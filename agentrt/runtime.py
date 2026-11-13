from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .tools import ToolError, ToolRegistry, ValidationError
from .trace import Trace


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Any


@dataclass(frozen=True)
class ModelTurn:
    """One reply from the model: text, tool calls, or both. No tool calls means it is done."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class Model(Protocol):
    def next_turn(self, messages: list[dict], tools: list[dict]) -> ModelTurn: ...


class ScriptedModel:
    """A model that replies from a fixed script. Each item is a ModelTurn or a function of the messages."""

    def __init__(self, script: list) -> None:
        self.script, self.seen = list(script), []

    def next_turn(self, messages: list[dict], tools: list[dict]) -> ModelTurn:
        self.seen.append([dict(m) for m in messages])
        if not self.script:
            raise RuntimeError("scripted model ran out of turns")
        step = self.script.pop(0)
        return step(messages) if callable(step) else step


@dataclass
class Policy:
    """Limits that hold no matter what the model says."""

    max_steps: int = 8                    # model turns
    max_tool_calls: int = 20
    max_seconds: float = 120.0
    max_result_chars: int = 4_000         # tool output is truncated before the model sees it
    max_identical_calls: int = 2          # the same call with the same arguments, repeated
    approve: Callable[[ToolCall], bool] | None = None   # asked before any tool with side effects runs


@dataclass
class AgentResult:
    output: str
    stop_reason: str          # "completed" or the limit that ended the run
    steps: int
    trace: Trace

    @property
    def completed(self) -> bool:
        return self.stop_reason == "completed"


class Agent:
    """The loop: ask the model, run the tools it asks for, feed results back, repeat.

    The model decides what to do. The runtime decides what it is allowed to do:
    arguments are validated, tools with side effects need approval, every tool
    has a timeout, and the loop has hard limits. A tool failure is returned to
    the model as an error message so it can correct itself; it never crashes the run.
    """

    def __init__(self, model: Model, tools: ToolRegistry, policy: Policy | None = None, system: str = "",
                 clock: Callable[[], float] = time.monotonic, executor: Callable[[ToolCall], dict] | None = None) -> None:
        self.model, self.tools, self.policy, self.system = model, tools, policy or Policy(), system
        self._clock = clock
        self._executor = executor     # set by replay to return recorded results instead of running tools

    def run(self, task: str, trace: Trace | None = None) -> AgentResult:
        trace = trace or Trace()
        policy = self.policy
        messages: list[dict] = [{"role": "user", "content": task}]
        trace.add("start", task=task, tools=self.tools.names())
        started, tool_calls, seen_calls, last_text = self._clock(), 0, {}, ""

        def stop(reason: str, steps: int) -> AgentResult:
            trace.add("stop", reason=reason, steps=steps)
            return AgentResult(last_text, reason, steps, trace)

        for step in range(1, policy.max_steps + 1):
            if self._clock() - started > policy.max_seconds:
                return stop("time_limit", step - 1)
            turn = self.model.next_turn(messages, self.tools.specs())
            last_text = turn.text or last_text
            trace.add("model_turn", step=step, text=turn.text,
                      tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in turn.tool_calls])
            if not turn.tool_calls:
                return stop("completed", step)
            messages.append({"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls})

            if tool_calls + len(turn.tool_calls) > policy.max_tool_calls:
                return stop("tool_call_limit", step)
            tool_calls += len(turn.tool_calls)

            runnable, results = [], {}
            for call in turn.tool_calls:
                key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
                seen_calls[key] = seen_calls.get(key, 0) + 1
                if seen_calls[key] > policy.max_identical_calls:
                    results[call.id] = {"ok": False, "error_type": "loop", "content":
                                        "You already made this exact call. Repeating it will not change the result; try something else or answer."}
                else:
                    runnable.append(call)
            # Calls within one turn are independent by construction, so run them together.
            if len(runnable) > 1 and self._executor is None:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(runnable)) as pool:
                    for call, result in zip(runnable, pool.map(self._execute, runnable)):
                        results[call.id] = result
            else:
                for call in runnable:
                    results[call.id] = self._execute(call)

            for call in turn.tool_calls:   # report in the order the model asked
                result = results[call.id]
                trace.add("tool_result", step=step, call_id=call.id, name=call.name, **result)
                messages.append({"role": "tool", "tool_call_id": call.id, "is_error": not result["ok"], "content": result["content"]})
        return stop("step_limit", policy.max_steps)

    def _execute(self, call: ToolCall) -> dict:
        if self._executor is not None:
            return self._executor(call)
        tool = self.tools.get(call.name)
        if tool is None:
            return self._fail("unknown_tool", f"No tool named {call.name!r}. Available: {self.tools.names()}")
        try:
            arguments = tool.validate(call.arguments)
        except ValidationError as exc:
            return self._fail("invalid_arguments", f"Invalid arguments for {tool.name}: {exc}")
        if tool.side_effects:
            if self.policy.approve is None or not self.policy.approve(call):
                return self._fail("denied", f"{tool.name} changes things and was not approved. Do not retry it; explain what you would have done.")
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.fn, **arguments)
        try:
            value = future.result(timeout=tool.timeout_s)
        except concurrent.futures.TimeoutError:
            return self._fail("timeout", f"{tool.name} did not finish within {tool.timeout_s} s")
        except ToolError as exc:
            return self._fail("tool_error", str(exc))
        except Exception as exc:  # noqa: BLE001 - a buggy tool must not take the agent down
            return self._fail("exception", f"{tool.name} raised {type(exc).__name__}: {exc}")
        finally:
            pool.shutdown(wait=False)
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        limit = self.policy.max_result_chars
        if len(text) > limit:
            text = text[:limit] + f"\n[truncated: {len(text) - limit} more characters]"
        return {"ok": True, "content": text, "error_type": None}

    @staticmethod
    def _fail(error_type: str, message: str) -> dict:
        return {"ok": False, "content": message, "error_type": error_type}
