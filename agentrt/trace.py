"""Every step is recorded. A trace can be saved, inspected and replayed without the model or the tools."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Trace:
    events: list[dict] = field(default_factory=list)
    clock: Callable[[], float] = time.time

    def add(self, kind: str, **data: Any) -> None:
        self.events.append({"seq": len(self.events), "at": self.clock(), "kind": kind, **data})

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, default=str) + "\n")

    @classmethod
    def load(cls, path: str) -> "Trace":
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(line) for line in f if line.strip()])

    def summary(self) -> dict:
        results = self.of("tool_result")
        return {
            "model_turns": len(self.of("model_turn")),
            "tool_calls": len(results),
            "tool_errors": sum(1 for r in results if not r["ok"]),
            "denied": sum(1 for r in results if r.get("error_type") == "denied"),
            "stop_reason": (self.of("stop") or [{"reason": None}])[-1]["reason"],
        }


def replay(trace: Trace):
    """Rebuild a model and a tool executor from a trace.

    The returned pair feeds an Agent the recorded model turns and tool results,
    so a production failure can be re-run step by step with no API calls and no
    side effects. If the agent asks for something the trace did not record, the
    replay raises: the code path has diverged from what happened.
    """
    from .runtime import ModelTurn, ToolCall

    turns = [ModelTurn(e["text"], [ToolCall(**c) for c in e["tool_calls"]]) for e in trace.of("model_turn")]
    recorded = {(e["call_id"]): e for e in trace.of("tool_result")}

    class ReplayModel:
        def __init__(self) -> None:
            self.position = 0

        def next_turn(self, messages, tools):
            if self.position >= len(turns):
                raise RuntimeError("replay diverged: the agent asked for a model turn that was not recorded")
            self.position += 1
            return turns[self.position - 1]

    def executor(call) -> dict:
        if call.id not in recorded:
            raise RuntimeError(f"replay diverged: no recorded result for call {call.id}")
        e = recorded[call.id]
        return {"ok": e["ok"], "content": e["content"], "error_type": e.get("error_type")}

    return ReplayModel(), executor
