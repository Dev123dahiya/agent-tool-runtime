import threading
import time
from typing import Literal, Optional

import pytest

from agentrt import Agent, ModelTurn, Policy, ScriptedModel, ToolCall, ToolError, ToolRegistry, Trace, ValidationError, replay


def make_tools():
    reg = ToolRegistry()
    state = {"writes": [], "active": 0, "peak": 0}
    lock = threading.Lock()

    @reg.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @reg.tool
    def search(query: str, limit: int = 3, sort: Literal["new", "top"] = "top", tags: Optional[list[str]] = None) -> list:
        """Search documents."""
        return [f"{query}-{i}" for i in range(limit)]

    @reg.tool(side_effects=True)
    def write_file(path: str, content: str) -> str:
        """Write a file."""
        state["writes"].append(path)
        return "written"

    @reg.tool
    def fail(reason: str) -> str:
        """Always fails."""
        if reason == "bug":
            raise KeyError("oops")
        raise ToolError(f"cannot do that: {reason}")

    @reg.tool(timeout_s=0.05)
    def slow() -> str:
        """Takes too long."""
        time.sleep(0.5)
        return "done"

    @reg.tool
    def busy(n: int) -> int:
        """Records how many calls overlap."""
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return n

    @reg.tool
    def big() -> str:
        """Returns a lot of text."""
        return "x" * 10_000

    return reg, state


def call(i, name, **arguments):
    return ToolCall(f"t{i}", name, arguments)


def run(script, policy=None, tools=None):
    reg, state = tools or make_tools()
    model = ScriptedModel(script)
    result = Agent(model, reg, policy).run("task")
    return result, model, state


def test_schema_is_derived_from_the_signature():
    reg, _ = make_tools()
    spec = reg.get("search").spec()
    assert spec["description"] == "Search documents."
    assert spec["input_schema"] == {
        "type": "object", "required": ["query"], "additionalProperties": False,
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}, "sort": {"enum": ["new", "top"]},
                       "tags": {"type": "array", "items": {"type": "string"}}},
    }
    assert reg.names() == ["add", "big", "busy", "fail", "search", "slow", "write_file"]


def test_registration_rejects_undocumented_or_untyped_tools():
    reg = ToolRegistry()
    with pytest.raises(TypeError, match="docstring"):
        @reg.tool
        def no_doc(x: int) -> int:
            return x
    with pytest.raises(TypeError, match="type hint"):
        @reg.tool
        def no_hint(x):
            """Has a docstring."""
            return x
    @reg.tool
    def ok() -> str:
        """Fine."""
        return "ok"
    with pytest.raises(ValueError, match="already registered"):
        reg.tool(ok)


@pytest.mark.parametrize("arguments, message", [
    ({"query": "x", "limt": 5}, "unknown argument"),
    ({}, "missing required"),
    ({"query": 5}, "expected string"),
    ({"query": "x", "limit": "5"}, "expected integer"),
    ({"query": "x", "limit": True}, "expected integer"),
    ({"query": "x", "sort": "random"}, "must be one of"),
    ({"query": "x", "tags": ["a", 2]}, "tags[1]"),
    ("query=x", "must be a JSON object"),
])
def test_validation_rejects_bad_arguments(arguments, message):
    reg, _ = make_tools()
    with pytest.raises(ValidationError, match=message.replace("[", r"\[").replace("]", r"\]")):
        reg.get("search").validate(arguments)


def test_tool_results_are_fed_back_and_the_final_text_is_returned():
    result, model, _ = run([ModelTurn("adding", [call(1, "add", a=2, b=3)]), ModelTurn("The answer is 5.")])
    assert result.completed and result.output == "The answer is 5." and result.steps == 2
    second = model.seen[1]
    assert [m["role"] for m in second] == ["user", "assistant", "tool"]
    assert second[2] == {"role": "tool", "tool_call_id": "t1", "is_error": False, "content": "5"}


def test_errors_reach_the_model_as_messages_instead_of_crashing():
    result, model, _ = run([
        ModelTurn("", [call(1, "nope"), call(2, "add", a="2", b=3), call(3, "fail", reason="locked"), call(4, "fail", reason="bug"), call(5, "slow")]),
        ModelTurn("giving up"),
    ])
    assert result.completed
    errors = {e["call_id"]: (e["error_type"], e["content"]) for e in result.trace.of("tool_result")}
    assert errors["t1"][0] == "unknown_tool" and "Available:" in errors["t1"][1]
    assert errors["t2"][0] == "invalid_arguments" and "expected integer" in errors["t2"][1]
    assert errors["t3"] == ("tool_error", "cannot do that: locked")
    assert errors["t4"][0] == "exception" and "KeyError" in errors["t4"][1]
    assert errors["t5"][0] == "timeout"
    assert all(m["is_error"] for m in model.seen[1] if m["role"] == "tool")


def test_side_effect_tools_need_approval():
    script = lambda: [ModelTurn("", [call(1, "write_file", path="/etc/passwd", content="x")]), ModelTurn("done")]
    denied, _, state = run(script())
    assert state["writes"] == [] and denied.trace.of("tool_result")[0]["error_type"] == "denied"

    asked = []
    policy = Policy(approve=lambda c: asked.append(c) or c.arguments["path"].startswith("/tmp/"))
    refused, _, state = run(script(), policy)
    assert state["writes"] == [] and asked[0].name == "write_file"
    allowed, _, state = run([ModelTurn("", [call(1, "write_file", path="/tmp/a", content="x")]), ModelTurn("done")], policy)
    assert state["writes"] == ["/tmp/a"] and allowed.trace.of("tool_result")[0]["ok"]

    never_asked = []
    run([ModelTurn("", [call(1, "add", a=1, b=1)]), ModelTurn("done")], Policy(approve=lambda c: never_asked.append(c) or True))
    assert never_asked == []   # read-only tools do not prompt


def test_step_limit_stops_a_model_that_never_finishes():
    forever = [ModelTurn(f"still going {i}", [call(i, "add", a=i, b=1)]) for i in range(50)]
    result, _, _ = run(forever, Policy(max_steps=4))
    assert result.stop_reason == "step_limit" and result.steps == 4 and not result.completed
    assert result.output == "still going 3"


def test_tool_call_and_time_limits():
    many = [ModelTurn("", [call(i * 10 + j, "add", a=i, b=j) for j in range(4)]) for i in range(5)]
    result, _, _ = run(many, Policy(max_tool_calls=6))
    assert result.stop_reason == "tool_call_limit" and len(result.trace.of("tool_result")) == 4

    now = [0.0]
    reg, _ = make_tools()

    @reg.tool
    def wait() -> str:
        """Advances the fake clock."""
        now[0] += 40
        return "waited"

    model = ScriptedModel([ModelTurn("", [call(i, "wait")]) for i in range(10)])
    result = Agent(model, reg, Policy(max_seconds=100, max_identical_calls=99), clock=lambda: now[0]).run("task")
    assert result.stop_reason == "time_limit" and result.steps == 3


def test_repeated_identical_calls_are_cut_off_with_guidance():
    same = [ModelTurn("", [call(i, "search", query="x")]) for i in range(4)] + [ModelTurn("ok, answering")]
    result, model, _ = run(same)
    kinds = [e["error_type"] for e in result.trace.of("tool_result")]
    assert kinds == [None, None, "loop", "loop"]
    assert "already made this exact call" in model.seen[3][-1]["content"]


def test_calls_in_one_turn_run_in_parallel_and_report_in_order():
    result, model, state = run([ModelTurn("", [call(i, "busy", n=i) for i in range(4)]), ModelTurn("done")])
    assert state["peak"] >= 2
    assert [m["content"] for m in model.seen[1] if m["role"] == "tool"] == ["0", "1", "2", "3"]


def test_long_tool_output_is_truncated_before_the_model_sees_it():
    result, model, _ = run([ModelTurn("", [call(1, "big")]), ModelTurn("done")], Policy(max_result_chars=100))
    content = model.seen[1][-1]["content"]
    assert content.startswith("x" * 100) and content.endswith("[truncated: 9900 more characters]")


def test_trace_round_trip_and_replay_without_touching_tools(tmp_path):
    reg, state = make_tools()
    policy = Policy(approve=lambda c: True)
    script = [ModelTurn("", [call(1, "write_file", path="/tmp/a", content="x"), call(2, "fail", reason="locked")]), ModelTurn("final")]
    original = Agent(ScriptedModel(script), reg, policy).run("task")
    assert state["writes"] == ["/tmp/a"]

    path = str(tmp_path / "trace.jsonl")
    original.trace.save(path)
    loaded = Trace.load(path)
    assert loaded.summary() == original.trace.summary() == {
        "model_turns": 2, "tool_calls": 2, "tool_errors": 1, "denied": 0, "stop_reason": "completed"}

    model, executor = replay(loaded)
    replayed = Agent(model, reg, policy, executor=executor).run("task")
    assert replayed.output == "final" and replayed.trace.summary() == original.trace.summary()
    assert state["writes"] == ["/tmp/a"]            # the side effect did not happen a second time
    assert [e["content"] for e in replayed.trace.of("tool_result")] == [e["content"] for e in loaded.of("tool_result")]


def test_replay_detects_divergence():
    reg, _ = make_tools()
    trace = Agent(ScriptedModel([ModelTurn("", [call(1, "add", a=1, b=2)]), ModelTurn("3")]), reg).run("task").trace
    model, executor = replay(trace)
    with pytest.raises(RuntimeError, match="diverged"):
        executor(ToolCall("t99", "add", {"a": 1, "b": 2}))
    model.next_turn([], [])
    model.next_turn([], [])
    with pytest.raises(RuntimeError, match="diverged"):
        model.next_turn([], [])
