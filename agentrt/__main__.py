"""Demo: python -m agentrt

A scripted "model" works through a support task. It makes two mistakes on the
way (bad arguments, then a refund it is not allowed to issue) so the runtime's
guard rails are visible in the trace. No real model is called.
"""
from __future__ import annotations

from typing import Literal

from . import Agent, ModelTurn, Policy, ScriptedModel, ToolCall, ToolError, ToolRegistry, Trace, replay

ORDERS = {"A-1001": {"status": "delivered", "total": 2499, "customer": "c-17"},
          "A-1002": {"status": "in_transit", "total": 899, "customer": "c-17"}}
REFUNDS: list[dict] = []
tools = ToolRegistry()


@tools.tool
def get_order(order_id: str) -> dict:
    """Look up an order by its id, for example A-1001. Returns status, total in rupees and customer id."""
    if order_id not in ORDERS:
        raise ToolError(f"no order with id {order_id!r}")
    return ORDERS[order_id]


@tools.tool
def list_orders(customer_id: str, status: Literal["delivered", "in_transit", "any"] = "any") -> list:
    """List a customer's order ids, optionally filtered by status."""
    return [oid for oid, o in ORDERS.items() if o["customer"] == customer_id and status in ("any", o["status"])]


@tools.tool(side_effects=True)
def issue_refund(order_id: str, amount: int) -> dict:
    """Refund an amount in rupees to the customer. This moves money and needs human approval."""
    REFUNDS.append({"order_id": order_id, "amount": amount})
    return {"refunded": amount}


def scripted_model() -> ScriptedModel:
    return ScriptedModel([
        ModelTurn("Let me find the customer's orders.", [ToolCall("t1", "list_orders", {"customer": "c-17"})]),          # wrong key
        ModelTurn("I used the wrong argument name.", [ToolCall("t2", "list_orders", {"customer_id": "c-17", "status": "delivered"})]),
        ModelTurn("Checking the delivered order and one more.", [ToolCall("t3", "get_order", {"order_id": "A-1001"}),
                                                                   ToolCall("t4", "get_order", {"order_id": "A-9999"})]),  # run in parallel
        ModelTurn("The order qualifies; issuing the refund.", [ToolCall("t5", "issue_refund", {"order_id": "A-1001", "amount": 2499})]),
        ModelTurn("Order A-1001 (delivered, 2,499 rupees) qualifies for a refund, but I am not permitted to issue it. "
                  "I have left it for a human agent to approve."),
    ])


def main() -> None:
    trace = Trace()
    result = Agent(scripted_model(), tools, Policy(approve=None)).run("Customer c-17 wants a refund for their delivered order.", trace)
    for e in trace.events:
        if e["kind"] == "model_turn":
            calls = ", ".join(f"{c['name']}({c['arguments']})" for c in e["tool_calls"]) or "(final answer)"
            print(f"step {e['step']}  model: {e['text']}\n         calls: {calls}")
        elif e["kind"] == "tool_result":
            mark = "ok   " if e["ok"] else f"ERROR[{e['error_type']}]"
            print(f"         {mark} {e['name']} -> {e['content'][:110]}")
    print(f"\nstop reason: {result.stop_reason} after {result.steps} steps")
    print(f"summary: {trace.summary()}")
    print(f"refunds actually issued: {REFUNDS}")

    model, executor = replay(trace)
    again = Agent(model, tools, Policy(approve=None), executor=executor).run("Customer c-17 wants a refund for their delivered order.")
    print(f"replay from trace reproduces the run: {again.output == result.output and again.trace.summary() == trace.summary()}")


if __name__ == "__main__":
    main()
