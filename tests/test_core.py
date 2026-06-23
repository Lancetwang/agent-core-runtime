import unittest

from agent_core.core import (
    CallableNode,
    Flow,
    FlowError,
    Node,
    RunContext,
    get_current_context,
    make_trace_options,
)
from agent_core.core.trace import TRACE_KEY


class CoreFlowTests(unittest.TestCase):
    def test_action_routes_to_one_successor(self) -> None:
        def classify(payload: dict) -> tuple[str, dict]:
            return "question", payload

        def answer(payload: dict) -> dict:
            payload["reply"] = "ok"
            return payload

        start = CallableNode(classify)
        answer_node = CallableNode(answer)
        start - "question" >> answer_node

        result = Flow(start).run({"input": "hello?"})

        self.assertEqual(result.action, "default")
        self.assertEqual(result.payload["reply"], "ok")
        self.assertEqual(result.path, ["CallableNode", "CallableNode"])

    def test_retry(self) -> None:
        calls = {"count": 0}

        class FlakyNode(Node):
            def exec(self, payload):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise ValueError("try again")
                return "default", payload

        Flow(FlakyNode(max_retries=2)).run({})

        self.assertEqual(calls["count"], 2)

    def test_max_steps_guard(self) -> None:
        node = CallableNode(lambda payload: payload)
        node >> node

        with self.assertRaises(FlowError):
            Flow(node).run({}, max_steps=2)

    def test_flow_collects_trace_events(self) -> None:
        node = CallableNode(lambda payload: payload)

        result = Flow(node).run({}, trace=True)

        self.assertEqual(result.path, ["CallableNode"])
        self.assertNotIn(TRACE_KEY, result.payload)
        self.assertEqual(
            [event.event for event in result.trace],
            ["node.start", "node.end", "flow.end"],
        )
        self.assertEqual(result.trace[0].step, 1)
        self.assertEqual(result.trace[1].action, "default")

    def test_flow_trace_can_filter_categories(self) -> None:
        node = CallableNode(lambda payload: payload)
        trace = make_trace_options(include=["flow"])

        result = Flow(node).run({}, trace=trace)

        self.assertEqual([event.category for event in result.trace], ["flow"])

    def test_flow_trace_can_emit_structured_events(self) -> None:
        events = []
        node = CallableNode(lambda payload: payload)

        Flow(node).run({}, trace=make_trace_options(on_event=events.append))

        self.assertEqual(events[0].event, "node.start")
        self.assertEqual(events[0].node, "CallableNode")

    def test_flow_trace_does_not_leak_into_original_payload(self) -> None:
        def copy_payload(payload: dict) -> dict:
            return {"ok": payload.get("ok")}

        payload = {"ok": True}

        Flow(CallableNode(copy_payload)).run(payload, trace=True)

        self.assertNotIn(TRACE_KEY, payload)

    def test_flow_exposes_run_context_events(self) -> None:
        result = Flow(CallableNode(lambda payload: {"ok": True})).run({})

        self.assertIsNotNone(result.context)
        self.assertEqual(result.context.state["ok"], True)
        self.assertEqual(
            [event.type for event in result.context.events],
            ["node.start", "node.end", "flow.end"],
        )

    def test_node_can_emit_context_events(self) -> None:
        def add_message(payload: dict) -> dict:
            context = get_current_context()
            self.assertIsNotNone(context)
            context.add_message("assistant", "hello")
            return payload

        result = Flow(CallableNode(add_message)).run({})

        self.assertEqual(result.context.messages[0]["content"], "hello")
        self.assertIn(
            "message.add",
            [event.type for event in result.context.events],
        )

    def test_flow_can_use_supplied_run_context(self) -> None:
        events = []
        context = RunContext.from_payload({"seed": 1}, on_event=events.append)

        result = Flow(CallableNode(lambda payload: {"answer": payload["seed"] + 1})).run(
            {"seed": 1},
            context=context,
        )

        self.assertIs(result.context, context)
        self.assertEqual(result.context.state["answer"], 2)
        self.assertEqual(events[-1].type, "flow.end")


if __name__ == "__main__":
    unittest.main()
