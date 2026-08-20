import threading
import unittest

from agent_core.core import (
    CallableNode,
    ExecResult,
    Flow,
    FlowCancelled,
    FlowError,
    Node,
    RunContext,
    get_current_context,
    make_trace_options,
)


class CoreFlowTests(unittest.TestCase):
    def test_action_routes_to_one_successor(self) -> None:
        def classify(payload: dict) -> ExecResult:
            return ExecResult("question", payload)

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

    def test_plain_tuple_return_stays_payload(self) -> None:
        result = Flow(
            CallableNode(
                lambda payload: ("user_id", "123"),
                route_plain_tuples=False,
            )
        ).run({})

        self.assertEqual(result.action, "default")
        self.assertEqual(result.payload, ("user_id", "123"))

    def test_plain_tuple_routing_remains_available_for_0_1_compatibility(self) -> None:
        target = CallableNode(lambda payload: {**payload, "reached": True})
        source = CallableNode(lambda payload: ("go", payload))
        source - "go" >> target

        with self.assertWarns(DeprecationWarning):
            result = Flow(source).run({})

        self.assertTrue(result.payload["reached"])

    def test_exec_result_routes_and_payload_passes_through(self) -> None:
        result = Flow(
            CallableNode(lambda payload: ExecResult("next", payload))
        ).run({})

        self.assertEqual(result.action, "next")
        self.assertEqual(result.payload, {})
        self.assertIsInstance(result.action, str)

    def test_empty_action_routes_to_default_successor(self) -> None:
        target = CallableNode(lambda payload: {**payload, "reached": True})
        source = CallableNode(lambda payload: ExecResult("", payload))
        source - "" >> target

        result = Flow(source).run({})

        self.assertEqual(result.action, "default")
        self.assertEqual(result.payload["reached"], True)
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

    def test_terminal_node_can_finish_on_last_step(self) -> None:
        result = Flow(CallableNode(lambda payload: payload)).run({}, max_steps=1)

        self.assertEqual(result.path, ["CallableNode"])
        self.assertEqual(result.action, "default")

    def test_flow_records_node_and_flow_errors(self) -> None:
        context = RunContext()

        def fail(payload):
            raise ValueError("broken")

        with self.assertRaisesRegex(ValueError, "broken"):
            Flow(CallableNode(fail)).run({}, context=context)

        self.assertEqual(
            [event.type for event in context.events],
            ["node.start", "node.error", "flow.error"],
        )
        self.assertEqual(context.events[-1].data["error_type"], "ValueError")

    def test_flow_records_max_steps_error(self) -> None:
        node = CallableNode(lambda payload: payload)
        node >> node
        context = RunContext()

        with self.assertRaises(FlowError):
            Flow(node).run({}, max_steps=1, context=context)

        self.assertEqual(context.events[-1].type, "flow.error")
        self.assertEqual(context.events[-1].data["error_type"], "FlowError")

    def test_flow_collects_trace_events(self) -> None:
        node = CallableNode(lambda payload: payload)

        result = Flow(node).run({}, trace=True)

        self.assertEqual(result.path, ["CallableNode"])
        self.assertEqual(
            [event.type for event in result.trace],
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

        self.assertEqual(events[0].type, "node.start")
        self.assertEqual(events[0].node, "CallableNode")

    def test_flow_exposes_run_context_events_without_mirroring_payload(self) -> None:
        result = Flow(CallableNode(lambda payload: {"ok": True})).run({})

        self.assertIsNotNone(result.context)
        self.assertEqual(result.payload["ok"], True)
        self.assertFalse(hasattr(result.context, "state"))
        self.assertFalse(hasattr(result.context, "payload"))
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
        context = RunContext(on_event=events.append)

        result = Flow(CallableNode(lambda payload: {"answer": payload["seed"] + 1})).run(
            {"seed": 1},
            context=context,
        )

        self.assertIs(result.context, context)
        self.assertEqual(result.payload["answer"], 2)
        self.assertEqual(events[-1].type, "flow.end")

    def test_context_notify_is_live_only(self) -> None:
        events = []
        observations = []
        context = RunContext(on_event=events.append, on_observation=observations.append)

        context.notify("tool.progress", category="tool", data={"content": "working"})

        self.assertEqual([event.type for event in events], ["tool.progress"])
        self.assertEqual(context.events, [])
        self.assertEqual(observations, [])

    def test_cancel_before_run_raises_flow_cancelled(self) -> None:
        cancel = threading.Event()
        cancel.set()
        context = RunContext()

        with self.assertRaisesRegex(FlowCancelled, "cancelled"):
            Flow(CallableNode(lambda payload: payload)).run({}, context=context, cancel=cancel)

        self.assertEqual(context.events[-1].type, "flow.cancel")

    def test_cancel_between_steps_stops_the_flow(self) -> None:
        cancel = threading.Event()

        def trigger(payload: dict) -> dict:
            cancel.set()
            return payload

        first = CallableNode(trigger)
        second = CallableNode(lambda payload: {**payload, "reached": True})
        first >> second

        with self.assertRaises(FlowCancelled):
            Flow(first).run({}, cancel=cancel, max_steps=5)

    def test_nested_flow_inherits_the_cancel_event(self) -> None:
        cancel = threading.Event()

        def inner_trigger(payload: dict) -> dict:
            cancel.set()
            return payload

        inner_start = CallableNode(inner_trigger)
        inner_start >> inner_start
        inner_agent_flow = Flow(inner_start)
        outer_node = CallableNode(lambda payload: inner_agent_flow.run(payload, max_steps=5))

        with self.assertRaises(FlowCancelled):
            Flow(outer_node).run({}, cancel=cancel, max_steps=5)


if __name__ == "__main__":
    unittest.main()
