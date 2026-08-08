import json
import threading
import time
import unittest
from typing import Annotated, Any, Literal, TypedDict

from agent_core.core import Flow
from agent_core.tools import Tool, ToolCall, ToolCallNode, ToolDefinitionError, ToolExecutor, get_current_tool_call, tool


def get_weather(city: str) -> dict[str, str]:
    return {"city": city, "condition": "sunny", "source": "mock"}


def weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get mocked weather.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        fn=get_weather,
    )


def sleeper_tool(name: str, *, parallel: bool) -> Tool:
    def slow(value: str) -> str:
        time.sleep(0.15)
        return f"{name}:{value}"

    return Tool(
        name=name,
        description="Sleeps briefly and echoes its input.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        fn=slow,
        parallel=parallel,
    )


def _openai_call(call_id: str, name: str, value: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"value": value})},
    }


class ToolTests(unittest.TestCase):
    def test_parallel_tools_run_concurrently(self) -> None:
        executor = ToolExecutor([sleeper_tool("slow_read", parallel=True)])
        assistant = {
            "tool_calls": [_openai_call(f"call_{i}", "slow_read", str(i)) for i in range(3)]
        }

        started = time.monotonic()
        results = executor.execute_all(executor.parse_tool_calls(assistant))
        elapsed = time.monotonic() - started

        # Three 150ms sleeps in parallel finish well under the 450ms serial sum.
        self.assertLess(elapsed, 0.4)
        self.assertEqual(
            [result.content for result in results],
            ["slow_read:0", "slow_read:1", "slow_read:2"],
        )

    def test_serial_tools_run_one_after_another(self) -> None:
        executor = ToolExecutor([sleeper_tool("slow_write", parallel=False)])
        assistant = {
            "tool_calls": [_openai_call(f"call_{i}", "slow_write", str(i)) for i in range(3)]
        }

        started = time.monotonic()
        results = executor.execute_all(executor.parse_tool_calls(assistant))
        elapsed = time.monotonic() - started

        self.assertGreaterEqual(elapsed, 0.4)
        self.assertEqual(
            [result.content for result in results],
            ["slow_write:0", "slow_write:1", "slow_write:2"],
        )

    def test_mixed_batch_preserves_result_order(self) -> None:
        executor = ToolExecutor(
            [
                sleeper_tool("read", parallel=True),
                sleeper_tool("write", parallel=False),
                sleeper_tool("grep", parallel=True),
            ]
        )
        assistant = {
            "tool_calls": [
                _openai_call("call_1", "write", "w"),
                _openai_call("call_2", "grep", "g"),
                _openai_call("call_3", "read", "r"),
            ]
        }

        results = executor.execute_all(executor.parse_tool_calls(assistant))

        self.assertEqual(
            [(result.tool_call_id, result.content) for result in results],
            [("call_1", "write:w"), ("call_2", "grep:g"), ("call_3", "read:r")],
        )

    def test_serial_tool_is_a_barrier_between_parallel_batches(self) -> None:
        active: set[str] = set()
        lock = threading.Lock()
        serial_overlapped = False

        def parallel(name: str, delay: float) -> Tool:
            def run(value: str) -> str:
                with lock:
                    active.add(name)
                try:
                    time.sleep(delay)
                    return value
                finally:
                    with lock:
                        active.remove(name)

            return Tool(
                name=name,
                description="Parallel test tool.",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
                fn=run,
                parallel=True,
            )

        def serial(value: str) -> str:
            nonlocal serial_overlapped
            with lock:
                serial_overlapped = bool(active)
            return value

        serial_tool = Tool(
            name="write",
            description="Serial test tool.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            fn=serial,
        )
        executor = ToolExecutor(
            [parallel("read", 0.03), serial_tool, parallel("grep", 0.2)]
        )
        calls = executor.parse_tool_calls(
            {
                "tool_calls": [
                    _openai_call("call_1", "read", "r"),
                    _openai_call("call_2", "write", "w"),
                    _openai_call("call_3", "grep", "g"),
                ]
            }
        )

        results = executor.execute_all(calls)

        self.assertFalse(serial_overlapped)
        self.assertEqual([result.content for result in results], ["r", "w", "g"])

    def test_results_carry_elapsed_ms(self) -> None:
        executor = ToolExecutor([sleeper_tool("slow_read", parallel=True)])
        assistant = {"tool_calls": [_openai_call("call_1", "slow_read", "x")]}

        results = executor.execute_all(executor.parse_tool_calls(assistant))

        self.assertIsNotNone(results[0].elapsed_ms)
        self.assertGreaterEqual(results[0].elapsed_ms, 100)

    def test_tool_call_node_emits_batch_events_in_order_with_elapsed_ms(self) -> None:
        node = ToolCallNode(
            executor=ToolExecutor(
                [
                    sleeper_tool("slow_read", parallel=True),
                    sleeper_tool("slow_write", parallel=False),
                ]
            ),
            next_action="chat",
        )
        payload = {
            "assistant_message": {
                "tool_calls": [
                    _openai_call("call_1", "slow_read", "r"),
                    _openai_call("call_2", "slow_write", "w"),
                ]
            },
            "history": [],
        }

        result = Flow(node).run(payload, trace=True)
        tool_events = [event for event in result.context.events if event.category == "tool"]

        self.assertEqual(
            [event.type for event in tool_events],
            ["tool.call", "tool.call", "tool.result", "tool.result"],
        )
        self.assertEqual(
            [event.data["tool_call_id"] for event in tool_events],
            ["call_1", "call_2", "call_1", "call_2"],
        )
        self.assertIsNotNone(tool_events[2].data.get("elapsed_ms"))
        self.assertEqual(tool_events[2].data["is_error"], False)
    def test_tool_executes_function(self) -> None:
        tool = weather_tool()

        result = tool.execute(city="Shanghai")

        self.assertEqual(tool.name, "get_weather")
        self.assertEqual(result["city"], "Shanghai")
        self.assertEqual(result["source"], "mock")

    def test_tool_decorator_builds_schema_from_signature(self) -> None:
        @tool(description="Get mocked weather.")
        def decorated_weather(
            city: Annotated[
                Literal["Shanghai", "Tokyo"],
                "City whose weather should be queried.",
            ],
            include_source: Annotated[bool, "Whether to include source metadata."] = True,
        ) -> dict[str, str]:
            return {"city": city, "source": "mock" if include_source else ""}

        llm_format = decorated_weather.to_llm_format()
        parameters = llm_format["function"]["parameters"]

        self.assertEqual(decorated_weather.name, "decorated_weather")
        self.assertEqual(decorated_weather(city="Shanghai")["city"], "Shanghai")
        self.assertEqual(parameters["required"], ["city"])
        self.assertEqual(
            parameters["properties"]["city"],
            {
                "type": "string",
                "enum": ["Shanghai", "Tokyo"],
                "description": "City whose weather should be queried.",
            },
        )
        self.assertEqual(parameters["properties"]["include_source"]["type"], "boolean")
        self.assertEqual(
            parameters["properties"]["include_source"]["description"],
            "Whether to include source metadata.",
        )
        self.assertTrue(parameters["properties"]["include_source"]["default"])

    def test_typed_dict_schema_and_current_tool_call(self) -> None:
        class Replacement(TypedDict):
            old_text: Annotated[str, "Exact text to replace."]
            new_text: str

        @tool(description="Apply replacements.")
        def batch_edit(edits: list[Replacement]) -> dict[str, Any]:
            call = get_current_tool_call()
            return {"call_id": call.id if call else "", "count": len(edits)}

        items = batch_edit.parameters["properties"]["edits"]["items"]
        result = ToolExecutor([batch_edit]).execute(
            ToolCall("edit-1", "batch_edit", {"edits": [{"old_text": "a", "new_text": "b"}]})
        )

        self.assertEqual(items["required"], ["old_text", "new_text"])
        self.assertEqual(items["properties"]["old_text"]["description"], "Exact text to replace.")
        self.assertIn('\"call_id\": \"edit-1\"', result.content)

    def test_tool_decorator_requires_annotations(self) -> None:
        with self.assertRaises(ToolDefinitionError):

            @tool(description="Missing parameter type.")
            def missing_param_type(city) -> dict[str, str]:
                return {"city": city}

        with self.assertRaises(ToolDefinitionError):

            @tool(description="Missing return type.")
            def missing_return_type(city: str):
                return {"city": city}

    def test_executor_runs_openai_style_tool_call(self) -> None:
        executor = ToolExecutor([weather_tool()])
        assistant_message = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Shanghai"}',
                    },
                }
            ]
        }

        tool_calls = executor.parse_tool_calls(assistant_message)
        results = executor.execute_all(tool_calls)

        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertIn('"city": "Shanghai"', results[0].content)
        self.assertFalse(results[0].is_error)

    def test_executor_handles_unknown_tool(self) -> None:
        executor = ToolExecutor()
        tool_call = executor.parse_tool_calls(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "missing", "arguments": "{}"},
                    }
                ]
            }
        )[0]

        result = executor.execute(tool_call)

        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)

    def test_executor_rejects_duplicate_tool_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate tool names: get_weather"):
            ToolExecutor([weather_tool(), weather_tool()])

    def test_executor_contains_malformed_calls_and_results(self) -> None:
        broken = Tool(
            name="broken",
            description="Return an unserializable value.",
            parameters={"type": "object", "properties": {}},
            fn=lambda: object(),
        )
        executor = ToolExecutor([broken])
        calls = executor.parse_tool_calls(
            {
                "tool_calls": [
                    None,
                    {"id": "call_1", "function": {"name": "broken", "arguments": "{}"}},
                ]
            }
        )

        result = executor.execute(calls[0])

        self.assertEqual(len(calls), 1)
        self.assertTrue(result.is_error)
        self.assertIn("TypeError", result.content)

    def test_tool_call_node_appends_tool_messages(self) -> None:
        node = ToolCallNode(executor=ToolExecutor([weather_tool()]), next_action="chat")
        payload = {
            "assistant_message": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Shanghai"}',
                        },
                    }
                ]
            },
            "history": [],
        }

        action, state = node.exec(payload)

        self.assertEqual(action, "chat")
        self.assertEqual(payload["history"], [])
        self.assertEqual(state["history"][0]["role"], "tool")
        self.assertEqual(state["history"][0]["tool_call_id"], "call_1")

    def test_tool_call_node_emits_trace_events(self) -> None:
        node = ToolCallNode(executor=ToolExecutor([weather_tool()]), next_action="chat")
        payload = {
            "assistant_message": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Shanghai"}',
                        },
                    }
                ]
            },
            "history": [],
        }

        result = Flow(node).run(payload, trace=True)
        tool_events = [event for event in result.trace if event.category == "tool"]
        runtime_tool_events = [
            event for event in result.context.events if event.category == "tool"
        ]

        self.assertEqual([event.type for event in tool_events], ["tool.call", "tool.result"])
        self.assertEqual(tool_events[0].data["name"], "get_weather")
        self.assertEqual(tool_events[0].data["arguments"], {"city": "Shanghai"})
        self.assertFalse(tool_events[1].data["is_error"])
        self.assertEqual(
            [event.type for event in runtime_tool_events],
            ["tool.call", "tool.result"],
        )
        self.assertEqual(result.context.messages[-1]["role"], "tool")
        self.assertEqual(result.context.messages[-1]["tool_call_id"], "call_1")
        self.assertNotIn("is_error", result.context.messages[-1])


if __name__ == "__main__":
    unittest.main()
