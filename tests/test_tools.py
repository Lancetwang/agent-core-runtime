import unittest
from typing import Annotated, Literal

from agent_core.core import Flow
from agent_core.tools import Tool, ToolCallNode, ToolDefinitionError, ToolExecutor, tool


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


class ToolTests(unittest.TestCase):
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

        self.assertEqual([event.event for event in tool_events], ["tool.call", "tool.result"])
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
