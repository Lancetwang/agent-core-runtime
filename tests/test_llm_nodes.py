import unittest
from typing import Annotated

from agent_core import Agent, ModelNode, ToolRouterNode, tool
from agent_core.core import Flow


class FakeChatModel:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
        self.requests.append(
            {
                "messages": list(messages),
                "tools": list(tools or []),
                "tool_choice": tool_choice,
                "kwargs": kwargs,
            }
        )
        return self.responses.pop(0)


class DeltaFakeChatModel:
    def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
        on_delta = kwargs.get("on_delta")
        if on_delta is not None:
            on_delta("hel")
            on_delta("lo")
        return {"role": "assistant", "content": "hello"}


@tool(description="Look up demo weather.")
def get_weather(city: Annotated[str, "City name."]) -> dict[str, str]:
    return {"city": city, "condition": "sunny"}


def build_messages(payload: dict) -> list[dict]:
    return [{"role": "system", "content": "Use tools when useful."}, *payload["history"]]


class LlmNodeTests(unittest.TestCase):
    def test_model_node_stores_assistant_message(self) -> None:
        model = FakeChatModel(
            [{"role": "assistant", "content": "hello", "usage": {"total_tokens": 3}}]
        )
        node = ModelNode(
            model=model,
            messages=build_messages,
            tools=[get_weather],
            chat_kwargs={"temperature": 0},
        )

        result = Flow(node).run({"history": [{"role": "user", "content": "hi"}]})

        self.assertEqual(result.payload["assistant_message"]["content"], "hello")
        self.assertEqual(result.payload["history"][-1]["content"], "hello")
        self.assertEqual(model.requests[0]["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(model.requests[0]["kwargs"]["temperature"], 0)
        self.assertIn(
            "model.response",
            [event.type for event in result.context.events],
        )

    def test_tool_router_preserves_content_when_tool_calls_exist(self) -> None:
        assistant_message = {
            "role": "assistant",
            "content": "I will check that.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Shanghai"}',
                    },
                }
            ],
        }

        action, state = ToolRouterNode().exec({"assistant_message": assistant_message})

        self.assertEqual(action, "tool_call")
        self.assertEqual(state["assistant_message"]["content"], "I will check that.")

    def test_model_node_emits_delta_events(self) -> None:
        node = ModelNode(model=DeltaFakeChatModel(), messages=build_messages)

        result = Flow(node).run({"history": [{"role": "user", "content": "hi"}]})

        delta_events = [
            event for event in result.context.events if event.type == "model.delta"
        ]
        self.assertEqual(
            [event.data["content"] for event in delta_events],
            ["hel", "lo"],
        )

    def test_agent_runs_default_tool_loop(self) -> None:
        model = FakeChatModel(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Shanghai"}',
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Shanghai is sunny.",
                },
            ]
        )
        agent = Agent(
            model=model,
            tools=[get_weather],
            chat_kwargs={"tool_choice": "auto"},
        )
        context = agent.new_context()
        context.add_message("system", "Use tools when useful.")
        context.add_message("user", "Shanghai weather?")

        result = agent.run({}, context=context)

        self.assertEqual(result.payload["answer"], "Shanghai is sunny.")
        self.assertEqual(
            result.path,
            ["ModelNode", "ToolRouterNode", "ToolCallNode", "ModelNode", "ToolRouterNode"],
        )
        self.assertEqual(model.requests[0]["tool_choice"], "auto")
        self.assertEqual(model.requests[1]["messages"][-1]["role"], "tool")
        self.assertIn('"condition": "sunny"', model.requests[1]["messages"][-1]["content"])
        self.assertEqual(
            [message["role"] for message in result.context.messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertIn("tool_calls", result.context.messages[2])
        self.assertNotIn("tool_calls", result.context.messages[-1])
        self.assertEqual(
            [event.type for event in result.context.events if event.category == "tool"],
            ["tool.observe", "tool.call", "tool.result", "tool.observe"],
        )


if __name__ == "__main__":
    unittest.main()
