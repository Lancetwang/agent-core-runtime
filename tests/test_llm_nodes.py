import unittest
from typing import Annotated

from agent_core import (
    Agent,
    ModelNode,
    RunContext,
    ToolCallNode,
    ToolExecutor,
    ToolRouterNode,
    tool,
)
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
            [
                {
                    "role": "assistant",
                    "content": "hello",
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                }
            ]
        )
        node = ModelNode(
            model=model,
            messages=build_messages,
            tools=[get_weather],
            chat_kwargs={"temperature": 0},
        )

        result = Flow(node).run({"history": [{"role": "user", "content": "hi"}]})

        self.assertEqual(result.payload["assistant_message"]["content"], "hello")
        # The context scope is the canonical store; state history no longer
        # mirrors assistant messages during a flow run.
        self.assertEqual(result.payload["history"], [{"role": "user", "content": "hi"}])
        self.assertEqual(result.context.messages[-1]["role"], "assistant")
        self.assertEqual(result.context.messages[-1]["content"], "hello")
        self.assertEqual(model.requests[0]["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(model.requests[0]["kwargs"]["temperature"], 0)
        self.assertIn(
            "model.response",
            [event.type for event in result.context.events],
        )
        self.assertEqual(
            result.usage.to_dict(),
            {"requests": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        )

    def test_model_node_does_not_mutate_input_history(self) -> None:
        model = FakeChatModel([{"role": "assistant", "content": "hello"}])
        payload = {"history": [{"role": "user", "content": "hi"}]}

        result = Flow(ModelNode(model=model)).run(payload)

        self.assertEqual(payload["history"], [{"role": "user", "content": "hi"}])
        self.assertEqual(result.payload["history"], [{"role": "user", "content": "hi"}])
        self.assertEqual(result.context.messages[-1]["content"], "hello")

    def test_context_imports_payload_history_before_tool_loop(self) -> None:
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
                {"role": "assistant", "content": "sunny"},
            ]
        )
        model_node = ModelNode(model=model, tools=[get_weather], action="observe")
        router_node = ToolRouterNode(tool_action="tool_call", done_action="final")
        tool_node = ToolCallNode(executor=ToolExecutor([get_weather]), next_action="chat")
        model_node - "observe" >> router_node
        router_node - "tool_call" >> tool_node
        tool_node - "chat" >> model_node

        Flow(model_node).run({"history": [{"role": "user", "content": "weather?"}]})

        self.assertEqual(
            [message["role"] for message in model.requests[1]["messages"]],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(model.requests[1]["messages"][0]["content"], "weather?")

    def test_model_node_observes_full_payload_without_retaining_it(self) -> None:
        model = FakeChatModel([{"role": "assistant", "content": "hello"}])
        node = ModelNode(
            model=model,
            messages=build_messages,
            tools=[get_weather],
            chat_kwargs={"temperature": 0},
        )
        observations = []
        context = RunContext()
        context.on_observation = observations.append

        result = Flow(node).run({"history": [{"role": "user", "content": "hi"}]}, context=context)

        request = next(event for event in observations if event.type == "model.request.payload")
        response = next(event for event in observations if event.type == "model.response.payload")
        self.assertEqual(request.data["messages"][-1]["content"], "hi")
        self.assertEqual(request.data["tools"][0]["function"]["name"], "get_weather")
        self.assertEqual(response.data["message"]["content"], "hello")
        self.assertNotIn("model.request.payload", [event.type for event in result.context.events])

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

        result = Flow(node).run({"history": [{"role": "user", "content": "hi"}]}, trace=True)

        delta_events = [
            event for event in result.trace if event.type == "model.delta"
        ]
        self.assertEqual(
            [event.data["content"] for event in delta_events],
            ["hel", "lo"],
        )
        # Streamed deltas are live-only and never retained in context events.
        self.assertNotIn(
            "model.delta",
            [event.type for event in result.context.events],
        )

    def test_model_node_emits_reasoning_deltas(self) -> None:
        class ReasoningModel:
            def chat_message(self, messages, **kwargs):
                kwargs["on_reasoning_delta"]("think")
                return {"role": "assistant", "content": "done", "reasoning_content": "think"}

        result = Flow(
            ModelNode(model=ReasoningModel(), chat_kwargs={"on_reasoning_delta": lambda _: None})
        ).run({"history": [{"role": "user", "content": "hi"}]}, trace=True)

        events = [event for event in result.trace if event.type == "model.reasoning.delta"]
        self.assertEqual([event.data["content"] for event in events], ["think"])
        self.assertNotIn(
            "model.reasoning.delta",
            [event.type for event in result.context.events],
        )

    def test_unscoped_ambient_messages_are_adopted_into_agent_scope(self) -> None:
        model = FakeChatModel(
            [
                {"role": "assistant", "content": "answer one"},
                {"role": "assistant", "content": "answer two"},
            ]
        )
        tools = [get_weather]
        model_node = ModelNode(model=model, tools=tools, action="observe")
        router_node = ToolRouterNode(tool_action="tool_call", done_action="final")
        tool_node = ToolCallNode(executor=ToolExecutor(tools), next_action="chat")
        model_node - "observe" >> router_node
        router_node - "tool_call" >> tool_node
        tool_node - "chat" >> model_node
        agent = Agent(Flow(model_node))

        context = RunContext()
        context.add_message("system", "SYSTEM")
        context.add_message("user", "first?")

        agent.run({"turn": 1}, context=context, trace=False)
        context.add_message("user", "second?")

        agent.run({"turn": 2}, context=context, trace=False)

        self.assertEqual(
            [message["content"] for message in model.requests[0]["messages"]],
            ["SYSTEM", "first?"],
        )
        self.assertEqual(
            [message["content"] for message in model.requests[1]["messages"]],
            ["SYSTEM", "first?", "answer one", "second?"],
        )

    def test_state_history_retains_reasoning_only_for_tool_calls(self) -> None:
        model = FakeChatModel(
            [
                {
                    "role": "assistant",
                    "content": "final",
                    "reasoning_content": "private thinking",
                }
            ]
        )
        action, state = ModelNode(model=model, messages=build_messages).exec(
            {"history": [{"role": "user", "content": "hi"}]}
        )

        self.assertEqual(action, "default")
        self.assertEqual(state["assistant_message"]["reasoning_content"], "private thinking")
        self.assertNotIn("reasoning_content", state["history"][-1])

        tool_call_model = FakeChatModel(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "think",
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
            ]
        )
        action, state = ModelNode(model=tool_call_model, messages=build_messages).exec(
            {"history": [{"role": "user", "content": "weather?"}]}
        )

        self.assertEqual(state["history"][-1]["reasoning_content"], "think")

    def test_agent_runs_default_tool_loop(self) -> None:
        model = FakeChatModel(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I should use the weather tool.",
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
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
                    "reasoning_content": "The tool result answers the question.",
                    "usage": {"prompt_tokens": 8, "completion_tokens": 3},
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
        self.assertEqual(
            result.context.messages[2]["reasoning_content"],
            "I should use the weather tool.",
        )
        self.assertNotIn("tool_calls", result.context.messages[-1])
        self.assertNotIn("reasoning_content", result.context.messages[-1])
        self.assertEqual(
            [event.type for event in result.context.events if event.category == "tool"],
            ["tool.observe", "tool.call", "tool.result", "tool.observe"],
        )
        self.assertEqual(result.usage.to_dict()["total_tokens"], 16)


if __name__ == "__main__":
    unittest.main()
