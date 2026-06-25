import os
import unittest
from typing import Annotated

from agent_core import Agent, CallableNode, Flow, ModelNode, Node, ToolRouterNode, tool


class AgentCorePackageTests(unittest.TestCase):
    def test_agent_core_can_be_used_directly(self) -> None:
        node = CallableNode(lambda payload: {"message": payload["message"].upper()})
        result = Agent(Flow(node)).run({"message": "ok"})

        self.assertEqual(result.payload["message"], "OK")

    def test_agent_core_exports_tool_decorator(self) -> None:
        @tool(description="Echo text.")
        def echo(text: str) -> str:
            return text

        self.assertEqual(echo.execute(text="hello"), "hello")
        self.assertEqual(echo.to_llm_format()["function"]["name"], "echo")

    def test_agent_core_exports_llm_nodes(self) -> None:
        self.assertEqual(ModelNode.__name__, "ModelNode")
        self.assertEqual(ToolRouterNode.__name__, "ToolRouterNode")

    def test_agent_can_be_declared_from_model_prompt_and_tools(self) -> None:
        class FakeChatModel:
            def __init__(self) -> None:
                self.requests = []
                self.responses = [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"text": "hello"}',
                                },
                            }
                        ],
                    },
                    {"role": "assistant", "content": "Echoed hello."},
                ]

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

        @tool(description="Echo text.")
        def echo(text: Annotated[str, "Text to echo."]) -> str:
            return text

        model = FakeChatModel()
        agent = Agent(
            model=model,
            instructions="Use tools when helpful.",
            tools=[echo],
            chat_kwargs={"tool_choice": "auto"},
        )
        context = agent.new_context()

        answer = agent.chat("please echo hello", context=context)

        self.assertEqual(answer, "Echoed hello.")
        self.assertEqual(context.messages[0]["role"], "system")
        self.assertEqual(context.messages[1]["role"], "user")
        self.assertEqual(model.requests[0]["messages"][0]["role"], "system")
        self.assertEqual(model.requests[0]["tools"][0]["function"]["name"], "echo")

    def test_agent_can_be_used_as_a_node(self) -> None:
        def inner_step(payload: dict) -> dict:
            payload["inner"] = "done"
            return payload

        def outer_step(payload: dict) -> dict:
            payload["outer"] = f"saw {payload['inner']}"
            return payload

        inner_agent = Agent(Flow(CallableNode(inner_step)), action="complete")
        outer_node = CallableNode(outer_step)
        inner_agent - "complete" >> outer_node

        result = Flow(inner_agent).run({})

        self.assertIsInstance(inner_agent, Node)
        self.assertEqual(result.payload["inner"], "done")
        self.assertEqual(result.payload["outer"], "saw done")
        self.assertEqual(result.path, ["Agent", "CallableNode"])
        self.assertIn(
            "CallableNode",
            [event.node for event in result.context.events if event.category == "node"],
        )

    def test_agent_can_preserve_inner_flow_action(self) -> None:
        inner_agent = Agent(
            Flow(CallableNode(lambda payload: ("special", payload))),
        )
        target = CallableNode(lambda payload: {"ok": True})
        inner_agent - "special" >> target

        result = Flow(inner_agent).run({})

        self.assertEqual(result.action, "default")
        self.assertEqual(result.payload["ok"], True)
        self.assertEqual(result.path, ["Agent", "CallableNode"])

    def test_agent_rejects_none_action(self) -> None:
        with self.assertRaises(TypeError):
            Agent(Flow(CallableNode(lambda payload: payload)), action=None)

    def test_nested_agent_flow_inherits_trace(self) -> None:
        inner_agent = Agent(
            Flow(CallableNode(lambda payload: ("done", payload))),
        )

        result = Flow(inner_agent).run({}, trace=True)

        self.assertEqual(
            [(event.event, event.node) for event in result.trace if event.category == "node"],
            [
                ("node.start", "Agent"),
                ("node.start", "CallableNode"),
                ("node.end", "CallableNode"),
                ("node.end", "Agent"),
            ],
        )

    def test_nested_agents_add_each_instruction_once(self) -> None:
        first = Agent(
            Flow(CallableNode(lambda payload: {"seen": ["first"]})),
            instructions="First role.",
        )
        second = Agent(
            Flow(CallableNode(lambda payload: {"seen": [*payload["seen"], "second"]})),
            instructions="Second role.",
        )
        first >> second

        result = Flow(first).run({})
        system_messages = [
            message["content"]
            for message in result.context.messages
            if message["role"] == "system"
        ]

        self.assertEqual(result.payload["seen"], ["first", "second"])
        self.assertEqual(system_messages, ["First role.", "Second role."])

    def test_nested_chat_agents_use_isolated_message_scopes(self) -> None:
        class FakeChatModel:
            def __init__(self, content: str) -> None:
                self.content = content
                self.requests = []

            def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
                self.requests.append(list(messages))
                return {"role": "assistant", "content": self.content}

        first_model = FakeChatModel("first answer")
        second_model = FakeChatModel("second answer")
        first = Agent(model=first_model, instructions="First role.")
        second = Agent(model=second_model, instructions="Second role.")
        first - "final" >> second

        result = Flow(first).run({})

        self.assertEqual(result.path, ["Agent", "Agent"])
        self.assertEqual([message["content"] for message in first_model.requests[0]], ["First role."])
        self.assertEqual([message["content"] for message in second_model.requests[0]], ["Second role."])
        self.assertNotIn("First role.", [message["content"] for message in second_model.requests[0]])
        self.assertEqual(len(result.context.message_scopes), 2)

    def test_chat_works_without_instructions(self) -> None:
        class FakeChatModel:
            def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
                return {"role": "assistant", "content": f"saw {messages[-1]['content']}"}

        agent = Agent(model=FakeChatModel())

        self.assertEqual(agent.chat("hello"), "saw hello")

    def test_agent_chat_streams_by_default_and_can_be_overridden(self) -> None:
        class FakeChatModel:
            def __init__(self) -> None:
                self.stream_values = []

            def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
                self.stream_values.append(kwargs.get("stream"))
                on_delta = kwargs.get("on_delta")
                if on_delta:
                    on_delta("hi")
                return {"role": "assistant", "content": "hi"}

        model = FakeChatModel()
        deltas = []
        agent = Agent(model=model)

        self.assertEqual(agent.chat("hello", on_delta=deltas.append), "hi")
        self.assertEqual(agent.chat("again", stream=False), "hi")
        self.assertEqual(model.stream_values, [True, False])
        self.assertEqual(deltas, ["hi"])

    def test_agent_stream_constructor_option(self) -> None:
        class FakeChatModel:
            def __init__(self) -> None:
                self.stream_values = []

            def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
                self.stream_values.append(kwargs.get("stream"))
                return {"role": "assistant", "content": "ok"}

        model = FakeChatModel()
        agent = Agent(model=model, stream=False)

        self.assertEqual(agent.chat("hello"), "ok")
        self.assertEqual(model.stream_values, [False])

    def test_agent_can_use_default_env_llm(self) -> None:
        previous = os.environ.get("LLM_API_KEY")
        os.environ["LLM_API_KEY"] = "test"
        try:
            agent = Agent(instructions="Use the default env-backed LLM.")
        finally:
            if previous is None:
                os.environ.pop("LLM_API_KEY", None)
            else:
                os.environ["LLM_API_KEY"] = previous

        self.assertEqual(agent.flow.start.__class__.__name__, "ModelNode")


if __name__ == "__main__":
    unittest.main()
