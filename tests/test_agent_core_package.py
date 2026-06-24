import unittest
from typing import Annotated

from agent_core import Agent, CallableNode, Flow, ModelNode, Node, Tool, ToolRouterNode, tool


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
            action=None,
        )
        target = CallableNode(lambda payload: {"ok": True})
        inner_agent - "special" >> target

        result = Flow(inner_agent).run({})

        self.assertEqual(result.action, "default")
        self.assertEqual(result.payload["ok"], True)
        self.assertEqual(result.path, ["Agent", "CallableNode"])

    def test_nested_agent_flow_inherits_trace(self) -> None:
        inner_agent = Agent(
            Flow(CallableNode(lambda payload: ("done", payload))),
            action=None,
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

    def test_chat_works_without_instructions(self) -> None:
        class FakeChatModel:
            def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
                return {"role": "assistant", "content": f"saw {messages[-1]['content']}"}

        agent = Agent(model=FakeChatModel())

        self.assertEqual(agent.chat("hello"), "saw hello")


if __name__ == "__main__":
    unittest.main()
