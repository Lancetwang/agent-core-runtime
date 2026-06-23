import unittest

from agent_core import Agent, CallableNode, Flow, ModelNode, Tool, ToolRouterNode, tool


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

    def test_agent_core_exports_agent_loop_nodes(self) -> None:
        self.assertEqual(ModelNode.__name__, "ModelNode")
        self.assertEqual(ToolRouterNode.__name__, "ToolRouterNode")


if __name__ == "__main__":
    unittest.main()
