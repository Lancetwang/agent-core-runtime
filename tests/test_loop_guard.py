import unittest
from typing import Annotated

from agent_core import Agent, PayloadKeys, RunContext, ToolLoopGuardNode, tool


class LoopGuardTests(unittest.TestCase):
    def test_malformed_prior_guard_state_is_ignored(self) -> None:
        node = ToolLoopGuardNode()
        action, state = node.exec(
            {
                PayloadKeys.LOOP_GUARD: {"rounds": None, "warned": "invalid"},
                PayloadKeys.ASSISTANT_MESSAGE: {},
            }
        )

        self.assertEqual(action, "continue")
        self.assertEqual(state[PayloadKeys.LOOP_GUARD], {"rounds": [], "warned": {}})

    def test_repeated_tool_results_warn_then_force_text_only_finish(self) -> None:
        executions = 0

        @tool(description="Inspect an unchanged value.")
        def inspect_value(value: Annotated[str, "Value to inspect."]) -> dict[str, str]:
            nonlocal executions
            executions += 1
            return {"value": value, "status": "unchanged"}

        class RepeatingModel:
            def __init__(self) -> None:
                self.requests = 0
                self.tool_counts: list[int] = []

            def chat_message(self, messages, *, tools=None, **kwargs):
                self.requests += 1
                self.tool_counts.append(len(tools or []))
                if tools:
                    return {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"inspect_{self.requests}",
                                "type": "function",
                                "function": {
                                    "name": "inspect_value",
                                    "arguments": '{"value":"same"}',
                                },
                            }
                        ],
                    }
                return {"role": "assistant", "content": "No further progress is possible."}

        model = RepeatingModel()
        context = RunContext()
        agent = Agent(model=model, tools=[inspect_value])

        answer = agent.chat("keep checking", context=context)

        self.assertEqual(answer, "No further progress is possible.")
        self.assertEqual(model.requests, 5)
        self.assertEqual(executions, 4)
        self.assertEqual(model.tool_counts, [1, 1, 1, 1, 0])
        self.assertEqual(
            [event.type for event in context.events if event.type.startswith("loop.")],
            ["loop.warning", "loop.guard"],
        )
        self.assertEqual(
            [
                message["agent_internal"]
                for message in context.messages
                if message.get("agent_internal")
            ],
            [True, True],
        )


if __name__ == "__main__":
    unittest.main()
