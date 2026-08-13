"""Integration tests: multi-agent teams, cross-turn reuse, worker contexts."""

from __future__ import annotations

import json
import unittest

from agent_core import (
    Agent,
    Flow,
    RunContext,
    ToolExecutor,
    get_current_tool_call,
    tool,
)


def _openai_call(call_id: str, name: str, value: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"value": value})},
    }


class WorkerContextTests(unittest.TestCase):
    def test_parallel_tool_sees_its_own_call_in_worker_thread(self) -> None:
        @tool(description="Echo the active call id.", parallel=True)
        def echo_id(value: str) -> str:
            call = get_current_tool_call()
            return f"{value}:{call.id if call else 'none'}"

        executor = ToolExecutor([echo_id], max_workers=4)
        assistant = {
            "tool_calls": [_openai_call(f"call_{i}", "echo_id", str(i)) for i in range(3)]
        }

        results = executor.execute_all(executor.parse_tool_calls(assistant))

        self.assertEqual(
            [result.content for result in results],
            ["0:call_0", "1:call_1", "2:call_2"],
        )


class MultiAgentTeamTests(unittest.TestCase):
    def test_team_shares_events_and_usage_with_isolated_scopes(self) -> None:
        class FakeChatModel:
            def __init__(self, content: str, prompt: int, completion: int) -> None:
                self.content = content
                self.prompt = prompt
                self.completion = completion
                self.requests: list[list[dict]] = []

            def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
                self.requests.append(list(messages))
                return {
                    "role": "assistant",
                    "content": self.content,
                    "usage": {
                        "prompt_tokens": self.prompt,
                        "completion_tokens": self.completion,
                    },
                }

        first_model = FakeChatModel("from research", 5, 2)
        second_model = FakeChatModel("final text", 8, 3)
        first = Agent(model=first_model, instructions="Researcher.")
        second = Agent(model=second_model, instructions="Writer.")
        first - "final" >> second

        context = RunContext()
        result = Flow(first).run({}, context=context)

        self.assertEqual(result.payload.get("answer"), "final text")
        self.assertEqual(
            result.usage.to_dict(),
            {"requests": 2, "input_tokens": 13, "output_tokens": 5, "total_tokens": 18},
        )
        self.assertEqual(context.usage.to_dict()["total_tokens"], 18)
        self.assertEqual(len(context.message_scopes), 2)
        self.assertEqual(
            [message["content"] for message in first_model.requests[0]],
            ["Researcher."],
        )
        self.assertEqual(
            [message["content"] for message in second_model.requests[0]],
            ["Writer."],
        )
        self.assertNotIn(
            "Researcher.",
            [message["content"] for message in second_model.requests[0]],
        )


class ChatTurnTests(unittest.TestCase):
    def test_chat_turns_accumulate_usage_and_preserve_history(self) -> None:
        class FakeChatModel:
            def __init__(self) -> None:
                self.requests: list[list[dict]] = []
                self.responses = [
                    {
                        "role": "assistant",
                        "content": "one",
                        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                    },
                    {
                        "role": "assistant",
                        "content": "two",
                        "usage": {"prompt_tokens": 6, "completion_tokens": 2},
                    },
                ]

            def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs):
                self.requests.append(list(messages))
                return self.responses.pop(0)

        model = FakeChatModel()
        agent = Agent(model=model, stream=False)
        context = agent.new_context()

        self.assertEqual(agent.chat("hello", context=context), "one")
        self.assertEqual(agent.chat("again", context=context), "two")

        self.assertEqual(
            context.usage.to_dict(),
            {"requests": 2, "input_tokens": 9, "output_tokens": 3, "total_tokens": 12},
        )
        self.assertEqual(
            [message["content"] for message in model.requests[1]],
            ["hello", "one", "again"],
        )


if __name__ == "__main__":
    unittest.main()
