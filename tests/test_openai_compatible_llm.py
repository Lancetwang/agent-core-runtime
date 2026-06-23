import unittest
from types import SimpleNamespace

from agent_core import OpenAICompatibleChatModel


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self.response


class FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


class OpenAICompatibleChatModelTests(unittest.TestCase):
    def test_chat_message_returns_openai_style_message(self) -> None:
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Shanghai"}',
            },
        }
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="", tool_calls=[tool_call]),
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                }
            ),
        )
        client = FakeClient(response)
        model = OpenAICompatibleChatModel(
            api_key="test",
            base_url="https://api.example.com",
            model="demo-model",
            client=client,
        )

        message = model.chat_message(
            [{"role": "user", "content": "weather"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
            temperature=0,
        )

        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["tool_calls"], [tool_call])
        self.assertEqual(message["usage"]["total_tokens"], 5)
        self.assertEqual(client.chat.completions.last_request["model"], "demo-model")
        self.assertEqual(client.chat.completions.last_request["tool_choice"], "auto")
        self.assertEqual(client.chat.completions.last_request["temperature"], 0)

    def test_default_extra_body_is_sent(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                )
            ],
            usage=None,
        )
        client = FakeClient(response)
        model = OpenAICompatibleChatModel(
            api_key="test",
            base_url="https://api.deepseek.com",
            model="demo-model",
            client=client,
            default_extra_body={"thinking": {"type": "disabled"}},
        )

        message = model.chat_message([{"role": "user", "content": "hello"}])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(
            client.chat.completions.last_request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )


if __name__ == "__main__":
    unittest.main()

