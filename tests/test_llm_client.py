import unittest
from types import SimpleNamespace

from agent_core import LLM


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


class FakeAsyncCompletions:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    async def create(self, **kwargs):
        self.last_request = kwargs
        return self.response


class FakeAsyncClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeAsyncCompletions(response))


class AsyncChunks:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            yield chunk


class LLMTests(unittest.TestCase):
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
                    message=SimpleNamespace(
                        content="",
                        reasoning_content="I should check the weather.",
                        tool_calls=[tool_call],
                    ),
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
        model = LLM(
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
        self.assertEqual(message["reasoning_content"], "I should check the weather.")
        self.assertEqual(message["tool_calls"], [tool_call])
        self.assertEqual(message["usage"]["total_tokens"], 5)
        self.assertEqual(client.chat.completions.last_request["model"], "demo-model")
        self.assertEqual(client.chat.completions.last_request["tool_choice"], "auto")
        self.assertEqual(client.chat.completions.last_request["temperature"], 0)

    def test_extra_body_is_sent(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                )
            ],
            usage=None,
        )
        client = FakeClient(response)
        model = LLM(
            api_key="test",
            base_url="https://api.deepseek.com",
            model="demo-model",
            client=client,
            extra_body={"thinking": {"type": "disabled"}},
        )

        message = model.chat_message([{"role": "user", "content": "hello"}])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(
            client.chat.completions.last_request["extra_body"],
            {"thinking": {"type": "disabled"}},
        )

    def test_streaming_chat_message_aggregates_content(self) -> None:
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Hello",
                            tool_calls=[],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=" world",
                            tool_calls=[],
                        )
                    )
                ],
                usage=SimpleNamespace(
                    model_dump=lambda: {
                        "prompt_tokens": 4,
                        "completion_tokens": 3,
                        "total_tokens": 7,
                    }
                ),
            ),
        ]
        client = FakeClient(chunks)
        model = LLM(
            api_key="test",
            base_url="https://api.example.com",
            model="demo-model",
            client=client,
        )
        deltas = []

        message = model.chat_message(
            [{"role": "user", "content": "hello"}],
            stream=True,
            on_delta=deltas.append,
        )

        self.assertEqual(message["content"], "Hello world")
        self.assertEqual(deltas, ["Hello", " world"])
        self.assertEqual(message["usage"]["total_tokens"], 7)
        self.assertNotIn("tool_calls", message)
        self.assertTrue(client.chat.completions.last_request["stream"])
        self.assertEqual(
            client.chat.completions.last_request["stream_options"],
            {"include_usage": True},
        )

    def test_streaming_chat_message_preserves_tool_calls(self) -> None:
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="",
                            reasoning_content="Check ",
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="get_weather",
                                        arguments="",
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content="the weather.",
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    type=None,
                                    function=SimpleNamespace(
                                        name=None,
                                        arguments='{"city": "Bei',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    type=None,
                                    function=SimpleNamespace(
                                        name=None,
                                        arguments='jing"}',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
        ]
        client = FakeClient(chunks)
        model = LLM(
            api_key="test",
            base_url="https://api.example.com",
            model="demo-model",
            client=client,
        )

        reasoning_deltas = []
        message = model.chat_message(
            [{"role": "user", "content": "weather"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
            on_reasoning_delta=reasoning_deltas.append,
        )

        self.assertEqual(
            message["tool_calls"],
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Beijing"}',
                    },
                }
            ],
        )

        self.assertEqual(message["reasoning_content"], "Check the weather.")
        self.assertEqual(reasoning_deltas, ["Check ", "the weather."])

    def test_indexless_streamed_tool_calls_stay_separate(self) -> None:
        def chunk(tool_calls):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=tool_calls))
                ]
            )

        def call_delta(*, call_id=None, name=None, arguments=None):
            return SimpleNamespace(
                index=None,
                id=call_id,
                type="function" if call_id else None,
                function=SimpleNamespace(name=name, arguments=arguments),
            )

        chunks = [
            chunk([call_delta(call_id="a1", name="Read")]),
            chunk([call_delta(arguments='{"path":')]),
            chunk([call_delta(arguments='"x.txt"}')]),
            chunk(
                [
                    call_delta(
                        call_id="b2",
                        name="Bash",
                        arguments='{"command":"ls"}',
                    )
                ]
            ),
            chunk([call_delta(call_id="b2", name="Bash")]),
            chunk(
                [
                    call_delta(
                        call_id="c3",
                        name="Glob",
                        arguments='{"pattern":"*"}',
                    )
                ]
            ),
        ]
        model = LLM(
            api_key="test",
            model="demo-model",
            client=FakeClient(chunks),
        )

        message = model.chat_message(
            [{"role": "user", "content": "inspect"}],
            stream=True,
        )

        self.assertEqual(
            [
                (
                    item["id"],
                    item["function"]["name"],
                    item["function"]["arguments"],
                )
                for item in message["tool_calls"]
            ],
            [
                ("a1", "Read", '{"path":"x.txt"}'),
                ("b2", "Bash", '{"command":"ls"}'),
                ("c3", "Glob", '{"pattern":"*"}'),
            ],
        )

    def test_parallel_indexless_fragments_follow_chunk_position(self) -> None:
        def chunk(tool_calls):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=tool_calls))
                ]
            )

        def call_delta(*, call_id=None, name=None, arguments=None):
            return SimpleNamespace(
                index=None,
                id=call_id,
                type="function" if call_id else None,
                function=SimpleNamespace(name=name, arguments=arguments),
            )

        chunks = [
            chunk(
                [
                    call_delta(call_id="duplicate", name="first", arguments='{"a":'),
                    call_delta(call_id="duplicate", name="second", arguments='{"b":'),
                ]
            ),
            chunk(
                [
                    call_delta(arguments="1}"),
                    call_delta(arguments="2}"),
                ]
            ),
        ]
        model = LLM(api_key="test", model="demo-model", client=FakeClient(chunks))

        message = model.chat_message([{"role": "user", "content": "run both"}], stream=True)

        self.assertEqual(
            [
                (item["id"], item["function"]["name"], item["function"]["arguments"])
                for item in message["tool_calls"]
            ],
            [
                ("duplicate", "first", '{"a":1}'),
                ("duplicate_1", "second", '{"b":2}'),
            ],
        )


class AsyncLLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_streaming_client_aggregates_content_and_usage(self) -> None:
        chunks = AsyncChunks(
            [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="hello", tool_calls=[]))]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(delta=SimpleNamespace(content=" async", tool_calls=[]))
                    ],
                    usage={"prompt_tokens": 2, "completion_tokens": 1},
                ),
            ]
        )
        client = FakeAsyncClient(chunks)
        model = LLM(model="demo-model", async_client=client)
        deltas = []

        message = await model.achat_message(
            [{"role": "user", "content": "hello"}],
            stream=True,
            on_delta=deltas.append,
        )

        self.assertEqual(message["content"], "hello async")
        self.assertEqual(message["usage"], {"prompt_tokens": 2, "completion_tokens": 1})
        self.assertEqual(deltas, ["hello", " async"])
        self.assertTrue(client.chat.completions.last_request["stream"])


if __name__ == "__main__":
    unittest.main()
