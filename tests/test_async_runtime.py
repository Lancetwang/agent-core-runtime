import asyncio
import time
import unittest
from typing import Annotated

from agent_core import (
    Agent,
    CallableNode,
    ExecResult,
    Flow,
    RunContext,
    Tool,
    ToolCall,
    ToolExecutor,
    report_tool_progress,
    tool,
)


class AsyncRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_flow_arun_routes_native_async_callable(self) -> None:
        async def classify(payload: dict) -> ExecResult:
            await asyncio.sleep(0)
            return ExecResult("next", {**payload, "classified": True})

        finish = CallableNode(lambda payload: {**payload, "finished": True})
        start = CallableNode(classify)
        start - "next" >> finish

        result = await Flow(start).arun({"value": 1})

        self.assertEqual(result.path, ["CallableNode", "CallableNode"])
        self.assertTrue(result.payload["classified"])
        self.assertTrue(result.payload["finished"])

    async def test_sync_callable_does_not_block_async_flow(self) -> None:
        ticked = asyncio.Event()

        def blocking(payload: dict) -> dict:
            time.sleep(0.05)
            return payload

        async def ticker() -> None:
            await asyncio.sleep(0.01)
            ticked.set()

        ticker_task = asyncio.create_task(ticker())
        await Flow(CallableNode(blocking)).arun({})
        await ticker_task

        self.assertTrue(ticked.is_set())

    async def test_agent_achat_runs_async_model_tool_loop_and_live_progress(self) -> None:
        @tool(description="Echo text asynchronously.", parallel=True)
        async def echo(text: Annotated[str, "Text to echo."]) -> str:
            report_tool_progress("halfway")
            await asyncio.sleep(0)
            return text

        class AsyncModel:
            def __init__(self) -> None:
                self.requests: list[dict] = []

            async def achat_message(self, messages, *, tools=None, **kwargs):
                self.requests.append({"messages": list(messages), "tools": list(tools or [])})
                if len(self.requests) == 1:
                    return {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_echo",
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"text":"hello"}',
                                },
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 4,
                            "completion_tokens": 1,
                            "prompt_tokens_details": {"cached_tokens": 3},
                        },
                    }
                return {
                    "role": "assistant",
                    "content": "Echoed hello.",
                    "usage": {"prompt_tokens": 6, "completion_tokens": 2},
                }

        live_events = []
        context = RunContext(on_event=live_events.append)
        model = AsyncModel()
        agent = Agent(model=model, tools=[echo], instructions="Use tools.")

        answer = await agent.achat("echo hello", context=context)

        self.assertEqual(answer, "Echoed hello.")
        scoped_messages = next(iter(context.message_scopes.values()))
        self.assertEqual(
            [message["role"] for message in scoped_messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(model.requests[0]["tools"][0]["function"]["name"], "echo")
        self.assertIn("tool.progress", [event.type for event in live_events])
        self.assertNotIn("tool.progress", [event.type for event in context.events])
        self.assertEqual(context.usage.to_dict()["cached_tokens"], 3)
        sequences = [event.seq for event in context.events]
        self.assertEqual(sequences, sorted(set(sequences)))

    async def test_async_tool_batches_keep_serial_calls_as_barriers(self) -> None:
        active: set[str] = set()
        serial_overlapped = False

        def parallel_tool(name: str) -> Tool:
            async def run() -> str:
                active.add(name)
                try:
                    await asyncio.sleep(0.01)
                    return name
                finally:
                    active.remove(name)

            return Tool(
                name=name,
                description=name,
                parameters={"type": "object"},
                fn=run,
                parallel=True,
            )

        def serial() -> str:
            nonlocal serial_overlapped
            serial_overlapped = bool(active)
            return "serial"

        executor = ToolExecutor(
            [
                parallel_tool("one"),
                parallel_tool("two"),
                Tool(
                    name="serial",
                    description="serial",
                    parameters={"type": "object"},
                    fn=serial,
                ),
                parallel_tool("three"),
            ]
        )
        calls = [
            ToolCall(id=str(index), name=name, arguments={})
            for index, name in enumerate(["one", "two", "serial", "three"])
        ]

        results = await executor.aexecute_all(calls)

        self.assertFalse(serial_overlapped)
        self.assertEqual(
            [result.content for result in results],
            ["one", "two", "serial", "three"],
        )

    async def test_task_cancellation_emits_cancel_events(self) -> None:
        started = asyncio.Event()
        context = RunContext()

        async def wait_forever(payload: dict) -> dict:
            started.set()
            await asyncio.Event().wait()
            return payload

        task = asyncio.create_task(Flow(CallableNode(wait_forever)).arun({}, context=context))
        await started.wait()
        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(context.events[-2].type, "node.cancel")
        self.assertEqual(context.events[-1].type, "flow.cancel")


if __name__ == "__main__":
    unittest.main()
