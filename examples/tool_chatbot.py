import argparse
import json
import sys
from typing import Annotated, Any, Literal

from agent_core import (
    Agent,
    Tool,
    TraceOptions,
    build_tool_agent_flow,
    make_trace_options,
    tool,
)


@tool(description="Look up demo weather for a supported city.")
def get_weather(
    city: Annotated[
        Literal["Shanghai", "Tokyo"],
        "English city name.",
    ],
) -> dict[str, str]:
    weather = {
        "Shanghai": {"condition": "sunny", "temperature": "24C"},
        "Tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    return {"city": city, **weather[city], "source": "mock"}


@tool(description="Return a short demo joke for the requested topic.")
def tell_joke(
    topic: Annotated[str, "Optional joke topic."] = "weather",
) -> dict[str, str]:
    return {
        "topic": topic,
        "joke": "I asked the cloud for a forecast. It said it was feeling under the weather.",
        "source": "mock",
    }


class RuleBasedDemoModel:
    """Small local model stub that emits OpenAI-style assistant messages."""

    def chat_message(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del tools, tool_choice, kwargs
        if messages and messages[-1]["role"] == "tool":
            return self._answer_from_tool_results(messages)

        text = _last_user_text(messages).lower()
        tool_calls = []
        if "shanghai" in text:
            tool_calls.append(_tool_call("call_weather_shanghai", "get_weather", {"city": "Shanghai"}))
        if "tokyo" in text:
            tool_calls.append(_tool_call("call_weather_tokyo", "get_weather", {"city": "Tokyo"}))
        if "joke" in text:
            tool_calls.append(_tool_call("call_joke", "tell_joke", {"topic": "weather"}))

        if tool_calls:
            return {"role": "assistant", "content": "", "tool_calls": tool_calls}
        return {
            "role": "assistant",
            "content": "Ask me about Shanghai weather, Tokyo weather, or a weather joke.",
        }

    def _answer_from_tool_results(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        results = [
            json.loads(message["content"])
            for message in messages
            if message["role"] == "tool"
        ]
        lines = []
        for result in results:
            if "condition" in result:
                lines.append(
                    f"{result['city']}: {result['condition']}, {result['temperature']}."
                )
            if "joke" in result:
                lines.append(result["joke"])
        return {"role": "assistant", "content": " ".join(lines)}


def build_tools() -> list[Tool]:
    return [get_weather, tell_joke]


def build_agent() -> Agent:
    return Agent(
        build_tool_agent_flow(
            model=RuleBasedDemoModel(),
            messages=lambda payload: payload.get("history", []),
            tools=build_tools(),
            chat_kwargs={"tool_choice": "auto"},
        )
    )


def run_turn(
    agent: Agent,
    history: list[dict[str, str]],
    user_input: str,
    *,
    trace: TraceOptions | None = None,
) -> bool:
    if user_input.lower() in {"exit", "quit", "q"}:
        return False

    history.append({"role": "user", "content": user_input})
    result = agent.run({"history": history}, trace=trace)
    history[:] = [
        message
        for message in result.payload["history"]
        if message["role"] in {"user", "assistant"}
    ]
    safe_print(result.payload["answer"])
    return True


def run_scripted_demo(trace: TraceOptions | None = None) -> None:
    agent = build_agent()
    history: list[dict[str, str]] = []
    for user_input in [
        "What is the weather in Shanghai?",
        "Compare Shanghai and Tokyo weather, then tell one joke.",
    ]:
        print(f"> {user_input}")
        run_turn(agent, history, user_input, trace=trace)


def run_interactive(trace: TraceOptions | None = None) -> None:
    agent = build_agent()
    history: list[dict[str, str]] = []
    print("agent-core tool chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if not run_turn(agent, history, user_input, trace=trace):
            print("bye")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agent-core tool chatbot example.")
    parser.add_argument("--demo", action="store_true", help="Run scripted demo turns.")
    parser.add_argument("--trace", action="store_true", help="Print trace events while the flow runs.")
    parser.add_argument(
        "--trace-events",
        default="node,tool,flow",
        help="Comma-separated trace categories: node, tool, flow.",
    )
    return parser.parse_args()


def build_trace_options(args: argparse.Namespace) -> TraceOptions:
    categories = [item.strip() for item in args.trace_events.split(",") if item.strip()]
    return make_trace_options(
        enabled=args.trace,
        include=categories,
        print_to_console=args.trace,
        printer=safe_print,
    )


def _tool_call(id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return str(message.get("content", ""))
    return ""


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


if __name__ == "__main__":
    args = parse_args()
    trace_options = build_trace_options(args)
    if args.demo:
        run_scripted_demo(trace=trace_options)
    else:
        run_interactive(trace=trace_options)

