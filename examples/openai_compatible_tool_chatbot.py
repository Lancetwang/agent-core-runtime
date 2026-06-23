import argparse
import os
import sys
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from openai import OpenAI

from agent_core import (
    Agent,
    Tool,
    TraceOptions,
    build_tool_agent_flow,
    make_trace_options,
    tool,
)


SYSTEM_PROMPT = """
You are a small CLI assistant using the agent-core runtime.

# Tools

## get_weather
Look up mock weather for Shanghai or Tokyo.
Use this whenever the user asks for weather for either city.

## tell_joke
Return a short mock joke for a requested topic.
Use this whenever the user asks for a joke.

# Response

Answer naturally after tool results are available.
""".strip()


@tool(description="Look up mock weather for Shanghai or Tokyo.")
def get_weather(
    city: Annotated[
        Literal["Shanghai", "Tokyo"],
        "English city name. Normalize user language to one of: Shanghai, Tokyo.",
    ],
) -> dict[str, str]:
    weather = {
        "Shanghai": {"condition": "sunny", "temperature": "24C"},
        "Tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    return {"city": city, **weather[city], "source": "mock"}


@tool(description="Return a short mock joke for the requested topic.")
def tell_joke(
    topic: Annotated[
        str,
        "Joke topic, normalized to concise English.",
    ] = "weather",
) -> dict[str, str]:
    return {
        "topic": topic,
        "joke": "I asked the cloud for a forecast. It said it was feeling under the weather.",
        "source": "mock",
    }


class OpenAICompatibleChatModel:
    """ChatModel adapter for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat_message(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            **kwargs,
        }
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        extra_body = dict(request.pop("extra_body", {}) or {})
        thinking = os.getenv("OPENAI_THINKING") or os.getenv("DEEPSEEK_THINKING")
        if thinking:
            extra_body.setdefault("thinking", {"type": thinking})
        elif _is_deepseek_url(str(self.client.base_url)):
            extra_body.setdefault("thinking", {"type": "disabled"})
        if extra_body:
            request["extra_body"] = extra_body

        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        result: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            result["tool_calls"] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in tool_calls
            ]

        usage = getattr(response, "usage", None)
        if usage is not None:
            result["usage"] = usage.model_dump() if hasattr(usage, "model_dump") else usage
        return result


def build_tools() -> list[Tool]:
    return [get_weather, tell_joke]


def build_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, *payload.get("history", [])]


def build_agent() -> Agent:
    return Agent(
        build_tool_agent_flow(
            model=build_model_from_env(),
            messages=build_messages,
            tools=build_tools(),
            chat_kwargs={"temperature": 0, "tool_choice": "auto"},
        )
    )


def build_model_from_env() -> OpenAICompatibleChatModel:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env.")
    return OpenAICompatibleChatModel(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com",
        model=os.getenv("OPENAI_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-v4-flash",
    )


def run_turn(
    agent: Agent,
    history: list[dict[str, Any]],
    user_input: str,
    *,
    trace: TraceOptions | None = None,
) -> bool:
    if user_input.lower() in {"exit", "quit", "q"}:
        return False

    history.append({"role": "user", "content": user_input})
    result = agent.run({"history": history}, trace=trace, max_steps=12)
    history[:] = result.payload["history"]
    safe_print(result.payload["answer"])
    return True


def run_scripted_demo(trace: TraceOptions | None = None) -> None:
    agent = build_agent()
    history: list[dict[str, Any]] = []
    for user_input in [
        "What is the weather in Shanghai?",
        "Compare Shanghai and Tokyo weather, then tell one joke.",
    ]:
        print(f"> {user_input}")
        run_turn(agent, history, user_input, trace=trace)


def run_interactive(trace: TraceOptions | None = None) -> None:
    agent = build_agent()
    history: list[dict[str, Any]] = []
    print("agent-core OpenAI-compatible chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if not run_turn(agent, history, user_input, trace=trace):
            print("bye")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real OpenAI-compatible tool chatbot example."
    )
    parser.add_argument("--demo", action="store_true", help="Run scripted demo turns.")
    parser.add_argument("--trace", action="store_true", help="Print trace events.")
    parser.add_argument(
        "--trace-events",
        default="node,tool,flow,model",
        help="Comma-separated trace categories.",
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


def _is_deepseek_url(base_url: str) -> bool:
    return "deepseek" in base_url.lower()


def safe_print(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    args = parse_args()
    trace_options = build_trace_options(args)
    if args.demo:
        run_scripted_demo(trace=trace_options)
    else:
        run_interactive(trace=trace_options)
