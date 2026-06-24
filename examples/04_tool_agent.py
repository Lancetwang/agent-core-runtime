from __future__ import annotations

import argparse
import json
from typing import Annotated, Literal

from agent_core import Agent, RunContext, build_tool_agent_flow, make_trace_options, tool
from _openai_compatible import build_demo_model, safe_print


SYSTEM_PROMPT = """
You are a small demo agent.

Tools
- get_weather: query mock weather for Shanghai or Tokyo.
- tell_joke: return a short mock joke.

Use tools when the user asks for weather or jokes. After tool results return,
answer naturally and mention that tool data is mocked.
""".strip()


@tool(description="Look up mock weather for Shanghai or Tokyo.")
def get_weather(
    city: Annotated[Literal["Shanghai", "Tokyo"], "City to query."],
) -> dict[str, str]:
    data = {
        "Shanghai": {"condition": "sunny", "temperature": "24C"},
        "Tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    return {"city": city, **data[city], "source": "mock"}


@tool(description="Return a short mock joke for a requested topic.")
def tell_joke(
    topic: Annotated[str, "Joke topic."] = "weather",
) -> dict[str, str]:
    return {
        "topic": topic,
        "joke": "The weather report tried stand-up, but its delivery was scattered.",
        "source": "mock",
    }


def build_messages(payload: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, *payload.get("history", [])]


def build_agent() -> Agent:
    return Agent(
        build_tool_agent_flow(
            model=build_demo_model(),
            messages=build_messages,
            tools=[get_weather, tell_joke],
            chat_kwargs={"temperature": 0.2, "max_tokens": 500, "tool_choice": "auto"},
        )
    )


def run_turn(agent: Agent, history: list[dict], user_input: str, *, trace: bool) -> str:
    history.append({"role": "user", "content": user_input})
    context = RunContext()
    result = agent.run(
        {"history": history},
        context=context,
        trace=make_trace_options(
            enabled=trace,
            include=["node", "model", "tool", "flow"],
            print_to_console=trace,
            printer=safe_print,
        ),
        max_steps=12,
    )
    history[:] = result.payload["history"]
    return str(result.payload.get("answer", ""))


def run_demo(trace: bool) -> None:
    agent = build_agent()
    history: list[dict] = []
    questions = [
        "What is the weather in Shanghai?",
        "Compare Shanghai and Tokyo weather, then tell one weather joke.",
    ]
    for question in questions:
        safe_print(f"> {question}")
        safe_print(run_turn(agent, history, question, trace=trace))
        safe_print()


def run_interactive(trace: bool) -> None:
    agent = build_agent()
    history: list[dict] = []
    safe_print("agent-core tool agent. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            safe_print("bye")
            return
        if user_input:
            safe_print(run_turn(agent, history, user_input, trace=trace))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real OpenAI-compatible tool agent.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat.")
    parser.add_argument("--trace", action="store_true", help="Print runtime events.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.interactive:
        run_interactive(trace=args.trace)
    else:
        run_demo(trace=args.trace)
