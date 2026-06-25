from __future__ import annotations

import argparse
import sys
from typing import Annotated

from agent_core import Agent, tool

INSTRUCTIONS = """
You are a compact demo agent. Answer naturally.

Tools
- get_weather: get mock weather for a city.

Use get_weather whenever the user asks about weather. After the tool result
returns, answer with the mocked weather data.
""".strip()


@tool(description="Get mock weather for a city. Use this for any weather question.")
def get_weather(location: Annotated[str, "City name to query."]) -> str:
    return f"{location} weather is sunny, 24C."


def print_delta(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal Agent(instructions, tools) chat.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    stream = not args.no_stream
    agent = Agent(
        instructions=INSTRUCTIONS,
        tools=[get_weather],
        stream=stream,
        chat_kwargs={"temperature": 0.2, "max_tokens": 600, "tool_choice": "auto"},
    )
    context = agent.new_context()
    context.metadata["agent_name"] = "weather-demo"

    print("chat. Type 'exit' to quit.")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if not user_input:
            continue

        answer = agent.chat(
            user_input,
            context=context,
            max_steps=12,
            on_delta=print_delta if stream else None,
        )
        if stream:
            print()
        else:
            print(answer)


if __name__ == "__main__":
    main()
