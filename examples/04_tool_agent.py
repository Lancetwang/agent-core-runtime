from __future__ import annotations

import argparse
from typing import Annotated, Literal

from agent_core import Agent, AgentEvent, RunContext, build_tool_agent_flow, tool
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


def build_context(*, stream_events: bool = False) -> RunContext:
    context = RunContext(on_event=print_event if stream_events else None)
    context.metadata["example"] = "04_tool_agent"
    context.add_message("system", SYSTEM_PROMPT)
    return context


def build_agent() -> Agent:
    return Agent(
        build_tool_agent_flow(
            model=build_demo_model(),
            messages=None,
            tools=[get_weather, tell_joke],
            chat_kwargs={"temperature": 0.2, "max_tokens": 500, "tool_choice": "auto"},
        )
    )


def run_turn(agent: Agent, context: RunContext, user_input: str) -> str:
    turn_index = int(context.metadata.get("turn_count", 0)) + 1
    context.metadata["turn_count"] = turn_index
    context.add_message("user", user_input)

    result = agent.run(
        {"turn": turn_index},
        context=context,
        trace=False,
        max_steps=12,
    )
    answer = str(result.payload.get("answer", ""))
    context.set_artifact("last_answer", answer)
    return answer


def run_demo(stream_events: bool) -> None:
    agent = build_agent()
    context = build_context(stream_events=stream_events)
    questions = [
        "What is the weather in Shanghai?",
        "Compare Shanghai and Tokyo weather, then tell one weather joke.",
    ]
    for question in questions:
        safe_print(f"> {question}")
        safe_print(run_turn(agent, context, question))
        safe_print(f"context messages: {len(context.messages)}")
        safe_print(f"context events: {len(context.events)}")
        safe_print()


def run_interactive(stream_events: bool) -> None:
    agent = build_agent()
    context = build_context(stream_events=stream_events)
    safe_print("agent-core tool agent. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            safe_print("bye")
            return
        if user_input:
            safe_print(run_turn(agent, context, user_input))
            safe_print(f"[context] messages={len(context.messages)} events={len(context.events)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real OpenAI-compatible tool agent.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat.")
    parser.add_argument("--events", action="store_true", help="Stream RunContext events.")
    return parser.parse_args()


def print_event(event: AgentEvent) -> None:
    if event.category not in {"node", "model", "tool", "flow", "artifact"}:
        return
    parts = [f"[event:{event.category}]", event.type]
    if event.step is not None:
        parts.append(f"step={event.step}")
    if event.node:
        parts.append(f"node={event.node}")
    if event.action:
        parts.append(f"action={event.action}")
    if event.data:
        parts.append(f"data={event.data}")
    safe_print(" ".join(parts))


if __name__ == "__main__":
    args = parse_args()
    if args.interactive:
        run_interactive(stream_events=args.events)
    else:
        run_demo(stream_events=args.events)
