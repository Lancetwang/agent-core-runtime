from __future__ import annotations

import argparse
import json
import sys
from typing import Annotated, Literal

from agent_core import (
    Agent,
    CallableNode,
    Flow,
    ModelNode,
    RunContext,
    ToolCallNode,
    ToolExecutor,
    ToolRouterNode,
    tool,
)


SYSTEM_PROMPT = """
You are a small demo agent.

Tools
- get_weather: query mock weather for Shanghai or Tokyo.
- tell_joke: return a short mock joke.

Use tools when the user asks for weather or jokes. After tool results return,
answer naturally and mention that tool data is mocked.

When you call tools, keep that assistant message content empty. Do not stream
user-facing text until the tool results are available.
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


def build_context() -> RunContext:
    context = RunContext()
    context.metadata["example"] = "04_tool_agent"
    context.add_message("system", SYSTEM_PROMPT)
    return context


def build_agent() -> Agent:
    tools = [get_weather, tell_joke]
    model_node = ModelNode(
        messages=None,
        tools=tools,
        action="observe",
        chat_kwargs={"temperature": 0.2, "max_tokens": 500, "tool_choice": "auto"},
    )
    router_node = ToolRouterNode(tool_action="tool_call", done_action="final")
    tool_node = ToolCallNode(
        executor=ToolExecutor(tools),
        next_action="prepare_final",
    )
    prepare_final_node = CallableNode(prepare_final_response)

    model_node - "observe" >> router_node
    router_node - "tool_call" >> tool_node
    tool_node - "prepare_final" >> prepare_final_node
    prepare_final_node - "chat" >> model_node

    return Agent(Flow(model_node))


def prepare_final_response(payload: dict) -> tuple[str, dict]:
    state = dict(payload)
    if state.get("stream_answer"):
        state["chat_kwargs"] = {"stream": True, "on_delta": print_delta}
    return "chat", state


def run_turn(agent: Agent, context: RunContext, user_input: str, *, stream: bool) -> str:
    turn_index = int(context.metadata.get("turn_count", 0)) + 1
    context.metadata["turn_count"] = turn_index
    context.add_message("user", user_input)
    delta_count_before = count_model_deltas(context)

    result = agent.run(
        {"turn": turn_index, "stream_answer": stream},
        context=context,
        trace=False,
        max_steps=12,
    )
    answer = str(result.payload.get("answer", ""))
    context.set_artifact("last_answer", answer)
    context.metadata["last_answer_streamed"] = count_model_deltas(context) > delta_count_before
    return answer


def run_demo(*, stream_answer: bool, context_view: str) -> None:
    agent = build_agent()
    context = build_context()
    questions = [
        "What is the weather in Shanghai?",
        "Use tools to compare Shanghai and Tokyo weather, then tell one weather joke.",
    ]
    for question in questions:
        print(f"> {question}")
        answer = run_turn(agent, context, question, stream=stream_answer)
        if stream_answer and context.metadata.get("last_answer_streamed"):
            print()
        else:
            print(answer)
        print_context(context, context_view)
        print()


def run_interactive(*, stream_answer: bool, context_view: str) -> None:
    agent = build_agent()
    context = build_context()
    print("agent-core tool agent. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("bye")
            return
        if user_input:
            answer = run_turn(agent, context, user_input, stream=stream_answer)
            if stream_answer and context.metadata.get("last_answer_streamed"):
                print()
            else:
                print(answer)
            print_context(context, context_view)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real OpenAI-compatible tool agent.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat.")
    parser.add_argument("--stream", action="store_true", help="Stream assistant text deltas.")
    parser.add_argument(
        "--context",
        choices=["none", "summary", "messages", "events", "artifacts", "all"],
        default="summary",
        help="Print RunContext details after each turn.",
    )
    return parser.parse_args()


def print_delta(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write(text)
    sys.stdout.flush()


def print_context(context: RunContext, view: str) -> None:
    if view == "none":
        return
    if view == "summary":
        print(
            "[context] "
            f"messages={len(context.messages)} "
            f"events={len(context.events)} "
            f"artifacts={list(context.artifacts)}"
        )
        return

    snapshots = {
        "messages": context.messages,
        "events": [event.to_dict() for event in context.events],
        "artifacts": context.artifacts,
        "all": context.to_dict(),
    }
    print(f"[context:{view}]")
    print(json.dumps(snapshots[view], ensure_ascii=False, indent=2))


def count_model_deltas(context: RunContext) -> int:
    return sum(1 for event in context.events if event.type == "model.delta")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.interactive:
        run_interactive(
            stream_answer=args.stream,
            context_view=args.context,
        )
    else:
        run_demo(
            stream_answer=args.stream,
            context_view=args.context,
        )
