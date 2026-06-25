from __future__ import annotations

import argparse
import sys

from agent_core import Agent, Flow, ModelNode, RunContext, make_trace_options

SYSTEM_PROMPT = (
    "You are a concise runtime assistant. "
    "Explain agent-core concepts in plain English and keep the answer short."
)


def build_messages(payload: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": payload["system_prompt"]},
        {"role": "user", "content": payload["question"]},
    ]


def print_delta(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one model node with a custom prompt.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    stream = not args.no_stream
    context = RunContext()
    model_node = ModelNode(
        messages=build_messages,
        chat_kwargs={
            "temperature": 0.2,
            "max_tokens": 220,
            "stream": stream,
            "on_delta": print_delta if stream else None,
        },
    )
    result = Agent(Flow(model_node)).run(
        {
            "system_prompt": SYSTEM_PROMPT,
            "question": "What is the difference between Agent, Flow, and Node?",
        },
        context=context,
        trace=make_trace_options(enabled=True, include=["node", "model"]),
    )

    if stream:
        print()
    else:
        print(result.payload["assistant_message"]["content"])
    print(f"path: {' -> '.join(result.path)}")
    print(f"context events: {len(context.events)}")


if __name__ == "__main__":
    main()
