from __future__ import annotations

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


def build_agent() -> Agent:
    model_node = ModelNode(
        messages=build_messages,
        chat_kwargs={"temperature": 0.2, "max_tokens": 220},
    )
    return Agent(Flow(model_node))


def main() -> None:
    configure_stdout()
    context = RunContext()
    result = build_agent().run(
        {
            "system_prompt": SYSTEM_PROMPT,
            "question": "What is the difference between Agent, Flow, and Node?",
        },
        context=context,
        trace=make_trace_options(enabled=True, include=["node", "model"]),
    )

    assistant_message = result.payload["assistant_message"]
    print(assistant_message["content"])
    print(f"path: {' -> '.join(result.path)}")
    print(f"context events: {len(context.events)}")


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
