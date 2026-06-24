from __future__ import annotations

import sys
from typing import Annotated, Literal

from agent_core import Agent, build_model_from_env, get_current_context, tool


INSTRUCTIONS = """
You are a custom research brief agent.

Use tools when notes are useful. Return a compact answer with:
- recommendation
- evidence used
- next step
""".strip()


@tool(description="Search mock internal notes for a topic.")
def search_notes(
    topic: Annotated[str, "Topic to search in the private note base."],
    depth: Annotated[Literal["quick", "deep"], "Search depth."] = "quick",
) -> dict[str, str]:
    return {
        "topic": topic,
        "depth": depth,
        "result": (
            "Mock note: define a narrow scope, gather two pieces of evidence, "
            "then write a concise recommendation."
        ),
    }


@tool(description="Save a short working note to the agent context.")
def save_working_note(
    title: Annotated[str, "Short note title."],
    content: Annotated[str, "Note content."],
) -> dict[str, str]:
    context = get_current_context()
    if context is not None:
        context.set_artifact(f"note:{title}", content)
    return {"title": title, "content": content, "status": "saved in mock storage"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    agent = Agent(
        model=build_model_from_env(),
        instructions=INSTRUCTIONS,
        tools=[search_notes, save_working_note],
        chat_kwargs={"temperature": 0.2, "max_tokens": 600, "tool_choice": "auto"},
    )
    context = agent.new_context()
    context.metadata["agent_name"] = "brief-writer"

    answer = agent.chat(
        "Create a short plan for evaluating whether a small agent runtime is ready "
        "to be used in an application.",
        context=context,
        max_steps=12,
    )

    print(answer)
    print()
    print(f"agent name: {context.metadata['agent_name']}")
    print(f"messages stored in RunContext: {len(context.messages)}")
    print(f"events stored in RunContext: {len(context.events)}")
    print(f"artifacts: {list(context.artifacts)}")


if __name__ == "__main__":
    main()
