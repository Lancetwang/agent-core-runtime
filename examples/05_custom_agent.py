from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from agent_core import Agent, RunContext, build_tool_agent_flow, get_current_context, tool
from _openai_compatible import build_demo_model, safe_print


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


@dataclass
class CustomResearchAgent:
    """A small application-level wrapper around the reusable runtime.

    This is the intended customization style: define instructions, tools, and a
    flow once, then expose a simple method that your application can call.
    """

    name: str
    instructions: str
    agent: Agent
    context: RunContext

    @classmethod
    def create(cls, *, name: str, instructions: str) -> CustomResearchAgent:
        context = RunContext()
        context.metadata["agent_name"] = name
        context.add_message("system", instructions)

        flow = build_tool_agent_flow(
            model=build_demo_model(),
            tools=[search_notes, save_working_note],
            chat_kwargs={"temperature": 0.2, "max_tokens": 600, "tool_choice": "auto"},
        )
        return cls(name=name, instructions=instructions, agent=Agent(flow), context=context)

    def ask(self, text: str) -> str:
        self.context.add_message("user", text)
        result = self.agent.run({"request": text}, context=self.context, max_steps=12)
        answer = str(result.payload.get("answer", ""))
        self.context.set_artifact("last_answer", answer)
        return answer


def main() -> None:
    agent = CustomResearchAgent.create(
        name="brief-writer",
        instructions=(
            "You are a custom research brief agent. Use tools when notes are useful. "
            "Return a compact answer with a recommendation and next step."
        ),
    )

    answer = agent.ask(
        "Create a short plan for evaluating whether a small agent runtime is ready "
        "to be used in an application."
    )

    safe_print(answer)
    safe_print()
    safe_print(f"agent name: {agent.context.metadata['agent_name']}")
    safe_print(f"messages stored in RunContext: {len(agent.context.messages)}")
    safe_print(f"events stored in RunContext: {len(agent.context.events)}")
    safe_print(f"artifacts: {list(agent.context.artifacts)}")


if __name__ == "__main__":
    main()
