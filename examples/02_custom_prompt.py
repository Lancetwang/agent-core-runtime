from typing import Any

from agent_core import Agent, build_tool_agent_flow
from _openai_compatible import build_model_from_env, safe_print


SYSTEM_PROMPT = """
You are a practical materials-science writing assistant.
Answer in three compact bullet points.
Avoid hype.
""".strip()


def build_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": payload["question"]},
    ]


agent = Agent(
    build_tool_agent_flow(
        model=build_model_from_env(),
        messages=build_messages,
        tools=[],
        chat_kwargs={"temperature": 0},
    )
)

result = agent.run({"question": "How should I think about catalyst stability?"})
safe_print(result.payload["answer"])

