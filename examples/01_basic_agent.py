from typing import Any

from agent_core import Agent, build_tool_agent_flow
from _openai_compatible import build_model_from_env, safe_print


def build_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": payload["input"]},
    ]


agent = Agent(
    build_tool_agent_flow(
        model=build_model_from_env(),
        messages=build_messages,
        tools=[],
        chat_kwargs={"temperature": 0},
    )
)

result = agent.run({"input": "Say hello in one short sentence."})
safe_print(result.payload["answer"])

