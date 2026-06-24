from __future__ import annotations

import sys
from typing import Annotated

from agent_core import Agent, tool

INSTRUCTIONS = """
你是一个Agent，但是你每句话后面都会带上曼波。必要时会使用工具回答用户的问题。
## 可用工具
get_weather: 获取某地的天气信息
""".strip()


@tool(description="获取某地的天气信息")
def get_weather(location: Annotated[str, "要查询天气的城市名称"]) -> str:
    return f"{location}的天气是晴朗的, 24°C。"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    agent = Agent(
        instructions=INSTRUCTIONS,
        tools=[get_weather],
        chat_kwargs={"temperature": 0.2, "max_tokens": 600, "tool_choice": "auto"},
    )
    context = agent.new_context()
    context.metadata["agent_name"] = "brief-writer"

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
        )
        print(answer)


if __name__ == "__main__":
    main()
