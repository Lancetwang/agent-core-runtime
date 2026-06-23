from typing import Annotated, Any, Literal

from agent_core import Agent, Tool, build_tool_agent_flow, tool
from _openai_compatible import build_model_from_env, safe_print


SYSTEM_PROMPT = """
You are a small tool-using assistant.

# Tools

## get_weather
Use this when the user asks about weather for Shanghai or Tokyo.

## tell_joke
Use this when the user asks for a joke.

# Response

Answer naturally after tool results are available.
""".strip()


@tool(description="Look up mock weather for Shanghai or Tokyo.")
def get_weather(
    city: Annotated[
        Literal["Shanghai", "Tokyo"],
        "English city name. Normalize user language to one of: Shanghai, Tokyo.",
    ],
) -> dict[str, str]:
    weather = {
        "Shanghai": {"condition": "sunny", "temperature": "24C"},
        "Tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    return {"city": city, **weather[city], "source": "mock"}


@tool(description="Return a short mock joke for the requested topic.")
def tell_joke(
    topic: Annotated[str, "Joke topic, normalized to concise English."] = "weather",
) -> dict[str, str]:
    return {
        "topic": topic,
        "joke": "I asked the cloud for a forecast. It said it was feeling under the weather.",
        "source": "mock",
    }


def build_tools() -> list[Tool]:
    return [get_weather, tell_joke]


def build_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, *payload.get("history", [])]


agent = Agent(
    build_tool_agent_flow(
        model=build_model_from_env(),
        messages=build_messages,
        tools=build_tools(),
        chat_kwargs={"temperature": 0, "tool_choice": "auto"},
    )
)

history: list[dict[str, Any]] = []
for user_input in [
    "What is the weather in Shanghai?",
    "Compare Shanghai and Tokyo weather, then tell one joke.",
]:
    print(f"> {user_input}")
    history.append({"role": "user", "content": user_input})
    result = agent.run({"history": history}, max_steps=12)
    history[:] = result.payload["history"]
    safe_print(result.payload["answer"])

