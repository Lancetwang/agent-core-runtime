from __future__ import annotations

import json
from typing import Annotated, Literal

from agent_core import ToolCall, ToolExecutor, tool


@tool(description="Look up mock weather for a supported city.")
def get_weather(
    city: Annotated[Literal["Shanghai", "Tokyo"], "City to query."],
) -> dict[str, str]:
    data = {
        "Shanghai": {"condition": "sunny", "temperature": "24C"},
        "Tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    return {"city": city, **data[city], "source": "mock"}


@tool(description="Return a short mock joke for a topic.")
def tell_joke(
    topic: Annotated[str, "Joke topic."] = "weather",
) -> dict[str, str]:
    return {
        "topic": topic,
        "joke": "I asked the forecast for a punchline; it said the timing was cloudy.",
        "source": "mock",
    }


def main() -> None:
    print("OpenAI-compatible schema:")
    print(json.dumps(get_weather.to_llm_format(), indent=2))

    executor = ToolExecutor([get_weather, tell_joke])
    result = executor.execute(
        ToolCall(
            id="call_demo_weather",
            name="get_weather",
            arguments={"city": "Shanghai"},
        )
    )

    print("\nTool execution result:")
    print(result.content)


if __name__ == "__main__":
    main()
