from typing import Annotated, Literal

from agent_core import tool


@tool(description="Look up mock weather for a supported city.")
def get_weather(
    city: Annotated[
        Literal["Shanghai", "Tokyo"],
        "English city name.",
    ],
) -> dict[str, str]:
    weather = {
        "Shanghai": {"condition": "sunny", "temperature": "24C"},
        "Tokyo": {"condition": "rainy", "temperature": "18C"},
    }
    return {"city": city, **weather[city], "source": "mock"}


print(get_weather.to_llm_format())
print(get_weather(city="Shanghai"))

