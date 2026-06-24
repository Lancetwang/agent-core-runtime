from __future__ import annotations

from agent_core import Agent, CallableNode, Flow, make_trace_options


def classify(payload: dict) -> tuple[str, dict]:
    text = str(payload["input"]).strip()
    payload["kind"] = "question" if text.endswith("?") else "statement"
    return payload["kind"], payload


def answer_question(payload: dict) -> dict:
    payload["answer"] = f"Question received: {payload['input']}"
    return payload


def answer_statement(payload: dict) -> dict:
    payload["answer"] = f"Statement received: {payload['input']}"
    return payload


def build_agent() -> Agent:
    router = CallableNode(classify)
    question = CallableNode(answer_question)
    statement = CallableNode(answer_statement)

    router - "question" >> question
    router - "statement" >> statement

    return Agent(Flow(router))


def main() -> None:
    agent = build_agent()
    result = agent.run(
        {"input": "How does a flow choose the next node?"},
        trace=make_trace_options(enabled=True, include=["node", "flow"]),
    )

    print(result.payload["answer"])
    print("path:", " -> ".join(result.path))
    print("trace events:", len(result.trace))


if __name__ == "__main__":
    main()
