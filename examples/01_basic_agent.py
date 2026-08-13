from __future__ import annotations

from agent_core import Agent, CallableNode, ExecResult, Flow, PayloadKeys, make_trace_options


def classify(payload: dict) -> ExecResult:
    text = str(payload[PayloadKeys.INPUT]).strip()
    payload["kind"] = "question" if text.endswith("?") else "statement"
    return ExecResult(payload["kind"], payload)


def answer_question(payload: dict) -> dict:
    payload[PayloadKeys.ANSWER] = f"Question received: {payload[PayloadKeys.INPUT]}"
    return payload


def answer_statement(payload: dict) -> dict:
    payload[PayloadKeys.ANSWER] = f"Statement received: {payload[PayloadKeys.INPUT]}"
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
        {PayloadKeys.INPUT: "How does a flow choose the next node?"},
        trace=make_trace_options(enabled=True, include=["node", "flow"]),
    )

    print(result.payload[PayloadKeys.ANSWER])
    print("path:", " -> ".join(result.path))
    print("trace events:", len(result.trace))


if __name__ == "__main__":
    main()
