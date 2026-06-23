from agent_core import Agent, CallableNode, Flow


def classify(payload: dict) -> tuple[str, dict]:
    text = payload["input"]
    route = "question" if text.endswith("?") else "statement"
    payload["kind"] = route
    return route, payload


def answer_question(payload: dict) -> dict:
    payload["reply"] = f"Question received: {payload['input']}"
    return payload


def answer_statement(payload: dict) -> dict:
    payload["reply"] = f"Statement received: {payload['input']}"
    return payload


classify_node = CallableNode(classify)
question_node = CallableNode(answer_question)
statement_node = CallableNode(answer_statement)

classify_node - "question" >> question_node
classify_node - "statement" >> statement_node

result = Agent(Flow(classify_node)).run({"input": "Hello?"})
print(result.payload["reply"])
print(result.path)
