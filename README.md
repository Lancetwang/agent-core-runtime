# Agent Core Runtime

Agent Core Runtime is a small Python runtime for building agents from explicit, composable parts. It is intentionally independent of any model provider, application domain, API key, web framework, or knowledge base.

## What It Provides

- `Node`: one unit of work with `exec(payload) -> (action, payload)`.
- `Flow`: routes each action to at most one next node.
- `Agent`: a thin runner around a `Flow`.
- `RunContext`: per-run messages, artifacts, metadata, and UI-friendly runtime events.
- `Tool` and `@tool`: typed Python functions converted into LLM-callable tool schemas.
- `ToolExecutor` and `ToolCallNode`: tool-call parsing, execution, and message appending.
- `ChatModel`, `ModelNode`, and `ToolRouterNode`: provider-neutral model/tool/model loops.

The original payload contract remains the baseline. `RunContext` is an additional runtime layer for richer agent state and event streaming.

## Layout

```text
src/agent_core/
  agent.py              # Thin Agent runner
  core/                 # Node, Flow, RunContext, trace/runtime events
  models.py             # Provider-neutral ChatModel protocol
  nodes/                # Reusable agent-loop nodes
  tools/                # Tool decorator, executor, file tools, tool-call node
examples/
  basic_flow.py         # Minimal action routing
  tool_chatbot.py       # Local fake-model tool loop
tests/                  # Runtime-only unit tests
```

## Install

```powershell
uv sync
```

No model SDK or API key is required by the runtime package.

## Basic Flow

```python
from agent_core import Agent, CallableNode, Flow

def classify(payload: dict) -> tuple[str, dict]:
    return "question" if payload["text"].endswith("?") else "statement", payload

def answer(payload: dict) -> dict:
    payload["answer"] = "received"
    return payload

start = CallableNode(classify)
answer_node = CallableNode(answer)

start - "question" >> answer_node
start - "statement" >> answer_node

result = Agent(Flow(start)).run({"text": "Hello?"})
print(result.payload["answer"])
```

Run the included example:

```powershell
uv run python examples/basic_flow.py
```

## Tools

Use `@tool` to expose typed Python functions as tool schemas:

```python
from typing import Annotated, Literal

from agent_core import tool

@tool(description="Look up demo weather for a supported city.")
def get_weather(
    city: Annotated[Literal["Shanghai", "Tokyo"], "English city name."],
) -> dict[str, str]:
    return {"city": city, "condition": "sunny"}
```

The schema is derived from the function signature, type annotations, and `Annotated` descriptions.

## Standard Tool-Agent Loop

For the common model/tool/model pattern:

```text
ModelNode -> ToolRouterNode
              | tool_call -> ToolCallNode -> ModelNode
              | final     -> flow end
```

Use `build_tool_agent_flow(...)`:

```python
from agent_core import Agent, build_tool_agent_flow

agent = Agent(
    build_tool_agent_flow(
        model=my_chat_model,
        messages=lambda payload: payload["history"],
        tools=[get_weather],
        chat_kwargs={"tool_choice": "auto"},
    )
)
```

`my_chat_model` only needs to implement the `ChatModel` protocol:

```python
def chat_message(self, messages, *, tools=None, tool_choice=None, **kwargs) -> dict:
    ...
```

Run the local fake-model tool-loop example:

```powershell
uv run python examples/tool_chatbot.py --demo --trace
```

## Runtime Events

Every flow run returns a context:

```python
result = agent.run({"history": []})
events = [event.to_dict() for event in result.context.events]
messages = result.context.messages
```

Nodes can also emit events while running:

```python
from agent_core import get_current_context

context = get_current_context()
if context is not None:
    context.emit("custom.event", category="custom", data={"ok": True})
```

## Validation

```powershell
uv run python -m unittest discover -s tests
uv run python -m compileall src tests examples
```

## 中文说明

Agent Core Runtime 是一个轻量级 Python agent runtime。这个分支只提供通用 agent 元件，不包含任何具体应用、模型供应商、API key、Web UI 或知识库。

核心设计是：

- `Node` 负责一个处理步骤，输入 `payload`，输出 `(action, payload)`。
- `Flow` 根据 `action` 选择下一个节点。
- `RunContext` 保存一次运行中的消息、产物、元数据和 runtime events。
- `Tool` / `@tool` 把有类型标注的 Python 函数转换为可供模型调用的工具。
- `ModelNode` / `ToolRouterNode` / `ToolCallNode` 提供通用的工具调用 agent 回路。

`payload` 机制仍然保留，`RunContext` 只是额外的运行上下文层。

