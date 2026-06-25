# Agent Core Runtime

[English README](README.md)

Agent Core Runtime 是一个轻量级 Python agent runtime。它只保留几个明确的元件：`Node`、`Flow`、`RunContext`、`Tool` 和 `Agent`。

## 这个项目提供什么

它的目标是足够小、足够直接，也方便你替换其中任何一层：

- `Node`：一个工作单元。
- `Flow`：根据 action 名称把节点连接起来。
- `Agent`：本身也是 `Node`，既可以单独运行，也可以嵌进更大的 flow。
- `RunContext`：保存一次运行中的 messages、events、metadata 和 artifacts。
- `payload`：在节点之间传递明确的业务数据，并作为 flow 的结果返回。
- `@tool`：把带类型标注的 Python 函数转换成 OpenAI-compatible tool schema。
- `LLM`：默认 OpenAI-compatible 模型，会自己读取 `.env`。

你可以一行声明一个普通工具 agent；如果需要特殊循环，也可以自己连接节点。

## 运行结构

```mermaid
flowchart TD
    App["应用层"] --> Agent["Agent"]
    Agent -->|"直接聊天"| BuiltIn["内置 model/tool loop"]
    Agent -->|"自定义"| Flow["Flow"]
    Agent -. "Agent 也是 Node" .-> OuterFlow["另一个 Flow"]

    Flow --> Data["payload"]
    Data --> Node["Node"]
    Node -->|"action + payload"| Next["下一个 Node"]
    Next --> Node

    Flow -. "runtime context" .-> Context["RunContext"]
    Context --> Messages["messages"]
    Context --> Events["events"]
    Context --> Artifacts["artifacts"]
    Context --> Metadata["metadata"]

    BuiltIn --> ModelNode["ModelNode"]
    ModelNode --> ChatModel["ChatModel"]
    ChatModel --> OpenAI["OpenAI-compatible API"]
    ModelNode --> Router["ToolRouterNode"]
    Router -->|"tool_calls"| ToolNode["ToolCallNode"]
    ToolNode --> Tools["@tool 函数"]
    ToolNode --> ModelNode
    Router -->|"没有 tool_calls"| Answer["answer"]
```

## 项目结构

```text
src/agent_core/
  agent.py              # Agent：直接聊天入口，也是可嵌套的 Node
  core/                 # Node, Flow, RunContext, trace events
  llm/                  # LLM, ChatModel 协议, ModelNode, router
  tools/                # @tool, ToolExecutor, ToolCallNode, file tools
examples/
  01_basic_agent.py     # 只使用 Node 和 Flow
  02_custom_prompt.py   # 真实模型调用和自定义 prompt
  03_custom_tool.py     # 工具 schema 和执行
  04_tool_agent.py      # 手动连接 model-tool-model loop
  05_custom_agent.py    # 直接 Agent(instructions, tools)
tests/
```

## 安装

```powershell
uv sync
Copy-Item .env.example .env
```

在 `.env` 中填写：

```text
LLM_API_KEY=...
```

默认配置面向 DeepSeek：

```text
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

`.env` 已被 Git 忽略。

## 快速声明 Agent

```python
from typing import Annotated

from agent_core import Agent, tool

@tool(description="Search private notes.")
def search_notes(topic: Annotated[str, "Topic to search."]) -> dict[str, str]:
    return {"topic": topic, "result": "mock note"}

agent = Agent(
    instructions="You are a concise research assistant.",
    tools=[search_notes],
    stream=True,
    chat_kwargs={"tool_choice": "auto"},
)

context = agent.new_context()
answer = agent.chat("Draft a short evaluation plan.", context=context)
print(answer)
```

## 自定义 Flow

当你不想使用普通聊天循环时，可以直接连接节点：

```python
from agent_core import Agent, CallableNode, Flow

def classify(payload: dict) -> tuple[str, dict]:
    return "question" if payload["text"].endswith("?") else "statement", payload

def answer(payload: dict) -> dict:
    payload["answer"] = "received"
    return payload

router = CallableNode(classify)
answer_node = CallableNode(answer)

router - "question" >> answer_node
router - "statement" >> answer_node

result = Agent(Flow(router)).run({"text": "Hello?"})
print(result.payload["answer"])
```

因为 `Agent` 本身也是 `Node`，所以 agent 可以继续组合成更大的 agent：

```python
researcher = Agent(model=model, instructions="Research.", tools=[search_notes])
writer = Agent(model=model, instructions="Write the final response.")

researcher - "final" >> writer
team = Agent(Flow(researcher))
```

当 `Agent` 作为节点使用时，它默认会把内部 flow 的最终 action 暴露给外层 flow。只有当你希望它固定返回某个外部 action 时，才需要传 `action="some_action"`。

## 示例

按顺序运行：

```powershell
uv run python examples/01_basic_agent.py
uv run python examples/02_custom_prompt.py
uv run python examples/03_custom_tool.py
uv run python examples/04_tool_agent.py --context messages
uv run python examples/05_custom_agent.py
```

LLM 相关示例默认流式输出。使用 `--no-stream` 可以切换为完整回复结束后再打印。

`04_tool_agent.py` 还支持：

- `--interactive`：启动交互式循环。
- `--context summary|messages|events|artifacts|all|none`：查看运行上下文。

你自己的 agent 可以用 `Agent(..., stream=False)` 默认关闭流式，也可以用 `agent.chat(..., stream=False)` 单次覆盖。

## Runtime Events

每次运行都会返回 `RunContext`：

```python
result = agent.run({"text": "hello"})
messages = result.context.messages
events = [event.to_dict() for event in result.context.events]
```

节点内部也可以写入当前 context：

```python
from agent_core import get_current_context

context = get_current_context()
if context:
    context.set_artifact("note", "saved")
```

保持一个清晰边界：业务状态放在 `payload`，运行/会话数据放在 `RunContext`。例如 router decision、plan、artifact path 这类内容应该从 `result.payload` 获取；流式 delta、messages、UI events、artifact metadata 则放在 `result.context`。完整报告、PDF、长日志这类大产物应该放在文件、数据库或对象存储里，payload/context 只保留路径、ID、摘要或元数据。

在多 agent flow 里，`RunContext` 会共享 events、artifacts 和 metadata，但每个 `Agent` 都有自己隔离的 message scope 作为 LLM 输入。这样前端/日志仍然能看到统一运行过程，但一个 agent 的 prompt/history 不会泄露到另一个 agent 的模型调用里。

## 验证

```powershell
uv run python -m unittest discover -s tests
uv run python -m compileall src tests examples
```
