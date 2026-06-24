# Agent Core Runtime

Agent Core Runtime is a small Python runtime for building agents from explicit, composable parts. It includes a built-in OpenAI-compatible chat adapter, so a fresh clone can run real model examples after you fill in a local `.env`.

[中文 README](README.zh-CN.md)

## What It Provides

- `Node`: one unit of work with `exec(payload) -> (action, payload)`.
- `Flow`: routes each action to at most one next node.
- `Agent`: a thin runner around a `Flow`.
- `RunContext`: per-run messages, artifacts, metadata, and UI-friendly runtime events.
- `Tool` and `@tool`: typed Python functions converted into LLM-callable tool schemas.
- `ToolExecutor` and `ToolCallNode`: tool-call parsing, execution, and message appending.
- `ChatModel`, `ModelNode`, and `ToolRouterNode`: provider-neutral model/tool/model loops.
- `OpenAICompatibleChatModel`: built-in adapter for OpenAI-compatible chat completion APIs.

The original payload contract remains the baseline. `RunContext` is an additional runtime layer for richer agent state and event streaming.

## Runtime Execution Logic

```mermaid
flowchart TD
    App["Application code"] --> AgentRun["Agent.run(payload, context?)"]
    AgentRun --> FlowRun["Flow.run(max_steps)"]
    FlowRun --> InitContext["create or reuse RunContext"]
    InitContext --> PickNode["current node"]

    PickNode --> StartEvent["emit node.start"]
    StartEvent --> SyncBefore["context.sync_payload(payload)"]
    SyncBefore --> Exec["node._exec(payload)"]
    Exec --> Result["returns (action, payload)"]
    Result --> SyncAfter["context.sync_payload(payload)"]
    SyncAfter --> EndEvent["emit node.end"]
    EndEvent --> NextNode["successors[action]"]
    NextNode -->|"next node exists"| PickNode
    NextNode -->|"no next node"| FlowEnd["emit flow.end"]
    FlowEnd --> RunResult["FlowRunResult<br/>payload, path, trace, context"]

    subgraph State["Run state"]
        Payload["payload<br/>per-node handoff"]
        Context["RunContext<br/>per-run state"]
        Messages["messages"]
        Metadata["metadata"]
        Artifacts["artifacts"]
        Events["AgentEvent stream"]
    end

    Exec <--> Payload
    Exec -. "get_current_context()" .-> Context
    Context --> Messages
    Context --> Metadata
    Context --> Artifacts
    Context --> Events
```

`payload` is the direct node-to-node handoff. `RunContext` is the durable per-run layer for conversation messages, UI events, metadata, and artifacts.

## Tool Agent And Streaming Loop

```mermaid
flowchart TD
    User["User input"] --> AddUser["context.add_message('user', text)"]
    AddUser --> ModelNode["ModelNode"]
    ModelNode --> BuildRequest["messages from RunContext<br/>tools from @tool schemas"]
    BuildRequest --> Adapter["OpenAICompatibleChatModel"]
    Adapter -->|"non-stream or tool decision"| AssistantMessage["assistant message"]
    Adapter -->|"stream=True"| Delta["model.delta events<br/>optional on_delta callback"]
    Delta --> AssistantMessage
    AssistantMessage --> StoreAssistant["append assistant message<br/>to payload and RunContext"]
    StoreAssistant --> Router["ToolRouterNode"]

    Router -->|"tool_calls exist"| ToolNode["ToolCallNode"]
    ToolNode --> ParseCalls["ToolExecutor.parse_tool_calls"]
    ParseCalls --> ExecuteTools["execute @tool functions"]
    ExecuteTools --> ToolMessages["append tool messages<br/>to payload and RunContext"]
    ToolMessages --> ModelNode

    Router -->|"no tool_calls"| Final["final answer in payload"]
    Final --> Artifact["optional context.set_artifact"]
```

Use `build_tool_agent_flow(...)` when you want this common loop without manually wiring nodes.

## Layout

```text
src/agent_core/
  agent.py              # Thin Agent runner
  core/                 # Node, Flow, RunContext, trace/runtime events
  llm/                  # Built-in OpenAI-compatible ChatModel adapter
  models.py             # Provider-neutral ChatModel protocol
  nodes/                # Reusable agent-loop nodes
  tools/                # Tool decorator, executor, file tools, tool-call node
examples/
  01_basic_agent.py     # Pure Node/Flow action routing
  02_custom_prompt.py   # Real model through ModelNode and RunContext
  03_custom_tool.py     # @tool schema generation and ToolExecutor
  04_tool_agent.py      # Context-first model/tool/model loop
  05_custom_agent.py    # Application-level custom agent wrapper
  _openai_compatible.py # Shared example helper, not a public API
tests/                  # Runtime-only unit tests
```

## Install

```powershell
uv sync
```

Copy the env template:

```powershell
Copy-Item .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`. `DEEPSEEK_API_KEY` is also supported. The defaults target DeepSeek:

```text
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
```

The `.env` file is ignored by Git.

## Progressive Examples

Run them in order:

```powershell
uv run python examples/01_basic_agent.py
uv run python examples/02_custom_prompt.py
uv run python examples/03_custom_tool.py
uv run python examples/04_tool_agent.py --events
uv run python examples/04_tool_agent.py --stream --context messages
uv run python examples/05_custom_agent.py
```

The sequence is intentionally small:

- `01_basic_agent.py`: no LLM, only `CallableNode`, branch actions, `Flow`, and trace.
- `02_custom_prompt.py`: a real model call through `ModelNode`, with messages built from payload and events stored in `RunContext`.
- `03_custom_tool.py`: a Python function becomes a `Tool`, exports OpenAI-compatible schema, and runs through `ToolExecutor`.
- `04_tool_agent.py`: a complete model-tool-model loop where conversation messages, live events, metadata, and artifacts live in `RunContext`; payload only carries per-run input.
- `05_custom_agent.py`: a compact application-level wrapper showing how to create a custom agent from instructions, tools, a flow, and a persistent context.

Useful `04_tool_agent.py` flags:

- `--stream`: stream the final assistant text after tool results are available, while still returning a complete assistant message to the flow.
- `--events`: print live `RunContext` events such as node/model/tool activity.
- `--context summary|messages|events|artifacts|all|none`: inspect the accumulated context after each turn.

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

## Tools

```python
from typing import Annotated, Literal

from agent_core import tool

@tool(description="Look up demo weather for a supported city.")
def get_weather(
    city: Annotated[Literal["Shanghai", "Tokyo"], "English city name."],
) -> dict[str, str]:
    return {"city": city, "condition": "sunny"}
```

The tool schema is derived from the function signature, type annotations, and `Annotated` descriptions.

## Runtime Events

Every flow run returns a context:

```python
result = agent.run({"history": []})
events = [event.to_dict() for event in result.context.events]
messages = result.context.messages
```

For terminal demos, `examples/04_tool_agent.py --context all` prints the full context snapshot, including messages, artifacts, metadata, and runtime events.

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
