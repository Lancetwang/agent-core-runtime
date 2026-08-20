# Changelog

All notable changes to `friday-agent-core` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-08-20

### Added

- Native async execution across the node runtime: override `Node.aexec`, run
  any graph with `Flow.arun`, and use `Agent.arun` / `Agent.achat`. Existing
  synchronous nodes run in worker threads, while `ModelNode`, `ToolCallNode`,
  `ToolExecutor`, and the default `LLM` use native async paths when available.
- `AsyncChatModel` and `LLM.achat_message` with `AsyncOpenAI` streaming,
  reasoning deltas, tool calls, and usage aggregation.
- `ToolLoopGuardNode`, a regular routable node with `continue`, `warn`, and
  `halt` actions. The built-in Agent loop uses it to warn on repeated
  no-progress tool calls, then disables tools for a final text-only answer.
- `report_tool_progress(value)` lets sync or async tools publish live-only
  `tool.progress` events without adding runtime-only parameters to their JSON
  schema.
- Every `AgentEvent` has a monotonic per-context `seq`; `RunUsage` also tracks
  cached prompt tokens across common OpenAI and Anthropic usage shapes.
- Cooperative cancellation: synchronous entry points accept a
  `threading.Event`, while async entry points also accept `asyncio.Event` and
  propagate task cancellation. Events are checked between steps and tool
  calls. A cancelled run emits `flow.cancel` and raises the new
  `FlowCancelled` (a `FlowError` subclass, never retried by `Node`). Nested
  runs inherit the enclosing run's cancel event.
- `python -m agent_core chat` starts an interactive chat REPL built from
  `Agent` + `LLM` (`--instructions`, `--max-steps`, `--no-stream`).
- CI now lints with ruff, type-checks the public source with Pyright, enforces
  an 85% coverage floor, and runs the matrix on Windows as well as Ubuntu.

### Changed

- `CallableNode` routes only explicit `ExecResult(action, payload)` values by
  default. Plain `(str, value)` tuples are now unambiguously business payloads;
  `route_plain_tuples=True` temporarily restores the deprecated 0.1 behavior.
- `Agent.run` / `Agent.chat` now default their `max_steps` to the
  constructor budget instead of an independent hard-coded 100.
- Nested flows share one step counter: every inner node visit is debited from
  the enclosing run budget while each flow keeps its own local cap.
- The conversation history now has one canonical store: during a flow run,
  assistant and tool messages are written to the active context scope only.
  `state["history"]` no longer accumulates during runs; it remains an import
  seed for custom flows. Custom message builders receive the canonical scope
  under their configured history key, so tool-loop history is not lost.
  Reasoning content is kept only for tool-call turns.
- An `Agent` adopts unscoped ambient messages into its scope at run start,
  so a conversation seeded through unscoped `context.add_message` calls is
  visible to the model while other agents' scoped messages stay isolated.
- `model.delta` / `model.reasoning.delta` are live-only (`notify`, no longer
  retained). All `ToolCallNode` instances now retain metadata-only tool events
  and send full payloads through observe-only events, avoiding another copy
  in `tool.call` / `tool.result` records. Full retained payloads require the
  explicit `retain_event_payloads=True` compatibility option. `message.add`
  events likewise retain field names rather than message content. Trace
  collection captures live events regardless of retention.
- Default trace categories dropped the never-emitted `llm` and `plan`
  entries and are now `{"flow", "node", "tool", "model"}`.

### Fixed

- OpenAI-compatible streams without tool-call `index` fields no longer merge
  several parallel calls into one corrupted call. Missing and duplicate call
  IDs are normalized in both the model adapter and tool executor so assistant
  messages and results remain pairable.
- Tool schema generation now unwraps `Annotated` recursively inside
  containers, unions, and TypedDict fields, so nested item descriptions no
  longer raise `ToolDefinitionError`.
- Flows wired in the examples/04 style (system and user messages added to a
  plain `RunContext` without a scope) no longer send an empty message list to
  the model on the first request.
- `Agent.chat` now reports a missing `answer` key in custom flow output instead
  of silently returning an empty string.

## [0.1.10] - 2026-08-08

### Fixed

- Let a terminal node finish successfully on the final allowed `max_steps`
  slot and emit `node.error` / `flow.error` events on failures.
- Import initial payload history into an empty context scope so tool loops keep
  their original system and user messages.
- Avoid mutating caller-owned history lists in model and tool nodes.
- Treat serial tools as exclusive barriers between parallel batches.
- Delay construction of the default provider until the first model call.

## [0.1.9] - 2026-08-08

### Added

- `get_current_tool_call()` exposes the active invocation to tools that need
  to correlate transient progress with a UI entry.

## [0.1.8] - 2026-08-07

### Added

- `Tool.parallel` flag (also settable through the `@tool(parallel=True)`
  decorator) marks read-only tools that are safe to run concurrently.
- `ToolExecutor.execute_all` now executes parallel-capable calls in a bounded
  thread pool (`max_workers`, default 4) while serial tools keep running one
  at a time in declaration order. Results are always returned in the original
  call order, so hosts can append them to the conversation without re-sorting.
- `ToolCallNode` announces every `tool.call` before executing the batch and
  emits `tool.result` events in declaration order afterwards.
- `ToolResult.elapsed_ms` and the `tool.result` event data now carry the
  wall-clock execution time of each call, measured inside the executor —
  accurate per call even when the batch runs concurrently.

## [0.1.7] - 2026-08-07

### Fixed

- Harden tool schema generation, duplicate-name validation, malformed call
  parsing, unknown-tool errors, and unserializable tool-result handling.

## [0.1.6] - 2026-07-31

### Added

- Emit provider reasoning as a separate `model.reasoning.delta` stream so
  harnesses can render collapsible thinking without mixing it into answers.

## [0.1.5] - 2026-07-31

### Changed

- Retain `reasoning_content` in `RunContext` only for assistant tool-call
  messages that need it on the next request, avoiding needless context growth
  after final answers.

## [0.1.4] - 2026-07-31

### Fixed

- Preserve provider `reasoning_content` in streamed and non-streamed assistant
  messages so thinking models can continue tool-call conversations correctly.

## [0.1.3] - 2026-07-29

### Added

- OpenAI-compatible multimodal user-message content support.

## [0.1.2] - 2026-07-29

### Added

- `python -m agent_core --version` for installed-version inspection.
- `python -m agent_core check` for a credential-free local runtime smoke test.

## [0.1.1] - 2026-07-26

### Added

- `agent_core.__version__`, resolved from the installed distribution.
- `ChatModel` and `Message` are now exported from the top-level `agent_core` package.
- PEP 561 `py.typed` marker: consumers get full type checking for the public API.
- Docstrings across the public API (`Node`, `Flow`, `Agent`, `ModelNode`,
  `ToolRouterNode`, `ToolCallNode`, `Tool`, `tool`, `ToolExecutor`,
  `RunContext` methods, `TraceOptions`).
- CI workflow running tests on every push and pull request.

### Changed

- Actionable error messages throughout: unknown tools list the available tool
  names, tool definition errors name the offending function and parameter,
  `max_steps` overrun suggests the likely causes, and tool execution failures
  include the exception type.
- `Flow.run` on a flow without a start node now raises `FlowError` instead of
  silently returning an empty result.
- Wiring the same action of a node to a second successor now raises
  `ValueError` instead of silently replacing the first edge.
- `Node >> other` validates that `other` is a `Node`.

## [0.1.0] - 2026-07-26

### Added

- First PyPI release as `friday-agent-core` (import name `agent_core`).
- Runtime primitives: `Node`, `Flow`, `Agent` (an agent is a node),
  `RunContext` with scoped messages, events, artifacts, metadata, and exact
  usage accounting.
- Built-in model/tool chat loop: `ModelNode`, `ToolRouterNode`,
  `ToolCallNode`, `ToolExecutor`, and the `@tool` schema decorator.
- OpenAI-compatible `LLM` adapter with streaming and usage capture.
- Trace options for filtering, forwarding, and printing run events.
