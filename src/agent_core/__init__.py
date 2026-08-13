"""Reusable agent runtime primitives.

Distributed on PyPI as ``friday-agent-core``; the import name is ``agent_core``.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from agent_core.agent import Agent

try:
    __version__ = _version("friday-agent-core")
except PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0.0.0"
from agent_core.core import (
    Action,
    AgentEvent,
    CallableNode,
    ExecResult,
    Flow,
    FlowCancelled,
    FlowError,
    FlowRunResult,
    Node,
    PayloadKeys,
    RunContext,
    RunUsage,
    TraceEvent,
    TraceOptions,
    format_trace_event,
    get_current_context,
    make_trace_options,
    reset_current_context,
    set_current_context,
)
from agent_core.llm import (
    LLM,
    ChatModel,
    Message,
    ModelNode,
    ToolRouterNode,
)
from agent_core.tools import (
    Tool,
    ToolCall,
    ToolCallNode,
    ToolDefinitionError,
    ToolExecutor,
    ToolResult,
    get_current_tool_call,
    tool,
)

__all__ = [
    "Action",
    "Agent",
    "AgentEvent",
    "CallableNode",
    "ChatModel",
    "ExecResult",
    "Flow",
    "FlowCancelled",
    "FlowError",
    "FlowRunResult",
    "LLM",
    "Message",
    "ModelNode",
    "Node",
    "PayloadKeys",
    "RunContext",
    "RunUsage",
    "Tool",
    "ToolCall",
    "ToolCallNode",
    "ToolDefinitionError",
    "ToolExecutor",
    "ToolRouterNode",
    "ToolResult",
    "TraceEvent",
    "TraceOptions",
    "__version__",
    "format_trace_event",
    "get_current_tool_call",
    "get_current_context",
    "make_trace_options",
    "reset_current_context",
    "set_current_context",
    "tool",
]
