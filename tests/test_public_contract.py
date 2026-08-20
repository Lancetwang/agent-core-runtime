"""Contract tests for the published package surface and its error behavior."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from typing import Annotated

import agent_core
from agent_core import (
    CallableNode,
    ExecResult,
    Flow,
    FlowError,
    Node,
    ToolCall,
    ToolDefinitionError,
    ToolExecutor,
    tool,
)


class PackageSurfaceTests(unittest.TestCase):
    def test_version_is_exposed(self) -> None:
        self.assertIsInstance(agent_core.__version__, str)
        self.assertTrue(agent_core.__version__)

    def test_chat_model_protocol_is_public(self) -> None:
        self.assertIn("ChatModel", agent_core.__all__)
        self.assertIn("Message", agent_core.__all__)

    def test_package_ships_the_py_typed_marker(self) -> None:
        package_dir = Path(agent_core.__file__).parent
        self.assertTrue((package_dir / "py.typed").exists())

    def test_module_reports_version_and_passes_local_check(self) -> None:
        version = subprocess.run(
            [sys.executable, "-m", "agent_core", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        check = subprocess.run(
            [sys.executable, "-m", "agent_core", "check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn(agent_core.__version__, version.stdout)
        self.assertIn("OK (import, node, flow, context)", check.stdout)

    def test_module_exposes_the_chat_subcommand(self) -> None:
        help_run = subprocess.run(
            [sys.executable, "-m", "agent_core", "chat", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("interactive chat REPL", help_run.stdout)

    def test_module_without_a_command_prints_help(self) -> None:
        run = subprocess.run(
            [sys.executable, "-m", "agent_core"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("check", run.stdout)
        self.assertIn("chat", run.stdout)


class WiringValidationTests(unittest.TestCase):
    def test_flow_without_start_node_raises(self) -> None:
        with self.assertRaisesRegex(FlowError, "no start node"):
            Flow().run({})

    def test_duplicate_action_edge_raises(self) -> None:
        router = CallableNode(lambda payload: ExecResult("go", payload))
        first = CallableNode(lambda payload: payload)
        second = CallableNode(lambda payload: payload)
        router - "go" >> first
        with self.assertRaisesRegex(ValueError, "already routes action 'go'"):
            router - "go" >> second

    def test_rewiring_the_same_successor_is_idempotent(self) -> None:
        router = CallableNode(lambda payload: ExecResult("go", payload))
        target = CallableNode(lambda payload: payload)
        router - "go" >> target
        router - "go" >> target
        self.assertIs(router.successors["go"], target)

    def test_rshift_requires_a_node(self) -> None:
        node = CallableNode(lambda payload: payload)
        with self.assertRaisesRegex(TypeError, "must be a Node"):
            node - "go" >> "not a node"  # type: ignore[operator]

    def test_unimplemented_exec_names_the_subclass(self) -> None:
        class Incomplete(Node):
            pass

        with self.assertRaisesRegex(NotImplementedError, "Incomplete"):
            Incomplete().exec({})

    def test_add_edge_wires_programmatically(self) -> None:
        router = CallableNode(lambda payload: ExecResult("go", payload))
        target = CallableNode(lambda payload: {**payload, "hit": True})

        returned = router.add_edge("go", target)
        result = Flow(router).run({})

        self.assertIs(returned, target)
        self.assertEqual(result.path, ["CallableNode", "CallableNode"])
        self.assertTrue(result.payload["hit"])

    def test_add_edge_rejects_non_nodes(self) -> None:
        router = CallableNode(lambda payload: payload)

        with self.assertRaisesRegex(TypeError, "must be a Node"):
            router.add_edge("go", "not a node")  # type: ignore[arg-type]

    def test_add_edge_normalizes_empty_action(self) -> None:
        router = CallableNode(lambda payload: ExecResult("", payload))
        target = CallableNode(lambda payload: {**payload, "hit": True})

        router.add_edge("", target)
        result = Flow(router).run({})

        self.assertNotIn("", router.successors)
        self.assertIs(router.successors["default"], target)
        self.assertTrue(result.payload["hit"])

    def test_action_selection_does_not_mutate_the_node(self) -> None:
        node = CallableNode(lambda payload: payload)

        node - "go"

        self.assertEqual(node.successors, {})
        self.assertFalse(hasattr(node, "_action"))


class ToolErrorMessageTests(unittest.TestCase):
    def test_unknown_tool_error_lists_available_tools(self) -> None:
        @tool(description="Say hello.")
        def hello(name: Annotated[str, "Name."]) -> str:
            return f"hi {name}"

        executor = ToolExecutor([hello])
        result = executor.execute(ToolCall(id="1", name="missing", arguments={}))

        self.assertTrue(result.is_error)
        self.assertIn("not found", result.content)
        self.assertIn("hello", result.content)

    def test_tool_failure_reports_exception_type(self) -> None:
        @tool(description="Always fails.")
        def broken() -> str:
            raise ValueError("bad input")

        executor = ToolExecutor([broken])
        result = executor.execute(ToolCall(id="1", name="broken", arguments={}))

        self.assertTrue(result.is_error)
        self.assertIn("ValueError", result.content)
        self.assertIn("bad input", result.content)

    def test_tool_definition_error_names_the_function(self) -> None:
        with self.assertRaisesRegex(ToolDefinitionError, "unannotated"):

            @tool(description="Missing parameter annotation.")
            def unannotated(value) -> str:  # type: ignore[no-untyped-def]
                return str(value)


if __name__ == "__main__":
    unittest.main()
