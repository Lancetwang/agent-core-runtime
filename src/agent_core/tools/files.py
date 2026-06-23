from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_core.tools.base import Tool


def build_file_tools(root: str | Path) -> list[Tool]:
    return [
        build_read_file_tool(root),
        build_write_file_tool(root),
        build_edit_file_tool(root),
    ]


def build_read_file_tool(root: str | Path) -> Tool:
    workspace = _workspace(root)

    def read_file(path: str, start_line: int = 1, max_chars: int = 4000) -> dict[str, Any]:
        resolved = _resolve_in_workspace(workspace, path)
        text = resolved.read_text(encoding="utf-8")
        lines = text.splitlines()
        start = max(1, start_line)
        excerpt_lines: list[str] = []
        char_count = 0
        for line_number, line in enumerate(lines[start - 1 :], start=start):
            rendered = f"{line_number}: {line}"
            if excerpt_lines and char_count + len(rendered) + 1 > max_chars:
                break
            excerpt_lines.append(rendered)
            char_count += len(rendered) + 1
            if char_count >= max_chars:
                break
        return {
            "path": str(resolved),
            "start_line": start,
            "line_count": len(lines),
            "content": "\n".join(excerpt_lines),
        }

    return Tool(
        name="read_file",
        description="Read a UTF-8 text file from the report workspace with line numbers.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path inside the report workspace."},
                "start_line": {
                    "type": "integer",
                    "description": "1-based line number to start reading from.",
                    "default": 1,
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of characters to return.",
                    "default": 4000,
                },
            },
            "required": ["path"],
        },
        fn=read_file,
    )


def build_write_file_tool(root: str | Path) -> Tool:
    workspace = _workspace(root)

    def write_file(path: str, content: str) -> dict[str, Any]:
        resolved = _resolve_in_workspace(workspace, path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return {
            "path": str(resolved),
            "chars": len(content),
            "line_count": len(content.splitlines()),
        }

    return Tool(
        name="write_file",
        description="Create or overwrite a UTF-8 text file in the report workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path inside the report workspace."},
                "content": {"type": "string", "description": "Full file content to write."},
            },
            "required": ["path", "content"],
        },
        fn=write_file,
    )


def build_edit_file_tool(root: str | Path) -> Tool:
    workspace = _workspace(root)

    def edit_file(path: str, start_line: int, end_line: int, replacement: str) -> dict[str, Any]:
        resolved = _resolve_in_workspace(workspace, path)
        lines = resolved.read_text(encoding="utf-8").splitlines()
        start = max(1, start_line)
        end = max(start, end_line)
        replacement_lines = replacement.splitlines()
        new_lines = [
            *lines[: start - 1],
            *replacement_lines,
            *lines[end:],
        ]
        content = "\n".join(new_lines)
        if resolved.read_text(encoding="utf-8").endswith("\n"):
            content += "\n"
        resolved.write_text(content, encoding="utf-8")
        return {
            "path": str(resolved),
            "start_line": start,
            "end_line": end,
            "replacement_line_count": len(replacement_lines),
            "line_count": len(new_lines),
        }

    return Tool(
        name="edit_file",
        description="Replace an inclusive line range in a UTF-8 text file in the report workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path inside the report workspace."},
                "start_line": {"type": "integer", "description": "1-based first line to replace."},
                "end_line": {"type": "integer", "description": "1-based last line to replace."},
                "replacement": {"type": "string", "description": "Replacement text for the line range."},
            },
            "required": ["path", "start_line", "end_line", "replacement"],
        },
        fn=edit_file,
    )


def _workspace(root: str | Path) -> Path:
    path = Path(root).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_in_workspace(root: Path, path: str) -> Path:
    candidate = (root / path).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"Path escapes report workspace: {path}")
    return candidate
