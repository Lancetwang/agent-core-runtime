from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints


class ToolDefinitionError(ValueError):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def to_llm_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


def tool(description: str, *, name: str | None = None) -> Callable[[Callable[..., Any]], Tool]:
    def decorator(fn: Callable[..., Any]) -> Tool:
        signature = inspect.signature(fn)
        type_hints = get_type_hints(fn, include_extras=True)
        if "return" not in type_hints:
            raise ToolDefinitionError(f"tool '{fn.__name__}' must have a return type annotation.")

        properties: dict[str, Any] = {}
        required: list[str] = []
        for param_name, parameter in signature.parameters.items():
            if parameter.kind not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                raise ToolDefinitionError(
                    f"tool parameter '{param_name}' must be positional-or-keyword or keyword-only."
                )
            annotation = type_hints.get(param_name)
            if annotation is None:
                raise ToolDefinitionError(
                    f"tool parameter '{param_name}' must have a type annotation."
                )

            schema, param_description = _annotation_to_schema(annotation)
            if param_description:
                schema["description"] = param_description
            if parameter.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                schema["default"] = parameter.default
            properties[param_name] = schema

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            parameters["required"] = required

        return Tool(
            name=name or fn.__name__,
            description=description,
            parameters=parameters,
            fn=fn,
        )

    return decorator


def _annotation_to_schema(annotation: Any) -> tuple[dict[str, Any], str | None]:
    description = None
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        description = next((item for item in args[1:] if isinstance(item, str)), None)

    return _type_to_schema(annotation), description


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = list(args)
        value_type = type(values[0]) if values else str
        schema = _type_to_schema(value_type)
        schema["enum"] = values
        return schema

    if origin in {list, tuple}:
        item_type = args[0] if args else Any
        return {"type": "array", "items": _type_to_schema(item_type)}

    if origin is dict:
        return {"type": "object"}

    if origin in {Union, UnionType}:
        schemas = [_type_to_schema(arg) for arg in args if arg is not type(None)]
        if len(schemas) == 1:
            return schemas[0]
        return {"anyOf": schemas}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is dict:
        return {"type": "object"}
    if annotation is list:
        return {"type": "array"}

    raise ToolDefinitionError(f"unsupported tool annotation: {annotation!r}.")
