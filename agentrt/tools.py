"""Tools are plain Python functions. Their signatures are the schema."""
from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Literal, Union, get_args, get_origin


class ToolError(Exception):
    """Raise inside a tool to report a failure the model can read and react to."""


class ValidationError(Exception):
    """The model's arguments do not match the tool's signature."""


_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _schema_for(annotation: Any) -> dict:
    origin = get_origin(annotation)
    if origin is Literal:
        return {"enum": list(get_args(annotation))}
    if origin in (Union, types.UnionType):
        options = [a for a in get_args(annotation) if a is not type(None)]
        return _schema_for(options[0]) if len(options) == 1 else {"anyOf": [_schema_for(o) for o in options]}
    if origin in (list, typing.List):
        (item,) = get_args(annotation) or (Any,)
        return {"type": "array", "items": _schema_for(item)} if item is not Any else {"type": "array"}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    return {}


def _check(value: Any, schema: dict, path: str) -> None:
    if "enum" in schema:
        if value not in schema["enum"]:
            raise ValidationError(f"{path}: must be one of {schema['enum']}, got {value!r}")
        return
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                return _check(value, option, path)
            except ValidationError:
                continue
        raise ValidationError(f"{path}: {value!r} matches none of the allowed types")
    kind = schema.get("type")
    ok = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        None: lambda v: True,
    }[kind](value)
    if not ok:
        raise ValidationError(f"{path}: expected {kind}, got {type(value).__name__}")
    if kind == "array" and "items" in schema:
        for i, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{i}]")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict           # JSON Schema for the arguments object
    side_effects: bool         # True if the tool changes something outside the conversation
    timeout_s: float

    def validate(self, arguments: Any) -> dict:
        """Check arguments against the schema. Unknown or missing keys are errors, never silently fixed."""
        if not isinstance(arguments, dict):
            raise ValidationError("arguments must be a JSON object")
        props, required = self.parameters["properties"], self.parameters["required"]
        unknown = sorted(set(arguments) - set(props))
        if unknown:
            raise ValidationError(f"unknown argument(s): {unknown}. Allowed: {sorted(props)}")
        missing = [r for r in required if r not in arguments]
        if missing:
            raise ValidationError(f"missing required argument(s): {missing}")
        for key, value in arguments.items():
            _check(value, props[key], key)
        return arguments

    def spec(self) -> dict:
        """The shape model APIs expect for a tool definition."""
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, fn: Callable | None = None, *, side_effects: bool = False, timeout_s: float = 10.0, name: str | None = None):
        """Register a function. Its type hints become the schema; its docstring the description."""

        def register(f: Callable) -> Callable:
            signature = inspect.signature(f)
            hints = typing.get_type_hints(f)
            properties, required = {}, []
            for pname, param in signature.parameters.items():
                if pname not in hints:
                    raise TypeError(f"{f.__name__}: parameter {pname!r} needs a type hint")
                properties[pname] = _schema_for(hints[pname])
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
            doc = inspect.getdoc(f)
            if not doc:
                raise TypeError(f"{f.__name__}: tools need a docstring; it is what the model reads")
            tool_name = name or f.__name__
            if tool_name in self._tools:
                raise ValueError(f"tool {tool_name!r} is already registered")
            self._tools[tool_name] = Tool(tool_name, doc, f, {"type": "object", "properties": properties,
                                                              "required": required, "additionalProperties": False},
                                          side_effects, timeout_s)
            return f

        return register(fn) if fn is not None else register

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict]:
        return [self._tools[n].spec() for n in self.names()]
