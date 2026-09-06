"""Registry for discovering and invoking Agent-callable tools."""

from __future__ import annotations

from copy import deepcopy
import logging
import time
from typing import Callable

from .rag_tool import search_knowledge_base


ToolHandler = Callable[..., dict]
RAG_TOOL_NAME = "search_knowledge_base"
LOGGER = logging.getLogger(__name__)


class ToolRegistryError(RuntimeError):
    """Base error for Tool Registry operations."""


class ToolAlreadyRegisteredError(ToolRegistryError):
    """Raised when a Tool name is registered more than once."""


class ToolNotFoundError(ToolRegistryError):
    """Raised when a requested Tool is not registered."""


class ToolValidationError(ToolRegistryError):
    """Raised when Tool name or input arguments are invalid."""


class ToolExecutionError(ToolRegistryError):
    """Raised when a registered Tool handler fails during execution."""


class ToolRegistry:
    """Register, describe, and execute tools through one interface."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, object]] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self._call_counts: dict[str, int] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, object],
        output_schema: dict[str, object],
        handler: ToolHandler,
    ) -> None:
        """Register one Tool and its public input/output contract."""
        if not isinstance(name, str) or not name.strip():
            raise ToolValidationError("Tool name 必须是非空字符串")
        normalized_name = name.strip()
        if normalized_name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"Tool 已注册: {normalized_name!r}"
            )
        if not isinstance(description, str) or not description.strip():
            raise ToolValidationError("Tool description 必须是非空字符串")
        if not isinstance(input_schema, dict):
            raise ToolValidationError("Tool input_schema 必须是 dict")
        if not isinstance(output_schema, dict):
            raise ToolValidationError("Tool output_schema 必须是 dict")
        if not callable(handler):
            raise ToolValidationError("Tool handler 必须可调用")

        self._tools[normalized_name] = {
            "name": normalized_name,
            "description": description.strip(),
            "input": deepcopy(input_schema),
            "output": deepcopy(output_schema),
        }
        self._handlers[normalized_name] = handler

    def get(self, name: str) -> dict[str, object]:
        """Return a copy of a registered Tool's public description."""
        self._validate_name(name)
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"未注册的 Tool: {name!r}") from exc
        return deepcopy(tool)

    def list_tools(self) -> list[dict[str, object]]:
        """Return public descriptions for all registered tools."""
        return [deepcopy(tool) for tool in self._tools.values()]

    def execute(self, name: str, **tool_input: object) -> dict:
        """Find and execute a Tool using its registered handler."""
        started_at = time.perf_counter()
        display_name = name if isinstance(name, str) and name else repr(name)
        call_count = self._call_counts.get(name, 0) if isinstance(name, str) else 0
        try:
            self._validate_name(name)
            try:
                handler = self._handlers[name]
            except KeyError as exc:
                raise ToolNotFoundError(f"未注册的 Tool: {name!r}") from exc
            call_count += 1
            self._call_counts[name] = call_count
            self._validate_input(name, tool_input)
            result = handler(**tool_input)
            if not isinstance(result, dict):
                raise ToolExecutionError(f"Tool {name!r} 返回值必须是 dict")
        except ToolRegistryError as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            LOGGER.error(
                "[ToolRegistry] tool=%s success=False duration_ms=%.2f call_count=%d error=%s",
                display_name,
                duration_ms,
                call_count,
                exc,
            )
            raise
        except Exception as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000
            LOGGER.error(
                "[ToolRegistry] tool=%s success=False duration_ms=%.2f call_count=%d error=%s",
                display_name,
                duration_ms,
                call_count,
                exc,
            )
            raise ToolExecutionError(f"Tool {name!r} 执行失败: {exc}") from exc

        duration_ms = (time.perf_counter() - started_at) * 1000
        LOGGER.info(
            "[ToolRegistry] tool=%s success=True duration_ms=%.2f call_count=%d",
            display_name,
            duration_ms,
            call_count,
        )
        return result

    def get_call_count(self, name: str) -> int:
        """Return the number of executions for one registered Tool."""
        self._validate_name(name)
        if name not in self._tools:
            raise ToolNotFoundError(f"未注册的 Tool: {name!r}")
        return self._call_counts.get(name, 0)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ToolValidationError("Tool name 必须是非空字符串")

    @staticmethod
    def _validate_input(name: str, tool_input: dict[str, object]) -> None:
        if name != RAG_TOOL_NAME:
            return
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolValidationError("query 必须是非空字符串")
        if "top_k" in tool_input:
            top_k = tool_input["top_k"]
            if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
                raise ToolValidationError("top_k 必须是大于 0 的整数")
        if "filters" in tool_input:
            filters = tool_input["filters"]
            if filters is not None and not isinstance(filters, dict):
                raise ToolValidationError("filters 必须是 dict 或 None")


DEFAULT_TOOL_REGISTRY = ToolRegistry()
DEFAULT_TOOL_REGISTRY.register(
    name=RAG_TOOL_NAME,
    description="检索当前项目知识库，并返回带来源和距离的结构化 Evidence。",
    input_schema={
        "query": "str (required)",
        "top_k": "int (optional, default=4)",
        "filters": "dict | None (optional)",
    },
    output_schema={
        "query": "str",
        "results": [
            {
                "content": "str",
                "source": "str | None",
                "page": "int | str | None",
                "distance": "float",
            }
        ],
    },
    handler=search_knowledge_base,
)


__all__ = [
    "DEFAULT_TOOL_REGISTRY",
    "RAG_TOOL_NAME",
    "ToolAlreadyRegisteredError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolRegistry",
    "ToolRegistryError",
    "ToolValidationError",
]
