import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel

from config import (
    AVIATIONSTACK_API_KEY,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
)

load_dotenv()


def _python_executable():
    return os.getenv("PYTHON_EXECUTABLE", sys.executable)


aviation_python = os.getenv("AVIATIONSTACK_MCP_PYTHON", _python_executable())
aviation_module = os.getenv("AVIATIONSTACK_MCP_MODULE", "aviationstack_mcp")
weather_script = os.path.join(os.path.dirname(__file__), "custom_meather_mcp_server.py")


client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        },
        "aviationstack": {
            "transport": "stdio",
            "command": aviation_python,
            "args": ["-m", aviation_module, "mcp", "run"],
            "env": {"AVIATIONSTACK_API_KEY": AVIATIONSTACK_API_KEY or ""},
        },
        "weather": {
            "transport": "stdio",
            "command": _python_executable(),
            "args": [weather_script],
            "env": {"OPENWEATHER_API_KEY": OPENWEATHER_API_KEY or ""},
        },
    }
)


class ToolResult(BaseModel):
    """Normalized, typed envelope returned by every MCP tool."""

    status: Literal["ok", "no_data", "error"]
    data: Any = None
    source: str
    timestamp: str
    error: str = ""


_tools_cache = None


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_result(raw: Any, source: str) -> ToolResult:
    if isinstance(raw, ToolResult):
        return raw

    if isinstance(raw, dict) and raw.get("error"):
        return ToolResult(
            status="error",
            data=None,
            source=source,
            timestamp=_timestamp(),
            error=str(raw.get("error")),
        )

    if raw is None or (isinstance(raw, str) and not raw.strip()) or raw == [] or raw == {}:
        return ToolResult(
            status="no_data",
            data=None,
            source=source,
            timestamp=_timestamp(),
            error="Tool returned no data.",
        )

    if isinstance(raw, dict):
        payload = raw.get("data", raw.get("results"))
        if payload == [] or (payload is None and "data" in raw):
            return ToolResult(
                status="no_data",
                data=None,
                source=source,
                timestamp=_timestamp(),
                error="Tool returned no data.",
            )

    return ToolResult(
        status="ok",
        data=raw,
        source=source,
        timestamp=_timestamp(),
    )


async def get_tools():
    """Load MCP tools without letting one broken server cancel the others."""
    global _tools_cache

    if _tools_cache is not None:
        return _tools_cache

    loaded_tools = []
    errors = []

    for server_name in client.connections:
        try:
            loaded_tools.extend(await client.get_tools(server_name=server_name))
        except Exception as exc:
            errors.append(f"{server_name}: {type(exc).__name__}: {exc}")

    if not loaded_tools and errors:
        raise RuntimeError("MCP tool loading failed: " + " | ".join(errors))

    _tools_cache = loaded_tools
    return _tools_cache


async def call_tool(tool_name: str, args: dict | None = None, source: str = "MCP") -> ToolResult:
    try:
        tools = await get_tools()
        tool = next((tool for tool in tools if tool.name == tool_name), None)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        raw = await tool.ainvoke(args or {})
        return _normalize_result(raw, source)
    except Exception as exc:
        return ToolResult(
            status="error",
            data=None,
            source=source,
            timestamp=_timestamp(),
            error=f"{type(exc).__name__}: {exc}",
        )


async def tavily_search(query: str):
    return await call_tool(
        "tavily_search",
        {"query": query},
        source="Tavily Search",
    )


async def list_airports(search: str = "", limit: int = 10):
    return await call_tool(
        "list_airports",
        {"search": search, "limit": limit, "offset": 0},
        source="AviationStack",
    )


async def list_airlines(search: str = "", limit: int = 10):
    return await call_tool(
        "list_airlines",
        {"search": search, "limit": limit, "offset": 0},
        source="AviationStack",
    )


async def current_weather(city: str):
    return await call_tool(
        "get_current_weather",
        {"city": city},
        source="OpenWeather",
    )


async def forecast(city: str):
    return await call_tool(
        "get_forecast",
        {"city": city},
        source="OpenWeather",
    )
