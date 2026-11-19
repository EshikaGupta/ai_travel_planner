import os
import sys
from datetime import datetime, timezone
from typing import Any, Literal
import json
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel
from langsmith import traceable
from config import (
    OPENWEATHER_API_KEY,
)

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

GOOGLE_FLIGHTS_MCP_ROOT = os.path.join(
    PROJECT_ROOT,
    "google-flights-mcp",
)

GOOGLE_FLIGHTS_PYTHON = os.path.join(
    GOOGLE_FLIGHTS_MCP_ROOT,
    ".venv",
    "Scripts",
    "python.exe",
)

GOOGLE_FLIGHTS_SERVER = os.path.join(
    GOOGLE_FLIGHTS_MCP_ROOT,
    "src",
    "mcp_server_google_flights",
    "server.py",
)

HASDATA_API_KEY = os.getenv("HASDATA_API_KEY")

def _get_mcp_error(raw):
    if not isinstance(raw, list):
        return None

    for item in raw:
        if not isinstance(item, dict):
            continue

        text = item.get("text")

        if not isinstance(text, str):
            continue

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict) and payload.get("error"):
            return payload["error"]

    return None

class ToolResult(BaseModel):
    status: Literal["ok", "error", "no_data"]
    data: Any = None
    source: str
    timestamp: str
    error: str | None = None


def _python_executable():
    return os.getenv("PYTHON_EXECUTABLE", sys.executable)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


weather_script = os.path.join(
    os.path.dirname(__file__),
    "custom_meather_mcp_server.py",
)


SERVER_CONFIG = {
    "google_hotels": {
    "transport": "streamable_http",
    "url": "https://mcp.hasdata.com/mcp?apis=google_travel_hotels",
    "headers": {
        "x-api-key": HASDATA_API_KEY,
        },
    },
    "google_flights": {
        "transport": "stdio",
        "command": GOOGLE_FLIGHTS_PYTHON,
        "args": [GOOGLE_FLIGHTS_SERVER],
        "env": {},
    },
    "weather": {
        "transport": "stdio",
        "command": _python_executable(),
        "args": [weather_script],
        "env": {
            "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY or "",
        },
    },
}


def _exception_text(exc: BaseException) -> str:
    """Keep useful details when an MCP library wraps an error in ExceptionGroup."""
    if isinstance(exc, BaseExceptionGroup):
        details = []
        for child in exc.exceptions:
            details.append(_exception_text(child))
        return "; ".join(details) or str(exc)

    return f"{type(exc).__name__}: {exc}"

@traceable(name="MCP tool execution", run_type="tool")
async def _call_server_tool(
    server_name: str,
    tool_name: str,
    args: dict | None = None,
) -> ToolResult:
    source = {
        "google_hotels": "Google Hotels",
        "google_flights": "Google Flights",
        "weather": "OpenWeather",
    }.get(server_name, server_name)

    try:
        server_client = MultiServerMCPClient(
            {server_name: SERVER_CONFIG[server_name]}
        )

        tools = await server_client.get_tools()

        tool = next(
            (item for item in tools if item.name == tool_name),
            None,
        )

        if tool is None:
            raise ValueError(
                f"Tool '{tool_name}' not found on MCP server "
                f"'{server_name}'."
            )

        raw = await tool.ainvoke(args or {})
        mcp_error = _get_mcp_error(raw)

        if mcp_error:
            return ToolResult(
                status="error",
                source=source,
                timestamp=_timestamp(),
                error=mcp_error,
            )
        
        if raw is None or raw == "" or raw == [] or raw == {}:
            return ToolResult(
                status="no_data",
                data=None,
                source=source,
                timestamp=_timestamp(),
            )

        return ToolResult(
            status="ok",
            data=raw,
            source=source,
            timestamp=_timestamp(),
        )

    except Exception as exc:
        return ToolResult(
            status="error",
            source=source,
            timestamp=_timestamp(),
            error=_exception_text(exc),
        )

async def search_google_hotels(
    destination: str,
    check_in: str,
    check_out: str,
    adults: int = 1,
    currency: str = "INR",
    country_code: str | None = None,
):
    args = {
        "q": f"hotels in {destination}",
        "checkInDate": check_in,
        "checkOutDate": check_out,
        "adults": adults,
        "currency": currency,
    }

    if country_code:
        args["gl"] = country_code.lower()

    return await _call_server_tool(
        "google_hotels",
        "hasdata_google_travel_hotels_getGoogleHotels",
        args,
    )

async def search_one_way_flights(
    origin: str,
    destination: str,
    date: str,
    adults: int = 1,
    children: int = 0,
    infants_in_seat: int = 0,
    infants_on_lap: int = 0,
    seat_type: str = "economy",
    return_cheapest_only: bool = False,
    max_results: int = 10,
    compact_mode: bool = True,
) -> ToolResult:
    return await _call_server_tool(
        "google_flights",
        "search_one_way_flights",
        {
            "origin": origin,
            "destination": destination,
            "date": date,
            "adults": adults,
            "children": children,
            "infants_in_seat": infants_in_seat,
            "infants_on_lap": infants_on_lap,
            "seat_type": seat_type,
            "return_cheapest_only": return_cheapest_only,
            "max_results": max_results,
            "compact_mode": compact_mode,
        },
    )


async def search_round_trip_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    adults: int = 1,
    children: int = 0,
    infants_in_seat: int = 0,
    infants_on_lap: int = 0,
    seat_type: str = "economy",
    max_stops: int = 2,
    return_cheapest_only: bool = False,
    max_results: int = 10,
    compact_mode: bool = True,
) -> ToolResult:
    return await _call_server_tool(
        "google_flights",
        "search_round_trip_flights",
        {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "adults": adults,
            "children": children,
            "infants_in_seat": infants_in_seat,
            "infants_on_lap": infants_on_lap,
            "seat_type": seat_type,
            "max_stops": max_stops,
            "return_cheapest_only": return_cheapest_only,
            "max_results": max_results,
            "compact_mode": compact_mode,
        },
    )


async def generate_google_flights_url(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    children: int = 0,
    seat_type: str = "economy",
) -> ToolResult:
    args = {
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "adults": adults,
        "children": children,
        "seat_type": seat_type,
    }

    if return_date:
        args["return_date"] = return_date

    return await _call_server_tool(
        "google_flights",
        "generate_google_flights_url",
        args,
    )


async def current_weather(city: str) -> ToolResult:
    return await _call_server_tool(
        "weather",
        "get_current_weather",
        {"city": city},
    )


async def forecast(city: str) -> ToolResult:
    return await _call_server_tool(
        "weather",
        "get_forecast",
        {"city": city},
    )
