import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Any, Literal
from langsmith import traceable
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from mcp_client import (
    current_weather,
    forecast,
    search_one_way_flights,
    search_round_trip_flights,
    generate_google_flights_url,
    tavily_search,
)
from state import TravelState
from airport_resolver import resolve_airports
from config import get_llm, get_active_provider

def _format_weather_response(current_data, forecast_data):

    current_text = _tool_result_text(current_data)
    forecast_text = _tool_result_text(forecast_data)

    try:
        current = json.loads(current_text)
        forecast = json.loads(forecast_text)
    except (json.JSONDecodeError, TypeError):
        return f"""
### Current Weather
{current_text}

### Forecast
{forecast_text}
"""

    lines = [
        f"## Weather in {current.get('city', 'Unknown')}",
        "",
        "### Current Conditions",
        f"- Temperature: {current.get('temperature_c')}°C",
        f"- Feels like: {current.get('feels_like_c')}°C",
        f"- Condition: {current.get('condition')}",
        f"- Humidity: {current.get('humidity')}%",
        f"- Wind speed: {current.get('wind_speed')} m/s",
        "",
        "### Forecast",
    ]

    for item in forecast.get("forecast", []):
        lines.append(
            f"- {item['datetime']}: "
            f"{item['temperature']}°C, {item['weather']}"
        )

    return "\n".join(lines)

def _get_llm():
    return get_llm(get_active_provider())

# Structured outputs are used only where the graph needs machine-readable data.
# Normal specialist/final responses remain plain text.
class GuardrailOutput(BaseModel):
    allowed: bool = Field(description="Whether this is a travel-related request")
    reason: str = Field(default="", description="Short reason when the request is rejected")


class TripConstraints(BaseModel):
    destination: str = ""
    origin: str = ""
    duration: str = ""
    budget: str = ""
    departure_date: str = ""
    return_date: str = ""
    travel_style: str = ""
    special_preferences: list[str] = Field(default_factory=list)


class SupervisorOutput(BaseModel):
    request_type: Literal[
        "flight_only",
        "hotel_only",
        "weather_only",
        "budget_only",
        "itinerary_planning",
        "multi_service",
    ] = "itinerary_planning"
    selected_agents: list[str] = Field(default_factory=list)
    trip_constraints: TripConstraints = Field(default_factory=TripConstraints)
    reasoning: str = ""

def _get_weather_city(state: TravelState) -> str:
    constraints = state.get("trip_constraints") or {}

    city = (constraints.get("destination") or "").strip()

    if city:
        return city

    query = state.get("user_query", "")

    match = re.search(
        r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z\s-]*?)(?:[?.!,]|$)",
        query,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return ""

def _run_async(coro):
    """Run an async MCP call from a synchronous LangGraph node."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import threading
    result = []
    error = []

    def runner():
        try:
            result.append(asyncio.run(coro))
        except Exception as exc:
            error.append(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0] if result else None


def _safe_async(coro, default=""):
    try:
        return _run_async(coro)
    except Exception as exc:
        print(f"\n========== MCP ERROR ==========")
        print(type(exc).__name__)
        print(str(exc))
        print("===============================\n")
        return default

# ============================================================
# LLM HELPERS
# ============================================================

def _tool_result_text(result: Any) -> str:
    """Extract displayable text from a structured MCP ToolResult."""
    if result is None:
        return ""

    data = getattr(result, "data", result)

    # MCP content blocks, e.g. [{"type": "text", "text": "..."}]
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(_content_to_text(item))
        return "\n\n".join(part for part in parts if part).strip()

    if isinstance(data, dict):
        if isinstance(data.get("text"), str):
            return data["text"].strip()
        return json.dumps(data, ensure_ascii=False, indent=2)

    return _content_to_text(data).strip()


def _format_tavily_result(result: Any) -> str:
    """Turn Tavily's MCP response into readable Markdown."""
    text = _tool_result_text(result)
    if not text:
        return "Hotel search data is currently unavailable."

    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text

    if not isinstance(payload, dict):
        return text

    sections = []

    answer = payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        sections.append(answer.strip())

    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        url = str(item.get("url", "")).strip()

        if not title and not content:
            continue

        if title and url:
            sections.append(f"### [{title}]({url})")
        elif title:
            sections.append(f"### {title}")

        if content:
            sections.append(content)

    return "\n\n".join(sections).strip() or "Hotel search data is currently unavailable."


def _content_to_text(content: Any) -> str:
    """Normalize LangChain content from Gemini, Groq, etc. to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return _content_to_text(content["content"])
        return ""
    if isinstance(content, (list, tuple)):
        return "\n".join(
            text for text in (_content_to_text(item) for item in content) if text
        )
    return str(content)


def _llm_text(system: str, prompt: str) -> str:
    """Invoke any configured chat model and always return plain text."""
    try:
        response = _get_llm().invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ]
        )
        text = _content_to_text(getattr(response, "content", response)).strip()
        if not text:
            raise ValueError("LLM returned an empty response.")
        return text
    except Exception as exc:
        raise RuntimeError(f"LLM invocation failed: {exc}") from exc


async def _llm_text_async(system: str, prompt: str) -> str:
    """Async version used by specialist agents that run in parallel."""
    try:
        response = await _get_llm().ainvoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ]
        )
        text = _content_to_text(getattr(response, "content", response)).strip()
        if not text:
            raise ValueError("LLM returned an empty response.")
        return text
    except Exception as exc:
        raise RuntimeError(f"LLM invocation failed: {exc}") from exc

@traceable(name="Structured LLM invocation", run_type="chain")
def _structured_invoke(model_class, system: str, prompt: str):
    """Use provider-native structured output, with a safe text fallback."""
    model = _get_llm()

    try:
        structured_model = model.with_structured_output(
            model_class,
            include_raw=True,
        )
        result = structured_model.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ]
        )

        parsed = result.get("parsed") if isinstance(result, dict) else None
        parsing_error = result.get("parsing_error") if isinstance(result, dict) else None

        if parsed is not None and not parsing_error:
            return parsed

        print("Structured output parsing failed; falling back to text JSON parsing.")
        if parsing_error:
            print(type(parsing_error).__name__, str(parsing_error))

        raw = result.get("raw") if isinstance(result, dict) else result
        raw_text = _content_to_text(getattr(raw, "content", raw)).strip()

        if raw_text:
            return _json_fallback(raw_text, model_class)

    except Exception as exc:
        print(
            f"Structured output unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    # Explicit JSON fallback
    fallback_prompt = f"""
    {system}

    {prompt}

    IMPORTANT OUTPUT REQUIREMENTS:
    - Return ONLY one valid JSON object.
    - Do not use Markdown.
    - Do not use bold formatting.
    - Do not add explanations before or after the JSON.
    - Use double quotes around all JSON keys and string values.
    - Use null for missing values.
    """

    raw_text = _llm_text(
        "You are a strict JSON generator. Return valid JSON only.",
        fallback_prompt,
    )

    return _json_fallback(raw_text, model_class)


def _json_fallback(text: str, model_class):
    """Last-resort parser for providers/models without structured-output support."""
    text = _content_to_text(text).strip()
    if not text:
        raise ValueError("LLM returned an empty response.")

    cleaned = (
        text.replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    # First attempt: direct JSON parsing
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        data = None

        # Second attempt: extract a JSON object from surrounding text
        for index, character in enumerate(cleaned):
            if character != "{":
                continue

            try:
                data, _ = decoder.raw_decode(cleaned[index:])
                break
            except json.JSONDecodeError:
                continue

        if data is None:
            raise ValueError(
                f"Could not parse JSON from LLM response:\n{text}"
            )

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from the LLM.")

    return model_class.model_validate(data)


def _fallback_supervisor(query: str) -> dict:
    """Deterministic fallback that preserves the user's primary intent."""
    q = query.lower()

    flight = any(word in q for word in (
        "flight", "flights", "fly", "flying", "airfare",
        "airline", "airport", "nonstop", "non-stop", "direct flight",
    ))
    hotel = any(word in q for word in (
        "hotel", "hotels", "stay", "accommodation", "resort",
        "where to stay",
    ))
    weather = any(word in q for word in (
        "weather", "forecast", "temperature", "climate", "rain",
        "snow", "packing",
    ))
    itinerary = any(word in q for word in (
        "itinerary", "trip plan", "travel plan", "plan my trip",
        "plan a trip", "day-by-day", "days in", "things to do",
    ))

    # Explicit single-service requests always stay single-service.
    # A price/budget constraint does NOT activate budget_agent for a
    # flight/hotel/weather request.
    if flight and not itinerary:
        selected = ["flight_agent"]
        request_type = "flight_only"
    elif hotel and not itinerary:
        selected = ["hotel_agent"]
        request_type = "hotel_only"
    elif weather and not itinerary:
        selected = ["weather_agent"]
        request_type = "weather_only"
    elif itinerary:
        selected = ["itinerary_agent"]
        if flight:
            selected.insert(0, "flight_agent")
        if hotel:
            selected.insert(-1, "hotel_agent")
        if weather:
            selected.insert(-1, "weather_agent")
        if any(word in q for word in (
            "budget", "under ", "cost", "price", "affordable",
            "lakh", "rupee", "inr", "usd", "$",
        )):
            selected.insert(-1, "budget_agent")
        request_type = "itinerary_planning"
    else:
        selected = ["itinerary_agent"]
        request_type = "itinerary_planning"

    destination = ""
    origin = ""
    if " from " in q and " to " in q:
        origin = query.lower().split(" from ", 1)[1].split(" to ", 1)[0].strip()
        destination = query.lower().split(" to ", 1)[1].split(" under ", 1)[0].split(" in ", 1)[0].strip()
    else:
        for marker in (" to ", " in ", " for "):
            if marker in q:
                destination = query.lower().split(marker, 1)[1].split(" under ", 1)[0].split(" for ", 1)[0].strip()
                break

    return {
        "request_type": request_type,
        "selected_agents": list(dict.fromkeys(selected)),
        "trip_constraints": {
            "destination": destination,
            "origin": origin,
            "duration": "",
            "budget": "",
            "departure_date": "",
            "return_date": "",
            "travel_style": "",
            "special_preferences": [],
        },
        "reasoning": "Used deterministic intent routing because structured supervisor output was unavailable.",
    }


# ============================================================
# SUPERVISOR AGENT
# ============================================================
@traceable(name="Supervisor Agent", run_type="chain")
def supervisor_agent(state: TravelState):
    query = state["user_query"]

    # --------------------------------------------------------
    # INPUT GUARDRAIL
    # --------------------------------------------------------

    guardrail_prompt = f"""
Determine whether the following request is a valid travel planning request.

Return ONLY a JSON object.
Do not use Markdown.
Do not add explanations before or after the JSON.

Required format:

{{
    "allowed": true,
    "reason": ""
}}

Rules:
- allowed = true if the user is asking for travel planning, flights,
  hotels, destinations, itineraries, weather for a trip, travel budget,
  transportation, activities, or related travel information.
- allowed = false if the request is clearly unrelated to travel.

User request:
{query}
"""

    guardrail_result = _structured_invoke(
        GuardrailOutput,
        "You are an input validation guardrail. Classify the request as travel-related or not.",
        guardrail_prompt,
    )

    print("\n========== GUARDRAIL STRUCTURED RESPONSE ==========")
    print(guardrail_result.model_dump_json(indent=2))
    print("===================================================\n")

    # --------------------------------------------------------
    # GUARDRAIL DECISION
    # --------------------------------------------------------

    if not guardrail_result.allowed:
        reason = guardrail_result.reason or "Request rejected by input guardrail."

        return {
            "selected_agents": [],
            "trip_constraints": {},
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(
                    content=f"Guardrail blocked request: {reason}"
                )
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    # --------------------------------------------------------
    # SUPERVISOR LOGIC
    # --------------------------------------------------------

    prompt = f"""
You are the routing supervisor for a multi-agent travel application.
Your ONLY job is to identify the user's PRIMARY REQUEST TYPE and select the minimum agents required to answer that request.

AVAILABLE AGENTS

flight_agent
- Only for flight-related requests: flights, routes, airfare, airlines, airports,
  departure/arrival, direct/nonstop flights, flight dates, or flight price limits.

hotel_agent
- Only for hotel/accommodation requests: hotels, stays, resorts, neighborhoods,
  accommodation, or where to stay.

weather_agent
- Only for weather-related requests: current weather, forecast, temperature,
  climate, rain, snow, or packing/weather conditions.

budget_agent
- For analyzing the overall budget/cost of a trip.
- DO NOT select budget_agent merely because another request contains a price,
  fare limit, currency, or words such as "under INR 50,000".
- Example: "flights from Delhi to Dubai under INR 50,000" = flight_agent ONLY.

itinerary_agent
- ONLY when the user asks to plan a trip, create an itinerary, create a day-by-day
  plan, plan activities, or asks for a complete travel plan.
- DO NOT select itinerary_agent for a standalone flight, hotel, weather, or budget query.

ROUTING RULES

1. Identify what the user wants as the final answer, not every constraint mentioned.
2. A constraint does NOT become a separate task.
   - "flights under INR 50,000" -> flight_agent only
   - "hotel under INR 10,000/night" -> hotel_agent only
   - "weather in Dubai next week" -> weather_agent only
3. For an explicit single-service request, select exactly ONE agent.
4. Only use multiple agents when the user explicitly asks for multiple services or a
   complete trip plan.
5. itinerary_agent is required only for actual trip planning/itinerary requests.
6. If the user asks to "plan a trip" and mentions flights, hotels, weather, or budget,
   select the relevant specialist agents plus itinerary_agent.
7. Do not infer additional services just because they might be useful.

REQUEST TYPE
Choose exactly one:
- flight_only
- hotel_only
- weather_only
- budget_only
- itinerary_planning
- multi_service

Examples
- "What's the weather in Delhi?" -> weather_only -> [weather_agent]
- "Flights from Delhi to Dubai in October under INR 50,000" -> flight_only -> [flight_agent]
- "Hotels in Dubai under INR 10,000" -> hotel_only -> [hotel_agent]
- "How much would a Dubai trip cost?" -> budget_only -> [budget_agent]
- "Plan a 5-day Dubai trip" -> itinerary_planning -> [itinerary_agent]
- "Plan a 5-day Dubai trip including flights and hotels under INR 1 lakh" -> itinerary_planning -> [flight_agent, hotel_agent, budget_agent, itinerary_agent]
- "Give me Dubai weather and hotels" -> multi_service -> [weather_agent, hotel_agent]

Return the structured fields exactly as requested.

IMPORTANT DATE EXTRACTION RULES:
- Extract trip duration whenever the user gives one, such as "7-day trip",
  "7 days", "5 nights", etc.
- If the user gives a departure date and a duration but no explicit return
  date, DO NOT ask the user for a return date.
- Preserve the duration in trip_constraints.duration.
- The flight agent will derive the return date from departure_date + duration.
- Example: "7-day trip from Delhi on 10 October" means departure_date =
  2026-10-10 and duration = "7 days". The flight agent should derive the
  return date as 2026-10-17.
- If an explicit return date is provided, use that date instead.

User request:
{query}
"""

    parsed = _structured_invoke(
        SupervisorOutput,
        "You route work to specialist travel agents. Choose only useful agents and extract constraints.",
        prompt,
    )

    print("\n========== SUPERVISOR STRUCTURED RESPONSE ==========")
    print(parsed.model_dump_json(indent=2))
    print("=====================================================\n")

    # --------------------------------------------------------
    # Defensive validation
    # --------------------------------------------------------

    allowed_agents = {
        "flight_agent",
        "hotel_agent",
        "weather_agent",
        "budget_agent",
        "itinerary_agent",
    }

    selected = [
        agent
        for agent in parsed.selected_agents
        if agent in allowed_agents
    ]

    # Enforce the supervisor's primary-intent contract in code.
    # This prevents a model from activating budget/itinerary merely because
    # those words appear as constraints in a single-service request.
    exclusive_routes = {
        "flight_only": ["flight_agent"],
        "hotel_only": ["hotel_agent"],
        "weather_only": ["weather_agent"],
        "budget_only": ["budget_agent"],
    }

    if parsed.request_type in exclusive_routes:
        selected = exclusive_routes[parsed.request_type]

    elif parsed.request_type == "itinerary_planning":
        # Itinerary planning must actually include itinerary_agent.
        selected = [agent for agent in selected if agent != "itinerary_agent"]
        selected.append("itinerary_agent")

    # Never let an empty/invalid structured response silently run every agent.
    if not selected:
        fallback = _fallback_supervisor(query)
        selected = fallback["selected_agents"]
        trip_constraints = fallback["trip_constraints"]
        reasoning = fallback["reasoning"]
    else:
        trip_constraints = parsed.trip_constraints.model_dump()
        reasoning = parsed.reasoning

    return {
        "selected_agents": selected,
        "trip_constraints": trip_constraints,
        "supervisor_reasoning": reasoning,
        "messages": [
            AIMessage(
                content="Supervisor created the agent plan."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# FLIGHT AGENT
# ============================================================
@traceable(name="Flight Agent", run_type="chain")
async def flight_agent(state: TravelState):
    query = state["user_query"]
    constraints = state["trip_constraints"]

    origin_input = constraints.get("origin", "").strip()
    destination_input = constraints.get("destination", "").strip()
    departure_date = constraints.get("departure_date", "").strip()
    return_date = constraints.get("return_date", "").strip()

    print("\n========== FLIGHT AGENT INPUT ==========")
    print("Query:", query)
    print("Origin:", origin_input)
    print("Destination:", destination_input)
    print("Departure date:", departure_date)
    print("Return date:", return_date)
    print("========================================\n")

    # ------------------------------------------------------------
    # Required information
    # ------------------------------------------------------------

    if not origin_input or not destination_input:
        result = (
            "I need both the departure location and destination "
            "to search for flights."
        )

        return {
            "flight_results": result,
            "messages": [
                AIMessage(content="Flight search requires origin and destination.")
            ],
            "llm_calls": state.get("llm_calls", 0),
        }

    if not departure_date:
        result = (
            "I need a specific departure date to search live flight options. "
            "Please provide the departure date."
        )

        return {
            "flight_results": result,
            "messages": [
                AIMessage(content="Flight search requires a departure date.")
            ],
            "llm_calls": state.get("llm_calls", 0),
        }

    # ------------------------------------------------------------
    # Derive return date from trip duration when the user did not
    # explicitly provide one.
    #
    # Example:
    #   7-day trip starting 2026-10-10
    #   -> return date 2026-10-17
    #
    # A "7-day trip" is treated as 7 elapsed days from departure.
    # An explicit return_date always takes precedence.
    # ------------------------------------------------------------

    if not return_date:
        duration_text = str(
            constraints.get("duration", "")
        ).strip()

        duration_match = re.search(
            r"(\d+)\s*(?:-\s*)?(?:day|days|night|nights)\b",
            duration_text.lower(),
        )

        if duration_match:
            trip_duration = int(duration_match.group(1))

            try:
                departure_dt = datetime.strptime(
                    departure_date,
                    "%Y-%m-%d",
                )

                return_dt = departure_dt + timedelta(days=trip_duration)
                return_date = return_dt.strftime("%Y-%m-%d")

                print(
                    "Derived return date from duration:",
                    return_date,
                )

            except ValueError:
                print(
                    "Could not derive return date. Invalid departure date:",
                    departure_date,
                )

    # If the supervisor failed to extract duration, try the original
    # user query as a deterministic backup.
    if not return_date:
        duration_match = re.search(
            r"(\d+)\s*(?:-\s*)?(?:day|days|night|nights)\b",
            query.lower(),
        )

        if duration_match:
            trip_duration = int(duration_match.group(1))

            try:
                departure_dt = datetime.strptime(
                    departure_date,
                    "%Y-%m-%d",
                )

                return_dt = departure_dt + timedelta(days=trip_duration)
                return_date = return_dt.strftime("%Y-%m-%d")

                print(
                    "Derived return date from user query:",
                    return_date,
                )

            except ValueError:
                pass

    # Only ask if neither an explicit return date nor a usable duration
    # exists. A duration-based trip should NEVER trigger this question.
    if not return_date:
        result = (
            "I have the departure date, but I need either a return date "
            "or a trip duration to search round-trip flights."
        )

        return {
            "flight_results": result,
            "messages": [
                AIMessage(content="Return date could not be determined.")
            ],
            "llm_calls": state.get("llm_calls", 0),
        }

    # Keep the derived date in shared state so hotel, budget and itinerary
    # agents all see the same trip dates.
    constraints["return_date"] = return_date

    # ------------------------------------------------------------
    # Resolve city / country / airport -> real IATA codes
    # ------------------------------------------------------------

    try:
        origin_airports = resolve_airports(
            origin_input,
            max_results=2,
        )

        destination_airports = resolve_airports(
            destination_input,
            max_results=5,
        )

    except Exception as exc:
        print(
            f"Airport resolution failed: "
            f"{type(exc).__name__}: {exc}"
        )

        result = (
            "I could not resolve the departure or destination "
            "to a valid airport. Please provide a city or airport name."
        )

        return {
            "flight_results": result,
            "messages": [
                AIMessage(content="Airport resolution failed.")
            ],
            "llm_calls": state.get("llm_calls", 0),
        }

    if not origin_airports:
        result = (
            f"I couldn't find a suitable airport for '{origin_input}'. "
            "Please provide a city, airport name, or IATA code."
        )

        return {
            "flight_results": result,
            "messages": [
                AIMessage(content="Origin airport could not be resolved.")
            ],
            "llm_calls": state.get("llm_calls", 0),
        }

    if not destination_airports:
        result = (
            f"I couldn't find a suitable airport for '{destination_input}'. "
            "Please provide a destination city, airport name, or IATA code."
        )

        return {
            "flight_results": result,
            "messages": [
                AIMessage(content="Destination airport could not be resolved.")
            ],
            "llm_calls": state.get("llm_calls", 0),
        }

    print("\n========== AIRPORT RESOLUTION ==========")
    print(
        "Origin:",
        [(a.iata, a.name) for a in origin_airports],
    )
    print(
        "Destination:",
        [(a.iata, a.name) for a in destination_airports],
    )
    print("========================================\n")

    # ------------------------------------------------------------
    # Helper: determine whether the MCP response actually contains
    # usable flight data. The MCP wrapper can return status="ok"
    # even when the underlying server returns a JSON error object.
    # ------------------------------------------------------------

    def _usable_flight_result(raw_result: Any) -> tuple[bool, str]:
        text = _tool_result_text(raw_result).strip()

        if not text:
            return False, ""

        if getattr(raw_result, "status", None) == "error":
            return False, ""

        try:
            payload = json.loads(text)

            if isinstance(payload, dict):
                if payload.get("error"):
                    return False, ""

                # Some successful MCP responses contain a list/count/data
                # structure. An explicit error is the important failure case.
                if payload.get("status") == "error":
                    return False, ""

        except (TypeError, json.JSONDecodeError):
            # Non-JSON text can still be a valid successful result.
            pass

        return True, text

    # ------------------------------------------------------------
    # 1. PRIMARY SEARCH: ROUND TRIP ONLY
    #
    # Search round-trip combinations first. We do NOT expose one-way
    # results if a valid round-trip result is available.
    # ------------------------------------------------------------

    round_trip_results = []

    max_combinations = 8
    combinations = [
        (origin_airport, destination_airport)
        for origin_airport in origin_airports
        for destination_airport in destination_airports
    ][:max_combinations]

    for origin_airport, destination_airport in combinations:
        origin_code = origin_airport.iata
        destination_code = destination_airport.iata

        try:
            print(
                f"\nSearching ROUND TRIP: "
                f"{origin_code} -> {destination_code}"
            )

            raw_result = await search_round_trip_flights(
                origin=origin_code,
                destination=destination_code,
                departure_date=departure_date,
                return_date=return_date,
                adults=1,
                seat_type="economy",
                max_stops=2,
                return_cheapest_only=False,
                max_results=10,
                compact_mode=True,
            )

            usable, result_text = _usable_flight_result(raw_result)

            if usable:
                round_trip_results.append(
                    f"### Round trip: {origin_code} -> {destination_code}\n"
                    f"Origin airport: {origin_airport.name}\n"
                    f"Destination airport: {destination_airport.name}\n"
                    f"Departure: {departure_date}\n"
                    f"Return: {return_date}\n"
                    f"{result_text}"
                )

                print(
                    f"Round-trip data available for "
                    f"{origin_code} -> {destination_code}"
                )
            else:
                print(
                    f"No usable round-trip data for "
                    f"{origin_code} -> {destination_code}"
                )

        except Exception as exc:
            print(
                f"Round-trip search failed for "
                f"{origin_code} -> {destination_code}: "
                f"{type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------
    # 2. FALLBACK: TWO ONE-WAY SEARCHES
    #
    # Only run this section if NO usable round-trip result was found.
    #
    # Outbound:
    #   origin -> destination on departure date
    #
    # Return:
    #   destination -> origin on return date
    #
    # We keep the two directions separate so the LLM cannot mistake
    # two unrelated one-way prices for a round-trip fare.
    # ------------------------------------------------------------

    fallback_one_way_results = []

    if not round_trip_results:
        print(
            "\n========== ROUND TRIP UNAVAILABLE =========="
        )
        print(
            "No usable round-trip results. "
            "Falling back to outbound + return one-way searches."
        )
        print("=============================================\n")

        # Keep the fallback bounded. We use the same airport combinations
        # as the round-trip attempt, but each successful combination requires
        # two one-way searches.
        for origin_airport, destination_airport in combinations:
            origin_code = origin_airport.iata
            destination_code = destination_airport.iata

            outbound_text = ""
            return_text = ""

            # -----------------------------
            # Outbound
            # -----------------------------
            try:
                print(
                    f"\nSearching ONE WAY OUTBOUND: "
                    f"{origin_code} -> {destination_code} "
                    f"on {departure_date}"
                )

                raw_outbound = await search_one_way_flights(
                    origin=origin_code,
                    destination=destination_code,
                    date=departure_date,
                    adults=1,
                    seat_type="economy",
                    return_cheapest_only=False,
                    max_results=10,
                    compact_mode=True,
                )

                outbound_usable, outbound_text = _usable_flight_result(
                    raw_outbound
                )

                if not outbound_usable:
                    outbound_text = ""

            except Exception as exc:
                print(
                    f"Outbound one-way search failed for "
                    f"{origin_code} -> {destination_code}: "
                    f"{type(exc).__name__}: {exc}"
                )

            # -----------------------------
            # Return
            # -----------------------------
            try:
                print(
                    f"\nSearching ONE WAY RETURN: "
                    f"{destination_code} -> {origin_code} "
                    f"on {return_date}"
                )

                raw_return = await search_one_way_flights(
                    origin=destination_code,
                    destination=origin_code,
                    date=return_date,
                    adults=1,
                    seat_type="economy",
                    return_cheapest_only=False,
                    max_results=10,
                    compact_mode=True,
                )

                return_usable, return_text = _usable_flight_result(
                    raw_return
                )

                if not return_usable:
                    return_text = ""

            except Exception as exc:
                print(
                    f"Return one-way search failed for "
                    f"{destination_code} -> {origin_code}: "
                    f"{type(exc).__name__}: {exc}"
                )

            # Keep only combinations where at least one direction
            # returned actual usable data.
            if outbound_text or return_text:
                fallback_one_way_results.append(
                    f"### One-way fallback: "
                    f"{origin_code} <-> {destination_code}\n"
                    f"Origin airport: {origin_airport.name}\n"
                    f"Destination airport: {destination_airport.name}\n\n"
                    f"**Outbound ({departure_date}): "
                    f"{origin_code} -> {destination_code}**\n"
                    f"{outbound_text or 'No outbound flight data available.'}\n\n"
                    f"**Return ({return_date}): "
                    f"{destination_code} -> {origin_code}**\n"
                    f"{return_text or 'No return flight data available.'}"
                )

    # ------------------------------------------------------------
    # Choose what to send to the LLM.
    #
    # Round-trip results ALWAYS win. One-way results are used only
    # when no usable round-trip result exists.
    # ------------------------------------------------------------

    if round_trip_results:
        combined_flight_data = "\n\n".join(round_trip_results)
        search_mode = "round_trip"
    elif fallback_one_way_results:
        combined_flight_data = "\n\n".join(fallback_one_way_results)
        search_mode = "one_way_fallback"
    else:
        combined_flight_data = (
            "No usable flight results were returned for the requested "
            "airport combinations."
        )
        search_mode = "none"

    print("\n========== GOOGLE FLIGHTS RESULT ==========")
    print("Search mode:", search_mode)
    print(combined_flight_data[:16000])
    print("===========================================\n")

    # ------------------------------------------------------------
    # LLM formatting
    # ------------------------------------------------------------

    prompt = f"""
Answer ONLY the user's flight-related request.

Do not create an itinerary, hotel plan, general trip budget,
weather plan, or activity plan.

Search mode:
{search_mode}

The flight data below comes from the Google Flights MCP.

IMPORTANT:
- If search mode is "round_trip", present ONLY the round-trip
  flight results.
- If search mode is "one_way_fallback", explain that no usable
  round-trip result was found and present the outbound and return
  one-way searches separately.
- NEVER combine two one-way fares and call the result a verified
  round-trip fare.
- Do NOT invent a round-trip total from one-way prices.

Use ONLY the supplied flight data.

Do NOT invent:
- flight numbers
- airlines
- prices
- departure times
- arrival times
- durations
- stops
- availability

If information is unavailable, say so.

User request:
{query}

Trip constraints:
{constraints}

Resolved origin:
{origin_input} -> {[airport.iata for airport in origin_airports]}

Resolved destination:
{destination_input} -> {[airport.iata for airport in destination_airports]}

Google Flights MCP data:
{combined_flight_data[:16000]}

When data is available, include:
- airline
- departure time
- arrival time
- duration
- number of stops
- price
- flight type
- Google Flights link

If multiple airports were searched, clearly identify
which airport each option uses.

For round-trip results, clearly state that the price is the
round-trip price when the MCP data says so.

For one-way fallback results, keep:
- outbound ({departure_date}: {origin_input} -> {destination_input})
- return ({return_date}: {destination_input} -> {origin_input})
separate.

Do not discuss hotels, sightseeing, day-by-day itineraries,
weather, or the overall trip budget.

The response does not need to be JSON.
"""

    try:
        result = await _llm_text_async(
            "You are a flight search and comparison specialist. "
            "Use only verified MCP flight data and never fabricate flight details. "
            "Prefer round-trip results; one-way data is fallback only.",
            prompt,
        )

    except Exception as exc:
        print(
            f"Flight result formatting failed: "
            f"{type(exc).__name__}: {exc}"
        )

        result = combined_flight_data

    print("\n========== FLIGHT AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "flight_results": result,
        "messages": [
            AIMessage(content="Flight agent completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# HOTEL AGENT
# ============================================================
@traceable(name="Hotel Agent", run_type="chain")
async def hotel_agent(state: TravelState):
    query = (
        f"Find hotel/accommodation information relevant to this request. "
        f"Do not plan flights, weather, activities, or a full itinerary. "
        f"User request: {state['user_query']}"
    )

    print("\n========== HOTEL AGENT INPUT ==========")
    print(query)
    print("=======================================\n")

    try:
        result = await tavily_search(query)
        result = _format_tavily_result(result)
    except Exception as exc:
        print(f"Hotel search failed: {type(exc).__name__}: {exc}")
        result = "Hotel search data is currently unavailable."

    print("\n========== HOTEL SEARCH RESULT ==========")
    print(result)
    print("=========================================\n")

    return {
        "hotel_results": result,
        "messages": [
            AIMessage(content="Hotel agent completed.")
        ],
    }


# ============================================================
# WEATHER AGENT
# ============================================================
@traceable(name="Weather Agent", run_type="chain")
async def weather_agent(state: TravelState):

    city = _get_weather_city(state)

    print("\n========== WEATHER AGENT INPUT ==========")
    print("User query:", state.get("user_query"))
    print("Extracted city:", city)
    print("=========================================\n")

    if not city:
        result = "Please specify a city for the weather search."

        return {
            "weather_results": result,
            "messages": [AIMessage(content=result)],
        }

    try:
        weather_data = await current_weather(city)
    except Exception as exc:
        print(f"Current weather lookup failed: {exc}")
        weather_data = "Current weather data unavailable."

    try:
        forecast_data = await forecast(city)
    except Exception as exc:
        print(f"Forecast lookup failed: {exc}")
        forecast_data = "Forecast data unavailable."

    result = result = _format_weather_response(
        weather_data,
        forecast_data,
    )

    return {
        "weather_results": result,
        "messages": [
            AIMessage(content="Weather agent completed.")
        ],
    }

# ============================================================
# BUDGET AGENT
# ============================================================
@traceable(name="Budget Agent", run_type="chain")
def budget_agent(state: TravelState):

    print("\n========== BUDGET AGENT INPUT ==========")
    print("Trip Constraints:")
    print(state.get("trip_constraints"))

    print("\nFlight Results:")
    print(state.get("flight_results"))

    print("\nHotel Results:")
    print(state.get("hotel_results"))

    print("\nWeather Results:")
    print(state.get("weather_results"))
    print("=========================================\n")

    prompt = f"""
Analyze whether this trip plan is realistic for the user's budget.

User request:
{state['user_query']}

Constraints:
{state.get('trip_constraints', {})}

Flight results:
{state.get('flight_results', '')}

Hotel results:
{state.get('hotel_results', '')}

Weather results:
{state.get('weather_results', '')}

Return a concise budget assessment with:

1. estimated cost categories
2. risk areas
3. money-saving suggestions
4. whether the plan seems feasible
"""

    result = _llm_text(
        "You are a practical travel budget analyst. Analyze overall trip affordability only; do not create an itinerary or independently plan flights/hotels.",
        prompt,
    )

    print("\n========== BUDGET AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "budget_results": result,
        "messages": [
            AIMessage(content="Budget agent completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# ITINERARY AGENT
# ============================================================

@traceable(name="Itinerary Agent", run_type="chain")
def itinerary_agent(state: TravelState):

    print("\n========== ITINERARY AGENT INPUT ==========")

    print("Trip Constraints:")
    print(state.get("trip_constraints"))

    print("\nFlight Results:")
    print(state.get("flight_results"))

    print("\nHotel Results:")
    print(state.get("hotel_results"))

    print("\nWeather Results:")
    print(state.get("weather_results"))

    print("\nBudget Results:")
    print(state.get("budget_results"))

    print("===========================================\n")

    prompt = f"""
Create a complete travel itinerary ONLY because the supervisor routed this request
to itinerary_agent. Do not invent missing requirements. Use the supplied specialist
results when available. If a specialist was not requested, do not add that service
just because it might be useful.


User request:
{state['user_query']}

Trip constraints:
{state.get('trip_constraints', {})}

Flight results:
{state.get('flight_results', '')}

Hotel results:
{state.get('hotel_results', '')}

Weather results:
{state.get('weather_results', '')}

Budget results:
{state.get('budget_results', '')}

Make the output structured, practical, and ready for human review.
"""

    result = _llm_text(
        "You are an expert itinerary planner. This agent is only used for explicit trip-planning requests.",
        prompt,
    )

    print("\n========== ITINERARY OUTPUT ==========")
    print(result)
    print("======================================\n")

    approval_request = f"""
Please review this draft travel plan.

{result}

Reply with approval or feedback.
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [
            AIMessage(
                content="Draft itinerary created for human review."
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ============================================================
# HUMAN APPROVAL AGENT
# ============================================================

@traceable(name="Human Approval Agent", run_type="chain")
def human_approval_agent(state: TravelState):

    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get(
                "itinerary",
                "",
            ),
            "approval_request": state.get(
                "approval_request",
                "",
            ),
            "expected_response": {
                "approved": True,
                "feedback": "Optional feedback for revision",
            },
        }
    )

    feedback = feedback if isinstance(feedback, dict) else {}
    approved = bool(feedback.get("approved", False))
    human_feedback = str(feedback.get("feedback", ""))

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [
            AIMessage(
                content="Human approval step completed."
            )
        ],
    }


# ============================================================
# FINAL RESPONSE AGENT
# ============================================================
@traceable(name="Final Response Agent", run_type="chain")
def final_response_agent(state: TravelState):

    print("\n========== FINAL AGENT INPUT ==========")
    print("Approved:", state.get("approved"))
    print("Feedback:", state.get("human_feedback"))
    print("=======================================\n")

    if state["approved"]:

        prompt = f"""
The human approved this draft itinerary.

Produce the final polished travel plan.

Draft itinerary:
{state['itinerary']}

Budget notes:
{state['budget_results']}
"""

    else:

        prompt = f"""
The human did not approve the draft.

Original user request:
{state['user_query']}

Draft itinerary:
{state['itinerary']}

Human feedback:
{state['human_feedback']}

Budget notes:
{state['budget_results']}
"""

    result = _llm_text(
        "You produce final user-ready travel plans.",
        prompt,
    )

    print("\n========== FINAL RESPONSE ==========")
    print(result)
    print("====================================\n")

    return {
        "final_response": result,
        "messages": [
            AIMessage(content=result)
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }