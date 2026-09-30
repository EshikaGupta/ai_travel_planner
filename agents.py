import asyncio
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import get_llm
from mcp_client import (
    current_weather,
    forecast,
    list_airlines,
    list_airports,
    tavily_search,
)
from state import TravelState
import streamlit as st

def _get_llm():
    provider = st.session_state.get("llm_provider", "gemini")
    return get_llm(provider)


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

def _content_to_text(content: Any) -> str:
    """Normalize LangChain content from Gemini, Groq, OpenAI, etc. to text."""
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
        return _json_fallback(raw_text, model_class)

    except Exception as exc:
        # Some provider/model combinations may not expose structured output.
        # Keep the application working by falling back to normal text output.
        print(f"Structured output unavailable: {type(exc).__name__}: {exc}")
        raw_text = _llm_text(system, prompt)
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

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        data = None
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                data, _ = decoder.raw_decode(cleaned[index:])
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            raise ValueError(f"Could not parse JSON from LLM response:\n{text}")

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
            "travel_style": "",
            "special_preferences": [],
        },
        "reasoning": "Used deterministic intent routing because structured supervisor output was unavailable.",
    }


# ============================================================
# SUPERVISOR AGENT
# ============================================================

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

async def flight_agent(state: TravelState):
    query = state["user_query"]
    constraints = state["trip_constraints"]
    destination = constraints.get("destination", "")

    print("\n========== FLIGHT AGENT INPUT ==========")
    print("Query:", query)
    print("Constraints:", constraints)
    print("========================================\n")

    try:
        airports = await list_airports(destination, limit=10)
    except Exception as exc:
        print(f"Flight airport lookup failed: {type(exc).__name__}: {exc}")
        airports = "Airport data unavailable."

    try:
        airlines = await list_airlines("", limit=10)
    except Exception as exc:
        print(f"Flight airline lookup failed: {type(exc).__name__}: {exc}")
        airlines = "Airline data unavailable."

    print("\n========== AIRPORT MCP DATA ==========")
    print(airports)
    print("======================================\n")

    print("\n========== AIRLINE MCP DATA ==========")
    print(airlines)
    print("======================================\n")

    prompt = f"""
Answer ONLY the user's flight-related request.
Do not create an itinerary, hotel plan, general trip budget, weather plan, or activity plan.
If the user gave a fare limit such as "under INR 50,000", treat it only as a flight-price constraint.
If exact fares are not available from the supplied MCP data, clearly say that instead of inventing prices.

User request:
{query}

Trip constraints:
{constraints}

Airport MCP data:
{str(airports)[:3000]}

Airline MCP data:
{str(airlines)[:3000]}

Include:

- likely departure/arrival airports
- relevant airlines
- estimated duration
- fare range
- peak season warning
- booking advice relevant to the requested flight

Do not discuss hotels, sightseeing, day-by-day itineraries, or an overall trip budget.
The response does not need to be JSON.
"""

    result = await _llm_text_async(
        "You are a flight planning specialist.",
        prompt,
    )

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
    except Exception as exc:
        print(f"Hotel search failed: {type(exc).__name__}: {exc}")
        result = "Hotel search data is currently unavailable."

    print("\n========== HOTEL SEARCH RESULT ==========")
    print(result)
    print("=========================================\n")

    return {
        "hotel_results": str(result),
        "messages": [
            AIMessage(content="Hotel agent completed.")
        ],
    }


# ============================================================
# WEATHER AGENT
# ============================================================

async def weather_agent(state: TravelState):
    constraints = state["trip_constraints"]
    city = constraints.get("destination", "")

    print("\n========== WEATHER AGENT INPUT ==========")
    print("City:", city)
    print("=========================================\n")

    try:
        weather_data = await current_weather(city)
    except Exception as exc:
        print(f"Current weather lookup failed: {type(exc).__name__}: {exc}")
        weather_data = "Current weather data unavailable."

    try:
        forecast_data = await forecast(city)
    except Exception as exc:
        print(f"Forecast lookup failed: {type(exc).__name__}: {exc}")
        forecast_data = "Forecast data unavailable."

    print("\n========== CURRENT WEATHER ==========")
    print(weather_data)
    print("=====================================\n")

    print("\n========== WEATHER FORECAST ==========")
    print(forecast_data)
    print("======================================\n")

    result = f"""
Current weather:
{weather_data}

Forecast:
{forecast_data}
"""

    print("\n========== WEATHER AGENT OUTPUT ==========")
    print(result)
    print("==========================================\n")

    return {
        "weather_results": result,
        "messages": [
            AIMessage(content="Weather agent completed.")
        ],
    }


# ============================================================
# BUDGET AGENT
# ============================================================

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