import asyncio

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from agents import (
    budget_agent,
    final_response_agent,
    flight_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    supervisor_agent,
    weather_agent,
    _run_async,
)
from config import DATABASE_URL
from state import TravelState


AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents") or []
    return [agent for agent in AGENT_ORDER if agent in selected]


async def _parallel_specialist_agents(state: TravelState):
    selected = _selected_agents(state)

    tasks = []

    if "flight_agent" in selected:
        tasks.append(flight_agent(state))

    if "hotel_agent" in selected:
        tasks.append(hotel_agent(state))

    if "weather_agent" in selected:
        tasks.append(weather_agent(state))

    if not tasks:
        return {}

    results = await asyncio.gather(*tasks)

    combined = {}

    for result in results:
        combined.update(result)

    return combined


def parallel_specialist_agents(state: TravelState):
    return _run_async(_parallel_specialist_agents(state))


def route_after_parallel(state: TravelState) -> str:
    selected = _selected_agents(state)

    if "budget_agent" in selected:
        return "budget_agent"

    if "itinerary_agent" in selected:
        return "itinerary_agent"

    return END


def route_after_budget(state: TravelState) -> str:
    if "itinerary_agent" in _selected_agents(state):
        return "itinerary_agent"

    return END


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("parallel_specialists", parallel_specialist_agents)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_response", final_response_agent)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "parallel_specialists")

    graph.add_conditional_edges(
        "parallel_specialists",
        route_after_parallel,
        {
            "budget_agent": "budget_agent",
            "itinerary_agent": "itinerary_agent",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "budget_agent",
        route_after_budget,
        {
            "itinerary_agent": "itinerary_agent",
            END: END,
        },
    )

    graph.add_edge("itinerary_agent", "human_approval")
    graph.add_edge("human_approval", "final_response")
    graph.add_edge("final_response", END)

    if DATABASE_URL:
        # PostgresSaver.setup() creates indexes using CONCURRENTLY, which
        # cannot run inside a normal transaction.
        conn = psycopg.connect(DATABASE_URL, autocommit=True)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        return graph.compile(checkpointer=checkpointer)

    return graph.compile()


app = build_graph()
