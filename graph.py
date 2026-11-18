import asyncio

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.config import get_stream_writer
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
        tasks.append(
            _run_and_stream(
                "flight_agent",
                flight_agent(state),
                "flight_results",
            )
        )

    if "hotel_agent" in selected:
        tasks.append(
            _run_and_stream(
                "hotel_agent",
                hotel_agent(state),
                "hotel_results",
            )
        )

    if "weather_agent" in selected:
        tasks.append(
            _run_and_stream(
                "weather_agent",
                weather_agent(state),
                "weather_results",
            )
        )

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

def _emit_agent_result(agent_name: str, result: dict, result_key: str):
    """Send a completed agent result to the frontend immediately."""
    content = result.get(result_key, "")

    if not content:
        return

    writer = get_stream_writer()
    writer({
        "type": "agent_result",
        "agent": agent_name,
        "content": content,
    })


async def _run_and_stream(agent_name: str, coro, result_key: str):
    """Run a specialist and stream its result as soon as it finishes."""
    result = await coro
    _emit_agent_result(agent_name, result, result_key)
    return result


def budget_agent_with_stream(state: TravelState):
    result = budget_agent(state)
    _emit_agent_result("budget_agent", result, "budget_results")
    return result


def itinerary_agent_with_stream(state: TravelState):
    result = itinerary_agent(state)
    _emit_agent_result("itinerary_agent", result, "itinerary")
    return result


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("parallel_specialists", parallel_specialist_agents)
    graph.add_node("budget_agent", budget_agent_with_stream)
    graph.add_node("itinerary_agent", itinerary_agent_with_stream)
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
