import os
from typing import TypedDict, Annotated
import operator
import psycopg
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (AnyMessage,HumanMessage, AIMessage,SystemMessage)

from mcp_client import get_llm,tavily_mcp_search,aviation_mcp_call,get_airlines,get_airports,extract_destination,weather_mcp_search,forecast_mcp_search

import asyncio
from dotenv import load_dotenv
load_dotenv()


DATABASE_URL=os.getenv("DATABASE_URL")

class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str
    flight_results:str
    hotel_results:str
    itinerary:str
    llm_calls:int
    weather_results: str

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""



# Flight Agent
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:

        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = get_llm().invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def hotel_agent(state: TravelState):
    print("\nINSIDE HOTEL AGENT\n")

    query = f"""
    Find specific hotel properties suitable for this travel request:

    {state['user_query']}

    Focus on:
    - Actual hotel names
    - Relevant cities
    - Approximate prices
    - Hotel features
    - Budget suitability

    Prefer specific hotels over generic travel guides.
    """

    try:
        raw_results = asyncio.run(
            tavily_mcp_search(query)
        )

        # Extract only useful information from Tavily response
        if isinstance(raw_results, dict):
            results = raw_results.get("results", [])
        else:
            results = raw_results

        hotel_sources = []

        if isinstance(results, list):
            for result in results[:8]:
                if isinstance(result, dict):
                    title = result.get("title", "")
                    url = result.get("url", "")
                    content = result.get("content", "")

                    hotel_sources.append(
                        f"""
                        TITLE: {title}
                        URL: {url}
                        CONTENT:
                        {content[:2500]}
                        """
                    )

        hotel_context = "\n\n".join(hotel_sources)

        if not hotel_context:
            return {
                "hotel_results": "No useful hotel information was found.",
                "messages": [
                    AIMessage(content="Hotel information unavailable")
                ]
            }

        # Convert raw search results into clean hotel recommendations
        hotel_prompt = f"""
You are a hotel recommendation expert.

User travel request:
{state['user_query']}

Below are web search results:

{hotel_context}

Extract and organize ONLY useful hotel information.

Requirements:
1. Mention specific hotel/property names where available.
2. Do NOT reproduce the raw search results.
3. Do NOT output JSON.
4. Do NOT mention search engine results.
5. Remove irrelevant travel-guide information.
6. Include approximate prices only when supported by the sources.
7. Include city/location.
8. Include important features/amenities when available.
9. Include the source URL for each recommendation.
10. If exact prices are unavailable, say "Price not available in source".
11. Do not invent hotel names, prices, ratings, or amenities.
12. If the sources contain mostly general travel guides rather than specific hotels, clearly say so.

Return the result in this format:

HOTEL RECOMMENDATIONS

1. Hotel Name
   Location:
   Approx. price:
   Why it fits:
   Source:

2. Hotel Name
   Location:
   Approx. price:
   Why it fits:
   Source:

Also provide a short:
BUDGET NOTE:
Explain whether the available hotel options appear compatible with the user's stated budget.
"""

        response = get_llm().invoke([
            SystemMessage(
                content="You are an expert hotel recommendation assistant."
            ),
            HumanMessage(content=hotel_prompt)
        ])

        hotel_results = response.content

        return {
            "hotel_results": hotel_results,
            "messages": [
                AIMessage(content="Hotel recommendations generated")
            ],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

    except Exception as e:

        return {
            "hotel_results": f"Hotel information unavailable: {str(e)}",
            "messages": [
                AIMessage(content="Hotel search failed")
            ]
        }

def weather_agent(state: TravelState):

    city = extract_destination(state["user_query"])

    weather_data = asyncio.run(
        weather_mcp_search(city)
    )

    forecast_data = asyncio.run(
        forecast_mcp_search(city)
    )

    return {
        "weather_results": f"""
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """,
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ]
    }


def itinerary_agent(state:TravelState):
    prompt=f"""Create a travel itinerary. 
    User query: {state['user_query']}
    Flight results:{state['flight_results']}
    Weather results:{state['weather_results']}
    Hotel results:{state['hotel_results']} """

    response=get_llm().invoke([
        SystemMessage(
           content="You are an expert travel planner" 
        ),
        HumanMessage(content=prompt)]
    )
    return {"itinerary":response.content, "messages":[response],"llm_calls":state.get("llm_calls",0)+1}

# def final_agent(state:TravelState):
#     final_prompt=f"""
#     Generate final travel response.
#     Flights:{state['flight_results']}
#     Hotels:{state['hotel_results']}
#     Itinerary:{state['itinerary']}"""
#     response=get_llm().invoke([
#         HumanMessage(content=final_prompt)
#     ])
#     return {"messages":[response],"llm_calls":state.get("llm_calls",0)+1}

graph=StateGraph(TravelState)
graph.add_node("flight_agent",flight_agent)
graph.add_node("hotel_agent",hotel_agent)
graph.add_node("weather_agent",weather_agent)
graph.add_node("itinerary_agent",itinerary_agent)


graph.add_edge(START,"flight_agent")
graph.add_edge("flight_agent","hotel_agent")
graph.add_edge("hotel_agent","weather_agent")
graph.add_edge("weather_agent","itinerary_agent")
graph.add_edge("itinerary_agent",END)

_conn=psycopg.connect(DATABASE_URL,autocommit=True)
checkpointer=PostgresSaver(_conn)
checkpointer.setup()

app=graph.compile(checkpointer=checkpointer)

if __name__=="__main__":
    config={"configurable":{
        "thread_id":"user1"
    }}
    user_input=input("Enter travel request:")
    result=app.invoke({
        "messages":[HumanMessage(content=user_input)],
        "user_query":user_input,
        "flight_results":"",
        "hotel_results":"",
        "weather_results":"",
        "itinerary":"",
        "llm_calls":0
    },config=config)
    print("\nFINAL RESPONSE:\n")
    for msg in result["messages"]:
        print(msg.content)