# AI Travel Planner

A multi-agent AI travel planning system built using **LangGraph, MCP
(Model Context Protocol), and LLMs**. It uses specialized agents to help
users plan trips by researching flights, hotels, weather conditions,
budgets, and itineraries.

The application includes human approval before presenting the final
travel plan and supports persistent conversation state through
PostgreSQL.

## Features

-   **Multi-Agent Architecture:** Uses LangGraph to coordinate
    specialized travel-planning agents.
-   **Supervisor Agent:** Understands the user's request, extracts trip
    constraints, and selects the relevant agents.
-   **Flight Search:** Integrates with Google Flights MCP to search
    one-way and round-trip flights.
-   **Hotel Research:** Uses Tavily search to find accommodation options
    and relevant information.
-   **Weather Information:** Uses a custom OpenWeatherMap MCP server to
    retrieve current weather and forecasts.
-   **Budget Analysis:** Estimates and analyses trip expenses using an
    LLM and the available specialist results.
-   **Itinerary Generation:** Generates a travel itinerary based on the
    user's preferences and research results.
-   **Human-in-the-Loop:** Allows users to review and approve the
    proposed plan or provide feedback.
-   **PostgreSQL Checkpointing:** Supports persistent LangGraph state
    and conversation resumption.
-   **Streamlit Interface:** Provides an interactive UI for planning
    trips and reviewing agent outputs.
-   **Multiple LLM Providers:** Supports Gemini, Groq, and OpenAI.

## Tech Stack

-   **Language:** Python
-   **Agent Orchestration:** LangGraph, LangChain
-   **LLM Providers:** Google Gemini, Groq, OpenAI
-   **Tool Integration:** MCP, LangChain MCP Adapters
-   **Search:** Tavily
-   **Weather API:** OpenWeatherMap
-   **Flight Search:** Google Flights MCP
-   **Database:** PostgreSQL, LangGraph Checkpoint Postgres
-   **Frontend:** Streamlit

## Architecture

The application follows a supervisor-based multi-agent architecture.

1.  **Supervisor Agent:** Analyses the user's request, identifies trip
    constraints, and determines which specialist agents are required.
2.  **Specialist Agents:** Flight, hotel, and weather agents execute
    their respective tasks. These agents can run concurrently.
3.  **Budget Agent:** Analyses estimated costs using the user's
    requirements and specialist results.
4.  **Itinerary Agent:** Creates a structured itinerary using the
    available research and trip constraints.
5.  **Human Approval:** Presents the proposed plan for user approval or
    feedback.
6.  **Final Response Agent:** Generates the final response based on the
    approval status and any feedback.

The workflow is implemented using LangGraph conditional routing,
parallel execution, and human-in-the-loop interrupts.

## Project Structure

``` text
ai_travel_planner/
│
├── agents.py                       # Agent definitions and LLM logic
├── airport_resolver.py             # Airport and IATA code resolution
├── config.py                       # LLM and API configuration
├── custom_meather_mcp_server.py    # Custom OpenWeatherMap MCP server
├── frontend.py                     # Streamlit application
├── graph.py                        # LangGraph workflow and orchestration
├── mcp_client.py                   # MCP client and tool integrations
├── state.py                        # Shared graph state
│
├── aviationstack-mcp/               # Existing submodule
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Prerequisites

-   Python 3.10+
-   PostgreSQL (optional; required for persistent checkpointing)
-   API keys for the services you intend to use:
    -   Google Gemini
    -   Tavily
    -   OpenWeatherMap
-   Google Flights MCP server set up locally

## Installation and Setup

### 1. Clone the repository

``` bash
git clone https://github.com/EshikaGupta/ai_travel_planner.git
cd ai_travel_planner
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

Linux/macOS:

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a
reference.

``` env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_gemini_model

TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweathermap_api_key

AVIATIONSTACK_API_KEY=your_aviationstack_api_key

DATABASE_URL=postgresql://username:password@localhost:5432/langgraph_memory_demo
```

The application also supports provider configuration through
`LLM_PROVIDER` (`gemini`, `groq`, or `openai`). Provider-specific model
settings and temperature can be configured through the variables read in
`config.py`.

### 5. Set up the Google Flights MCP server

The flight integration currently uses a separate Google Flights MCP
server, not the AviationStack MCP submodule.

The application expects the server at:

``` text
ai_travel_planner/
└── google-flights-mcp/
    └── src/
        └── mcp_server_google_flights/
            └── server.py
```

The MCP client also expects the server's Python environment at:

``` text
google-flights-mcp/.venv/Scripts/python.exe
```

Set up the Google Flights MCP project separately and ensure its server
file and Python environment are available at these expected paths.
Update the paths in `mcp_client.py` if your setup differs.

### 6. Weather MCP server

The project includes a custom OpenWeatherMap MCP server in:

``` text
custom_meather_mcp_server.py
```

It exposes tools for:

-   `get_current_weather(city)`
-   `get_forecast(city)`

The server is launched by the MCP client, so it does not need to be
started separately. Ensure `OPENWEATHER_API_KEY` is configured.

### 7. Tavily integration

Tavily is integrated through its hosted MCP endpoint in `mcp_client.py`.
Configure `TAVILY_API_KEY` in your `.env` file. A separate local Tavily
MCP server installation is not required.

### 8. PostgreSQL setup (optional)

PostgreSQL is used for persistent LangGraph checkpointing.

Create a database:

``` sql
CREATE DATABASE langgraph_memory_demo;
```

Configure `DATABASE_URL` in your `.env` file.

If `DATABASE_URL` is configured, the application uses `PostgresSaver`
for checkpoint persistence. Without it, the graph runs without
PostgreSQL persistence.

## Running the Application

Launch the Streamlit interface from the project root:

``` bash
streamlit run frontend.py
```

The interface allows users to:

-   Enter travel requirements and preferences.
-   Select an LLM provider.
-   Follow the progress of the specialist agents.
-   Review research results and the generated itinerary.
-   Approve the proposed plan or provide feedback.
-   View and download the final travel plan.

## Example Query

``` text
Plan a 5-day trip to Japan from Delhi with a budget of
₹1,00,000. I prefer cultural experiences, local food,
and comfortable accommodation.
```

The supervisor analyses the request and routes it to the relevant
agents. The system researches available travel information, estimates
the budget, generates an itinerary, and requests user approval before
producing the final response.

## Notes

-   Flight search depends on the separately configured Google Flights
    MCP server.
-   Hotel research uses web search results and does not represent
    verified hotel availability or booking.
-   Budget estimates are LLM-generated estimates, not guaranteed prices.
-   PostgreSQL persistence is optional; without a configured database,
    conversation checkpointing is not persisted across application
    restarts.
