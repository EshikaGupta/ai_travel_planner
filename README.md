# AI Travel Planner

A supervisor-orchestrated, multi-agent travel planning application built with **Python, LangGraph, LangChain, MCP, and Streamlit**. Users describe a travel request in natural language, and a supervisor selects the specialist agents needed to handle it. The application can research flights and accommodation, retrieve weather information, estimate trip costs, draft an itinerary, and request human approval before generating a final response.

The project also includes **LangSmith tracing** for inspecting graph execution, agent runs, LLM calls, and MCP tool execution.

## Features

- **Supervisor-based routing:** A travel guardrail checks whether a request is travel-related. The supervisor then identifies the primary request type, extracts trip constraints, and selects the relevant agents.
- **Specialist agents:** Separate agents handle flights, accommodation research, weather, budget analysis, and itinerary generation.
- **Parallel research:** Flight, hotel, and weather agents are run concurrently when selected. Their completed results can be streamed to the Streamlit interface as they finish.
- **Flight search:** Integrates with a Google Flights MCP server. It resolves city, country, or airport inputs to IATA codes using an OurAirports-based local data cache, searches round-trip flights, and falls back to separate outbound and return one-way searches if usable round-trip data is unavailable.
- **Accommodation research:** Uses the hosted Google Hotels MCP endpoint to search for relevant accommodation information and formats returned search results for display.
- **Weather information:** Includes a custom OpenWeatherMap MCP server with current-weather and forecast tools.
- **Budget analysis:** Uses an LLM to assess overall trip affordability, cost categories, risk areas, and potential savings based on the request and available specialist results.
- **Itinerary generation:** Creates a structured trip plan using the user's request, constraints, and available specialist outputs.
- **Human-in-the-loop review:** Uses a LangGraph interrupt to pause for approval or feedback on the draft itinerary. A final-response agent produces the user-facing response after the review decision.
- **Conversation checkpointing:** Supports PostgreSQL-backed LangGraph checkpoint persistence when `DATABASE_URL` is configured. Without it, the graph runs without PostgreSQL checkpoint persistence.
- **Multiple LLM providers:** Supports Google Gemini and Groq. The Streamlit interface allows the user to select the provider.
- **LangSmith observability:** Uses `@traceable` spans and LangGraph/LangChain tracing to inspect supervisor, specialist-agent, structured LLM, and MCP tool execution.
- **Streamlit interface:** Includes example trip prompts, recent-trip session history, agent outputs, human approval controls, final response display, and response download.

## Architecture

The application is implemented as a LangGraph state graph.

1. **Input and guardrail:** The supervisor checks whether the request is travel-related.
2. **Supervisor:** Classifies the request, extracts trip constraints, and chooses the agents required for the user's primary intent.
3. **Parallel specialist stage:** Selected flight, hotel, and weather agents execute concurrently. Each specialist returns its result into the shared graph state.
4. **Budget stage:** When selected, the budget agent reviews the user's budget and the available research results.
5. **Itinerary stage:** When selected, the itinerary agent generates a draft using the request, constraints, and specialist outputs.
6. **Human approval:** The graph pauses with an interrupt so the user can approve the draft or provide feedback.
7. **Final response:** The final-response agent creates a polished response based on the approval decision and feedback.

The graph uses conditional routing to avoid running specialist agents that are not needed. Flight, hotel, and weather work is performed in parallel; budget and itinerary work run downstream so they can use the specialist results.

### High-level flow

```mermaid
flowchart TD
    A([User request]) --> B[Supervisor and travel guardrail]
    B --> C[Select specialist agents]
    C --> D{Parallel specialist stage}
    D --> E[Flight agent]
    D --> F[Hotel agent]
    D --> G[Weather agent]
    E --> H[Merge specialist results]
    F --> H
    G --> H
    H --> I{Budget agent selected?}
    I -- Yes --> J[Budget analysis]
    I -- No --> K{Itinerary agent selected?}
    J --> K
    K -- Yes --> L[Generate itinerary draft]
    K -- No --> M([End])
    L --> N[Human approval interrupt]
    N --> O[Final response agent]
    O --> P([Final travel plan])
```

## Tech Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Agent orchestration | LangGraph |
| LLM framework | LangChain |
| LLM providers | Google Gemini, Groq |
| Tracing and observability | LangSmith |
| Tool protocol and adapters | Model Context Protocol (MCP), LangChain MCP Adapters |
| Flight search | Google Flights MCP server |
| Accommodation/web research | Hosted Google Hotels MCP endpoint |
| Weather | Custom OpenWeatherMap MCP server |
| Airport resolution | OurAirports CSV data |
| Checkpoint persistence | PostgreSQL, LangGraph Postgres checkpointer |
| User interface | Streamlit |

## Project Structure

```text
ai_travel_planner/
├── agents.py                       # Guardrail, supervisor, specialist and response agents
├── airport_resolver.py             # City/airport/country to IATA airport resolution
├── config.py                       # LLM provider and API/database configuration
├── custom_meather_mcp_server.py    # Custom OpenWeatherMap MCP server
├── frontend.py                     # Streamlit interface and streamed agent results
├── graph.py                        # LangGraph state graph, routing and streaming helpers
├── mcp_client.py                   # MCP server configuration, wrappers and tracing
├── state.py                        # Shared LangGraph state schema
├── data/
│   ├── airports.csv                # OurAirports airport reference data
│   └── countries.csv               # OurAirports country reference data
├── google-flights-mcp/             # External MCP project; set up separately (not included)
├── .env.example                    # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

The `google-flights-mcp/` directory is an external dependency and is not included in this repository tree. The current MCP client expects it to be installed alongside the application.

## Prerequisites

- Python 3.10 or newer.
- API key for at least one supported LLM provider: Google Gemini or Groq.
- HasData (Google hotels) API key for accommodation/web research.
- OpenWeatherMap API key for weather tools.
- Google Flights MCP server set up locally for flight search.
- PostgreSQL is optional and is needed only when persistent graph checkpointing is desired.
- LangSmith API key is optional, but required to send traces to a LangSmith project.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/EshikaGupta/ai_travel_planner.git
cd ai_travel_planner
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows**

```powershell
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a template.

```env
# Choose a provider: gemini or groq
LLM_PROVIDER=gemini

# Google Gemini
GEMINI_API_KEY=
GEMINI_MODEL=

# Groq
GROQ_API_KEY=
GROQ_MODEL=

# External services
HASDATA_API_KEY=
OPENWEATHER_API_KEY=

# Optional PostgreSQL checkpointing
DATABASE_URL=

# Optional LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=ai-travel-planner

# Optional model temperature
LLM_TEMPERATURE=0.2

# Optional: Python executable used to launch the local weather MCP server
# Defaults to the Python executable running the application.
PYTHON_EXECUTABLE=
```

Only configure the provider you intend to use. The Streamlit sidebar also provides a provider selector for Gemini and Groq.

Keep your real `.env` file and API keys out of source control.

### 5. Set up the Google Flights MCP server

Flight search uses a separate local Google Flights MCP project. It is not bundled in this repository.

The current `mcp_client.py` expects the following layout:

```text
ai_travel_planner/
└── google-flights-mcp/
    ├── .venv/
    │   └── Scripts/
    │       └── python.exe
    └── src/
        └── mcp_server_google_flights/
            └── server.py
```

The client launches the server using the Python executable at `google-flights-mcp/.venv/Scripts/python.exe` and the server script at `google-flights-mcp/src/mcp_server_google_flights/server.py`.

Set up the external Google Flights MCP project and its virtual environment at those paths. If your installation differs, update the paths in `mcp_client.py`. The client also supports setting `PYTHON_EXECUTABLE` for the weather MCP server; the Google Flights paths are currently defined directly in `mcp_client.py`.

### 6. Weather MCP server

The repository includes `custom_meather_mcp_server.py`, a custom MCP server exposing:

- `get_current_weather(city)`
- `get_forecast(city)`

The server is launched by `mcp_client.py` over stdio when a weather tool is called. It does not need to be started separately. Configure `OPENWEATHER_API_KEY` before using weather functionality.

### 7. Google Hotels MCP integration

Accommodation research uses HASDATA's hosted MCP endpoint configured in `mcp_client.py`. Set `HASDATA_API_KEY` in `.env`. A separate local Google Hotels server installation is not required.

### 8. Airport reference data

`airport_resolver.py` resolves airport names, city names, country names, and IATA codes using OurAirports CSV datasets. The repository includes the `data/airports.csv` and `data/countries.csv` reference files. If either file is missing locally, the resolver downloads it from OurAirports and caches it in the `data/` directory.

### 9. PostgreSQL checkpointing (optional)

PostgreSQL is optional. When `DATABASE_URL` is set, `graph.py` creates a PostgreSQL connection, initializes the LangGraph `PostgresSaver`, and compiles the graph with checkpointing enabled.

Create a database, for example:

```sql
CREATE DATABASE langgraph_memory_demo;
```

Then configure a connection string in `.env`:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/langgraph_memory_demo
```

When `DATABASE_URL` is empty or unset, the graph is compiled without the PostgreSQL checkpointer. In that configuration, graph state is not persisted across application restarts.

### 10. LangSmith tracing (optional)

LangSmith is listed in `requirements.txt`. Configure:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=ai-travel-planner
```

The code uses `@traceable` on selected agent functions, the structured LLM invocation, and the central MCP tool execution wrapper. LangGraph and LangChain model execution tracing can also appear in the project when tracing is enabled. Inspect the LangSmith project to explore graph runs, agent spans, model calls, and MCP tool calls.

Tracing is optional for running the application. Do not commit your LangSmith API key.

## Running the Application

From the project root, run:

```bash
streamlit run frontend.py
```

The Streamlit app provides:

- Quick-start trip prompts (Dubai, Japan, Paris, Bali, and Rome).
- A natural-language travel request input.
- Gemini/Groq provider selection.
- Recent-trip history within the current Streamlit session.
- Agent result sections displayed as completed agent events are streamed to the interface.
- Draft itinerary review and approval/feedback controls.
- Final response display and text-file download.

## Example Requests

**Complete trip plan**

```text
Plan a 7-day Japan trip from Delhi including flights, hotels and sightseeing under INR 2 lakh.
```

**Flight-only request**

```text
Find round-trip flights from Delhi to Dubai for 10 December 2026 to 15 December 2026.
```

**Accommodation research**

```text
Find affordable places to stay in Bali for a 10-day backpacking trip.
```

**Weather request**

```text
What is the current weather and forecast in Jaipur?
```

The supervisor is intended to route each request to the minimum set of agents needed for its primary intent. For example, a standalone weather request should not automatically trigger itinerary generation.

## Observability

LangSmith is used to inspect execution rather than just the final answer. Current explicit traceable spans include:

- Structured LLM invocation.
- Supervisor agent.
- Flight, hotel, weather, budget, and itinerary agents.
- Human approval and final response agents.
- Central MCP tool execution wrapper.

LangGraph execution and underlying LangChain model calls can also appear as nested runs. Internal LangChain Runnables (for example, sequences, parallel steps, and parsing/fallback wrappers) are implementation details; they should not be counted as separate agents or model calls in application-level analytics.

## Important Notes and Limitations

- **Flight search requires an external local dependency.** The Google Flights MCP server must be installed and configured separately.
- **Flight search requires exact dates.** If a departure date is missing, the flight agent asks for one; it does not perform a month-wide flight search.
- **Flight results are search results, not bookings.** The application does not purchase or reserve tickets.
- **Accommodation results are research, not confirmed inventory.** Google Hotels search results do not establish live room availability, final prices, or a hotel booking.
- **Budget results are estimates.** They are generated by an LLM from the request and available research and should not be treated as guaranteed quotes.
- **Weather forecasts are API data.** The custom server retrieves current conditions and a short forecast from OpenWeatherMap.
- **PostgreSQL is optional.** Without `DATABASE_URL`, graph checkpoint persistence is disabled.
- **LangSmith is optional.** Without LangSmith credentials and tracing configuration, remote tracing is not available.
- **Human approval applies to itinerary workflows.** The graph pauses for review before the final response stage when an itinerary is generated.

## License

No license file is currently included in the repository. Add a license file if you intend to specify reuse or distribution terms.
