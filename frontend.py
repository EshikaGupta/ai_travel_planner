import os
import uuid
from datetime import datetime

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Travel Booking System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

def init_state():
    defaults = {
        "user_id": "demo_user",
        "thread_id": None,
        "llm_provider": "openai",
        "latest_result": None,
        "waiting_for_approval": False,
        "history": [],
        "active_history_index": None,
        "query_text": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.thread_id:
        st.session_state.thread_id = new_thread_id(st.session_state.user_id)


def new_thread_id(user_id: str) -> str:
    safe_user = (user_id or "user").strip().replace(" ", "_")
    return f"{safe_user}_{uuid.uuid4().hex[:8]}"


init_state()


# ============================================================
# QUICK PACKAGES
# ============================================================

PACKAGES = [
    {
        "name": "Dubai Weekend",
        "short": "🇦🇪 Dubai",
        "query": "Plan a 3-day Dubai weekend trip including flights, hotels and sightseeing.",
        "image": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=700&q=80",
    },
    {
        "name": "Japan Explorer",
        "short": "🇯🇵 Japan",
        "query": "Plan a 7-day Japan trip including flights, hotels and sightseeing under INR 2 lakh.",
        "image": "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=700&q=80",
    },
    {
        "name": "Paris Escape",
        "short": "🇫🇷 Paris",
        "query": "Plan a 5-day Paris trip including flights, hotels and major attractions.",
        "image": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=700&q=80",
    },
    {
        "name": "Bali Backpacking",
        "short": "🇮🇩 Bali",
        "query": "Plan a 10-day budget backpacking trip to Bali with affordable stays and activities.",
        "image": "https://images.unsplash.com/photo-1537996194471-e657df975ab4?w=700&q=80",
    },
    {
        "name": "Rome Getaway",
        "short": "🇮🇹 Rome",
        "query": "Plan a 5-day Rome trip including flights, hotels, sightseeing and food experiences.",
        "image": "https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=700&q=80",
    },
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #070b12;
}

/* Keep Streamlit's header and native sidebar controls visible. */
[data-testid="stHeader"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: transparent !important;
}

/* Never hide the native sidebar expand/collapse controls. */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarNavCollapseIcon"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    z-index: 1000000 !important;
}

/* Keep the header/toolbar available because Streamlit places the sidebar toggle there. */
[data-testid="stToolbar"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Hide only the old menu/footer, never the header. */
#MainMenu, footer {
    display: none !important;
}

/* Make the sidebar a little wider and keep its background distinct. */
section[data-testid="stSidebar"] {
    min-width: 290px !important;
    max-width: 290px !important;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] {
    background: #080d17 !important;
    border-right: 1px solid #182337;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

.sidebar-brand {
    font-size: 0.95rem;
    font-weight: 700;
    color: #eaf4ff;
    margin-bottom: 1.3rem;
}

.sidebar-muted {
    color: #6f89a5;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
    margin: 1rem 0 0.45rem;
}

.sidebar-tech {
    background: #0d1625;
    border: 1px solid #172942;
    border-radius: 8px;
    padding: 0.48rem 0.7rem;
    color: #9bb6d1;
    font-size: 0.78rem;
    margin-bottom: 0.35rem;
}

.history-empty {
    color: #526b84;
    font-size: 0.78rem;
    line-height: 1.5;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: #0d1625;
    color: #a9bfd6;
    border: 1px solid #172942;
    border-radius: 8px;
    font-size: 0.78rem;
    padding: 0.48rem 0.65rem;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: #11223a;
    border-color: #2a72c8;
    color: #ffffff;
}

/* ---------------- Hero ---------------- */
.hero {
    height: 250px;
    border-radius: 16px;
    overflow: hidden;
    position: relative;
    margin-bottom: 1.3rem;
    border: 1px solid #1b2b42;
    background:
        linear-gradient(90deg, rgba(4,9,17,0.94), rgba(4,9,17,0.58), rgba(4,9,17,0.78)),
        url('https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1600&q=85') center/cover;
}

.hero-inner {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 2.4rem;
}

.hero-badge {
    display: inline-block;
    width: fit-content;
    background: rgba(20,105,205,0.2);
    border: 1px solid rgba(69,157,255,0.45);
    color: #77b8ff;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}

.hero-title {
    color: white;
    font-size: 2.45rem;
    line-height: 1.1;
    font-weight: 800;
    margin-bottom: 0.55rem;
}

.hero-subtitle {
    color: #a4bad1;
    max-width: 680px;
    font-size: 0.92rem;
    line-height: 1.6;
}

/* ---------------- Destination cards ---------------- */
.package-card {
    height: 105px;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    border: 1px solid #20334c;
    background-size: cover;
    background-position: center;
}

.package-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(transparent 25%, rgba(3,7,14,0.88));
}

.package-title {
    position: absolute;
    z-index: 2;
    left: 0.75rem;
    bottom: 0.55rem;
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
}

/* ---------------- Main input ---------------- */
.section-label {
    color: #72b3f7;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.3rem 0 0.5rem;
}

div[data-testid="stTextArea"] textarea {
    background: #09121f !important;
    color: #e8f4ff !important;
    border: 1px solid #27405e !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #3289e8 !important;
    box-shadow: 0 0 0 1px #3289e8 !important;
}

/* ---------------- Buttons ---------------- */
div[data-testid="stButton"] > button[kind="primary"] {
    min-height: 48px;
    border: 0 !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #0878e8, #0756ad) !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 24px rgba(0,100,220,0.25);
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1590ff, #0865c8) !important;
    box-shadow: 0 10px 30px rgba(0,100,220,0.4);
}

/* Quick package buttons */
.quick-package + div button {
    min-height: 40px !important;
    font-size: 0.75rem !important;
}

/* ---------------- Results ---------------- */
.result-card {
    background: #0b1421;
    border: 1px solid #1b304a;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    min-height: 100px;
}

.result-title {
    color: #9bcaff;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    margin-bottom: 0.55rem;
}

.final-card {
    background: linear-gradient(145deg, #0c1c2f, #09121e);
    border: 1px solid #244a70;
    border-left: 4px solid #1685f5;
    border-radius: 12px;
    padding: 1.25rem 1.35rem;
    color: #d7e8f8;
    line-height: 1.7;
}

.agent-pill {
    display: inline-block;
    padding: 0.3rem 0.65rem;
    border-radius: 999px;
    background: #0d2035;
    border: 1px solid #1d4368;
    color: #83bcf0;
    font-size: 0.72rem;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
}

.metric-card {
    background: #0b1421;
    border: 1px solid #1b304a;
    border-radius: 10px;
    padding: 0.75rem;
    text-align: center;
}

.metric-number {
    color: #54aaff;
    font-size: 1.35rem;
    font-weight: 800;
}

.metric-label {
    color: #67829e;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

hr { border-color: #17283d !important; }

.stAlert {
    background: #0c1827 !important;
}


/* ---------------- Markdown / Tables ---------------- */
.stMarkdown,
.stMarkdown p,
.stMarkdown span,
.stMarkdown li,
.stMarkdown strong,
.stMarkdown em {
    color: #ffffff !important;
}

.stMarkdown table {
    color: #ffffff !important;
    border-color: #26384d !important;
}

.stMarkdown table th,
.stMarkdown table td {
    color: #ffffff !important;
    border-color: #26384d !important;
}

.stMarkdown table th {
    color: #ffffff !important;
    background: #0d1625 !important;
}

.stMarkdown table td {
    color: #ffffff !important;
    background: #0b1421 !important;
}

</style>
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌐 AI Travel Planner</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-muted">Session</div>', unsafe_allow_html=True)

    user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        label_visibility="collapsed",
        key="sidebar_user_id",
    )

    if user_id != st.session_state.user_id:
        st.session_state.user_id = user_id
        st.session_state.thread_id = new_thread_id(user_id)

    if st.button("＋ New conversation"):
        st.session_state.thread_id = new_thread_id(st.session_state.user_id)
        st.session_state.latest_result = None
        st.session_state.waiting_for_approval = False
        st.session_state.query_text = ""
        st.session_state.active_history_index = None
        st.rerun()

    st.markdown('<div class="sidebar-muted">LLM</div>', unsafe_allow_html=True)
    st.session_state.llm_provider = st.selectbox(
        "Select LLM",
        ["openai", "gemini", "groq"],
        index=["openai", "gemini", "groq"].index(st.session_state.llm_provider),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-muted">Powered by</div>', unsafe_allow_html=True)
    for item in [
        "🔗 LangGraph",
        "🧠 OpenAI / Gemini / Groq",
        "🐘 PostgreSQL",
        "🔎 Tavily Search",
        "✈️ AviationStack",
    ]:
        st.markdown(f'<div class="sidebar-tech">{item}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-muted">Recent trips</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div class="history-empty">Your generated travel requests will appear here.</div>',
            unsafe_allow_html=True,
        )
    else:
        for index, item in enumerate(reversed(st.session_state.history[-12:])):
            original_index = len(st.session_state.history) - 1 - index
            title = item["query"].strip().replace("\n", " ")
            if len(title) > 34:
                title = title[:34] + "…"

            if st.button(
                f"🗺️ {title}",
                key=f"history_{original_index}",
                help=item["query"],
            ):
                st.session_state.query_text = item["query"]
                st.session_state.latest_result = item.get("result")
                st.session_state.waiting_for_approval = item.get("waiting_for_approval", False)
                st.session_state.thread_id = item.get(
                    "thread_id",
                    st.session_state.thread_id,
                )
                st.session_state.active_history_index = original_index
                st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-inner">
        <div class="hero-badge">✦ Multi-Agent AI System</div>
        <div class="hero-title">✈️ AI Travel Booking System</div>
        <div class="hero-subtitle">
            Your AI travel team for flights, hotels, weather, budgets and complete trip planning —
            with each specialist agent running only when your request needs it.
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PREDEFINED PACKAGES
# ============================================================

st.markdown('<div class="section-label">✦ Explore ready-made trips</div>', unsafe_allow_html=True)

package_cols = st.columns(len(PACKAGES))
for col, package in zip(package_cols, PACKAGES):
    with col:
        st.markdown(
            f"""
            <div class="package-card" style="background-image:url('{package['image']}')">
                <div class="package-title">{package['short']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Buttons are intentionally separate from image cards for reliable Streamlit interaction.
quick_cols = st.columns(len(PACKAGES))
for col, package in zip(quick_cols, PACKAGES):
    with col:
        if st.button(
            package["name"],
            key=f"package_{package['name']}",
            use_container_width=True,
        ):
            st.session_state.query_text = package["query"]
            st.rerun()


# ============================================================
# QUERY INPUT
# ============================================================

st.markdown('<div class="section-label">🗺️ Describe your request</div>', unsafe_allow_html=True)

query = st.text_area(
    "Travel request",
    placeholder=(
        "e.g. Plan a 7-day Japan trip including flights, hotels and sightseeing under INR 2 lakh"
    ),
    height=95,
    label_visibility="collapsed",
    key="query_text",
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if st.button("🚀 Generate My Travel Plan", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a travel request first.")
    else:
        st.session_state.latest_result = None
        st.session_state.waiting_for_approval = False

        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        input_state = {
            "messages": [HumanMessage(content=query)],
            "user_id": st.session_state.user_id,
            "user_query": query,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
        }

        with st.status("🤖 Running the travel agents...", expanded=True) as status:
            try:
                result = app.invoke(input_state, config=config)
                status.update(label="✅ Request completed", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="❌ Request failed", state="error", expanded=True)
                st.exception(exc)
                result = None

        if result is not None:
            waiting = "__interrupt__" in result
            st.session_state.latest_result = result
            st.session_state.waiting_for_approval = waiting

            # Avoid duplicate history entries if Streamlit reruns the page.
            st.session_state.history.append(
                {
                    "query": query,
                    "result": result,
                    "thread_id": st.session_state.thread_id,
                    "waiting_for_approval": waiting,
                    "created_at": datetime.now().strftime("%d %b %Y, %H:%M"),
                }
            )

            st.rerun()


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.get("latest_result")

if result:
    selected_agents = result.get("selected_agents", [])

    st.markdown("---")
    st.markdown('<div class="section-label">🤖 Agent pipeline</div>', unsafe_allow_html=True)

    if selected_agents:
        pills = "".join(
            f'<span class="agent-pill">{agent.replace("_", " ").title()}</span>'
            for agent in selected_agents
        )
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.caption("No specialist agents selected.")

    reasoning = result.get("supervisor_reasoning", "")
    if reasoning:
        with st.expander("Supervisor decision", expanded=False):
            st.write(reasoning)

    # Show only outputs that actually exist / were selected.
    output_sections = [
        ("flight_agent", "✈️ Flight Information", result.get("flight_results", "")),
        ("hotel_agent", "🏨 Hotel Information", result.get("hotel_results", "")),
        ("weather_agent", "🌤️ Weather Information", result.get("weather_results", "")),
        ("budget_agent", "💰 Budget Analysis", result.get("budget_results", "")),
    ]

    visible_sections = [
        (agent, title, text)
        for agent, title, text in output_sections
        if agent in selected_agents and text
    ]

    if visible_sections:
        cols = st.columns(min(2, len(visible_sections)))
        for index, (_, title, text) in enumerate(visible_sections):
            with cols[index % len(cols)]:
                st.markdown(
                    f'<div class="result-card"><div class="result-title">{title}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(text)
                st.markdown("</div>", unsafe_allow_html=True)

    # Itinerary only exists for explicit planning requests.
    draft = result.get("itinerary", "")
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"]
        if interrupt_data:
            try:
                draft = interrupt_data[0].value.get("draft_itinerary", draft)
            except AttributeError:
                pass

    if draft:
        st.markdown('<div class="section-label">🗓️ Draft itinerary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="final-card">{draft}</div>', unsafe_allow_html=True)

    # Small metrics row.
    agents_run = len(selected_agents)
    llm_calls = result.get("llm_calls", 0)
    metrics = st.columns(3)
    with metrics[0]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-number">{agents_run}</div>'
            '<div class="metric-label">Agents selected</div></div>',
            unsafe_allow_html=True,
        )
    with metrics[1]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-number">{llm_calls}</div>'
            '<div class="metric-label">LLM calls</div></div>',
            unsafe_allow_html=True,
        )
    with metrics[2]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-number">{"⏳" if st.session_state.waiting_for_approval else "✓"}</div>'
            '<div class="metric-label">Status</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# HUMAN APPROVAL
# ============================================================

if st.session_state.get("waiting_for_approval"):
    st.markdown("---")
    st.markdown('<div class="section-label">👤 Human approval</div>', unsafe_allow_html=True)

    approved = st.radio(
        "Approve this draft?",
        ["Yes", "No, revise it"],
        horizontal=True,
        key="approval_choice",
    )

    feedback = st.text_area(
        "Feedback",
        placeholder="Optional feedback for the itinerary agent...",
        disabled=approved == "Yes",
        key="approval_feedback",
    )

    if st.button("Submit approval", type="primary", use_container_width=True):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        with st.spinner("Preparing the final response..."):
            final_result = app.invoke(
                Command(
                    resume={
                        "approved": approved == "Yes",
                        "feedback": feedback,
                    }
                ),
                config=config,
            )

        st.session_state.latest_result = final_result
        st.session_state.waiting_for_approval = False

        # Update the current history item with the final response.
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                if item.get("thread_id") == st.session_state.thread_id:
                    item["result"] = final_result
                    item["waiting_for_approval"] = False
                    break

        st.rerun()


# ============================================================
# FINAL RESPONSE
# ============================================================

final_result = st.session_state.get("latest_result")
if final_result and final_result.get("final_response"):
    st.markdown("---")
    st.markdown('<div class="section-label">✨ Final travel plan</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="final-card">{final_result["final_response"]}</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇️ Download response",
        data=final_result["final_response"],
        file_name="travel_plan.txt",
        mime="text/plain",
        use_container_width=True,
    )
