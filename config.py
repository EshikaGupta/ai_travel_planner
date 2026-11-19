import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

_ACTIVE_PROVIDER = None

def set_provider(provider: str):
    global _ACTIVE_PROVIDER
    _ACTIVE_PROVIDER = provider

def get_active_provider():
    return _ACTIVE_PROVIDER or os.getenv("LLM_PROVIDER")

def get_llm(provider):
    """Create the configured chat model.

    Set LLM_PROVIDER to gemini, groq, or openai.
    If it is not set, the first provider with an API key is used.
    """

    if not provider:
        if os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.getenv("GROQ_API_KEY"):
            provider = "groq"
        # elif os.getenv("OPENAI_API_KEY"):
        #     provider = "openai"
        else:
            raise RuntimeError(
                "Set one of GEMINI_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY."
            )

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL") or "gemini-3.8-flash",
            google_api_key=key,
            temperature=temperature,
        )

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b",
            groq_api_key=key,
            temperature=temperature,
            max_tokens=int(os.getenv("GROQ_MAX_OUTPUT_TOKENS", "1024"))
        )

    # if provider == "openai":
    #     key = os.getenv("OPENAI_API_KEY")
    #     if not key:
    #         raise RuntimeError("OPENAI_API_KEY is not set.")
    #     return ChatOpenAI(
    #         model=os.getenv("OPENAI_MODEL") or "gpt-5-mini",
    #         api_key=key,
    #         temperature=temperature,
    #     )

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use gemini, groq, or openai."
    )
