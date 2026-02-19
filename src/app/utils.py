import time
from functools import wraps

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from src.app.config import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_PROVIDER, GROQ_MODEL,
    GEMINI_PRICING, GROQ_PRICING
)
from src.app.logger import get_logger

logger = get_logger(__name__)


def log_token_usage(response):
    """
    Log token usage and estimated cost for an LLM response.
    
    Args:
        response: The LangChain response object (AIMessage).
    """
    usage = response.response_metadata.get("token_usage", {})
    if not usage:
        # Fallback for some providers if structure differs
        usage = response.usage_metadata or {}

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    if LLM_PROVIDER == "groq":
        pricing = GROQ_PRICING
        model_name = GROQ_MODEL
    else:
        pricing = GEMINI_PRICING
        model_name = LLM_MODEL

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    logger.info(
        f"💰 Usage [{model_name}]: "
        f"\nIn: {input_tokens} | Out: {output_tokens} | Total: {total_tokens} "
        f"| Cost: ${total_cost:.6f}"
    )


def timer(base_function):
    """Decorator to measure and log function execution time."""
    @wraps(base_function)
    def enhanced_function(*args, **kwargs):
        start_time = time.time()
        result = base_function(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{base_function.__name__} completed in {elapsed:.2f}s")
        return result
    return enhanced_function


def get_llm():
    """
    Initialize and return the configured LLM.

    Uses LLM_PROVIDER config to select between Gemini and Groq.

    Returns:
        LangChain chat model instance.
    """
    if LLM_PROVIDER == "groq":
        logger.info(f"Using Groq LLM: {GROQ_MODEL}")
        return ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)
    else:
        logger.info(f"Using Gemini LLM: {LLM_MODEL}")
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=LLM_TEMPERATURE
        )