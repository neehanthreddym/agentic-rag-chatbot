"""
Agentic query router — classifies user queries to determine
the optimal knowledge source for answering.

Routes:
  - document_search : use RAG pipeline (retrieve + cite)
  - memory_lookup   : answer from stored user/company memory
  - general         : conversational / general knowledge
"""
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.generation.prompts import ROUTER_PROMPT
from src.app.utils import get_llm, log_token_usage
from src.app.logger import get_logger

logger = get_logger(__name__)

VALID_ROUTES = {"document_search", "memory_lookup", "general"}


def route_query(query: str, has_vectorstore: bool) -> str:
    """
    Classify a user query into one of the supported routes.

    Uses a single LLM call to determine whether the query should be
    answered from uploaded documents, stored memory, or general knowledge.

    Args:
        query: The user's question.
        has_vectorstore: Whether a document vectorstore is currently loaded.

    Returns:
        One of "document_search", "memory_lookup", or "general".
    """
    llm = get_llm()

    prompt = ROUTER_PROMPT.format(
        query=query,
        has_documents="Yes" if has_vectorstore else "No",
    )

    messages = [
        SystemMessage(content="You are a query classifier. Respond with ONLY the route name."),
        HumanMessage(content=prompt),
    ]

    try:
        logger.info(f"🧭 Routing query: {query[:80]}...")
        response = llm.invoke(messages)
        log_token_usage(response)
        raw = response.content.strip().lower()

        # Extract the route from the response
        for route in VALID_ROUTES:
            if route in raw:
                # Fallback: if no vectorstore and LLM says document_search
                if route == "document_search" and not has_vectorstore:
                    logger.info("🧭 No documents loaded — downgrading document_search -> general")
                    return "general"
                logger.info(f"🧭 Route: {route}")
                return route

        # Default fallback
        logger.warning(f"⚠️ Could not parse route from: {raw!r} — defaulting to general")
        return "general"

    except Exception as e:
        logger.warning(f"⚠️ Router failed: {e} — defaulting to general")
        return "general"
