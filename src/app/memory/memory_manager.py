"""
Memory manager — orchestrates memory extraction and persistence.

This is the single public entry point for the memory subsystem.
Other modules call `process_memory()` after each conversation turn.
"""
from src.app.config import (
    USER_MEMORY_PATH,
    COMPANY_MEMORY_PATH,
    MEMORY_CONFIDENCE_THRESHOLD,
)
from src.app.memory.memory_extractor import extract_memory
from src.app.memory.memory_writer import append_facts
from src.app.logger import get_logger

logger = get_logger(__name__)


def process_memory(
    user_message: str,
    assistant_response: str,
    confidence_threshold: float = MEMORY_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Full memory pipeline for one conversation turn.

    Steps:
        1. Call extract_memory() — LLM decides what facts are worth saving
        2. Check if confidence >= threshold (default 0.7)
        3. If yes, append facts to the appropriate markdown files
        4. Return a summary of what was written

    Args:
        user_message: The user's message.
        assistant_response: The assistant's reply.
        confidence_threshold: Minimum confidence to trigger a write.

    Returns:
        Dict summarizing the memory operation:
        {
            "memory_saved": bool,
            "user_facts_written": int,
            "company_facts_written": int,
            "confidence": float,
        }
    """
    # Step 1: Extract facts via LLM
    decision = extract_memory(user_message, assistant_response)

    result = {
        "memory_saved": False,
        "user_facts_written": 0,
        "company_facts_written": 0,
        "confidence": decision.confidence,
    }

    # Step 2: Confidence gate
    if not decision.should_save or decision.confidence < confidence_threshold:
        logger.info(
            f"🧠 Memory skipped — should_save={decision.should_save}, "
            f"confidence={decision.confidence:.2f} "
            f"(threshold={confidence_threshold})"
        )
        return result

    # Step 3: Write user facts
    if decision.user_facts:
        user_written = append_facts(USER_MEMORY_PATH, decision.user_facts)
        result["user_facts_written"] = user_written

    # Step 4: Write company facts
    if decision.company_facts:
        company_written = append_facts(COMPANY_MEMORY_PATH, decision.company_facts)
        result["company_facts_written"] = company_written

    result["memory_saved"] = (
        result["user_facts_written"] > 0 or result["company_facts_written"] > 0
    )

    logger.info(
        f"🧠 Memory result: saved={result['memory_saved']}, "
        f"user={result['user_facts_written']}, "
        f"company={result['company_facts_written']}"
    )
    return result
