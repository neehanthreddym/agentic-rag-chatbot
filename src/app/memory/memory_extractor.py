"""
Memory extractor — uses the LLM to identify high-signal facts
from a conversation turn that are worth persisting.

The LLM returns a structured JSON decision:
    {should_save, user_facts, company_facts, confidence}
"""
import json
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from src.app.generation.prompts import MEMORY_EXTRACTION_PROMPT
from src.app.utils import get_llm, log_token_usage
from src.app.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryDecision:
    """Result of memory extraction for a single conversation turn."""
    should_save: bool = False
    user_facts: list[str] = field(default_factory=list)
    company_facts: list[str] = field(default_factory=list)
    confidence: float = 0.0


def extract_memory(
    user_message: str,
    assistant_response: str,
) -> MemoryDecision:
    """
    Analyze a conversation turn and decide which facts to persist.

    Invokes the LLM with MEMORY_EXTRACTION_PROMPT, parses the JSON
    response, and returns a typed MemoryDecision.

    Args:
        user_message: The user's message in this turn.
        assistant_response: The assistant's reply.

    Returns:
        MemoryDecision with extracted facts and confidence score.
        On any failure (JSON parse error, LLM timeout), returns an
        empty decision rather than crashing.
    """
    llm = get_llm()

    prompt = MEMORY_EXTRACTION_PROMPT.format(
        user_message=user_message,
        assistant_response=assistant_response,
    )

    messages = [
        SystemMessage(content="You are a memory curator. Respond ONLY with valid JSON."),
        HumanMessage(content=prompt),
    ]

    try:
        logger.info("🧠 Extracting memory from conversation turn...")
        response = llm.invoke(messages)
        log_token_usage(response)
        raw = response.content.strip()

        # Strip markdown code fences if the LLM wraps its JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]  # remove first line (```json)
            raw = raw.rsplit("```", 1)[0]  # remove closing ```
            raw = raw.strip()

        data = json.loads(raw)

        decision = MemoryDecision(
            should_save=data.get("should_save", False),
            user_facts=data.get("user_facts", []),
            company_facts=data.get("company_facts", []),
            confidence=float(data.get("confidence", 0.0)),
        )

        logger.info(
            f"🧠 Memory decision: save={decision.should_save}, "
            f"confidence={decision.confidence:.2f}, "
            f"user_facts={len(decision.user_facts)}, "
            f"company_facts={len(decision.company_facts)}"
        )
        return decision

    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ Failed to parse memory JSON: {e}")
        return MemoryDecision()

    except Exception as e:
        logger.warning(f"⚠️ Memory extraction failed: {e}")
        return MemoryDecision()
