"""
Grounded answer generator with citation support.

Uses Gemini or Groq LLM to generate answers grounded in retrieved
context, with structured citations. Supports three modes:
- rag: Document search with citations
- memory: Answer from stored memory
- general: Conversational without context
"""
import re
import os

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.app.generation.prompts import (
    RAG_SYSTEM_PROMPT,
    MEMORY_ANSWER_PROMPT,
    GENERAL_ANSWER_PROMPT,
)
from src.app.retrieval.retriever import format_context
from src.app.utils import timer, get_llm, log_token_usage
from src.app.logger import get_logger
from src.app.config import USER_MEMORY_PATH, COMPANY_MEMORY_PATH

logger = get_logger(__name__)


def _read_memory_file(path: str) -> str:
    """
    Read memory from a file, stripping HTML comments.
    
    Args:
        path: Path to the memory file (USER_MEMORY_PATH or COMPANY_MEMORY_PATH).
    
    Returns:
        The memory content, or empty string if file doesn't exist.
    """
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r") as f:
            content = f.read()
        # Strip multi-line HTML comments
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)
        return content.strip()
    except Exception as e:
        logger.warning(f"Failed to read memory file {path}: {e}")
        return ""


def _format_memory_context() -> str:
    """
    Read and format stored user and company memory.
    
    Returns:
        Concatenated memory context string, or message if no memory exists.
    """
    user_mem = _read_memory_file(USER_MEMORY_PATH)
    company_mem = _read_memory_file(COMPANY_MEMORY_PATH)
    
    parts = []
    if user_mem:
        parts.append(f"USER PROFILE:\n{user_mem}")
    if company_mem:
        parts.append(f"COMPANY/ORGANIZATION:\n{company_mem}")
    
    if not parts:
        return "No stored memory yet."
    
    return "\n\n".join(parts)


def _extract_citations(answer_text: str, docs: list[Document]) -> list[dict]:
    """
    Extract citation references from the answer text.

    Looks for [Source: filename, Chunk N] patterns and maps them
    to the actual retrieved documents.

    Args:
        answer_text: The LLM's generated answer.
        docs: The retrieved documents used as context.

    Returns:
        List of citation dicts with source, chunk_id, and snippet.
    """
    pattern = r"\[Source:\s*(.+?),\s*Chunk\s*(\d+)\]"
    matches = re.findall(pattern, answer_text)

    citations = []
    seen = set()

    for source, chunk_id_str in matches:
        key = (source.strip(), int(chunk_id_str))
        if key in seen:
            continue
        seen.add(key)

        # Find the matching document
        snippet = ""
        for doc in docs:
            if (doc.metadata.get("source") == key[0] and
                    doc.metadata.get("chunk_id") == key[1]):
                snippet = doc.page_content[:200]
                break

        citations.append({
            "source": key[0],
            "chunk_id": key[1],
            "snippet": snippet,
        })

    return citations

@timer
def generate_rag_answer(query: str, context_docs: list[Document]) -> dict:
    """
    Generate a grounded answer with citations from retrieved documents.

    Args:
        query: The user's question.
        context_docs: Retrieved documents providing context.

    Returns:
        Dict with keys:
          - answer: The generated answer text.
          - citations: List of citation dicts.
          - sources_used: List of unique source filenames.
    """
    llm = get_llm()

    # Format the context with citation markers
    context = format_context(context_docs)

    # Build the system prompt with context injected
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    logger.info(f"🤖 Generating RAG answer for: {query}")
    response = llm.invoke(messages)
    log_token_usage(response)
    answer_text = response.content

    # Extract structured citations
    citations = _extract_citations(answer_text, context_docs)
    sources_used = list({c["source"] for c in citations})

    # Strip inline citation markers so the displayed answer is clean
    clean_answer = re.sub(r"\s*\[Source:\s*.+?,\s*Chunk\s*\d+\]", "", answer_text)
    clean_answer = clean_answer.strip()

    logger.info(
        f"✅ RAG answer generated with {len(citations)} citation(s) "
        f"from {len(sources_used)} source(s)"
    )

    return {
        "answer": clean_answer,
        "citations": citations,
        "sources_used": sources_used,
    }


@timer
def generate_memory_answer(query: str) -> dict:
    """
    Generate an answer using stored user and company memory.

    Args:
        query: The user's question.

    Returns:
        Dict with keys:
          - answer: The generated answer text.
          - citations: Empty list (no documents cited).
          - sources_used: Empty list.
    """
    llm = get_llm()

    # Retrieve stored memory
    memory_context = _format_memory_context()
    system_prompt = MEMORY_ANSWER_PROMPT.format(memory_context=memory_context)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    logger.info(f"🧠 Generating memory-based answer for: {query}")
    response = llm.invoke(messages)
    log_token_usage(response)
    answer_text = response.content.strip()

    logger.info("✅ Memory answer generated")

    return {
        "answer": answer_text,
        "citations": [],
        "sources_used": [],
    }


@timer
def generate_general_answer(query: str) -> dict:
    """
    Generate a conversational answer without document or memory context.

    Args:
        query: The user's question or message.

    Returns:
        Dict with keys:
          - answer: The generated answer text.
          - citations: Empty list.
          - sources_used: Empty list.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=GENERAL_ANSWER_PROMPT),
        HumanMessage(content=query),
    ]

    logger.info(f"💬 Generating general answer for: {query}")
    response = llm.invoke(messages)
    log_token_usage(response)
    answer_text = response.content.strip()

    logger.info("✅ General answer generated")

    return {
        "answer": answer_text,
        "citations": [],
        "sources_used": [],
    }


@timer
def generate_answer(query: str, context_docs: list[Document], mode: str = "rag") -> dict:
    """
    Unified answer generator supporting multiple modes.

    Dispatches to mode-specific generators:
    - "rag": Generate from retrieved documents with citations
    - "memory": Generate from stored memory
    - "general": Generate conversational response

    Args:
        query: The user's question.
        context_docs: Retrieved documents (used only in "rag" mode).
        mode: Generation mode - "rag", "memory", or "general". Default: "rag".

    Returns:
        Dict with keys:
          - answer: The generated answer text.
          - citations: List of citations (only for rag mode).
          - sources_used: List of source filenames (only for rag mode).
    """
    mode = mode.lower().strip()

    if mode == "rag":
        return generate_rag_answer(query, context_docs)
    elif mode == "memory":
        return generate_memory_answer(query)
    elif mode == "general":
        return generate_general_answer(query)
    else:
        logger.warning(f"Unknown mode: {mode} — defaulting to 'general'")
        return generate_general_answer(query)
