"""
Memory writer — file I/O layer for persistent markdown memory files.

Reads existing memory files and appends new facts with deduplication
to avoid storing the same knowledge twice.
"""
import os
from datetime import datetime

from src.app.logger import get_logger

logger = get_logger(__name__)


def read_memory(file_path: str) -> str:
    """
    Read the contents of a memory markdown file.

    Args:
        file_path: Path to the memory file.

    Returns:
        File contents as a string, or empty string if file doesn't exist.
    """
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _is_duplicate(fact: str, existing_content: str) -> bool:
    """
    Check if a fact already exists in the memory file.

    Uses case-insensitive substring matching to catch near-duplicates.

    Args:
        fact: The fact to check.
        existing_content: Current contents of the memory file.

    Returns:
        True if the fact is already present.
    """
    return fact.strip().lower() in existing_content.lower()


def append_facts(
    file_path: str,
    facts: list[str],
    header: str = "",
) -> int:
    """
    Append high-signal facts to a memory markdown file.

    - Reads existing content to deduplicate
    - Skips facts that already appear in the file
    - Writes new facts as markdown bullet points under a timestamped header
    - Creates the file if it doesn't exist

    Args:
        file_path: Path to the memory file (e.g. USER_MEMORY.md).
        facts: List of fact strings to append.
        header: Optional custom header; defaults to timestamped session header.

    Returns:
        Number of new facts actually written (after deduplication).
    """
    if not facts:
        return 0

    existing = read_memory(file_path)

    # Filter out duplicates
    new_facts = [f for f in facts if f.strip() and not _is_duplicate(f, existing)]

    if not new_facts:
        logger.info(f"📝 No new facts to write to {file_path}")
        return 0

    # Build the entry
    if not header:
        header = f"## Session — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    lines = ["\n", header, ""]
    for fact in new_facts:
        lines.append(f"- {fact}")
    lines.append("")  # trailing newline

    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"📝 Wrote {len(new_facts)} fact(s) to {file_path}")
    return len(new_facts)
