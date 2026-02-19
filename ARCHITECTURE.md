# Architecture Overview

## Goal
Provide a brief, readable overview of how the Agentic RAG Chatbot works:
- ingestion
- indexing
- retrieval + grounding with citations
- memory writing

---

## High-Level Flow

### 1) Ingestion (Upload → Parse → Chunk)
- Supported inputs: PDF files (arXiv papers)
- Parsing approach: `unstructured` library with `hi_res` strategy for tables/images
- Chunking strategy: Section-aware chunking via `unstructured`'s `chunk_by_title` with overlap
- Metadata captured per chunk:
  - source filename
  - page/section (if available)
  - chunk_id
  - content type flags (has_tables, has_images)
  - AI-generated summary

### 2) Indexing / Storage
- Vector store: ChromaDB with persistent storage
- Embeddings: Google `gemini-embedding-001`
- Persistence: `chroma_db/` directory (configurable)

### 3) Retrieval + Grounded Answering
- Retrieval method: Top-k similarity search (default k=5)
- How citations are built:
  - citation includes: source filename, chunk locator (Chunk N), snippet
  - LLM instructed to use `[Source: filename, Chunk N]` format
- Failure behavior:
  - When retrieval is empty/low confidence, LLM responds with "I don't have enough information"
  - Refusal tests verify no hallucinated citations

### 4) Memory System (Selective)
- What counts as "high-signal" memory:
  - User preferences (e.g., "prefers weekly summaries on Mondays")
  - Stated expertise or role (e.g., "Project Finance Analyst")
  - Project-specific decisions (e.g., "chose PostgreSQL for the backend")
  - Organizational policies, standards, or workflow patterns
  - Repeated interests or domain topics
- What you explicitly do NOT store (PII/secrets/raw transcript):
  - No raw conversation transcripts — only distilled facts
  - No PII, passwords, API keys, or sensitive credentials
  - No generic greetings, small talk, or transient information
- How you decide when to write:
  - Each conversation turn is analyzed by the LLM using `MEMORY_EXTRACTION_PROMPT`
  - The LLM returns a structured decision: `{should_save, user_facts, company_facts, confidence}`
  - Facts are only saved when `should_save=true` AND `confidence >= 0.7` (configurable threshold)
  - Deduplication: case-insensitive substring matching prevents storing the same fact twice
- Format written to:
  - `USER_MEMORY.md` — timestamped markdown bullets for user-specific facts
  - `COMPANY_MEMORY.md` — timestamped markdown bullets for org-wide learnings

---

## Tradeoffs & Next Steps
- Why this design?
  - Modular: memory system is decoupled from RAG pipeline — can be enabled/disabled independently
  - LLM-as-judge: leverages the LLM's understanding to curate high-signal facts rather than rules-based extraction
  - Confidence gating: the threshold prevents low-quality facts from polluting memory
- What you would improve with more time:
  - Semantic deduplication (embeddings-based) instead of substring matching
  - Memory retrieval: feed stored memories back into the RAG prompt for personalized answers
  - Memory expiry/decay for stale facts
  - Multi-user support with separate memory files per user