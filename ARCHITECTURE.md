# Architecture Overview

## Goal
Provide a brief, readable overview of how the Agentic RAG Chatbot works:
- ingestion
- indexing
- retrieval + grounding with citations
- memory writing

---

## High-Level Flow

### 0) Agentic Query Router
The router is the first step in the response pipeline. It classifies incoming queries to determine the optimal knowledge source.

**Three Routes:**
1. **document_search** — Query is about uploaded documents
   - Example: "What are the main findings of this paper?"
   - Response: Retrieves relevant chunks, generates answer with citations
   
2. **memory_lookup** — Query is about stored user/company information
   - Example: "What's my role?" or "What tech stack did we choose?"
   - Response: Generates answer grounded in stored memory
   
3. **general** — Query is conversational or general knowledge
   - Example: "Hi!" or "What's the weather?" or "Explain quantum computing"
   - Response: Generates conversational reply without document/memory grounding

**How it Works:**
- Router uses an LLM call to classify the query into one of the three routes
- Takes context: whether documents are currently loaded (has_vectorstore flag)
- **Safety: document_search is automatically downgraded to general if no vectorstore is loaded**
- Graceful fallback: if classification fails, defaults to "general" mode

**Implementation:**
- Location: `src/app/routing/router.py`
- Uses `ROUTER_PROMPT` from `src/app/generation/prompts.py`
- Returns route name: "document_search", "memory_lookup", or "general"

### 0.5) Answer Generation (Route-Specific)
Once the router determines the best route, the generator produces an appropriate response:

**RAG Mode (document_search route):**
- Retrieves top-k relevant chunks from the document vectorstore
- Uses `RAG_SYSTEM_PROMPT` with context injected
- LLM instructed to cite with `[Source: filename, Chunk N]` format
- Citations are extracted, deduplicated, and formatted for display
- Returns: answer + citation list + sources used

**Memory Mode (memory_lookup route):**
- Reads stored user and company memory from markdown files
- Uses `MEMORY_ANSWER_PROMPT` with memory context injected
- LLM generates conversational response grounded in stored facts
- Falls back gracefully: "I don't have that stored in my memory yet..."
- Returns: answer (citations are empty)

**General Mode (general route):**
- No context provided to the LLM
- Uses `GENERAL_ANSWER_PROMPT` for conversational tone
- LLM can provide general knowledge/small talk responses
- Returns: answer (citations are empty)

**Implementation:**
- Location: `src/app/generation/generator.py`
- Functions: `generate_rag_answer()`, `generate_memory_answer()`, `generate_general_answer()`
- Unified entry point: `generate_answer(query, context_docs, mode)`
- All functions are timed via `@timer` decorator for performance tracking

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

## Application Flow (UI Layer)

**In `app.py` (Streamlit interface):**

1. User uploads a PDF (optional) — triggers ingestion pipeline
2. User asks a question in the chat input
3. System flow:
   - **Route:** Call `route_query(prompt, has_vectorstore)` → determines "document_search" | "memory_lookup" | "general"
   - **Retrieve (if document_search):** Use retriever to fetch top-k chunks
   - **Generate:** Call `generate_answer(prompt, docs, mode)` → returns answer + citations
   - **Memory:** Call `process_memory(prompt, answer)` → extracts & saves high-signal facts
   - **Display:** Show answer + citations + memory indicator in UI

---

## Tradeoffs & Next Steps
- Why this design?
  - Modular: memory system is decoupled from RAG pipeline — can be enabled/disabled independently
  - LLM-as-judge: leverages the LLM's understanding to curate high-signal facts rather than rules-based extraction
  - Confidence gating: the threshold prevents low-quality facts from polluting memory
  - **Agentic routing:** A separate LLM classifier intelligently dispatches queries to the right mode (documents, memory, or conversational), rather than using heuristics
- What you would improve with more time:
  - Semantic deduplication (embeddings-based) instead of substring matching
  - Memory retrieval: feed stored memories back into the RAG prompt for personalized answers
  - Memory expiry/decay for stale facts
  - Multi-user support with separate memory files per user
  - **Router enhancement:** Use multi-turn context in routing decisions (e.g., "What does that mean?" → understand context from previous exchange)
  - **Router caching:** Cache routing decisions for follow-up questions to reduce LLM calls