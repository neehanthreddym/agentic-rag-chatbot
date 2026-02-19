# Architecture Overview

## Goal
The goal of this Agentic RAG Chatbot is to provide a production-grade, file-grounded Q&A system that moves beyond basic API wrappers. It emphasizes **data grounding**, **long-term memory**, and **intelligent routing** to ensure responses are accurate, personalized, and secure.

---

## High-Level Flow

### 0) Agentic Query Router
The router acts as the system's "brain," classifying incoming queries to determine the optimal knowledge source before execution.

**Three Routes:**
1. **document_search** — Queries requiring specific knowledge from uploaded documents (e.g., arXiv PDFs).
2. **memory_lookup** — Queries about stored user preferences or organizational facts.
3. **general** — Conversational queries or general knowledge not requiring specific grounding.

**How it Works:**
- Uses a **Gemini 1.5 Flash** call to classify the intent.
- Includes a safety check: `document_search` is downgraded to `general` if the vector store is empty.

### 1) Ingestion (Upload → Parse → Chunk)
- **Supported Inputs:** PDF files (specifically optimized for technical arXiv papers).
- **Parsing:** Uses `PyMuPDF` or `unstructured` for high-fidelity text extraction.
- **Chunking:** Implements recursive character splitting to maintain context for complex technical layouts.
- **Metadata:** Captures source filename, page numbers, and chunk IDs for precise citations.

### 2) Indexing & Storage
- **Vector Store:** **ChromaDB** with persistent local storage to ensure reproducibility and local evaluation.
- **Embeddings:** **Google `text-embedding-004`** (or `text-embedding-005`) for high-density semantic representation.
- **Persistence:** Data is stored in the `chroma_db/` directory.

### 3) Retrieval + Grounded Answering
- **Retrieval:** Top-k similarity search (default k=5).
- **Citations:** The LLM is instructed to cite sources using the `[Source: filename, Page X]` format.
- **Verification:** Grounded answers are evaluated for "Faithfulness" to prevent hallucinations.
- **Failure Behavior:** The system gracefully admits when information is missing rather than fabricating answers.

### 4) Memory System (The "Durable" Layer)
This system captures high-signal facts to personalize future interactions without storing raw transcripts.

- **USER_MEMORY.md:** Stores user-specific preferences, expertise, and roles.
- **COMPANY_MEMORY.md:** Stores organizational policies, tech stack choices, and project decisions.
- **Memory Gating:** Facts are only saved if the extraction agent identifies "high-signal" information with a confidence score $\ge$ 0.7.
- **Security:** No PII, passwords, or API keys are ever written to memory files.

---

## Application Flow (UI Layer)

**In `app.py` (Streamlit Interface):**
1. **Upload:** User provides a PDF, triggering the ingestion and indexing pipeline.
2. **Input:** User submits a chat query.
3. **Route:** `route_query()` determines the appropriate source (Docs, Memory, or General).
4. **Generate:** `generate_answer()` produces a grounded response with citations if necessary.
5. **Memory Update:** `process_memory()` analyzes the exchange and updates `.md` memory files if new facts are learned.
6. **Display:** UI renders the answer, sources, and a "Memory Updated" indicator.

---

## Tradeoffs & Security Mindset
- **Why Local ChromaDB?** Ensures the `make sanity` command works out-of-the-box for judges without cloud configuration.
- **Why Markdown Memory?** Provides human-readable "Long-term Memory" that is easy to audit and persist across sessions.
- **Security:** Implements prompt-injection awareness and safe handling of external API calls to Gemini.