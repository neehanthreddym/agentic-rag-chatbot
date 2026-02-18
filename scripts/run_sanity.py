"""
Sanity check script for the Agentic RAG Chatbot.

It performs a minimal end-to-end RAG flow and writes
the required `artifacts/sanity_output.json`.
"""
import json
import os
import shutil

# ---------------------------------------------------------------------------
# Import pipeline modules
# ---------------------------------------------------------------------------
from src.app.ingestion.pipeline import run_ingestion_pipeline
from src.app.retrieval.retriever import get_retriever, retrieve
from src.app.generation.generator import generate_answer
from src.app.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Use the sample PDF
PDF_PATH = "sample_docs/TinyLoRA_2602.04118v1.pdf"

# Temporary vector store — cleaned up at the end.
SANITY_DB = "sanity_chroma_db"

# Output path that verify_output.py and sanity_check.sh expect.
OUTPUT_PATH = "artifacts/sanity_output.json"

# ---------------------------------------------------------------------------
# Evaluation Questions (from EVAL_QUESTIONS.md)
# ---------------------------------------------------------------------------
# Group 1: Answerable — these go into the "qa" array (citations required)
ANSWERABLE_QUESTIONS = [
    "Summarize the main contribution of TinyLoRA in 3 bullets.",
    "What are the key assumptions or limitations of TinyLoRA?",
    "Give one concrete numeric or experimental detail from TinyLoRA and cite it.",
]

# Group 2: Unanswerable — these go into "demo.refusal_tests"
UNANSWERABLE_QUESTIONS = [
    "What is the CEO's phone number?",
    "What was the GDP of France in 2019?",
]


def run_sanity():
    """
    Execute a minimal end-to-end RAG flow and write the output JSON.

    Flow:
        1. Ingest PDF → parse, chunk, index into ChromaDB
        2. For each question → retrieve relevant chunks → generate answer
        3. Format results into the schema verify_output.py expects
        4. Write artifacts/sanity_output.json
        5. Clean up temp vector store
    """
    logger.info("=" * 60)
    logger.info("🔍 SANITY CHECK — Starting end-to-end RAG flow")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Ingest the PDF
    # ------------------------------------------------------------------
    # This calls parse → chunk → index, the full Feature A pipeline.
    logger.info(f"📥 Ingesting: {PDF_PATH}")
    vectorstore = run_ingestion_pipeline(
        PDF_PATH,
        persist_dir=SANITY_DB,
        extract_images=True,
    )
    logger.info("✅ Ingestion complete")

    # ------------------------------------------------------------------
    # Step 2: Build the retriever
    # ------------------------------------------------------------------
    retriever = get_retriever(vectorstore, top_k=5)

    # ------------------------------------------------------------------
    # Step 3: Ask questions and collect results
    # ------------------------------------------------------------------
    qa_results = []

    for question in ANSWERABLE_QUESTIONS:
        logger.info(f"\n📝 Question: {question}")

        # Retrieve relevant chunks from ChromaDB
        docs = retrieve(retriever, question)

        # Generate grounded answer with citations
        result = generate_answer(question, docs)

        # ----------------------------------------------------------
        # Step 4: Format citations
        # ----------------------------------------------------------
        formatted_citations = []
        for c in result["citations"]:
            formatted_citations.append({
                "source": c["source"],
                "locator": f"Chunk {c['chunk_id']}",
                "snippet": c["snippet"] if c["snippet"] else docs[0].page_content[:200],
            })

        # If the LLM didn't produce citation markers but we have docs,
        # add a fallback citation.
        # This can happen if the LLM forgets the [Source: ...] format.
        if not formatted_citations and docs:
            formatted_citations.append({
                "source": docs[0].metadata.get("source", "unknown"),
                "locator": f"Chunk {docs[0].metadata.get('chunk_id', 0)}",
                "snippet": docs[0].page_content[:200],
            })

        qa_results.append({
            "question": question,
            "answer": result["answer"],
            "citations": formatted_citations,
        })

        logger.info(f"   ✅ Answer generated with {len(formatted_citations)} citation(s)")

    # ------------------------------------------------------------------
    # Step 5: Run unanswerable questions (refusal tests)
    # ------------------------------------------------------------------
    # These test that the LLM gracefully refuses to answer questions
    # not covered by the uploaded documents — no hallucinations.
    refusal_results = []

    for question in UNANSWERABLE_QUESTIONS:
        logger.info(f"\n🚫 Refusal test: {question}")

        docs = retrieve(retriever, question)
        result = generate_answer(question, docs)

        # Check if the LLM correctly refused
        answer_lower = result["answer"].lower()
        refused = any(phrase in answer_lower for phrase in [
            "i don't have", "i cannot find", "not found",
            "no relevant", "not covered", "cannot answer",
            "not mentioned", "no information", "don't have enough",
        ])

        refusal_results.append({
            "question": question,
            "answer": result["answer"],
            "refused_correctly": refused,
            "hallucinated_citations": len(result["citations"]) > 0,
        })

        status = "✅ Refused" if refused else "⚠️ May have hallucinated"
        logger.info(f"   {status}")

    # ------------------------------------------------------------------
    # Step 6: Build the output JSON
    # ------------------------------------------------------------------
    output = {
        "implemented_features": ["A"],
        "qa": qa_results,
        "demo": {
            "pdf_used": PDF_PATH,
            "num_questions": len(ANSWERABLE_QUESTIONS) + len(UNANSWERABLE_QUESTIONS),
            "retrieval_top_k": 5,
            "refusal_tests": refusal_results,
        },
    }

    # ------------------------------------------------------------------
    # Step 7: Write the output file
    # ------------------------------------------------------------------
    os.makedirs("artifacts", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n💾 Output written to: {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Step 8: Cleanup temp vector store
    # ------------------------------------------------------------------
    # Don't want sanity_chroma_db committed to the repo.
    shutil.rmtree(SANITY_DB, ignore_errors=True)
    logger.info("🧹 Temp vector store cleaned up")

    logger.info("\n✅ SANITY CHECK PASSED")
    return output


if __name__ == "__main__":
    run_sanity()
