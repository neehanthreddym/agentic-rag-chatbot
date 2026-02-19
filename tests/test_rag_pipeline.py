"""
Unit tests for the RAG pipeline.

Tests cover config loading, content-type classification, context formatting,
citation extraction, and the full ingestion + retrieval + generation flow.

Run with:  python -m pytest tests/ -v
"""
import json
import os
import shutil
import tempfile

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


# =====================================================================
# Config tests
# =====================================================================

class TestConfig:
    """Verify all required config values are present and valid."""

    def test_api_key_loaded(self):
        from src.app.config import GOOGLE_API_KEY
        assert GOOGLE_API_KEY is not None, (
            "GOOGLE_API_KEY not set — create a .env file"
        )

    def test_llm_model_defined(self):
        from src.app.config import LLM_MODEL
        assert LLM_MODEL and isinstance(LLM_MODEL, str)

    def test_embedding_model_defined(self):
        from src.app.config import EMBEDDING_MODEL
        assert EMBEDDING_MODEL and isinstance(EMBEDDING_MODEL, str)

    def test_llm_provider_valid(self):
        from src.app.config import LLM_PROVIDER
        assert LLM_PROVIDER in ("gemini", "groq")

    def test_chunk_params_positive(self):
        from src.app.config import CHUNK_MAX_CHARACTERS, CHUNK_NEW_AFTER_N_CHARS, CHUNK_OVERLAP
        assert CHUNK_MAX_CHARACTERS > 0
        assert CHUNK_NEW_AFTER_N_CHARS > 0
        assert CHUNK_OVERLAP >= 0

    def test_top_k_positive(self):
        from src.app.config import TOP_K
        assert TOP_K > 0


# =====================================================================
# Logger tests
# =====================================================================

class TestLogger:
    """Verify the logger module works correctly."""

    def test_get_logger_returns_logger(self):
        from src.app.logger import get_logger
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"

    def test_get_logger_no_duplicate_handlers(self):
        from src.app.logger import get_logger
        logger1 = get_logger("dedup_test")
        handler_count = len(logger1.handlers)
        logger2 = get_logger("dedup_test")
        assert len(logger2.handlers) == handler_count

    def test_logger_outputs(self, capsys):
        from src.app.logger import get_logger
        logger = get_logger("output_test")
        logger.info("hello test")
        captured = capsys.readouterr()
        assert "hello test" in captured.out


# =====================================================================
# 3. Chunker — content type classification (unit, no API calls)
# =====================================================================

class TestSeparateContentTypes:
    """Test the separate_content_types function with mock elements."""

    def _make_mock_chunk(self, elements):
        """Create a mock chunk with orig_elements metadata."""
        chunk = MagicMock()
        chunk.metadata.orig_elements = elements
        return chunk

    def _make_element(self, class_name, text="sample", text_as_html=None, image_base64=None):
        el = MagicMock()
        type(el).__name__ = class_name
        el.__str__ = lambda self: text
        el.metadata.text_as_html = text_as_html
        el.metadata.image_base64 = image_base64
        return el

    def test_text_only_chunk(self):
        from src.app.ingestion.chunker import separate_content_types
        elements = [self._make_element("NarrativeText", text="Hello world")]
        chunk = self._make_mock_chunk(elements)
        result = separate_content_types(chunk)

        assert "text" in result["types"]
        assert result["tables_html"] == []
        assert result["images"] == []
        assert "Hello world" in result["text"]

    def test_table_chunk(self):
        from src.app.ingestion.chunker import separate_content_types
        elements = [
            self._make_element("Table", text_as_html="<table><tr><td>Content</td></tr></table>"),
        ]
        chunk = self._make_mock_chunk(elements)
        result = separate_content_types(chunk)

        assert "table" in result["types"]
        assert len(result["tables_html"]) == 1
        assert "<table>" in result["tables_html"][0]

    def test_image_chunk(self):
        from src.app.ingestion.chunker import separate_content_types
        elements = [
            self._make_element("Image", image_base64="base64data=="),
        ]
        chunk = self._make_mock_chunk(elements)
        result = separate_content_types(chunk)

        assert "image" in result["types"]
        assert len(result["images"]) == 1

    def test_mixed_chunk(self):
        from src.app.ingestion.chunker import separate_content_types
        elements = [
            self._make_element("NarrativeText", text="Some text"),
            self._make_element("Table", text_as_html="<table></table>"),
            self._make_element("Image", image_base64="img=="),
        ]
        chunk = self._make_mock_chunk(elements)
        result = separate_content_types(chunk)

        assert set(result["types"]) == {"image", "table", "text"}


# =====================================================================
# 4. Retriever — context formatting and citation markers (unit)
# =====================================================================

class TestFormatContext:
    """Test the format_context function."""

    def test_empty_docs(self):
        from src.app.retrieval.retriever import format_context
        result = format_context([])
        assert "No relevant documents found" in result

    def test_single_doc_formatting(self):
        from src.app.retrieval.retriever import format_context
        doc = Document(
            page_content="TinyLoRA reduces parameters.",
            metadata={"source": "TinyLoRA.pdf", "chunk_id": 3},
        )
        result = format_context([doc])

        assert "[Source: TinyLoRA.pdf, Chunk 3]" in result
        assert "TinyLoRA reduces parameters." in result
        assert "--- Document 1 ---" in result

    def test_multiple_docs(self):
        from src.app.retrieval.retriever import format_context
        docs = [
            Document(page_content="First.", metadata={"source": "a.pdf", "chunk_id": 1}),
            Document(page_content="Second.", metadata={"source": "b.pdf", "chunk_id": 2}),
        ]
        result = format_context(docs)

        assert "--- Document 1 ---" in result
        assert "--- Document 2 ---" in result
        assert "[Source: a.pdf, Chunk 1]" in result
        assert "[Source: b.pdf, Chunk 2]" in result

    def test_tables_included_when_present(self):
        from src.app.retrieval.retriever import format_context
        doc = Document(
            page_content="Table content.",
            metadata={
                "source": "paper.pdf",
                "chunk_id": 1,
                "has_tables": True,
                "original_content": json.dumps({
                    "raw_text": "text",
                    "tables_html": ["<table><tr><td>data</td></tr></table>"],
                    "images_base64": [],
                }),
            },
        )
        result = format_context([doc])
        assert "TABLES:" in result
        assert "<table>" in result


# =====================================================================
# 5. Generator — citation extraction (unit, no API calls)
# =====================================================================

class TestCitationExtraction:
    """Test the _extract_citations function."""

    def test_single_citation(self):
        from src.app.generation.generator import _extract_citations
        answer = "TinyLoRA reduces params [Source: TinyLoRA.pdf, Chunk 3]."
        docs = [
            Document(
                page_content="TinyLoRA is a technique...",
                metadata={"source": "TinyLoRA.pdf", "chunk_id": 3},
            ),
        ]
        citations = _extract_citations(answer, docs)

        assert len(citations) == 1
        assert citations[0]["source"] == "TinyLoRA.pdf"
        assert citations[0]["chunk_id"] == 3
        assert citations[0]["snippet"] != ""

    def test_multiple_citations(self):
        from src.app.generation.generator import _extract_citations
        answer = (
            "Claim A [Source: a.pdf, Chunk 1]. "
            "Claim B [Source: b.pdf, Chunk 5]."
        )
        docs = [
            Document(page_content="Content A", metadata={"source": "a.pdf", "chunk_id": 1}),
            Document(page_content="Content B", metadata={"source": "b.pdf", "chunk_id": 5}),
        ]
        citations = _extract_citations(answer, docs)
        assert len(citations) == 2

    def test_duplicate_citations_deduped(self):
        from src.app.generation.generator import _extract_citations
        answer = (
            "Fact [Source: a.pdf, Chunk 1]. "
            "Same fact [Source: a.pdf, Chunk 1]."
        )
        docs = [
            Document(page_content="Content", metadata={"source": "a.pdf", "chunk_id": 1}),
        ]
        citations = _extract_citations(answer, docs)
        assert len(citations) == 1

    def test_no_citations(self):
        from src.app.generation.generator import _extract_citations
        answer = "I don't have enough information to answer that."
        citations = _extract_citations(answer, [])
        assert len(citations) == 0


# =====================================================================
# 6. Parser — file validation (unit, no heavy PDF parsing)
# =====================================================================

class TestParser:
    """Test parser edge cases."""

    def test_missing_file_raises_error(self):
        from src.app.ingestion.parser import parse_pdf
        with pytest.raises(FileNotFoundError):
            parse_pdf("/nonexistent/path/to/file.pdf")

    def test_missing_directory_raises_error(self):
        from src.app.ingestion.parser import parse_directory
        with pytest.raises(NotADirectoryError):
            parse_directory("/nonexistent/directory")

    def test_empty_directory_returns_empty(self):
        from src.app.ingestion.parser import parse_directory
        with tempfile.TemporaryDirectory() as tmpdir:
            result = parse_directory(tmpdir)
            assert result == {}


# =====================================================================
# 7. Indexer — vector store lifecycle (unit with temp directory)
# =====================================================================

class TestIndexer:
    """Test indexer create/load operations."""

    def test_load_nonexistent_raises_error(self):
        from src.app.ingestion.indexer import load_vector_store
        with pytest.raises(FileNotFoundError):
            load_vector_store(persist_dir="/nonexistent/chroma_db")

    @pytest.mark.integration
    def test_create_and_load_vector_store(self):
        """Integration test: create a vector store and reload it."""
        from src.app.ingestion.indexer import create_vector_store, load_vector_store

        docs = [
            Document(
                page_content="Test document about machine learning.",
                metadata={"source": "test.pdf", "chunk_id": 1},
            ),
            Document(
                page_content="Another document about neural networks.",
                metadata={"source": "test.pdf", "chunk_id": 2},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = os.path.join(tmpdir, "chroma_db")

            # Create
            vs = create_vector_store(docs, persist_dir=persist_dir)
            assert vs is not None

            # Load
            vs_loaded = load_vector_store(persist_dir=persist_dir)
            count = vs_loaded._collection.count()
            assert count == 2

    @pytest.mark.integration
    def test_similarity_search_returns_results(self):
        """Integration test: verify similarity search works."""
        from src.app.ingestion.indexer import create_vector_store

        docs = [
            Document(
                page_content="TinyLoRA reduces the number of trainable parameters.",
                metadata={"source": "tiny.pdf", "chunk_id": 1},
            ),
            Document(
                page_content="Transformers use self-attention mechanisms.",
                metadata={"source": "transformer.pdf", "chunk_id": 1},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_dir = os.path.join(tmpdir, "chroma_db")
            vs = create_vector_store(docs, persist_dir=persist_dir)

            results = vs.similarity_search("parameter reduction", k=1)
            assert len(results) == 1
            assert "TinyLoRA" in results[0].page_content


# =====================================================================
# 8. Generator — generate_answer with mocked LLM (unit)
# =====================================================================

class TestGenerateAnswer:
    """Test generate_answer with a mocked LLM to avoid API calls."""

    @patch("src.app.generation.generator.get_llm")
    def test_generate_answer_structure(self, mock_get_llm):
        from src.app.generation.generator import generate_answer

        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="TinyLoRA reduces parameters to as few as 1 [Source: tiny.pdf, Chunk 1]."
        )
        mock_get_llm.return_value = mock_llm

        docs = [
            Document(
                page_content="TinyLoRA is a technique for reducing trainable params.",
                metadata={"source": "tiny.pdf", "chunk_id": 1},
            ),
        ]

        result = generate_answer("What is TinyLoRA?", docs)

        assert "answer" in result
        assert "citations" in result
        assert "sources_used" in result
        assert isinstance(result["citations"], list)
        assert len(result["citations"]) == 1
        assert result["citations"][0]["source"] == "tiny.pdf"

    @patch("src.app.generation.generator.get_llm")
    def test_generate_answer_no_info(self, mock_get_llm):
        from src.app.generation.generator import generate_answer

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="I don't have enough information in the uploaded documents to answer that question."
        )
        mock_get_llm.return_value = mock_llm

        result = generate_answer("What is quantum computing?", [])

        assert result["citations"] == []
        assert result["sources_used"] == []


# =====================================================================
# 9. Memory Extractor — LLM-based fact extraction (unit, mocked LLM)
# =====================================================================

class TestMemoryExtractor:
    """Test memory extraction with a mocked LLM to avoid API calls."""

    @patch("src.app.memory.memory_extractor.get_llm")
    def test_extract_high_signal_facts(self, mock_get_llm):
        from src.app.memory.memory_extractor import extract_memory

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "should_save": True,
                "user_facts": ["User prefers weekly summaries on Mondays"],
                "company_facts": ["Asset Management interfaces with Project Finance"],
                "confidence": 0.9,
            })
        )
        mock_get_llm.return_value = mock_llm

        decision = extract_memory(
            "I prefer weekly summaries on Mondays.",
            "Noted! I'll remember that preference.",
        )

        assert decision.should_save is True
        assert len(decision.user_facts) == 1
        assert len(decision.company_facts) == 1
        assert decision.confidence == 0.9

    @patch("src.app.memory.memory_extractor.get_llm")
    def test_extract_no_signal(self, mock_get_llm):
        from src.app.memory.memory_extractor import extract_memory

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "should_save": False,
                "user_facts": [],
                "company_facts": [],
                "confidence": 0.0,
            })
        )
        mock_get_llm.return_value = mock_llm

        decision = extract_memory("Hello!", "Hi there!")

        assert decision.should_save is False
        assert decision.user_facts == []
        assert decision.company_facts == []
        assert decision.confidence == 0.0

    @patch("src.app.memory.memory_extractor.get_llm")
    def test_malformed_json_graceful(self, mock_get_llm):
        from src.app.memory.memory_extractor import extract_memory

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="This is not valid JSON at all!"
        )
        mock_get_llm.return_value = mock_llm

        decision = extract_memory("test", "test")

        # Should return empty decision, not crash
        assert decision.should_save is False
        assert decision.user_facts == []
        assert decision.confidence == 0.0

    @patch("src.app.memory.memory_extractor.get_llm")
    def test_markdown_fenced_json(self, mock_get_llm):
        """Test that JSON wrapped in markdown code fences is handled."""
        from src.app.memory.memory_extractor import extract_memory

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='```json\n{"should_save": true, "user_facts": ["Role: analyst"], "company_facts": [], "confidence": 0.8}\n```'
        )
        mock_get_llm.return_value = mock_llm

        decision = extract_memory("I'm an analyst.", "Got it!")

        assert decision.should_save is True
        assert len(decision.user_facts) == 1
        assert decision.confidence == 0.8


# =====================================================================
# 10. Memory Writer — file I/O with deduplication (unit, temp files)
# =====================================================================

class TestMemoryWriter:
    """Test memory writer operations using temp directories."""

    def test_append_facts_creates_content(self):
        from src.app.memory.memory_writer import append_facts, read_memory

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "TEST_MEMORY.md")

            # Write initial content
            with open(path, "w") as f:
                f.write("# TEST MEMORY\n")

            written = append_facts(path, ["User is a data scientist"])
            assert written == 1

            content = read_memory(path)
            assert "User is a data scientist" in content

    def test_append_facts_deduplicates(self):
        from src.app.memory.memory_writer import append_facts, read_memory

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "TEST_MEMORY.md")

            with open(path, "w") as f:
                f.write("# TEST MEMORY\n")

            # Write same fact twice
            append_facts(path, ["User prefers Python"])
            written2 = append_facts(path, ["User prefers Python"])

            assert written2 == 0  # second write should be deduped

            content = read_memory(path)
            assert content.count("User prefers Python") == 1

    def test_read_memory_nonexistent_file(self):
        from src.app.memory.memory_writer import read_memory

        content = read_memory("/nonexistent/path/MEMORY.md")
        assert content == ""

    def test_append_empty_facts(self):
        from src.app.memory.memory_writer import append_facts

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "TEST_MEMORY.md")
            written = append_facts(path, [])
            assert written == 0


# =====================================================================
# 11. Memory Manager — orchestration (unit, mocked extractor)
# =====================================================================

class TestMemoryManager:
    """Test the memory manager orchestration logic."""

    @patch("src.app.memory.memory_manager.append_facts")
    @patch("src.app.memory.memory_manager.extract_memory")
    def test_process_memory_high_confidence(self, mock_extract, mock_append):
        from src.app.memory.memory_manager import process_memory
        from src.app.memory.memory_extractor import MemoryDecision

        mock_extract.return_value = MemoryDecision(
            should_save=True,
            user_facts=["Prefers morning meetings"],
            company_facts=["Standard meeting time is 9 AM"],
            confidence=0.9,
        )
        mock_append.return_value = 1

        result = process_memory("I prefer morning meetings.", "Noted!")

        assert result["memory_saved"] is True
        assert mock_append.call_count == 2  # once for user, once for company

    @patch("src.app.memory.memory_manager.append_facts")
    @patch("src.app.memory.memory_manager.extract_memory")
    def test_process_memory_low_confidence(self, mock_extract, mock_append):
        from src.app.memory.memory_manager import process_memory
        from src.app.memory.memory_extractor import MemoryDecision

        mock_extract.return_value = MemoryDecision(
            should_save=True,
            user_facts=["Something vague"],
            company_facts=[],
            confidence=0.3,  # below threshold
        )

        result = process_memory("Maybe something.", "Okay.")

        assert result["memory_saved"] is False
        mock_append.assert_not_called()

    @patch("src.app.memory.memory_manager.append_facts")
    @patch("src.app.memory.memory_manager.extract_memory")
    def test_process_memory_should_save_false(self, mock_extract, mock_append):
        from src.app.memory.memory_manager import process_memory
        from src.app.memory.memory_extractor import MemoryDecision

        mock_extract.return_value = MemoryDecision(
            should_save=False,
            user_facts=[],
            company_facts=[],
            confidence=0.0,
        )

        result = process_memory("Hello!", "Hi!")

        assert result["memory_saved"] is False
        mock_append.assert_not_called()
