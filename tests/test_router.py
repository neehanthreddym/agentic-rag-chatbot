"""
Unit tests for the agentic query router.

Tests verify that queries are correctly classified into routes:
- document_search: Queries about document content
- memory_lookup: Queries about stored memory (user/company facts)
- general: Conversational or general knowledge questions

Run with:  python -m pytest tests/test_router.py -v
"""
import pytest
from unittest.mock import patch, MagicMock


# =====================================================================
# Router Classification Tests
# =====================================================================

class TestRouterClassification:
    """Test route_query() with various query types."""

    @patch("src.app.routing.router.get_llm")
    def test_route_document_search(self, mock_get_llm):
        """Verify document_search route is detected."""
        from src.app.routing.router import route_query

        # Mock LLM response for document search query
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "document_search"
        mock_get_llm.return_value = mock_llm

        query = "What are the main findings in the paper about transformer architectures?"
        route = route_query(query, has_vectorstore=True)

        assert route == "document_search"
        mock_llm.invoke.assert_called_once()

    @patch("src.app.routing.router.get_llm")
    def test_route_memory_lookup(self, mock_get_llm):
        """Verify memory_lookup route is detected."""
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "memory_lookup"
        mock_get_llm.return_value = mock_llm

        query = "What's my role at the company?"
        route = route_query(query, has_vectorstore=True)

        assert route == "memory_lookup"

    @patch("src.app.routing.router.get_llm")
    def test_route_general(self, mock_get_llm):
        """Verify general route is detected."""
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "general"
        mock_get_llm.return_value = mock_llm

        query = "What's the weather like today?"
        route = route_query(query, has_vectorstore=True)

        assert route == "general"

    @patch("src.app.routing.router.get_llm")
    def test_document_search_downgrade_no_vectorstore(self, mock_get_llm):
        """
        Verify document_search is downgraded to general when no vectorstore.
        """
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        # LLM would suggest document_search
        mock_llm.invoke.return_value.content = "document_search"
        mock_get_llm.return_value = mock_llm

        query = "What does the paper say about this?"
        route = route_query(query, has_vectorstore=False)

        # Should be downgraded to general since no documents loaded
        assert route == "general"

    @patch("src.app.routing.router.get_llm")
    def test_router_handles_whitespace(self, mock_get_llm):
        """Verify router handles responses with whitespace."""
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        # Response with leading/trailing whitespace
        mock_llm.invoke.return_value.content = "  memory_lookup  \n"
        mock_get_llm.return_value = mock_llm

        query = "Tell me about myself"
        route = route_query(query, has_vectorstore=True)

        assert route == "memory_lookup"

    @patch("src.app.routing.router.get_llm")
    def test_router_case_insensitive(self, mock_get_llm):
        """Verify router is case-insensitive."""
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "DOCUMENT_SEARCH"
        mock_get_llm.return_value = mock_llm

        query = "What is in the uploaded document?"
        route = route_query(query, has_vectorstore=True)

        assert route == "document_search"

    @patch("src.app.routing.router.get_llm")
    def test_router_fallback_on_invalid_response(self, mock_get_llm):
        """Verify router defaults to general on unparseable response."""
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        # Invalid/unparseable response
        mock_llm.invoke.return_value.content = "invalid_route_name"
        mock_get_llm.return_value = mock_llm

        query = "Some query"
        route = route_query(query, has_vectorstore=True)

        # Should default to general
        assert route == "general"

    @patch("src.app.routing.router.get_llm")
    def test_router_fallback_on_exception(self, mock_get_llm):
        """Verify router defaults to general when LLM call fails."""
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        # Simulate LLM failure
        mock_llm.invoke.side_effect = Exception("LLM error")
        mock_get_llm.return_value = mock_llm

        query = "Some query"
        route = route_query(query, has_vectorstore=True)

        # Should default to general
        assert route == "general"


# =====================================================================
# Integration Tests
# =====================================================================

class TestRouterIntegration:
    """Test router with generator modes."""

    @patch("src.app.routing.router.get_llm")
    @patch("src.app.generation.generator.get_llm")
    def test_router_rag_mode_integration(self, mock_gen_llm, mock_router_llm):
        """Verify router correctly routes to RAG mode."""
        from src.app.routing.router import route_query
        from src.app.generation.generator import generate_answer
        from langchain_core.documents import Document

        # Router identifies document_search
        mock_router_llm.return_value.invoke.return_value.content = "document_search"

        # Generator processes RAG
        mock_gen_llm.return_value.invoke.return_value.content = "Based on the document..."

        query = "What is the main topic?"
        route = route_query(query, has_vectorstore=True)

        assert route == "document_search"

        # Simulate RAG generation
        doc = Document(
            page_content="Sample content",
            metadata={"source": "test.pdf", "chunk_id": 0}
        )
        result = generate_answer(query, [doc], mode="rag")

        assert result["answer"] is not None
        assert "sources_used" in result

    @patch("src.app.routing.router.get_llm")
    @patch("src.app.generation.generator.get_llm")
    def test_router_memory_mode_integration(self, mock_gen_llm, mock_router_llm):
        """Verify router correctly routes to memory mode."""
        from src.app.routing.router import route_query
        from src.app.generation.generator import generate_answer

        # Router identifies memory_lookup
        mock_router_llm.return_value.invoke.return_value.content = "memory_lookup"

        # Generator answers from memory
        mock_gen_llm.return_value.invoke.return_value.content = "You mentioned earlier..."

        query = "What is my role?"
        route = route_query(query, has_vectorstore=True)

        assert route == "memory_lookup"

        # Simulate memory-based generation
        result = generate_answer(query, [], mode="memory")

        assert result["answer"] is not None
        assert len(result["citations"]) == 0

    @patch("src.app.routing.router.get_llm")
    @patch("src.app.generation.generator.get_llm")
    def test_router_general_mode_integration(self, mock_gen_llm, mock_router_llm):
        """Verify router correctly routes to general mode."""
        from src.app.routing.router import route_query
        from src.app.generation.generator import generate_answer

        # Router identifies general
        mock_router_llm.return_value.invoke.return_value.content = "general"

        # Generator provides conversational response
        mock_gen_llm.return_value.invoke.return_value.content = "Hello! How can I help?"

        query = "Hi there!"
        route = route_query(query, has_vectorstore=False)

        assert route == "general"

        # Simulate general generation
        result = generate_answer(query, [], mode="general")

        assert result["answer"] is not None
        assert len(result["citations"]) == 0


# =====================================================================
# Edge Cases
# =====================================================================

class TestRouterEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("src.app.routing.router.get_llm")
    def test_empty_query(self, mock_get_llm):
        """Test router with empty query."""
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "general"
        mock_get_llm.return_value = mock_llm

        route = route_query("", has_vectorstore=True)
        assert route == "general"

    @patch("src.app.routing.router.get_llm")
    def test_very_long_query(self, mock_get_llm):
        """Test router with very long query."""
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "document_search"
        mock_get_llm.return_value = mock_llm

        long_query = "What is " + ("long " * 500) + "content?"
        route = route_query(long_query, has_vectorstore=True)
        assert route == "document_search"

    @patch("src.app.routing.router.get_llm")
    def test_special_characters_in_query(self, mock_get_llm):
        """Test router with special characters."""
        from src.app.routing.router import route_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "general"
        mock_get_llm.return_value = mock_llm

        query = "What's @#$% going on? <>&\"'"
        route = route_query(query, has_vectorstore=True)
        assert route in ["document_search", "memory_lookup", "general"]

