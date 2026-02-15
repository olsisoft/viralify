"""
Unit Tests for RAG Service

Tests the main RAGService orchestrator.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch


# Mock enums and dataclasses for testing
class MockDocumentStatus(str, Enum):
    PENDING = "pending"
    SCANNING = "scanning"
    PARSING = "parsing"
    VECTORIZING = "vectorizing"
    READY = "ready"
    FAILED = "failed"
    SCAN_FAILED = "scan_failed"


class MockDocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    URL = "url"


@dataclass
class MockDocument:
    """Mock Document for testing."""
    id: str
    user_id: str
    filename: str
    document_type: MockDocumentType = MockDocumentType.PDF
    course_id: Optional[str] = None
    status: MockDocumentStatus = MockDocumentStatus.PENDING
    error_message: Optional[str] = None
    raw_content: str = ""
    file_path: str = ""
    file_size_bytes: int = 0
    page_count: int = 0
    word_count: int = 0
    created_at: datetime = None
    processed_at: datetime = None
    security_scan: object = None
    extracted_metadata: dict = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class MockSecurityScanResult:
    """Mock security scan result."""
    is_safe: bool = True
    threats_found: List[str] = None

    def __post_init__(self):
        if self.threats_found is None:
            self.threats_found = []


# Import the service (with mocked dependencies)
from ..services.rag_service import RAGService


class TestRAGService:
    """Tests for RAGService class."""

    @pytest.fixture
    def service(self):
        """Create RAG service without initialization."""
        return RAGService(
            vector_backend="memory",
            storage_path="/tmp/test",
            reranker_backend="tfidf",
        )

    # ==========================================================================
    # Initialization Tests
    # ==========================================================================

    def test_create_service(self, service):
        """Test creating RAG service."""
        assert service.vector_backend == "memory"
        assert service.storage_path == "/tmp/test"
        assert not service._initialized

    def test_default_configuration(self):
        """Test default configuration values."""
        service = RAGService()

        assert service.MAX_CONTEXT_TOKENS == 8000
        assert service.MAX_PROMPT_TOKENS == 100000
        assert service.RERANK_TOP_K == 30

    # ==========================================================================
    # Token Counting Tests
    # ==========================================================================

    def test_count_tokens_without_tokenizer(self, service):
        """Test token counting without tiktoken."""
        service._tokenizer = None

        count = service.count_tokens("Hello world this is a test")

        # Rough estimate: len / 4
        assert count > 0

    def test_count_tokens_empty(self, service):
        """Test counting tokens in empty string."""
        service._tokenizer = None

        assert service.count_tokens("") == 0
        assert service.count_tokens(None) == 0

    def test_count_tokens_with_mock_tokenizer(self, service):
        """Test token counting with mock tokenizer."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(10))
        service._tokenizer = mock_tokenizer

        count = service.count_tokens("Test text")

        assert count == 10
        mock_tokenizer.encode.assert_called_once_with("Test text")

    # ==========================================================================
    # Get Document Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_get_document(self, service):
        """Test getting a document."""
        mock_doc = MockDocument(
            id="doc_123",
            user_id="user_1",
            filename="test.pdf",
        )

        # Mock repository
        service._initialized = True
        service._repository = AsyncMock()
        service._repository.get.return_value = mock_doc

        result = await service.get_document("doc_123")

        assert result is mock_doc
        service._repository.get.assert_called_once_with("doc_123")

    @pytest.mark.asyncio
    async def test_get_document_auto_initialize(self, service):
        """Test that get_document auto-initializes."""
        service._initialized = False

        with patch.object(service, 'initialize', new_callable=AsyncMock) as mock_init:
            service._repository = AsyncMock()
            service._repository.get.return_value = None

            await service.get_document("doc_123")

            mock_init.assert_called_once()

    # ==========================================================================
    # Get Documents By User Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_get_documents_by_user(self, service):
        """Test getting documents by user."""
        mock_docs = [
            MockDocument(id="doc_1", user_id="user_1", filename="file1.pdf"),
            MockDocument(id="doc_2", user_id="user_1", filename="file2.pdf"),
        ]

        service._initialized = True
        service._repository = AsyncMock()
        service._repository.get_by_user.return_value = mock_docs

        result = await service.get_documents_by_user("user_1")

        assert len(result) == 2
        service._repository.get_by_user.assert_called_once_with("user_1")

    # ==========================================================================
    # Delete Document Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_delete_document(self, service):
        """Test deleting a document."""
        mock_doc = MockDocument(
            id="doc_123",
            user_id="user_1",
            filename="test.pdf",
            file_path="/path/to/file.pdf",
        )

        service._initialized = True
        service._repository = AsyncMock()
        service._repository.get.return_value = mock_doc
        service._storage = AsyncMock()
        service._vectorization = AsyncMock()

        result = await service.delete_document("doc_123", "user_1")

        assert result is True
        service._vectorization.delete_document.assert_called_once_with("doc_123")
        service._storage.delete_file.assert_called_once()
        service._repository.delete.assert_called_once_with("doc_123")

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, service):
        """Test deleting nonexistent document."""
        service._initialized = True
        service._repository = AsyncMock()
        service._repository.get.return_value = None

        result = await service.delete_document("nonexistent", "user_1")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_document_wrong_user(self, service):
        """Test deleting document owned by different user."""
        mock_doc = MockDocument(
            id="doc_123",
            user_id="user_1",  # Different user
            filename="test.pdf",
        )

        service._initialized = True
        service._repository = AsyncMock()
        service._repository.get.return_value = mock_doc

        result = await service.delete_document("doc_123", "user_2")

        assert result is False

    # ==========================================================================
    # Last Weighted Result Tests
    # ==========================================================================

    def test_get_last_weighted_result(self, service):
        """Test getting last weighted result."""
        from ..models.scoring import WeightedRAGResult

        mock_result = WeightedRAGResult(
            combined_context="test",
            document_scores=[],
        )
        service._last_weighted_result = mock_result

        result = service.get_last_weighted_result()

        assert result is mock_result

    def test_get_last_weighted_result_none(self, service):
        """Test getting last weighted result when none."""
        assert service.get_last_weighted_result() is None

    # ==========================================================================
    # Close Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_close(self, service):
        """Test closing the service."""
        service._repository = AsyncMock()

        await service.close()

        service._repository.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_repository(self, service):
        """Test closing when repository not initialized."""
        service._repository = None

        await service.close()  # Should not raise


class TestRAGServiceIntegration:
    """Integration-style tests for RAGService."""

    @pytest.fixture
    def service_with_mocks(self):
        """Create service with mocked dependencies."""
        service = RAGService()
        service._initialized = True

        # Mock all dependencies
        service._repository = AsyncMock()
        service._storage = AsyncMock()
        service._security_scanner = MagicMock()
        service._document_parser = AsyncMock()
        service._vectorization = AsyncMock()
        service._weighted_rag = MagicMock()
        service._structure_extractor = MagicMock()
        service._ai_structure_generator = AsyncMock()
        service._context_builder = MagicMock()
        service._tokenizer = None

        return service

    @pytest.mark.asyncio
    async def test_get_context_no_documents(self, service_with_mocks):
        """Test getting context with no documents."""
        service = service_with_mocks
        service._repository.get.return_value = None

        result = await service.get_context_for_course_generation(
            topic="Test Topic",
            description="Test description",
            document_ids=["doc_1"],
            user_id="user_1",
        )

        # Should return empty or structure only
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_get_context_single_document(self, service_with_mocks):
        """Test getting context with single document."""
        service = service_with_mocks

        mock_doc = MockDocument(
            id="doc_1",
            user_id="user_1",
            filename="test.pdf",
            raw_content="This is the document content about Apache Kafka.",
            status=MockDocumentStatus.READY,
        )

        service._repository.get.return_value = mock_doc
        service._structure_extractor.extract.return_value = MagicMock(headings=[])

        result = await service.get_context_for_course_generation(
            topic="Apache Kafka",
            description=None,
            document_ids=["doc_1"],
            user_id="user_1",
            use_weighted=False,  # Single doc, no weighting
        )

        assert "Apache Kafka" in result or "document content" in result.lower()


class TestRAGServiceEdgeCases:
    """Edge case tests for RAGService."""

    @pytest.fixture
    def service(self):
        return RAGService()

    def test_reranker_backend_options(self):
        """Test different reranker backend options."""
        for backend in ["auto", "cross-encoder", "tfidf"]:
            service = RAGService(reranker_backend=backend)
            assert service.reranker_backend == backend

    def test_vector_backend_options(self):
        """Test different vector backend options."""
        for backend in ["memory", "chroma", "pgvector"]:
            service = RAGService(vector_backend=backend)
            assert service.vector_backend == backend

    @pytest.mark.asyncio
    async def test_multiple_initializations(self):
        """Test that multiple initialize calls are safe."""
        service = RAGService()
        service._initialized = True

        # Second call should be a no-op
        service._repository = AsyncMock()
        await service.initialize()

        # Repository should not be reinitialized
        assert service._initialized


class TestRAGServiceConfiguration:
    """Tests for RAGService configuration."""

    def test_max_tokens_configuration(self):
        """Test token limit configuration."""
        service = RAGService()

        assert service.MAX_CONTEXT_TOKENS == 8000
        assert service.MAX_PROMPT_TOKENS == 100000

    def test_rerank_configuration(self):
        """Test re-ranking configuration."""
        service = RAGService()

        assert service.RERANK_TOP_K == 30
        assert service.RERANK_FINAL_K == 15
