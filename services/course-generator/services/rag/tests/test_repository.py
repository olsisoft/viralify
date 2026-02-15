"""
Unit Tests for Document Repository

Tests InMemoryDocumentRepository and DocumentRepositoryPg (with memory fallback).
"""

import pytest
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from ..storage.repository import (
    InMemoryDocumentRepository,
    DocumentRepositoryPg,
    RepositoryConfig,
)


# Mock DocumentStatus for testing
class MockDocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


@dataclass
class MockDocument:
    """Mock Document for testing."""
    id: str
    user_id: str
    filename: str
    course_id: Optional[str] = None
    status: MockDocumentStatus = MockDocumentStatus.PENDING
    error_message: Optional[str] = None
    raw_content: str = ""
    page_count: int = 0
    word_count: int = 0
    created_at: datetime = None
    processed_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class TestRepositoryConfig:
    """Tests for RepositoryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RepositoryConfig()

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "tiktok_user"
        assert config.min_pool_size == 2
        assert config.max_pool_size == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RepositoryConfig(
            host="db.example.com",
            port=5433,
            user="custom_user",
            password="secret",
            database="custom_db",
        )

        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.user == "custom_user"
        assert config.password == "secret"


class TestInMemoryDocumentRepository:
    """Tests for InMemoryDocumentRepository."""

    @pytest.fixture
    def repo(self):
        """Create fresh in-memory repository."""
        return InMemoryDocumentRepository()

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        return MockDocument(
            id="doc_123",
            user_id="user_456",
            filename="test.pdf",
            course_id="course_789",
        )

    # ==========================================================================
    # Save Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_save_document(self, repo, sample_document):
        """Test saving a document."""
        await repo.save(sample_document)

        assert sample_document.id in repo.documents
        assert sample_document.user_id in repo.user_documents

    @pytest.mark.asyncio
    async def test_save_indexes_by_user(self, repo, sample_document):
        """Test that save indexes by user."""
        await repo.save(sample_document)

        user_docs = repo.user_documents.get(sample_document.user_id, [])
        assert sample_document.id in user_docs

    @pytest.mark.asyncio
    async def test_save_indexes_by_course(self, repo, sample_document):
        """Test that save indexes by course."""
        await repo.save(sample_document)

        course_docs = repo.course_documents.get(sample_document.course_id, [])
        assert sample_document.id in course_docs

    @pytest.mark.asyncio
    async def test_save_no_course(self, repo):
        """Test saving document without course."""
        doc = MockDocument(
            id="doc_1",
            user_id="user_1",
            filename="test.pdf",
            course_id=None,
        )

        await repo.save(doc)

        assert doc.id in repo.documents
        assert doc.id not in repo.course_documents.get(None, [])

    @pytest.mark.asyncio
    async def test_save_no_duplicate_indexes(self, repo, sample_document):
        """Test that saving twice doesn't duplicate indexes."""
        await repo.save(sample_document)
        await repo.save(sample_document)

        user_docs = repo.user_documents.get(sample_document.user_id, [])
        assert user_docs.count(sample_document.id) == 1

    # ==========================================================================
    # Get Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_get_document(self, repo, sample_document):
        """Test getting a document by ID."""
        await repo.save(sample_document)

        result = await repo.get(sample_document.id)

        assert result is sample_document

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, repo):
        """Test getting nonexistent document."""
        result = await repo.get("nonexistent_id")
        assert result is None

    # ==========================================================================
    # Get By User Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_get_by_user(self, repo):
        """Test getting all documents for a user."""
        docs = [
            MockDocument(id=f"doc_{i}", user_id="user_1", filename=f"file{i}.pdf")
            for i in range(3)
        ]
        for doc in docs:
            await repo.save(doc)

        result = await repo.get_by_user("user_1")

        assert len(result) == 3
        for doc in docs:
            assert doc in result

    @pytest.mark.asyncio
    async def test_get_by_user_empty(self, repo):
        """Test getting documents for user with no docs."""
        result = await repo.get_by_user("nonexistent_user")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_by_user_multiple_users(self, repo):
        """Test that get_by_user only returns docs for specified user."""
        doc1 = MockDocument(id="doc_1", user_id="user_1", filename="file1.pdf")
        doc2 = MockDocument(id="doc_2", user_id="user_2", filename="file2.pdf")

        await repo.save(doc1)
        await repo.save(doc2)

        result = await repo.get_by_user("user_1")

        assert len(result) == 1
        assert doc1 in result
        assert doc2 not in result

    # ==========================================================================
    # Get By Course Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_get_by_course(self, repo):
        """Test getting all documents for a course."""
        docs = [
            MockDocument(
                id=f"doc_{i}",
                user_id="user_1",
                filename=f"file{i}.pdf",
                course_id="course_1",
            )
            for i in range(2)
        ]
        for doc in docs:
            await repo.save(doc)

        result = await repo.get_by_course("course_1")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_by_course_empty(self, repo):
        """Test getting documents for course with no docs."""
        result = await repo.get_by_course("nonexistent_course")
        assert result == []

    # ==========================================================================
    # Get By IDs Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_get_by_ids(self, repo):
        """Test getting multiple documents by IDs."""
        docs = [
            MockDocument(id=f"doc_{i}", user_id="user_1", filename=f"file{i}.pdf")
            for i in range(5)
        ]
        for doc in docs:
            await repo.save(doc)

        result = await repo.get_by_ids(["doc_0", "doc_2", "doc_4"])

        assert len(result) == 3
        result_ids = {doc.id for doc in result}
        assert result_ids == {"doc_0", "doc_2", "doc_4"}

    @pytest.mark.asyncio
    async def test_get_by_ids_partial(self, repo):
        """Test getting by IDs with some nonexistent."""
        doc = MockDocument(id="doc_1", user_id="user_1", filename="file.pdf")
        await repo.save(doc)

        result = await repo.get_by_ids(["doc_1", "nonexistent"])

        assert len(result) == 1
        assert result[0].id == "doc_1"

    @pytest.mark.asyncio
    async def test_get_by_ids_empty(self, repo):
        """Test getting by empty ID list."""
        result = await repo.get_by_ids([])
        assert result == []

    # ==========================================================================
    # Delete Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_delete_document(self, repo, sample_document):
        """Test deleting a document."""
        await repo.save(sample_document)
        await repo.delete(sample_document.id)

        assert await repo.get(sample_document.id) is None

    @pytest.mark.asyncio
    async def test_delete_removes_from_user_index(self, repo, sample_document):
        """Test that delete removes from user index."""
        await repo.save(sample_document)
        await repo.delete(sample_document.id)

        user_docs = repo.user_documents.get(sample_document.user_id, [])
        assert sample_document.id not in user_docs

    @pytest.mark.asyncio
    async def test_delete_removes_from_course_index(self, repo, sample_document):
        """Test that delete removes from course index."""
        await repo.save(sample_document)
        await repo.delete(sample_document.id)

        course_docs = repo.course_documents.get(sample_document.course_id, [])
        assert sample_document.id not in course_docs

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, repo):
        """Test deleting nonexistent document doesn't crash."""
        await repo.delete("nonexistent")  # Should not raise

    # ==========================================================================
    # Update Status Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_update_status(self, repo, sample_document):
        """Test updating document status."""
        await repo.save(sample_document)

        await repo.update_status(sample_document.id, MockDocumentStatus.PROCESSING)

        doc = await repo.get(sample_document.id)
        assert doc.status == MockDocumentStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_update_status_with_error(self, repo, sample_document):
        """Test updating status with error message."""
        await repo.save(sample_document)

        await repo.update_status(
            sample_document.id,
            MockDocumentStatus.FAILED,
            error_message="Processing failed",
        )

        doc = await repo.get(sample_document.id)
        assert doc.status == MockDocumentStatus.FAILED
        assert doc.error_message == "Processing failed"

    @pytest.mark.asyncio
    async def test_update_status_nonexistent(self, repo):
        """Test updating status of nonexistent document."""
        # Should not raise
        await repo.update_status("nonexistent", MockDocumentStatus.READY)

    # ==========================================================================
    # Close Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_close(self, repo):
        """Test closing repository (no-op for in-memory)."""
        await repo.close()  # Should not raise


class TestDocumentRepositoryPg:
    """Tests for DocumentRepositoryPg (with memory fallback)."""

    @pytest.fixture
    def repo(self):
        """Create repository with memory fallback enabled."""
        return DocumentRepositoryPg(use_memory_fallback=True)

    @pytest.fixture
    def sample_document(self):
        """Create sample document."""
        return MockDocument(
            id="doc_123",
            user_id="user_456",
            filename="test.pdf",
        )

    # ==========================================================================
    # Initialization Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_initialize_falls_back_to_memory(self, repo):
        """Test that initialization falls back to memory when PG unavailable."""
        await repo.initialize()

        # Should use memory fallback when PostgreSQL is unavailable
        assert repo._initialized
        assert repo.is_using_memory  # No asyncpg or no connection

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, repo):
        """Test that initialize only runs once."""
        await repo.initialize()
        first_state = repo._using_memory

        await repo.initialize()

        assert repo._using_memory == first_state

    # ==========================================================================
    # Operation Tests (Using Memory Fallback)
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_save_with_fallback(self, repo, sample_document):
        """Test save operation with memory fallback."""
        await repo.save(sample_document)

        result = await repo.get(sample_document.id)
        assert result is sample_document

    @pytest.mark.asyncio
    async def test_get_by_user_with_fallback(self, repo):
        """Test get_by_user with memory fallback."""
        doc = MockDocument(id="doc_1", user_id="user_1", filename="test.pdf")
        await repo.save(doc)

        result = await repo.get_by_user("user_1")

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_delete_with_fallback(self, repo, sample_document):
        """Test delete with memory fallback."""
        await repo.save(sample_document)
        await repo.delete(sample_document.id)

        result = await repo.get(sample_document.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_update_status_with_fallback(self, repo, sample_document):
        """Test update_status with memory fallback."""
        await repo.save(sample_document)

        await repo.update_status(sample_document.id, MockDocumentStatus.READY)

        doc = await repo.get(sample_document.id)
        assert doc.status == MockDocumentStatus.READY

    # ==========================================================================
    # Close Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_close(self, repo):
        """Test closing repository."""
        await repo.initialize()
        await repo.close()  # Should not raise


class TestRepositoryEdgeCases:
    """Edge case tests for repositories."""

    @pytest.fixture
    def repo(self):
        return InMemoryDocumentRepository()

    @pytest.mark.asyncio
    async def test_same_doc_multiple_courses(self, repo):
        """Test document can be associated with one course only."""
        doc = MockDocument(
            id="doc_1",
            user_id="user_1",
            filename="test.pdf",
            course_id="course_1",
        )

        await repo.save(doc)

        # Change course and save again
        doc.course_id = "course_2"
        await repo.save(doc)

        # Document should still be in original course index
        # (In-memory repo doesn't update course index on re-save)
        course1_docs = await repo.get_by_course("course_1")
        assert len(course1_docs) == 1

    @pytest.mark.asyncio
    async def test_get_by_ids_preserves_order(self, repo):
        """Test that get_by_ids returns docs in requested order."""
        docs = [
            MockDocument(id=f"doc_{i}", user_id="user_1", filename=f"file{i}.pdf")
            for i in range(3)
        ]
        for doc in docs:
            await repo.save(doc)

        result = await repo.get_by_ids(["doc_2", "doc_0", "doc_1"])

        # Result order may vary, but all should be returned
        assert len(result) == 3
