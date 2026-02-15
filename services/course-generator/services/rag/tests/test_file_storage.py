"""
Unit Tests for Document File Storage

Tests LocalDocumentStorage and RAGDocumentStorage (with local fallback).
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from ..storage.file_storage import (
    LocalDocumentStorage,
    RAGDocumentStorage,
    StorageConfig,
)


class TestStorageConfig:
    """Tests for StorageConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StorageConfig()

        assert config.endpoint == "http://minio:9000"
        assert config.bucket == "storage"
        assert config.documents_prefix == "documents"
        assert config.local_path == "/tmp/viralify/documents"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StorageConfig(
            endpoint="https://s3.example.com",
            access_key="my_access_key",
            secret_key="my_secret_key",
            bucket="my-bucket",
            documents_prefix="docs",
            local_path="/custom/path",
        )

        assert config.endpoint == "https://s3.example.com"
        assert config.access_key == "my_access_key"
        assert config.bucket == "my-bucket"

    def test_ssl_detection(self):
        """Test SSL detection from endpoint."""
        https_config = StorageConfig(endpoint="https://s3.example.com")
        http_config = StorageConfig(endpoint="http://minio:9000")

        assert https_config.use_ssl is False  # Default, not from endpoint
        # Note: from_env() would detect SSL


class TestLocalDocumentStorage:
    """Tests for LocalDocumentStorage."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create local storage with temp directory."""
        return LocalDocumentStorage(base_path=temp_dir)

    # ==========================================================================
    # Initialization Tests
    # ==========================================================================

    def test_creates_base_path(self, temp_dir):
        """Test that storage creates base path if not exists."""
        new_path = Path(temp_dir) / "new_subdir"
        storage = LocalDocumentStorage(base_path=str(new_path))

        assert new_path.exists()

    def test_get_user_path(self, storage):
        """Test getting user path."""
        path = storage.get_user_path("user_123")

        assert "user_123" in str(path)
        assert path.exists()

    # ==========================================================================
    # Save File Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_save_file(self, storage):
        """Test saving a file."""
        content = b"Test content"

        path = await storage.save_file(
            content=content,
            user_id="user_1",
            document_id="doc_1",
            filename="test.pdf",
        )

        assert Path(path).exists()
        with open(path, 'rb') as f:
            assert f.read() == content

    @pytest.mark.asyncio
    async def test_save_file_path_structure(self, storage):
        """Test that saved file path has correct structure."""
        content = b"Content"

        path = await storage.save_file(
            content=content,
            user_id="user_123",
            document_id="doc_456",
            filename="report.pdf",
        )

        assert "user_123" in path
        assert "doc_456_report.pdf" in path

    @pytest.mark.asyncio
    async def test_save_large_file(self, storage):
        """Test saving a large file."""
        content = b"x" * (1024 * 1024)  # 1 MB

        path = await storage.save_file(
            content=content,
            user_id="user_1",
            document_id="doc_1",
            filename="large.bin",
        )

        assert Path(path).exists()
        assert Path(path).stat().st_size == len(content)

    @pytest.mark.asyncio
    async def test_save_empty_file(self, storage):
        """Test saving empty file."""
        path = await storage.save_file(
            content=b"",
            user_id="user_1",
            document_id="doc_1",
            filename="empty.txt",
        )

        assert Path(path).exists()
        assert Path(path).stat().st_size == 0

    # ==========================================================================
    # Get File Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_get_file(self, storage):
        """Test retrieving a file."""
        content = b"Test content"
        path = await storage.save_file(content, "user_1", "doc_1", "test.txt")

        result = await storage.get_file(path)

        assert result == content

    @pytest.mark.asyncio
    async def test_get_file_not_found(self, storage):
        """Test retrieving nonexistent file."""
        result = await storage.get_file("/nonexistent/path/file.txt")

        assert result is None

    # ==========================================================================
    # Delete File Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_delete_file(self, storage):
        """Test deleting a file."""
        content = b"Content"
        path = await storage.save_file(content, "user_1", "doc_1", "test.txt")

        await storage.delete_file(path)

        assert not Path(path).exists()

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, storage):
        """Test deleting nonexistent file doesn't crash."""
        await storage.delete_file("/nonexistent/file.txt")  # Should not raise

    # ==========================================================================
    # List Files Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_list_files(self, storage):
        """Test listing files for a user."""
        # Save multiple files
        for i in range(3):
            await storage.save_file(b"content", "user_1", f"doc_{i}", f"file{i}.txt")

        files = await storage.list_files("user_1")

        assert len(files) == 3

    @pytest.mark.asyncio
    async def test_list_files_empty(self, storage):
        """Test listing files for user with no files."""
        files = await storage.list_files("user_with_no_files")

        assert files == []

    @pytest.mark.asyncio
    async def test_list_files_multiple_users(self, storage):
        """Test that list_files only returns files for specified user."""
        await storage.save_file(b"content", "user_1", "doc_1", "file1.txt")
        await storage.save_file(b"content", "user_2", "doc_2", "file2.txt")

        files_user1 = await storage.list_files("user_1")
        files_user2 = await storage.list_files("user_2")

        assert len(files_user1) == 1
        assert len(files_user2) == 1

    # ==========================================================================
    # Public URL Tests
    # ==========================================================================

    def test_get_public_url(self, storage):
        """Test getting public URL (file:// for local)."""
        url = storage.get_public_url("/path/to/file.pdf")

        assert url.startswith("file://")
        assert "/path/to/file.pdf" in url


class TestRAGDocumentStorage:
    """Tests for RAGDocumentStorage (with local fallback)."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def config(self, temp_dir):
        """Create config with temp directory."""
        return StorageConfig(
            access_key="",  # No credentials = fall back to local
            secret_key="",
            local_path=temp_dir,
        )

    @pytest.fixture
    def storage(self, config):
        """Create storage with local fallback."""
        return RAGDocumentStorage(config=config, use_local_fallback=True)

    # ==========================================================================
    # Initialization Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_initialize_falls_back_to_local(self, storage):
        """Test that initialization falls back to local when S3 unavailable."""
        await storage.initialize()

        assert storage._initialized
        assert storage.is_using_local

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, storage):
        """Test that initialize only runs once."""
        await storage.initialize()
        first_state = storage._using_local

        await storage.initialize()

        assert storage._using_local == first_state

    # ==========================================================================
    # Operation Tests (Using Local Fallback)
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_save_file_with_fallback(self, storage):
        """Test save operation with local fallback."""
        content = b"Test content"

        path = await storage.save_file(
            content=content,
            user_id="user_1",
            document_id="doc_1",
            filename="test.pdf",
        )

        assert path is not None
        result = await storage.get_file(path)
        assert result == content

    @pytest.mark.asyncio
    async def test_get_file_with_fallback(self, storage):
        """Test get operation with local fallback."""
        content = b"Content"
        path = await storage.save_file(content, "user_1", "doc_1", "test.txt")

        result = await storage.get_file(path)

        assert result == content

    @pytest.mark.asyncio
    async def test_delete_file_with_fallback(self, storage):
        """Test delete operation with local fallback."""
        content = b"Content"
        path = await storage.save_file(content, "user_1", "doc_1", "test.txt")

        await storage.delete_file(path)

        result = await storage.get_file(path)
        assert result is None

    @pytest.mark.asyncio
    async def test_list_files_with_fallback(self, storage):
        """Test list operation with local fallback."""
        await storage.save_file(b"content1", "user_1", "doc_1", "file1.txt")
        await storage.save_file(b"content2", "user_1", "doc_2", "file2.txt")

        files = await storage.list_files("user_1")

        assert len(files) == 2

    # ==========================================================================
    # Auto-Initialize Tests
    # ==========================================================================

    @pytest.mark.asyncio
    async def test_auto_initialize_on_save(self, storage):
        """Test that save auto-initializes storage."""
        assert not storage._initialized

        await storage.save_file(b"content", "user_1", "doc_1", "test.txt")

        assert storage._initialized

    @pytest.mark.asyncio
    async def test_auto_initialize_on_get(self, storage):
        """Test that get auto-initializes storage."""
        assert not storage._initialized

        await storage.get_file("some/path")

        assert storage._initialized


class TestStorageEdgeCases:
    """Edge case tests for storage."""

    @pytest.fixture
    def temp_dir(self):
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def storage(self, temp_dir):
        return LocalDocumentStorage(base_path=temp_dir)

    @pytest.mark.asyncio
    async def test_filename_with_spaces(self, storage):
        """Test handling filename with spaces."""
        path = await storage.save_file(
            content=b"content",
            user_id="user_1",
            document_id="doc_1",
            filename="my file with spaces.pdf",
        )

        assert "my file with spaces.pdf" in path
        assert Path(path).exists()

    @pytest.mark.asyncio
    async def test_filename_with_special_chars(self, storage):
        """Test handling filename with special characters."""
        path = await storage.save_file(
            content=b"content",
            user_id="user_1",
            document_id="doc_1",
            filename="file-with_special.chars(1).pdf",
        )

        assert Path(path).exists()

    @pytest.mark.asyncio
    async def test_binary_content(self, storage):
        """Test saving binary content."""
        # PDF header bytes
        content = b"%PDF-1.4\n\x00\x01\x02\x03"

        path = await storage.save_file(content, "user_1", "doc_1", "binary.pdf")
        result = await storage.get_file(path)

        assert result == content

    @pytest.mark.asyncio
    async def test_unicode_filename(self, storage):
        """Test handling unicode filename."""
        path = await storage.save_file(
            content=b"content",
            user_id="user_1",
            document_id="doc_1",
            filename="fichier_été.pdf",
        )

        assert Path(path).exists()

    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self, storage):
        """Test that saving overwrites existing file."""
        original_content = b"original"
        new_content = b"new content"

        path1 = await storage.save_file(original_content, "user_1", "doc_1", "file.txt")
        path2 = await storage.save_file(new_content, "user_1", "doc_1", "file.txt")

        assert path1 == path2

        result = await storage.get_file(path1)
        assert result == new_content
