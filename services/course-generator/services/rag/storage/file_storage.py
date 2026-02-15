"""
RAG Document Storage - MinIO/S3 with Local Fallback

Provides document file storage with:
- MinIO/S3 backend for production (using boto3)
- Local filesystem backend for development/testing
- Automatic fallback when object storage is unavailable
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass


# Lazy imports for optional dependencies
boto3 = None
ClientError = Exception


def _get_boto3():
    """Lazy load boto3 to avoid import errors when not installed."""
    global boto3, ClientError
    if boto3 is None:
        try:
            import boto3 as _boto3
            from botocore.exceptions import ClientError as _ClientError
            boto3 = _boto3
            ClientError = _ClientError
        except ImportError:
            boto3 = False
    return boto3 if boto3 else None


@dataclass
class StorageConfig:
    """Configuration for document storage."""
    # MinIO/S3 configuration
    endpoint: str = "http://minio:9000"
    access_key: str = ""
    secret_key: str = ""
    region: str = "us-east-1"
    use_ssl: bool = False
    public_url: str = ""

    # Bucket and prefix configuration
    bucket: str = "storage"
    documents_prefix: str = "documents"

    # Local fallback path
    local_path: str = "/tmp/viralify/documents"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Load configuration from environment variables."""
        endpoint = os.getenv("STORAGE_ENDPOINT", "http://minio:9000")
        use_ssl = endpoint.startswith("https://") or os.getenv("STORAGE_USE_SSL", "").lower() == "true"

        return cls(
            endpoint=endpoint,
            access_key=os.getenv("STORAGE_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "")),
            secret_key=os.getenv("STORAGE_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "")),
            region=os.getenv("STORAGE_REGION", "us-east-1"),
            use_ssl=use_ssl,
            public_url=os.getenv("STORAGE_PUBLIC_URL", ""),
            bucket=os.getenv("STORAGE_BUCKET", "storage"),
            documents_prefix=os.getenv("STORAGE_DOCUMENTS_PREFIX", "documents"),
            local_path=os.getenv("DOCUMENT_STORAGE_PATH", "/tmp/viralify/documents"),
        )


class LocalDocumentStorage:
    """
    Local filesystem document storage.

    Used for development/testing or as fallback when MinIO/S3 is unavailable.
    """

    def __init__(self, base_path: str = "/tmp/viralify/documents"):
        """
        Initialize local storage.

        Args:
            base_path: Root directory for document storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_user_path(self, user_id: str) -> Path:
        """Get storage path for user."""
        path = self.base_path / user_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_file(
        self,
        content: bytes,
        user_id: str,
        document_id: str,
        filename: str,
    ) -> str:
        """
        Save file to local storage.

        Args:
            content: File content as bytes
            user_id: Owner user ID
            document_id: Document ID
            filename: Original filename

        Returns:
            Path to saved file
        """
        user_path = self.get_user_path(user_id)
        file_path = user_path / f"{document_id}_{filename}"

        # Run blocking I/O in thread pool
        await asyncio.to_thread(self._write_file, file_path, content)

        return str(file_path)

    def _write_file(self, path: Path, content: bytes) -> None:
        """Write file synchronously (run in thread pool)."""
        with open(path, 'wb') as f:
            f.write(content)

    async def get_file(self, file_path: str) -> Optional[bytes]:
        """
        Retrieve file from local storage.

        Args:
            file_path: Path to file

        Returns:
            File content as bytes, or None if not found
        """
        path = Path(file_path)
        if path.exists():
            return await asyncio.to_thread(self._read_file, path)
        return None

    def _read_file(self, path: Path) -> bytes:
        """Read file synchronously (run in thread pool)."""
        with open(path, 'rb') as f:
            return f.read()

    async def delete_file(self, file_path: str) -> None:
        """
        Delete file from local storage.

        Args:
            file_path: Path to file
        """
        path = Path(file_path)
        if path.exists():
            await asyncio.to_thread(path.unlink)

    async def list_files(self, user_id: str) -> List[str]:
        """
        List all files for a user.

        Args:
            user_id: User ID

        Returns:
            List of file paths
        """
        user_path = self.get_user_path(user_id)
        return [str(f) for f in user_path.iterdir() if f.is_file()]

    def get_public_url(self, file_path: str) -> str:
        """
        Get public URL for file (returns file path for local storage).

        Args:
            file_path: Path to file

        Returns:
            File path (no public URL for local storage)
        """
        return f"file://{file_path}"


class S3DocumentStorage:
    """
    MinIO/S3-compatible document storage using boto3.

    Stores documents in a single bucket with prefix structure:
        {bucket}/{prefix}/{user_id}/{document_id}_{filename}

    Example:
        storage/documents/user_123/doc_456_report.pdf
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize S3 storage.

        Args:
            config: Storage configuration
        """
        self.config = config
        self._client = None
        self._connected = False

    def _get_client(self):
        """Get or create S3 client."""
        if self._client is not None:
            return self._client

        s3 = _get_boto3()
        if not s3:
            raise ImportError("boto3 is required for S3 storage")

        from botocore.config import Config as BotoConfig

        self._client = s3.client(
            's3',
            endpoint_url=self.config.endpoint,
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
            region_name=self.config.region,
            config=BotoConfig(
                signature_version='s3v4',
                s3={'addressing_style': 'path'},
            ),
        )
        return self._client

    def _get_key(self, user_id: str, document_id: str, filename: str) -> str:
        """Generate S3 key for document."""
        return f"{self.config.documents_prefix}/{user_id}/{document_id}_{filename}"

    async def save_file(
        self,
        content: bytes,
        user_id: str,
        document_id: str,
        filename: str,
    ) -> str:
        """
        Save file to S3.

        Args:
            content: File content as bytes
            user_id: Owner user ID
            document_id: Document ID
            filename: Original filename

        Returns:
            S3 key of saved file
        """
        key = self._get_key(user_id, document_id, filename)

        # Detect content type
        content_type = self._detect_content_type(filename)

        # Run S3 upload in thread pool
        await asyncio.to_thread(
            self._upload_bytes,
            key,
            content,
            content_type,
        )

        print(f"[RAG_STORAGE] Uploaded: s3://{self.config.bucket}/{key}", flush=True)
        return key

    def _upload_bytes(self, key: str, content: bytes, content_type: str) -> None:
        """Upload bytes to S3 synchronously."""
        client = self._get_client()
        client.put_object(
            Bucket=self.config.bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
        )

    async def get_file(self, key_or_path: str) -> Optional[bytes]:
        """
        Retrieve file from S3.

        Args:
            key_or_path: S3 key or full path

        Returns:
            File content as bytes, or None if not found
        """
        # Extract key from path if needed
        key = key_or_path
        if key.startswith(self.config.documents_prefix):
            pass  # Already a key
        elif '/' in key_or_path:
            # Try to extract key from path
            parts = key_or_path.split('/')
            if self.config.documents_prefix in parts:
                idx = parts.index(self.config.documents_prefix)
                key = '/'.join(parts[idx:])

        try:
            content = await asyncio.to_thread(self._download_bytes, key)
            return content
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise

    def _download_bytes(self, key: str) -> bytes:
        """Download bytes from S3 synchronously."""
        client = self._get_client()
        response = client.get_object(Bucket=self.config.bucket, Key=key)
        return response['Body'].read()

    async def delete_file(self, key_or_path: str) -> None:
        """
        Delete file from S3.

        Args:
            key_or_path: S3 key or full path
        """
        key = key_or_path
        if key.startswith(self.config.documents_prefix):
            pass
        elif '/' in key_or_path:
            parts = key_or_path.split('/')
            if self.config.documents_prefix in parts:
                idx = parts.index(self.config.documents_prefix)
                key = '/'.join(parts[idx:])

        await asyncio.to_thread(self._delete_object, key)
        print(f"[RAG_STORAGE] Deleted: s3://{self.config.bucket}/{key}", flush=True)

    def _delete_object(self, key: str) -> None:
        """Delete object from S3 synchronously."""
        client = self._get_client()
        client.delete_object(Bucket=self.config.bucket, Key=key)

    async def list_files(self, user_id: str) -> List[str]:
        """
        List all files for a user.

        Args:
            user_id: User ID

        Returns:
            List of S3 keys
        """
        prefix = f"{self.config.documents_prefix}/{user_id}/"
        keys = await asyncio.to_thread(self._list_objects, prefix)
        return keys

    def _list_objects(self, prefix: str) -> List[str]:
        """List objects with prefix synchronously."""
        client = self._get_client()
        response = client.list_objects_v2(Bucket=self.config.bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]

    def get_public_url(self, key: str) -> str:
        """
        Get public URL for file.

        Args:
            key: S3 key

        Returns:
            Public URL for accessing the file
        """
        if self.config.public_url:
            return f"{self.config.public_url}/{key}"
        return f"{self.config.endpoint}/{self.config.bucket}/{key}"

    def _detect_content_type(self, filename: str) -> str:
        """Detect content type from filename."""
        ext = Path(filename).suffix.lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
        }
        return content_types.get(ext, 'application/octet-stream')


class RAGDocumentStorage:
    """
    Document storage with MinIO/S3 primary and local fallback.

    Automatically falls back to local filesystem if MinIO/S3 is unavailable.

    Usage:
        storage = RAGDocumentStorage()
        await storage.initialize()

        path = await storage.save_file(content, user_id, doc_id, filename)
        content = await storage.get_file(path)
    """

    def __init__(self, config: StorageConfig = None, use_local_fallback: bool = True):
        """
        Initialize storage with optional configuration.

        Args:
            config: Storage configuration (loads from env if not provided)
            use_local_fallback: If True, fall back to local when S3 fails
        """
        self.config = config or StorageConfig.from_env()
        self.use_local_fallback = use_local_fallback
        self._s3_storage: Optional[S3DocumentStorage] = None
        self._local_storage: Optional[LocalDocumentStorage] = None
        self._using_local = False
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the storage, attempting S3 first."""
        if self._initialized:
            return

        # Check if boto3 is available and credentials are configured
        s3 = _get_boto3()
        if s3 and self.config.access_key and self.config.secret_key:
            try:
                self._s3_storage = S3DocumentStorage(self.config)
                # Test connection by listing bucket
                await asyncio.to_thread(self._test_s3_connection)
                self._using_local = False
                print(f"[RAG_STORAGE] Using S3 backend: {self.config.endpoint}", flush=True)
            except Exception as e:
                print(f"[RAG_STORAGE] S3 unavailable: {e}", flush=True)
                if self.use_local_fallback:
                    self._local_storage = LocalDocumentStorage(self.config.local_path)
                    self._using_local = True
                    print(f"[RAG_STORAGE] Falling back to local: {self.config.local_path}", flush=True)
                else:
                    raise
        else:
            if self.use_local_fallback:
                self._local_storage = LocalDocumentStorage(self.config.local_path)
                self._using_local = True
                reason = "boto3 not installed" if not s3 else "credentials not configured"
                print(f"[RAG_STORAGE] {reason}, using local: {self.config.local_path}", flush=True)
            else:
                raise ValueError("S3 storage requires boto3 and valid credentials")

        self._initialized = True

    def _test_s3_connection(self) -> None:
        """Test S3 connection by listing bucket."""
        client = self._s3_storage._get_client()
        # Try to access the bucket
        client.head_bucket(Bucket=self.config.bucket)

    @property
    def _storage(self):
        """Get the active storage backend."""
        if self._using_local:
            return self._local_storage
        return self._s3_storage

    async def save_file(
        self,
        content: bytes,
        user_id: str,
        document_id: str,
        filename: str,
    ) -> str:
        """
        Save file to storage.

        Args:
            content: File content as bytes
            user_id: Owner user ID
            document_id: Document ID
            filename: Original filename

        Returns:
            Path/key of saved file
        """
        if not self._initialized:
            await self.initialize()
        return await self._storage.save_file(content, user_id, document_id, filename)

    async def get_file(self, file_path: str) -> Optional[bytes]:
        """
        Retrieve file from storage.

        Args:
            file_path: Path/key of file

        Returns:
            File content as bytes, or None if not found
        """
        if not self._initialized:
            await self.initialize()
        return await self._storage.get_file(file_path)

    async def delete_file(self, file_path: str) -> None:
        """
        Delete file from storage.

        Args:
            file_path: Path/key of file
        """
        if not self._initialized:
            await self.initialize()
        await self._storage.delete_file(file_path)

    async def list_files(self, user_id: str) -> List[str]:
        """
        List all files for a user.

        Args:
            user_id: User ID

        Returns:
            List of file paths/keys
        """
        if not self._initialized:
            await self.initialize()
        return await self._storage.list_files(user_id)

    def get_public_url(self, file_path: str) -> str:
        """
        Get public URL for file.

        Args:
            file_path: Path/key of file

        Returns:
            Public URL for accessing the file
        """
        return self._storage.get_public_url(file_path)

    @property
    def is_using_local(self) -> bool:
        """Check if using local fallback."""
        return self._using_local


# Module-level factory
_default_storage = None


async def get_document_storage(
    config: StorageConfig = None,
    use_local_fallback: bool = True,
) -> RAGDocumentStorage:
    """
    Get or create a document storage instance.

    Args:
        config: Optional configuration
        use_local_fallback: Fall back to local if S3 unavailable

    Returns:
        Initialized RAGDocumentStorage instance
    """
    global _default_storage
    if _default_storage is None:
        _default_storage = RAGDocumentStorage(config, use_local_fallback)
        await _default_storage.initialize()
    return _default_storage
