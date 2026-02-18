"""
Document Repository - PostgreSQL with Memory Fallback

Provides document metadata storage with:
- PostgreSQL backend for production (asyncpg)
- In-memory backend for development/testing
- Automatic fallback when database is unavailable
"""

import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Lazy imports for optional dependencies
asyncpg = None


def _get_asyncpg():
    """Lazy load asyncpg to avoid import errors when not installed."""
    global asyncpg
    if asyncpg is None:
        try:
            import asyncpg as _asyncpg
            asyncpg = _asyncpg
        except ImportError:
            asyncpg = False
    return asyncpg if asyncpg else None


# Import Document model - handle both possible locations
try:
    from models.document_models import Document, DocumentStatus
except ImportError:
    Document = None
    DocumentStatus = None


@dataclass
class RepositoryConfig:
    """Configuration for document repository."""
    host: str = "localhost"
    port: int = 5432
    user: str = "viralify_prod"
    password: str = ""
    database: str = "viralify_production"
    min_pool_size: int = 2
    max_pool_size: int = 10

    @classmethod
    def from_env(cls) -> "RepositoryConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("DATABASE_HOST", "localhost"),
            port=int(os.getenv("DATABASE_PORT", "5432")),
            user=os.getenv("DATABASE_USER", "viralify_prod"),
            password=os.getenv("DATABASE_PASSWORD", ""),
            database=os.getenv("DATABASE_NAME", "viralify_production"),
            min_pool_size=int(os.getenv("DB_MIN_POOL_SIZE", "2")),
            max_pool_size=int(os.getenv("DB_MAX_POOL_SIZE", "10")),
        )


class InMemoryDocumentRepository:
    """
    Simple in-memory document repository.

    Used for development/testing or as fallback when PostgreSQL is unavailable.
    """

    def __init__(self):
        self.documents: Dict[str, "Document"] = {}
        self.user_documents: Dict[str, List[str]] = {}  # user_id -> [doc_ids]
        self.course_documents: Dict[str, List[str]] = {}  # course_id -> [doc_ids]

    async def save(self, document: "Document") -> None:
        """Save document to repository."""
        self.documents[document.id] = document

        # Index by user
        if document.user_id not in self.user_documents:
            self.user_documents[document.user_id] = []
        if document.id not in self.user_documents[document.user_id]:
            self.user_documents[document.user_id].append(document.id)

        # Index by course
        if document.course_id:
            if document.course_id not in self.course_documents:
                self.course_documents[document.course_id] = []
            if document.id not in self.course_documents[document.course_id]:
                self.course_documents[document.course_id].append(document.id)

    async def get(self, document_id: str) -> Optional["Document"]:
        """Get document by ID."""
        return self.documents.get(document_id)

    async def get_by_user(self, user_id: str) -> List["Document"]:
        """Get all documents for a user."""
        doc_ids = self.user_documents.get(user_id, [])
        return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]

    async def get_by_course(self, course_id: str) -> List["Document"]:
        """Get all documents for a course."""
        doc_ids = self.course_documents.get(course_id, [])
        return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]

    async def get_by_ids(self, document_ids: List[str]) -> List["Document"]:
        """Get multiple documents by IDs."""
        return [self.documents[doc_id] for doc_id in document_ids if doc_id in self.documents]

    async def delete(self, document_id: str) -> None:
        """Delete document."""
        doc = self.documents.pop(document_id, None)
        if doc:
            # Remove from user index
            if doc.user_id in self.user_documents:
                self.user_documents[doc.user_id] = [
                    d for d in self.user_documents[doc.user_id] if d != document_id
                ]
            # Remove from course index
            if doc.course_id and doc.course_id in self.course_documents:
                self.course_documents[doc.course_id] = [
                    d for d in self.course_documents[doc.course_id] if d != document_id
                ]

    async def update_status(
        self,
        document_id: str,
        status: "DocumentStatus",
        error_message: Optional[str] = None,
    ) -> None:
        """Update document status."""
        doc = self.documents.get(document_id)
        if doc:
            doc.status = status
            if error_message:
                doc.error_message = error_message
            if DocumentStatus and status == DocumentStatus.READY:
                doc.processed_at = datetime.utcnow()

    async def close(self) -> None:
        """Close repository (no-op for in-memory)."""
        pass


class PostgresDocumentRepository:
    """
    PostgreSQL-backed document repository using asyncpg.

    Table Schema:
        CREATE TABLE IF NOT EXISTS rag_documents (
            id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            course_id VARCHAR(255),
            filename VARCHAR(500) NOT NULL,
            document_type VARCHAR(50),
            file_path TEXT,
            file_size_bytes BIGINT,
            raw_content TEXT,
            content_summary TEXT,
            page_count INT DEFAULT 0,
            word_count INT DEFAULT 0,
            chunk_count INT DEFAULT 0,
            status VARCHAR(50) DEFAULT 'pending',
            error_message TEXT,
            security_scan JSONB,
            extracted_metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            processed_at TIMESTAMP WITH TIME ZONE
        );

        CREATE INDEX IF NOT EXISTS idx_rag_documents_user ON rag_documents(user_id);
        CREATE INDEX IF NOT EXISTS idx_rag_documents_course ON rag_documents(course_id);
        CREATE INDEX IF NOT EXISTS idx_rag_documents_status ON rag_documents(status);
    """

    TABLE_NAME = "rag_documents"

    def __init__(self, config: RepositoryConfig = None):
        """
        Initialize PostgreSQL repository.

        Args:
            config: Database configuration (loads from env if not provided)
        """
        self.config = config or RepositoryConfig.from_env()
        self._pool = None
        self._connected = False

    async def _ensure_pool(self):
        """Ensure connection pool is initialized."""
        if self._pool is not None:
            return

        pg = _get_asyncpg()
        if not pg:
            raise ImportError("asyncpg is required for PostgreSQL repository")

        try:
            self._pool = await pg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database=self.config.database,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
            )
            self._connected = True
            print(f"[RAG_REPO] PostgreSQL pool created: {self.config.host}:{self.config.port}", flush=True)
        except Exception as e:
            print(f"[RAG_REPO] Failed to create pool: {e}", flush=True)
            raise

    async def save(self, document: "Document") -> None:
        """Save document to PostgreSQL."""
        await self._ensure_pool()

        import json

        query = f"""
            INSERT INTO {self.TABLE_NAME} (
                id, user_id, course_id, filename, document_type,
                file_path, file_size_bytes, raw_content, content_summary,
                page_count, word_count, chunk_count, status, error_message,
                security_scan, extracted_metadata, created_at, processed_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
            )
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                raw_content = EXCLUDED.raw_content,
                content_summary = EXCLUDED.content_summary,
                page_count = EXCLUDED.page_count,
                word_count = EXCLUDED.word_count,
                chunk_count = EXCLUDED.chunk_count,
                security_scan = EXCLUDED.security_scan,
                extracted_metadata = EXCLUDED.extracted_metadata,
                processed_at = EXCLUDED.processed_at
        """

        doc_type = document.document_type.value if document.document_type else None
        status = document.status.value if document.status else "pending"
        security_scan = json.dumps(document.security_scan.__dict__) if document.security_scan else None
        metadata = json.dumps(document.extracted_metadata) if document.extracted_metadata else None

        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                document.id,
                document.user_id,
                document.course_id,
                document.filename,
                doc_type,
                document.file_path,
                document.file_size_bytes,
                document.raw_content,
                getattr(document, 'content_summary', None),
                document.page_count,
                document.word_count,
                getattr(document, 'chunk_count', 0),
                status,
                document.error_message,
                security_scan,
                metadata,
                document.created_at,
                getattr(document, 'processed_at', None),
            )

    async def get(self, document_id: str) -> Optional["Document"]:
        """Get document by ID."""
        await self._ensure_pool()

        query = f"SELECT * FROM {self.TABLE_NAME} WHERE id = $1"

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, document_id)
            return self._row_to_document(row) if row else None

    async def get_by_user(self, user_id: str) -> List["Document"]:
        """Get all documents for a user."""
        await self._ensure_pool()

        query = f"SELECT * FROM {self.TABLE_NAME} WHERE user_id = $1 ORDER BY created_at DESC"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, user_id)
            return [self._row_to_document(row) for row in rows]

    async def get_by_course(self, course_id: str) -> List["Document"]:
        """Get all documents for a course."""
        await self._ensure_pool()

        query = f"SELECT * FROM {self.TABLE_NAME} WHERE course_id = $1 ORDER BY created_at DESC"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, course_id)
            return [self._row_to_document(row) for row in rows]

    async def get_by_ids(self, document_ids: List[str]) -> List["Document"]:
        """Get multiple documents by IDs."""
        if not document_ids:
            return []

        await self._ensure_pool()

        query = f"SELECT * FROM {self.TABLE_NAME} WHERE id = ANY($1)"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, document_ids)
            return [self._row_to_document(row) for row in rows]

    async def delete(self, document_id: str) -> None:
        """Delete document."""
        await self._ensure_pool()

        query = f"DELETE FROM {self.TABLE_NAME} WHERE id = $1"

        async with self._pool.acquire() as conn:
            await conn.execute(query, document_id)

    async def update_status(
        self,
        document_id: str,
        status: "DocumentStatus",
        error_message: Optional[str] = None,
    ) -> None:
        """Update document status."""
        await self._ensure_pool()

        status_val = status.value if hasattr(status, 'value') else str(status)
        processed_at = datetime.utcnow() if status_val == "ready" else None

        if error_message:
            query = f"""
                UPDATE {self.TABLE_NAME}
                SET status = $2, error_message = $3, processed_at = $4
                WHERE id = $1
            """
            params = (document_id, status_val, error_message, processed_at)
        else:
            query = f"""
                UPDATE {self.TABLE_NAME}
                SET status = $2, processed_at = $3
                WHERE id = $1
            """
            params = (document_id, status_val, processed_at)

        async with self._pool.acquire() as conn:
            await conn.execute(query, *params)

    def _row_to_document(self, row) -> Optional["Document"]:
        """Convert database row to Document object."""
        if not row or not Document:
            return None

        import json

        try:
            from models.document_models import DocumentType, DocumentStatus as DS

            doc = Document(
                id=row["id"],
                user_id=row["user_id"],
                course_id=row["course_id"],
                filename=row["filename"],
                document_type=DocumentType(row["document_type"]) if row["document_type"] else None,
                file_path=row["file_path"],
                file_size_bytes=row["file_size_bytes"],
                raw_content=row["raw_content"],
                page_count=row["page_count"] or 0,
                word_count=row["word_count"] or 0,
                status=DS(row["status"]) if row["status"] else DS.PENDING,
                error_message=row["error_message"],
                extracted_metadata=json.loads(row["extracted_metadata"]) if row["extracted_metadata"] else None,
                created_at=row["created_at"],
            )
            doc.processed_at = row["processed_at"]
            return doc
        except Exception as e:
            print(f"[RAG_REPO] Error converting row to Document: {e}", flush=True)
            return None

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._connected = False


class DocumentRepositoryPg:
    """
    Document repository with PostgreSQL primary and in-memory fallback.

    Automatically falls back to in-memory storage if PostgreSQL is unavailable.

    Usage:
        repo = DocumentRepositoryPg()
        await repo.initialize()

        await repo.save(document)
        doc = await repo.get(document_id)
    """

    def __init__(self, config: RepositoryConfig = None, use_memory_fallback: bool = True):
        """
        Initialize repository with optional PostgreSQL config.

        Args:
            config: PostgreSQL configuration (loads from env if not provided)
            use_memory_fallback: If True, fall back to in-memory when PostgreSQL fails
        """
        self.config = config or RepositoryConfig.from_env()
        self.use_memory_fallback = use_memory_fallback
        self._pg_repo: Optional[PostgresDocumentRepository] = None
        self._memory_repo: Optional[InMemoryDocumentRepository] = None
        self._using_memory = False
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the repository, attempting PostgreSQL first."""
        if self._initialized:
            return

        # Check if asyncpg is available
        pg = _get_asyncpg()
        if pg:
            try:
                self._pg_repo = PostgresDocumentRepository(self.config)
                await self._pg_repo._ensure_pool()
                self._using_memory = False
                print("[RAG_REPO] Using PostgreSQL backend", flush=True)
            except Exception as e:
                print(f"[RAG_REPO] PostgreSQL unavailable: {e}", flush=True)
                if self.use_memory_fallback:
                    self._memory_repo = InMemoryDocumentRepository()
                    self._using_memory = True
                    print("[RAG_REPO] Falling back to in-memory backend", flush=True)
                else:
                    raise
        else:
            if self.use_memory_fallback:
                self._memory_repo = InMemoryDocumentRepository()
                self._using_memory = True
                print("[RAG_REPO] asyncpg not installed, using in-memory backend", flush=True)
            else:
                raise ImportError("asyncpg is required for PostgreSQL repository")

        self._initialized = True

    @property
    def _repo(self):
        """Get the active repository."""
        if self._using_memory:
            return self._memory_repo
        return self._pg_repo

    async def save(self, document: "Document") -> None:
        """Save document to repository."""
        if not self._initialized:
            await self.initialize()
        await self._repo.save(document)

    async def get(self, document_id: str) -> Optional["Document"]:
        """Get document by ID."""
        if not self._initialized:
            await self.initialize()
        return await self._repo.get(document_id)

    async def get_by_user(self, user_id: str) -> List["Document"]:
        """Get all documents for a user."""
        if not self._initialized:
            await self.initialize()
        return await self._repo.get_by_user(user_id)

    async def get_by_course(self, course_id: str) -> List["Document"]:
        """Get all documents for a course."""
        if not self._initialized:
            await self.initialize()
        return await self._repo.get_by_course(course_id)

    async def get_by_ids(self, document_ids: List[str]) -> List["Document"]:
        """Get multiple documents by IDs."""
        if not self._initialized:
            await self.initialize()
        return await self._repo.get_by_ids(document_ids)

    async def delete(self, document_id: str) -> None:
        """Delete document."""
        if not self._initialized:
            await self.initialize()
        await self._repo.delete(document_id)

    async def update_status(
        self,
        document_id: str,
        status: "DocumentStatus",
        error_message: Optional[str] = None,
    ) -> None:
        """Update document status."""
        if not self._initialized:
            await self.initialize()
        await self._repo.update_status(document_id, status, error_message)

    async def close(self) -> None:
        """Close repository."""
        if self._repo:
            await self._repo.close()

    @property
    def is_using_memory(self) -> bool:
        """Check if using in-memory fallback."""
        return self._using_memory


# Module-level factory
_default_repository = None


async def get_document_repository(
    config: RepositoryConfig = None,
    use_memory_fallback: bool = True,
) -> DocumentRepositoryPg:
    """
    Get or create a document repository instance.

    Args:
        config: Optional configuration
        use_memory_fallback: Fall back to memory if PostgreSQL unavailable

    Returns:
        Initialized DocumentRepositoryPg instance
    """
    global _default_repository
    if _default_repository is None:
        _default_repository = DocumentRepositoryPg(config, use_memory_fallback)
        await _default_repository.initialize()
    return _default_repository
