"""
Source Library Service

Manages the persistent source library allowing users to save
and reuse sources across multiple courses.
"""
import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import httpx
from openai import AsyncOpenAI

from models.document_models import DocumentChunk, DocumentType, EXTENSION_TO_TYPE
from models.source_models import (
    Source,
    SourceType,
    SourceStatus,
    PedagogicalRole,
    CourseSource,
    SourceResponse,
    SourceSuggestion,
    CreateSourceRequest,
    UpdateSourceRequest,
    SOURCES_TABLE_SQL,
    COURSE_SOURCES_TABLE_SQL,
    SOURCE_CHUNKS_TABLE_SQL,
)
from services.security_scanner import SecurityScanner
from services.document_parser import DocumentParser, WebContentParser
from services.vector_store import VectorizationService


class SourceStorage:
    """File storage for source files"""

    def __init__(self, base_path: str = "/tmp/viralify/sources"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_user_path(self, user_id: str) -> Path:
        """Get storage path for user"""
        path = self.base_path / user_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def save_file(
        self,
        content: bytes,
        user_id: str,
        source_id: str,
        filename: str,
    ) -> str:
        """Save file to storage"""
        user_path = self.get_user_path(user_id)
        file_path = user_path / f"{source_id}_{filename}"

        with open(file_path, 'wb') as f:
            f.write(content)

        return str(file_path)

    async def get_file(self, file_path: str) -> Optional[bytes]:
        """Retrieve file from storage"""
        path = Path(file_path)
        if path.exists():
            with open(path, 'rb') as f:
                return f.read()
        return None

    async def delete_file(self, file_path: str) -> None:
        """Delete file from storage"""
        path = Path(file_path)
        if path.exists():
            path.unlink()


class SourceRepository:
    """
    PostgreSQL-based source repository.
    Falls back to in-memory storage if no database connection.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.pool: Optional[asyncpg.Pool] = None
        self._use_memory = not self.database_url

        # In-memory fallback
        self.sources: Dict[str, Source] = {}
        self.user_sources: Dict[str, List[str]] = {}
        self.course_sources: Dict[str, List[CourseSource]] = {}

    async def initialize(self) -> None:
        """Initialize database connection and create tables"""
        if self._use_memory:
            print("[SOURCE_LIBRARY] ⚠️ WARNING: Using in-memory storage (no DATABASE_URL)", flush=True)
            print("[SOURCE_LIBRARY] ⚠️ Sources will be LOST on restart!", flush=True)
            return

        print(f"[SOURCE_LIBRARY] Connecting to PostgreSQL...", flush=True)

        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )

            # Verify connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                print(f"[SOURCE_LIBRARY] ✅ PostgreSQL connected: {version[:50]}...", flush=True)

                # Create tables (they may already exist from init script)
                await conn.execute(SOURCES_TABLE_SQL)
                await conn.execute(COURSE_SOURCES_TABLE_SQL)

                # Note: SOURCE_CHUNKS_TABLE_SQL requires pgvector extension
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    await conn.execute(SOURCE_CHUNKS_TABLE_SQL)
                    print("[SOURCE_LIBRARY] ✅ Tables verified/created (with pgvector)", flush=True)
                except Exception as e:
                    print(f"[SOURCE_LIBRARY] ⚠️ pgvector not available, chunks will use memory: {e}", flush=True)

                # Count existing sources
                count = await conn.fetchval("SELECT COUNT(*) FROM sources")
                print(f"[SOURCE_LIBRARY] 📊 Found {count} existing sources in database", flush=True)

            self._use_memory = False
            print("[SOURCE_LIBRARY] ✅ Repository ready (persistent mode)", flush=True)

        except Exception as e:
            print(f"[SOURCE_LIBRARY] ❌ Database connection FAILED: {e}", flush=True)
            print(f"[SOURCE_LIBRARY] ⚠️ FALLING BACK to in-memory storage - sources will be LOST on restart!", flush=True)
            self._use_memory = True

    async def close(self) -> None:
        """Close database connection"""
        if self.pool:
            await self.pool.close()

    # ==========================================================================
    # Source CRUD
    # ==========================================================================

    async def save_source(self, source: Source) -> None:
        """Save source to database"""
        if self._use_memory:
            self.sources[source.id] = source
            if source.user_id not in self.user_sources:
                self.user_sources[source.user_id] = []
            if source.id not in self.user_sources[source.user_id]:
                self.user_sources[source.user_id].append(source.id)
            print(f"[SOURCE_LIBRARY] ⚠️ Source {source.id[:8]}... saved to MEMORY (will be lost on restart)", flush=True)
            return

        print(f"[SOURCE_LIBRARY] Saving source {source.id[:8]}... to PostgreSQL", flush=True)

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sources (
                    id, user_id, name, source_type, pedagogical_role, filename, document_type,
                    file_size_bytes, file_path, source_url, note_content,
                    status, error_message, raw_content, content_summary,
                    word_count, chunk_count, is_vectorized, extracted_metadata,
                    tags, usage_count, last_used_at, created_at, updated_at, processed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    pedagogical_role = EXCLUDED.pedagogical_role,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message,
                    raw_content = EXCLUDED.raw_content,
                    content_summary = EXCLUDED.content_summary,
                    word_count = EXCLUDED.word_count,
                    chunk_count = EXCLUDED.chunk_count,
                    is_vectorized = EXCLUDED.is_vectorized,
                    extracted_metadata = EXCLUDED.extracted_metadata,
                    tags = EXCLUDED.tags,
                    usage_count = EXCLUDED.usage_count,
                    last_used_at = EXCLUDED.last_used_at,
                    updated_at = EXCLUDED.updated_at,
                    processed_at = EXCLUDED.processed_at
            """,
                source.id,
                source.user_id,
                source.name,
                source.source_type.value,
                source.pedagogical_role.value,
                source.filename,
                source.document_type.value if source.document_type else None,
                source.file_size_bytes,
                source.file_path,
                source.source_url,
                source.note_content,
                source.status.value,
                source.error_message,
                source.raw_content,
                source.content_summary,
                source.word_count,
                source.chunk_count,
                source.is_vectorized,
                json.dumps(source.extracted_metadata),
                source.tags,
                source.usage_count,
                source.last_used_at,
                source.created_at,
                source.updated_at,
                source.processed_at,
            )
        print(f"[SOURCE_LIBRARY] ✅ Source {source.id[:8]}... saved to PostgreSQL (status={source.status.value})", flush=True)

    async def get_source(self, source_id: str) -> Optional[Source]:
        """Get source by ID"""
        if self._use_memory:
            source = self.sources.get(source_id)
            if not source:
                print(f"[SOURCE_LIBRARY] Source {source_id[:8]}... not found in memory (have {len(self.sources)} sources)", flush=True)
            return source

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM sources WHERE id = $1",
                source_id,
            )
            if row:
                return self._row_to_source(row)
            print(f"[SOURCE_LIBRARY] Source {source_id[:8]}... not found in database", flush=True)
            return None

    async def get_sources_by_user(
        self,
        user_id: str,
        source_type: Optional[SourceType] = None,
        status: Optional[SourceStatus] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[Source], int]:
        """Get all sources for a user with filtering and pagination"""
        if self._use_memory:
            source_ids = self.user_sources.get(user_id, [])
            sources = [self.sources[sid] for sid in source_ids if sid in self.sources]

            # Apply filters
            if source_type:
                sources = [s for s in sources if s.source_type == source_type]
            if status:
                sources = [s for s in sources if s.status == status]
            if tags:
                sources = [s for s in sources if any(t in s.tags for t in tags)]
            if search:
                search_lower = search.lower()
                sources = [s for s in sources if search_lower in s.name.lower()]

            total = len(sources)
            start = (page - 1) * page_size
            end = start + page_size
            return sources[start:end], total

        # Build query with filters
        query = "SELECT * FROM sources WHERE user_id = $1"
        params = [user_id]
        param_idx = 2

        if source_type:
            query += f" AND source_type = ${param_idx}"
            params.append(source_type.value)
            param_idx += 1

        if status:
            query += f" AND status = ${param_idx}"
            params.append(status.value)
            param_idx += 1

        if tags:
            query += f" AND tags && ${param_idx}"
            params.append(tags)
            param_idx += 1

        if search:
            query += f" AND name ILIKE ${param_idx}"
            params.append(f"%{search}%")
            param_idx += 1

        # Count total
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")

        # Add pagination
        query += f" ORDER BY updated_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([page_size, (page - 1) * page_size])

        async with self.pool.acquire() as conn:
            total = await conn.fetchval(count_query, *params[:-2])
            rows = await conn.fetch(query, *params)
            sources = [self._row_to_source(row) for row in rows]
            return sources, total

    async def delete_source(self, source_id: str) -> bool:
        """Delete source"""
        if self._use_memory:
            source = self.sources.pop(source_id, None)
            if source:
                if source.user_id in self.user_sources:
                    self.user_sources[source.user_id] = [
                        sid for sid in self.user_sources[source.user_id] if sid != source_id
                    ]
                return True
            return False

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM sources WHERE id = $1",
                source_id,
            )
            return result == "DELETE 1"

    async def update_source(
        self,
        source_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Source]:
        """Update source fields"""
        source = await self.get_source(source_id)
        if not source:
            return None

        for key, value in updates.items():
            if hasattr(source, key):
                setattr(source, key, value)

        source.updated_at = datetime.utcnow()
        await self.save_source(source)
        return source

    # ==========================================================================
    # Course-Source Links
    # ==========================================================================

    async def link_source_to_course(
        self,
        course_id: str,
        source_id: str,
        user_id: str,
        relevance_score: Optional[float] = None,
        is_primary: bool = False,
    ) -> CourseSource:
        """Link a source to a course"""
        link = CourseSource(
            course_id=course_id,
            source_id=source_id,
            user_id=user_id,
            relevance_score=relevance_score,
            is_primary=is_primary,
        )

        if self._use_memory:
            if course_id not in self.course_sources:
                self.course_sources[course_id] = []
            # Check for existing link
            existing = [l for l in self.course_sources[course_id] if l.source_id == source_id]
            if not existing:
                self.course_sources[course_id].append(link)
            return link

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO course_sources (id, course_id, source_id, user_id, relevance_score, is_primary, added_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (course_id, source_id) DO UPDATE SET
                    relevance_score = EXCLUDED.relevance_score,
                    is_primary = EXCLUDED.is_primary
            """,
                link.id,
                link.course_id,
                link.source_id,
                link.user_id,
                link.relevance_score,
                link.is_primary,
                link.added_at,
            )

        # Update usage count
        source = await self.get_source(source_id)
        if source:
            source.usage_count += 1
            source.last_used_at = datetime.utcnow()
            await self.save_source(source)

        return link

    async def unlink_source_from_course(
        self,
        course_id: str,
        source_id: str,
    ) -> bool:
        """Remove source from course"""
        if self._use_memory:
            if course_id in self.course_sources:
                self.course_sources[course_id] = [
                    l for l in self.course_sources[course_id] if l.source_id != source_id
                ]
                return True
            return False

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM course_sources WHERE course_id = $1 AND source_id = $2",
                course_id,
                source_id,
            )
            return result == "DELETE 1"

    async def get_course_sources(
        self,
        course_id: str,
        user_id: str,
    ) -> List[Tuple[CourseSource, Source]]:
        """Get all sources linked to a course"""
        if self._use_memory:
            links = self.course_sources.get(course_id, [])
            results = []
            for link in links:
                if link.user_id == user_id:
                    source = self.sources.get(link.source_id)
                    if source:
                        results.append((link, source))
            return results

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT cs.*, s.*
                FROM course_sources cs
                JOIN sources s ON cs.source_id = s.id
                WHERE cs.course_id = $1 AND cs.user_id = $2
                ORDER BY cs.is_primary DESC, cs.added_at ASC
            """,
                course_id,
                user_id,
            )

            results = []
            for row in rows:
                link = CourseSource(
                    id=row['id'],
                    course_id=row['course_id'],
                    source_id=row['source_id'],
                    user_id=row['user_id'],
                    relevance_score=row['relevance_score'],
                    is_primary=row['is_primary'],
                    added_at=row['added_at'],
                )
                source = self._row_to_source(row)
                results.append((link, source))

            return results

    def _row_to_source(self, row: asyncpg.Record) -> Source:
        """Convert database row to Source model"""
        # Parse pedagogical role with fallback to AUTO
        pedagogical_role = PedagogicalRole.AUTO
        if row.get('pedagogical_role'):
            try:
                pedagogical_role = PedagogicalRole(row['pedagogical_role'])
            except ValueError:
                pedagogical_role = PedagogicalRole.AUTO

        return Source(
            id=row['id'],
            user_id=row['user_id'],
            name=row['name'],
            source_type=SourceType(row['source_type']),
            pedagogical_role=pedagogical_role,
            filename=row.get('filename'),
            document_type=DocumentType(row['document_type']) if row.get('document_type') else None,
            file_size_bytes=row.get('file_size_bytes', 0),
            file_path=row.get('file_path'),
            source_url=row.get('source_url'),
            note_content=row.get('note_content'),
            status=SourceStatus(row['status']),
            error_message=row.get('error_message'),
            raw_content=row.get('raw_content'),
            content_summary=row.get('content_summary'),
            word_count=row.get('word_count', 0),
            chunk_count=row.get('chunk_count', 0),
            is_vectorized=row.get('is_vectorized', False),
            extracted_metadata=json.loads(row['extracted_metadata']) if row.get('extracted_metadata') else {},
            tags=row.get('tags', []),
            usage_count=row.get('usage_count', 0),
            last_used_at=row.get('last_used_at'),
            created_at=row.get('created_at', datetime.utcnow()),
            updated_at=row.get('updated_at', datetime.utcnow()),
            processed_at=row.get('processed_at'),
        )


class SourceLibraryService:
    """
    Main service for managing the source library.
    """

    def __init__(
        self,
        vector_backend: str = "memory",
        storage_path: str = "/tmp/viralify/sources",
        database_url: Optional[str] = None,
    ):
        self.repository = SourceRepository(database_url)
        self.storage = SourceStorage(storage_path)
        self.security_scanner = SecurityScanner()
        self.document_parser = DocumentParser()
        self.web_parser = WebContentParser()
        self.vectorization = VectorizationService(vector_backend=vector_backend)

        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0,
        )

        print("[SOURCE_LIBRARY] Service initialized", flush=True)

    async def initialize(self) -> None:
        """Initialize the service"""
        await self.repository.initialize()
        print("[SOURCE_LIBRARY] Repository initialized", flush=True)

    async def close(self) -> None:
        """Close connections"""
        await self.repository.close()

    # ==========================================================================
    # Source Creation
    # ==========================================================================

    async def create_source_from_file(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Source:
        """Create a source from an uploaded file"""
        print(f"[SOURCE_LIBRARY] Creating file source: {filename}", flush=True)

        # Determine document type
        ext = Path(filename).suffix.lower()
        if ext not in EXTENSION_TO_TYPE:
            raise ValueError(f"Unsupported file extension: {ext}")

        document_type = EXTENSION_TO_TYPE[ext]

        # Generate unique name to avoid duplicates
        base_name = name or filename
        unique_name = await self._get_unique_name(user_id, base_name)

        # Create source record
        source = Source(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=unique_name,
            source_type=SourceType.FILE,
            filename=self.security_scanner.sanitize_filename(filename),
            document_type=document_type,
            file_size_bytes=len(file_content),
            status=SourceStatus.PROCESSING,
            tags=tags or [],
        )

        await self.repository.save_source(source)

        try:
            # Security scan
            scan_result = await self.security_scanner.scan_file(
                file_content,
                filename,
                document_type,
            )

            if not scan_result.is_safe:
                source.status = SourceStatus.FAILED
                source.error_message = f"Security scan failed: {', '.join(scan_result.threats_found)}"
                await self.repository.save_source(source)
                raise ValueError(source.error_message)

            # Save file
            file_path = await self.storage.save_file(
                file_content,
                user_id,
                source.id,
                source.filename,
            )
            source.file_path = file_path

            # Parse document
            raw_text, chunks, metadata = await self.document_parser.parse_document(
                file_content,
                filename,
                document_type,
            )

            source.raw_content = raw_text
            source.extracted_metadata = metadata
            source.word_count = len(raw_text.split())

            # Vectorize
            vectorized_chunks = await self.vectorization.vectorize_chunks(
                chunks,
                source.id,
                user_id,
            )

            source.chunks = vectorized_chunks
            source.chunk_count = len(vectorized_chunks)
            source.is_vectorized = True

            # Generate summary
            source.content_summary = await self._generate_summary(raw_text[:4000])

            # Mark ready
            source.status = SourceStatus.READY
            source.processed_at = datetime.utcnow()
            await self.repository.save_source(source)

            print(f"[SOURCE_LIBRARY] File source ready: {source.id}", flush=True)
            return source

        except Exception as e:
            print(f"[SOURCE_LIBRARY] File processing failed: {e}", flush=True)
            source.status = SourceStatus.FAILED
            source.error_message = str(e)
            await self.repository.save_source(source)
            raise

    async def create_source_from_url(
        self,
        url: str,
        user_id: str,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Source:
        """Create a source from a URL"""
        print(f"[SOURCE_LIBRARY] Creating URL source: {url}", flush=True)

        # Determine source type
        is_youtube = "youtube.com" in url or "youtu.be" in url
        source_type = SourceType.YOUTUBE if is_youtube else SourceType.URL

        # Initial name (will be updated after parsing)
        initial_name = name or url[:100]
        unique_name = await self._get_unique_name(user_id, initial_name)

        source = Source(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=unique_name,
            source_type=source_type,
            source_url=url,
            status=SourceStatus.PROCESSING,
            tags=tags or [],
        )

        await self.repository.save_source(source)

        try:
            # Parse content
            if is_youtube:
                raw_text, metadata = await self.web_parser.parse_youtube(url)
                # Update name from metadata if no custom name was provided
                if not name:
                    parsed_name = metadata.get('title', 'YouTube Video')[:100]
                    source.name = await self._get_unique_name(user_id, parsed_name)
            else:
                raw_text, metadata = await self.web_parser.parse_url(url)
                if not name:
                    parsed_name = metadata.get('title', 'Web Page')[:100]
                    source.name = await self._get_unique_name(user_id, parsed_name)

            source.raw_content = raw_text
            source.extracted_metadata = metadata
            source.word_count = len(raw_text.split())

            # Create chunks and vectorize
            chunks = self.document_parser._create_chunks(raw_text, metadata)
            vectorized_chunks = await self.vectorization.vectorize_chunks(
                chunks,
                source.id,
                user_id,
            )

            source.chunks = vectorized_chunks
            source.chunk_count = len(vectorized_chunks)
            source.is_vectorized = True

            # Generate summary
            source.content_summary = await self._generate_summary(raw_text[:4000])

            # Mark ready
            source.status = SourceStatus.READY
            source.processed_at = datetime.utcnow()
            await self.repository.save_source(source)

            print(f"[SOURCE_LIBRARY] URL source ready: {source.id}", flush=True)
            return source

        except Exception as e:
            print(f"[SOURCE_LIBRARY] URL processing failed: {e}", flush=True)
            source.status = SourceStatus.FAILED
            source.error_message = str(e)
            await self.repository.save_source(source)
            raise

    async def create_note_source(
        self,
        content: str,
        user_id: str,
        name: str,
        tags: Optional[List[str]] = None,
    ) -> Source:
        """Create a source from user notes"""
        print(f"[SOURCE_LIBRARY] Creating note source: {name}", flush=True)

        # Generate unique name to avoid duplicates
        unique_name = await self._get_unique_name(user_id, name)

        source = Source(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=unique_name,
            source_type=SourceType.NOTE,
            note_content=content,
            raw_content=content,
            word_count=len(content.split()),
            status=SourceStatus.PROCESSING,
            tags=tags or [],
        )

        await self.repository.save_source(source)

        try:
            # Create chunks and vectorize
            chunks = self.document_parser._create_chunks(content, {"title": name})
            vectorized_chunks = await self.vectorization.vectorize_chunks(
                chunks,
                source.id,
                user_id,
            )

            source.chunks = vectorized_chunks
            source.chunk_count = len(vectorized_chunks)
            source.is_vectorized = True

            # Generate summary if content is long
            if source.word_count > 100:
                source.content_summary = await self._generate_summary(content[:4000])

            source.status = SourceStatus.READY
            source.processed_at = datetime.utcnow()
            await self.repository.save_source(source)

            print(f"[SOURCE_LIBRARY] Note source ready: {source.id}", flush=True)
            return source

        except Exception as e:
            print(f"[SOURCE_LIBRARY] Note processing failed: {e}", flush=True)
            source.status = SourceStatus.FAILED
            source.error_message = str(e)
            await self.repository.save_source(source)
            raise

    # ==========================================================================
    # Source Management
    # ==========================================================================

    async def get_source(
        self,
        source_id: str,
        user_id: str,
    ) -> Optional[Source]:
        """Get source with access control"""
        source = await self.repository.get_source(source_id)
        if source and source.user_id == user_id:
            return source
        return None

    async def list_sources(
        self,
        user_id: str,
        source_type: Optional[SourceType] = None,
        status: Optional[SourceStatus] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[Source], int]:
        """List user's sources with filtering"""
        return await self.repository.get_sources_by_user(
            user_id=user_id,
            source_type=source_type,
            status=status,
            tags=tags,
            search=search,
            page=page,
            page_size=page_size,
        )

    async def update_source(
        self,
        source_id: str,
        user_id: str,
        request: UpdateSourceRequest,
    ) -> Optional[Source]:
        """Update source metadata"""
        source = await self.get_source(source_id, user_id)
        if not source:
            return None

        updates = {}
        if request.name is not None:
            updates['name'] = request.name
        if request.tags is not None:
            updates['tags'] = request.tags
        if request.pedagogical_role is not None:
            updates['pedagogical_role'] = request.pedagogical_role
        if request.note_content is not None and source.source_type == SourceType.NOTE:
            updates['note_content'] = request.note_content
            updates['raw_content'] = request.note_content
            updates['word_count'] = len(request.note_content.split())

        return await self.repository.update_source(source_id, updates)

    async def delete_source(
        self,
        source_id: str,
        user_id: str,
    ) -> bool:
        """Delete source with access control"""
        source = await self.get_source(source_id, user_id)
        if not source:
            return False

        # Delete from vector store
        await self.vectorization.delete_document(source_id)

        # Delete file if exists
        if source.file_path:
            await self.storage.delete_file(source.file_path)

        # Delete from repository
        return await self.repository.delete_source(source_id)

    # ==========================================================================
    # Course-Source Links
    # ==========================================================================

    async def link_to_course(
        self,
        course_id: str,
        source_id: str,
        user_id: str,
        is_primary: bool = False,
    ) -> Optional[CourseSource]:
        """Link a source to a course"""
        source = await self.get_source(source_id, user_id)
        if not source or source.status != SourceStatus.READY:
            return None

        return await self.repository.link_source_to_course(
            course_id=course_id,
            source_id=source_id,
            user_id=user_id,
            is_primary=is_primary,
        )

    async def unlink_from_course(
        self,
        course_id: str,
        source_id: str,
        user_id: str,
    ) -> bool:
        """Remove a source from a course"""
        source = await self.get_source(source_id, user_id)
        if not source:
            return False

        return await self.repository.unlink_source_from_course(course_id, source_id)

    async def get_course_sources(
        self,
        course_id: str,
        user_id: str,
    ) -> List[Tuple[CourseSource, Source]]:
        """Get all sources linked to a course"""
        return await self.repository.get_course_sources(course_id, user_id)

    async def get_context_for_course(
        self,
        course_id: str,
        topic: str,
        user_id: str,
        max_tokens: int = 4000,
    ) -> str:
        """Get RAG context from course sources"""
        links = await self.get_course_sources(course_id, user_id)

        if not links:
            return ""

        source_ids = [link.source_id for link, _ in links]

        # Use vectorization service for search
        from models.document_models import RAGQueryRequest
        from services.retrieval_service import RAGService

        # Create a temporary RAG service for the query
        results = await self.vectorization.search(
            query=topic,
            user_id=user_id,
            document_ids=source_ids,
            top_k=10,
            similarity_threshold=0.5,
        )

        # Build context
        context_parts = []
        current_tokens = 0

        for result in results:
            # Find the source name
            source_name = "Unknown"
            for _, source in links:
                if source.id == result.document_id:
                    source_name = source.name
                    break

            chunk_text = f"[Source: {source_name}]\n{result.content}"
            chunk_tokens = result.token_count

            if current_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            current_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    async def get_context_from_source_ids(
        self,
        source_ids: List[str],
        topic: str,
        description: Optional[str],
        user_id: str,
        max_tokens: int = 6000,
    ) -> str:
        """
        Get RAG context directly from a list of source IDs.

        This is used when generating courses with sources that may not be
        linked to a specific course yet.

        The context is organized by pedagogical role to help the AI understand
        which sources provide theory, examples, references, etc.

        Args:
            source_ids: List of source IDs to get context from
            topic: Course topic for semantic search
            description: Course description for search
            user_id: User ID for access control
            max_tokens: Maximum tokens to return

        Returns:
            Combined context string from the sources, organized by pedagogical role
        """
        import tiktoken

        try:
            tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        def count_tokens(text: str) -> int:
            if not text:
                return 0
            return len(tokenizer.encode(text))

        print(f"[SOURCE_LIBRARY] get_context_from_source_ids called:", flush=True)
        print(f"[SOURCE_LIBRARY]   - source_ids: {source_ids}", flush=True)
        print(f"[SOURCE_LIBRARY]   - user_id: {user_id}", flush=True)
        print(f"[SOURCE_LIBRARY]   - topic: {topic}", flush=True)
        print(f"[SOURCE_LIBRARY]   - max_tokens: {max_tokens}", flush=True)
        print(f"[SOURCE_LIBRARY] Repository has {len(self.repository.sources)} sources in memory", flush=True)
        print(f"[SOURCE_LIBRARY] Repository source keys: {list(self.repository.sources.keys())[:5]}...", flush=True)

        # Collect all sources and their content, organized by pedagogical role
        sources_by_role: Dict[str, List[Dict[str, Any]]] = {
            PedagogicalRole.THEORY.value: [],
            PedagogicalRole.EXAMPLE.value: [],
            PedagogicalRole.REFERENCE.value: [],
            PedagogicalRole.OPINION.value: [],
            PedagogicalRole.DATA.value: [],
            PedagogicalRole.CONTEXT.value: [],
            PedagogicalRole.AUTO.value: [],
        }
        valid_sources = []
        total_raw_tokens = 0

        for source_id in source_ids:
            source = await self.repository.get_source(source_id)
            print(f"[SOURCE_LIBRARY] Source {source_id}: found={source is not None}", flush=True)

            if source:
                # Check user_id match - allow 'anonymous' as fallback for documents uploaded before profile selection
                user_id_match = (
                    source.user_id == user_id or
                    source.user_id == 'anonymous' or  # Docs uploaded before profile selected
                    user_id == 'anonymous'  # Generation without profile
                )
                print(f"[SOURCE_LIBRARY]   - user_id match: {user_id_match} (source: {source.user_id}, request: {user_id})", flush=True)
                print(f"[SOURCE_LIBRARY]   - status: {source.status}", flush=True)
                print(f"[SOURCE_LIBRARY]   - pedagogical_role: {source.pedagogical_role.value}", flush=True)
                print(f"[SOURCE_LIBRARY]   - has raw_content: {bool(source.raw_content)}", flush=True)

            # Allow access if:
            # 1. Direct user_id match
            # 2. Source was uploaded as 'anonymous' (before profile selection)
            # 3. Request is 'anonymous' (generation without profile)
            user_id_allowed = (
                source.user_id == user_id or
                source.user_id == 'anonymous' or
                user_id == 'anonymous'
            ) if source else False

            if source and user_id_allowed and source.status == SourceStatus.READY:
                if source.raw_content:
                    role = source.pedagogical_role.value
                    doc_header = f"[{source.name}]\n"
                    if source.content_summary:
                        doc_header += f"Summary: {source.content_summary}\n\n"

                    doc_content = doc_header + source.raw_content
                    doc_tokens = count_tokens(doc_content)

                    source_data = {
                        "source": source,
                        "content": doc_content,
                        "tokens": doc_tokens,
                    }
                    sources_by_role[role].append(source_data)
                    valid_sources.append(source_data)
                    total_raw_tokens += doc_tokens
                    print(f"[SOURCE_LIBRARY] Source {source.name} ({role}): {doc_tokens} tokens", flush=True)

        if not valid_sources:
            print("[SOURCE_LIBRARY] No valid sources found with content", flush=True)
            return ""

        # Build context organized by pedagogical role
        def build_role_context(role_sources: List[Dict], role_name: str) -> str:
            if not role_sources:
                return ""
            role_header = self._get_role_header(role_name)
            parts = [f"\n{'='*60}\n{role_header}\n{'='*60}"]
            for s in role_sources:
                parts.append(s["content"])
            return "\n\n".join(parts)

        # If total content fits, use all of it organized by role
        if total_raw_tokens <= max_tokens:
            print(f"[SOURCE_LIBRARY] Using FULL source content ({total_raw_tokens} tokens)", flush=True)
            context_parts = []

            # Order roles by pedagogical importance
            role_order = [
                PedagogicalRole.THEORY.value,
                PedagogicalRole.REFERENCE.value,
                PedagogicalRole.CONTEXT.value,
                PedagogicalRole.EXAMPLE.value,
                PedagogicalRole.DATA.value,
                PedagogicalRole.OPINION.value,
                PedagogicalRole.AUTO.value,
            ]

            for role in role_order:
                role_context = build_role_context(sources_by_role[role], role)
                if role_context:
                    context_parts.append(role_context)

            return "\n".join(context_parts)

        # If too large, use semantic search but still organize by role
        print(f"[SOURCE_LIBRARY] Content too large ({total_raw_tokens} tokens), using search", flush=True)

        query = topic
        if description:
            query += f" {description}"

        results = await self.vectorization.search(
            query=query,
            user_id=user_id,
            document_ids=source_ids,
            top_k=20,
            similarity_threshold=0.3,
        )

        # Build context with source names and roles
        context_parts = []
        context_parts.append("=== SOURCE SUMMARIES BY ROLE ===")

        # Add summaries grouped by role
        role_order = [
            PedagogicalRole.THEORY.value,
            PedagogicalRole.REFERENCE.value,
            PedagogicalRole.CONTEXT.value,
            PedagogicalRole.EXAMPLE.value,
            PedagogicalRole.DATA.value,
            PedagogicalRole.OPINION.value,
            PedagogicalRole.AUTO.value,
        ]

        for role in role_order:
            role_sources = sources_by_role[role]
            if role_sources:
                context_parts.append(f"\n[{self._get_role_header(role)}]")
                for s in role_sources:
                    if s["source"].content_summary:
                        context_parts.append(f"  - {s['source'].name}: {s['source'].content_summary}")

        context_parts.append("\n=== RELEVANT CONTENT ===")

        current_tokens = count_tokens("\n".join(context_parts))

        for result in results:
            # Find source info
            source_name = "Unknown"
            source_role = "auto"
            for s in valid_sources:
                if s["source"].id == result.document_id:
                    source_name = s["source"].name
                    source_role = s["source"].pedagogical_role.value
                    break

            chunk_text = f"\n[Source: {source_name} | Role: {source_role}]\n{result.content}"
            chunk_tokens = result.token_count

            if current_tokens + chunk_tokens > max_tokens:
                break

            context_parts.append(chunk_text)
            current_tokens += chunk_tokens

        combined = "\n".join(context_parts)
        print(f"[SOURCE_LIBRARY] Returning context: {count_tokens(combined)} tokens", flush=True)

        return combined

    def _get_role_header(self, role: str) -> str:
        """Get human-readable header for a pedagogical role."""
        headers = {
            PedagogicalRole.THEORY.value: "📚 THEORY SOURCES (Definitions, Concepts, Explanations)",
            PedagogicalRole.EXAMPLE.value: "💡 EXAMPLE SOURCES (Practical Examples, Demos, Tutorials)",
            PedagogicalRole.REFERENCE.value: "📖 REFERENCE SOURCES (Official Documentation, Specifications)",
            PedagogicalRole.OPINION.value: "💭 OPINION SOURCES (Personal Notes, Perspectives)",
            PedagogicalRole.DATA.value: "📊 DATA SOURCES (Statistics, Studies, Research)",
            PedagogicalRole.CONTEXT.value: "🔍 CONTEXT SOURCES (Background, History, Prerequisites)",
            PedagogicalRole.AUTO.value: "📄 OTHER SOURCES",
        }
        return headers.get(role, "📄 OTHER SOURCES")

    async def get_sources_for_traceability(
        self,
        source_ids: List[str],
        user_id: str,
    ) -> Tuple[List[Source], List[Dict[str, Any]]]:
        """
        Get sources and their chunks for traceability building.

        Args:
            source_ids: List of source IDs
            user_id: User ID for access control

        Returns:
            Tuple of (sources list, chunks list with embeddings and metadata)
        """
        print(f"[SOURCE_LIBRARY] Getting sources for traceability: {len(source_ids)} IDs", flush=True)

        sources = []
        all_chunks = []

        for source_id in source_ids:
            source = await self.repository.get_source(source_id)
            if source and source.user_id == user_id and source.status == SourceStatus.READY:
                sources.append(source)

                # Collect chunks with metadata
                for i, chunk in enumerate(source.chunks):
                    chunk_data = {
                        "source_id": source.id,
                        "content": chunk.content,
                        "chunk_index": i,
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title,
                        "embedding": chunk.embedding,
                        "token_count": chunk.token_count,
                    }
                    all_chunks.append(chunk_data)

        print(f"[SOURCE_LIBRARY] Found {len(sources)} sources, {len(all_chunks)} chunks", flush=True)
        return sources, all_chunks

    # ==========================================================================
    # AI Suggestions
    # ==========================================================================

    async def suggest_sources(
        self,
        topic: str,
        description: Optional[str],
        user_id: str,
        language: str = "fr",
        max_suggestions: int = 5,
    ) -> Tuple[List[SourceSuggestion], List[Source]]:
        """Suggest sources for a course topic"""
        print(f"[SOURCE_LIBRARY] Suggesting sources for: {topic}", flush=True)

        # First, find relevant existing sources
        existing_sources, _ = await self.list_sources(
            user_id=user_id,
            status=SourceStatus.READY,
        )

        relevant_existing = []
        if existing_sources:
            # Use AI to find relevant existing sources
            existing_info = "\n".join([
                f"- {s.name} ({s.source_type.value}): {s.content_summary or 'No summary'}"
                for s in existing_sources[:20]
            ])

            relevance_prompt = f"""Given this course topic: "{topic}"
{f'Description: {description}' if description else ''}

Here are the user's existing sources:
{existing_info}

List the IDs of sources that would be relevant (return as JSON array of names):
"""
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": relevance_prompt}],
                    temperature=0.3,
                    max_tokens=500,
                )
                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                relevant_names = json.loads(content)

                for source in existing_sources:
                    if source.name in relevant_names:
                        relevant_existing.append(source)
            except Exception as e:
                print(f"[SOURCE_LIBRARY] Relevance check failed: {e}", flush=True)

        # Generate new source suggestions
        # Request more suggestions than needed to account for invalid URLs
        request_count = max_suggestions * 2
        lang_name = "français" if language == "fr" else "English"

        suggestion_prompt = f"""Suggest {request_count} sources for creating a course about: "{topic}"
{f'Description: {description}' if description else ''}

Language: {lang_name}

For each suggestion provide:
1. Type: "url" (web article/documentation) or "youtube" (video tutorial)
2. A specific title/name
3. A real, working URL (for web/youtube sources)
4. Why it's relevant
5. Relevance score (0-1)
6. Key keywords

Focus on high-quality, educational resources.

Respond in JSON format:
{{
    "suggestions": [
        {{
            "type": "url|youtube",
            "title": "Source title",
            "url": "https://...",
            "description": "Why this is relevant",
            "relevance_score": 0.9,
            "keywords": ["keyword1", "keyword2"]
        }}
    ]
}}

Only respond with valid JSON."""

        suggestions = []
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": suggestion_prompt}],
                temperature=0.5,
                max_tokens=1500,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            for item in data.get("suggestions", []):
                suggestion = SourceSuggestion(
                    suggestion_type=SourceType.YOUTUBE if item.get("type") == "youtube" else SourceType.URL,
                    title=item.get("title", ""),
                    url=item.get("url"),
                    description=item.get("description", ""),
                    relevance_score=float(item.get("relevance_score", 0.5)),
                    keywords=item.get("keywords", []),
                )
                suggestions.append(suggestion)

        except Exception as e:
            print(f"[SOURCE_LIBRARY] Suggestion generation failed: {e}", flush=True)

        # Verify URLs before returning suggestions
        if suggestions:
            print(f"[SOURCE_LIBRARY] Verifying {len(suggestions)} suggested URLs...", flush=True)
            suggestions = await self._verify_urls_batch(suggestions)
            # Limit to requested number after verification
            suggestions = suggestions[:max_suggestions]

        print(f"[SOURCE_LIBRARY] Found {len(relevant_existing)} relevant, suggested {len(suggestions)} verified URLs", flush=True)
        return suggestions, relevant_existing

    # ==========================================================================
    # Helpers
    # ==========================================================================

    async def _verify_url(self, url: str, timeout: float = 15.0, min_content_length: int = 500) -> bool:
        """
        Verify that a URL is accessible and has real content.
        Returns True only if:
        - URL responds with 2xx status code
        - Response has meaningful content (not empty or error page)
        """
        if not url:
            return False

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5,fr;q=0.3",
                }
            ) as client:
                response = await client.get(url)

                # Only accept 2xx status codes
                if not (200 <= response.status_code < 300):
                    print(f"[SOURCE_LIBRARY] URL returned {response.status_code}: {url}", flush=True)
                    return False

                # Check content length
                content_length = len(response.content)
                if content_length < min_content_length:
                    print(f"[SOURCE_LIBRARY] URL has insufficient content ({content_length} bytes): {url}", flush=True)
                    return False

                # Check for common error page indicators in the content
                content_text = response.text.lower()
                error_indicators = [
                    "404 not found",
                    "page not found",
                    "cette page n'existe pas",
                    "page introuvable",
                    "error 404",
                    "not found</title>",
                    "access denied",
                    "forbidden</title>",
                ]
                for indicator in error_indicators:
                    if indicator in content_text:
                        print(f"[SOURCE_LIBRARY] URL appears to be error page: {url}", flush=True)
                        return False

                print(f"[SOURCE_LIBRARY] URL verified OK ({content_length} bytes): {url}", flush=True)
                return True

        except httpx.TimeoutException:
            print(f"[SOURCE_LIBRARY] URL verification timeout: {url}", flush=True)
            return False
        except httpx.RequestError as e:
            print(f"[SOURCE_LIBRARY] URL verification failed: {url} - {e}", flush=True)
            return False
        except Exception as e:
            print(f"[SOURCE_LIBRARY] URL verification error: {url} - {e}", flush=True)
            return False

    async def _verify_urls_batch(
        self,
        suggestions: List["SourceSuggestion"],
        max_concurrent: int = 5,
    ) -> List["SourceSuggestion"]:
        """
        Verify multiple URLs concurrently and return only valid ones.
        """
        if not suggestions:
            return []

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def verify_with_semaphore(suggestion: "SourceSuggestion") -> Optional["SourceSuggestion"]:
            async with semaphore:
                if suggestion.url and await self._verify_url(suggestion.url):
                    return suggestion
                return None

        # Run verifications concurrently
        tasks = [verify_with_semaphore(s) for s in suggestions]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        valid_suggestions = [s for s in results if s is not None]
        print(f"[SOURCE_LIBRARY] URL verification: {len(valid_suggestions)}/{len(suggestions)} valid", flush=True)

        return valid_suggestions

    async def _get_unique_name(
        self,
        user_id: str,
        base_name: str,
    ) -> str:
        """
        Generate a unique name for a source by appending a number if needed.
        e.g., "file.pdf" -> "file (1).pdf" -> "file (2).pdf"
        """
        # Get all sources for this user to check for duplicates
        sources, _ = await self.repository.get_sources_by_user(
            user_id=user_id,
            page=1,
            page_size=1000,  # Get all to check names
        )

        existing_names = {s.name for s in sources}

        # If name doesn't exist, use it as-is
        if base_name not in existing_names:
            return base_name

        # Find a unique name by appending a number
        # Split name and extension for files
        import re
        match = re.match(r'^(.+?)(\.[^.]+)?$', base_name)
        if match:
            name_part = match.group(1)
            ext_part = match.group(2) or ""
        else:
            name_part = base_name
            ext_part = ""

        # Remove any existing " (N)" suffix
        clean_name = re.sub(r'\s*\(\d+\)$', '', name_part)

        # Find the next available number
        counter = 1
        while True:
            new_name = f"{clean_name} ({counter}){ext_part}"
            if new_name not in existing_names:
                return new_name
            counter += 1
            if counter > 1000:  # Safety limit
                raise ValueError("Too many duplicates with this name")

    async def _generate_summary(self, text: str) -> str:
        """Generate AI summary of source content"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a brief summary (2-3 sentences) of the following content. Focus on the main topics and key points.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[SOURCE_LIBRARY] Summary generation failed: {e}", flush=True)
            return ""
