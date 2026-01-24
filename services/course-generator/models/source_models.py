"""
Source Library Models for RAG System

Defines models for the persistent source library that allows users
to save and reuse sources across multiple courses.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from models.document_models import DocumentType, DocumentStatus, DocumentChunk


class SourceType(str, Enum):
    """Types of sources in the library"""
    FILE = "file"           # Uploaded file (PDF, DOCX, etc.)
    URL = "url"             # Web page
    YOUTUBE = "youtube"     # YouTube video
    NOTE = "note"           # User text note


class PedagogicalRole(str, Enum):
    """
    Role of the source in course building.
    Determines how the content is used in the generated course.
    """
    THEORY = "theory"           # Definitions, concepts, explanations (books, papers, docs)
    EXAMPLE = "example"         # Practical examples, demos, tutorials (videos, code samples)
    REFERENCE = "reference"     # Official documentation, specifications
    OPINION = "opinion"         # Personal notes, opinions, specific perspectives
    DATA = "data"               # Statistics, studies, research data
    CONTEXT = "context"         # Background information, history, prerequisites
    AUTO = "auto"               # Let AI determine based on content analysis


class SourceStatus(str, Enum):
    """Source processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Source(BaseModel):
    """A source in the user's library - persisted and reusable across courses"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="Owner user ID")

    # Source identification
    name: str = Field(..., description="User-friendly name for the source")
    source_type: SourceType = Field(..., description="Type of source")
    pedagogical_role: PedagogicalRole = Field(
        default=PedagogicalRole.AUTO,
        description="Role of this source in course generation (theory, example, etc.)"
    )

    # File-specific fields
    filename: Optional[str] = Field(None, description="Original filename for file sources")
    document_type: Optional[DocumentType] = Field(None, description="Document type for file sources")
    file_size_bytes: int = Field(default=0)
    file_path: Optional[str] = Field(None, description="Storage path for files")

    # URL-specific fields
    source_url: Optional[str] = Field(None, description="URL for web/youtube sources")

    # Note-specific fields
    note_content: Optional[str] = Field(None, description="Content for text notes")

    # Processing status
    status: SourceStatus = Field(default=SourceStatus.PENDING)
    error_message: Optional[str] = Field(None)

    # Extracted content
    raw_content: Optional[str] = Field(None, description="Extracted/parsed text content")
    content_summary: Optional[str] = Field(None, description="AI-generated summary")
    word_count: int = Field(default=0)

    # RAG chunks
    chunks: List[DocumentChunk] = Field(default_factory=list)
    chunk_count: int = Field(default=0)
    is_vectorized: bool = Field(default=False)

    # Metadata
    extracted_metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list, description="User tags for organization")

    # Usage tracking
    usage_count: int = Field(default=0, description="Number of courses using this source")
    last_used_at: Optional[datetime] = Field(None)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "src-123",
                "user_id": "user-456",
                "name": "Python Best Practices Guide",
                "source_type": "file",
                "filename": "python_guide.pdf",
                "document_type": "pdf",
                "status": "ready",
                "word_count": 15000,
                "chunk_count": 50,
                "tags": ["python", "programming"],
                "usage_count": 3,
            }
        }


class CourseSource(BaseModel):
    """Links a source to a course - many-to-many relationship"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    course_id: str = Field(..., description="Course ID")
    source_id: str = Field(..., description="Source ID from library")
    user_id: str = Field(..., description="Owner user ID")

    # Usage context
    relevance_score: Optional[float] = Field(None, description="AI-calculated relevance to course topic")
    is_primary: bool = Field(default=False, description="Primary source for the course")

    # Timestamps
    added_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "cs-123",
                "course_id": "course-456",
                "source_id": "src-789",
                "user_id": "user-456",
                "relevance_score": 0.85,
                "is_primary": True,
            }
        }


# =============================================================================
# Request Models
# =============================================================================

class CreateSourceRequest(BaseModel):
    """Request to create a new source"""
    user_id: str = Field(..., description="User ID")
    name: str = Field(..., description="Source name", min_length=1, max_length=200)
    source_type: SourceType = Field(..., description="Type of source")
    pedagogical_role: PedagogicalRole = Field(
        default=PedagogicalRole.AUTO,
        description="Role of this source (theory, example, opinion, etc.)"
    )

    # For URL/YouTube sources
    source_url: Optional[str] = Field(None, description="URL for web/youtube sources")

    # For note sources
    note_content: Optional[str] = Field(None, description="Content for text notes")

    # Optional tags
    tags: List[str] = Field(default_factory=list, max_length=10)

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "name": "Python Tutorial",
                "source_type": "url",
                "source_url": "https://docs.python.org/3/tutorial/",
                "tags": ["python", "tutorial"],
            }
        }


class BulkCreateSourceRequest(BaseModel):
    """Request to create multiple sources at once"""
    user_id: str = Field(..., description="User ID")
    sources: List[CreateSourceRequest] = Field(..., description="Sources to create", min_length=1, max_length=50)


class UpdateSourceRequest(BaseModel):
    """Request to update a source"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    tags: Optional[List[str]] = Field(None, max_length=10)
    note_content: Optional[str] = Field(None, description="Update content for note sources")
    pedagogical_role: Optional[PedagogicalRole] = Field(None, description="Update pedagogical role")


class LinkSourceToCourseRequest(BaseModel):
    """Request to link a source to a course"""
    course_id: str = Field(..., description="Course ID")
    source_id: str = Field(..., description="Source ID from library")
    user_id: str = Field(..., description="User ID")
    is_primary: bool = Field(default=False)


class BulkLinkSourcesRequest(BaseModel):
    """Request to link multiple sources to a course"""
    course_id: str = Field(..., description="Course ID")
    source_ids: List[str] = Field(..., description="Source IDs to link", min_length=1)
    user_id: str = Field(..., description="User ID")


class SuggestSourcesRequest(BaseModel):
    """Request for AI source suggestions"""
    user_id: str = Field(..., description="User ID")
    topic: str = Field(..., description="Course topic", min_length=3)
    description: Optional[str] = Field(None, description="Course description")
    language: str = Field(default="fr", description="Language for suggestions")
    max_suggestions: int = Field(default=5, ge=1, le=10)


# =============================================================================
# Response Models
# =============================================================================

class SourceResponse(BaseModel):
    """Response with source details"""
    id: str
    user_id: str
    name: str
    source_type: SourceType
    pedagogical_role: PedagogicalRole
    filename: Optional[str]
    document_type: Optional[DocumentType]
    file_size_bytes: int
    source_url: Optional[str]
    note_content: Optional[str]
    status: SourceStatus
    error_message: Optional[str]
    content_summary: Optional[str]
    word_count: int
    chunk_count: int
    is_vectorized: bool
    tags: List[str]
    usage_count: int
    last_used_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_source(cls, source: Source) -> "SourceResponse":
        return cls(
            id=source.id,
            user_id=source.user_id,
            name=source.name,
            source_type=source.source_type,
            pedagogical_role=source.pedagogical_role,
            filename=source.filename,
            document_type=source.document_type,
            file_size_bytes=source.file_size_bytes,
            source_url=source.source_url,
            note_content=source.note_content,
            status=source.status,
            error_message=source.error_message,
            content_summary=source.content_summary,
            word_count=source.word_count,
            chunk_count=source.chunk_count,
            is_vectorized=source.is_vectorized,
            tags=source.tags,
            usage_count=source.usage_count,
            last_used_at=source.last_used_at,
            created_at=source.created_at,
            updated_at=source.updated_at,
        )


class SourceListResponse(BaseModel):
    """Response listing sources"""
    sources: List[SourceResponse]
    total: int
    page: int
    page_size: int


class SourceSuggestion(BaseModel):
    """An AI-suggested source"""
    suggestion_type: SourceType
    title: str = Field(..., description="Suggested title/name")
    url: Optional[str] = Field(None, description="URL if web/youtube source")
    description: str = Field(..., description="Why this source is relevant")
    relevance_score: float = Field(..., ge=0, le=1)
    keywords: List[str] = Field(default_factory=list)


class SuggestSourcesResponse(BaseModel):
    """Response with AI source suggestions"""
    topic: str
    suggestions: List[SourceSuggestion]
    existing_relevant_sources: List[SourceResponse] = Field(
        default_factory=list,
        description="Existing sources in user's library that are relevant"
    )


class CourseSourceResponse(BaseModel):
    """Response with course-source link details"""
    id: str
    course_id: str
    source_id: str
    source: SourceResponse
    relevance_score: Optional[float]
    is_primary: bool
    added_at: datetime


class CourseSourcesResponse(BaseModel):
    """Response listing all sources for a course"""
    course_id: str
    sources: List[CourseSourceResponse]
    total: int


# =============================================================================
# Database table definitions (for SQL migrations)
# =============================================================================

SOURCES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    name VARCHAR(200) NOT NULL,
    source_type VARCHAR(20) NOT NULL,
    pedagogical_role VARCHAR(20) DEFAULT 'auto',
    filename VARCHAR(255),
    document_type VARCHAR(20),
    file_size_bytes BIGINT DEFAULT 0,
    file_path TEXT,
    source_url TEXT,
    note_content TEXT,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    raw_content TEXT,
    content_summary TEXT,
    word_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    is_vectorized BOOLEAN DEFAULT FALSE,
    extracted_metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,

    -- Indexes
    CONSTRAINT sources_user_id_idx UNIQUE (user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_sources_user_id ON sources(user_id);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);
CREATE INDEX IF NOT EXISTS idx_sources_source_type ON sources(source_type);
CREATE INDEX IF NOT EXISTS idx_sources_pedagogical_role ON sources(pedagogical_role);
CREATE INDEX IF NOT EXISTS idx_sources_tags ON sources USING GIN(tags);
"""

COURSE_SOURCES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS course_sources (
    id VARCHAR(36) PRIMARY KEY,
    course_id VARCHAR(36) NOT NULL,
    source_id VARCHAR(36) NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    user_id VARCHAR(36) NOT NULL,
    relevance_score FLOAT,
    is_primary BOOLEAN DEFAULT FALSE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique source per course
    CONSTRAINT unique_course_source UNIQUE (course_id, source_id)
);

CREATE INDEX IF NOT EXISTS idx_course_sources_course_id ON course_sources(course_id);
CREATE INDEX IF NOT EXISTS idx_course_sources_source_id ON course_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_course_sources_user_id ON course_sources(user_id);
"""

SOURCE_CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS source_chunks (
    id VARCHAR(36) PRIMARY KEY,
    source_id VARCHAR(36) NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    section_title VARCHAR(255),
    embedding vector(1536),
    embedding_model VARCHAR(50) DEFAULT 'text-embedding-3-small',
    token_count INTEGER DEFAULT 0,

    -- Ensure unique chunk per source
    CONSTRAINT unique_source_chunk UNIQUE (source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_source_chunks_source_id ON source_chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_source_chunks_embedding ON source_chunks USING ivfflat (embedding vector_cosine_ops);
"""
