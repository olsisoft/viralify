"""
Document Models for RAG System

Defines models for document upload, parsing, and retrieval.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    PPTX = "pptx"
    PPT = "ppt"
    TXT = "txt"
    MD = "md"
    XLSX = "xlsx"
    XLS = "xls"
    CSV = "csv"
    URL = "url"
    YOUTUBE = "youtube"


class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    SCANNING = "scanning"          # Security scan in progress
    SCAN_FAILED = "scan_failed"    # Security scan failed
    PARSING = "parsing"            # Content extraction in progress
    PARSE_FAILED = "parse_failed"  # Content extraction failed
    VECTORIZING = "vectorizing"    # Creating embeddings
    READY = "ready"                # Ready for RAG
    FAILED = "failed"              # General failure


class SecurityScanResult(BaseModel):
    """Result of security scan"""
    is_safe: bool = Field(..., description="Whether the document is safe")
    scan_timestamp: datetime = Field(default_factory=datetime.utcnow)
    threats_found: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Validation details
    mime_type_valid: bool = Field(default=True)
    extension_matches_content: bool = Field(default=True)
    no_macros: bool = Field(default=True)
    no_embedded_objects: bool = Field(default=True)
    file_size_ok: bool = Field(default=True)

    scan_details: Dict[str, Any] = Field(default_factory=dict)


class ExtractedImage(BaseModel):
    """An image extracted from a document (diagram, chart, etc.)"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(..., description="Parent document ID")

    # Image data
    file_path: str = Field(..., description="Path to saved image file")
    file_name: str = Field(..., description="Image filename")
    image_format: str = Field(default="png", description="Image format (png, jpg)")
    width: int = Field(default=0, description="Image width in pixels")
    height: int = Field(default=0, description="Image height in pixels")
    file_size_bytes: int = Field(default=0)

    # Context from document
    page_number: Optional[int] = Field(None, description="Page/slide number where found")
    context_text: Optional[str] = Field(None, description="Surrounding text context")
    caption: Optional[str] = Field(None, description="Caption if detected")

    # AI-generated metadata
    description: Optional[str] = Field(None, description="AI-generated image description")
    detected_type: str = Field(default="unknown", description="diagram, chart, screenshot, photo, etc.")
    relevance_keywords: List[str] = Field(default_factory=list, description="Keywords for matching")

    # For RAG visual matching
    embedding: Optional[List[float]] = Field(None, description="Text embedding of description")


class DocumentChunk(BaseModel):
    """A chunk of document content with embedding and semantic metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Text content of the chunk (enriched format)")
    chunk_index: int = Field(..., description="Index of chunk in document")

    # Position metadata
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    section_title: Optional[str] = Field(None, description="Section title if detected")

    # Embedding
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    embedding_model: str = Field(default="text-embedding-3-small")

    # Token count for context management
    token_count: int = Field(default=0)

    # Semantic metadata from SemanticChunker
    # Includes: content_type, is_key_content, contains_definition, contains_example,
    # contains_code, section_hierarchy, keywords, context_hint, timestamps, etc.
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Semantic metadata for better LLM understanding"
    )


class Document(BaseModel):
    """A document uploaded for RAG"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="Owner user ID")
    course_id: Optional[str] = Field(None, description="Associated course ID")

    # File info
    filename: str = Field(..., description="Original filename")
    document_type: DocumentType = Field(..., description="Document type")
    file_size_bytes: int = Field(default=0)
    file_path: Optional[str] = Field(None, description="Storage path")

    # URL info (for web sources)
    source_url: Optional[str] = Field(None, description="URL if web source")

    # Processing status
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    error_message: Optional[str] = Field(None)

    # Security scan
    security_scan: Optional[SecurityScanResult] = Field(None)

    # Extracted content
    raw_content: Optional[str] = Field(None, description="Extracted text content")
    content_summary: Optional[str] = Field(None, description="AI-generated summary")
    page_count: int = Field(default=0)
    word_count: int = Field(default=0)

    # Chunks for RAG
    chunks: List[DocumentChunk] = Field(default_factory=list)
    chunk_count: int = Field(default=0)

    # Extracted images (diagrams, charts, etc.)
    extracted_images: List[ExtractedImage] = Field(default_factory=list)
    image_count: int = Field(default=0)

    # Metadata extracted from document
    extracted_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = Field(None)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc-123",
                "user_id": "user-456",
                "filename": "python_guide.pdf",
                "document_type": "pdf",
                "file_size_bytes": 1024000,
                "status": "ready",
                "page_count": 25,
                "word_count": 15000,
                "chunk_count": 50,
            }
        }


class DocumentUploadRequest(BaseModel):
    """Request to upload a document"""
    user_id: str = Field(..., description="User ID")
    course_id: Optional[str] = Field(None, description="Course ID to associate with")

    # For URL-based sources
    source_url: Optional[str] = Field(None, description="URL to fetch content from")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user-123",
                "course_id": "course-456",
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    document_id: str
    filename: str
    document_type: DocumentType
    status: DocumentStatus
    message: str


class DocumentListResponse(BaseModel):
    """Response listing documents"""
    documents: List[Document]
    total: int
    page: int
    page_size: int


class RAGQueryRequest(BaseModel):
    """Request to query documents using RAG"""
    query: str = Field(..., description="Query text", min_length=3)
    document_ids: Optional[List[str]] = Field(None, description="Specific documents to search")
    course_id: Optional[str] = Field(None, description="Course ID to filter documents")
    user_id: str = Field(..., description="User ID for access control")

    # Search parameters
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    similarity_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum similarity score")

    # Context options
    include_metadata: bool = Field(default=True)
    max_tokens: int = Field(default=4000, description="Max tokens for context")


class RAGChunkResult(BaseModel):
    """A single RAG search result"""
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    similarity_score: float
    page_number: Optional[int]
    section_title: Optional[str]
    token_count: int


class RAGQueryResponse(BaseModel):
    """Response from RAG query"""
    query: str
    results: List[RAGChunkResult]
    total_results: int
    total_tokens: int

    # Combined context for LLM
    combined_context: str = Field(..., description="Combined context for LLM prompt")


# File size limits (in bytes)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_FILE_SIZE_BY_TYPE = {
    DocumentType.PDF: 50 * 1024 * 1024,
    DocumentType.DOCX: 25 * 1024 * 1024,
    DocumentType.PPTX: 100 * 1024 * 1024,
    DocumentType.XLSX: 10 * 1024 * 1024,
    DocumentType.TXT: 5 * 1024 * 1024,
    DocumentType.MD: 5 * 1024 * 1024,
    DocumentType.CSV: 10 * 1024 * 1024,
}

# Allowed MIME types
ALLOWED_MIME_TYPES = {
    "application/pdf": DocumentType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.DOCX,
    "application/msword": DocumentType.DOC,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentType.PPTX,
    "application/vnd.ms-powerpoint": DocumentType.PPT,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentType.XLSX,
    "application/vnd.ms-excel": DocumentType.XLS,
    "text/plain": DocumentType.TXT,
    "text/markdown": DocumentType.MD,
    "text/csv": DocumentType.CSV,
}

# Extension to type mapping
EXTENSION_TO_TYPE = {
    ".pdf": DocumentType.PDF,
    ".docx": DocumentType.DOCX,
    ".doc": DocumentType.DOC,
    ".pptx": DocumentType.PPTX,
    ".ppt": DocumentType.PPT,
    ".xlsx": DocumentType.XLSX,
    ".xls": DocumentType.XLS,
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MD,
    ".csv": DocumentType.CSV,
}
