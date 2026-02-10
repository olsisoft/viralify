"""
RAG Retrieval Service

Orchestrates document upload, processing, and retrieval for
Retrieval-Augmented Generation in course creation.
"""
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from openai import AsyncOpenAI

from models.document_models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    DocumentType,
    RAGChunkResult,
    RAGQueryRequest,
    RAGQueryResponse,
    EXTENSION_TO_TYPE,
)
from services.security_scanner import SecurityScanner
from services.document_parser import DocumentParser, WebContentParser
from services.vector_store import VectorizationService
from services.reranker import get_reranker, RerankerBase
from dataclasses import dataclass, field
import re


# =============================================================================
# WEIGHTED MULTI-SOURCE RAG - Models and Algorithm
# =============================================================================

@dataclass
class DocumentRelevanceScore:
    """Relevance score for a document relative to a topic."""
    document_id: str
    filename: str

    # Individual scores (0.0 - 1.0)
    semantic_similarity: float = 0.0      # Embedding-based similarity to topic
    keyword_coverage: float = 0.0         # % of topic keywords found in document
    freshness_score: float = 1.0          # Based on document date (newer = better)
    document_type_score: float = 1.0      # PDF/official > notes

    # Weighted final score
    final_score: float = 0.0

    # Metadata for traceability
    matched_keywords: List[str] = field(default_factory=list)
    document_type: str = "unknown"
    created_at: Optional[datetime] = None

    # Token allocation
    allocated_tokens: int = 0
    contribution_percentage: float = 0.0


@dataclass
class WeightedRAGResult:
    """Result of weighted multi-source RAG retrieval."""
    # Combined context from all relevant sources
    combined_context: str

    # Per-document breakdown for traceability
    document_scores: List[DocumentRelevanceScore]

    # Statistics
    total_documents_provided: int = 0
    documents_included: int = 0
    documents_excluded: int = 0
    total_tokens_used: int = 0

    # Source contribution map (for traceability)
    source_contributions: Dict[str, float] = field(default_factory=dict)


class WeightedMultiSourceRAG:
    """
    Weighted Multi-Source RAG Algorithm

    Ensures balanced, relevance-weighted content from multiple sources:
    1. Score each document for relevance to the topic
    2. Filter out irrelevant documents (below threshold)
    3. Allocate tokens proportionally based on scores
    4. Retrieve content respecting each document's budget
    5. Track contributions for traceability

    Scoring weights (configurable):
    - Semantic similarity: 40%
    - Keyword coverage: 30%
    - Document freshness: 15%
    - Document type: 15%
    """

    # Scoring weights
    WEIGHT_SEMANTIC = 0.40
    WEIGHT_KEYWORDS = 0.30
    WEIGHT_FRESHNESS = 0.15
    WEIGHT_DOC_TYPE = 0.15

    # Minimum relevance threshold (documents below this are excluded)
    MIN_RELEVANCE_THRESHOLD = 0.25

    # Minimum guaranteed allocation (% of budget) for included documents
    MIN_ALLOCATION_PERCENT = 0.10

    # Document type scores (official docs score higher)
    DOC_TYPE_SCORES = {
        "pdf": 1.0,        # Official documentation
        "docx": 0.9,       # Formal documents
        "pptx": 0.85,      # Presentations
        "xlsx": 0.8,       # Structured data
        "md": 0.75,        # Technical docs
        "txt": 0.7,        # Plain text
        "url": 0.65,       # Web content
        "youtube": 0.6,    # Video transcripts
        "unknown": 0.5,
    }

    def __init__(self, embedding_service=None, tokenizer=None):
        self.embedding_service = embedding_service
        self.tokenizer = tokenizer or self._get_default_tokenizer()

    def _get_default_tokenizer(self):
        """Get default tokenizer for token counting."""
        try:
            import tiktoken
            return tiktoken.encoding_for_model("gpt-4")
        except (ImportError, KeyError):
            return None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4  # Rough estimate

    async def score_documents(
        self,
        documents: List[Document],
        topic: str,
        description: Optional[str] = None,
    ) -> List[DocumentRelevanceScore]:
        """
        Score each document for relevance to the topic.

        Args:
            documents: List of Document objects to score
            topic: Course/content topic
            description: Optional detailed description

        Returns:
            List of DocumentRelevanceScore sorted by relevance (highest first)
        """
        query = f"{topic} {description or ''}".strip()
        query_keywords = self._extract_keywords(query)

        scores = []

        for doc in documents:
            if not doc.raw_content:
                continue

            score = DocumentRelevanceScore(
                document_id=doc.id,
                filename=doc.filename,
                document_type=doc.document_type.value if doc.document_type else "unknown",
                created_at=doc.created_at,
            )

            # 1. Semantic similarity (using embeddings if available)
            if self.embedding_service and doc.raw_content:
                try:
                    score.semantic_similarity = await self._compute_semantic_similarity(
                        query, doc.raw_content[:5000]  # First 5000 chars for efficiency
                    )
                except Exception as e:
                    print(f"[WEIGHTED_RAG] Semantic scoring failed for {doc.filename}: {e}", flush=True)
                    score.semantic_similarity = 0.5  # Default
            else:
                # Fallback to keyword-based pseudo-similarity
                score.semantic_similarity = self._keyword_similarity(query, doc.raw_content)

            # 2. Keyword coverage
            doc_text = doc.raw_content.lower()
            matched = [kw for kw in query_keywords if kw.lower() in doc_text]
            score.keyword_coverage = len(matched) / len(query_keywords) if query_keywords else 0
            score.matched_keywords = matched

            # 3. Freshness score (documents from last year score 1.0, older decay)
            if doc.created_at:
                days_old = (datetime.utcnow() - doc.created_at).days
                score.freshness_score = max(0.5, 1.0 - (days_old / 730))  # 2 year decay
            else:
                score.freshness_score = 0.7  # Unknown date

            # 4. Document type score
            doc_type = doc.document_type.value if doc.document_type else "unknown"
            score.document_type_score = self.DOC_TYPE_SCORES.get(doc_type, 0.5)

            # Calculate weighted final score
            score.final_score = (
                self.WEIGHT_SEMANTIC * score.semantic_similarity +
                self.WEIGHT_KEYWORDS * score.keyword_coverage +
                self.WEIGHT_FRESHNESS * score.freshness_score +
                self.WEIGHT_DOC_TYPE * score.document_type_score
            )

            scores.append(score)

            print(f"[WEIGHTED_RAG] {doc.filename}: "
                  f"semantic={score.semantic_similarity:.2f}, "
                  f"keywords={score.keyword_coverage:.2f}, "
                  f"fresh={score.freshness_score:.2f}, "
                  f"type={score.document_type_score:.2f} "
                  f"→ FINAL={score.final_score:.2f}", flush=True)

        # Sort by final score (highest first)
        scores.sort(key=lambda x: x.final_score, reverse=True)

        return scores

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant keywords from text."""
        # Remove common words and extract meaningful terms
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et',
            'en', 'est', 'que', 'qui', 'dans', 'ce', 'il', 'ne', 'sur',
            'se', 'pas', 'plus', 'par', 'pour', 'au', 'avec', 'son',
        }

        words = re.findall(r'\b[a-zA-Z\u00C0-\u017F]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]

        # Return unique keywords, preserving order
        seen = set()
        return [kw for kw in keywords if not (kw in seen or seen.add(kw))]

    def _keyword_similarity(self, query: str, document: str) -> float:
        """Compute keyword-based similarity (fallback when no embeddings)."""
        query_kw = set(self._extract_keywords(query))
        doc_kw = set(self._extract_keywords(document[:10000]))  # First 10k chars

        if not query_kw:
            return 0.5

        overlap = query_kw.intersection(doc_kw)
        return len(overlap) / len(query_kw)

    async def _compute_semantic_similarity(self, query: str, document: str) -> float:
        """Compute semantic similarity using embeddings."""
        if not self.embedding_service:
            return 0.5

        try:
            query_emb = await self.embedding_service.embed(query)
            doc_emb = await self.embedding_service.embed(document)

            # Cosine similarity
            import numpy as np
            similarity = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
            )
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.5

    def allocate_tokens(
        self,
        scores: List[DocumentRelevanceScore],
        total_budget: int,
    ) -> List[DocumentRelevanceScore]:
        """
        Allocate token budget to documents based on relevance scores.

        Documents below MIN_RELEVANCE_THRESHOLD are excluded.
        Included documents get proportional allocation with minimum guarantee.

        Args:
            scores: Scored documents (sorted by relevance)
            total_budget: Total tokens available

        Returns:
            Updated scores with token allocations
        """
        # Filter to relevant documents only
        relevant = [s for s in scores if s.final_score >= self.MIN_RELEVANCE_THRESHOLD]
        excluded = [s for s in scores if s.final_score < self.MIN_RELEVANCE_THRESHOLD]

        print(f"[WEIGHTED_RAG] Token allocation: {len(relevant)} relevant, "
              f"{len(excluded)} excluded (threshold={self.MIN_RELEVANCE_THRESHOLD})", flush=True)

        if not relevant:
            return scores

        # Calculate total score for normalization
        total_score = sum(s.final_score for s in relevant)

        # Allocate tokens proportionally
        for score in relevant:
            # Proportional allocation
            proportion = score.final_score / total_score

            # Ensure minimum allocation
            proportion = max(proportion, self.MIN_ALLOCATION_PERCENT)

            score.allocated_tokens = int(total_budget * proportion)
            score.contribution_percentage = proportion * 100

            print(f"[WEIGHTED_RAG] {score.filename}: "
                  f"{score.contribution_percentage:.1f}% → {score.allocated_tokens} tokens", flush=True)

        # Adjust to not exceed budget
        total_allocated = sum(s.allocated_tokens for s in relevant)
        if total_allocated > total_budget:
            scale = total_budget / total_allocated
            for score in relevant:
                score.allocated_tokens = int(score.allocated_tokens * scale)

        return scores

    async def retrieve_weighted_context(
        self,
        documents: List[Document],
        scores: List[DocumentRelevanceScore],
        topic: str,
        chunks_by_doc: Dict[str, List[DocumentChunk]] = None,
    ) -> WeightedRAGResult:
        """
        Retrieve context from each document respecting its token budget.

        Args:
            documents: Source documents
            scores: Scored and allocated documents
            topic: Topic for chunk relevance
            chunks_by_doc: Pre-computed chunks per document

        Returns:
            WeightedRAGResult with combined context and traceability
        """
        doc_map = {d.id: d for d in documents}
        context_parts = []
        source_contributions = {}

        included = [s for s in scores if s.allocated_tokens > 0]
        excluded = [s for s in scores if s.allocated_tokens == 0]

        for score in included:
            doc = doc_map.get(score.document_id)
            if not doc or not doc.raw_content:
                continue

            # Build document context within budget
            doc_content = self._extract_within_budget(
                doc.raw_content,
                score.allocated_tokens,
                doc.filename,
            )

            if doc_content:
                header = f"\n=== SOURCE: {doc.filename} (Relevance: {score.final_score:.0%}) ===\n"
                context_parts.append(header + doc_content)
                source_contributions[doc.filename] = score.contribution_percentage

        combined = "\n".join(context_parts)

        return WeightedRAGResult(
            combined_context=combined,
            document_scores=scores,
            total_documents_provided=len(documents),
            documents_included=len(included),
            documents_excluded=len(excluded),
            total_tokens_used=self.count_tokens(combined),
            source_contributions=source_contributions,
        )

    def _extract_within_budget(
        self,
        content: str,
        token_budget: int,
        filename: str,
    ) -> str:
        """Extract content within token budget, prioritizing beginning."""
        if not content:
            return ""

        current_tokens = self.count_tokens(content)

        if current_tokens <= token_budget:
            return content

        # Truncate to budget (rough estimate: 4 chars per token)
        char_budget = token_budget * 4
        truncated = content[:char_budget]

        # Try to end at sentence boundary
        last_period = truncated.rfind('.')
        if last_period > char_budget * 0.7:
            truncated = truncated[:last_period + 1]

        return truncated + f"\n[...truncated to {token_budget} tokens...]"


class DocumentStorage:
    """
    Simple file-based document storage.
    In production, use S3 or similar object storage.
    """

    def __init__(self, base_path: str = "/tmp/viralify/documents"):
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
        document_id: str,
        filename: str,
    ) -> str:
        """Save file to storage"""
        user_path = self.get_user_path(user_id)
        file_path = user_path / f"{document_id}_{filename}"

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


class DocumentRepository:
    """
    Simple in-memory document repository.
    In production, use PostgreSQL or similar database.
    """

    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.user_documents: Dict[str, List[str]] = {}  # user_id -> [doc_ids]
        self.course_documents: Dict[str, List[str]] = {}  # course_id -> [doc_ids]

    async def save(self, document: Document) -> None:
        """Save document to repository"""
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

    async def get(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(document_id)

    async def get_by_user(self, user_id: str) -> List[Document]:
        """Get all documents for a user"""
        doc_ids = self.user_documents.get(user_id, [])
        return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]

    async def get_by_course(self, course_id: str) -> List[Document]:
        """Get all documents for a course"""
        doc_ids = self.course_documents.get(course_id, [])
        return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]

    async def delete(self, document_id: str) -> None:
        """Delete document"""
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
        status: DocumentStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Update document status"""
        doc = self.documents.get(document_id)
        if doc:
            doc.status = status
            if error_message:
                doc.error_message = error_message
            if status == DocumentStatus.READY:
                doc.processed_at = datetime.utcnow()


class RAGService:
    """
    Main RAG service orchestrating document processing and retrieval.

    Features:
    - Multi-format document parsing (PDF, DOCX, PPTX, etc.)
    - Vector search with embedding similarity
    - Cross-Encoder re-ranking for improved precision
    - Token-aware context building
    """

    # Token limits for different contexts
    MAX_CONTEXT_TOKENS = 8000  # Max tokens for RAG context in prompts
    MAX_PROMPT_TOKENS = 100000  # Safety limit for total prompt size

    # Re-ranking configuration
    RERANK_TOP_K = 30  # Get more candidates for re-ranking
    RERANK_FINAL_K = 15  # Return top results after re-ranking

    def __init__(
        self,
        vector_backend: str = "memory",
        storage_path: str = "/tmp/viralify/documents",
        reranker_backend: str = "auto",
    ):
        self.security_scanner = SecurityScanner()
        self.document_parser = DocumentParser()
        self.web_parser = WebContentParser()
        self.vectorization = VectorizationService(vector_backend=vector_backend)
        self.storage = DocumentStorage(base_path=storage_path)
        self.repository = DocumentRepository()

        # Initialize Cross-Encoder reranker
        self.reranker: Optional[RerankerBase] = None
        self.reranker_backend = reranker_backend
        self._init_reranker()

        # OpenAI client for context synthesis
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0,
        )

        # Initialize tokenizer for GPT-4
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Store last weighted RAG result for traceability
        self._last_weighted_result: Optional[WeightedRAGResult] = None

        print(f"[RAG] Service initialized (vector: {vector_backend}, reranker: {reranker_backend})", flush=True)

    def _init_reranker(self):
        """Initialize the reranker (lazy loading)"""
        try:
            self.reranker = get_reranker(self.reranker_backend)
            print(f"[RAG] Reranker ready: {self.reranker.__class__.__name__}", flush=True)
        except Exception as e:
            print(f"[RAG] Reranker initialization deferred: {e}", flush=True)
            self.reranker = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if not text:
            return ""

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        # Truncate and add indicator
        truncated_tokens = tokens[:max_tokens - 20]  # Leave room for truncation message
        truncated_text = self.tokenizer.decode(truncated_tokens)
        return truncated_text + "\n\n[... content truncated due to length ...]"

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        course_id: Optional[str] = None,
    ) -> Document:
        """
        Upload and process a document.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            user_id: Owner user ID
            course_id: Optional course to associate with

        Returns:
            Processed Document object
        """
        print(f"[RAG] Uploading document: {filename} for user {user_id}", flush=True)

        # Determine document type from extension
        ext = Path(filename).suffix.lower()
        if ext not in EXTENSION_TO_TYPE:
            raise ValueError(f"Unsupported file extension: {ext}")

        document_type = EXTENSION_TO_TYPE[ext]

        # Create document record
        document = Document(
            id=str(uuid.uuid4()),
            user_id=user_id,
            course_id=course_id,
            filename=self.security_scanner.sanitize_filename(filename),
            document_type=document_type,
            file_size_bytes=len(file_content),
            status=DocumentStatus.SCANNING,
        )

        await self.repository.save(document)

        try:
            # 1. Security scan
            print(f"[RAG] Security scanning: {document.id}", flush=True)
            scan_result = await self.security_scanner.scan_file(
                file_content,
                filename,
                document_type,
            )

            document.security_scan = scan_result

            if not scan_result.is_safe:
                document.status = DocumentStatus.SCAN_FAILED
                document.error_message = f"Security scan failed: {', '.join(scan_result.threats_found)}"
                await self.repository.save(document)
                raise ValueError(document.error_message)

            # 2. Save file to storage
            file_path = await self.storage.save_file(
                file_content,
                user_id,
                document.id,
                document.filename,
            )
            document.file_path = file_path

            # 3. Parse document
            document.status = DocumentStatus.PARSING
            await self.repository.save(document)

            print(f"[RAG] Parsing document: {document.id}", flush=True)
            raw_text, chunks, metadata = await self.document_parser.parse_document(
                file_content,
                filename,
                document_type,
            )

            document.raw_content = raw_text
            document.extracted_metadata = metadata
            document.page_count = metadata.get("page_count", 0)
            document.word_count = len(raw_text.split())

            # 3.5. Extract images (diagrams, charts) for visual RAG
            if metadata.get("has_images", False) and document_type in [
                DocumentType.PDF, DocumentType.PPTX, DocumentType.DOCX
            ]:
                print(f"[RAG] Extracting images from: {document.id}", flush=True)
                images_output_dir = str(self.storage.get_user_path(user_id) / "images")
                try:
                    extracted_images = await self.document_parser.extract_images(
                        file_content,
                        filename,
                        document_type,
                        document.id,
                        images_output_dir,
                    )
                    document.extracted_images = extracted_images
                    document.image_count = len(extracted_images)
                    print(f"[RAG] Extracted {len(extracted_images)} images", flush=True)
                except Exception as img_error:
                    print(f"[RAG] Image extraction warning: {img_error}", flush=True)
                    # Continue without images - not critical

            # 4. Vectorize chunks
            document.status = DocumentStatus.VECTORIZING
            await self.repository.save(document)

            print(f"[RAG] Vectorizing {len(chunks)} chunks: {document.id}", flush=True)
            vectorized_chunks = await self.vectorization.vectorize_chunks(
                chunks,
                document.id,
                user_id,
            )

            document.chunks = vectorized_chunks
            document.chunk_count = len(vectorized_chunks)

            # 5. Generate summary
            document.content_summary = await self._generate_summary(raw_text[:4000])

            # Mark as ready
            document.status = DocumentStatus.READY
            document.processed_at = datetime.utcnow()
            await self.repository.save(document)

            print(f"[RAG] Document ready: {document.id}", flush=True)

            return document

        except Exception as e:
            print(f"[RAG] Processing failed: {e}", flush=True)
            if document.status != DocumentStatus.SCAN_FAILED:
                document.status = DocumentStatus.FAILED
                document.error_message = str(e)
            await self.repository.save(document)
            raise

    async def upload_from_url(
        self,
        url: str,
        user_id: str,
        course_id: Optional[str] = None,
    ) -> Document:
        """
        Upload document from URL.

        Args:
            url: URL to fetch content from
            user_id: Owner user ID
            course_id: Optional course to associate with

        Returns:
            Processed Document object
        """
        print(f"[RAG] Fetching URL: {url} for user {user_id}", flush=True)

        # Determine document type
        if "youtube.com" in url or "youtu.be" in url:
            document_type = DocumentType.YOUTUBE
            raw_text, metadata = await self.web_parser.parse_youtube(url)
            filename = f"youtube_{metadata.get('video_id', 'video')}.txt"
        else:
            document_type = DocumentType.URL
            raw_text, metadata = await self.web_parser.parse_url(url)
            filename = f"url_{metadata.get('title', 'page')[:50]}.txt"

        # Create document record
        document = Document(
            id=str(uuid.uuid4()),
            user_id=user_id,
            course_id=course_id,
            filename=filename,
            document_type=document_type,
            source_url=url,
            status=DocumentStatus.PARSING,
        )

        await self.repository.save(document)

        try:
            document.raw_content = raw_text
            document.extracted_metadata = metadata
            document.word_count = len(raw_text.split())

            # Create chunks
            chunks = self.document_parser._create_chunks(raw_text, metadata)

            # Vectorize
            document.status = DocumentStatus.VECTORIZING
            await self.repository.save(document)

            vectorized_chunks = await self.vectorization.vectorize_chunks(
                chunks,
                document.id,
                user_id,
            )

            document.chunks = vectorized_chunks
            document.chunk_count = len(vectorized_chunks)

            # Generate summary
            document.content_summary = await self._generate_summary(raw_text[:4000])

            # Mark as ready
            document.status = DocumentStatus.READY
            document.processed_at = datetime.utcnow()
            await self.repository.save(document)

            print(f"[RAG] URL document ready: {document.id}", flush=True)

            return document

        except Exception as e:
            print(f"[RAG] URL processing failed: {e}", flush=True)
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            await self.repository.save(document)
            raise

    async def query(self, request: RAGQueryRequest) -> RAGQueryResponse:
        """
        Query documents using RAG with Cross-Encoder re-ranking.

        Pipeline:
        1. Vector search (bi-encoder) - fast, retrieves candidates
        2. Cross-Encoder re-ranking - accurate, filters noise
        3. Context building - token-aware aggregation

        Args:
            request: RAG query request

        Returns:
            RAGQueryResponse with relevant chunks and combined context
        """
        print(f"[RAG] Query: {request.query[:100]}...", flush=True)

        # Get document IDs to search
        document_ids = request.document_ids

        if not document_ids and request.course_id:
            # Get all documents for the course
            docs = await self.repository.get_by_course(request.course_id)
            document_ids = [d.id for d in docs if d.status == DocumentStatus.READY]

        if not document_ids:
            # Get all user documents
            docs = await self.repository.get_by_user(request.user_id)
            document_ids = [d.id for d in docs if d.status == DocumentStatus.READY]

        # STEP 1: Vector search (bi-encoder) - get more candidates for re-ranking
        # We fetch more results than needed because re-ranking will filter noise
        vector_top_k = max(request.top_k, self.RERANK_TOP_K)

        results = await self.vectorization.search(
            query=request.query,
            user_id=request.user_id,
            document_ids=document_ids,
            top_k=vector_top_k,
            similarity_threshold=request.similarity_threshold,
        )

        print(f"[RAG] Vector search returned {len(results)} candidates", flush=True)

        # Enrich results with document names
        for result in results:
            doc = await self.repository.get(result.document_id)
            if doc:
                result.document_name = doc.filename

        # STEP 2: Cross-Encoder re-ranking
        if results and self.reranker is not None:
            results = await self._rerank_results(request.query, results, request.top_k)

        # STEP 3: Build combined context respecting token limit
        combined_context = self._build_context(results, request.max_tokens)

        # Calculate total tokens
        total_tokens = sum(r.token_count for r in results)

        response = RAGQueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            total_tokens=total_tokens,
            combined_context=combined_context,
        )

        print(f"[RAG] Returning {len(results)} re-ranked chunks", flush=True)

        return response

    async def _rerank_results(
        self,
        query: str,
        results: List[RAGChunkResult],
        top_k: int,
    ) -> List[RAGChunkResult]:
        """
        Re-rank results using Cross-Encoder for improved precision.

        Cross-encoders process query and document together, providing
        more accurate relevance scoring than bi-encoder similarity.

        Args:
            query: Original search query
            results: Results from vector search
            top_k: Number of results to return after re-ranking

        Returns:
            Re-ranked results sorted by relevance
        """
        if not results:
            return results

        try:
            print(f"[RAG] Re-ranking {len(results)} results with CrossEncoder...", flush=True)

            # Extract document contents for re-ranking
            documents = [r.content for r in results]

            # Get re-ranked scores
            reranked = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
            )

            # Build re-ordered results list
            reranked_results = []
            for original_idx, rerank_score in reranked:
                result = results[original_idx]
                # Store both scores for debugging/analysis
                result.rerank_score = rerank_score
                reranked_results.append(result)

            print(f"[RAG] Re-ranking complete. Top score: {reranked[0][1]:.3f} -> {reranked[-1][1]:.3f}", flush=True)

            return reranked_results

        except Exception as e:
            print(f"[RAG] Re-ranking failed, using original order: {e}", flush=True)
            # Fallback to original results
            return results[:top_k]

    async def get_document(self, document_id: str, user_id: str) -> Optional[Document]:
        """Get document by ID with access control"""
        doc = await self.repository.get(document_id)
        if doc and doc.user_id == user_id:
            return doc
        return None

    async def list_documents(
        self,
        user_id: str,
        course_id: Optional[str] = None,
    ) -> List[Document]:
        """List documents for user or course"""
        if course_id:
            docs = await self.repository.get_by_course(course_id)
        else:
            docs = await self.repository.get_by_user(user_id)

        # Filter by user for security
        return [d for d in docs if d.user_id == user_id]

    async def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete document with access control"""
        doc = await self.repository.get(document_id)

        if not doc or doc.user_id != user_id:
            return False

        # Delete from vector store
        await self.vectorization.delete_document(document_id)

        # Delete file
        if doc.file_path:
            await self.storage.delete_file(doc.file_path)

        # Delete from repository
        await self.repository.delete(document_id)

        print(f"[RAG] Deleted document: {document_id}", flush=True)

        return True

    def _build_context(
        self,
        results: List[RAGChunkResult],
        max_tokens: int,
    ) -> str:
        """
        Build combined context from search results.

        Prioritizes key content (definitions, examples) and sorts by relevance.
        Uses enriched chunk format from SemanticChunker for better LLM understanding.
        """
        if not results:
            return ""

        # Sort results to prioritize key content
        # 1. First by similarity score (relevance)
        # 2. Then boost key content (definitions, examples, etc.)
        sorted_results = self._prioritize_chunks(results)

        context_parts = []
        current_tokens = 0

        for result in sorted_results:
            # The content is already in enriched format from SemanticChunker
            # It includes source info, content type, semantic markers, etc.
            chunk_text = result.content

            # Count actual tokens
            chunk_tokens = self.count_tokens(chunk_text)

            if current_tokens + chunk_tokens > max_tokens:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only add if we can fit meaningful content
                    truncated = self.truncate_to_tokens(chunk_text, remaining_tokens)
                    context_parts.append(truncated)
                break

            context_parts.append(chunk_text)
            current_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    def _prioritize_chunks(self, results: List[RAGChunkResult]) -> List[RAGChunkResult]:
        """
        Prioritize chunks based on content importance.

        Boosts scores for:
        - Key content (definitions, important concepts)
        - Examples
        - Content with images
        """
        scored_results = []

        for result in results:
            # Start with similarity score
            priority_score = result.similarity_score

            # Check for key content markers in the enriched content
            content_lower = result.content.lower()

            # Boost definitions
            if '[contains: definition' in content_lower or 'key concept' in content_lower:
                priority_score += 0.15

            # Boost examples
            if '[contains: example' in content_lower or 'contains: example' in content_lower:
                priority_score += 0.10

            # Boost content with images
            if '[associated visuals:' in content_lower:
                priority_score += 0.05

            # Boost code examples for technical content
            if '[content type: code' in content_lower or 'contains: code' in content_lower:
                priority_score += 0.05

            scored_results.append((result, priority_score))

        # Sort by priority score (highest first)
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [r[0] for r in scored_results]

    async def _generate_summary(self, text: str) -> str:
        """Generate AI summary of document content"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Generate a brief summary (2-3 sentences) of the following document content. Focus on the main topics and key points.",
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[RAG] Summary generation failed: {e}", flush=True)
            return ""

    async def _generate_structure_with_ai(
        self,
        doc: Document,
        is_youtube: bool,
    ) -> str:
        """
        Use AI to generate structure for documents without explicit headings.

        This is especially useful for YouTube videos without chapters or
        documents that are unstructured text.
        """
        if not doc.raw_content:
            return ""

        # Take first 4000 chars for analysis (enough to understand structure)
        content_sample = doc.raw_content[:4000]

        if is_youtube:
            source_type = "vidéo YouTube"
            instruction = """Analyse cette transcription YouTube et identifie les principaux sujets/sections abordés.
Retourne UNIQUEMENT une liste de sections au format:
┌── [Sujet 1]
┌── [Sujet 2]
...

Identifie 3-7 sections principales basées sur les changements de sujet dans la transcription.
Ne retourne RIEN d'autre que la liste des sections."""
        else:
            source_type = "document"
            instruction = """Analyse ce document et identifie sa structure (chapitres, sections principales).
Retourne UNIQUEMENT une liste de sections au format:
┌── [Section 1]
   ├── [Sous-section 1.1]
┌── [Section 2]
...

Identifie la structure logique du document. Ne retourne RIEN d'autre que la liste."""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"Tu es un expert en analyse de contenu. Tu analyses un {source_type} pour en extraire la structure."
                    },
                    {
                        "role": "user",
                        "content": f"{instruction}\n\nCONTENU À ANALYSER:\n{content_sample}"
                    }
                ],
                max_tokens=500,
                temperature=0.3,
            )

            ai_structure = response.choices[0].message.content.strip()

            if ai_structure:
                if is_youtube:
                    title = doc.extracted_metadata.get('title', doc.filename)
                    header = f"\n🎬 VIDEO YOUTUBE: {title}\n   STRUCTURE DÉTECTÉE PAR IA (basée sur le contenu):"
                else:
                    header = f"\n📄 DOCUMENT: {doc.filename}\n   STRUCTURE DÉTECTÉE PAR IA:"

                # Indent the AI structure
                indented = "\n".join(f"   {line}" for line in ai_structure.split('\n'))
                result = f"{header}\n{indented}"

                # Add summary if available
                if doc.content_summary:
                    result += f"\n\n   SUMMARY: {doc.content_summary}"

                # Add stats
                word_count = doc.word_count or len(doc.raw_content.split())
                if is_youtube:
                    result += f"\n   STATS: {word_count} mots dans la transcription"
                else:
                    result += f"\n   STATS: {doc.page_count} pages, {word_count} words"

                print(f"[RAG] AI-generated structure for {doc.filename}: {len(ai_structure.split(chr(10)))} sections", flush=True)
                return result

        except Exception as e:
            print(f"[RAG] AI structure generation failed for {doc.filename}: {e}", flush=True)

        return ""

    async def _extract_document_structure(
        self,
        document_ids: List[str],
        user_id: str,
    ) -> str:
        """
        Extract document structure (headings, table of contents) from documents.

        This creates an explicit outline that the LLM must follow for course structure.
        Handles PDFs, DOCX, and YouTube videos with chapters.

        Returns:
            Formatted structure string showing document organization
        """
        structure_parts = []
        docs_without_structure = []

        for doc_id in document_ids:
            doc = await self.repository.get(doc_id)
            if not doc or doc.user_id != user_id or doc.status != DocumentStatus.READY:
                continue

            # Check if this is a YouTube video
            is_youtube = (
                doc.document_type == DocumentType.URL and
                doc.source_url and
                ('youtube.com' in doc.source_url or 'youtu.be' in doc.source_url)
            )

            if is_youtube:
                doc_structure = [f"\n🎬 VIDEO YOUTUBE: {doc.extracted_metadata.get('title', doc.filename)}"]
                duration = doc.extracted_metadata.get('duration_seconds', 0)
                if duration:
                    minutes = duration // 60
                    doc_structure.append(f"   DURÉE: {minutes} minutes")
            else:
                doc_structure = [f"\n📄 DOCUMENT: {doc.filename}"]

            # Extract headings from metadata
            headings = doc.extracted_metadata.get("headings", [])

            if headings:
                if is_youtube:
                    doc_structure.append("   CHAPITRES YOUTUBE (structure obligatoire):")
                    for heading in headings:
                        text = heading.get("text", "").strip()
                        start_time = heading.get("start_time", 0)
                        if text:
                            # Format timestamp
                            mins = int(start_time // 60)
                            secs = int(start_time % 60)
                            timestamp = f"{mins:02d}:{secs:02d}"
                            doc_structure.append(f"   ┌── [{timestamp}] {text}")
                else:
                    doc_structure.append("   TABLE OF CONTENTS:")
                    for heading in headings:
                        level = heading.get("level", 1)
                        text = heading.get("text", "").strip()
                        if text:
                            indent = "   " * level
                            prefix = "├──" if level > 1 else "┌──"
                            doc_structure.append(f"   {indent}{prefix} {text}")
            else:
                # If no headings, try to extract structure from content
                detected_sections = []

                if doc.raw_content:
                    # Look for markdown-style headers or numbered sections
                    lines = doc.raw_content.split('\n')
                    for line in lines[:100]:  # Check first 100 lines
                        line = line.strip()
                        # Markdown headers
                        if line.startswith('#'):
                            level = len(line.split()[0])  # Count #s
                            text = line.lstrip('#').strip()
                            if text and len(text) > 3:
                                detected_sections.append((level, text))
                        # Numbered sections like "1. Introduction" or "1.1 Overview"
                        elif re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
                            # Count dots for level
                            match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)', line)
                            if match:
                                level = match.group(1).count('.') + 1
                                text = match.group(2).strip()
                                if len(text) > 3:
                                    detected_sections.append((level, text))
                        # ALL CAPS lines (often section titles)
                        elif line.isupper() and len(line) > 5 and len(line) < 80:
                            detected_sections.append((1, line.title()))

                if detected_sections:
                    doc_structure.append("   DETECTED SECTIONS:")
                    for level, text in detected_sections[:20]:  # Limit to 20 sections
                        indent = "   " * level
                        prefix = "├──" if level > 1 else "┌──"
                        doc_structure.append(f"   {indent}{prefix} {text}")
                else:
                    # No structure detected - flag for AI analysis
                    docs_without_structure.append((doc_id, doc, is_youtube))

            # Add content summary if available
            if doc.content_summary:
                doc_structure.append(f"\n   SUMMARY: {doc.content_summary}")

            # Add document stats
            if is_youtube:
                word_count = doc.word_count or len((doc.raw_content or '').split())
                doc_structure.append(f"   STATS: {word_count} mots dans la transcription")
            else:
                doc_structure.append(f"   STATS: {doc.page_count} pages, {doc.word_count} words")

            structure_parts.append("\n".join(doc_structure))

        # For documents without structure, use AI to generate structure
        if docs_without_structure:
            for doc_id, doc, is_youtube in docs_without_structure:
                ai_structure = await self._generate_structure_with_ai(doc, is_youtube)
                if ai_structure:
                    structure_parts.append(ai_structure)

        if not structure_parts:
            return ""

        header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           DOCUMENT STRUCTURE - YOUR COURSE MUST FOLLOW THIS OUTLINE          ║
╚══════════════════════════════════════════════════════════════════════════════╝

The following is the EXACT structure extracted from the source documents.
Your course sections and lectures MUST map directly to this structure.

DO NOT invent new topics. DO NOT reorganize. FOLLOW THIS STRUCTURE.
"""
        return header + "\n".join(structure_parts)

    async def get_context_for_course_generation(
        self,
        topic: str,
        description: Optional[str],
        document_ids: List[str],
        user_id: str,
        max_tokens: int = 8000,  # Increased for deeper RAG integration
        use_weighted: bool = True,  # Use weighted multi-source algorithm
    ) -> str:
        """
        Get relevant context from documents for course generation.

        This is the main integration point with the course generator.
        For multiple documents, uses Weighted Multi-Source RAG for balanced coverage.

        Args:
            topic: Course topic
            description: Course description
            document_ids: Documents to search
            user_id: User ID
            max_tokens: Maximum context tokens
            use_weighted: Use weighted multi-source algorithm (recommended for multiple docs)

        Returns:
            Combined context string for course generation
        """
        # Enforce maximum context size to prevent API token errors
        effective_max_tokens = min(max_tokens, self.MAX_CONTEXT_TOKENS)

        print(f"[RAG] Getting context for course: {topic[:50]}... (max {effective_max_tokens} tokens)", flush=True)
        print(f"[RAG] Searching in {len(document_ids)} documents", flush=True)

        # CRITICAL: Extract document structure FIRST - this guides course structure
        document_structure = await self._extract_document_structure(document_ids, user_id)
        structure_tokens = self.count_tokens(document_structure)
        print(f"[RAG] Extracted document structure: {structure_tokens} tokens", flush=True)

        # Reserve tokens for structure (it's more important than content for course planning)
        content_max_tokens = effective_max_tokens - structure_tokens - 100  # 100 token buffer

        # Use weighted approach for multiple documents (better balance and traceability)
        if use_weighted and len(document_ids) > 1:
            print(f"[RAG] Using WEIGHTED multi-source algorithm for {len(document_ids)} documents", flush=True)
            weighted_result = await self.get_weighted_context_for_course_generation(
                topic=topic,
                description=description,
                document_ids=document_ids,
                user_id=user_id,
                max_tokens=content_max_tokens,  # Reserve space for structure
            )
            # Store the weighted result for traceability (accessible via get_last_weighted_result)
            self._last_weighted_result = weighted_result
            # Prepend document structure to guide course organization
            if document_structure:
                return document_structure + "\n\n" + weighted_result.combined_context
            return weighted_result.combined_context

        # Single document or weighted disabled: use simpler approach
        print(f"[RAG] Repository has {len(self.repository.documents)} documents in memory", flush=True)

        # STRATEGY 1: Get ALL raw content from documents (for comprehensive coverage)
        all_document_content = []
        total_raw_tokens = 0

        for doc_id in document_ids:
            doc = await self.repository.get(doc_id)
            print(f"[RAG] Document {doc_id}: found={doc is not None}", flush=True)
            if doc:
                print(f"[RAG]   - user_id match: {doc.user_id == user_id} (doc: {doc.user_id}, request: {user_id})", flush=True)
                print(f"[RAG]   - status: {doc.status}", flush=True)
                print(f"[RAG]   - has raw_content: {bool(doc.raw_content)}", flush=True)
            if doc and doc.user_id == user_id and doc.status == DocumentStatus.READY:
                if doc.raw_content:
                    doc_header = f"=== DOCUMENT: {doc.filename} ===\n"
                    if doc.content_summary:
                        doc_header += f"Summary: {doc.content_summary}\n\n"

                    doc_content = doc_header + doc.raw_content
                    doc_tokens = self.count_tokens(doc_content)

                    all_document_content.append({
                        "content": doc_content,
                        "tokens": doc_tokens,
                        "filename": doc.filename
                    })
                    total_raw_tokens += doc_tokens
                    print(f"[RAG] Document {doc.filename}: {doc_tokens} tokens", flush=True)

        # If total content fits, use all of it
        if total_raw_tokens <= content_max_tokens:
            print(f"[RAG] Using FULL document content ({total_raw_tokens} tokens)", flush=True)
            content = "\n\n".join([d["content"] for d in all_document_content])
            # Prepend document structure to guide course organization
            if document_structure:
                return document_structure + "\n\n" + content
            return content

        # STRATEGY 2: If too large, combine semantic search with document excerpts
        print(f"[RAG] Content too large ({total_raw_tokens} tokens), using hybrid approach", flush=True)

        # Build search query from topic and description
        query = f"{topic}"
        if description:
            query += f" {description}"

        request = RAGQueryRequest(
            query=query,
            document_ids=document_ids,
            user_id=user_id,
            top_k=30,  # Increased: Get more chunks for comprehensive coverage
            similarity_threshold=0.3,  # Lowered: Include more potentially relevant content
            max_tokens=effective_max_tokens,
        )

        response = await self.query(request)

        # Build combined context with document structure
        context_parts = []
        current_tokens = 0

        # First, add document summaries for context
        summaries_section = "=== DOCUMENT SUMMARIES ===\n"
        for doc_data in all_document_content:
            doc_id = document_ids[all_document_content.index(doc_data)]
            doc = await self.repository.get(doc_id)
            if doc and doc.content_summary:
                summaries_section += f"- {doc.filename}: {doc.content_summary}\n"

        summaries_tokens = self.count_tokens(summaries_section)
        if summaries_tokens < effective_max_tokens * 0.1:  # Reserve 10% for summaries
            context_parts.append(summaries_section)
            current_tokens += summaries_tokens

        # Add relevant chunks from search
        context_parts.append("\n=== RELEVANT CONTENT FROM DOCUMENTS ===\n")
        context_parts.append(response.combined_context)

        combined = "\n".join(context_parts)
        final_tokens = self.count_tokens(combined)

        if final_tokens > effective_max_tokens:
            print(f"[RAG] Final context too large ({final_tokens} tokens), truncating", flush=True)
            combined = self.truncate_to_tokens(combined, effective_max_tokens)

        print(f"[RAG] Returning context: {self.count_tokens(combined)} tokens", flush=True)

        # Prepend document structure to guide course organization
        if document_structure:
            combined = document_structure + "\n\n" + combined
            print(f"[RAG] Added document structure, total: {self.count_tokens(combined)} tokens", flush=True)

        return combined

    def get_last_weighted_result(self) -> Optional[WeightedRAGResult]:
        """
        Get the last weighted RAG result for traceability.

        Returns the WeightedRAGResult from the most recent
        get_context_for_course_generation call that used weighted retrieval.
        """
        return getattr(self, '_last_weighted_result', None)

    async def get_weighted_context_for_course_generation(
        self,
        topic: str,
        description: Optional[str],
        document_ids: List[str],
        user_id: str,
        max_tokens: int = 8000,
    ) -> WeightedRAGResult:
        """
        Get weighted, balanced context from multiple documents for course generation.

        This method implements the Weighted Multi-Source RAG algorithm:
        1. Score each document for relevance to the topic
        2. Filter out irrelevant documents (noise reduction)
        3. Allocate tokens proportionally based on scores
        4. Retrieve content respecting each document's budget
        5. Return combined context with traceability info

        Args:
            topic: Course topic
            description: Course description
            document_ids: Documents to use as sources
            user_id: User ID
            max_tokens: Maximum context tokens

        Returns:
            WeightedRAGResult with balanced context and source contributions
        """
        print(f"[WEIGHTED_RAG] Starting weighted retrieval for: {topic[:50]}...", flush=True)
        print(f"[WEIGHTED_RAG] Documents: {len(document_ids)}, Budget: {max_tokens} tokens", flush=True)

        # Load documents
        documents = []
        for doc_id in document_ids:
            doc = await self.repository.get(doc_id)
            if doc and doc.user_id == user_id and doc.status == DocumentStatus.READY:
                if doc.raw_content:
                    documents.append(doc)
                    print(f"[WEIGHTED_RAG] Loaded: {doc.filename} ({len(doc.raw_content)} chars)", flush=True)
                else:
                    print(f"[WEIGHTED_RAG] Skipped {doc.filename}: no raw_content", flush=True)

        if not documents:
            print("[WEIGHTED_RAG] No documents loaded, returning empty result", flush=True)
            return WeightedRAGResult(
                combined_context="",
                document_scores=[],
                total_documents_provided=len(document_ids),
                documents_included=0,
                documents_excluded=len(document_ids),
                total_tokens_used=0,
                source_contributions={},
            )

        # Initialize weighted RAG with embedding service for semantic scoring
        weighted_rag = WeightedMultiSourceRAG(
            embedding_service=self.vectorization.embedding_service,
            tokenizer=self.tokenizer,
        )

        # Step 1: Score documents for relevance
        print("[WEIGHTED_RAG] Step 1: Scoring documents...", flush=True)
        scores = await weighted_rag.score_documents(documents, topic, description)

        # Step 2: Allocate token budget
        print("[WEIGHTED_RAG] Step 2: Allocating token budget...", flush=True)
        scores = weighted_rag.allocate_tokens(scores, max_tokens)

        # Step 3: Retrieve weighted context
        print("[WEIGHTED_RAG] Step 3: Retrieving weighted context...", flush=True)
        result = await weighted_rag.retrieve_weighted_context(
            documents=documents,
            scores=scores,
            topic=topic,
        )

        # Summary
        print(f"[WEIGHTED_RAG] Result summary:", flush=True)
        print(f"  - Documents provided: {result.total_documents_provided}", flush=True)
        print(f"  - Documents included: {result.documents_included}", flush=True)
        print(f"  - Documents excluded: {result.documents_excluded}", flush=True)
        print(f"  - Total tokens used: {result.total_tokens_used}", flush=True)
        for filename, contrib in result.source_contributions.items():
            print(f"  - {filename}: {contrib:.1f}%", flush=True)

        return result

    async def get_images_for_topic(
        self,
        topic: str,
        document_ids: List[str],
        user_id: str,
        image_types: List[str] = None,
        max_images: int = 5,
        min_relevance: float = 0.3,
    ) -> List[dict]:
        """
        Get relevant images from documents for a specific topic.

        Used for diagram slides when RAG images are available.

        Args:
            topic: Topic to match images against
            document_ids: Documents to search
            user_id: User ID for access control
            image_types: Filter by image types (diagram, chart, screenshot, etc.)
            max_images: Maximum number of images to return
            min_relevance: Minimum relevance score (0-1) to include an image

        Returns:
            List of image dictionaries with path, description, and relevance
        """
        print(f"[RAG_IMAGES] Searching images for topic: {topic[:50]}...", flush=True)

        matching_images = []
        topic_lower = topic.lower()
        # Remove common stopwords for better matching
        stopwords = {'the', 'a', 'an', 'is', 'are', 'of', 'and', 'or', 'to', 'in', 'for', 'with', 'on', 'at', 'by'}
        topic_words = set(w for w in topic_lower.split() if w not in stopwords and len(w) > 2)

        if not topic_words:
            print(f"[RAG_IMAGES] No meaningful words in topic, skipping", flush=True)
            return []

        # Collect images from all specified documents
        for doc_id in document_ids:
            doc = await self.repository.get(doc_id)

            if not doc or doc.user_id != user_id:
                continue

            if not doc.extracted_images:
                continue

            for img in doc.extracted_images:
                # Filter by image type if specified
                if image_types and img.detected_type not in image_types:
                    continue

                # Calculate relevance score based on multiple factors
                # Each factor contributes to a normalized 0-1 score
                score_components = []

                # Factor 1: Context text matching (weight: 40%)
                context_score = 0.0
                if img.context_text:
                    context_lower = img.context_text.lower()
                    matching_words = sum(1 for w in topic_words if w in context_lower)
                    # Normalize by number of topic words
                    context_score = min(1.0, matching_words / max(1, len(topic_words)))
                score_components.append(('context', context_score, 0.4))

                # Factor 2: Caption matching (weight: 25%)
                caption_score = 0.0
                if img.caption:
                    caption_lower = img.caption.lower()
                    matching_words = sum(1 for w in topic_words if w in caption_lower)
                    caption_score = min(1.0, matching_words / max(1, len(topic_words)))
                score_components.append(('caption', caption_score, 0.25))

                # Factor 3: AI description matching (weight: 20%)
                desc_score = 0.0
                if img.description:
                    desc_lower = img.description.lower()
                    matching_words = sum(1 for w in topic_words if w in desc_lower)
                    desc_score = min(1.0, matching_words / max(1, len(topic_words)))
                score_components.append(('description', desc_score, 0.2))

                # Factor 4: Keywords matching (weight: 15%)
                keyword_score = 0.0
                if img.relevance_keywords:
                    keywords_lower = [kw.lower() for kw in img.relevance_keywords]
                    matching_keywords = sum(1 for w in topic_words if any(w in kw for kw in keywords_lower))
                    keyword_score = min(1.0, matching_keywords / max(1, len(topic_words)))
                score_components.append(('keywords', keyword_score, 0.15))

                # Calculate weighted score
                relevance_score = sum(score * weight for _, score, weight in score_components)

                # Bonus for diagram/chart types (useful for educational content)
                if img.detected_type in ["diagram", "chart", "architecture", "flowchart"]:
                    relevance_score = min(1.0, relevance_score + 0.1)

                # Only include if above minimum relevance
                if relevance_score >= min_relevance:
                    matching_images.append({
                        "image_id": img.id,
                        "document_id": img.document_id,
                        "file_path": img.file_path,
                        "file_name": img.file_name,
                        "width": img.width,
                        "height": img.height,
                        "detected_type": img.detected_type,
                        "context_text": img.context_text[:300] if img.context_text else None,
                        "caption": img.caption,
                        "description": img.description,
                        "page_number": img.page_number,
                        "document_name": doc.filename,
                        "relevance_score": round(relevance_score, 3),
                    })

        # Sort by relevance and limit
        matching_images.sort(key=lambda x: x["relevance_score"], reverse=True)
        result = matching_images[:max_images]

        if result:
            print(f"[RAG_IMAGES] Found {len(result)} relevant images (best score: {result[0]['relevance_score']:.2f})", flush=True)
        else:
            print(f"[RAG_IMAGES] No images found with relevance >= {min_relevance}", flush=True)

        return result
