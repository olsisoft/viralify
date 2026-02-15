"""
RAG Service - Main Orchestrator

Coordinates document processing and retrieval for course generation.
This is a slim orchestrator that delegates to specialized modules.
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..algorithms import WeightedMultiSourceRAG, get_weighted_rag
from ..models import DocumentRelevanceScore, WeightedRAGResult
from ..storage import DocumentRepositoryPg, RAGDocumentStorage
from ..processors import StructureExtractor, AIStructureGenerator
from ..retrieval import ContextBuilder, ChunkPrioritizer, ImageRetriever

# Optional imports - graceful degradation
try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# Import models from main course-generator
try:
    from models.document_models import (
        Document,
        DocumentChunk,
        DocumentStatus,
        DocumentType,
        RAGChunkResult,
        EXTENSION_TO_TYPE,
    )
except ImportError:
    Document = None
    DocumentStatus = None
    DocumentType = None
    EXTENSION_TO_TYPE = {}

# Import services from main course-generator
try:
    from services.security_scanner import SecurityScanner
    from services.document_parser import DocumentParser, WebContentParser
    from services.vector_store import VectorizationService
    from services.reranker import get_reranker, RerankerBase
except ImportError:
    SecurityScanner = None
    DocumentParser = None
    WebContentParser = None
    VectorizationService = None
    get_reranker = None


class RAGService:
    """
    Main RAG Service orchestrating document processing and retrieval.

    This is a slim orchestrator that coordinates:
    - Document upload and security scanning
    - Document parsing and vectorization
    - Weighted multi-source retrieval
    - Context building with structure extraction

    Usage:
        service = RAGService()
        await service.initialize()

        # Upload document
        doc = await service.upload_document(content, filename, user_id)

        # Get context for course generation
        context = await service.get_context_for_course_generation(
            topic="Apache Kafka",
            document_ids=["doc_123"],
            user_id="user_456"
        )
    """

    # Token limits
    MAX_CONTEXT_TOKENS = 8000
    MAX_PROMPT_TOKENS = 100000

    # Re-ranking configuration
    RERANK_TOP_K = 30
    RERANK_FINAL_K = 15

    def __init__(
        self,
        vector_backend: str = "memory",
        storage_path: str = "/tmp/viralify/documents",
        reranker_backend: str = "auto",
    ):
        """
        Initialize the RAG service.

        Args:
            vector_backend: Backend for vector store (memory, chroma, pgvector)
            storage_path: Path for local document storage
            reranker_backend: Re-ranker backend (auto, cross-encoder, tfidf)
        """
        self.vector_backend = vector_backend
        self.storage_path = storage_path
        self.reranker_backend = reranker_backend

        # Lazy-initialized components
        self._repository: Optional[DocumentRepositoryPg] = None
        self._storage: Optional[RAGDocumentStorage] = None
        self._security_scanner = None
        self._document_parser = None
        self._web_parser = None
        self._vectorization = None
        self._reranker = None
        self._openai_client = None
        self._tokenizer = None

        # Specialized modules
        self._weighted_rag: Optional[WeightedMultiSourceRAG] = None
        self._structure_extractor: Optional[StructureExtractor] = None
        self._ai_structure_generator: Optional[AIStructureGenerator] = None
        self._context_builder: Optional[ContextBuilder] = None

        # State
        self._initialized = False
        self._last_weighted_result: Optional[WeightedRAGResult] = None

    async def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        # Initialize repository and storage
        self._repository = DocumentRepositoryPg()
        await self._repository.initialize()

        self._storage = RAGDocumentStorage()
        await self._storage.initialize()

        # Initialize processing components
        if SecurityScanner:
            self._security_scanner = SecurityScanner()
        if DocumentParser:
            self._document_parser = DocumentParser()
        if WebContentParser:
            self._web_parser = WebContentParser()
        if VectorizationService:
            self._vectorization = VectorizationService(vector_backend=self.vector_backend)

        # Initialize re-ranker
        if get_reranker:
            try:
                self._reranker = get_reranker(self.reranker_backend)
                print(f"[RAG] Re-ranker ready: {self._reranker.__class__.__name__}", flush=True)
            except Exception as e:
                print(f"[RAG] Re-ranker initialization failed: {e}", flush=True)

        # Initialize OpenAI client
        if AsyncOpenAI:
            self._openai_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=60.0,
            )

        # Initialize tokenizer
        if tiktoken:
            try:
                self._tokenizer = tiktoken.encoding_for_model("gpt-4")
            except KeyError:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize specialized modules
        self._weighted_rag = get_weighted_rag()
        self._structure_extractor = StructureExtractor()
        self._ai_structure_generator = AIStructureGenerator(self._openai_client)
        self._context_builder = ContextBuilder(self._tokenizer)

        self._initialized = True
        print(
            f"[RAG] Service initialized "
            f"(vector: {self.vector_backend}, "
            f"storage: {'S3' if not self._storage.is_using_local else 'local'})",
            flush=True
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return len(text) // 4

    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        course_id: Optional[str] = None,
    ) -> "Document":
        """
        Upload and process a document.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            user_id: Owner user ID
            course_id: Optional course association

        Returns:
            Processed Document object
        """
        if not self._initialized:
            await self.initialize()

        print(f"[RAG] Uploading: {filename} for user {user_id}", flush=True)

        # Determine document type
        ext = Path(filename).suffix.lower()
        if ext not in EXTENSION_TO_TYPE:
            raise ValueError(f"Unsupported file extension: {ext}")

        document_type = EXTENSION_TO_TYPE[ext]

        # Create document record
        document = Document(
            id=str(uuid.uuid4()),
            user_id=user_id,
            course_id=course_id,
            filename=self._security_scanner.sanitize_filename(filename) if self._security_scanner else filename,
            document_type=document_type,
            file_size_bytes=len(file_content),
            status=DocumentStatus.SCANNING,
        )

        await self._repository.save(document)

        try:
            # 1. Security scan
            if self._security_scanner:
                scan_result = await self._security_scanner.scan_file(
                    file_content, filename, document_type
                )
                document.security_scan = scan_result

                if not scan_result.is_safe:
                    document.status = DocumentStatus.SCAN_FAILED
                    document.error_message = f"Security scan failed: {', '.join(scan_result.threats_found)}"
                    await self._repository.save(document)
                    raise ValueError(document.error_message)

            # 2. Save file
            file_path = await self._storage.save_file(
                file_content, user_id, document.id, document.filename
            )
            document.file_path = file_path

            # 3. Parse document
            document.status = DocumentStatus.PARSING
            await self._repository.save(document)

            if self._document_parser:
                raw_text, chunks, metadata = await self._document_parser.parse_document(
                    file_content, filename, document_type
                )

                document.raw_content = raw_text
                document.extracted_metadata = metadata
                document.page_count = metadata.get("page_count", 0)
                document.word_count = len(raw_text.split())

            # 4. Vectorize
            document.status = DocumentStatus.VECTORIZING
            await self._repository.save(document)

            if self._vectorization and document.raw_content:
                await self._vectorization.add_document(document)

            # 5. Mark ready
            document.status = DocumentStatus.READY
            document.processed_at = datetime.utcnow()
            await self._repository.save(document)

            print(f"[RAG] Document ready: {document.id}", flush=True)
            return document

        except Exception as e:
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            await self._repository.save(document)
            raise

    async def get_document(self, document_id: str) -> Optional["Document"]:
        """Get document by ID."""
        if not self._initialized:
            await self.initialize()
        return await self._repository.get(document_id)

    async def get_documents_by_user(self, user_id: str) -> List["Document"]:
        """Get all documents for a user."""
        if not self._initialized:
            await self.initialize()
        return await self._repository.get_by_user(user_id)

    async def delete_document(self, document_id: str, user_id: str) -> bool:
        """Delete a document."""
        if not self._initialized:
            await self.initialize()

        document = await self._repository.get(document_id)
        if not document or document.user_id != user_id:
            return False

        # Delete from vector store
        if self._vectorization:
            await self._vectorization.delete_document(document_id)

        # Delete file
        if document.file_path:
            await self._storage.delete_file(document.file_path)

        # Delete record
        await self._repository.delete(document_id)

        print(f"[RAG] Deleted document: {document_id}", flush=True)
        return True

    async def get_context_for_course_generation(
        self,
        topic: str,
        description: Optional[str],
        document_ids: List[str],
        user_id: str,
        max_tokens: int = 8000,
        use_weighted: bool = True,
    ) -> str:
        """
        Get relevant context from documents for course generation.

        This is the main integration point with the course generator.

        Args:
            topic: Course topic
            description: Course description
            document_ids: Documents to search
            user_id: User ID
            max_tokens: Maximum context tokens
            use_weighted: Use weighted multi-source algorithm

        Returns:
            Combined context string
        """
        if not self._initialized:
            await self.initialize()

        effective_max_tokens = min(max_tokens, self.MAX_CONTEXT_TOKENS)

        print(f"[RAG] Getting context for: {topic[:50]}... ({len(document_ids)} docs)", flush=True)

        # Extract document structure first
        document_structure = await self._extract_document_structure(document_ids, user_id)
        structure_tokens = self.count_tokens(document_structure)
        print(f"[RAG] Document structure: {structure_tokens} tokens", flush=True)

        # Calculate remaining budget for content
        content_max_tokens = effective_max_tokens - structure_tokens - 100

        # Get weighted context
        if use_weighted and len(document_ids) > 1:
            print(f"[RAG] Using weighted multi-source algorithm", flush=True)
            weighted_result = await self._get_weighted_context(
                topic, description, document_ids, user_id, content_max_tokens
            )
            self._last_weighted_result = weighted_result

            if document_structure:
                return f"{document_structure}\n\n{weighted_result.combined_context}"
            return weighted_result.combined_context

        # Single document: simpler approach
        return await self._get_simple_context(
            topic, description, document_ids, user_id, content_max_tokens, document_structure
        )

    async def _extract_document_structure(
        self,
        document_ids: List[str],
        user_id: str,
    ) -> str:
        """Extract structure from all documents."""
        structures = []
        docs_without_structure = []

        for doc_id in document_ids:
            doc = await self._repository.get(doc_id)
            if not doc or doc.user_id != user_id:
                continue
            if DocumentStatus and doc.status != DocumentStatus.READY:
                continue

            structure = self._structure_extractor.extract(doc)

            if structure.headings:
                formatted = self._structure_extractor.format_structure(structure)
                structures.append(formatted)
            else:
                # Need AI generation
                docs_without_structure.append((doc_id, doc, structure.is_youtube))

        # Generate structure for docs without explicit structure
        for doc_id, doc, is_youtube in docs_without_structure:
            ai_structure = await self._ai_structure_generator.generate(doc, is_youtube)
            if ai_structure:
                formatted = self._ai_structure_generator.format_ai_structure(ai_structure)
                structures.append(formatted)

        if not structures:
            return ""

        header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           DOCUMENT STRUCTURE - YOUR COURSE MUST FOLLOW THIS OUTLINE          ║
╚══════════════════════════════════════════════════════════════════════════════╝

The following is the EXACT structure extracted from the source documents.
Your course sections and lectures MUST map directly to this structure.

DO NOT invent new topics. DO NOT reorganize. FOLLOW THIS STRUCTURE.
"""
        return header + "\n".join(structures)

    async def _get_weighted_context(
        self,
        topic: str,
        description: Optional[str],
        document_ids: List[str],
        user_id: str,
        max_tokens: int,
    ) -> WeightedRAGResult:
        """Get weighted context from multiple documents."""
        # Get documents
        documents = []
        for doc_id in document_ids:
            doc = await self._repository.get(doc_id)
            if doc and doc.user_id == user_id:
                if DocumentStatus and doc.status == DocumentStatus.READY:
                    documents.append(doc)

        if not documents:
            return WeightedRAGResult(
                combined_context="",
                document_scores=[],
            )

        # Score documents
        scores = await self._weighted_rag.score_documents(documents, topic, description)

        # Allocate tokens
        scores = self._weighted_rag.allocate_tokens(scores, max_tokens)

        # Retrieve context
        result = await self._weighted_rag.retrieve_weighted_context(
            documents, scores, topic
        )

        return result

    async def _get_simple_context(
        self,
        topic: str,
        description: Optional[str],
        document_ids: List[str],
        user_id: str,
        max_tokens: int,
        document_structure: str,
    ) -> str:
        """Get context using simple vector search (single document or weighted disabled)."""
        # Get raw content from documents
        all_content = []

        for doc_id in document_ids:
            doc = await self._repository.get(doc_id)
            if not doc or doc.user_id != user_id:
                continue
            if doc.raw_content:
                all_content.append(f"=== {doc.filename} ===\n{doc.raw_content}")

        if not all_content:
            return document_structure

        combined = "\n\n".join(all_content)

        # Truncate to fit budget
        if self._tokenizer:
            tokens = self._tokenizer.encode(combined)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens - 50]
                combined = self._tokenizer.decode(truncated_tokens)
                combined += "\n\n[... content truncated ...]"

        if document_structure:
            return f"{document_structure}\n\n{combined}"
        return combined

    def get_last_weighted_result(self) -> Optional[WeightedRAGResult]:
        """Get the last weighted RAG result for traceability."""
        return self._last_weighted_result

    async def close(self) -> None:
        """Close all resources."""
        if self._repository:
            await self._repository.close()


# Module-level factory
_default_service = None


async def get_rag_service(
    vector_backend: str = "memory",
    storage_path: str = "/tmp/viralify/documents",
) -> RAGService:
    """
    Get or create a RAG service instance.

    Args:
        vector_backend: Vector store backend
        storage_path: Document storage path

    Returns:
        Initialized RAGService instance
    """
    global _default_service
    if _default_service is None:
        _default_service = RAGService(vector_backend, storage_path)
        await _default_service.initialize()
    return _default_service
