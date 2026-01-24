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

    async def get_context_for_course_generation(
        self,
        topic: str,
        description: Optional[str],
        document_ids: List[str],
        user_id: str,
        max_tokens: int = 8000,  # Increased for deeper RAG integration
    ) -> str:
        """
        Get relevant context from documents for course generation.

        This is the main integration point with the course generator.
        Returns comprehensive document content to ensure deep RAG integration.

        Args:
            topic: Course topic
            description: Course description
            document_ids: Documents to search
            user_id: User ID
            max_tokens: Maximum context tokens

        Returns:
            Combined context string for course generation
        """
        # Enforce maximum context size to prevent API token errors
        effective_max_tokens = min(max_tokens, self.MAX_CONTEXT_TOKENS)

        print(f"[RAG] Getting context for course: {topic[:50]}... (max {effective_max_tokens} tokens)", flush=True)
        print(f"[RAG] Searching in {len(document_ids)} documents", flush=True)
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
        if total_raw_tokens <= effective_max_tokens:
            print(f"[RAG] Using FULL document content ({total_raw_tokens} tokens)", flush=True)
            return "\n\n".join([d["content"] for d in all_document_content])

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

        return combined

    async def get_images_for_topic(
        self,
        topic: str,
        document_ids: List[str],
        user_id: str,
        image_types: List[str] = None,
        max_images: int = 5,
    ) -> List[dict]:
        """
        Get relevant images from documents for a specific topic.

        Used as visual fallback when diagram generation fails.

        Args:
            topic: Topic to match images against
            document_ids: Documents to search
            user_id: User ID for access control
            image_types: Filter by image types (diagram, chart, screenshot, etc.)
            max_images: Maximum number of images to return

        Returns:
            List of image dictionaries with path, description, and relevance
        """
        print(f"[RAG] Searching images for topic: {topic[:50]}...", flush=True)

        matching_images = []
        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())

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

                # Calculate relevance score based on context matching
                relevance_score = 0.0

                # Check context text for topic words
                if img.context_text:
                    context_lower = img.context_text.lower()
                    matching_words = sum(1 for w in topic_words if w in context_lower)
                    relevance_score += matching_words * 0.2

                # Check caption for topic words
                if img.caption:
                    caption_lower = img.caption.lower()
                    matching_words = sum(1 for w in topic_words if w in caption_lower)
                    relevance_score += matching_words * 0.3

                # Check AI description if available
                if img.description:
                    desc_lower = img.description.lower()
                    matching_words = sum(1 for w in topic_words if w in desc_lower)
                    relevance_score += matching_words * 0.3

                # Check keywords
                if img.relevance_keywords:
                    matching_keywords = sum(1 for kw in img.relevance_keywords if kw.lower() in topic_lower)
                    relevance_score += matching_keywords * 0.2

                # Bonus for diagrams in educational content
                if img.detected_type in ["diagram", "chart"]:
                    relevance_score += 0.1

                if relevance_score > 0:
                    matching_images.append({
                        "image_id": img.id,
                        "file_path": img.file_path,
                        "file_name": img.file_name,
                        "width": img.width,
                        "height": img.height,
                        "detected_type": img.detected_type,
                        "context_text": img.context_text[:200] if img.context_text else None,
                        "caption": img.caption,
                        "description": img.description,
                        "page_number": img.page_number,
                        "document_name": doc.filename,
                        "relevance_score": relevance_score,
                    })

        # Sort by relevance and limit
        matching_images.sort(key=lambda x: x["relevance_score"], reverse=True)
        result = matching_images[:max_images]

        print(f"[RAG] Found {len(result)} relevant images", flush=True)

        return result
