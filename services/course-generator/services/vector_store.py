"""
Vector Store Service

Handles document embeddings and vector similarity search.
Supports multiple backends: ChromaDB (default), Pinecone, pgvector.
"""
import asyncio
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import tiktoken
import numpy as np
from openai import AsyncOpenAI

from models.document_models import DocumentChunk, RAGChunkResult


class EmbeddingService:
    """
    Service for generating text embeddings using OpenAI.
    """

    # OpenAI embedding API limits
    MAX_TOKENS_PER_BATCH = 100000  # Safe limit for batch requests
    MAX_TEXTS_PER_BATCH = 100  # Max texts in single request

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=120.0,  # Increased timeout for large batches
        )
        self.model = model
        self.dimensions = 1536  # text-embedding-3-small dimensions

        # Initialize tokenizer for counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with automatic batching"""
        if not texts:
            return []

        # Create batches respecting token and count limits
        batches = self._create_batches(texts)
        print(f"[EMBEDDING] Processing {len(texts)} texts in {len(batches)} batches", flush=True)

        # Process all batches
        all_embeddings = [None] * len(texts)

        for batch_idx, (batch_texts, batch_indices) in enumerate(batches):
            print(f"[EMBEDDING] Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_texts)} texts)", flush=True)

            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch_texts,
                )

                # Map embeddings back to original indices
                for item in response.data:
                    original_idx = batch_indices[item.index]
                    all_embeddings[original_idx] = item.embedding

            except Exception as e:
                print(f"[EMBEDDING] Batch {batch_idx + 1} failed: {e}", flush=True)
                # Try individual texts in case of batch failure
                for i, text in enumerate(batch_texts):
                    try:
                        embedding = await self.embed_text(text)
                        all_embeddings[batch_indices[i]] = embedding
                    except Exception as inner_e:
                        print(f"[EMBEDDING] Single text embedding failed: {inner_e}", flush=True)
                        all_embeddings[batch_indices[i]] = None

            # Small delay between batches to avoid rate limiting
            if batch_idx < len(batches) - 1:
                await asyncio.sleep(0.1)

        return all_embeddings

    def _create_batches(self, texts: List[str]) -> List[Tuple[List[str], List[int]]]:
        """Create batches of texts respecting token and count limits"""
        batches = []
        current_batch_texts = []
        current_batch_indices = []
        current_tokens = 0

        for i, text in enumerate(texts):
            text_tokens = self.count_tokens(text)

            # Check if adding this text would exceed limits
            would_exceed_tokens = current_tokens + text_tokens > self.MAX_TOKENS_PER_BATCH
            would_exceed_count = len(current_batch_texts) >= self.MAX_TEXTS_PER_BATCH

            if would_exceed_tokens or would_exceed_count:
                # Save current batch and start new one
                if current_batch_texts:
                    batches.append((current_batch_texts, current_batch_indices))
                current_batch_texts = []
                current_batch_indices = []
                current_tokens = 0

            # Handle texts that are too large for a single request
            if text_tokens > self.MAX_TOKENS_PER_BATCH:
                # Truncate the text to fit
                truncated = self._truncate_text(text, self.MAX_TOKENS_PER_BATCH - 100)
                current_batch_texts.append(truncated)
            else:
                current_batch_texts.append(text)

            current_batch_indices.append(i)
            current_tokens += text_tokens

        # Don't forget the last batch
        if current_batch_texts:
            batches.append((current_batch_texts, current_batch_indices))

        return batches

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)


class VectorStoreBase(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        user_id: str,
    ) -> None:
        """Add document chunks to the store"""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[RAGChunkResult]:
        """Search for similar chunks"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document"""
        pass

    @abstractmethod
    async def delete_user_documents(self, user_id: str) -> None:
        """Delete all documents for a user"""
        pass


class ChromaVectorStore(VectorStoreBase):
    """
    Vector store using ChromaDB (recommended for development/small scale).
    """

    def __init__(self, persist_directory: str = "/tmp/viralify/chroma"):
        try:
            import chromadb
            from chromadb.config import Settings

            self.persist_directory = persist_directory
            Path(persist_directory).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            ))

            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="document_chunks",
                metadata={"hnsw:space": "cosine"},
            )

            print(f"[VECTOR] ChromaDB initialized at {persist_directory}", flush=True)

        except ImportError:
            raise ImportError("chromadb is required. Install with: pip install chromadb")

    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        user_id: str,
    ) -> None:
        """Add chunks to ChromaDB"""
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "user_id": user_id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number or -1,
                "section_title": chunk.section_title or "",
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"[VECTOR] Added {len(chunks)} chunks for document {document_id}", flush=True)

    async def search(
        self,
        query_embedding: List[float],
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[RAGChunkResult]:
        """Search ChromaDB for similar chunks"""
        # Build where filter
        where_filter = {"user_id": user_id}

        if document_ids:
            where_filter = {
                "$and": [
                    {"user_id": user_id},
                    {"document_id": {"$in": document_ids}},
                ]
            }

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to RAGChunkResult
        chunk_results = []

        if results and results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                # ChromaDB returns distances, convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance

                if similarity < similarity_threshold:
                    continue

                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                content = results['documents'][0][i] if results['documents'] else ""

                chunk_results.append(RAGChunkResult(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    document_name="",  # Will be filled by retrieval service
                    content=content,
                    similarity_score=similarity,
                    page_number=metadata.get("page_number") if metadata.get("page_number", -1) >= 0 else None,
                    section_title=metadata.get("section_title") or None,
                    token_count=metadata.get("token_count", 0),
                ))

        return chunk_results

    async def delete_document(self, document_id: str) -> None:
        """Delete document chunks from ChromaDB"""
        self.collection.delete(
            where={"document_id": document_id}
        )
        print(f"[VECTOR] Deleted chunks for document {document_id}", flush=True)

    async def delete_user_documents(self, user_id: str) -> None:
        """Delete all user documents from ChromaDB"""
        self.collection.delete(
            where={"user_id": user_id}
        )
        print(f"[VECTOR] Deleted all documents for user {user_id}", flush=True)


class PgVectorStore(VectorStoreBase):
    """
    Vector store using PostgreSQL with pgvector extension.
    Recommended for production: scalable, persistent, ACID compliant.
    """

    def __init__(
        self,
        database_url: str = None,
        pool_size: int = 10,
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://tiktok_user:tiktok_secure_pass_2024@postgres:5432/tiktok_platform"
        )
        self.pool_size = pool_size
        self.pool = None
        self._initialized = False
        print(f"[VECTOR] PgVector store configured", flush=True)

    async def _ensure_pool(self):
        """Lazily create connection pool"""
        if self.pool is None:
            try:
                import asyncpg
                from pgvector.asyncpg import register_vector

                self.pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=2,
                    max_size=self.pool_size,
                    init=register_vector,
                )
                self._initialized = True
                print(f"[VECTOR] PgVector connection pool created", flush=True)
            except ImportError:
                raise ImportError("asyncpg and pgvector are required. Install with: pip install asyncpg pgvector")

    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        user_id: str,
    ) -> None:
        """Add chunks to PostgreSQL with pgvector"""
        if not chunks:
            return

        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            # Use batch insert for better performance
            values = [
                (
                    chunk.id,
                    document_id,
                    user_id,
                    chunk.content,
                    chunk.embedding,
                    chunk.chunk_index,
                    chunk.page_number,
                    chunk.section_title,
                    chunk.token_count,
                )
                for chunk in chunks
                if chunk.embedding
            ]

            await conn.executemany(
                """
                INSERT INTO document_chunks
                    (chunk_id, document_id, user_id, content, embedding, chunk_index, page_number, section_title, token_count)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    chunk_index = EXCLUDED.chunk_index,
                    page_number = EXCLUDED.page_number,
                    section_title = EXCLUDED.section_title,
                    token_count = EXCLUDED.token_count
                """,
                values,
            )

        print(f"[VECTOR] Added {len(chunks)} chunks for document {document_id}", flush=True)

    async def search(
        self,
        query_embedding: List[float],
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.5,  # Lowered from 0.7 for better recall
    ) -> List[RAGChunkResult]:
        """Search pgvector for similar chunks using cosine similarity"""
        await self._ensure_pool()

        # Convert to numpy array for pgvector
        query_vector = np.array(query_embedding)

        print(f"[VECTOR] Searching for user {user_id}, docs={document_ids}, threshold={similarity_threshold}", flush=True)

        async with self.pool.acquire() as conn:
            # Build query with optional document filter
            if document_ids:
                query = """
                    SELECT
                        chunk_id,
                        document_id,
                        content,
                        page_number,
                        section_title,
                        token_count,
                        1 - (embedding <=> $1) as similarity
                    FROM document_chunks
                    WHERE user_id = $2
                      AND document_id = ANY($3)
                      AND 1 - (embedding <=> $1) >= $4
                    ORDER BY embedding <=> $1
                    LIMIT $5
                """
                rows = await conn.fetch(
                    query,
                    query_vector,
                    user_id,
                    document_ids,
                    similarity_threshold,
                    top_k,
                )
            else:
                query = """
                    SELECT
                        chunk_id,
                        document_id,
                        content,
                        page_number,
                        section_title,
                        token_count,
                        1 - (embedding <=> $1) as similarity
                    FROM document_chunks
                    WHERE user_id = $2
                      AND 1 - (embedding <=> $1) >= $3
                    ORDER BY embedding <=> $1
                    LIMIT $4
                """
                rows = await conn.fetch(
                    query,
                    query_vector,
                    user_id,
                    similarity_threshold,
                    top_k,
                )

            print(f"[VECTOR] Found {len(rows)} matching chunks", flush=True)

            results = []
            for row in rows:
                results.append(RAGChunkResult(
                    chunk_id=row['chunk_id'],
                    document_id=row['document_id'],
                    document_name="",  # Will be filled by retrieval service
                    content=row['content'],
                    similarity_score=float(row['similarity']),
                    page_number=row['page_number'],
                    section_title=row['section_title'],
                    token_count=row['token_count'],
                ))

            return results

    async def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document"""
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM document_chunks WHERE document_id = $1",
                document_id,
            )

        print(f"[VECTOR] Deleted chunks for document {document_id}", flush=True)

    async def delete_user_documents(self, user_id: str) -> None:
        """Delete all documents for a user"""
        await self._ensure_pool()

        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM document_chunks WHERE user_id = $1",
                user_id,
            )

        print(f"[VECTOR] Deleted all documents for user {user_id}", flush=True)

    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            print("[VECTOR] PgVector connection pool closed", flush=True)


class InMemoryVectorStore(VectorStoreBase):
    """
    Simple in-memory vector store for testing/development.
    Data is not persisted between restarts.
    """

    def __init__(self):
        self.chunks: dict = {}  # chunk_id -> (chunk_data, embedding)
        self.document_index: dict = {}  # document_id -> [chunk_ids]
        self.user_index: dict = {}  # user_id -> [document_ids]
        print("[VECTOR] InMemory vector store initialized", flush=True)

    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        user_id: str,
    ) -> None:
        """Add chunks to memory"""
        if document_id not in self.document_index:
            self.document_index[document_id] = []

        if user_id not in self.user_index:
            self.user_index[user_id] = []

        if document_id not in self.user_index[user_id]:
            self.user_index[user_id].append(document_id)

        for chunk in chunks:
            self.chunks[chunk.id] = {
                "chunk": chunk,
                "document_id": document_id,
                "user_id": user_id,
                "embedding": np.array(chunk.embedding) if chunk.embedding else None,
            }
            self.document_index[document_id].append(chunk.id)

        print(f"[VECTOR] Added {len(chunks)} chunks for document {document_id}", flush=True)

    async def search(
        self,
        query_embedding: List[float],
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[RAGChunkResult]:
        """Search using cosine similarity"""
        query_vec = np.array(query_embedding)

        # Get candidate chunks
        candidate_doc_ids = document_ids or self.user_index.get(user_id, [])

        candidates = []
        for doc_id in candidate_doc_ids:
            chunk_ids = self.document_index.get(doc_id, [])
            for chunk_id in chunk_ids:
                chunk_data = self.chunks.get(chunk_id)
                if chunk_data and chunk_data["embedding"] is not None:
                    candidates.append(chunk_data)

        # Calculate similarities
        results = []
        for data in candidates:
            embedding = data["embedding"]
            # Cosine similarity
            similarity = float(np.dot(query_vec, embedding) / (np.linalg.norm(query_vec) * np.linalg.norm(embedding)))

            if similarity >= similarity_threshold:
                chunk = data["chunk"]
                results.append((similarity, RAGChunkResult(
                    chunk_id=chunk.id,
                    document_id=data["document_id"],
                    document_name="",
                    content=chunk.content,
                    similarity_score=similarity,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    token_count=chunk.token_count,
                )))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:top_k]]

    async def delete_document(self, document_id: str) -> None:
        """Delete document from memory"""
        chunk_ids = self.document_index.pop(document_id, [])
        for chunk_id in chunk_ids:
            self.chunks.pop(chunk_id, None)

        # Update user index
        for user_id, doc_ids in self.user_index.items():
            if document_id in doc_ids:
                doc_ids.remove(document_id)

        print(f"[VECTOR] Deleted document {document_id}", flush=True)

    async def delete_user_documents(self, user_id: str) -> None:
        """Delete all user documents"""
        doc_ids = self.user_index.pop(user_id, [])
        for doc_id in doc_ids:
            await self.delete_document(doc_id)
        print(f"[VECTOR] Deleted all documents for user {user_id}", flush=True)


class VectorStoreFactory:
    """Factory for creating vector store instances"""

    @staticmethod
    def create(backend: str = "chroma", **kwargs) -> VectorStoreBase:
        """
        Create vector store instance.

        Args:
            backend: "chroma", "memory", "pinecone", or "pgvector"
            **kwargs: Backend-specific configuration

        Returns:
            VectorStoreBase instance
        """
        if backend == "memory":
            return InMemoryVectorStore()

        elif backend == "chroma":
            persist_dir = kwargs.get("persist_directory", "/tmp/viralify/chroma")
            return ChromaVectorStore(persist_directory=persist_dir)

        elif backend == "pinecone":
            # TODO: Implement Pinecone backend
            raise NotImplementedError("Pinecone backend not yet implemented")

        elif backend == "pgvector":
            database_url = kwargs.get("database_url")
            pool_size = kwargs.get("pool_size", 10)
            return PgVectorStore(database_url=database_url, pool_size=pool_size)

        else:
            raise ValueError(f"Unknown vector store backend: {backend}")


class VectorizationService:
    """
    High-level service for document vectorization.
    Combines embedding generation and vector storage.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        vector_backend: str = "memory",
        **backend_kwargs,
    ):
        self.embedding_service = EmbeddingService(model=embedding_model)
        self.vector_store = VectorStoreFactory.create(vector_backend, **backend_kwargs)

    async def vectorize_chunks(
        self,
        chunks: List[DocumentChunk],
        document_id: str,
        user_id: str,
    ) -> List[DocumentChunk]:
        """
        Generate embeddings for chunks and store them.

        Args:
            chunks: Document chunks without embeddings
            document_id: Parent document ID
            user_id: Owner user ID

        Returns:
            Chunks with embeddings added
        """
        print(f"[VECTORIZE] Processing {len(chunks)} chunks for document {document_id}", flush=True)

        # Extract texts for batch embedding
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings in batch
        embeddings = await self.embedding_service.embed_texts(texts)

        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk.document_id = document_id

        # Store in vector store
        await self.vector_store.add_chunks(chunks, document_id, user_id)

        print(f"[VECTORIZE] Completed vectorization for document {document_id}", flush=True)

        return chunks

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        return await self.embedding_service.embed_text(query)

    async def search(
        self,
        query: str,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[RAGChunkResult]:
        """
        Search for relevant chunks using semantic similarity.

        Args:
            query: Search query text
            user_id: User ID for access control
            document_ids: Optional list of specific documents to search
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of matching chunks with similarity scores
        """
        # Embed query
        query_embedding = await self.embed_query(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            document_ids=document_ids,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        return results

    async def delete_document(self, document_id: str) -> None:
        """Remove document from vector store"""
        await self.vector_store.delete_document(document_id)

    async def delete_user_documents(self, user_id: str) -> None:
        """Remove all user documents from vector store"""
        await self.vector_store.delete_user_documents(user_id)
