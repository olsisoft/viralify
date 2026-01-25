"""
pgvector Store for WeaveGraph

Stores concepts and edges in PostgreSQL with pgvector
for efficient similarity search.
"""

import os
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import json

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    asyncpg = None

from .models import ConceptNode, ConceptEdge, WeaveGraph, RelationType, ConceptSource


class WeaveGraphPgVectorStore:
    """
    PostgreSQL + pgvector storage for WeaveGraph.

    Stores concepts with embeddings and edges for
    efficient similarity search and graph traversal.
    """

    # SQL statements
    CREATE_CONCEPTS_TABLE = """
    CREATE TABLE IF NOT EXISTS weave_concepts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        canonical_name VARCHAR(255) NOT NULL,
        name VARCHAR(500) NOT NULL,
        language VARCHAR(10) DEFAULT 'en',
        embedding vector(1024),
        source_document_ids TEXT[] DEFAULT '{}',
        frequency INT DEFAULT 1,
        source_type VARCHAR(50) DEFAULT 'nlp',
        aliases TEXT[] DEFAULT '{}',
        user_id VARCHAR(255),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(canonical_name, user_id)
    );

    CREATE INDEX IF NOT EXISTS idx_weave_concepts_user ON weave_concepts(user_id);
    CREATE INDEX IF NOT EXISTS idx_weave_concepts_name ON weave_concepts(canonical_name);
    CREATE INDEX IF NOT EXISTS idx_weave_concepts_lang ON weave_concepts(language);
    """

    CREATE_CONCEPTS_EMBEDDING_INDEX = """
    CREATE INDEX IF NOT EXISTS idx_weave_concepts_embedding
    ON weave_concepts USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    """

    CREATE_EDGES_TABLE = """
    CREATE TABLE IF NOT EXISTS weave_edges (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_concept_id UUID NOT NULL,
        target_concept_id UUID NOT NULL,
        relation_type VARCHAR(50) DEFAULT 'similar',
        weight FLOAT DEFAULT 1.0,
        bidirectional BOOLEAN DEFAULT TRUE,
        user_id VARCHAR(255),
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        UNIQUE(source_concept_id, target_concept_id, relation_type)
    );

    CREATE INDEX IF NOT EXISTS idx_weave_edges_source ON weave_edges(source_concept_id);
    CREATE INDEX IF NOT EXISTS idx_weave_edges_target ON weave_edges(target_concept_id);
    CREATE INDEX IF NOT EXISTS idx_weave_edges_user ON weave_edges(user_id);
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the store.

        Args:
            connection_string: PostgreSQL connection string
        """
        if not HAS_ASYNCPG:
            raise ImportError("asyncpg is required for WeaveGraphPgVectorStore. Install with: pip install asyncpg")

        self.connection_string = connection_string or self._build_connection_string()
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._init_lock: Optional[asyncio.Lock] = None  # Created lazily in async context

    def _build_connection_string(self) -> str:
        """Build connection string from environment variables"""
        # First, check for DATABASE_URL (Docker/production style)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # asyncpg needs postgresql:// not postgres://
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            return database_url

        # Fallback to individual variables
        host = os.getenv("DATABASE_HOST", "localhost")
        port = os.getenv("DATABASE_PORT", "5432")
        user = os.getenv("DATABASE_USER", "tiktok_user")
        password = os.getenv("DATABASE_PASSWORD", "")
        database = os.getenv("DATABASE_NAME", "tiktok_platform")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    async def initialize(self) -> None:
        """Initialize connection pool and create tables"""
        if self._initialized and self._pool:
            return

        # Create lock lazily (must be in async context)
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized and self._pool:
                return

            try:
                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=2,
                    max_size=10,
                    command_timeout=60
                )

                async with self._pool.acquire() as conn:
                    # Create tables
                    await conn.execute(self.CREATE_CONCEPTS_TABLE)
                    await conn.execute(self.CREATE_EDGES_TABLE)

                    # Try to create embedding index (may fail if no embeddings yet)
                    try:
                        await conn.execute(self.CREATE_CONCEPTS_EMBEDDING_INDEX)
                    except Exception as e:
                        print(f"[WEAVE_GRAPH] Note: Embedding index creation deferred: {e}", flush=True)

                self._initialized = True
                print("[WEAVE_GRAPH] pgvector store initialized", flush=True)

            except Exception as e:
                print(f"[WEAVE_GRAPH] Failed to initialize: {e}", flush=True)
                raise

    async def close(self) -> None:
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False

    async def store_concept(self, concept: ConceptNode, user_id: str) -> str:
        """
        Store or update a concept.

        Returns the concept ID.
        """
        await self.initialize()

        async with self._pool.acquire() as conn:
            # Check if exists
            existing = await conn.fetchrow(
                "SELECT id, frequency FROM weave_concepts WHERE canonical_name = $1 AND user_id = $2",
                concept.canonical_name, user_id
            )

            if existing:
                # Update existing
                await conn.execute("""
                    UPDATE weave_concepts
                    SET frequency = frequency + $1,
                        source_document_ids = array_cat(source_document_ids, $2::text[]),
                        updated_at = NOW()
                    WHERE id = $3
                """, concept.frequency, concept.source_document_ids, existing['id'])
                return str(existing['id'])
            else:
                # Insert new
                embedding_str = None
                if concept.embedding:
                    embedding_str = '[' + ','.join(str(x) for x in concept.embedding) + ']'

                result = await conn.fetchrow("""
                    INSERT INTO weave_concepts
                    (canonical_name, name, language, embedding, source_document_ids,
                     frequency, source_type, aliases, user_id)
                    VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9)
                    RETURNING id
                """,
                    concept.canonical_name,
                    concept.name,
                    concept.language,
                    embedding_str,
                    concept.source_document_ids,
                    concept.frequency,
                    concept.source_type.value,
                    concept.aliases,
                    user_id
                )
                return str(result['id'])

    async def store_concepts_batch(self, concepts: List[ConceptNode], user_id: str) -> List[str]:
        """Store multiple concepts efficiently"""
        ids = []
        for concept in concepts:
            cid = await self.store_concept(concept, user_id)
            ids.append(cid)
        return ids

    async def update_concept_embedding(self, concept_id: str, embedding: List[float]) -> None:
        """Update the embedding for a concept"""
        await self.initialize()

        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'

        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE weave_concepts
                SET embedding = $1::vector, updated_at = NOW()
                WHERE id = $2::uuid
            """, embedding_str, concept_id)

    async def store_edge(self, edge: ConceptEdge, user_id: str) -> str:
        """Store an edge between concepts"""
        await self.initialize()

        async with self._pool.acquire() as conn:
            try:
                result = await conn.fetchrow("""
                    INSERT INTO weave_edges
                    (source_concept_id, target_concept_id, relation_type, weight, bidirectional, user_id, metadata)
                    VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7)
                    ON CONFLICT (source_concept_id, target_concept_id, relation_type)
                    DO UPDATE SET weight = GREATEST(weave_edges.weight, $4)
                    RETURNING id
                """,
                    edge.source_id,
                    edge.target_id,
                    edge.relation_type.value,
                    edge.weight,
                    edge.bidirectional,
                    user_id,
                    json.dumps(edge.metadata)
                )
                return str(result['id'])
            except Exception as e:
                print(f"[WEAVE_GRAPH] Failed to store edge: {e}", flush=True)
                return ""

    async def store_edges_batch(self, edges: List[ConceptEdge], user_id: str) -> int:
        """Store multiple edges efficiently"""
        count = 0
        for edge in edges:
            if await self.store_edge(edge, user_id):
                count += 1
        return count

    async def find_similar_concepts(
        self,
        embedding: List[float],
        user_id: str,
        limit: int = 20,
        threshold: float = 0.5
    ) -> List[Tuple[ConceptNode, float]]:
        """
        Find concepts similar to the given embedding.

        Returns list of (concept, similarity_score) tuples.
        """
        await self.initialize()

        embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    id, canonical_name, name, language, frequency, source_type, aliases,
                    1 - (embedding <=> $1::vector) as similarity
                FROM weave_concepts
                WHERE user_id = $2
                  AND embedding IS NOT NULL
                  AND 1 - (embedding <=> $1::vector) > $3
                ORDER BY similarity DESC
                LIMIT $4
            """, embedding_str, user_id, threshold, limit)

            results = []
            for row in rows:
                concept = ConceptNode(
                    id=str(row['id']),
                    canonical_name=row['canonical_name'],
                    name=row['name'],
                    language=row['language'],
                    frequency=row['frequency'],
                    source_type=ConceptSource(row['source_type']),
                    aliases=list(row['aliases']) if row['aliases'] else []
                )
                results.append((concept, row['similarity']))

            return results

    async def get_concept_by_name(self, canonical_name: str, user_id: str) -> Optional[ConceptNode]:
        """Get a concept by its canonical name"""
        await self.initialize()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, canonical_name, name, language, frequency, source_type, aliases, source_document_ids
                FROM weave_concepts
                WHERE canonical_name = $1 AND user_id = $2
            """, canonical_name, user_id)

            if not row:
                return None

            return ConceptNode(
                id=str(row['id']),
                canonical_name=row['canonical_name'],
                name=row['name'],
                language=row['language'],
                frequency=row['frequency'],
                source_type=ConceptSource(row['source_type']),
                aliases=list(row['aliases']) if row['aliases'] else [],
                source_document_ids=list(row['source_document_ids']) if row['source_document_ids'] else []
            )

    async def get_concept_neighbors(
        self,
        concept_id: str,
        user_id: str,
        max_depth: int = 1,
        min_weight: float = 0.5
    ) -> List[Tuple[ConceptNode, float, str]]:
        """
        Get neighboring concepts through edges.

        Returns list of (concept, weight, relation_type) tuples.
        """
        await self.initialize()

        async with self._pool.acquire() as conn:
            # Get direct neighbors
            rows = await conn.fetch("""
                SELECT DISTINCT
                    c.id, c.canonical_name, c.name, c.language, c.frequency, c.source_type, c.aliases,
                    e.weight, e.relation_type
                FROM weave_concepts c
                JOIN weave_edges e ON (
                    (e.source_concept_id = $1::uuid AND e.target_concept_id = c.id) OR
                    (e.bidirectional = TRUE AND e.target_concept_id = $1::uuid AND e.source_concept_id = c.id)
                )
                WHERE c.user_id = $2 AND e.weight >= $3
                ORDER BY e.weight DESC
            """, concept_id, user_id, min_weight)

            results = []
            for row in rows:
                concept = ConceptNode(
                    id=str(row['id']),
                    canonical_name=row['canonical_name'],
                    name=row['name'],
                    language=row['language'],
                    frequency=row['frequency'],
                    source_type=ConceptSource(row['source_type']),
                    aliases=list(row['aliases']) if row['aliases'] else []
                )
                results.append((concept, row['weight'], row['relation_type']))

            return results

    async def expand_query(
        self,
        query_terms: List[str],
        user_id: str,
        max_expansions: int = 10
    ) -> Dict[str, List[str]]:
        """
        Expand query terms using the graph.

        Returns dict mapping original terms to their expansions.
        """
        await self.initialize()

        expansions = {}

        for term in query_terms:
            canonical = term.lower().replace(' ', '_')
            concept = await self.get_concept_by_name(canonical, user_id)

            if concept:
                neighbors = await self.get_concept_neighbors(concept.id, user_id)
                expanded = [concept.name]  # Include original
                for neighbor, weight, rel_type in neighbors[:max_expansions]:
                    if neighbor.name not in expanded:
                        expanded.append(neighbor.name)
                expansions[term] = expanded
            else:
                expansions[term] = [term]  # No expansion found

        return expansions

    async def get_user_graph_stats(self, user_id: str) -> Dict:
        """Get statistics about a user's concept graph"""
        await self.initialize()

        async with self._pool.acquire() as conn:
            concept_count = await conn.fetchval(
                "SELECT COUNT(*) FROM weave_concepts WHERE user_id = $1",
                user_id
            )
            edge_count = await conn.fetchval(
                "SELECT COUNT(*) FROM weave_edges WHERE user_id = $1",
                user_id
            )
            languages = await conn.fetch(
                "SELECT DISTINCT language FROM weave_concepts WHERE user_id = $1",
                user_id
            )
            top_concepts = await conn.fetch("""
                SELECT name, frequency
                FROM weave_concepts
                WHERE user_id = $1
                ORDER BY frequency DESC
                LIMIT 10
            """, user_id)

            return {
                "total_concepts": concept_count,
                "total_edges": edge_count,
                "languages": [r['language'] for r in languages],
                "top_concepts": [(r['name'], r['frequency']) for r in top_concepts]
            }

    async def delete_user_graph(self, user_id: str) -> None:
        """Delete all concepts and edges for a user"""
        await self.initialize()

        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM weave_edges WHERE user_id = $1", user_id)
            await conn.execute("DELETE FROM weave_concepts WHERE user_id = $1", user_id)
            print(f"[WEAVE_GRAPH] Deleted graph for user {user_id}", flush=True)


# Singleton instance
_store_instance: Optional[WeaveGraphPgVectorStore] = None


def get_weave_graph_store() -> WeaveGraphPgVectorStore:
    """Get or create the singleton store instance"""
    global _store_instance
    if _store_instance is None:
        _store_instance = WeaveGraphPgVectorStore()
    return _store_instance
