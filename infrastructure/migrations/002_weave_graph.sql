-- =============================================================================
-- Migration 002: WeaveGraph Tables (Phase 2 & 3)
-- =============================================================================
-- Description: Creates tables for the WeaveGraph concept graph system
--              Used for RAG query expansion and resonance matching
-- Date: 2026-01-25
-- Phases: Phase 2 (WeaveGraph) + Phase 3 (Resonate Match)
-- =============================================================================

-- Ensure pgvector extension is available
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Table: weave_concepts
-- Stores concepts extracted from documents with E5-large embeddings
-- =============================================================================

CREATE TABLE IF NOT EXISTS weave_concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Concept identification
    canonical_name VARCHAR(255) NOT NULL,      -- Normalized form: "apache_kafka"
    name VARCHAR(500) NOT NULL,                -- Display form: "Apache Kafka"
    language VARCHAR(10) DEFAULT 'en',         -- 'en', 'fr', etc.

    -- E5-large multilingual embedding (1024 dimensions)
    embedding vector(1024),

    -- Source tracking
    source_document_ids TEXT[] DEFAULT '{}',   -- Which documents mention this concept
    frequency INT DEFAULT 1,                   -- How often it appears
    source_type VARCHAR(50) DEFAULT 'nlp',     -- 'nlp', 'keyword', 'entity', 'technical', 'llm'
    aliases TEXT[] DEFAULT '{}',               -- Alternative names

    -- Ownership
    user_id VARCHAR(255),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    UNIQUE(canonical_name, user_id)
);

-- Comment
COMMENT ON TABLE weave_concepts IS 'WeaveGraph concepts with E5-large multilingual embeddings for cross-language RAG';
COMMENT ON COLUMN weave_concepts.embedding IS 'E5-large multilingual embedding (1024 dimensions) for semantic similarity';
COMMENT ON COLUMN weave_concepts.canonical_name IS 'Normalized lowercase form with underscores (e.g., apache_kafka)';

-- =============================================================================
-- Table: weave_edges
-- Stores relationships between concepts
-- =============================================================================

CREATE TABLE IF NOT EXISTS weave_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Edge endpoints
    source_concept_id UUID NOT NULL REFERENCES weave_concepts(id) ON DELETE CASCADE,
    target_concept_id UUID NOT NULL REFERENCES weave_concepts(id) ON DELETE CASCADE,

    -- Relationship properties
    relation_type VARCHAR(50) DEFAULT 'similar',  -- similar, translation, part_of, prerequisite, synonym, hypernym, hyponym
    weight FLOAT DEFAULT 1.0,                     -- Relationship strength (0-1)
    bidirectional BOOLEAN DEFAULT TRUE,           -- True for similarity, False for hierarchical

    -- Ownership
    user_id VARCHAR(255),

    -- Additional metadata
    metadata JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    UNIQUE(source_concept_id, target_concept_id, relation_type)
);

-- Comment
COMMENT ON TABLE weave_edges IS 'Semantic relationships between concepts (similarity, translation, hierarchy)';
COMMENT ON COLUMN weave_edges.weight IS 'Relationship strength from 0 to 1, used for resonance propagation';
COMMENT ON COLUMN weave_edges.relation_type IS 'Type: similar, translation, part_of, prerequisite, synonym, hypernym, hyponym';

-- =============================================================================
-- Indexes for weave_concepts
-- =============================================================================

-- User filtering (most common query pattern)
CREATE INDEX IF NOT EXISTS idx_weave_concepts_user
ON weave_concepts(user_id);

-- Lookup by canonical name
CREATE INDEX IF NOT EXISTS idx_weave_concepts_name
ON weave_concepts(canonical_name);

-- Language filtering
CREATE INDEX IF NOT EXISTS idx_weave_concepts_lang
ON weave_concepts(language);

-- Combined user + name lookup
CREATE INDEX IF NOT EXISTS idx_weave_concepts_user_name
ON weave_concepts(user_id, canonical_name);

-- HNSW index for fast vector similarity search
-- m=16: connections per layer (memory vs accuracy tradeoff)
-- ef_construction=64: index build quality
CREATE INDEX IF NOT EXISTS idx_weave_concepts_embedding
ON weave_concepts USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- Indexes for weave_edges
-- =============================================================================

-- Source concept lookup (for graph traversal)
CREATE INDEX IF NOT EXISTS idx_weave_edges_source
ON weave_edges(source_concept_id);

-- Target concept lookup (for reverse traversal)
CREATE INDEX IF NOT EXISTS idx_weave_edges_target
ON weave_edges(target_concept_id);

-- User filtering
CREATE INDEX IF NOT EXISTS idx_weave_edges_user
ON weave_edges(user_id);

-- Relation type filtering
CREATE INDEX IF NOT EXISTS idx_weave_edges_type
ON weave_edges(relation_type);

-- Combined source + type (for typed traversal)
CREATE INDEX IF NOT EXISTS idx_weave_edges_source_type
ON weave_edges(source_concept_id, relation_type);

-- =============================================================================
-- Trigger: Auto-update updated_at
-- =============================================================================

-- Function (may already exist from other migrations)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for weave_concepts
DROP TRIGGER IF EXISTS update_weave_concepts_updated_at ON weave_concepts;
CREATE TRIGGER update_weave_concepts_updated_at
    BEFORE UPDATE ON weave_concepts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Verification
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE '=== Migration 002: WeaveGraph Tables ===';
    RAISE NOTICE 'Tables created: weave_concepts, weave_edges';
    RAISE NOTICE 'Indexes created: 9 indexes (including HNSW for vector search)';
    RAISE NOTICE 'Triggers created: update_weave_concepts_updated_at';
    RAISE NOTICE 'Migration completed successfully!';
END $$;
