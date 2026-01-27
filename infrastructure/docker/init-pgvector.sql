-- =============================================================================
-- pgvector Extension and Tables for RAG Vector Store
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- Document Chunks Table with Vector Embeddings
-- =============================================================================

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id VARCHAR(255) UNIQUE NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI text-embedding-3-small dimensions
    chunk_index INTEGER NOT NULL DEFAULT 0,
    page_number INTEGER,
    section_title VARCHAR(500),
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes for filtering
    CONSTRAINT fk_chunk_id UNIQUE (chunk_id)
);

-- Index for user filtering
CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON document_chunks(user_id);

-- Index for document filtering
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);

-- Index for combined user + document filtering
CREATE INDEX IF NOT EXISTS idx_chunks_user_document ON document_chunks(user_id, document_id);

-- HNSW index for fast vector similarity search (cosine distance)
-- ef_construction: higher = more accurate but slower index build
-- m: connections per layer, higher = more accurate but more memory
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- =============================================================================
-- Documents Metadata Table (optional, for document tracking)
-- =============================================================================

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL DEFAULT 0,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for user's documents listing
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Lecture Components Table (for lecture editing)
-- =============================================================================

CREATE TABLE IF NOT EXISTS lecture_components (
    id VARCHAR(255) PRIMARY KEY,
    lecture_id VARCHAR(255) NOT NULL,
    job_id VARCHAR(255) NOT NULL,

    -- JSON columns for complex data (slides, voiceover, params)
    slides_json JSONB NOT NULL DEFAULT '[]',
    voiceover_json JSONB,
    generation_params_json JSONB NOT NULL DEFAULT '{}',

    -- Scalar columns
    total_duration FLOAT NOT NULL DEFAULT 0.0,
    video_url TEXT,
    presentation_job_id VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'completed',
    is_edited BOOLEAN NOT NULL DEFAULT FALSE,
    error TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for looking up components by lecture_id (most common query)
CREATE INDEX IF NOT EXISTS idx_lecture_components_lecture_id ON lecture_components(lecture_id);

-- Index for looking up all components for a job
CREATE INDEX IF NOT EXISTS idx_lecture_components_job_id ON lecture_components(job_id);

-- Index for finding edited components
CREATE INDEX IF NOT EXISTS idx_lecture_components_edited ON lecture_components(is_edited) WHERE is_edited = TRUE;

-- Trigger for updating updated_at
DROP TRIGGER IF EXISTS update_lecture_components_updated_at ON lecture_components;
CREATE TRIGGER update_lecture_components_updated_at
    BEFORE UPDATE ON lecture_components
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE document_chunks IS 'Stores document chunks with vector embeddings for RAG retrieval';
COMMENT ON COLUMN document_chunks.embedding IS 'OpenAI text-embedding-3-small vector (1536 dimensions)';
COMMENT ON TABLE documents IS 'Metadata for uploaded documents';
COMMENT ON TABLE lecture_components IS 'Stores editable lecture components for video editing and regeneration';

-- =============================================================================
-- WeaveGraph Tables (Concept Graph for RAG Enhancement)
-- =============================================================================

-- Concepts with E5-large embeddings (1024 dimensions)
CREATE TABLE IF NOT EXISTS weave_concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_name VARCHAR(255) NOT NULL,
    name VARCHAR(500) NOT NULL,
    language VARCHAR(10) DEFAULT 'en',
    embedding vector(1024),  -- E5-large multilingual embeddings
    source_document_ids TEXT[] DEFAULT '{}',
    frequency INT DEFAULT 1,
    source_type VARCHAR(50) DEFAULT 'nlp',
    aliases TEXT[] DEFAULT '{}',
    user_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(canonical_name, user_id)
);

-- Indexes for concept queries
CREATE INDEX IF NOT EXISTS idx_weave_concepts_user ON weave_concepts(user_id);
CREATE INDEX IF NOT EXISTS idx_weave_concepts_name ON weave_concepts(canonical_name);
CREATE INDEX IF NOT EXISTS idx_weave_concepts_lang ON weave_concepts(language);

-- HNSW index for fast concept similarity search
CREATE INDEX IF NOT EXISTS idx_weave_concepts_embedding
ON weave_concepts USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Edges connecting concepts (relationships)
CREATE TABLE IF NOT EXISTS weave_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_concept_id UUID NOT NULL REFERENCES weave_concepts(id) ON DELETE CASCADE,
    target_concept_id UUID NOT NULL REFERENCES weave_concepts(id) ON DELETE CASCADE,
    relation_type VARCHAR(50) DEFAULT 'similar',  -- similar, translation, part_of, prerequisite, synonym
    weight FLOAT DEFAULT 1.0,  -- Relationship strength (0-1)
    bidirectional BOOLEAN DEFAULT TRUE,
    user_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(source_concept_id, target_concept_id, relation_type)
);

-- Indexes for edge queries
CREATE INDEX IF NOT EXISTS idx_weave_edges_source ON weave_edges(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_weave_edges_target ON weave_edges(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_weave_edges_user ON weave_edges(user_id);
CREATE INDEX IF NOT EXISTS idx_weave_edges_type ON weave_edges(relation_type);

-- Trigger for updating weave_concepts.updated_at
DROP TRIGGER IF EXISTS update_weave_concepts_updated_at ON weave_concepts;
CREATE TRIGGER update_weave_concepts_updated_at
    BEFORE UPDATE ON weave_concepts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE weave_concepts IS 'WeaveGraph concepts with E5-large multilingual embeddings for cross-language RAG';
COMMENT ON TABLE weave_edges IS 'Semantic relationships between concepts (similarity, translation, etc.)';

-- =============================================================================
-- Source Library Tables (Persistent Source Storage)
-- =============================================================================

-- Sources table - stores uploaded documents, URLs, YouTube videos, notes
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

    -- Unique constraint for name per user
    CONSTRAINT sources_user_name_unique UNIQUE (user_id, name)
);

-- Indexes for sources
CREATE INDEX IF NOT EXISTS idx_sources_user_id ON sources(user_id);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);
CREATE INDEX IF NOT EXISTS idx_sources_source_type ON sources(source_type);
CREATE INDEX IF NOT EXISTS idx_sources_pedagogical_role ON sources(pedagogical_role);
CREATE INDEX IF NOT EXISTS idx_sources_tags ON sources USING GIN(tags);

-- Course-Source links table - many-to-many relationship
CREATE TABLE IF NOT EXISTS course_sources (
    id VARCHAR(36) PRIMARY KEY,
    course_id VARCHAR(36) NOT NULL,
    source_id VARCHAR(36) NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    user_id VARCHAR(36) NOT NULL,
    relevance_score FLOAT,
    is_primary BOOLEAN DEFAULT FALSE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Unique constraint for source per course
    CONSTRAINT unique_course_source UNIQUE (course_id, source_id)
);

-- Indexes for course_sources
CREATE INDEX IF NOT EXISTS idx_course_sources_course_id ON course_sources(course_id);
CREATE INDEX IF NOT EXISTS idx_course_sources_source_id ON course_sources(source_id);
CREATE INDEX IF NOT EXISTS idx_course_sources_user_id ON course_sources(user_id);

-- Source chunks table - vectorized chunks for RAG
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

    -- Unique constraint for chunk per source
    CONSTRAINT unique_source_chunk UNIQUE (source_id, chunk_index)
);

-- Indexes for source_chunks
CREATE INDEX IF NOT EXISTS idx_source_chunks_source_id ON source_chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_source_chunks_embedding ON source_chunks USING ivfflat (embedding vector_cosine_ops);

-- Trigger for updating sources.updated_at
DROP TRIGGER IF EXISTS update_sources_updated_at ON sources;
CREATE TRIGGER update_sources_updated_at
    BEFORE UPDATE ON sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE sources IS 'Source Library - stores uploaded documents, URLs, YouTube videos, and notes for RAG';
COMMENT ON TABLE course_sources IS 'Links sources to courses (many-to-many relationship)';
COMMENT ON TABLE source_chunks IS 'Vectorized chunks of source content for RAG retrieval';
