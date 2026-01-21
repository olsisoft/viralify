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
