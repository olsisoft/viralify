-- =============================================================================
-- Create lecture_components table for lecture editing feature
-- Run this script on your PostgreSQL database
-- =============================================================================

-- Function for updating updated_at timestamp (if not exists)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Table pour stocker les composants de lecture (pour édition)
CREATE TABLE IF NOT EXISTS lecture_components (
    id VARCHAR(255) PRIMARY KEY,
    lecture_id VARCHAR(255) NOT NULL,
    job_id VARCHAR(255) NOT NULL,
    slides_json JSONB NOT NULL DEFAULT '[]',
    voiceover_json JSONB,
    generation_params_json JSONB NOT NULL DEFAULT '{}',
    total_duration FLOAT NOT NULL DEFAULT 0.0,
    video_url TEXT,
    presentation_job_id VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'completed',
    is_edited BOOLEAN NOT NULL DEFAULT FALSE,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index pour recherche par lecture_id (requête la plus fréquente)
CREATE INDEX IF NOT EXISTS idx_lecture_components_lecture_id ON lecture_components(lecture_id);

-- Index pour recherche par job_id
CREATE INDEX IF NOT EXISTS idx_lecture_components_job_id ON lecture_components(job_id);

-- Index pour composants édités
CREATE INDEX IF NOT EXISTS idx_lecture_components_edited ON lecture_components(is_edited) WHERE is_edited = TRUE;

-- Trigger pour mettre à jour updated_at automatiquement
DROP TRIGGER IF EXISTS update_lecture_components_updated_at ON lecture_components;
CREATE TRIGGER update_lecture_components_updated_at
    BEFORE UPDATE ON lecture_components
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Commentaire sur la table
COMMENT ON TABLE lecture_components IS 'Stores editable lecture components for video editing and regeneration';

-- Verification
SELECT 'lecture_components table created successfully!' as status;
