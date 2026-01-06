-- ===========================================
-- TikTok Viral Platform Database Schema
-- ===========================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ===========================================
-- USERS & AUTHENTICATION
-- ===========================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),
    full_name VARCHAR(255),
    avatar_url TEXT,
    tiktok_user_id VARCHAR(100) UNIQUE,
    tiktok_username VARCHAR(100),
    tiktok_display_name VARCHAR(255),
    tiktok_avatar_url TEXT,
    tiktok_follower_count BIGINT DEFAULT 0,
    tiktok_following_count BIGINT DEFAULT 0,
    tiktok_likes_count BIGINT DEFAULT 0,
    access_token_encrypted TEXT,
    refresh_token_encrypted TEXT,
    token_expires_at TIMESTAMP WITH TIME ZONE,
    token_scope TEXT,
    plan_type VARCHAR(50) DEFAULT 'free' CHECK (plan_type IN ('free', 'pro', 'business', 'enterprise')),
    plan_expires_at TIMESTAMP WITH TIME ZONE,
    monthly_posts_limit INTEGER DEFAULT 10,
    monthly_posts_used INTEGER DEFAULT 0,
    monthly_ai_generations_limit INTEGER DEFAULT 50,
    monthly_ai_generations_used INTEGER DEFAULT 0,
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    email_verified BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    device_info JSONB,
    ip_address INET,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    default_privacy_level VARCHAR(50) DEFAULT 'PUBLIC_TO_EVERYONE',
    default_allow_comments BOOLEAN DEFAULT TRUE,
    default_allow_duet BOOLEAN DEFAULT TRUE,
    default_allow_stitch BOOLEAN DEFAULT TRUE,
    auto_add_music BOOLEAN DEFAULT TRUE,
    preferred_posting_times JSONB DEFAULT '[]',
    content_categories JSONB DEFAULT '[]',
    target_audience JSONB DEFAULT '{}',
    brand_voice TEXT,
    notification_email BOOLEAN DEFAULT TRUE,
    notification_push BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- CONTENT & POSTS
-- ===========================================

CREATE TABLE content_drafts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(300),
    caption TEXT,
    script TEXT,
    hashtags JSONB DEFAULT '[]',
    mentions JSONB DEFAULT '[]',
    video_url TEXT,
    video_duration_seconds INTEGER,
    thumbnail_url TEXT,
    ai_suggestions JSONB DEFAULT '{}',
    trend_data JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN ('draft', 'ready', 'scheduled', 'published', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE scheduled_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    draft_id UUID REFERENCES content_drafts(id) ON DELETE SET NULL,
    title VARCHAR(300) NOT NULL,
    caption TEXT,
    hashtags JSONB DEFAULT '[]',
    video_url TEXT NOT NULL,
    video_size_bytes BIGINT,
    video_duration_seconds INTEGER,
    thumbnail_url TEXT,
    scheduled_at TIMESTAMP WITH TIME ZONE NOT NULL,
    privacy_level VARCHAR(50) DEFAULT 'PUBLIC_TO_EVERYONE',
    allow_comments BOOLEAN DEFAULT TRUE,
    allow_duet BOOLEAN DEFAULT TRUE,
    allow_stitch BOOLEAN DEFAULT TRUE,
    commercial_content BOOLEAN DEFAULT FALSE,
    branded_content BOOLEAN DEFAULT FALSE,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'uploading', 'published', 'failed', 'cancelled')),
    -- Multi-platform support
    target_platforms JSONB DEFAULT '["TIKTOK"]',
    tiktok_post_id VARCHAR(100),
    tiktok_share_url TEXT,
    publish_id VARCHAR(100),
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    published_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE post_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID REFERENCES scheduled_posts(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    tiktok_post_id VARCHAR(100),
    views BIGINT DEFAULT 0,
    likes BIGINT DEFAULT 0,
    comments BIGINT DEFAULT 0,
    shares BIGINT DEFAULT 0,
    saves BIGINT DEFAULT 0,
    reach BIGINT DEFAULT 0,
    impressions BIGINT DEFAULT 0,
    engagement_rate DECIMAL(10, 4) DEFAULT 0,
    avg_watch_time_seconds DECIMAL(10, 2) DEFAULT 0,
    completion_rate DECIMAL(10, 4) DEFAULT 0,
    audience_demographics JSONB DEFAULT '{}',
    traffic_sources JSONB DEFAULT '{}',
    captured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- MULTI-PLATFORM SUPPORT
-- ===========================================

-- Platform accounts for multi-platform publishing (TikTok, Instagram, YouTube)
CREATE TABLE platform_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    platform VARCHAR(50) NOT NULL CHECK (platform IN ('TIKTOK', 'INSTAGRAM', 'YOUTUBE')),
    platform_user_id VARCHAR(255) NOT NULL,
    platform_username VARCHAR(255),
    platform_display_name VARCHAR(255),
    platform_avatar_url TEXT,
    follower_count BIGINT DEFAULT 0,
    following_count BIGINT DEFAULT 0,
    likes_count BIGINT DEFAULT 0,
    access_token_encrypted TEXT,
    refresh_token_encrypted TEXT,
    token_expires_at TIMESTAMP WITH TIME ZONE,
    token_scope TEXT,
    account_status VARCHAR(50) DEFAULT 'active' CHECK (account_status IN ('active', 'expired', 'revoked', 'error')),
    last_sync_at TIMESTAMP WITH TIME ZONE,
    platform_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, platform, platform_user_id)
);

-- Per-platform publishing status for scheduled posts
CREATE TABLE scheduled_post_platforms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scheduled_post_id UUID REFERENCES scheduled_posts(id) ON DELETE CASCADE,
    platform VARCHAR(50) NOT NULL CHECK (platform IN ('TIKTOK', 'INSTAGRAM', 'YOUTUBE')),
    platform_account_id UUID REFERENCES platform_accounts(id) ON DELETE SET NULL,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'uploading', 'published', 'failed', 'cancelled', 'skipped')),
    platform_post_id VARCHAR(255),
    platform_share_url TEXT,
    publish_id VARCHAR(255),
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    published_at TIMESTAMP WITH TIME ZONE,
    -- Adapted content for this platform
    adapted_caption TEXT,
    adapted_hashtags JSONB DEFAULT '[]',
    adapted_title VARCHAR(300),
    adapted_duration_seconds INTEGER,
    -- Platform-specific settings
    platform_specific_settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(scheduled_post_id, platform)
);

-- Platform-specific analytics
CREATE TABLE platform_post_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scheduled_post_platform_id UUID REFERENCES scheduled_post_platforms(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    platform VARCHAR(50) NOT NULL CHECK (platform IN ('TIKTOK', 'INSTAGRAM', 'YOUTUBE')),
    platform_post_id VARCHAR(255),
    views BIGINT DEFAULT 0,
    likes BIGINT DEFAULT 0,
    comments BIGINT DEFAULT 0,
    shares BIGINT DEFAULT 0,
    saves BIGINT DEFAULT 0,
    reach BIGINT DEFAULT 0,
    impressions BIGINT DEFAULT 0,
    engagement_rate DECIMAL(10, 4) DEFAULT 0,
    avg_watch_time_seconds DECIMAL(10, 2) DEFAULT 0,
    completion_rate DECIMAL(10, 4) DEFAULT 0,
    -- Platform-specific metrics (e.g., YouTube: subscribers gained, Instagram: profile visits)
    platform_specific_metrics JSONB DEFAULT '{}',
    audience_demographics JSONB DEFAULT '{}',
    traffic_sources JSONB DEFAULT '{}',
    captured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- TRENDS & ANALYSIS
-- ===========================================

CREATE TABLE trending_hashtags (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hashtag VARCHAR(255) NOT NULL,
    hashtag_normalized VARCHAR(255) NOT NULL,
    region VARCHAR(10) DEFAULT 'global',
    category VARCHAR(100),
    view_count BIGINT DEFAULT 0,
    video_count BIGINT DEFAULT 0,
    trend_score DECIMAL(10, 4) DEFAULT 0,
    growth_rate DECIMAL(10, 4) DEFAULT 0,
    peak_time TIMESTAMP WITH TIME ZONE,
    is_trending BOOLEAN DEFAULT TRUE,
    trend_started_at TIMESTAMP WITH TIME ZONE,
    captured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE trending_sounds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sound_id VARCHAR(100) NOT NULL,
    sound_title VARCHAR(500),
    sound_author VARCHAR(255),
    sound_url TEXT,
    duration_seconds INTEGER,
    region VARCHAR(10) DEFAULT 'global',
    category VARCHAR(100),
    usage_count BIGINT DEFAULT 0,
    trend_score DECIMAL(10, 4) DEFAULT 0,
    growth_rate DECIMAL(10, 4) DEFAULT 0,
    is_trending BOOLEAN DEFAULT TRUE,
    captured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE trending_formats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    format_name VARCHAR(255) NOT NULL,
    format_description TEXT,
    format_template JSONB,
    example_videos JSONB DEFAULT '[]',
    region VARCHAR(10) DEFAULT 'global',
    category VARCHAR(100),
    popularity_score DECIMAL(10, 4) DEFAULT 0,
    engagement_avg DECIMAL(10, 4) DEFAULT 0,
    optimal_duration_seconds INTEGER,
    is_trending BOOLEAN DEFAULT TRUE,
    captured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE trend_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    snapshot_type VARCHAR(50) NOT NULL,
    region VARCHAR(10) DEFAULT 'global',
    data JSONB NOT NULL,
    analysis JSONB DEFAULT '{}',
    captured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- AI AGENTS & GENERATIONS
-- ===========================================

CREATE TABLE ai_agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    agent_type VARCHAR(50) NOT NULL CHECK (agent_type IN ('trend_analyzer', 'script_generator', 'optimizer', 'strategy', 'moderator')),
    model_provider VARCHAR(50) DEFAULT 'openai',
    model_name VARCHAR(100) DEFAULT 'gpt-4',
    system_prompt TEXT,
    capabilities JSONB DEFAULT '[]',
    config JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE ai_conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES ai_agents(id) ON DELETE SET NULL,
    title VARCHAR(255),
    context JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE ai_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES ai_conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    tool_calls JSONB DEFAULT '[]',
    tool_results JSONB DEFAULT '[]',
    tokens_used INTEGER DEFAULT 0,
    model_used VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE ai_generations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES ai_agents(id) ON DELETE SET NULL,
    generation_type VARCHAR(50) NOT NULL CHECK (generation_type IN ('script', 'caption', 'hashtags', 'hook', 'strategy', 'optimization')),
    input_data JSONB NOT NULL,
    output_data JSONB NOT NULL,
    model_used VARCHAR(100),
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    latency_ms INTEGER,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- NOTIFICATIONS & EVENTS
-- ===========================================

CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT,
    data JSONB DEFAULT '{}',
    is_read BOOLEAN DEFAULT FALSE,
    read_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE event_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    event_source VARCHAR(100) NOT NULL,
    event_data JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- INDEXES
-- ===========================================

-- Users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_tiktok_user_id ON users(tiktok_user_id);
CREATE INDEX idx_users_plan_type ON users(plan_type);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires ON user_sessions(expires_at);

-- Content
CREATE INDEX idx_content_drafts_user_id ON content_drafts(user_id);
CREATE INDEX idx_content_drafts_status ON content_drafts(status);
CREATE INDEX idx_scheduled_posts_user_id ON scheduled_posts(user_id);
CREATE INDEX idx_scheduled_posts_status ON scheduled_posts(status);
CREATE INDEX idx_scheduled_posts_scheduled_at ON scheduled_posts(scheduled_at) WHERE status = 'pending';
CREATE INDEX idx_post_analytics_post_id ON post_analytics(post_id);
CREATE INDEX idx_post_analytics_user_id ON post_analytics(user_id);
CREATE INDEX idx_post_analytics_captured_at ON post_analytics(captured_at);

-- Platform Accounts (Multi-platform)
CREATE INDEX idx_platform_accounts_user_id ON platform_accounts(user_id);
CREATE INDEX idx_platform_accounts_platform ON platform_accounts(platform);
CREATE INDEX idx_platform_accounts_user_platform ON platform_accounts(user_id, platform);
CREATE INDEX idx_platform_accounts_status ON platform_accounts(account_status);

-- Scheduled Post Platforms (Multi-platform)
CREATE INDEX idx_scheduled_post_platforms_post_id ON scheduled_post_platforms(scheduled_post_id);
CREATE INDEX idx_scheduled_post_platforms_platform ON scheduled_post_platforms(platform);
CREATE INDEX idx_scheduled_post_platforms_status ON scheduled_post_platforms(status);
CREATE INDEX idx_scheduled_post_platforms_pending ON scheduled_post_platforms(scheduled_post_id, platform) WHERE status = 'pending';

-- Platform Post Analytics
CREATE INDEX idx_platform_post_analytics_post_platform_id ON platform_post_analytics(scheduled_post_platform_id);
CREATE INDEX idx_platform_post_analytics_user_id ON platform_post_analytics(user_id);
CREATE INDEX idx_platform_post_analytics_platform ON platform_post_analytics(platform);

-- Trends
CREATE INDEX idx_trending_hashtags_region ON trending_hashtags(region);
CREATE INDEX idx_trending_hashtags_trending ON trending_hashtags(is_trending, trend_score DESC);
CREATE INDEX idx_trending_hashtags_captured ON trending_hashtags(captured_at);
CREATE INDEX idx_trending_sounds_region ON trending_sounds(region);
CREATE INDEX idx_trending_sounds_trending ON trending_sounds(is_trending, trend_score DESC);
CREATE INDEX idx_trending_formats_region ON trending_formats(region);

-- AI
CREATE INDEX idx_ai_conversations_user_id ON ai_conversations(user_id);
CREATE INDEX idx_ai_messages_conversation_id ON ai_messages(conversation_id);
CREATE INDEX idx_ai_generations_user_id ON ai_generations(user_id);
CREATE INDEX idx_ai_generations_type ON ai_generations(generation_type);

-- Notifications
CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_unread ON notifications(user_id, is_read) WHERE is_read = FALSE;
CREATE INDEX idx_event_logs_user_id ON event_logs(user_id);
CREATE INDEX idx_event_logs_type ON event_logs(event_type);
CREATE INDEX idx_event_logs_created ON event_logs(created_at);

-- ===========================================
-- TRIGGERS
-- ===========================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_content_drafts_updated_at BEFORE UPDATE ON content_drafts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_scheduled_posts_updated_at BEFORE UPDATE ON scheduled_posts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_agents_updated_at BEFORE UPDATE ON ai_agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_conversations_updated_at BEFORE UPDATE ON ai_conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_platform_accounts_updated_at BEFORE UPDATE ON platform_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_scheduled_post_platforms_updated_at BEFORE UPDATE ON scheduled_post_platforms
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- SEED DATA: AI AGENTS
-- ===========================================

INSERT INTO ai_agents (name, description, agent_type, model_provider, model_name, system_prompt, capabilities, config) VALUES
(
    'TrendScout',
    'Analyzes TikTok trends and identifies viral patterns',
    'trend_analyzer',
    'openai',
    'gpt-4-turbo-preview',
    'You are TrendScout, an expert AI agent specialized in analyzing TikTok trends. Your role is to:
1. Identify emerging trends before they peak
2. Analyze viral patterns in successful content
3. Predict trend longevity and relevance
4. Match trends to user niches and audiences
5. Provide actionable insights for content creation

Always provide data-driven insights with specific examples and metrics when available.',
    '["trend_detection", "pattern_analysis", "viral_prediction", "niche_matching"]',
    '{"temperature": 0.7, "max_tokens": 2000}'
),
(
    'ScriptGenius',
    'Generates engaging TikTok video scripts with viral hooks',
    'script_generator',
    'anthropic',
    'claude-3-opus-20240229',
    'You are ScriptGenius, a creative AI agent specialized in writing TikTok video scripts. Your expertise includes:
1. Creating attention-grabbing hooks in the first 3 seconds
2. Writing scripts optimized for different video lengths (15s, 30s, 60s, 3min)
3. Incorporating trending formats and sounds
4. Crafting compelling storytelling arcs
5. Including clear calls-to-action
6. Balancing entertainment with value delivery

Always consider the target audience, current trends, and platform best practices.',
    '["script_writing", "hook_creation", "storytelling", "cta_optimization"]',
    '{"temperature": 0.8, "max_tokens": 3000}'
),
(
    'ContentOptimizer',
    'Optimizes content for maximum engagement and reach',
    'optimizer',
    'openai',
    'gpt-4-turbo-preview',
    'You are ContentOptimizer, an AI agent focused on maximizing TikTok content performance. Your capabilities:
1. Optimize captions for discoverability and engagement
2. Suggest optimal posting times based on audience analytics
3. Recommend relevant hashtags with the right mix of popular and niche tags
4. Analyze video length optimization for specific content types
5. Provide A/B testing recommendations
6. Suggest content improvements based on performance data

Base recommendations on platform algorithms, audience behavior, and proven strategies.',
    '["caption_optimization", "hashtag_strategy", "timing_analysis", "ab_testing"]',
    '{"temperature": 0.6, "max_tokens": 1500}'
),
(
    'StrategyAdvisor',
    'Develops comprehensive content strategies for growth',
    'strategy',
    'openai',
    'gpt-4-turbo-preview',
    'You are StrategyAdvisor, a strategic AI agent for TikTok growth. Your expertise covers:
1. Developing long-term content calendars
2. Creating niche positioning strategies
3. Analyzing competitor content and strategies
4. Building audience engagement plans
5. Setting and tracking growth KPIs
6. Recommending collaboration opportunities
7. Planning content series and campaigns

Provide strategic, actionable advice that balances short-term wins with long-term growth.',
    '["strategy_planning", "competitor_analysis", "kpi_tracking", "campaign_planning"]',
    '{"temperature": 0.7, "max_tokens": 2500}'
),
(
    'ContentModerator',
    'Reviews content for compliance and brand safety',
    'moderator',
    'openai',
    'gpt-4-turbo-preview',
    'You are ContentModerator, an AI agent ensuring content compliance. Your responsibilities:
1. Check content against TikTok community guidelines
2. Identify potential copyright issues
3. Flag inappropriate or risky content
4. Ensure commercial content disclosure compliance
5. Verify brand safety standards
6. Suggest content modifications for compliance

Always prioritize safety while maintaining creative expression.',
    '["guideline_compliance", "copyright_check", "brand_safety", "content_review"]',
    '{"temperature": 0.3, "max_tokens": 1000}'
);

-- ===========================================
-- CONTENT GENERATION (AI Media)
-- ===========================================

-- Jobs de génération asynchrones
CREATE TABLE content_generation_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    job_type VARCHAR(50) NOT NULL CHECK (job_type IN ('image', 'video', 'voiceover', 'article', 'thumbnail', 'video_composition')),
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    priority INTEGER DEFAULT 0,
    input_data JSONB NOT NULL,
    output_data JSONB,
    provider VARCHAR(50), -- 'dalle3', 'midjourney', 'elevenlabs', 'openai_tts', 'runway', 'pexels'
    provider_job_id VARCHAR(255),
    credits_used INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Assets générés (images, vidéos, audio)
CREATE TABLE generated_assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    job_id UUID REFERENCES content_generation_jobs(id) ON DELETE SET NULL,
    asset_type VARCHAR(50) NOT NULL CHECK (asset_type IN ('image', 'video', 'audio', 'article', 'thumbnail')),
    storage_url TEXT NOT NULL,
    storage_provider VARCHAR(50) DEFAULT 's3',
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),
    duration_seconds DECIMAL(10, 2),
    dimensions JSONB, -- {width, height}
    metadata JSONB DEFAULT '{}',
    prompt TEXT,
    style_preset VARCHAR(100),
    is_public BOOLEAN DEFAULT FALSE,
    is_favorite BOOLEAN DEFAULT FALSE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Templates vidéo pré-construits
CREATE TABLE video_templates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    thumbnail_url TEXT,
    template_data JSONB NOT NULL, -- Timeline structure, layers, effects
    duration_seconds INTEGER,
    aspect_ratio VARCHAR(10) DEFAULT '9:16',
    is_premium BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compositions vidéo utilisateur
CREATE TABLE video_compositions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    template_id UUID REFERENCES video_templates(id) ON DELETE SET NULL,
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN ('draft', 'rendering', 'completed', 'failed')),
    duration_seconds INTEGER,
    resolution VARCHAR(20) DEFAULT '1080x1920',
    aspect_ratio VARCHAR(10) DEFAULT '9:16',
    composition_data JSONB NOT NULL, -- Timeline, layers, effects
    output_url TEXT,
    render_job_id UUID REFERENCES content_generation_jobs(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Articles/Blog générés
CREATE TABLE generated_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50) CHECK (content_type IN ('blog_post', 'thread', 'newsletter', 'linkedin_post', 'caption_long')),
    word_count INTEGER,
    reading_time_minutes INTEGER,
    seo_keywords JSONB DEFAULT '[]',
    meta_description TEXT,
    source_script_id UUID REFERENCES content_drafts(id) ON DELETE SET NULL,
    status VARCHAR(50) DEFAULT 'draft' CHECK (status IN ('draft', 'published', 'archived')),
    published_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- FAME COACHING SYSTEM
-- ===========================================

-- Badges et Achievements (créé avant missions pour la FK)
CREATE TABLE badges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    icon_url TEXT,
    category VARCHAR(50) CHECK (category IN ('milestone', 'streak', 'skill', 'special', 'challenge')),
    rarity VARCHAR(20) DEFAULT 'common' CHECK (rarity IN ('common', 'rare', 'epic', 'legendary')),
    xp_value INTEGER DEFAULT 0,
    requirements JSONB NOT NULL, -- {type: 'views_total', threshold: 100000}
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Profil de compétences utilisateur
CREATE TABLE user_skill_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE UNIQUE,
    current_level VARCHAR(50) DEFAULT 'beginner' CHECK (current_level IN ('beginner', 'creator', 'rising_star', 'influencer', 'celebrity')),
    experience_points INTEGER DEFAULT 0,
    level_progress_percent DECIMAL(5, 2) DEFAULT 0,
    skills JSONB DEFAULT '{"content_creation": 0, "engagement": 0, "consistency": 0, "trend_usage": 0, "storytelling": 0}',
    strengths JSONB DEFAULT '[]',
    areas_to_improve JSONB DEFAULT '[]',
    niche VARCHAR(100),
    content_style VARCHAR(100),
    onboarding_completed BOOLEAN DEFAULT FALSE,
    last_skill_assessment_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Plans de croissance personnalisés
CREATE TABLE growth_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    plan_type VARCHAR(50) NOT NULL CHECK (plan_type IN ('30_day', '60_day', '90_day', 'custom')),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    goals JSONB NOT NULL, -- [{metric: 'followers', target: 10000, current: 5000}]
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'abandoned')),
    progress_percent DECIMAL(5, 2) DEFAULT 0,
    milestones JSONB DEFAULT '[]', -- [{title, target_date, completed}]
    ai_recommendations JSONB DEFAULT '[]',
    weekly_focus JSONB DEFAULT '[]', -- Focus areas per week
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Missions de coaching (templates)
CREATE TABLE coaching_missions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    description TEXT,
    mission_type VARCHAR(50) NOT NULL CHECK (mission_type IN ('daily', 'weekly', 'challenge', 'bonus', 'onboarding')),
    category VARCHAR(50) CHECK (category IN ('content', 'engagement', 'growth', 'learning', 'collaboration')),
    difficulty VARCHAR(20) DEFAULT 'medium' CHECK (difficulty IN ('easy', 'medium', 'hard', 'expert')),
    xp_reward INTEGER NOT NULL,
    requirements JSONB NOT NULL, -- {action: 'post_video', count: 1, conditions: {...}}
    badge_reward_id UUID REFERENCES badges(id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT TRUE,
    min_level VARCHAR(50) DEFAULT 'beginner',
    order_index INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Progression des missions utilisateur
CREATE TABLE user_missions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    mission_id UUID REFERENCES coaching_missions(id) ON DELETE CASCADE,
    plan_id UUID REFERENCES growth_plans(id) ON DELETE SET NULL,
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'expired', 'skipped')),
    progress JSONB DEFAULT '{}', -- {current: 2, target: 5, details: {...}}
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    xp_earned INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Badges débloqués par utilisateur
CREATE TABLE user_badges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    badge_id UUID REFERENCES badges(id) ON DELETE CASCADE,
    earned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    progress_snapshot JSONB, -- State when badge was earned
    is_displayed BOOLEAN DEFAULT TRUE, -- Show on profile
    UNIQUE(user_id, badge_id)
);

-- Streaks utilisateur
CREATE TABLE user_streaks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    streak_type VARCHAR(50) NOT NULL CHECK (streak_type IN ('daily_post', 'engagement', 'login', 'mission_complete')),
    current_count INTEGER DEFAULT 0,
    longest_count INTEGER DEFAULT 0,
    last_activity_date DATE,
    streak_started_at DATE,
    freeze_count INTEGER DEFAULT 0, -- Streak freezes available
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, streak_type)
);

-- Tips de coaching personnalisés
CREATE TABLE coaching_tips (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    tip_type VARCHAR(50) NOT NULL CHECK (tip_type IN ('post_feedback', 'trend_alert', 'engagement_tip', 'growth_insight', 'reminder', 'celebration')),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    context_data JSONB, -- {post_id, performance_data, etc.}
    priority INTEGER DEFAULT 0, -- Higher = more important
    is_read BOOLEAN DEFAULT FALSE,
    is_dismissed BOOLEAN DEFAULT FALSE,
    action_url TEXT,
    action_label VARCHAR(100),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Historique XP
CREATE TABLE xp_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    xp_amount INTEGER NOT NULL,
    source_type VARCHAR(50) NOT NULL CHECK (source_type IN ('mission', 'badge', 'streak', 'post', 'engagement', 'bonus')),
    source_id UUID, -- Reference to mission_id, badge_id, etc.
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- ANALYTICS & VIRAL PREDICTION
-- ===========================================

-- Benchmarks par niche
CREATE TABLE niche_benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    niche VARCHAR(100) NOT NULL,
    platform VARCHAR(50) NOT NULL CHECK (platform IN ('TIKTOK', 'INSTAGRAM', 'YOUTUBE', 'ALL')),
    follower_range VARCHAR(50), -- '0-1k', '1k-10k', '10k-100k', etc.
    avg_views INTEGER,
    avg_likes INTEGER,
    avg_comments INTEGER,
    avg_shares INTEGER,
    avg_engagement_rate DECIMAL(10, 4),
    avg_completion_rate DECIMAL(10, 4),
    top_posting_times JSONB, -- [{day, hour, engagement_multiplier}]
    top_hashtags JSONB,
    top_content_types JSONB,
    sample_size INTEGER,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Prédictions virales
CREATE TABLE viral_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_draft_id UUID REFERENCES content_drafts(id) ON DELETE CASCADE,
    viral_score DECIMAL(5, 2), -- 0-100
    confidence DECIMAL(5, 2), -- Model confidence
    predicted_views_low INTEGER,
    predicted_views_high INTEGER,
    factors JSONB NOT NULL, -- {hook_strength: 8, trend_alignment: 7, ...}
    recommendations JSONB DEFAULT '[]', -- Suggestions to improve score
    model_version VARCHAR(50),
    actual_performance JSONB, -- Filled after posting
    prediction_accuracy DECIMAL(5, 2), -- Calculated after actual
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analyse des gaps de contenu
CREATE TABLE content_gap_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    niche VARCHAR(100),
    analysis_type VARCHAR(50) CHECK (analysis_type IN ('topic_gap', 'format_gap', 'timing_gap', 'audience_gap')),
    gaps_identified JSONB NOT NULL, -- Topics/formats user hasn't covered
    opportunities JSONB NOT NULL, -- High-potential opportunities
    competitor_data JSONB, -- What competitors are doing well
    recommendations JSONB NOT NULL,
    priority_score DECIMAL(5, 2), -- How urgent to address
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Snapshots de performance utilisateur
CREATE TABLE user_performance_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    period_type VARCHAR(20) CHECK (period_type IN ('daily', 'weekly', 'monthly')),
    metrics JSONB NOT NULL, -- {views, likes, followers, engagement_rate, posts_count}
    vs_previous_period JSONB, -- {views_change: +15%, likes_change: -5%}
    vs_niche_benchmark JSONB, -- {engagement_vs_avg: +2.5%}
    rank_in_niche INTEGER,
    insights JSONB DEFAULT '[]', -- AI-generated insights
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ===========================================
-- INDEXES pour nouvelles tables
-- ===========================================

-- Content Generation
CREATE INDEX idx_content_generation_jobs_user_id ON content_generation_jobs(user_id);
CREATE INDEX idx_content_generation_jobs_status ON content_generation_jobs(status);
CREATE INDEX idx_content_generation_jobs_type ON content_generation_jobs(job_type);
CREATE INDEX idx_generated_assets_user_id ON generated_assets(user_id);
CREATE INDEX idx_generated_assets_type ON generated_assets(asset_type);
CREATE INDEX idx_video_compositions_user_id ON video_compositions(user_id);
CREATE INDEX idx_generated_articles_user_id ON generated_articles(user_id);

-- Coaching
CREATE INDEX idx_user_skill_profiles_user_id ON user_skill_profiles(user_id);
CREATE INDEX idx_user_skill_profiles_level ON user_skill_profiles(current_level);
CREATE INDEX idx_growth_plans_user_id ON growth_plans(user_id);
CREATE INDEX idx_growth_plans_status ON growth_plans(status);
CREATE INDEX idx_coaching_missions_type ON coaching_missions(mission_type);
CREATE INDEX idx_coaching_missions_active ON coaching_missions(is_active, mission_type);
CREATE INDEX idx_user_missions_user_id ON user_missions(user_id);
CREATE INDEX idx_user_missions_status ON user_missions(status);
CREATE INDEX idx_user_badges_user_id ON user_badges(user_id);
CREATE INDEX idx_user_streaks_user_id ON user_streaks(user_id);
CREATE INDEX idx_coaching_tips_user_id ON coaching_tips(user_id);
CREATE INDEX idx_coaching_tips_unread ON coaching_tips(user_id, is_read) WHERE is_read = FALSE;
CREATE INDEX idx_xp_history_user_id ON xp_history(user_id);

-- Analytics
CREATE INDEX idx_niche_benchmarks_niche ON niche_benchmarks(niche);
CREATE INDEX idx_niche_benchmarks_platform ON niche_benchmarks(platform);
CREATE INDEX idx_viral_predictions_user_id ON viral_predictions(user_id);
CREATE INDEX idx_viral_predictions_draft_id ON viral_predictions(content_draft_id);
CREATE INDEX idx_content_gap_analyses_user_id ON content_gap_analyses(user_id);
CREATE INDEX idx_user_performance_snapshots_user_id ON user_performance_snapshots(user_id);
CREATE INDEX idx_user_performance_snapshots_date ON user_performance_snapshots(snapshot_date);

-- ===========================================
-- TRIGGERS pour nouvelles tables
-- ===========================================

CREATE TRIGGER update_content_generation_jobs_updated_at BEFORE UPDATE ON content_generation_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_video_templates_updated_at BEFORE UPDATE ON video_templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_video_compositions_updated_at BEFORE UPDATE ON video_compositions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_generated_articles_updated_at BEFORE UPDATE ON generated_articles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_skill_profiles_updated_at BEFORE UPDATE ON user_skill_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_growth_plans_updated_at BEFORE UPDATE ON growth_plans
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_streaks_updated_at BEFORE UPDATE ON user_streaks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- SEED DATA: BADGES
-- ===========================================

INSERT INTO badges (name, description, icon_url, category, rarity, xp_value, requirements) VALUES
-- Milestone badges
('First Post', 'Published your first video', '/badges/first-post.svg', 'milestone', 'common', 50, '{"type": "posts_count", "threshold": 1}'),
('Content Creator', 'Published 10 videos', '/badges/content-creator.svg', 'milestone', 'common', 100, '{"type": "posts_count", "threshold": 10}'),
('Prolific Creator', 'Published 50 videos', '/badges/prolific.svg', 'milestone', 'rare', 500, '{"type": "posts_count", "threshold": 50}'),
('First 1K Views', 'Reached 1,000 total views', '/badges/1k-views.svg', 'milestone', 'common', 100, '{"type": "views_total", "threshold": 1000}'),
('10K Club', 'Reached 10,000 total views', '/badges/10k-views.svg', 'milestone', 'rare', 250, '{"type": "views_total", "threshold": 10000}'),
('100K Club', 'Reached 100,000 total views', '/badges/100k-views.svg', 'milestone', 'epic', 1000, '{"type": "views_total", "threshold": 100000}'),
('Viral Sensation', 'Reached 1,000,000 total views', '/badges/1m-views.svg', 'milestone', 'legendary', 5000, '{"type": "views_total", "threshold": 1000000}'),
('Rising Star', 'Gained 1,000 followers', '/badges/rising-star.svg', 'milestone', 'rare', 500, '{"type": "followers", "threshold": 1000}'),
('Influencer', 'Gained 10,000 followers', '/badges/influencer.svg', 'milestone', 'epic', 2000, '{"type": "followers", "threshold": 10000}'),

-- Streak badges
('On Fire', '7-day posting streak', '/badges/on-fire.svg', 'streak', 'common', 150, '{"type": "streak", "streak_type": "daily_post", "threshold": 7}'),
('Unstoppable', '30-day posting streak', '/badges/unstoppable.svg', 'streak', 'epic', 1000, '{"type": "streak", "streak_type": "daily_post", "threshold": 30}'),
('Legend', '100-day posting streak', '/badges/legend.svg', 'streak', 'legendary', 5000, '{"type": "streak", "streak_type": "daily_post", "threshold": 100}'),

-- Skill badges
('Engagement Pro', 'Achieved 5% average engagement rate', '/badges/engagement-pro.svg', 'skill', 'rare', 300, '{"type": "engagement_rate", "threshold": 5.0}'),
('Hook Master', '80% average 3-second retention', '/badges/hook-master.svg', 'skill', 'epic', 500, '{"type": "retention_3s", "threshold": 80}'),
('Trend Rider', 'Used 10 trending sounds', '/badges/trend-rider.svg', 'skill', 'rare', 200, '{"type": "trending_sounds_used", "threshold": 10}'),

-- Challenge badges
('Weekly Warrior', 'Completed 4 weekly challenges', '/badges/weekly-warrior.svg', 'challenge', 'rare', 400, '{"type": "weekly_challenges", "threshold": 4}'),
('Mission Master', 'Completed 50 missions', '/badges/mission-master.svg', 'challenge', 'epic', 1000, '{"type": "missions_completed", "threshold": 50}');

-- ===========================================
-- SEED DATA: COACHING MISSIONS
-- ===========================================

INSERT INTO coaching_missions (title, description, mission_type, category, difficulty, xp_reward, requirements, min_level, order_index) VALUES
-- Daily missions
('Post a Video', 'Create and publish one video today', 'daily', 'content', 'easy', 50, '{"action": "post_video", "count": 1}', 'beginner', 1),
('Engage with Community', 'Reply to 5 comments on your videos', 'daily', 'engagement', 'easy', 30, '{"action": "reply_comments", "count": 5}', 'beginner', 2),
('Watch Trends', 'Check trending hashtags and sounds', 'daily', 'learning', 'easy', 20, '{"action": "view_trends", "count": 1}', 'beginner', 3),
('Optimize a Caption', 'Use AI to optimize a caption', 'daily', 'content', 'easy', 25, '{"action": "optimize_caption", "count": 1}', 'beginner', 4),

-- Weekly missions
('Consistency King', 'Post at least 5 videos this week', 'weekly', 'content', 'medium', 200, '{"action": "post_video", "count": 5, "period": "week"}', 'beginner', 1),
('Trend Surfer', 'Use 3 different trending sounds', 'weekly', 'content', 'medium', 150, '{"action": "use_trending_sound", "count": 3, "period": "week"}', 'creator', 2),
('Engagement Champion', 'Achieve 5% engagement on one post', 'weekly', 'engagement', 'hard', 300, '{"action": "engagement_rate", "threshold": 5.0, "period": "week"}', 'creator', 3),
('AI Power User', 'Generate 10 scripts with AI', 'weekly', 'learning', 'medium', 100, '{"action": "generate_script", "count": 10, "period": "week"}', 'beginner', 4),

-- Challenges
('7-Day Challenge', 'Post every day for 7 days', 'challenge', 'content', 'hard', 500, '{"action": "streak", "streak_type": "daily_post", "count": 7}', 'beginner', 1),
('Viral Attempt', 'Get 10,000 views on a single video', 'challenge', 'growth', 'expert', 1000, '{"action": "single_post_views", "threshold": 10000}', 'creator', 2),
('Collaboration Quest', 'Duet or stitch with another creator', 'challenge', 'collaboration', 'medium', 200, '{"action": "collaboration", "count": 1}', 'creator', 3),

-- Onboarding missions
('Complete Your Profile', 'Add your niche and content style', 'onboarding', 'learning', 'easy', 100, '{"action": "complete_profile", "fields": ["niche", "content_style"]}', 'beginner', 1),
('Connect TikTok', 'Link your TikTok account', 'onboarding', 'learning', 'easy', 150, '{"action": "connect_platform", "platform": "TIKTOK"}', 'beginner', 2),
('First AI Script', 'Generate your first AI script', 'onboarding', 'learning', 'easy', 75, '{"action": "generate_script", "count": 1}', 'beginner', 3);

-- ===========================================
-- SEED DATA: VIDEO TEMPLATES
-- ===========================================

INSERT INTO video_templates (name, description, category, template_data, duration_seconds, aspect_ratio, is_premium) VALUES
('Hook + Value', 'Attention-grabbing hook followed by valuable content', 'educational',
 '{"sections": [{"type": "hook", "duration": 3}, {"type": "problem", "duration": 5}, {"type": "solution", "duration": 15}, {"type": "cta", "duration": 3}]}',
 26, '9:16', false),
('Story Time', 'Narrative storytelling format', 'storytelling',
 '{"sections": [{"type": "setup", "duration": 5}, {"type": "buildup", "duration": 10}, {"type": "climax", "duration": 8}, {"type": "conclusion", "duration": 5}]}',
 28, '9:16', false),
('Tutorial Quick', 'Fast-paced how-to tutorial', 'tutorial',
 '{"sections": [{"type": "intro", "duration": 2}, {"type": "step1", "duration": 5}, {"type": "step2", "duration": 5}, {"type": "step3", "duration": 5}, {"type": "result", "duration": 3}]}',
 20, '9:16', false),
('Before/After', 'Transformation showcase', 'transformation',
 '{"sections": [{"type": "before", "duration": 5}, {"type": "process", "duration": 10}, {"type": "after", "duration": 5}, {"type": "reaction", "duration": 3}]}',
 23, '9:16', false),
('POV Story', 'First-person perspective narrative', 'entertainment',
 '{"sections": [{"type": "pov_setup", "duration": 3}, {"type": "scenario", "duration": 12}, {"type": "twist", "duration": 5}, {"type": "reaction", "duration": 3}]}',
 23, '9:16', true),
('Product Review', 'Quick product showcase and review', 'review',
 '{"sections": [{"type": "hook", "duration": 2}, {"type": "overview", "duration": 5}, {"type": "pros", "duration": 8}, {"type": "cons", "duration": 5}, {"type": "verdict", "duration": 5}]}',
 25, '9:16', false),
('Day in Life', 'Daily routine vlog style', 'lifestyle',
 '{"sections": [{"type": "morning", "duration": 8}, {"type": "midday", "duration": 8}, {"type": "evening", "duration": 8}, {"type": "reflection", "duration": 3}]}',
 27, '9:16', true);

-- ===========================================
-- SEED DATA: NICHE BENCHMARKS
-- ===========================================

INSERT INTO niche_benchmarks (niche, platform, follower_range, avg_views, avg_likes, avg_comments, avg_engagement_rate, avg_completion_rate, top_posting_times, calculated_at) VALUES
('fitness', 'TIKTOK', '1k-10k', 5000, 400, 25, 8.5, 45.0, '[{"day": "Monday", "hour": 7}, {"day": "Wednesday", "hour": 18}, {"day": "Saturday", "hour": 10}]', NOW()),
('fitness', 'TIKTOK', '10k-100k', 25000, 1800, 120, 7.8, 42.0, '[{"day": "Monday", "hour": 7}, {"day": "Wednesday", "hour": 18}, {"day": "Saturday", "hour": 10}]', NOW()),
('cooking', 'TIKTOK', '1k-10k', 8000, 600, 40, 8.0, 55.0, '[{"day": "Sunday", "hour": 12}, {"day": "Tuesday", "hour": 19}, {"day": "Friday", "hour": 18}]', NOW()),
('cooking', 'TIKTOK', '10k-100k', 40000, 2800, 180, 7.5, 52.0, '[{"day": "Sunday", "hour": 12}, {"day": "Tuesday", "hour": 19}, {"day": "Friday", "hour": 18}]', NOW()),
('comedy', 'TIKTOK', '1k-10k', 12000, 1000, 80, 9.0, 65.0, '[{"day": "Friday", "hour": 20}, {"day": "Saturday", "hour": 21}, {"day": "Sunday", "hour": 15}]', NOW()),
('comedy', 'TIKTOK', '10k-100k', 60000, 4500, 350, 8.2, 60.0, '[{"day": "Friday", "hour": 20}, {"day": "Saturday", "hour": 21}, {"day": "Sunday", "hour": 15}]', NOW()),
('education', 'TIKTOK', '1k-10k', 3500, 250, 35, 8.2, 48.0, '[{"day": "Monday", "hour": 8}, {"day": "Wednesday", "hour": 12}, {"day": "Thursday", "hour": 19}]', NOW()),
('fashion', 'TIKTOK', '1k-10k', 7000, 580, 45, 9.0, 50.0, '[{"day": "Tuesday", "hour": 18}, {"day": "Thursday", "hour": 19}, {"day": "Saturday", "hour": 14}]', NOW()),
('beauty', 'TIKTOK', '1k-10k', 9000, 750, 55, 9.0, 52.0, '[{"day": "Tuesday", "hour": 18}, {"day": "Thursday", "hour": 19}, {"day": "Saturday", "hour": 14}]', NOW()),
('tech', 'TIKTOK', '1k-10k', 4500, 320, 50, 8.3, 45.0, '[{"day": "Monday", "hour": 12}, {"day": "Wednesday", "hour": 18}, {"day": "Friday", "hour": 17}]', NOW());
