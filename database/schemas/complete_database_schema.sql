-- StreamVault Pro - Complete Database Schema
-- Production-ready IPTV platform database

-- ==========================================
-- USERS & AUTHENTICATION
-- ==========================================

CREATE DATABASE streamvault_pro;
\c streamvault_pro;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    date_of_birth DATE,
    country_code CHAR(2),
    language_preference VARCHAR(10) DEFAULT 'en',
    profile_image_url TEXT,
    subscription_tier VARCHAR(20) DEFAULT 'free',
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User devices
CREATE TABLE user_devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id VARCHAR(255) UNIQUE NOT NULL,
    device_name VARCHAR(100),
    device_type VARCHAR(50), -- mobile, tablet, tv, desktop
    platform VARCHAR(50), -- ios, android, web, smart_tv
    app_version VARCHAR(20),
    last_active_at TIMESTAMP WITH TIME ZONE,
    is_registered BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================
-- SUBSCRIPTIONS & BILLING
-- ==========================================

-- Subscription plans
CREATE TABLE subscription_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    price_monthly DECIMAL(10,2),
    price_yearly DECIMAL(10,2),
    features JSONB, -- JSON array of features
    max_devices INTEGER DEFAULT 3,
    max_concurrent_streams INTEGER DEFAULT 2,
    video_quality VARCHAR(20) DEFAULT 'HD', -- SD, HD, 4K
    has_ads BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User subscriptions
CREATE TABLE user_subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan_id UUID NOT NULL REFERENCES subscription_plans(id),
    status VARCHAR(20) DEFAULT 'active', -- active, cancelled, expired, suspended
    billing_cycle VARCHAR(20) DEFAULT 'monthly', -- monthly, yearly
    price_paid DECIMAL(10,2),
    starts_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ends_at TIMESTAMP WITH TIME ZONE,
    auto_renew BOOLEAN DEFAULT true,
    payment_method_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Payment methods
CREATE TABLE payment_methods (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(20) NOT NULL, -- card, paypal, bank_transfer
    provider VARCHAR(50), -- stripe, paypal
    provider_payment_id VARCHAR(255),
    card_last_four VARCHAR(4),
    card_brand VARCHAR(20),
    card_exp_month INTEGER,
    card_exp_year INTEGER,
    is_default BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Payment transactions
CREATE TABLE payment_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    subscription_id UUID REFERENCES user_subscriptions(id),
    payment_method_id UUID REFERENCES payment_methods(id),
    amount DECIMAL(10,2) NOT NULL,
    currency CHAR(3) DEFAULT 'USD',
    status VARCHAR(20) DEFAULT 'pending', -- pending, completed, failed, refunded
    transaction_type VARCHAR(20), -- subscription, addon, ppv
    provider_transaction_id VARCHAR(255),
    failure_reason TEXT,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================
-- CONTENT MANAGEMENT
-- ==========================================

-- Content categories
CREATE TABLE content_categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    parent_id UUID REFERENCES content_categories(id),
    sort_order INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Content providers (channels, studios)
CREATE TABLE content_providers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    logo_url TEXT,
    website_url TEXT,
    country_code CHAR(2),
    contact_email VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Live TV channels
CREATE TABLE live_channels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_number INTEGER,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    category_id UUID REFERENCES content_categories(id),
    provider_id UUID REFERENCES content_providers(id),
    logo_url TEXT,
    stream_url TEXT NOT NULL,
    backup_stream_url TEXT,
    quality VARCHAR(20) DEFAULT 'HD',
    language VARCHAR(10),
    country_code CHAR(2),
    is_adult BOOLEAN DEFAULT false,
    is_premium BOOLEAN DEFAULT false,
    sort_order INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- VOD content (movies, series, episodes)
CREATE TABLE vod_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    slug VARCHAR(500) UNIQUE NOT NULL,
    description TEXT,
    content_type VARCHAR(20) NOT NULL, -- movie, series, episode, documentary
    category_id UUID REFERENCES content_categories(id),
    provider_id UUID REFERENCES content_providers(id),
    parent_id UUID REFERENCES vod_content(id), -- for episodes linking to series
    season_number INTEGER,
    episode_number INTEGER,
    duration_minutes INTEGER,
    release_date DATE,
    rating VARCHAR(10), -- G, PG, PG-13, R, etc.
    imdb_id VARCHAR(20),
    tmdb_id VARCHAR(20),
    poster_url TEXT,
    backdrop_url TEXT,
    trailer_url TEXT,
    genres JSONB, -- array of genre strings
    cast_crew JSONB, -- array of cast/crew objects
    languages JSONB, -- array of available languages
    subtitles JSONB, -- array of subtitle languages
    is_premium BOOLEAN DEFAULT false,
    is_adult BOOLEAN DEFAULT false,
    view_count INTEGER DEFAULT 0,
    rating_average DECIMAL(3,2) DEFAULT 0,
    rating_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- VOD content files (different qualities/formats)
CREATE TABLE vod_content_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES vod_content(id) ON DELETE CASCADE,
    file_url TEXT NOT NULL,
    quality VARCHAR(20) NOT NULL, -- 240p, 360p, 480p, 720p, 1080p, 4K
    format VARCHAR(20) DEFAULT 'mp4', -- mp4, hls, dash
    size_bytes BIGINT,
    duration_seconds INTEGER,
    bitrate_kbps INTEGER,
    codec VARCHAR(50),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================
-- USER INTERACTIONS & ANALYTICS
-- ==========================================

-- User favorites/watchlist
CREATE TABLE user_favorites (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_type VARCHAR(20) NOT NULL, -- channel, movie, series
    content_id UUID NOT NULL, -- references live_channels.id or vod_content.id
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, content_type, content_id)
);

-- User viewing history
CREATE TABLE viewing_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_type VARCHAR(20) NOT NULL, -- channel, movie, episode
    content_id UUID NOT NULL,
    device_id UUID REFERENCES user_devices(id),
    watch_duration_seconds INTEGER DEFAULT 0,
    total_duration_seconds INTEGER,
    progress_percentage DECIMAL(5,2) DEFAULT 0,
    quality_watched VARCHAR(20),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_position_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User ratings and reviews
CREATE TABLE user_ratings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID NOT NULL REFERENCES vod_content(id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 10),
    review_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, content_id)
);

-- Content recommendations (ML-generated)
CREATE TABLE content_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID NOT NULL REFERENCES vod_content(id),
    recommendation_type VARCHAR(50), -- collaborative, content_based, trending, etc.
    score DECIMAL(5,4), -- 0.0 to 1.0
    reason TEXT, -- explanation for recommendation
    is_shown BOOLEAN DEFAULT false,
    is_clicked BOOLEAN DEFAULT false,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- ==========================================
-- LIVE TV PROGRAM GUIDE (EPG)
-- ==========================================

-- EPG programs
CREATE TABLE epg_programs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    channel_id UUID NOT NULL REFERENCES live_channels(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    episode_info VARCHAR(200), -- Season 1, Episode 5
    rating VARCHAR(10),
    actors JSONB,
    directors JSONB,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    duration_minutes INTEGER,
    is_live BOOLEAN DEFAULT false,
    is_repeat BOOLEAN DEFAULT false,
    poster_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==========================================
-- STREAMING ANALYTICS
-- ==========================================

-- Real-time streaming sessions
CREATE TABLE streaming_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id),
    device_id UUID REFERENCES user_devices(id),
    content_type VARCHAR(20) NOT NULL, -- channel, movie, episode
    content_id UUID NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    quality VARCHAR(20),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    total_duration_seconds INTEGER DEFAULT 0,
    bandwidth_kbps INTEGER,
    buffer_events INTEGER DEFAULT 0,
    error_events INTEGER DEFAULT 0,
    cdn_server VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true
);

-- CDN performance metrics
CREATE TABLE cdn_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    server_name VARCHAR(100) NOT NULL,
    region VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    bandwidth_mbps DECIMAL(10,2),
    concurrent_users INTEGER,
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_percent DECIMAL(5,2),
    response_time_ms INTEGER,
    error_rate_percent DECIMAL(5,4)
);

-- ==========================================
-- ADMIN & CONFIGURATION
-- ==========================================

-- Admin users
CREATE TABLE admin_users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(200),
    role VARCHAR(50) DEFAULT 'admin', -- super_admin, admin, moderator, support
    permissions JSONB, -- array of permission strings
    is_active BOOLEAN DEFAULT true,
    last_login_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System configuration
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(100) UNIQUE NOT NULL,
    value TEXT,
    description TEXT,
    data_type VARCHAR(20) DEFAULT 'string', -- string, integer, boolean, json
    is_public BOOLEAN DEFAULT false, -- can be accessed by frontend
    updated_by UUID REFERENCES admin_users(id),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API rate limiting
CREATE TABLE api_rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL, -- user_id, ip_address, api_key
    endpoint VARCHAR(255) NOT NULL,
    request_count INTEGER DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(identifier, endpoint, window_start)
);

-- ==========================================
-- INDEXES FOR PERFORMANCE
-- ==========================================

-- Users indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_subscription_tier ON users(subscription_tier);
CREATE INDEX idx_users_country_code ON users(country_code);

-- Sessions indexes
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires ON user_sessions(expires_at);

-- Content indexes
CREATE INDEX idx_live_channels_active ON live_channels(is_active);
CREATE INDEX idx_live_channels_category ON live_channels(category_id);
CREATE INDEX idx_live_channels_provider ON live_channels(provider_id);
CREATE INDEX idx_live_channels_country ON live_channels(country_code);

CREATE INDEX idx_vod_content_type ON vod_content(content_type);
CREATE INDEX idx_vod_content_category ON vod_content(category_id);
CREATE INDEX idx_vod_content_active ON vod_content(is_active);
CREATE INDEX idx_vod_content_premium ON vod_content(is_premium);
CREATE INDEX idx_vod_content_rating ON vod_content(rating_average);
CREATE INDEX idx_vod_content_views ON vod_content(view_count);

-- Full-text search indexes
CREATE INDEX idx_vod_content_title_search ON vod_content USING GIN(to_tsvector('english', title));
CREATE INDEX idx_vod_content_description_search ON vod_content USING GIN(to_tsvector('english', description));

-- Analytics indexes
CREATE INDEX idx_viewing_history_user ON viewing_history(user_id, started_at);
CREATE INDEX idx_viewing_history_content ON viewing_history(content_id, started_at);
CREATE INDEX idx_streaming_sessions_active ON streaming_sessions(is_active, start_time);

-- EPG indexes
CREATE INDEX idx_epg_programs_channel_time ON epg_programs(channel_id, start_time, end_time);
CREATE INDEX idx_epg_programs_time_range ON epg_programs(start_time, end_time);

-- ==========================================
-- SAMPLE DATA INSERTS
-- ==========================================

-- Insert sample subscription plans
INSERT INTO subscription_plans (name, description, price_monthly, price_yearly, features, max_devices, max_concurrent_streams, video_quality) VALUES 
('Free', 'Basic access with ads', 0, 0, '["Limited channels", "SD quality", "Ads included"]', 1, 1, 'SD'),
('Basic', 'Essential streaming package', 19.99, 199.99, '["500+ channels", "HD quality", "Limited ads", "7-day catch-up"]', 3, 2, 'HD'),
('Premium', 'Full streaming experience', 39.99, 399.99, '["1000+ channels", "4K quality", "No ads", "30-day catch-up", "Sports packages"]', 5, 4, '4K'),
('Enterprise', 'Commercial license', 99.99, 999.99, '["All content", "Multi-location", "Priority support", "Analytics dashboard"]', 20, 10, '4K');

-- Insert sample categories
INSERT INTO content_categories (name, slug, description) VALUES 
('Entertainment', 'entertainment', 'Movies, TV shows, and general entertainment'),
('Sports', 'sports', 'Live sports and sports-related content'),
('News', 'news', 'News channels and current affairs'),
('Kids', 'kids', 'Family-friendly content for children'),
('Music', 'music', 'Music videos and music-related content'),
('International', 'international', 'International channels and content');

-- Insert sample content providers
INSERT INTO content_providers (name, slug, description, country_code) VALUES 
('StreamVault Studios', 'streamvault-studios', 'Original content production', 'US'),
('Global Sports Network', 'global-sports', 'International sports content', 'UK'),
('WorldNews Corp', 'worldnews', 'Global news coverage', 'US'),
('Kids Entertainment Co', 'kids-entertainment', 'Family programming', 'CA');

-- Insert system configuration
INSERT INTO system_config (key, value, description, is_public) VALUES 
('platform_name', 'StreamVault Pro', 'Platform branding name', true),
('max_concurrent_streams_free', '1', 'Max streams for free users', false),
('video_transcode_qualities', '["240p", "360p", "480p", "720p", "1080p", "4K"]', 'Available video qualities', true),
('supported_languages', '["en", "es", "fr", "de", "pt", "it", "ru", "zh", "ja", "ko"]', 'Supported platform languages', true),
('customer_support_email', 'support@streamvault.pro', 'Customer support contact', true),
('free_trial_duration_days', '30', 'Free trial period length', true);

-- ==========================================
-- FUNCTIONS AND TRIGGERS
-- ==========================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_subscriptions_updated_at BEFORE UPDATE ON user_subscriptions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_vod_content_updated_at BEFORE UPDATE ON vod_content FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_ratings_updated_at BEFORE UPDATE ON user_ratings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update content rating average
CREATE OR REPLACE FUNCTION update_content_rating()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE vod_content 
    SET rating_average = (
        SELECT COALESCE(AVG(rating), 0) 
        FROM user_ratings 
        WHERE content_id = COALESCE(NEW.content_id, OLD.content_id)
    ),
    rating_count = (
        SELECT COUNT(*) 
        FROM user_ratings 
        WHERE content_id = COALESCE(NEW.content_id, OLD.content_id)
    )
    WHERE id = COALESCE(NEW.content_id, OLD.content_id);
    
    RETURN COALESCE(NEW, OLD);
END;
$$ language 'plpgsql';

-- Trigger for content rating updates
CREATE TRIGGER update_content_rating_trigger 
    AFTER INSERT OR UPDATE OR DELETE ON user_ratings 
    FOR EACH ROW EXECUTE FUNCTION update_content_rating();

-- Function to log viewing activity
CREATE OR REPLACE FUNCTION log_viewing_activity()
RETURNS TRIGGER AS $$
BEGIN
    -- Update view count for VOD content
    IF NEW.content_type IN ('movie', 'episode') THEN
        UPDATE vod_content 
        SET view_count = view_count + 1 
        WHERE id = NEW.content_id;
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for viewing history
CREATE TRIGGER log_viewing_activity_trigger 
    AFTER INSERT ON viewing_history 
    FOR EACH ROW EXECUTE FUNCTION log_viewing_activity();

COMMENT ON DATABASE streamvault_pro IS 'StreamVault Pro IPTV Platform Database - Complete production schema with all business logic, analytics, and optimization features';