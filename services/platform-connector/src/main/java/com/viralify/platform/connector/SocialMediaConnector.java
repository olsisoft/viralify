package com.viralify.platform.connector;

import com.viralify.platform.connector.dto.*;
import com.viralify.platform.connector.model.ContentLimits;
import com.viralify.platform.connector.model.Platform;

import java.util.UUID;

/**
 * Common interface for all social media platform connectors.
 * Implementations: TikTokConnector, InstagramConnector, YouTubeConnector
 */
public interface SocialMediaConnector {

    /**
     * Get the platform this connector handles
     */
    Platform getPlatform();

    /**
     * Get content limits for this platform
     */
    ContentLimits getContentLimits();

    /**
     * Publish a video to the platform
     */
    PublishResult publishVideo(PublishVideoRequest request);

    /**
     * Get the status of a publishing operation
     */
    PublishStatusResponse getPublishStatus(UUID userId, String publishId);

    /**
     * Get user information from the platform
     */
    PlatformUserInfo getUserInfo(UUID userId);

    /**
     * Get analytics for a specific video
     */
    VideoAnalytics getVideoAnalytics(UUID userId, String platformPostId);

    /**
     * Validate content before publishing
     */
    ContentValidationResult validateContent(ContentValidationRequest request);

    /**
     * Handle webhook events from the platform
     */
    void handleWebhook(PlatformWebhookEvent event);

    /**
     * Check if the user has a valid connected account for this platform
     */
    boolean hasValidConnection(UUID userId);

    /**
     * Refresh the access token for a user
     */
    void refreshToken(UUID userId);
}
