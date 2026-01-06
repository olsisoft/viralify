package com.viralify.platform.connector.instagram;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.viralify.platform.connector.SocialMediaConnector;
import com.viralify.platform.connector.dto.*;
import com.viralify.platform.connector.model.ContentLimits;
import com.viralify.platform.connector.model.Platform;
import com.viralify.platform.connector.service.ContentAdapterService;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.time.OffsetDateTime;
import java.util.*;

/**
 * Instagram Reels Connector using Instagram Graph API
 * Requires Facebook Business account with Instagram Professional Account linked
 *
 * API Documentation: https://developers.facebook.com/docs/instagram-api/guides/content-publishing
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class InstagramConnectorService implements SocialMediaConnector {

    private final WebClient.Builder webClientBuilder;
    private final RedisTemplate<String, Object> redisTemplate;
    private final ContentAdapterService contentAdapterService;

    @Value("${instagram.app-id:}")
    private String appId;

    @Value("${instagram.app-secret:}")
    private String appSecret;

    private static final String GRAPH_API_BASE = "https://graph.facebook.com/v18.0";
    private static final String INSTAGRAM_API_BASE = "https://graph.instagram.com/v18.0";

    @Override
    public Platform getPlatform() {
        return Platform.INSTAGRAM;
    }

    @Override
    public ContentLimits getContentLimits() {
        return Platform.INSTAGRAM.toContentLimits();
    }

    @Override
    public PublishResult publishVideo(PublishVideoRequest request) {
        log.info("Publishing Instagram Reel for user: {}", request.getUserId());

        try {
            String accessToken = getValidAccessToken(request.getUserId());
            String igUserId = getInstagramUserId(request.getUserId());

            // Adapt content for Instagram
            AdaptedContent adapted = contentAdapterService.adaptForPlatform(
                    request.getTitle(),
                    request.getCaption(),
                    request.getHashtags(),
                    request.getVideoDurationSeconds(),
                    Platform.INSTAGRAM
            );

            // Step 1: Create media container
            String containerId = createMediaContainer(igUserId, accessToken, request, adapted);

            if (containerId == null) {
                return PublishResult.failure(Platform.INSTAGRAM, "CONTAINER_ERROR",
                        "Failed to create media container");
            }

            // Step 2: Wait for media to be ready and publish
            // Instagram processes the video asynchronously
            return PublishResult.builder()
                    .platform(Platform.INSTAGRAM)
                    .success(true)
                    .publishId(containerId)
                    .build();

        } catch (Exception e) {
            log.error("Error publishing Instagram Reel: {}", e.getMessage(), e);
            return PublishResult.failure(Platform.INSTAGRAM, "INTERNAL_ERROR", e.getMessage());
        }
    }

    /**
     * Create a media container for the Reel
     * Instagram API requires two-step process: create container, then publish
     */
    private String createMediaContainer(String igUserId, String accessToken,
                                        PublishVideoRequest request, AdaptedContent adapted) {
        WebClient client = webClientBuilder.build();

        // Build caption with hashtags
        String fullCaption = adapted.getCaptionWithHashtags();

        Map<String, Object> params = new HashMap<>();
        params.put("media_type", "REELS");
        params.put("video_url", request.getVideoUrl());
        params.put("caption", fullCaption);

        // Add location if provided
        if (request.getLocationId() != null) {
            params.put("location_id", request.getLocationId());
        }

        // Add user tags if provided
        if (request.getUserTags() != null && !request.getUserTags().isEmpty()) {
            params.put("user_tags", request.getUserTags());
        }

        // Share to feed (Reels appear in both Reels tab and Feed)
        params.put("share_to_feed", true);

        try {
            InstagramMediaResponse response = client.post()
                    .uri(GRAPH_API_BASE + "/" + igUserId + "/media?access_token=" + accessToken)
                    .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                    .bodyValue(params)
                    .retrieve()
                    .bodyToMono(InstagramMediaResponse.class)
                    .timeout(Duration.ofSeconds(30))
                    .block();

            if (response != null && response.getId() != null) {
                return response.getId();
            }

            log.error("Instagram media container creation failed: {}", response);
            return null;

        } catch (Exception e) {
            log.error("Error creating Instagram media container: {}", e.getMessage(), e);
            return null;
        }
    }

    /**
     * Publish the media container (called after video is processed)
     */
    public PublishResult publishMediaContainer(UUID userId, String containerId) {
        String accessToken = getValidAccessToken(userId);
        String igUserId = getInstagramUserId(userId);

        WebClient client = webClientBuilder.build();

        try {
            // Check if container is ready
            InstagramContainerStatus status = checkContainerStatus(containerId, accessToken);

            if (!"FINISHED".equals(status.getStatusCode())) {
                return PublishResult.builder()
                        .platform(Platform.INSTAGRAM)
                        .success(false)
                        .publishId(containerId)
                        .errorCode("NOT_READY")
                        .errorMessage("Video is still processing: " + status.getStatusCode())
                        .build();
            }

            // Publish the container
            InstagramMediaResponse response = client.post()
                    .uri(GRAPH_API_BASE + "/" + igUserId + "/media_publish?creation_id=" + containerId +
                            "&access_token=" + accessToken)
                    .retrieve()
                    .bodyToMono(InstagramMediaResponse.class)
                    .timeout(Duration.ofSeconds(30))
                    .block();

            if (response != null && response.getId() != null) {
                // Get the media permalink
                String permalink = getMediaPermalink(response.getId(), accessToken);

                return PublishResult.builder()
                        .platform(Platform.INSTAGRAM)
                        .success(true)
                        .publishId(containerId)
                        .platformPostId(response.getId())
                        .shareUrl(permalink)
                        .publishedAt(OffsetDateTime.now())
                        .build();
            }

            return PublishResult.failure(Platform.INSTAGRAM, "PUBLISH_ERROR", "Failed to publish media");

        } catch (Exception e) {
            log.error("Error publishing Instagram media container: {}", e.getMessage(), e);
            return PublishResult.failure(Platform.INSTAGRAM, "INTERNAL_ERROR", e.getMessage());
        }
    }

    /**
     * Check the status of a media container
     */
    private InstagramContainerStatus checkContainerStatus(String containerId, String accessToken) {
        WebClient client = webClientBuilder.build();

        return client.get()
                .uri(GRAPH_API_BASE + "/" + containerId + "?fields=status_code&access_token=" + accessToken)
                .retrieve()
                .bodyToMono(InstagramContainerStatus.class)
                .timeout(Duration.ofSeconds(10))
                .block();
    }

    @Override
    public PublishStatusResponse getPublishStatus(UUID userId, String publishId) {
        String accessToken = getValidAccessToken(userId);

        InstagramContainerStatus status = checkContainerStatus(publishId, accessToken);

        PublishStatusResponse.PublishStatus mappedStatus = switch (status.getStatusCode()) {
            case "FINISHED" -> PublishStatusResponse.PublishStatus.PUBLISHED;
            case "IN_PROGRESS" -> PublishStatusResponse.PublishStatus.PROCESSING;
            case "ERROR" -> PublishStatusResponse.PublishStatus.FAILED;
            default -> PublishStatusResponse.PublishStatus.PENDING;
        };

        return PublishStatusResponse.builder()
                .platform(Platform.INSTAGRAM)
                .publishId(publishId)
                .status(mappedStatus)
                .failReason(status.getStatusCode().equals("ERROR") ? "Instagram processing error" : null)
                .build();
    }

    @Override
    public PlatformUserInfo getUserInfo(UUID userId) {
        String accessToken = getValidAccessToken(userId);
        String igUserId = getInstagramUserId(userId);

        WebClient client = webClientBuilder.build();

        String fields = "id,username,name,profile_picture_url,followers_count,follows_count,media_count";

        InstagramUserInfo response = client.get()
                .uri(GRAPH_API_BASE + "/" + igUserId + "?fields=" + fields + "&access_token=" + accessToken)
                .retrieve()
                .bodyToMono(InstagramUserInfo.class)
                .timeout(Duration.ofSeconds(10))
                .block();

        if (response != null) {
            return PlatformUserInfo.builder()
                    .platform(Platform.INSTAGRAM)
                    .platformUserId(response.getId())
                    .username(response.getUsername())
                    .displayName(response.getName())
                    .avatarUrl(response.getProfilePictureUrl())
                    .followerCount(response.getFollowersCount())
                    .followingCount(response.getFollowsCount())
                    .videoCount(response.getMediaCount())
                    .build();
        }

        throw new RuntimeException("Failed to get Instagram user info");
    }

    @Override
    public VideoAnalytics getVideoAnalytics(UUID userId, String platformPostId) {
        String accessToken = getValidAccessToken(userId);

        WebClient client = webClientBuilder.build();

        String fields = "id,like_count,comments_count,plays,shares";

        InstagramMediaInsights response = client.get()
                .uri(GRAPH_API_BASE + "/" + platformPostId + "?fields=" + fields + "&access_token=" + accessToken)
                .retrieve()
                .bodyToMono(InstagramMediaInsights.class)
                .timeout(Duration.ofSeconds(10))
                .block();

        if (response != null) {
            return VideoAnalytics.builder()
                    .platform(Platform.INSTAGRAM)
                    .platformPostId(platformPostId)
                    .views(response.getPlays())
                    .likes(response.getLikeCount())
                    .comments(response.getCommentsCount())
                    .shares(response.getShares())
                    .capturedAt(OffsetDateTime.now())
                    .build();
        }

        throw new RuntimeException("Failed to get Instagram video analytics");
    }

    @Override
    public ContentValidationResult validateContent(ContentValidationRequest request) {
        ContentLimits limits = getContentLimits();
        ContentValidationResult result = ContentValidationResult.valid(Platform.INSTAGRAM);

        // Check duration (90 seconds max for Reels)
        if (request.getVideoDurationSeconds() != null &&
                request.getVideoDurationSeconds() > limits.getMaxDurationSeconds()) {
            result.addWarning(String.format("Video duration (%ds) exceeds Instagram Reels limit (%ds). Video will be trimmed.",
                    request.getVideoDurationSeconds(), limits.getMaxDurationSeconds()));
            result.setSuggestedDurationSeconds(limits.getMaxDurationSeconds());
        }

        // Check hashtag count (strict 30 limit)
        if (request.getHashtags() != null && request.getHashtags().size() > 30) {
            result.addWarning(String.format("Hashtag count (%d) exceeds Instagram limit (30). Some will be removed.",
                    request.getHashtags().size()));
            result.setSuggestedHashtags(request.getHashtags().subList(0, 30));
        }

        // Check caption length
        if (request.getCaption() != null && request.getCaption().length() > limits.getMaxCaptionLength()) {
            result.addWarning("Caption exceeds Instagram limit and will be truncated.");
        }

        // Check file size
        if (request.getVideoSizeBytes() != null &&
                request.getVideoSizeBytes() > limits.getMaxFileSizeBytes()) {
            result.addError("Video file size exceeds Instagram limit of 4GB.");
        }

        return result;
    }

    @Override
    public void handleWebhook(PlatformWebhookEvent event) {
        log.info("Received Instagram webhook: {}", event.getEventType());
        // Instagram webhooks are handled via Facebook webhook system
        // Implement based on your Facebook App webhook configuration
    }

    @Override
    public boolean hasValidConnection(UUID userId) {
        try {
            String token = getValidAccessToken(userId);
            return token != null && !token.isEmpty();
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public void refreshToken(UUID userId) {
        // Instagram/Facebook long-lived tokens need to be refreshed
        // Call auth service to handle token refresh
        log.info("Refreshing Instagram token for user: {}", userId);
    }

    private String getValidAccessToken(UUID userId) {
        String cacheKey = "instagram:token:" + userId;
        String cachedToken = (String) redisTemplate.opsForValue().get(cacheKey);

        if (cachedToken != null) {
            return cachedToken;
        }

        throw new RuntimeException("No valid Instagram access token for user: " + userId);
    }

    private String getInstagramUserId(UUID userId) {
        String cacheKey = "instagram:user_id:" + userId;
        String cachedId = (String) redisTemplate.opsForValue().get(cacheKey);

        if (cachedId != null) {
            return cachedId;
        }

        throw new RuntimeException("No Instagram user ID found for user: " + userId);
    }

    private String getMediaPermalink(String mediaId, String accessToken) {
        WebClient client = webClientBuilder.build();

        try {
            Map<String, String> response = client.get()
                    .uri(GRAPH_API_BASE + "/" + mediaId + "?fields=permalink&access_token=" + accessToken)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .timeout(Duration.ofSeconds(10))
                    .block();

            if (response != null) {
                return (String) response.get("permalink");
            }
        } catch (Exception e) {
            log.warn("Failed to get Instagram media permalink: {}", e.getMessage());
        }

        return "https://www.instagram.com/reel/" + mediaId;
    }
}

// Instagram API Response DTOs
@Data
@NoArgsConstructor
@AllArgsConstructor
class InstagramMediaResponse {
    private String id;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class InstagramContainerStatus {
    @JsonProperty("status_code")
    private String statusCode; // IN_PROGRESS, FINISHED, ERROR
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class InstagramUserInfo {
    private String id;
    private String username;
    private String name;
    @JsonProperty("profile_picture_url")
    private String profilePictureUrl;
    @JsonProperty("followers_count")
    private Long followersCount;
    @JsonProperty("follows_count")
    private Long followsCount;
    @JsonProperty("media_count")
    private Long mediaCount;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class InstagramMediaInsights {
    private String id;
    @JsonProperty("like_count")
    private Long likeCount;
    @JsonProperty("comments_count")
    private Long commentsCount;
    private Long plays;
    private Long shares;
}
