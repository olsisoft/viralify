package com.viralify.platform.connector.youtube;

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
 * YouTube Shorts Connector using YouTube Data API v3
 * Requires Google Cloud Project with YouTube Data API enabled
 *
 * API Documentation: https://developers.google.com/youtube/v3
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class YouTubeConnectorService implements SocialMediaConnector {

    private final WebClient.Builder webClientBuilder;
    private final RedisTemplate<String, Object> redisTemplate;
    private final ContentAdapterService contentAdapterService;

    @Value("${youtube.api-key:}")
    private String apiKey;

    @Value("${youtube.client-id:}")
    private String clientId;

    @Value("${youtube.client-secret:}")
    private String clientSecret;

    private static final String YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3";
    private static final String YOUTUBE_UPLOAD_BASE = "https://www.googleapis.com/upload/youtube/v3";

    @Override
    public Platform getPlatform() {
        return Platform.YOUTUBE;
    }

    @Override
    public ContentLimits getContentLimits() {
        return Platform.YOUTUBE.toContentLimits();
    }

    @Override
    public PublishResult publishVideo(PublishVideoRequest request) {
        log.info("Publishing YouTube Short for user: {}", request.getUserId());

        try {
            String accessToken = getValidAccessToken(request.getUserId());

            // Adapt content for YouTube
            AdaptedContent adapted = contentAdapterService.adaptForPlatform(
                    request.getTitle(),
                    request.getCaption(),
                    request.getHashtags(),
                    request.getVideoDurationSeconds(),
                    Platform.YOUTUBE
            );

            // Step 1: Initialize resumable upload
            String uploadUrl = initializeResumableUpload(accessToken, request, adapted);

            if (uploadUrl == null) {
                return PublishResult.failure(Platform.YOUTUBE, "UPLOAD_INIT_ERROR",
                        "Failed to initialize upload");
            }

            // For PULL_FROM_URL strategy, we need to download and re-upload
            // YouTube doesn't support direct URL publishing like TikTok
            // In production, this would be handled by a background job

            // Return the upload URL as publishId for tracking
            String uploadId = extractUploadId(uploadUrl);

            return PublishResult.builder()
                    .platform(Platform.YOUTUBE)
                    .success(true)
                    .publishId(uploadId)
                    .build();

        } catch (Exception e) {
            log.error("Error publishing YouTube Short: {}", e.getMessage(), e);
            return PublishResult.failure(Platform.YOUTUBE, "INTERNAL_ERROR", e.getMessage());
        }
    }

    /**
     * Initialize a resumable upload session
     * Returns the upload URL for the video file
     */
    private String initializeResumableUpload(String accessToken, PublishVideoRequest request,
                                             AdaptedContent adapted) {
        WebClient client = webClientBuilder.build();

        // Build video metadata
        Map<String, Object> snippet = new HashMap<>();
        snippet.put("title", adapted.getTitle());
        snippet.put("description", adapted.getDescription());
        snippet.put("categoryId", request.getCategoryId() != null ? request.getCategoryId() : "22"); // People & Blogs

        // Add tags (converted from hashtags)
        if (adapted.getTags() != null && !adapted.getTags().isEmpty()) {
            snippet.put("tags", adapted.getTags());
        }

        // Status settings
        Map<String, Object> status = new HashMap<>();
        status.put("privacyStatus", request.getVisibility() != null ? request.getVisibility() : "public");
        status.put("selfDeclaredMadeForKids", false);

        // For Shorts, we need to indicate it's a short-form video
        // YouTube automatically detects Shorts based on aspect ratio and duration
        // but #Shorts in title/description helps with categorization

        Map<String, Object> body = new HashMap<>();
        body.put("snippet", snippet);
        body.put("status", status);

        try {
            // Initialize resumable upload
            String response = client.post()
                    .uri(YOUTUBE_UPLOAD_BASE + "/videos?uploadType=resumable&part=snippet,status")
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                    .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                    .header("X-Upload-Content-Type", "video/*")
                    .bodyValue(body)
                    .exchangeToMono(clientResponse -> {
                        // Get the upload URL from the Location header
                        String location = clientResponse.headers().header("Location").stream()
                                .findFirst().orElse(null);
                        return reactor.core.publisher.Mono.just(location != null ? location : "");
                    })
                    .timeout(Duration.ofSeconds(30))
                    .block();

            return response;

        } catch (Exception e) {
            log.error("Error initializing YouTube upload: {}", e.getMessage(), e);
            return null;
        }
    }

    /**
     * Complete the video upload (called with actual video data)
     */
    public PublishResult completeUpload(UUID userId, String uploadUrl, byte[] videoData) {
        WebClient client = webClientBuilder.build();

        try {
            YouTubeVideoResponse response = client.put()
                    .uri(uploadUrl)
                    .header(HttpHeaders.CONTENT_TYPE, "video/*")
                    .header(HttpHeaders.CONTENT_LENGTH, String.valueOf(videoData.length))
                    .bodyValue(videoData)
                    .retrieve()
                    .bodyToMono(YouTubeVideoResponse.class)
                    .timeout(Duration.ofMinutes(5))
                    .block();

            if (response != null && response.getId() != null) {
                String shareUrl = "https://www.youtube.com/shorts/" + response.getId();

                return PublishResult.builder()
                        .platform(Platform.YOUTUBE)
                        .success(true)
                        .platformPostId(response.getId())
                        .shareUrl(shareUrl)
                        .publishedAt(OffsetDateTime.now())
                        .build();
            }

            return PublishResult.failure(Platform.YOUTUBE, "UPLOAD_ERROR", "Failed to complete upload");

        } catch (Exception e) {
            log.error("Error completing YouTube upload: {}", e.getMessage(), e);
            return PublishResult.failure(Platform.YOUTUBE, "INTERNAL_ERROR", e.getMessage());
        }
    }

    @Override
    public PublishStatusResponse getPublishStatus(UUID userId, String publishId) {
        String accessToken = getValidAccessToken(userId);

        WebClient client = webClientBuilder.build();

        try {
            YouTubeVideoListResponse response = client.get()
                    .uri(YOUTUBE_API_BASE + "/videos?part=status,processingDetails&id=" + publishId)
                    .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                    .retrieve()
                    .bodyToMono(YouTubeVideoListResponse.class)
                    .timeout(Duration.ofSeconds(10))
                    .block();

            if (response != null && response.getItems() != null && !response.getItems().isEmpty()) {
                YouTubeVideoItem video = response.getItems().get(0);

                PublishStatusResponse.PublishStatus status = mapYouTubeStatus(video);

                return PublishStatusResponse.builder()
                        .platform(Platform.YOUTUBE)
                        .publishId(publishId)
                        .status(status)
                        .platformPostId(video.getId())
                        .shareUrl("https://www.youtube.com/shorts/" + video.getId())
                        .build();
            }

            return PublishStatusResponse.builder()
                    .platform(Platform.YOUTUBE)
                    .publishId(publishId)
                    .status(PublishStatusResponse.PublishStatus.PENDING)
                    .build();

        } catch (Exception e) {
            log.error("Error getting YouTube video status: {}", e.getMessage(), e);
            return PublishStatusResponse.builder()
                    .platform(Platform.YOUTUBE)
                    .publishId(publishId)
                    .status(PublishStatusResponse.PublishStatus.FAILED)
                    .failReason(e.getMessage())
                    .build();
        }
    }

    private PublishStatusResponse.PublishStatus mapYouTubeStatus(YouTubeVideoItem video) {
        if (video.getStatus() != null) {
            String uploadStatus = video.getStatus().getUploadStatus();
            if ("processed".equals(uploadStatus)) {
                return PublishStatusResponse.PublishStatus.PUBLISHED;
            } else if ("failed".equals(uploadStatus) || "rejected".equals(uploadStatus)) {
                return PublishStatusResponse.PublishStatus.FAILED;
            } else if ("uploaded".equals(uploadStatus)) {
                return PublishStatusResponse.PublishStatus.PROCESSING;
            }
        }
        return PublishStatusResponse.PublishStatus.PENDING;
    }

    @Override
    public PlatformUserInfo getUserInfo(UUID userId) {
        String accessToken = getValidAccessToken(userId);

        WebClient client = webClientBuilder.build();

        YouTubeChannelResponse response = client.get()
                .uri(YOUTUBE_API_BASE + "/channels?part=snippet,statistics&mine=true")
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                .retrieve()
                .bodyToMono(YouTubeChannelResponse.class)
                .timeout(Duration.ofSeconds(10))
                .block();

        if (response != null && response.getItems() != null && !response.getItems().isEmpty()) {
            YouTubeChannelItem channel = response.getItems().get(0);

            return PlatformUserInfo.builder()
                    .platform(Platform.YOUTUBE)
                    .platformUserId(channel.getId())
                    .username(channel.getSnippet().getCustomUrl())
                    .displayName(channel.getSnippet().getTitle())
                    .avatarUrl(channel.getSnippet().getThumbnails().getDefaultThumbnail().getUrl())
                    .followerCount(channel.getStatistics().getSubscriberCount())
                    .videoCount(channel.getStatistics().getVideoCount())
                    .likesCount(0L) // YouTube doesn't expose total likes received
                    .platformSpecific(Map.of(
                            "viewCount", channel.getStatistics().getViewCount(),
                            "channelDescription", channel.getSnippet().getDescription()
                    ))
                    .build();
        }

        throw new RuntimeException("Failed to get YouTube channel info");
    }

    @Override
    public VideoAnalytics getVideoAnalytics(UUID userId, String platformPostId) {
        String accessToken = getValidAccessToken(userId);

        WebClient client = webClientBuilder.build();

        YouTubeVideoListResponse response = client.get()
                .uri(YOUTUBE_API_BASE + "/videos?part=statistics&id=" + platformPostId)
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                .retrieve()
                .bodyToMono(YouTubeVideoListResponse.class)
                .timeout(Duration.ofSeconds(10))
                .block();

        if (response != null && response.getItems() != null && !response.getItems().isEmpty()) {
            YouTubeVideoItem video = response.getItems().get(0);
            YouTubeVideoStatistics stats = video.getStatistics();

            return VideoAnalytics.builder()
                    .platform(Platform.YOUTUBE)
                    .platformPostId(platformPostId)
                    .views(stats.getViewCount())
                    .likes(stats.getLikeCount())
                    .comments(stats.getCommentCount())
                    .capturedAt(OffsetDateTime.now())
                    .platformSpecificMetrics(Map.of(
                            "favoriteCount", stats.getFavoriteCount()
                    ))
                    .build();
        }

        throw new RuntimeException("Failed to get YouTube video analytics");
    }

    @Override
    public ContentValidationResult validateContent(ContentValidationRequest request) {
        ContentLimits limits = getContentLimits();
        ContentValidationResult result = ContentValidationResult.valid(Platform.YOUTUBE);

        // Check duration (60 seconds max for Shorts)
        if (request.getVideoDurationSeconds() != null &&
                request.getVideoDurationSeconds() > limits.getMaxDurationSeconds()) {
            result.addWarning(String.format("Video duration (%ds) exceeds YouTube Shorts limit (%ds). Video will be trimmed.",
                    request.getVideoDurationSeconds(), limits.getMaxDurationSeconds()));
            result.setSuggestedDurationSeconds(limits.getMaxDurationSeconds());
        }

        // Check title length
        if (request.getTitle() != null && request.getTitle().length() > limits.getMaxTitleLength()) {
            result.addWarning("Title exceeds YouTube limit (100 chars) and will be truncated.");
        }

        // Check title contains #Shorts
        if (request.getTitle() != null && !request.getTitle().toLowerCase().contains("#shorts")) {
            result.addWarning("Title should include #Shorts for proper categorization.");
        }

        // Check description length
        if (request.getCaption() != null && request.getCaption().length() > limits.getMaxCaptionLength()) {
            result.addWarning("Description exceeds YouTube limit and will be truncated.");
        }

        return result;
    }

    @Override
    public void handleWebhook(PlatformWebhookEvent event) {
        log.info("Received YouTube webhook: {}", event.getEventType());
        // YouTube uses PubSubHubbub for notifications
        // Implement based on your webhook subscription
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
        log.info("Refreshing YouTube token for user: {}", userId);
        // Call auth service to refresh Google OAuth token
    }

    private String getValidAccessToken(UUID userId) {
        String cacheKey = "youtube:token:" + userId;
        String cachedToken = (String) redisTemplate.opsForValue().get(cacheKey);

        if (cachedToken != null) {
            return cachedToken;
        }

        throw new RuntimeException("No valid YouTube access token for user: " + userId);
    }

    private String extractUploadId(String uploadUrl) {
        // Extract upload ID from the resumable upload URL
        if (uploadUrl != null && uploadUrl.contains("upload_id=")) {
            int start = uploadUrl.indexOf("upload_id=") + 10;
            int end = uploadUrl.indexOf("&", start);
            return end > 0 ? uploadUrl.substring(start, end) : uploadUrl.substring(start);
        }
        return UUID.randomUUID().toString();
    }
}

// YouTube API Response DTOs
@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeVideoResponse {
    private String id;
    private String kind;
    private YouTubeVideoSnippet snippet;
    private YouTubeVideoStatus status;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeVideoListResponse {
    private String kind;
    private List<YouTubeVideoItem> items;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeVideoItem {
    private String id;
    private YouTubeVideoSnippet snippet;
    private YouTubeVideoStatus status;
    private YouTubeVideoStatistics statistics;
    private YouTubeProcessingDetails processingDetails;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeVideoSnippet {
    private String title;
    private String description;
    private String categoryId;
    private List<String> tags;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeVideoStatus {
    private String uploadStatus;
    private String privacyStatus;
    private String failureReason;
    private String rejectionReason;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeVideoStatistics {
    private Long viewCount;
    private Long likeCount;
    private Long commentCount;
    private Long favoriteCount;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeProcessingDetails {
    private String processingStatus;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeChannelResponse {
    private String kind;
    private List<YouTubeChannelItem> items;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeChannelItem {
    private String id;
    private YouTubeChannelSnippet snippet;
    private YouTubeChannelStatistics statistics;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeChannelSnippet {
    private String title;
    private String description;
    private String customUrl;
    private YouTubeThumbnails thumbnails;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeThumbnails {
    @JsonProperty("default")
    private YouTubeThumbnail defaultThumbnail;
    private YouTubeThumbnail medium;
    private YouTubeThumbnail high;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeThumbnail {
    private String url;
    private Integer width;
    private Integer height;
}

@Data
@NoArgsConstructor
@AllArgsConstructor
class YouTubeChannelStatistics {
    private Long viewCount;
    private Long subscriberCount;
    private Long videoCount;
}
