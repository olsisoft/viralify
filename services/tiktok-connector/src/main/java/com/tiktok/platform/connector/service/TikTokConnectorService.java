package com.tiktok.platform.connector.service;

import com.tiktok.platform.connector.dto.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class TikTokConnectorService {

    private final WebClient.Builder webClientBuilder;
    private final RedisTemplate<String, Object> redisTemplate;
    private final RestTemplate restTemplate;

    @Value("${tiktok.client-key}")
    private String clientKey;

    @Value("${tiktok.client-secret}")
    private String clientSecret;

    private static final String TIKTOK_API_BASE = "https://open.tiktokapis.com/v2";
    private static final String CONTENT_POSTING_BASE = TIKTOK_API_BASE + "/post/publish";

    // ========================================
    // Video Publishing - Content Posting API
    // ========================================

    public PublishResult publishVideo(PublishVideoRequest request) {
        log.info("Publishing video for user: {}", request.getUserId());

        try {
            // Step 1: Get valid access token
            String accessToken = getValidAccessToken(request.getUserId());

            // Step 2: Get creator info to check permissions
            CreatorInfo creatorInfo = getCreatorInfo(accessToken);
            validatePrivacyLevel(request.getPrivacyLevel(), creatorInfo.getPrivacyLevelOptions());

            // Step 3: Initialize video upload
            VideoInitResponse initResponse = initializeVideoUpload(accessToken, request);

            if (initResponse.getError() != null && initResponse.getError().getCode() != null) {
                return PublishResult.builder()
                        .success(false)
                        .errorCode(initResponse.getError().getCode())
                        .errorMessage(initResponse.getError().getMessage())
                        .build();
            }

            // Step 4: For PULL_FROM_URL, the video is fetched by TikTok
            // We just need to return the publish_id for status tracking
            return PublishResult.builder()
                    .success(true)
                    .publishId(initResponse.getData().getPublishId())
                    .build();

        } catch (Exception e) {
            log.error("Error publishing video: {}", e.getMessage(), e);
            return PublishResult.builder()
                    .success(false)
                    .errorCode("INTERNAL_ERROR")
                    .errorMessage(e.getMessage())
                    .build();
        }
    }

    private VideoInitResponse initializeVideoUpload(String accessToken, PublishVideoRequest request) {
        WebClient client = webClientBuilder.build();

        Map<String, Object> postInfo = new HashMap<>();
        postInfo.put("title", request.getTitle());
        postInfo.put("privacy_level", request.getPrivacyLevel());
        postInfo.put("disable_comment", !request.isAllowComments());
        postInfo.put("disable_duet", !request.isAllowDuet());
        postInfo.put("disable_stitch", !request.isAllowStitch());
        postInfo.put("video_cover_timestamp_ms", 1000);

        if (request.isCommercialContent()) {
            postInfo.put("brand_content_toggle", true);
            postInfo.put("brand_organic_toggle", !request.isBrandedContent());
        }

        Map<String, Object> sourceInfo = new HashMap<>();
        sourceInfo.put("source", "PULL_FROM_URL");
        sourceInfo.put("video_url", request.getVideoUrl());

        Map<String, Object> body = new HashMap<>();
        body.put("post_info", postInfo);
        body.put("source_info", sourceInfo);

        return client.post()
                .uri(CONTENT_POSTING_BASE + "/video/init/")
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(VideoInitResponse.class)
                .timeout(Duration.ofSeconds(30))
                .block();
    }

    // ========================================
    // Publish Status Tracking
    // ========================================

    public PublishStatusResponse getPublishStatus(UUID userId, String publishId) {
        String accessToken = getValidAccessToken(userId);

        WebClient client = webClientBuilder.build();

        return client.post()
                .uri(CONTENT_POSTING_BASE + "/status/fetch/")
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .bodyValue(Map.of("publish_id", publishId))
                .retrieve()
                .bodyToMono(PublishStatusResponse.class)
                .timeout(Duration.ofSeconds(10))
                .block();
    }

    // ========================================
    // Creator Info
    // ========================================

    public CreatorInfo getCreatorInfo(String accessToken) {
        WebClient client = webClientBuilder.build();

        CreatorInfoResponse response = client.post()
                .uri(CONTENT_POSTING_BASE + "/creator_info/query/")
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .retrieve()
                .bodyToMono(CreatorInfoResponse.class)
                .timeout(Duration.ofSeconds(10))
                .block();

        if (response != null && response.getData() != null) {
            return response.getData();
        }

        throw new RuntimeException("Failed to get creator info");
    }

    // ========================================
    // User Info
    // ========================================

    public TikTokUserInfo getUserInfo(UUID userId) {
        String accessToken = getValidAccessToken(userId);

        WebClient client = webClientBuilder.build();

        String fields = "open_id,union_id,avatar_url,display_name,username,follower_count,following_count,likes_count,video_count";

        TikTokUserInfoResponse response = client.get()
                .uri(TIKTOK_API_BASE + "/user/info/?fields=" + fields)
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                .retrieve()
                .bodyToMono(TikTokUserInfoResponse.class)
                .timeout(Duration.ofSeconds(10))
                .block();

        if (response != null && response.getData() != null && response.getData().getUser() != null) {
            return response.getData().getUser();
        }

        throw new RuntimeException("Failed to get user info");
    }

    // ========================================
    // Video List
    // ========================================

    public VideoListResponse getUserVideos(UUID userId, int maxCount, String cursor) {
        String accessToken = getValidAccessToken(userId);

        WebClient client = webClientBuilder.build();

        String fields = "id,title,video_description,duration,cover_image_url,share_url,view_count,like_count,comment_count,share_count,create_time";

        Map<String, Object> body = new HashMap<>();
        body.put("max_count", Math.min(maxCount, 20));
        if (cursor != null) {
            body.put("cursor", cursor);
        }

        return client.post()
                .uri(TIKTOK_API_BASE + "/video/list/?fields=" + fields)
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(VideoListResponse.class)
                .timeout(Duration.ofSeconds(15))
                .block();
    }

    // ========================================
    // Video Analytics (if available)
    // ========================================

    public VideoAnalytics getVideoAnalytics(UUID userId, String videoId) {
        String accessToken = getValidAccessToken(userId);

        // Note: Detailed analytics requires Business API access
        // This returns basic metrics from video/query endpoint

        WebClient client = webClientBuilder.build();

        String fields = "id,view_count,like_count,comment_count,share_count";

        Map<String, Object> body = new HashMap<>();
        body.put("filters", Map.of("video_ids", List.of(videoId)));

        VideoQueryResponse response = client.post()
                .uri(TIKTOK_API_BASE + "/video/query/?fields=" + fields)
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + accessToken)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_JSON_VALUE)
                .bodyValue(body)
                .retrieve()
                .bodyToMono(VideoQueryResponse.class)
                .timeout(Duration.ofSeconds(10))
                .block();

        if (response != null && response.getData() != null && !response.getData().getVideos().isEmpty()) {
            VideoInfo video = response.getData().getVideos().get(0);
            return VideoAnalytics.builder()
                    .videoId(video.getId())
                    .viewCount(video.getViewCount())
                    .likeCount(video.getLikeCount())
                    .commentCount(video.getCommentCount())
                    .shareCount(video.getShareCount())
                    .build();
        }

        throw new RuntimeException("Failed to get video analytics");
    }

    // ========================================
    // Token Management
    // ========================================

    private String getValidAccessToken(UUID userId) {
        // In production, this would fetch from auth-service
        // For now, check Redis cache or call auth service
        String cacheKey = "tiktok:token:" + userId;
        String cachedToken = (String) redisTemplate.opsForValue().get(cacheKey);

        if (cachedToken != null) {
            return cachedToken;
        }

        // Call auth service to get token
        throw new RuntimeException("No valid access token for user: " + userId);
    }

    private void validatePrivacyLevel(String requested, List<String> allowed) {
        if (allowed == null || !allowed.contains(requested)) {
            throw new RuntimeException("Privacy level '" + requested + "' not allowed. Available: " + allowed);
        }
    }

    // ========================================
    // Webhook Handler
    // ========================================

    public void handleWebhook(TikTokWebhookEvent event) {
        log.info("Received TikTok webhook: {}", event.getEvent());

        switch (event.getEvent()) {
            case "post_publish_complete":
                handlePublishComplete(event);
                break;
            case "post_publish_failed":
                handlePublishFailed(event);
                break;
            default:
                log.warn("Unknown webhook event: {}", event.getEvent());
        }
    }

    private void handlePublishComplete(TikTokWebhookEvent event) {
        String publishId = event.getPublishId();
        String postId = event.getPostId();

        log.info("Post published successfully. PublishId: {}, PostId: {}", publishId, postId);

        // Update post status in scheduler service via message queue
        // rabbitTemplate.convertAndSend("post.status.update", ...);
    }

    private void handlePublishFailed(TikTokWebhookEvent event) {
        String publishId = event.getPublishId();
        String failReason = event.getFailReason();

        log.error("Post publish failed. PublishId: {}, Reason: {}", publishId, failReason);

        // Update post status in scheduler service
    }
}
