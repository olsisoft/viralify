package com.tiktok.platform.scheduler.service;

import com.tiktok.platform.scheduler.dto.*;
import com.tiktok.platform.scheduler.entity.ScheduledPost;
import com.tiktok.platform.scheduler.entity.ScheduledPostPlatform;
import com.tiktok.platform.scheduler.repository.ScheduledPostRepository;
import com.tiktok.platform.scheduler.repository.ScheduledPostPlatformRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class SchedulerService {

    private final ScheduledPostRepository postRepository;
    private final ScheduledPostPlatformRepository platformRepository;
    private final RedisTemplate<String, Object> redisTemplate;
    private final RabbitTemplate rabbitTemplate;
    private final WebClient.Builder webClientBuilder;

    @Value("${tiktok-connector.url}")
    private String tiktokConnectorUrl;

    @Value("${instagram-connector.url:http://localhost:8086}")
    private String instagramConnectorUrl;

    @Value("${youtube-connector.url:http://localhost:8087}")
    private String youtubeConnectorUrl;

    @Value("${platform-connector.url:http://localhost:8088}")
    private String platformConnectorUrl;

    private static final String PUBLISH_QUEUE = "post.publish";
    private static final String RATE_LIMIT_KEY = "ratelimit:publish:";
    private static final Set<String> VALID_PLATFORMS = Set.of("TIKTOK", "INSTAGRAM", "YOUTUBE");

    @Transactional
    public ScheduledPostResponse createScheduledPost(UUID userId, CreateScheduledPostRequest request) {
        if (request.getScheduledAt().isBefore(OffsetDateTime.now().plusMinutes(5))) {
            throw new RuntimeException("Scheduled time must be at least 5 minutes in the future");
        }

        List<String> targetPlatforms = request.getTargetPlatforms();
        if (targetPlatforms == null || targetPlatforms.isEmpty()) {
            targetPlatforms = List.of("TIKTOK");
        }
        for (String platform : targetPlatforms) {
            if (!VALID_PLATFORMS.contains(platform.toUpperCase())) {
                throw new RuntimeException("Invalid platform: " + platform);
            }
        }

        Long postsToday = postRepository.countPublishedSince(
            userId,
            OffsetDateTime.now().truncatedTo(ChronoUnit.DAYS)
        );

        if (postsToday >= 15) {
            throw new RuntimeException("Daily posting limit reached (15 posts/day)");
        }

        ScheduledPost post = ScheduledPost.builder()
                .userId(userId)
                .title(request.getTitle())
                .caption(request.getCaption())
                .hashtags(request.getHashtags())
                .videoUrl(request.getVideoUrl())
                .videoSizeBytes(request.getVideoSizeBytes())
                .videoDurationSeconds(request.getVideoDurationSeconds())
                .thumbnailUrl(request.getThumbnailUrl())
                .scheduledAt(request.getScheduledAt())
                .privacyLevel(request.getPrivacyLevel() != null ? request.getPrivacyLevel() : "PUBLIC_TO_EVERYONE")
                .allowComments(request.getAllowComments() != null ? request.getAllowComments() : true)
                .allowDuet(request.getAllowDuet() != null ? request.getAllowDuet() : true)
                .allowStitch(request.getAllowStitch() != null ? request.getAllowStitch() : true)
                .commercialContent(request.getCommercialContent() != null ? request.getCommercialContent() : false)
                .brandedContent(request.getBrandedContent() != null ? request.getBrandedContent() : false)
                .targetPlatforms(targetPlatforms)
                .status("pending")
                .build();

        post = postRepository.save(post);

        for (String platform : targetPlatforms) {
            ScheduledPostPlatform platformStatus = createPlatformEntry(post, platform, request);
            post.getPlatformStatuses().add(platformStatus);
        }

        post = postRepository.save(post);
        schedulePublishJob(post);

        log.info("Created scheduled post {} for user {} at {} targeting platforms: {}",
                post.getId(), userId, request.getScheduledAt(), targetPlatforms);

        return mapToResponse(post);
    }

    private ScheduledPostPlatform createPlatformEntry(ScheduledPost post, String platform, CreateScheduledPostRequest request) {
        ContentAdaptation adaptation = adaptContentForPlatform(
                post.getTitle(),
                post.getCaption(),
                post.getHashtags(),
                post.getVideoDurationSeconds(),
                platform
        );

        PlatformSettings customSettings = null;
        if (request.getPlatformSettings() != null) {
            customSettings = request.getPlatformSettings().get(platform);
        }

        return ScheduledPostPlatform.builder()
                .scheduledPost(post)
                .platform(platform.toUpperCase())
                .status("pending")
                .adaptedCaption(customSettings != null && customSettings.getCustomCaption() != null
                        ? customSettings.getCustomCaption()
                        : adaptation.getCaption())
                .adaptedHashtags(customSettings != null && customSettings.getCustomHashtags() != null
                        ? customSettings.getCustomHashtags()
                        : adaptation.getHashtags())
                .adaptedTitle(customSettings != null && customSettings.getCustomTitle() != null
                        ? customSettings.getCustomTitle()
                        : adaptation.getTitle())
                .build();
    }

    private ContentAdaptation adaptContentForPlatform(String title, String caption, List<String> hashtags, Integer duration, String platform) {
        try {
            WebClient client = webClientBuilder.baseUrl(platformConnectorUrl).build();

            return client.post()
                    .uri("/api/v1/content/adapt")
                    .bodyValue(Map.of(
                            "title", title != null ? title : "",
                            "caption", caption != null ? caption : "",
                            "hashtags", hashtags != null ? hashtags : List.of(),
                            "durationSeconds", duration != null ? duration : 0,
                            "platform", platform
                    ))
                    .retrieve()
                    .bodyToMono(ContentAdaptation.class)
                    .timeout(Duration.ofSeconds(10))
                    .block();
        } catch (Exception e) {
            log.warn("Failed to adapt content for {}, using original: {}", platform, e.getMessage());
            return ContentAdaptation.builder()
                    .platform(platform)
                    .title(title)
                    .caption(caption)
                    .hashtags(hashtags != null ? hashtags : List.of())
                    .build();
        }
    }

    public List<ScheduledPostResponse> getUserScheduledPosts(UUID userId) {
        return postRepository.findByUserIdOrderByScheduledAtDesc(userId)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    public List<ScheduledPostResponse> getUserPendingPosts(UUID userId) {
        return postRepository.findByUserIdAndStatusOrderByScheduledAtAsc(userId, "pending")
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    public ScheduledPostResponse getScheduledPost(UUID postId, UUID userId) {
        ScheduledPost post = postRepository.findById(postId)
                .orElseThrow(() -> new RuntimeException("Post not found"));

        if (!post.getUserId().equals(userId)) {
            throw new RuntimeException("Access denied");
        }

        return mapToResponse(post);
    }

    @Transactional
    public ScheduledPostResponse updateScheduledPost(UUID postId, UUID userId, UpdateScheduledPostRequest request) {
        ScheduledPost post = postRepository.findById(postId)
                .orElseThrow(() -> new RuntimeException("Post not found"));

        if (!post.getUserId().equals(userId)) {
            throw new RuntimeException("Access denied");
        }

        if (!"pending".equals(post.getStatus())) {
            throw new RuntimeException("Can only update pending posts");
        }

        if (request.getTitle() != null) post.setTitle(request.getTitle());
        if (request.getCaption() != null) post.setCaption(request.getCaption());
        if (request.getHashtags() != null) post.setHashtags(request.getHashtags());
        if (request.getScheduledAt() != null) {
            if (request.getScheduledAt().isBefore(OffsetDateTime.now().plusMinutes(5))) {
                throw new RuntimeException("Scheduled time must be at least 5 minutes in the future");
            }
            post.setScheduledAt(request.getScheduledAt());
        }
        if (request.getPrivacyLevel() != null) post.setPrivacyLevel(request.getPrivacyLevel());
        if (request.getAllowComments() != null) post.setAllowComments(request.getAllowComments());
        if (request.getAllowDuet() != null) post.setAllowDuet(request.getAllowDuet());
        if (request.getAllowStitch() != null) post.setAllowStitch(request.getAllowStitch());

        post = postRepository.save(post);

        return mapToResponse(post);
    }

    @Transactional
    public void cancelScheduledPost(UUID postId, UUID userId) {
        ScheduledPost post = postRepository.findById(postId)
                .orElseThrow(() -> new RuntimeException("Post not found"));

        if (!post.getUserId().equals(userId)) {
            throw new RuntimeException("Access denied");
        }

        if (!"pending".equals(post.getStatus())) {
            throw new RuntimeException("Can only cancel pending posts");
        }

        post.setStatus("cancelled");
        postRepository.save(post);

        log.info("Cancelled scheduled post {}", postId);
    }

    @Scheduled(fixedRate = 60000)
    @Transactional
    public void processScheduledPosts() {
        List<ScheduledPost> postsToPublish = postRepository.findPostsReadyToPublish(OffsetDateTime.now());

        for (ScheduledPost post : postsToPublish) {
            try {
                if (!checkRateLimit(post.getUserId())) {
                    log.warn("Rate limit exceeded for user {}, delaying post {}", post.getUserId(), post.getId());
                    continue;
                }

                post.setStatus("processing");
                postRepository.save(post);

                rabbitTemplate.convertAndSend(PUBLISH_QUEUE, post.getId().toString());

                log.info("Queued post {} for publishing", post.getId());

            } catch (Exception e) {
                log.error("Error processing scheduled post {}: {}", post.getId(), e.getMessage());
                post.setStatus("failed");
                post.setErrorMessage(e.getMessage());
                postRepository.save(post);
            }
        }
    }

    @Scheduled(fixedRate = 300000)
    @Transactional
    public void retryFailedPosts() {
        List<ScheduledPost> postsToRetry = postRepository.findPostsToRetry();

        for (ScheduledPost post : postsToRetry) {
            log.info("Retrying post {} (attempt {})", post.getId(), post.getRetryCount() + 1);

            post.setRetryCount(post.getRetryCount() + 1);
            post.setStatus("pending");
            post.setScheduledAt(OffsetDateTime.now().plusMinutes(5));
            postRepository.save(post);
        }
    }

    @Transactional
    public void processPublishJob(String postIdStr) {
        UUID postId = UUID.fromString(postIdStr);
        ScheduledPost post = postRepository.findById(postId)
                .orElseThrow(() -> new RuntimeException("Post not found: " + postId));

        try {
            post.setStatus("uploading");
            postRepository.save(post);

            List<ScheduledPostPlatform> platformStatuses = post.getPlatformStatuses();

            if (platformStatuses.isEmpty()) {
                PublishResult result = publishToTikTok(post, null);
                handleLegacyPublishResult(post, result);
            } else {
                for (ScheduledPostPlatform platformStatus : platformStatuses) {
                    if (!"pending".equals(platformStatus.getStatus())) {
                        continue;
                    }
                    publishToPlatform(post, platformStatus);
                }
                updateOverallPostStatus(post);
            }

        } catch (Exception e) {
            log.error("Error publishing post {}: {}", postId, e.getMessage());
            handlePublishFailure(post, "INTERNAL_ERROR", e.getMessage());
        }

        postRepository.save(post);
    }

    private void publishToPlatform(ScheduledPost post, ScheduledPostPlatform platformStatus) {
        String platform = platformStatus.getPlatform();
        log.info("Publishing post {} to {}", post.getId(), platform);

        try {
            platformStatus.setStatus("uploading");
            platformRepository.save(platformStatus);

            PublishResult result;
            switch (platform) {
                case "TIKTOK" -> result = publishToTikTok(post, platformStatus);
                case "INSTAGRAM" -> result = publishToInstagram(post, platformStatus);
                case "YOUTUBE" -> result = publishToYouTube(post, platformStatus);
                default -> {
                    log.error("Unknown platform: {}", platform);
                    result = PublishResult.builder()
                            .success(false)
                            .errorCode("UNKNOWN_PLATFORM")
                            .errorMessage("Platform not supported: " + platform)
                            .build();
                }
            }

            if (result.isSuccess()) {
                platformStatus.setStatus("published");
                platformStatus.setPlatformPostId(result.getTiktokPostId() != null ? result.getTiktokPostId() : result.getPublishId());
                platformStatus.setPlatformShareUrl(result.getShareUrl());
                platformStatus.setPublishedAt(OffsetDateTime.now());
                platformStatus.setErrorMessage(null);
                log.info("Successfully published post {} to {}: {}", post.getId(), platform, result.getPublishId());
            } else {
                platformStatus.setStatus("failed");
                platformStatus.setErrorMessage(String.format("[%s] %s", result.getErrorCode(), result.getErrorMessage()));
                platformStatus.setRetryCount(platformStatus.getRetryCount() + 1);
                log.error("Failed to publish post {} to {}: {}", post.getId(), platform, result.getErrorMessage());
            }

            platformRepository.save(platformStatus);

        } catch (Exception e) {
            log.error("Error publishing post {} to {}: {}", post.getId(), platform, e.getMessage());
            platformStatus.setStatus("failed");
            platformStatus.setErrorMessage("[EXCEPTION] " + e.getMessage());
            platformStatus.setRetryCount(platformStatus.getRetryCount() + 1);
            platformRepository.save(platformStatus);
        }
    }

    private PublishResult publishToInstagram(ScheduledPost post, ScheduledPostPlatform platformStatus) {
        WebClient client = webClientBuilder.baseUrl(platformConnectorUrl).build();

        String caption = platformStatus.getAdaptedCaption() != null
                ? platformStatus.getAdaptedCaption()
                : buildCaption(post);

        return client.post()
                .uri("/api/v1/instagram/publish")
                .bodyValue(Map.of(
                        "userId", post.getUserId().toString(),
                        "videoUrl", post.getVideoUrl(),
                        "caption", caption,
                        "hashtags", platformStatus.getAdaptedHashtags() != null
                                ? platformStatus.getAdaptedHashtags()
                                : post.getHashtags()
                ))
                .retrieve()
                .bodyToMono(PublishResult.class)
                .timeout(Duration.ofMinutes(5))
                .onErrorReturn(PublishResult.builder()
                        .success(false)
                        .errorCode("TIMEOUT")
                        .errorMessage("Instagram publish request timed out")
                        .build())
                .block();
    }

    private PublishResult publishToYouTube(ScheduledPost post, ScheduledPostPlatform platformStatus) {
        WebClient client = webClientBuilder.baseUrl(platformConnectorUrl).build();

        String title = platformStatus.getAdaptedTitle() != null
                ? platformStatus.getAdaptedTitle()
                : post.getTitle();
        String description = platformStatus.getAdaptedCaption() != null
                ? platformStatus.getAdaptedCaption()
                : post.getCaption();

        return client.post()
                .uri("/api/v1/youtube/publish")
                .bodyValue(Map.of(
                        "userId", post.getUserId().toString(),
                        "videoUrl", post.getVideoUrl(),
                        "title", title != null ? title : "",
                        "description", description != null ? description : "",
                        "tags", platformStatus.getAdaptedHashtags() != null
                                ? platformStatus.getAdaptedHashtags()
                                : post.getHashtags(),
                        "privacyStatus", mapPrivacyLevelForYouTube(post.getPrivacyLevel())
                ))
                .retrieve()
                .bodyToMono(PublishResult.class)
                .timeout(Duration.ofMinutes(10))
                .onErrorReturn(PublishResult.builder()
                        .success(false)
                        .errorCode("TIMEOUT")
                        .errorMessage("YouTube publish request timed out")
                        .build())
                .block();
    }

    private String mapPrivacyLevelForYouTube(String tiktokPrivacy) {
        return switch (tiktokPrivacy != null ? tiktokPrivacy : "PUBLIC_TO_EVERYONE") {
            case "PUBLIC_TO_EVERYONE" -> "public";
            case "MUTUAL_FOLLOW_FRIENDS", "FOLLOWER_OF_CREATOR" -> "unlisted";
            case "SELF_ONLY" -> "private";
            default -> "public";
        };
    }

    private void handleLegacyPublishResult(ScheduledPost post, PublishResult result) {
        if (result.isSuccess()) {
            post.setStatus("published");
            post.setTiktokPostId(result.getTiktokPostId());
            post.setTiktokShareUrl(result.getShareUrl());
            post.setPublishId(result.getPublishId());
            post.setPublishedAt(OffsetDateTime.now());
            post.setErrorMessage(null);
            log.info("Successfully published post {} to TikTok: {}", post.getId(), result.getTiktokPostId());
            incrementRateLimit(post.getUserId());
        } else {
            handlePublishFailure(post, result.getErrorCode(), result.getErrorMessage());
        }
    }

    private void updateOverallPostStatus(ScheduledPost post) {
        List<ScheduledPostPlatform> statuses = post.getPlatformStatuses();

        long publishedCount = statuses.stream().filter(s -> "published".equals(s.getStatus())).count();
        long failedCount = statuses.stream().filter(s -> "failed".equals(s.getStatus())).count();
        long pendingCount = statuses.stream().filter(s -> "pending".equals(s.getStatus()) || "uploading".equals(s.getStatus())).count();

        if (publishedCount == statuses.size()) {
            post.setStatus("published");
            post.setPublishedAt(OffsetDateTime.now());
            post.setErrorMessage(null);
            log.info("Post {} fully published to all {} platforms", post.getId(), statuses.size());
        } else if (pendingCount == 0 && failedCount > 0 && publishedCount > 0) {
            post.setStatus("partial");
            post.setErrorMessage(String.format("Published to %d/%d platforms", publishedCount, statuses.size()));
            log.warn("Post {} partially published: {}/{} platforms succeeded", post.getId(), publishedCount, statuses.size());
        } else if (pendingCount == 0 && failedCount == statuses.size()) {
            post.setStatus("failed");
            post.setErrorMessage("Failed to publish to all platforms");
            log.error("Post {} failed on all platforms", post.getId());
        }
    }

    private PublishResult publishToTikTok(ScheduledPost post, ScheduledPostPlatform platformStatus) {
        WebClient client = webClientBuilder.baseUrl(tiktokConnectorUrl).build();

        String caption = platformStatus != null && platformStatus.getAdaptedCaption() != null
                ? platformStatus.getAdaptedCaption()
                : buildCaption(post);

        return client.post()
                .uri("/api/v1/publish")
                .bodyValue(PublishRequest.builder()
                        .userId(post.getUserId())
                        .title(post.getTitle())
                        .caption(caption)
                        .videoUrl(post.getVideoUrl())
                        .privacyLevel(post.getPrivacyLevel())
                        .allowComments(post.getAllowComments())
                        .allowDuet(post.getAllowDuet())
                        .allowStitch(post.getAllowStitch())
                        .commercialContent(post.getCommercialContent())
                        .brandedContent(post.getBrandedContent())
                        .build())
                .retrieve()
                .bodyToMono(PublishResult.class)
                .timeout(Duration.ofMinutes(5))
                .onErrorReturn(PublishResult.builder()
                        .success(false)
                        .errorCode("TIMEOUT")
                        .errorMessage("Request timed out")
                        .build())
                .block();
    }

    private String buildCaption(ScheduledPost post) {
        StringBuilder caption = new StringBuilder();

        if (post.getCaption() != null) {
            caption.append(post.getCaption());
        }

        if (post.getHashtags() != null && !post.getHashtags().isEmpty()) {
            caption.append("\n\n");
            for (String hashtag : post.getHashtags()) {
                if (!hashtag.startsWith("#")) {
                    caption.append("#");
                }
                caption.append(hashtag).append(" ");
            }
        }

        return caption.toString().trim();
    }

    private void handlePublishFailure(ScheduledPost post, String errorCode, String errorMessage) {
        post.setStatus("failed");
        post.setErrorMessage(String.format("[%s] %s", errorCode, errorMessage));

        if (post.getRetryCount() < post.getMaxRetries() && isRetryableError(errorCode)) {
            log.info("Post {} failed with retryable error, will retry later", post.getId());
        } else {
            log.error("Post {} failed permanently: {}", post.getId(), errorMessage);
        }
    }

    private boolean isRetryableError(String errorCode) {
        return errorCode != null && (
            errorCode.contains("TIMEOUT") ||
            errorCode.contains("RATE_LIMIT") ||
            errorCode.contains("SERVICE_UNAVAILABLE") ||
            errorCode.contains("500")
        );
    }

    private boolean checkRateLimit(UUID userId) {
        String key = RATE_LIMIT_KEY + userId.toString();
        Long count = redisTemplate.opsForValue().increment(key);

        if (count == 1) {
            redisTemplate.expire(key, Duration.ofMinutes(1));
        }

        return count <= 6;
    }

    private void incrementRateLimit(UUID userId) {
        String dailyKey = RATE_LIMIT_KEY + "daily:" + userId.toString();
        redisTemplate.opsForValue().increment(dailyKey);
        redisTemplate.expire(dailyKey, Duration.ofDays(1));
    }

    private void schedulePublishJob(ScheduledPost post) {
        log.debug("Post {} scheduled for {}", post.getId(), post.getScheduledAt());
    }

    public SchedulerStatsResponse getStats(UUID userId) {
        List<ScheduledPost> userPosts = postRepository.findByUserIdOrderByScheduledAtDesc(userId);

        long pending = userPosts.stream().filter(p -> "pending".equals(p.getStatus())).count();
        long published = userPosts.stream().filter(p -> "published".equals(p.getStatus())).count();
        long failed = userPosts.stream().filter(p -> "failed".equals(p.getStatus())).count();

        OffsetDateTime todayStart = OffsetDateTime.now().truncatedTo(ChronoUnit.DAYS);
        OffsetDateTime weekStart = todayStart.minusDays(7);

        long publishedToday = userPosts.stream()
                .filter(p -> "published".equals(p.getStatus()) &&
                        p.getPublishedAt() != null &&
                        p.getPublishedAt().isAfter(todayStart))
                .count();

        long publishedThisWeek = userPosts.stream()
                .filter(p -> "published".equals(p.getStatus()) &&
                        p.getPublishedAt() != null &&
                        p.getPublishedAt().isAfter(weekStart))
                .count();

        return SchedulerStatsResponse.builder()
                .totalScheduled((long) userPosts.size())
                .pendingCount(pending)
                .publishedCount(published)
                .failedCount(failed)
                .publishedToday(publishedToday)
                .publishedThisWeek(publishedThisWeek)
                .build();
    }

    private ScheduledPostResponse mapToResponse(ScheduledPost post) {
        List<PlatformStatusResponse> platformStatuses = post.getPlatformStatuses() != null
                ? post.getPlatformStatuses().stream()
                        .map(this::mapPlatformStatus)
                        .collect(Collectors.toList())
                : List.of();

        return ScheduledPostResponse.builder()
                .id(post.getId())
                .userId(post.getUserId())
                .title(post.getTitle())
                .caption(post.getCaption())
                .hashtags(post.getHashtags())
                .videoUrl(post.getVideoUrl())
                .thumbnailUrl(post.getThumbnailUrl())
                .scheduledAt(post.getScheduledAt())
                .privacyLevel(post.getPrivacyLevel())
                .allowComments(post.getAllowComments())
                .allowDuet(post.getAllowDuet())
                .allowStitch(post.getAllowStitch())
                .commercialContent(post.getCommercialContent())
                .brandedContent(post.getBrandedContent())
                .status(post.getStatus())
                .targetPlatforms(post.getTargetPlatforms())
                .platformStatuses(platformStatuses)
                .tiktokPostId(post.getTiktokPostId())
                .tiktokShareUrl(post.getTiktokShareUrl())
                .errorMessage(post.getErrorMessage())
                .retryCount(post.getRetryCount())
                .publishedAt(post.getPublishedAt())
                .createdAt(post.getCreatedAt())
                .build();
    }

    private PlatformStatusResponse mapPlatformStatus(ScheduledPostPlatform platform) {
        return PlatformStatusResponse.builder()
                .platform(platform.getPlatform())
                .status(platform.getStatus())
                .platformPostId(platform.getPlatformPostId())
                .platformShareUrl(platform.getPlatformShareUrl())
                .errorMessage(platform.getErrorMessage())
                .retryCount(platform.getRetryCount())
                .adaptedCaption(platform.getAdaptedCaption())
                .adaptedHashtags(platform.getAdaptedHashtags())
                .publishedAt(platform.getPublishedAt())
                .build();
    }
}
