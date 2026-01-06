package com.tiktok.platform.scheduler.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Entity
@Table(name = "scheduled_posts")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ScheduledPost {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "user_id", nullable = false)
    private UUID userId;

    @Column(name = "draft_id")
    private UUID draftId;

    @Column(nullable = false, length = 300)
    private String title;

    @Column(columnDefinition = "TEXT")
    private String caption;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(columnDefinition = "jsonb")
    private List<String> hashtags;

    @Column(name = "video_url", nullable = false)
    private String videoUrl;

    @Column(name = "video_size_bytes")
    private Long videoSizeBytes;

    @Column(name = "video_duration_seconds")
    private Integer videoDurationSeconds;

    @Column(name = "thumbnail_url")
    private String thumbnailUrl;

    @Column(name = "scheduled_at", nullable = false)
    private OffsetDateTime scheduledAt;

    @Column(name = "privacy_level")
    @Builder.Default
    private String privacyLevel = "PUBLIC_TO_EVERYONE";

    @Column(name = "allow_comments")
    @Builder.Default
    private Boolean allowComments = true;

    @Column(name = "allow_duet")
    @Builder.Default
    private Boolean allowDuet = true;

    @Column(name = "allow_stitch")
    @Builder.Default
    private Boolean allowStitch = true;

    @Column(name = "commercial_content")
    @Builder.Default
    private Boolean commercialContent = false;

    @Column(name = "branded_content")
    @Builder.Default
    private Boolean brandedContent = false;

    @Column(length = 50)
    @Builder.Default
    private String status = "pending";

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "target_platforms", columnDefinition = "jsonb")
    @Builder.Default
    private List<String> targetPlatforms = List.of("TIKTOK");

    @OneToMany(mappedBy = "scheduledPost", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    @Builder.Default
    private List<ScheduledPostPlatform> platformStatuses = new ArrayList<>();

    @Column(name = "tiktok_post_id", length = 100)
    private String tiktokPostId;

    @Column(name = "tiktok_share_url")
    private String tiktokShareUrl;

    @Column(name = "publish_id", length = 100)
    private String publishId;

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @Column(name = "retry_count")
    @Builder.Default
    private Integer retryCount = 0;

    @Column(name = "max_retries")
    @Builder.Default
    private Integer maxRetries = 3;

    @Column(name = "published_at")
    private OffsetDateTime publishedAt;

    @CreationTimestamp
    @Column(name = "created_at")
    private OffsetDateTime createdAt;

    @UpdateTimestamp
    @Column(name = "updated_at")
    private OffsetDateTime updatedAt;

    public boolean isFullyPublished() {
        if (platformStatuses.isEmpty()) return false;
        return platformStatuses.stream().allMatch(ps -> "published".equals(ps.getStatus()));
    }

    public boolean hasAnyPermanentFailure() {
        return platformStatuses.stream().anyMatch(ps ->
            "failed".equals(ps.getStatus()) && ps.getRetryCount() >= 3);
    }
}
