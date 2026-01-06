package com.tiktok.platform.scheduler.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Entity
@Table(name = "scheduled_post_platforms",
       uniqueConstraints = @UniqueConstraint(columnNames = {"scheduled_post_id", "platform"}))
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ScheduledPostPlatform {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "scheduled_post_id", nullable = false)
    private ScheduledPost scheduledPost;

    @Column(nullable = false, length = 50)
    private String platform;

    @Column(name = "platform_account_id")
    private UUID platformAccountId;

    @Column(length = 50)
    @Builder.Default
    private String status = "pending";

    @Column(name = "platform_post_id", length = 255)
    private String platformPostId;

    @Column(name = "platform_share_url", columnDefinition = "TEXT")
    private String platformShareUrl;

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @Column(name = "retry_count")
    @Builder.Default
    private Integer retryCount = 0;

    @Column(name = "adapted_caption", columnDefinition = "TEXT")
    private String adaptedCaption;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "adapted_hashtags", columnDefinition = "jsonb")
    private List<String> adaptedHashtags;

    @Column(name = "adapted_title")
    private String adaptedTitle;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "platform_specific_settings", columnDefinition = "jsonb")
    private Map<String, Object> platformSpecificSettings;

    @Column(name = "published_at")
    private OffsetDateTime publishedAt;

    @CreationTimestamp
    @Column(name = "created_at")
    private OffsetDateTime createdAt;
}
