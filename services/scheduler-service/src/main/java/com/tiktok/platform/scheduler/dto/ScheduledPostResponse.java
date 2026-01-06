package com.tiktok.platform.scheduler.dto;

import lombok.*;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ScheduledPostResponse {
    private UUID id;
    private UUID userId;
    private String title;
    private String caption;
    private List<String> hashtags;
    private String videoUrl;
    private String thumbnailUrl;
    private OffsetDateTime scheduledAt;
    private String privacyLevel;
    private Boolean allowComments;
    private Boolean allowDuet;
    private Boolean allowStitch;
    private Boolean commercialContent;
    private Boolean brandedContent;
    private String status;
    private List<String> targetPlatforms;
    private List<PlatformStatusResponse> platformStatuses;
    private String tiktokPostId;
    private String tiktokShareUrl;
    private String errorMessage;
    private Integer retryCount;
    private OffsetDateTime publishedAt;
    private OffsetDateTime createdAt;
}
