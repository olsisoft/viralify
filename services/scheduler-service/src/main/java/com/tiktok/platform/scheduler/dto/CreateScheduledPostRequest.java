package com.tiktok.platform.scheduler.dto;

import lombok.*;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CreateScheduledPostRequest {
    private String title;
    private String caption;
    private List<String> hashtags;
    private String videoUrl;
    private Long videoSizeBytes;
    private Integer videoDurationSeconds;
    private String thumbnailUrl;
    private OffsetDateTime scheduledAt;
    private String privacyLevel;
    private Boolean allowComments;
    private Boolean allowDuet;
    private Boolean allowStitch;
    private Boolean commercialContent;
    private Boolean brandedContent;
    @Builder.Default
    private List<String> targetPlatforms = List.of("TIKTOK");
    private Map<String, PlatformSettings> platformSettings;
}
