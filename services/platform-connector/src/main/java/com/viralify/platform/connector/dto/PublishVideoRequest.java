package com.viralify.platform.connector.dto;

import com.viralify.platform.connector.model.Platform;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PublishVideoRequest {
    private UUID userId;
    private UUID scheduledPostId;
    private Platform platform;

    // Content
    private String title;
    private String caption;
    private List<String> hashtags;
    private String videoUrl;
    private Integer videoDurationSeconds;
    private Long videoSizeBytes;

    // Common settings
    private String privacyLevel;
    private boolean allowComments;

    // TikTok specific
    private boolean allowDuet;
    private boolean allowStitch;
    private boolean commercialContent;
    private boolean brandedContent;

    // Instagram specific
    private String locationId;
    private List<String> userTags;

    // YouTube specific
    private String playlistId;
    private String categoryId;
    private String visibility; // public, unlisted, private
    private List<String> tags; // YouTube tags (separate from hashtags)

    // Platform-specific settings as a flexible map
    @Builder.Default
    private Map<String, Object> platformSpecificSettings = Map.of();
}
