package com.viralify.platform.connector.dto;

import com.viralify.platform.connector.model.Platform;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class VideoAnalytics {
    private Platform platform;
    private String platformPostId;

    // Common metrics
    private Long views;
    private Long likes;
    private Long comments;
    private Long shares;
    private Long saves;
    private Long reach;
    private Long impressions;

    // Engagement
    private BigDecimal engagementRate;
    private BigDecimal avgWatchTimeSeconds;
    private BigDecimal completionRate;

    // Timestamps
    private OffsetDateTime capturedAt;

    // Platform-specific metrics (e.g., YouTube: subscribers gained, Instagram: profile visits)
    @Builder.Default
    private Map<String, Object> platformSpecificMetrics = Map.of();

    // Audience demographics
    @Builder.Default
    private Map<String, Object> audienceDemographics = Map.of();

    // Traffic sources
    @Builder.Default
    private Map<String, Object> trafficSources = Map.of();
}
