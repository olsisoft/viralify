package com.viralify.platform.connector.dto;

import com.viralify.platform.connector.model.Platform;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PlatformWebhookEvent {
    private Platform platform;
    private String eventType;
    private String publishId;
    private String platformPostId;
    private String shareUrl;
    private String failReason;
    private Long timestamp;

    // Raw platform-specific data
    @Builder.Default
    private Map<String, Object> rawData = Map.of();

    // Common event types
    public static final String EVENT_PUBLISH_COMPLETE = "publish_complete";
    public static final String EVENT_PUBLISH_FAILED = "publish_failed";
    public static final String EVENT_PUBLISH_PROGRESS = "publish_progress";
    public static final String EVENT_VIDEO_DELETED = "video_deleted";

    public boolean isPublishComplete() {
        return EVENT_PUBLISH_COMPLETE.equals(eventType);
    }

    public boolean isPublishFailed() {
        return EVENT_PUBLISH_FAILED.equals(eventType);
    }
}
