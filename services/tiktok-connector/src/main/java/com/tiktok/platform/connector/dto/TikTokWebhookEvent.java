package com.tiktok.platform.connector.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TikTokWebhookEvent {
    private String event;
    @JsonProperty("publish_id")
    private String publishId;
    @JsonProperty("post_id")
    private String postId;
    @JsonProperty("fail_reason")
    private String failReason;
    private Long timestamp;
}
