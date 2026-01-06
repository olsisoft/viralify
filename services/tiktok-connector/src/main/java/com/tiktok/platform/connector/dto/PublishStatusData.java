package com.tiktok.platform.connector.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PublishStatusData {
    private String status;
    @JsonProperty("publish_id")
    private String publishId;
    @JsonProperty("uploaded_bytes")
    private Long uploadedBytes;
    @JsonProperty("fail_reason")
    private String failReason;
    @JsonProperty("publicaly_available_post_id")
    private List<String> publicPostIds;
}
