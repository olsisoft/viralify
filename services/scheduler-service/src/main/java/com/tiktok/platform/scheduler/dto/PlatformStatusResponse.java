package com.tiktok.platform.scheduler.dto;

import lombok.*;
import java.time.OffsetDateTime;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PlatformStatusResponse {
    private String platform;
    private String status;
    private String platformPostId;
    private String platformShareUrl;
    private String errorMessage;
    private Integer retryCount;
    private String adaptedCaption;
    private List<String> adaptedHashtags;
    private OffsetDateTime publishedAt;
}
