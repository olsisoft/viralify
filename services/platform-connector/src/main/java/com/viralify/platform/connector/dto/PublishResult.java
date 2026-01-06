package com.viralify.platform.connector.dto;

import com.viralify.platform.connector.model.Platform;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.OffsetDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PublishResult {
    private Platform platform;
    private boolean success;
    private String publishId;
    private String platformPostId;
    private String shareUrl;
    private String errorCode;
    private String errorMessage;
    private OffsetDateTime publishedAt;

    public static PublishResult success(Platform platform, String publishId) {
        return PublishResult.builder()
                .platform(platform)
                .success(true)
                .publishId(publishId)
                .build();
    }

    public static PublishResult success(Platform platform, String publishId, String platformPostId, String shareUrl) {
        return PublishResult.builder()
                .platform(platform)
                .success(true)
                .publishId(publishId)
                .platformPostId(platformPostId)
                .shareUrl(shareUrl)
                .publishedAt(OffsetDateTime.now())
                .build();
    }

    public static PublishResult failure(Platform platform, String errorCode, String errorMessage) {
        return PublishResult.builder()
                .platform(platform)
                .success(false)
                .errorCode(errorCode)
                .errorMessage(errorMessage)
                .build();
    }
}
