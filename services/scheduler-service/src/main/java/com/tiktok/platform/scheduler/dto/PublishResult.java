package com.tiktok.platform.scheduler.dto;

import lombok.*;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PublishResult {
    private boolean success;
    private String publishId;
    private String tiktokPostId;
    private String shareUrl;
    private String errorCode;
    private String errorMessage;
}
