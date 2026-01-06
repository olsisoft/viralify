package com.viralify.platform.connector.dto;

import com.viralify.platform.connector.model.Platform;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PublishStatusResponse {
    private Platform platform;
    private String publishId;
    private PublishStatus status;
    private String platformPostId;
    private String shareUrl;
    private String failReason;
    private Long uploadedBytes;
    private Integer progressPercent;

    public enum PublishStatus {
        PENDING,
        PROCESSING,
        UPLOADING,
        PUBLISHED,
        FAILED,
        CANCELLED
    }

    public boolean isComplete() {
        return status == PublishStatus.PUBLISHED || status == PublishStatus.FAILED || status == PublishStatus.CANCELLED;
    }

    public boolean isSuccessful() {
        return status == PublishStatus.PUBLISHED;
    }
}
