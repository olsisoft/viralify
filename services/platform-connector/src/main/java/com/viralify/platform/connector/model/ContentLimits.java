package com.viralify.platform.connector.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ContentLimits {
    private Platform platform;
    private int maxDurationSeconds;
    private int maxCaptionLength;
    private int maxHashtags;
    private int maxTitleLength;
    private String aspectRatio;
    private long maxFileSizeBytes;
    private boolean hashtagsInCaption;

    @Builder.Default
    private List<String> supportedFormats = List.of("mp4", "mov", "webm");

    @Builder.Default
    private Map<String, Object> platformSpecific = Map.of();

    /**
     * Check if a video duration is valid for this platform
     */
    public boolean isValidDuration(int durationSeconds) {
        return durationSeconds > 0 && durationSeconds <= maxDurationSeconds;
    }

    /**
     * Check if caption length is valid
     */
    public boolean isValidCaptionLength(String caption) {
        return caption == null || caption.length() <= maxCaptionLength;
    }

    /**
     * Check if number of hashtags is valid
     */
    public boolean isValidHashtagCount(int count) {
        return count >= 0 && count <= maxHashtags;
    }

    /**
     * Check if file size is valid
     */
    public boolean isValidFileSize(long sizeBytes) {
        return sizeBytes > 0 && sizeBytes <= maxFileSizeBytes;
    }

    /**
     * Get warning message if content exceeds limits
     */
    public String getWarningMessage(int duration, String caption, int hashtagCount) {
        StringBuilder warnings = new StringBuilder();

        if (duration > maxDurationSeconds) {
            warnings.append(String.format("Video duration (%ds) exceeds %s limit (%ds). Video will be trimmed. ",
                    duration, platform.getDisplayName(), maxDurationSeconds));
        }

        if (caption != null && caption.length() > maxCaptionLength) {
            warnings.append(String.format("Caption length (%d) exceeds %s limit (%d). Caption will be truncated. ",
                    caption.length(), platform.getDisplayName(), maxCaptionLength));
        }

        if (hashtagCount > maxHashtags) {
            warnings.append(String.format("Hashtag count (%d) exceeds %s limit (%d). Some hashtags will be removed. ",
                    hashtagCount, platform.getDisplayName(), maxHashtags));
        }

        return warnings.length() > 0 ? warnings.toString().trim() : null;
    }
}
