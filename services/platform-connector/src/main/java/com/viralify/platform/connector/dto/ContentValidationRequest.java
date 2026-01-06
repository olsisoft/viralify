package com.viralify.platform.connector.dto;

import com.viralify.platform.connector.model.Platform;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ContentValidationRequest {
    private Platform platform;
    private String title;
    private String caption;
    private List<String> hashtags;
    private Integer videoDurationSeconds;
    private Long videoSizeBytes;
    private String videoFormat;
}
