package com.tiktok.platform.scheduler.dto;

import lombok.*;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ContentAdaptation {
    private String platform;
    private String title;
    private String caption;
    private String description;
    private List<String> hashtags;
    private List<String> tags;
    private Integer suggestedDurationSeconds;
    private boolean contentWasModified;
    private List<String> adaptationNotes;
}
