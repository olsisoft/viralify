package com.tiktok.platform.scheduler.dto;

import lombok.*;
import java.time.OffsetDateTime;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UpdateScheduledPostRequest {
    private String title;
    private String caption;
    private List<String> hashtags;
    private OffsetDateTime scheduledAt;
    private String privacyLevel;
    private Boolean allowComments;
    private Boolean allowDuet;
    private Boolean allowStitch;
}
