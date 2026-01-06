package com.tiktok.platform.scheduler.dto;

import lombok.*;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PublishRequest {
    private UUID userId;
    private String title;
    private String caption;
    private String videoUrl;
    private String privacyLevel;
    private Boolean allowComments;
    private Boolean allowDuet;
    private Boolean allowStitch;
    private Boolean commercialContent;
    private Boolean brandedContent;
}
