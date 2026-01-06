package com.tiktok.platform.connector.dto;

import lombok.*;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PublishVideoRequest {
    private UUID userId;
    private String title;
    private String caption;
    private String videoUrl;
    private String privacyLevel;
    private boolean allowComments;
    private boolean allowDuet;
    private boolean allowStitch;
    private boolean commercialContent;
    private boolean brandedContent;
}
