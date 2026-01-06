package com.tiktok.platform.connector.dto;

import lombok.*;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class VideoAnalytics {
    private String videoId;
    private Long viewCount;
    private Long likeCount;
    private Long commentCount;
    private Long shareCount;
}
