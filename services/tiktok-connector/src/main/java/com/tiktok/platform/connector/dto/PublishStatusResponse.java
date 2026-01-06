package com.tiktok.platform.connector.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PublishStatusResponse {
    private PublishStatusData data;
    private TikTokError error;
}
