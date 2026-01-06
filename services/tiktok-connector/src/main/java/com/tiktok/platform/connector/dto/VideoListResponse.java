package com.tiktok.platform.connector.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class VideoListResponse {
    private VideoListData data;
    private TikTokError error;
}
