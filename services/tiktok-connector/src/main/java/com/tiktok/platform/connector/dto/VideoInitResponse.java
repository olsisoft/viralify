package com.tiktok.platform.connector.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class VideoInitResponse {
    private VideoInitData data;
    private TikTokError error;
}
