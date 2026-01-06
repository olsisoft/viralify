package com.tiktok.platform.connector.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class VideoQueryResponse {
    private VideoQueryData data;
    private TikTokError error;
}
