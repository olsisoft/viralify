package com.tiktok.platform.connector.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TikTokUserInfoResponse {
    private TikTokUserData data;
    private TikTokError error;
}
