package com.tiktok.platform.auth.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TikTokUserInfoResponse {
    private TikTokUserData data;
    private TikTokError error;
}
