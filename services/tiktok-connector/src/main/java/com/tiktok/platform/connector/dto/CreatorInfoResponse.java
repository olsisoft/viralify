package com.tiktok.platform.connector.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CreatorInfoResponse {
    private CreatorInfo data;
    private TikTokError error;
}
