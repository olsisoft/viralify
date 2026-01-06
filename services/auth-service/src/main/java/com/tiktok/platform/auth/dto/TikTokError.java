package com.tiktok.platform.auth.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TikTokError {
    private String code;
    private String message;
    private String logId;
}
