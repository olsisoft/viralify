package com.tiktok.platform.auth.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PlatformConnectionResponse {
    private String platform;
    private boolean connected;
    private String platformUserId;
    private String platformUsername;
    private String platformDisplayName;
    private String platformAvatarUrl;
    private String errorMessage;
}
