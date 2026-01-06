package com.tiktok.platform.auth.dto;

import lombok.*;
import java.time.OffsetDateTime;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PlatformAccountDto {
    private UUID id;
    private String platform;
    private String platformUserId;
    private String platformUsername;
    private String platformDisplayName;
    private String platformAvatarUrl;
    private Long followerCount;
    private String accountStatus;
    private OffsetDateTime tokenExpiresAt;
    private OffsetDateTime connectedAt;
}
