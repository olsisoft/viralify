package com.tiktok.platform.auth.dto;

import lombok.*;
import java.time.OffsetDateTime;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserDto {
    private UUID id;
    private String email;
    private String fullName;
    private String avatarUrl;
    private String tiktokUserId;
    private String tiktokUsername;
    private String tiktokDisplayName;
    private String tiktokAvatarUrl;
    private Long tiktokFollowerCount;
    private Long tiktokFollowingCount;
    private Long tiktokLikesCount;
    private Boolean tiktokConnected;
    private String planType;
    private Integer monthlyPostsLimit;
    private Integer monthlyPostsUsed;
    private Integer monthlyAiGenerationsLimit;
    private Integer monthlyAiGenerationsUsed;
    private String timezone;
    private String language;
    private OffsetDateTime createdAt;
}
