package com.viralify.platform.connector.dto;

import com.viralify.platform.connector.model.Platform;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PlatformUserInfo {
    private Platform platform;
    private String platformUserId;
    private String username;
    private String displayName;
    private String avatarUrl;
    private Long followerCount;
    private Long followingCount;
    private Long likesCount;
    private Long videoCount;

    // Platform-specific data
    @Builder.Default
    private Map<String, Object> platformSpecific = Map.of();
}
