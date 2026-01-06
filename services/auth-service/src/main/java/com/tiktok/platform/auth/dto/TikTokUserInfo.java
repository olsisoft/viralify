package com.tiktok.platform.auth.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TikTokUserInfo {
    @JsonProperty("open_id")
    private String openId;

    @JsonProperty("union_id")
    private String unionId;

    @JsonProperty("avatar_url")
    private String avatarUrl;

    @JsonProperty("display_name")
    private String displayName;

    private String username;

    @JsonProperty("follower_count")
    private Long followerCount;

    @JsonProperty("following_count")
    private Long followingCount;

    @JsonProperty("likes_count")
    private Long likesCount;
}
