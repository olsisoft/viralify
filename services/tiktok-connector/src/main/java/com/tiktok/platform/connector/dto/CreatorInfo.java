package com.tiktok.platform.connector.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CreatorInfo {
    @JsonProperty("creator_avatar_url")
    private String creatorAvatarUrl;
    @JsonProperty("creator_username")
    private String creatorUsername;
    @JsonProperty("creator_nickname")
    private String creatorNickname;
    @JsonProperty("privacy_level_options")
    private List<String> privacyLevelOptions;
    @JsonProperty("comment_disabled")
    private Boolean commentDisabled;
    @JsonProperty("duet_disabled")
    private Boolean duetDisabled;
    @JsonProperty("stitch_disabled")
    private Boolean stitchDisabled;
    @JsonProperty("max_video_post_duration_sec")
    private Integer maxVideoPostDurationSec;
}
