package com.tiktok.platform.connector.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class VideoInfo {
    private String id;
    private String title;
    @JsonProperty("video_description")
    private String videoDescription;
    private Integer duration;
    @JsonProperty("cover_image_url")
    private String coverImageUrl;
    @JsonProperty("share_url")
    private String shareUrl;
    @JsonProperty("view_count")
    private Long viewCount;
    @JsonProperty("like_count")
    private Long likeCount;
    @JsonProperty("comment_count")
    private Long commentCount;
    @JsonProperty("share_count")
    private Long shareCount;
    @JsonProperty("create_time")
    private Long createTime;
}
