package com.tiktok.platform.connector.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class VideoListData {
    private List<VideoInfo> videos;
    private String cursor;
    @JsonProperty("has_more")
    private Boolean hasMore;
}
