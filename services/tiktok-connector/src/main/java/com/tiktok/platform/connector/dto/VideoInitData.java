package com.tiktok.platform.connector.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class VideoInitData {
    @JsonProperty("publish_id")
    private String publishId;
    @JsonProperty("upload_url")
    private String uploadUrl;
}
