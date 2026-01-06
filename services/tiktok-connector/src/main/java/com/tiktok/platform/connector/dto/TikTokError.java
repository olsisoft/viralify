package com.tiktok.platform.connector.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class TikTokError {
    private String code;
    private String message;
    @JsonProperty("log_id")
    private String logId;
}
