package com.tiktok.platform.scheduler.dto;

import lombok.*;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PlatformSettings {
    private String customCaption;
    private List<String> customHashtags;
    private String customTitle;
    private Boolean enabled;
}
