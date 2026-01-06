package com.tiktok.platform.connector.dto;

import lombok.*;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class VideoQueryData {
    private List<VideoInfo> videos;
}
