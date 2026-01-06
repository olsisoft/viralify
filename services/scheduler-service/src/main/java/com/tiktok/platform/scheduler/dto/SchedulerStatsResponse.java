package com.tiktok.platform.scheduler.dto;

import lombok.*;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SchedulerStatsResponse {
    private Long totalScheduled;
    private Long pendingCount;
    private Long publishedCount;
    private Long failedCount;
    private Long publishedToday;
    private Long publishedThisWeek;
}
