package com.tiktok.platform.scheduler.controller;

import com.tiktok.platform.scheduler.dto.*;
import com.tiktok.platform.scheduler.service.SchedulerService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/scheduler")
@RequiredArgsConstructor
public class SchedulerController {

    private final SchedulerService schedulerService;

    @PostMapping("/posts")
    public ResponseEntity<ScheduledPostResponse> createScheduledPost(
            @RequestHeader("X-User-Id") UUID userId,
            @RequestBody CreateScheduledPostRequest request
    ) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(schedulerService.createScheduledPost(userId, request));
    }

    @GetMapping("/posts")
    public ResponseEntity<List<ScheduledPostResponse>> getUserPosts(
            @RequestHeader("X-User-Id") UUID userId
    ) {
        return ResponseEntity.ok(schedulerService.getUserScheduledPosts(userId));
    }

    @GetMapping("/posts/pending")
    public ResponseEntity<List<ScheduledPostResponse>> getPendingPosts(
            @RequestHeader("X-User-Id") UUID userId
    ) {
        return ResponseEntity.ok(schedulerService.getUserPendingPosts(userId));
    }

    @GetMapping("/posts/{postId}")
    public ResponseEntity<ScheduledPostResponse> getPost(
            @RequestHeader("X-User-Id") UUID userId,
            @PathVariable UUID postId
    ) {
        return ResponseEntity.ok(schedulerService.getScheduledPost(postId, userId));
    }

    @PutMapping("/posts/{postId}")
    public ResponseEntity<ScheduledPostResponse> updatePost(
            @RequestHeader("X-User-Id") UUID userId,
            @PathVariable UUID postId,
            @RequestBody UpdateScheduledPostRequest request
    ) {
        return ResponseEntity.ok(schedulerService.updateScheduledPost(postId, userId, request));
    }

    @DeleteMapping("/posts/{postId}")
    public ResponseEntity<Void> cancelPost(
            @RequestHeader("X-User-Id") UUID userId,
            @PathVariable UUID postId
    ) {
        schedulerService.cancelScheduledPost(postId, userId);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/stats")
    public ResponseEntity<SchedulerStatsResponse> getStats(
            @RequestHeader("X-User-Id") UUID userId
    ) {
        return ResponseEntity.ok(schedulerService.getStats(userId));
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Scheduler Service is healthy");
    }
}
