package com.tiktok.platform.connector.controller;

import com.tiktok.platform.connector.dto.*;
import com.tiktok.platform.connector.service.TikTokConnectorService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.UUID;

@RestController
@RequestMapping("/api/tiktok")
@RequiredArgsConstructor
public class TikTokConnectorController {

    private final TikTokConnectorService tiktokConnectorService;

    @PostMapping("/publish")
    public ResponseEntity<PublishResult> publishVideo(@RequestBody PublishVideoRequest request) {
        PublishResult result = tiktokConnectorService.publishVideo(request);
        return ResponseEntity.ok(result);
    }

    @GetMapping("/publish/status/{userId}/{publishId}")
    public ResponseEntity<PublishStatusResponse> getPublishStatus(
            @PathVariable UUID userId,
            @PathVariable String publishId) {
        PublishStatusResponse response = tiktokConnectorService.getPublishStatus(userId, publishId);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/user/{userId}")
    public ResponseEntity<TikTokUserInfo> getUserInfo(@PathVariable UUID userId) {
        TikTokUserInfo userInfo = tiktokConnectorService.getUserInfo(userId);
        return ResponseEntity.ok(userInfo);
    }

    @GetMapping("/videos/{userId}")
    public ResponseEntity<VideoListResponse> getUserVideos(
            @PathVariable UUID userId,
            @RequestParam(defaultValue = "20") int maxCount,
            @RequestParam(required = false) String cursor) {
        VideoListResponse response = tiktokConnectorService.getUserVideos(userId, maxCount, cursor);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/video/{userId}/{videoId}/analytics")
    public ResponseEntity<VideoAnalytics> getVideoAnalytics(
            @PathVariable UUID userId,
            @PathVariable String videoId) {
        VideoAnalytics analytics = tiktokConnectorService.getVideoAnalytics(userId, videoId);
        return ResponseEntity.ok(analytics);
    }

    @PostMapping("/webhook")
    public ResponseEntity<Void> handleWebhook(@RequestBody TikTokWebhookEvent event) {
        tiktokConnectorService.handleWebhook(event);
        return ResponseEntity.ok().build();
    }
}
