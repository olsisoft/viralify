package com.tiktok.platform.auth.controller;

import com.tiktok.platform.auth.dto.*;
import com.tiktok.platform.auth.service.AuthService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/auth")
@RequiredArgsConstructor
public class AuthController {

    private final AuthService authService;

    // ========================================
    // Email/Password Authentication
    // ========================================

    @PostMapping("/register")
    public ResponseEntity<AuthResponse> register(@RequestBody RegisterRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(authService.register(request));
    }

    @PostMapping("/login")
    public ResponseEntity<AuthResponse> login(@RequestBody LoginRequest request) {
        return ResponseEntity.ok(authService.login(request));
    }

    @PostMapping("/refresh")
    public ResponseEntity<AuthResponse> refreshToken(@RequestBody Map<String, String> request) {
        return ResponseEntity.ok(authService.refreshAccessToken(request.get("refreshToken")));
    }

    @PostMapping("/logout")
    public ResponseEntity<Void> logout(@RequestHeader("Authorization") String authHeader) {
        String token = authHeader.replace("Bearer ", "");
        authService.logout(token);
        return ResponseEntity.noContent().build();
    }

    // ========================================
    // TikTok OAuth
    // ========================================

    @GetMapping("/tiktok/url")
    public ResponseEntity<Map<String, String>> getTikTokAuthUrl(
            @RequestParam(required = false) String state) {
        String url = authService.getTikTokAuthUrl(state);
        return ResponseEntity.ok(Map.of("url", url));
    }

    @GetMapping("/tiktok/callback")
    public ResponseEntity<AuthResponse> handleTikTokCallback(
            @RequestParam String code,
            @RequestParam String state) {
        return ResponseEntity.ok(authService.handleTikTokCallback(code, state));
    }

    @PostMapping("/tiktok/refresh")
    public ResponseEntity<TikTokTokenResponse> refreshTikTokToken(
            @RequestHeader("X-User-Id") UUID userId) {
        return ResponseEntity.ok(authService.refreshTikTokToken(userId));
    }

    // ========================================
    // Instagram OAuth
    // ========================================

    @GetMapping("/instagram/url")
    public ResponseEntity<Map<String, String>> getInstagramAuthUrl(
            @RequestParam(required = false) String state) {
        String url = authService.getInstagramAuthUrl(state);
        return ResponseEntity.ok(Map.of("url", url));
    }

    @GetMapping("/instagram/callback")
    public ResponseEntity<PlatformConnectionResponse> handleInstagramCallback(
            @RequestHeader("X-User-Id") UUID userId,
            @RequestParam String code,
            @RequestParam String state) {
        return ResponseEntity.ok(authService.handleInstagramCallback(userId, code, state));
    }

    // ========================================
    // YouTube OAuth
    // ========================================

    @GetMapping("/youtube/url")
    public ResponseEntity<Map<String, String>> getYouTubeAuthUrl(
            @RequestParam(required = false) String state) {
        String url = authService.getYouTubeAuthUrl(state);
        return ResponseEntity.ok(Map.of("url", url));
    }

    @GetMapping("/youtube/callback")
    public ResponseEntity<PlatformConnectionResponse> handleYouTubeCallback(
            @RequestHeader("X-User-Id") UUID userId,
            @RequestParam String code,
            @RequestParam String state) {
        return ResponseEntity.ok(authService.handleYouTubeCallback(userId, code, state));
    }

    // ========================================
    // Platform Management
    // ========================================

    @GetMapping("/platforms")
    public ResponseEntity<List<PlatformAccountDto>> getUserPlatforms(
            @RequestHeader("X-User-Id") UUID userId) {
        return ResponseEntity.ok(authService.getUserPlatformAccounts(userId));
    }

    @DeleteMapping("/platforms/{platform}")
    public ResponseEntity<Void> disconnectPlatform(
            @RequestHeader("X-User-Id") UUID userId,
            @PathVariable String platform) {
        authService.disconnectPlatform(userId, platform);
        return ResponseEntity.noContent().build();
    }

    // ========================================
    // User Profile
    // ========================================

    @GetMapping("/profile")
    public ResponseEntity<UserDto> getProfile(@RequestHeader("X-User-Id") UUID userId) {
        return ResponseEntity.ok(authService.getUserProfile(userId));
    }

    @PutMapping("/profile")
    public ResponseEntity<UserDto> updateProfile(
            @RequestHeader("X-User-Id") UUID userId,
            @RequestBody UpdateProfileRequest request) {
        return ResponseEntity.ok(authService.updateUserProfile(userId, request));
    }

    // ========================================
    // Token Management (Internal use by other services)
    // ========================================

    @GetMapping("/token/{platform}")
    public ResponseEntity<Map<String, String>> getPlatformToken(
            @RequestHeader("X-User-Id") UUID userId,
            @PathVariable String platform) {
        String token = authService.getDecryptedPlatformToken(userId, platform.toUpperCase());
        return ResponseEntity.ok(Map.of("token", token));
    }

    // ========================================
    // Health Check
    // ========================================

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Auth Service is healthy");
    }
}
