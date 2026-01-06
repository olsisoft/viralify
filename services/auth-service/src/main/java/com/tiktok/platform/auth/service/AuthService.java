package com.tiktok.platform.auth.service;

import com.tiktok.platform.auth.dto.*;
import com.tiktok.platform.auth.entity.User;
import com.tiktok.platform.auth.repository.UserRepository;
import com.tiktok.platform.auth.security.JwtTokenProvider;
import com.tiktok.platform.auth.security.EncryptionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.*;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;
import java.time.Duration;
import java.time.OffsetDateTime;
import java.util.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuthService {
    
    private final UserRepository userRepository;
    private final JwtTokenProvider jwtTokenProvider;
    private final PasswordEncoder passwordEncoder;
    private final EncryptionService encryptionService;
    private final RedisTemplate<String, Object> redisTemplate;
    private final RestTemplate restTemplate;
    
    @Value("${tiktok.client-key}")
    private String tiktokClientKey;

    @Value("${tiktok.client-secret}")
    private String tiktokClientSecret;

    @Value("${tiktok.redirect-uri}")
    private String tiktokRedirectUri;

    // Instagram/Facebook OAuth Config
    @Value("${instagram.app-id:}")
    private String instagramAppId;

    @Value("${instagram.app-secret:}")
    private String instagramAppSecret;

    @Value("${instagram.redirect-uri:}")
    private String instagramRedirectUri;

    // YouTube/Google OAuth Config
    @Value("${youtube.client-id:}")
    private String youtubeClientId;

    @Value("${youtube.client-secret:}")
    private String youtubeClientSecret;

    @Value("${youtube.redirect-uri:}")
    private String youtubeRedirectUri;

    // TikTok API URLs
    private static final String TIKTOK_AUTH_URL = "https://www.tiktok.com/v2/auth/authorize/";
    private static final String TIKTOK_TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/";
    private static final String TIKTOK_USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/";

    // Instagram/Facebook API URLs
    private static final String INSTAGRAM_AUTH_URL = "https://api.instagram.com/oauth/authorize";
    private static final String INSTAGRAM_TOKEN_URL = "https://api.instagram.com/oauth/access_token";
    private static final String INSTAGRAM_GRAPH_URL = "https://graph.instagram.com";
    private static final String FACEBOOK_GRAPH_URL = "https://graph.facebook.com/v18.0";

    // YouTube/Google API URLs
    private static final String GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth";
    private static final String GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token";
    private static final String YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3";
    
    // ========================================
    // Email/Password Authentication
    // ========================================
    
    @Transactional
    public AuthResponse register(RegisterRequest request) {
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new RuntimeException("Email already registered");
        }
        
        User user = User.builder()
                .email(request.getEmail())
                .passwordHash(passwordEncoder.encode(request.getPassword()))
                .fullName(request.getFullName())
                .build();
        
        user = userRepository.save(user);
        
        String accessToken = jwtTokenProvider.generateAccessToken(user);
        String refreshToken = jwtTokenProvider.generateRefreshToken(user);
        
        cacheUserSession(user.getId(), accessToken);
        
        return AuthResponse.builder()
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .expiresIn(jwtTokenProvider.getAccessTokenExpiration())
                .user(mapToUserDto(user))
                .build();
    }
    
    public AuthResponse login(LoginRequest request) {
        User user = userRepository.findByEmail(request.getEmail())
                .orElseThrow(() -> new RuntimeException("Invalid credentials"));
        
        if (!passwordEncoder.matches(request.getPassword(), user.getPasswordHash())) {
            throw new RuntimeException("Invalid credentials");
        }
        
        if (!user.getIsActive()) {
            throw new RuntimeException("Account is deactivated");
        }
        
        user.setLastLoginAt(OffsetDateTime.now());
        userRepository.save(user);
        
        String accessToken = jwtTokenProvider.generateAccessToken(user);
        String refreshToken = jwtTokenProvider.generateRefreshToken(user);
        
        cacheUserSession(user.getId(), accessToken);
        
        return AuthResponse.builder()
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .expiresIn(jwtTokenProvider.getAccessTokenExpiration())
                .user(mapToUserDto(user))
                .build();
    }
    
    // ========================================
    // TikTok OAuth 2.0
    // ========================================
    
    public String getTikTokAuthUrl(String state) {
        String csrfState = state != null ? state : UUID.randomUUID().toString();
        
        // Cache state for validation
        redisTemplate.opsForValue().set(
                "tiktok:oauth:state:" + csrfState,
                csrfState,
                Duration.ofMinutes(10)
        );
        
        return UriComponentsBuilder.fromHttpUrl(TIKTOK_AUTH_URL)
                .queryParam("client_key", tiktokClientKey)
                .queryParam("response_type", "code")
                .queryParam("scope", "user.info.basic,user.info.profile,user.info.stats,video.upload,video.publish")
                .queryParam("redirect_uri", tiktokRedirectUri)
                .queryParam("state", csrfState)
                .build()
                .toUriString();
    }
    
    @Transactional
    public AuthResponse handleTikTokCallback(String code, String state) {
        // Validate state
        String cachedState = (String) redisTemplate.opsForValue().get("tiktok:oauth:state:" + state);
        if (cachedState == null || !cachedState.equals(state)) {
            throw new RuntimeException("Invalid OAuth state");
        }
        redisTemplate.delete("tiktok:oauth:state:" + state);
        
        // Exchange code for tokens
        TikTokTokenResponse tokenResponse = exchangeCodeForToken(code);
        
        // Get user info from TikTok
        TikTokUserInfo userInfo = getTikTokUserInfo(tokenResponse.getAccessToken());
        
        // Find or create user
        User user = userRepository.findByTiktokUserId(userInfo.getOpenId())
                .orElseGet(() -> createUserFromTikTok(userInfo, tokenResponse));
        
        // Update tokens
        updateUserTikTokTokens(user, tokenResponse);
        updateUserTikTokProfile(user, userInfo);
        
        user.setLastLoginAt(OffsetDateTime.now());
        user = userRepository.save(user);
        
        // Generate JWT
        String accessToken = jwtTokenProvider.generateAccessToken(user);
        String refreshToken = jwtTokenProvider.generateRefreshToken(user);
        
        cacheUserSession(user.getId(), accessToken);
        
        return AuthResponse.builder()
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .expiresIn(jwtTokenProvider.getAccessTokenExpiration())
                .user(mapToUserDto(user))
                .tiktokConnected(true)
                .build();
    }
    
    private TikTokTokenResponse exchangeCodeForToken(String code) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
        
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("client_key", tiktokClientKey);
        body.add("client_secret", tiktokClientSecret);
        body.add("code", code);
        body.add("grant_type", "authorization_code");
        body.add("redirect_uri", tiktokRedirectUri);
        
        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(body, headers);
        
        ResponseEntity<TikTokTokenResponse> response = restTemplate.exchange(
                TIKTOK_TOKEN_URL,
                HttpMethod.POST,
                request,
                TikTokTokenResponse.class
        );
        
        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Failed to exchange TikTok authorization code");
        }
        
        return response.getBody();
    }
    
    private TikTokUserInfo getTikTokUserInfo(String accessToken) {
        HttpHeaders headers = new HttpHeaders();
        headers.setBearerAuth(accessToken);
        
        String url = UriComponentsBuilder.fromHttpUrl(TIKTOK_USER_INFO_URL)
                .queryParam("fields", "open_id,union_id,avatar_url,display_name,username,follower_count,following_count,likes_count")
                .build()
                .toUriString();
        
        HttpEntity<Void> request = new HttpEntity<>(headers);
        
        ResponseEntity<TikTokUserInfoResponse> response = restTemplate.exchange(
                url,
                HttpMethod.GET,
                request,
                TikTokUserInfoResponse.class
        );
        
        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Failed to get TikTok user info");
        }
        
        return response.getBody().getData().getUser();
    }
    
    private User createUserFromTikTok(TikTokUserInfo userInfo, TikTokTokenResponse tokenResponse) {
        return User.builder()
                .email(userInfo.getOpenId() + "@tiktok.placeholder.com") // TikTok doesn't provide email
                .tiktokUserId(userInfo.getOpenId())
                .tiktokUsername(userInfo.getUsername())
                .tiktokDisplayName(userInfo.getDisplayName())
                .tiktokAvatarUrl(userInfo.getAvatarUrl())
                .tiktokFollowerCount(userInfo.getFollowerCount())
                .tiktokFollowingCount(userInfo.getFollowingCount())
                .tiktokLikesCount(userInfo.getLikesCount())
                .accessTokenEncrypted(encryptionService.encrypt(tokenResponse.getAccessToken()))
                .refreshTokenEncrypted(encryptionService.encrypt(tokenResponse.getRefreshToken()))
                .tokenExpiresAt(OffsetDateTime.now().plusSeconds(tokenResponse.getExpiresIn()))
                .tokenScope(tokenResponse.getScope())
                .build();
    }
    
    private void updateUserTikTokTokens(User user, TikTokTokenResponse tokenResponse) {
        user.setAccessTokenEncrypted(encryptionService.encrypt(tokenResponse.getAccessToken()));
        user.setRefreshTokenEncrypted(encryptionService.encrypt(tokenResponse.getRefreshToken()));
        user.setTokenExpiresAt(OffsetDateTime.now().plusSeconds(tokenResponse.getExpiresIn()));
        user.setTokenScope(tokenResponse.getScope());
    }
    
    private void updateUserTikTokProfile(User user, TikTokUserInfo userInfo) {
        user.setTiktokUsername(userInfo.getUsername());
        user.setTiktokDisplayName(userInfo.getDisplayName());
        user.setTiktokAvatarUrl(userInfo.getAvatarUrl());
        user.setTiktokFollowerCount(userInfo.getFollowerCount());
        user.setTiktokFollowingCount(userInfo.getFollowingCount());
        user.setTiktokLikesCount(userInfo.getLikesCount());
    }

    // ========================================
    // Instagram OAuth 2.0
    // ========================================

    public String getInstagramAuthUrl(String state) {
        String csrfState = state != null ? state : UUID.randomUUID().toString();

        redisTemplate.opsForValue().set(
                "instagram:oauth:state:" + csrfState,
                csrfState,
                Duration.ofMinutes(10)
        );

        // Instagram Basic Display API / Instagram Graph API
        return UriComponentsBuilder.fromHttpUrl(INSTAGRAM_AUTH_URL)
                .queryParam("client_id", instagramAppId)
                .queryParam("redirect_uri", instagramRedirectUri)
                .queryParam("scope", "instagram_basic,instagram_content_publish,instagram_manage_insights")
                .queryParam("response_type", "code")
                .queryParam("state", csrfState)
                .build()
                .toUriString();
    }

    @Transactional
    public PlatformConnectionResponse handleInstagramCallback(UUID userId, String code, String state) {
        // Validate state
        String cachedState = (String) redisTemplate.opsForValue().get("instagram:oauth:state:" + state);
        if (cachedState == null || !cachedState.equals(state)) {
            throw new RuntimeException("Invalid OAuth state");
        }
        redisTemplate.delete("instagram:oauth:state:" + state);

        // Exchange code for short-lived token
        Map<String, String> tokenResponse = exchangeInstagramCode(code);
        String shortLivedToken = tokenResponse.get("access_token");
        String igUserId = tokenResponse.get("user_id");

        // Exchange for long-lived token
        Map<String, Object> longLivedToken = exchangeForLongLivedInstagramToken(shortLivedToken);
        String accessToken = (String) longLivedToken.get("access_token");
        Long expiresIn = ((Number) longLivedToken.get("expires_in")).longValue();

        // Get user info
        Map<String, Object> userInfo = getInstagramUserInfo(accessToken, igUserId);

        // Save platform account
        savePlatformAccount(userId, "INSTAGRAM", igUserId, userInfo, accessToken, null, expiresIn);

        // Cache token for platform-connector service
        redisTemplate.opsForValue().set(
                "instagram:token:" + userId,
                accessToken,
                Duration.ofSeconds(expiresIn - 3600) // Refresh 1 hour before expiry
        );
        redisTemplate.opsForValue().set(
                "instagram:user_id:" + userId,
                igUserId,
                Duration.ofDays(30)
        );

        return PlatformConnectionResponse.builder()
                .platform("INSTAGRAM")
                .connected(true)
                .platformUserId(igUserId)
                .platformUsername((String) userInfo.get("username"))
                .build();
    }

    private Map<String, String> exchangeInstagramCode(String code) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("client_id", instagramAppId);
        body.add("client_secret", instagramAppSecret);
        body.add("grant_type", "authorization_code");
        body.add("redirect_uri", instagramRedirectUri);
        body.add("code", code);

        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(body, headers);

        ResponseEntity<Map> response = restTemplate.exchange(
                INSTAGRAM_TOKEN_URL,
                HttpMethod.POST,
                request,
                Map.class
        );

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Failed to exchange Instagram code");
        }

        return response.getBody();
    }

    private Map<String, Object> exchangeForLongLivedInstagramToken(String shortLivedToken) {
        String url = UriComponentsBuilder.fromHttpUrl(INSTAGRAM_GRAPH_URL + "/access_token")
                .queryParam("grant_type", "ig_exchange_token")
                .queryParam("client_secret", instagramAppSecret)
                .queryParam("access_token", shortLivedToken)
                .build()
                .toUriString();

        ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Failed to exchange for long-lived Instagram token");
        }

        return response.getBody();
    }

    private Map<String, Object> getInstagramUserInfo(String accessToken, String userId) {
        String url = UriComponentsBuilder.fromHttpUrl(INSTAGRAM_GRAPH_URL + "/" + userId)
                .queryParam("fields", "id,username,account_type,media_count")
                .queryParam("access_token", accessToken)
                .build()
                .toUriString();

        ResponseEntity<Map> response = restTemplate.getForEntity(url, Map.class);

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Failed to get Instagram user info");
        }

        return response.getBody();
    }

    // ========================================
    // YouTube OAuth 2.0
    // ========================================

    public String getYouTubeAuthUrl(String state) {
        String csrfState = state != null ? state : UUID.randomUUID().toString();

        redisTemplate.opsForValue().set(
                "youtube:oauth:state:" + csrfState,
                csrfState,
                Duration.ofMinutes(10)
        );

        return UriComponentsBuilder.fromHttpUrl(GOOGLE_AUTH_URL)
                .queryParam("client_id", youtubeClientId)
                .queryParam("redirect_uri", youtubeRedirectUri)
                .queryParam("response_type", "code")
                .queryParam("scope", "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly")
                .queryParam("access_type", "offline")
                .queryParam("prompt", "consent")
                .queryParam("state", csrfState)
                .build()
                .toUriString();
    }

    @Transactional
    public PlatformConnectionResponse handleYouTubeCallback(UUID userId, String code, String state) {
        // Validate state
        String cachedState = (String) redisTemplate.opsForValue().get("youtube:oauth:state:" + state);
        if (cachedState == null || !cachedState.equals(state)) {
            throw new RuntimeException("Invalid OAuth state");
        }
        redisTemplate.delete("youtube:oauth:state:" + state);

        // Exchange code for tokens
        Map<String, Object> tokenResponse = exchangeYouTubeCode(code);
        String accessToken = (String) tokenResponse.get("access_token");
        String refreshToken = (String) tokenResponse.get("refresh_token");
        Long expiresIn = ((Number) tokenResponse.get("expires_in")).longValue();

        // Get channel info
        Map<String, Object> channelInfo = getYouTubeChannelInfo(accessToken);

        String channelId = (String) channelInfo.get("id");
        String channelTitle = (String) channelInfo.get("title");
        String customUrl = (String) channelInfo.get("customUrl");

        // Save platform account
        savePlatformAccount(userId, "YOUTUBE", channelId, channelInfo, accessToken, refreshToken, expiresIn);

        // Cache token for platform-connector service
        redisTemplate.opsForValue().set(
                "youtube:token:" + userId,
                accessToken,
                Duration.ofSeconds(expiresIn - 300)
        );

        return PlatformConnectionResponse.builder()
                .platform("YOUTUBE")
                .connected(true)
                .platformUserId(channelId)
                .platformUsername(customUrl != null ? customUrl : channelTitle)
                .build();
    }

    private Map<String, Object> exchangeYouTubeCode(String code) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("client_id", youtubeClientId);
        body.add("client_secret", youtubeClientSecret);
        body.add("code", code);
        body.add("grant_type", "authorization_code");
        body.add("redirect_uri", youtubeRedirectUri);

        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(body, headers);

        ResponseEntity<Map> response = restTemplate.exchange(
                GOOGLE_TOKEN_URL,
                HttpMethod.POST,
                request,
                Map.class
        );

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Failed to exchange YouTube code");
        }

        return response.getBody();
    }

    private Map<String, Object> getYouTubeChannelInfo(String accessToken) {
        HttpHeaders headers = new HttpHeaders();
        headers.setBearerAuth(accessToken);

        String url = UriComponentsBuilder.fromHttpUrl(YOUTUBE_API_URL + "/channels")
                .queryParam("part", "snippet,statistics")
                .queryParam("mine", "true")
                .build()
                .toUriString();

        HttpEntity<Void> request = new HttpEntity<>(headers);

        ResponseEntity<Map> response = restTemplate.exchange(
                url,
                HttpMethod.GET,
                request,
                Map.class
        );

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Failed to get YouTube channel info");
        }

        List<Map<String, Object>> items = (List<Map<String, Object>>) response.getBody().get("items");
        if (items == null || items.isEmpty()) {
            throw new RuntimeException("No YouTube channel found");
        }

        Map<String, Object> channel = items.get(0);
        Map<String, Object> snippet = (Map<String, Object>) channel.get("snippet");
        Map<String, Object> statistics = (Map<String, Object>) channel.get("statistics");

        Map<String, Object> result = new HashMap<>();
        result.put("id", channel.get("id"));
        result.put("title", snippet.get("title"));
        result.put("customUrl", snippet.get("customUrl"));
        result.put("description", snippet.get("description"));
        result.put("subscriberCount", statistics.get("subscriberCount"));
        result.put("videoCount", statistics.get("videoCount"));

        return result;
    }

    // ========================================
    // Platform Account Management
    // ========================================

    private void savePlatformAccount(UUID userId, String platform, String platformUserId,
                                     Map<String, Object> userInfo, String accessToken,
                                     String refreshToken, Long expiresIn) {
        // This would typically save to platform_accounts table
        // For now, we cache the tokens
        String cacheKey = platform.toLowerCase() + ":account:" + userId;

        Map<String, Object> accountData = new HashMap<>();
        accountData.put("platformUserId", platformUserId);
        accountData.put("accessToken", encryptionService.encrypt(accessToken));
        if (refreshToken != null) {
            accountData.put("refreshToken", encryptionService.encrypt(refreshToken));
        }
        accountData.put("expiresAt", OffsetDateTime.now().plusSeconds(expiresIn).toString());
        accountData.put("userInfo", userInfo);

        redisTemplate.opsForHash().putAll(cacheKey, accountData);
        redisTemplate.expire(cacheKey, Duration.ofDays(90));

        log.info("Saved {} account for user {}", platform, userId);
    }

    public List<PlatformAccountDto> getUserPlatformAccounts(UUID userId) {
        List<PlatformAccountDto> accounts = new ArrayList<>();

        // Check TikTok (from user table)
        userRepository.findById(userId).ifPresent(user -> {
            if (user.getTiktokUserId() != null) {
                accounts.add(PlatformAccountDto.builder()
                        .platform("TIKTOK")
                        .platformUserId(user.getTiktokUserId())
                        .platformUsername(user.getTiktokUsername())
                        .platformDisplayName(user.getTiktokDisplayName())
                        .platformAvatarUrl(user.getTiktokAvatarUrl())
                        .followerCount(user.getTiktokFollowerCount())
                        .accountStatus("active")
                        .build());
            }
        });

        // Check Instagram
        Map<Object, Object> igAccount = redisTemplate.opsForHash().entries("instagram:account:" + userId);
        if (!igAccount.isEmpty()) {
            Map<String, Object> userInfo = (Map<String, Object>) igAccount.get("userInfo");
            accounts.add(PlatformAccountDto.builder()
                    .platform("INSTAGRAM")
                    .platformUserId((String) igAccount.get("platformUserId"))
                    .platformUsername(userInfo != null ? (String) userInfo.get("username") : null)
                    .accountStatus("active")
                    .build());
        }

        // Check YouTube
        Map<Object, Object> ytAccount = redisTemplate.opsForHash().entries("youtube:account:" + userId);
        if (!ytAccount.isEmpty()) {
            Map<String, Object> userInfo = (Map<String, Object>) ytAccount.get("userInfo");
            accounts.add(PlatformAccountDto.builder()
                    .platform("YOUTUBE")
                    .platformUserId((String) ytAccount.get("platformUserId"))
                    .platformUsername(userInfo != null ? (String) userInfo.get("customUrl") : null)
                    .platformDisplayName(userInfo != null ? (String) userInfo.get("title") : null)
                    .accountStatus("active")
                    .build());
        }

        return accounts;
    }

    public void disconnectPlatform(UUID userId, String platform) {
        switch (platform.toUpperCase()) {
            case "TIKTOK" -> {
                userRepository.findById(userId).ifPresent(user -> {
                    user.setTiktokUserId(null);
                    user.setTiktokUsername(null);
                    user.setAccessTokenEncrypted(null);
                    user.setRefreshTokenEncrypted(null);
                    userRepository.save(user);
                });
                redisTemplate.delete("tiktok:token:" + userId);
            }
            case "INSTAGRAM" -> {
                redisTemplate.delete("instagram:account:" + userId);
                redisTemplate.delete("instagram:token:" + userId);
                redisTemplate.delete("instagram:user_id:" + userId);
            }
            case "YOUTUBE" -> {
                redisTemplate.delete("youtube:account:" + userId);
                redisTemplate.delete("youtube:token:" + userId);
            }
            default -> throw new RuntimeException("Unknown platform: " + platform);
        }
        log.info("Disconnected {} for user {}", platform, userId);
    }

    public String getDecryptedPlatformToken(UUID userId, String platform) {
        String cacheKey = platform.toLowerCase() + ":token:" + userId;
        String cachedToken = (String) redisTemplate.opsForValue().get(cacheKey);

        if (cachedToken != null) {
            return cachedToken;
        }

        // Try to get from account data
        Map<Object, Object> accountData = redisTemplate.opsForHash()
                .entries(platform.toLowerCase() + ":account:" + userId);

        if (!accountData.isEmpty() && accountData.get("accessToken") != null) {
            return encryptionService.decrypt((String) accountData.get("accessToken"));
        }

        throw new RuntimeException("No " + platform + " token found for user: " + userId);
    }

    // ========================================
    // Token Refresh
    // ========================================
    
    public AuthResponse refreshAccessToken(String refreshToken) {
        if (!jwtTokenProvider.validateToken(refreshToken)) {
            throw new RuntimeException("Invalid refresh token");
        }
        
        UUID userId = jwtTokenProvider.getUserIdFromToken(refreshToken);
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        String newAccessToken = jwtTokenProvider.generateAccessToken(user);
        String newRefreshToken = jwtTokenProvider.generateRefreshToken(user);
        
        cacheUserSession(user.getId(), newAccessToken);
        
        return AuthResponse.builder()
                .accessToken(newAccessToken)
                .refreshToken(newRefreshToken)
                .expiresIn(jwtTokenProvider.getAccessTokenExpiration())
                .user(mapToUserDto(user))
                .build();
    }
    
    @Transactional
    public TikTokTokenResponse refreshTikTokToken(UUID userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        if (user.getRefreshTokenEncrypted() == null) {
            throw new RuntimeException("No TikTok refresh token available");
        }
        
        String refreshToken = encryptionService.decrypt(user.getRefreshTokenEncrypted());
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);
        
        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("client_key", tiktokClientKey);
        body.add("client_secret", tiktokClientSecret);
        body.add("grant_type", "refresh_token");
        body.add("refresh_token", refreshToken);
        
        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(body, headers);
        
        ResponseEntity<TikTokTokenResponse> response = restTemplate.exchange(
                TIKTOK_TOKEN_URL,
                HttpMethod.POST,
                request,
                TikTokTokenResponse.class
        );
        
        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new RuntimeException("Failed to refresh TikTok token");
        }
        
        TikTokTokenResponse tokenResponse = response.getBody();
        updateUserTikTokTokens(user, tokenResponse);
        userRepository.save(user);
        
        return tokenResponse;
    }
    
    // ========================================
    // Session Management
    // ========================================
    
    public void logout(String accessToken) {
        UUID userId = jwtTokenProvider.getUserIdFromToken(accessToken);
        redisTemplate.delete("user:session:" + userId);
    }
    
    private void cacheUserSession(UUID userId, String accessToken) {
        redisTemplate.opsForValue().set(
                "user:session:" + userId,
                accessToken,
                Duration.ofHours(24)
        );
    }
    
    public Optional<User> getCurrentUser(String accessToken) {
        if (!jwtTokenProvider.validateToken(accessToken)) {
            return Optional.empty();
        }
        UUID userId = jwtTokenProvider.getUserIdFromToken(accessToken);
        return userRepository.findById(userId);
    }
    
    // ========================================
    // User Operations
    // ========================================
    
    public UserDto getUserProfile(UUID userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));
        return mapToUserDto(user);
    }
    
    @Transactional
    public UserDto updateUserProfile(UUID userId, UpdateProfileRequest request) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        if (request.getFullName() != null) {
            user.setFullName(request.getFullName());
        }
        if (request.getTimezone() != null) {
            user.setTimezone(request.getTimezone());
        }
        if (request.getLanguage() != null) {
            user.setLanguage(request.getLanguage());
        }
        
        user = userRepository.save(user);
        return mapToUserDto(user);
    }
    
    public String getDecryptedTikTokAccessToken(UUID userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        if (user.getAccessTokenEncrypted() == null) {
            throw new RuntimeException("TikTok not connected");
        }
        
        // Check if token is expired
        if (user.getTokenExpiresAt() != null && user.getTokenExpiresAt().isBefore(OffsetDateTime.now())) {
            // Token expired, try to refresh
            refreshTikTokToken(userId);
            user = userRepository.findById(userId).orElseThrow();
        }
        
        return encryptionService.decrypt(user.getAccessTokenEncrypted());
    }
    
    // ========================================
    // Helpers
    // ========================================
    
    private UserDto mapToUserDto(User user) {
        return UserDto.builder()
                .id(user.getId())
                .email(user.getEmail())
                .fullName(user.getFullName())
                .avatarUrl(user.getAvatarUrl())
                .tiktokUserId(user.getTiktokUserId())
                .tiktokUsername(user.getTiktokUsername())
                .tiktokDisplayName(user.getTiktokDisplayName())
                .tiktokAvatarUrl(user.getTiktokAvatarUrl())
                .tiktokFollowerCount(user.getTiktokFollowerCount())
                .tiktokFollowingCount(user.getTiktokFollowingCount())
                .tiktokLikesCount(user.getTiktokLikesCount())
                .tiktokConnected(user.getTiktokUserId() != null)
                .planType(user.getPlanType())
                .monthlyPostsLimit(user.getMonthlyPostsLimit())
                .monthlyPostsUsed(user.getMonthlyPostsUsed())
                .monthlyAiGenerationsLimit(user.getMonthlyAiGenerationsLimit())
                .monthlyAiGenerationsUsed(user.getMonthlyAiGenerationsUsed())
                .timezone(user.getTimezone())
                .language(user.getLanguage())
                .createdAt(user.getCreatedAt())
                .build();
    }
}
