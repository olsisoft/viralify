package com.tiktok.platform.auth.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;
import java.time.OffsetDateTime;
import java.util.UUID;

@Entity
@Table(name = "users")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class User {
    
    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;
    
    @Column(unique = true, nullable = false)
    private String email;
    
    @Column(name = "password_hash")
    private String passwordHash;
    
    @Column(name = "full_name")
    private String fullName;
    
    @Column(name = "avatar_url")
    private String avatarUrl;
    
    @Column(name = "tiktok_user_id", unique = true)
    private String tiktokUserId;
    
    @Column(name = "tiktok_username")
    private String tiktokUsername;
    
    @Column(name = "tiktok_display_name")
    private String tiktokDisplayName;
    
    @Column(name = "tiktok_avatar_url")
    private String tiktokAvatarUrl;
    
    @Column(name = "tiktok_follower_count")
    private Long tiktokFollowerCount;
    
    @Column(name = "tiktok_following_count")
    private Long tiktokFollowingCount;
    
    @Column(name = "tiktok_likes_count")
    private Long tiktokLikesCount;
    
    @Column(name = "access_token_encrypted")
    private String accessTokenEncrypted;
    
    @Column(name = "refresh_token_encrypted")
    private String refreshTokenEncrypted;
    
    @Column(name = "token_expires_at")
    private OffsetDateTime tokenExpiresAt;
    
    @Column(name = "token_scope")
    private String tokenScope;
    
    @Column(name = "plan_type")
    @Builder.Default
    private String planType = "free";
    
    @Column(name = "plan_expires_at")
    private OffsetDateTime planExpiresAt;
    
    @Column(name = "monthly_posts_limit")
    @Builder.Default
    private Integer monthlyPostsLimit = 10;
    
    @Column(name = "monthly_posts_used")
    @Builder.Default
    private Integer monthlyPostsUsed = 0;
    
    @Column(name = "monthly_ai_generations_limit")
    @Builder.Default
    private Integer monthlyAiGenerationsLimit = 50;
    
    @Column(name = "monthly_ai_generations_used")
    @Builder.Default
    private Integer monthlyAiGenerationsUsed = 0;
    
    @Builder.Default
    private String timezone = "UTC";
    
    @Builder.Default
    private String language = "en";
    
    @Column(name = "email_verified")
    @Builder.Default
    private Boolean emailVerified = false;
    
    @Column(name = "is_active")
    @Builder.Default
    private Boolean isActive = true;
    
    @Column(name = "last_login_at")
    private OffsetDateTime lastLoginAt;
    
    @CreationTimestamp
    @Column(name = "created_at")
    private OffsetDateTime createdAt;
    
    @UpdateTimestamp
    @Column(name = "updated_at")
    private OffsetDateTime updatedAt;
}
