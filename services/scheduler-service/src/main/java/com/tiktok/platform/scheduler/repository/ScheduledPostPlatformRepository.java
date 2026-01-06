package com.tiktok.platform.scheduler.repository;

import com.tiktok.platform.scheduler.entity.ScheduledPostPlatform;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface ScheduledPostPlatformRepository extends JpaRepository<ScheduledPostPlatform, UUID> {

    List<ScheduledPostPlatform> findByScheduledPostId(UUID scheduledPostId);

    Optional<ScheduledPostPlatform> findByScheduledPostIdAndPlatform(UUID scheduledPostId, String platform);

    @Query("SELECT p FROM ScheduledPostPlatform p WHERE p.status = 'failed' AND p.retryCount < 3")
    List<ScheduledPostPlatform> findPlatformsToRetry();

    @Query("SELECT p FROM ScheduledPostPlatform p WHERE p.scheduledPost.userId = :userId AND p.platform = :platform AND p.status = 'published'")
    List<ScheduledPostPlatform> findPublishedByUserAndPlatform(@Param("userId") UUID userId, @Param("platform") String platform);
}
