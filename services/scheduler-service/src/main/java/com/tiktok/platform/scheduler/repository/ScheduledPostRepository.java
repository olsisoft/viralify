package com.tiktok.platform.scheduler.repository;

import com.tiktok.platform.scheduler.entity.ScheduledPost;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import java.time.OffsetDateTime;
import java.util.List;
import java.util.UUID;

public interface ScheduledPostRepository extends JpaRepository<ScheduledPost, UUID> {

    List<ScheduledPost> findByUserIdOrderByScheduledAtDesc(UUID userId);

    List<ScheduledPost> findByUserIdAndStatusOrderByScheduledAtAsc(UUID userId, String status);

    @Query("SELECT p FROM ScheduledPost p WHERE p.status = 'pending' AND p.scheduledAt <= :now ORDER BY p.scheduledAt ASC")
    List<ScheduledPost> findPostsReadyToPublish(@Param("now") OffsetDateTime now);

    @Query("SELECT p FROM ScheduledPost p WHERE p.status = 'failed' AND p.retryCount < p.maxRetries ORDER BY p.scheduledAt ASC")
    List<ScheduledPost> findPostsToRetry();

    @Query("SELECT COUNT(p) FROM ScheduledPost p WHERE p.userId = :userId AND p.status = 'published' AND p.publishedAt >= :since")
    Long countPublishedSince(@Param("userId") UUID userId, @Param("since") OffsetDateTime since);

    List<ScheduledPost> findByStatusIn(List<String> statuses);
}
