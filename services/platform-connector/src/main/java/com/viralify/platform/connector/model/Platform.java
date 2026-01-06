package com.viralify.platform.connector.model;

import lombok.Getter;

@Getter
public enum Platform {
    TIKTOK(
        "TikTok",
        600,        // 10 minutes max
        2200,       // caption length
        30,         // max hashtags (soft limit)
        300,        // title length
        "9:16",     // aspect ratio
        4L * 1024 * 1024 * 1024, // 4GB max file size
        true,       // hashtags in caption
        true,       // supports duet
        true        // supports stitch
    ),
    INSTAGRAM(
        "Instagram Reels",
        90,         // 90 seconds max
        2200,       // caption length
        30,         // max hashtags
        0,          // no separate title
        "9:16",     // aspect ratio
        4L * 1024 * 1024 * 1024, // 4GB max file size
        true,       // hashtags in caption
        false,      // no duet
        false       // no stitch
    ),
    YOUTUBE(
        "YouTube Shorts",
        60,         // 60 seconds max for Shorts
        5000,       // description length
        500,        // tags total chars (not count)
        100,        // title length
        "9:16",     // aspect ratio
        256L * 1024 * 1024 * 1024, // 256GB max (general YouTube)
        false,      // hashtags/tags separate from description
        false,      // no duet
        false       // no stitch
    );

    private final String displayName;
    private final int maxDurationSeconds;
    private final int maxCaptionLength;
    private final int maxHashtags;
    private final int maxTitleLength;
    private final String aspectRatio;
    private final long maxFileSizeBytes;
    private final boolean hashtagsInCaption;
    private final boolean supportsDuet;
    private final boolean supportsStitch;

    Platform(String displayName, int maxDurationSeconds, int maxCaptionLength,
             int maxHashtags, int maxTitleLength, String aspectRatio,
             long maxFileSizeBytes, boolean hashtagsInCaption,
             boolean supportsDuet, boolean supportsStitch) {
        this.displayName = displayName;
        this.maxDurationSeconds = maxDurationSeconds;
        this.maxCaptionLength = maxCaptionLength;
        this.maxHashtags = maxHashtags;
        this.maxTitleLength = maxTitleLength;
        this.aspectRatio = aspectRatio;
        this.maxFileSizeBytes = maxFileSizeBytes;
        this.hashtagsInCaption = hashtagsInCaption;
        this.supportsDuet = supportsDuet;
        this.supportsStitch = supportsStitch;
    }

    public ContentLimits toContentLimits() {
        return ContentLimits.builder()
                .platform(this)
                .maxDurationSeconds(maxDurationSeconds)
                .maxCaptionLength(maxCaptionLength)
                .maxHashtags(maxHashtags)
                .maxTitleLength(maxTitleLength)
                .aspectRatio(aspectRatio)
                .maxFileSizeBytes(maxFileSizeBytes)
                .hashtagsInCaption(hashtagsInCaption)
                .build();
    }
}
