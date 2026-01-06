package com.viralify.platform.connector.service;

import com.viralify.platform.connector.dto.AdaptedContent;
import com.viralify.platform.connector.model.ContentLimits;
import com.viralify.platform.connector.model.Platform;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Service responsible for adapting content to meet platform-specific requirements.
 * Handles duration limits, caption formatting, hashtag optimization, etc.
 */
@Service
@Slf4j
public class ContentAdapterService {

    /**
     * Adapt content for a specific platform
     */
    public AdaptedContent adaptForPlatform(String title, String caption, List<String> hashtags,
                                           Integer durationSeconds, Platform platform) {
        log.info("Adapting content for platform: {}", platform.getDisplayName());

        return switch (platform) {
            case TIKTOK -> adaptForTikTok(title, caption, hashtags, durationSeconds);
            case INSTAGRAM -> adaptForInstagram(title, caption, hashtags, durationSeconds);
            case YOUTUBE -> adaptForYouTube(title, caption, hashtags, durationSeconds);
        };
    }

    /**
     * Adapt content for TikTok
     * - Max 10 min duration
     * - Hashtags in caption
     * - 2200 char caption limit
     */
    private AdaptedContent adaptForTikTok(String title, String caption, List<String> hashtags,
                                          Integer durationSeconds) {
        AdaptedContent adapted = AdaptedContent.builder()
                .platform(Platform.TIKTOK)
                .title(truncateTitle(title, Platform.TIKTOK.getMaxTitleLength()))
                .caption(caption)
                .hashtags(hashtags != null ? new ArrayList<>(hashtags) : new ArrayList<>())
                .contentWasModified(false)
                .build();

        ContentLimits limits = Platform.TIKTOK.toContentLimits();

        // Check duration
        if (durationSeconds != null && durationSeconds > limits.getMaxDurationSeconds()) {
            adapted.setSuggestedDurationSeconds(limits.getMaxDurationSeconds());
            adapted.addNote(String.format("Video will be trimmed from %ds to %ds (TikTok max)",
                    durationSeconds, limits.getMaxDurationSeconds()));
            adapted.setContentWasModified(true);
        }

        // Limit hashtags
        if (adapted.getHashtags().size() > limits.getMaxHashtags()) {
            List<String> limitedHashtags = adapted.getHashtags().subList(0, limits.getMaxHashtags());
            adapted.addNote(String.format("Reduced hashtags from %d to %d",
                    adapted.getHashtags().size(), limits.getMaxHashtags()));
            adapted.setHashtags(new ArrayList<>(limitedHashtags));
            adapted.setContentWasModified(true);
        }

        // Check total caption length with hashtags
        String fullCaption = adapted.getCaptionWithHashtags();
        if (fullCaption.length() > limits.getMaxCaptionLength()) {
            adapted.setCaption(truncateCaption(caption, hashtags, limits.getMaxCaptionLength()));
            adapted.addNote("Caption truncated to fit TikTok limit");
            adapted.setContentWasModified(true);
        }

        return adapted;
    }

    /**
     * Adapt content for Instagram Reels
     * - Max 90 seconds duration
     * - Hashtags in caption (max 30)
     * - 2200 char caption limit
     */
    private AdaptedContent adaptForInstagram(String title, String caption, List<String> hashtags,
                                             Integer durationSeconds) {
        AdaptedContent adapted = AdaptedContent.builder()
                .platform(Platform.INSTAGRAM)
                .title(null) // Instagram doesn't have separate title
                .caption(caption)
                .hashtags(hashtags != null ? new ArrayList<>(hashtags) : new ArrayList<>())
                .contentWasModified(false)
                .build();

        ContentLimits limits = Platform.INSTAGRAM.toContentLimits();

        // Check duration - Instagram Reels max 90 seconds
        if (durationSeconds != null && durationSeconds > limits.getMaxDurationSeconds()) {
            adapted.setSuggestedDurationSeconds(limits.getMaxDurationSeconds());
            adapted.addNote(String.format("Video will be trimmed from %ds to %ds (Instagram Reels max)",
                    durationSeconds, limits.getMaxDurationSeconds()));
            adapted.setContentWasModified(true);
        }

        // Instagram strictly enforces 30 hashtag limit
        if (adapted.getHashtags().size() > 30) {
            // Keep the first 30 hashtags (prioritize user's first choices)
            List<String> limitedHashtags = adapted.getHashtags().subList(0, 30);
            adapted.addNote(String.format("Reduced hashtags from %d to 30 (Instagram max)",
                    adapted.getHashtags().size()));
            adapted.setHashtags(new ArrayList<>(limitedHashtags));
            adapted.setContentWasModified(true);
        }

        // Check total caption length
        String fullCaption = adapted.getCaptionWithHashtags();
        if (fullCaption.length() > limits.getMaxCaptionLength()) {
            adapted.setCaption(truncateCaption(caption, adapted.getHashtags(), limits.getMaxCaptionLength()));
            adapted.addNote("Caption truncated to fit Instagram limit");
            adapted.setContentWasModified(true);
        }

        return adapted;
    }

    /**
     * Adapt content for YouTube Shorts
     * - Max 60 seconds duration
     * - Title required (max 100 chars)
     * - Description (caption) max 5000 chars
     * - Tags separate from description
     * - Must include #Shorts
     */
    private AdaptedContent adaptForYouTube(String title, String caption, List<String> hashtags,
                                           Integer durationSeconds) {
        AdaptedContent adapted = AdaptedContent.builder()
                .platform(Platform.YOUTUBE)
                .title(title)
                .caption(caption)
                .hashtags(new ArrayList<>()) // YouTube doesn't use hashtags in description same way
                .tags(new ArrayList<>())
                .contentWasModified(false)
                .build();

        ContentLimits limits = Platform.YOUTUBE.toContentLimits();

        // Check duration - YouTube Shorts max 60 seconds
        if (durationSeconds != null && durationSeconds > limits.getMaxDurationSeconds()) {
            adapted.setSuggestedDurationSeconds(limits.getMaxDurationSeconds());
            adapted.addNote(String.format("Video will be trimmed from %ds to %ds (YouTube Shorts max)",
                    durationSeconds, limits.getMaxDurationSeconds()));
            adapted.setContentWasModified(true);
        }

        // Ensure #Shorts is in title or description
        String adaptedTitle = ensureShortsTag(title, limits.getMaxTitleLength());
        adapted.setTitle(adaptedTitle);

        // Convert hashtags to YouTube tags
        if (hashtags != null && !hashtags.isEmpty()) {
            List<String> youtubeTags = convertToYouTubeTags(hashtags, limits.getMaxHashtags());
            adapted.setTags(youtubeTags);
            adapted.addNote("Hashtags converted to YouTube tags");
            adapted.setContentWasModified(true);
        }

        // Build YouTube description
        StringBuilder description = new StringBuilder();
        if (caption != null && !caption.isEmpty()) {
            description.append(caption);
        }

        // Add some hashtags to description for discoverability
        if (hashtags != null && !hashtags.isEmpty()) {
            description.append("\n\n");
            int hashtagsToAdd = Math.min(5, hashtags.size()); // Add up to 5 hashtags
            for (int i = 0; i < hashtagsToAdd; i++) {
                String tag = hashtags.get(i);
                if (!tag.startsWith("#")) {
                    description.append("#");
                }
                description.append(tag).append(" ");
            }
        }

        adapted.setDescription(truncateText(description.toString().trim(), limits.getMaxCaptionLength()));

        return adapted;
    }

    /**
     * Ensure YouTube title contains #Shorts for proper categorization
     */
    private String ensureShortsTag(String title, int maxLength) {
        if (title == null || title.isEmpty()) {
            return "#Shorts";
        }

        // Check if #Shorts already present
        if (title.toLowerCase().contains("#shorts")) {
            return truncateTitle(title, maxLength);
        }

        // Add #Shorts to title
        String withShorts = title + " #Shorts";
        if (withShorts.length() <= maxLength) {
            return withShorts;
        }

        // Need to truncate title to fit #Shorts
        int availableLength = maxLength - " #Shorts".length();
        return truncateTitle(title, availableLength) + " #Shorts";
    }

    /**
     * Convert hashtags to YouTube tags format
     * YouTube tags are without # prefix and have a total character limit
     */
    private List<String> convertToYouTubeTags(List<String> hashtags, int maxTotalChars) {
        List<String> tags = new ArrayList<>();
        int totalChars = 0;

        for (String hashtag : hashtags) {
            // Remove # prefix for YouTube tags
            String tag = hashtag.startsWith("#") ? hashtag.substring(1) : hashtag;

            // Check if adding this tag would exceed limit
            if (totalChars + tag.length() + 1 > maxTotalChars) { // +1 for separator
                break;
            }

            tags.add(tag);
            totalChars += tag.length() + 1;
        }

        return tags;
    }

    /**
     * Truncate title to max length, trying to preserve whole words
     */
    private String truncateTitle(String title, int maxLength) {
        if (title == null || title.isEmpty() || maxLength <= 0) {
            return title;
        }

        if (title.length() <= maxLength) {
            return title;
        }

        // Try to cut at word boundary
        String truncated = title.substring(0, maxLength);
        int lastSpace = truncated.lastIndexOf(' ');

        if (lastSpace > maxLength * 0.7) { // Only cut at space if it's not too far back
            return truncated.substring(0, lastSpace).trim();
        }

        return truncated.trim();
    }

    /**
     * Truncate text to max length
     */
    private String truncateText(String text, int maxLength) {
        if (text == null || text.length() <= maxLength) {
            return text;
        }
        return text.substring(0, maxLength - 3) + "...";
    }

    /**
     * Truncate caption while preserving space for hashtags
     */
    private String truncateCaption(String caption, List<String> hashtags, int maxLength) {
        if (caption == null) return null;

        // Calculate space needed for hashtags
        int hashtagsLength = 0;
        if (hashtags != null) {
            hashtagsLength = hashtags.stream()
                    .mapToInt(h -> h.length() + 2) // +2 for # and space
                    .sum();
        }

        // Reserve space for hashtags + newlines
        int availableForCaption = maxLength - hashtagsLength - 4; // -4 for \n\n separators

        if (availableForCaption < 50) {
            // Not enough room, prioritize some caption
            availableForCaption = Math.min(100, maxLength / 2);
        }

        return truncateText(caption, availableForCaption);
    }

    /**
     * Get all platform adaptations at once (for preview)
     */
    public List<AdaptedContent> adaptForAllPlatforms(String title, String caption,
                                                      List<String> hashtags, Integer durationSeconds,
                                                      List<Platform> platforms) {
        return platforms.stream()
                .map(platform -> adaptForPlatform(title, caption, hashtags, durationSeconds, platform))
                .collect(Collectors.toList());
    }
}
