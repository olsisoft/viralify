package com.viralify.platform.connector.dto;

import com.viralify.platform.connector.model.Platform;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AdaptedContent {
    private Platform platform;

    // Adapted content
    private String title;
    private String caption;
    private List<String> hashtags;
    private Integer suggestedDurationSeconds;

    // YouTube specific
    private List<String> tags;
    private String description;

    // Adaptation info
    private boolean contentWasModified;

    @Builder.Default
    private List<String> adaptationNotes = new ArrayList<>();

    public void addNote(String note) {
        if (adaptationNotes == null) adaptationNotes = new ArrayList<>();
        adaptationNotes.add(note);
    }

    /**
     * Build caption with hashtags for platforms that include them in caption
     */
    public String getCaptionWithHashtags() {
        if (hashtags == null || hashtags.isEmpty()) {
            return caption;
        }

        if (!platform.isHashtagsInCaption()) {
            return caption;
        }

        StringBuilder sb = new StringBuilder();
        if (caption != null && !caption.isEmpty()) {
            sb.append(caption);
            if (!caption.endsWith("\n")) {
                sb.append("\n\n");
            }
        }

        for (String hashtag : hashtags) {
            if (!hashtag.startsWith("#")) {
                sb.append("#");
            }
            sb.append(hashtag).append(" ");
        }

        return sb.toString().trim();
    }
}
