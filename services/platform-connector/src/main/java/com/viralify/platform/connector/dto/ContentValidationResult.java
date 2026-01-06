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
public class ContentValidationResult {
    private Platform platform;
    private boolean valid;

    @Builder.Default
    private List<String> errors = new ArrayList<>();

    @Builder.Default
    private List<String> warnings = new ArrayList<>();

    // Suggested adaptations
    private Integer suggestedDurationSeconds;
    private String suggestedCaption;
    private List<String> suggestedHashtags;

    public static ContentValidationResult valid(Platform platform) {
        return ContentValidationResult.builder()
                .platform(platform)
                .valid(true)
                .build();
    }

    public static ContentValidationResult invalid(Platform platform, List<String> errors) {
        return ContentValidationResult.builder()
                .platform(platform)
                .valid(false)
                .errors(errors)
                .build();
    }

    public void addError(String error) {
        if (errors == null) errors = new ArrayList<>();
        errors.add(error);
        valid = false;
    }

    public void addWarning(String warning) {
        if (warnings == null) warnings = new ArrayList<>();
        warnings.add(warning);
    }

    public boolean hasWarnings() {
        return warnings != null && !warnings.isEmpty();
    }

    public boolean hasErrors() {
        return errors != null && !errors.isEmpty();
    }
}
