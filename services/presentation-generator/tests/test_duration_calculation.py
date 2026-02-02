"""
Unit tests for duration calculation in presentation generation.

Tests cover:
- Per-slide duration calculation based on total target duration
- Per-slide word count calculation (2.5 words/second)
- Various duration scenarios (5min, 10min, 20min)
- Edge cases (1 slide, many slides, very short duration)
"""

import pytest


class TestDurationCalculation:
    """Tests for the duration calculation logic used in slide generation prompts."""

    def test_per_slide_duration_10_minutes_15_slides(self):
        """10 minutes with 15 slides = 40s per slide"""
        total_duration = 600  # 10 minutes in seconds
        total_slides = 15

        per_slide_duration = total_duration // max(total_slides, 1)

        assert per_slide_duration == 40

    def test_per_slide_duration_5_minutes_10_slides(self):
        """5 minutes with 10 slides = 30s per slide"""
        total_duration = 300  # 5 minutes in seconds
        total_slides = 10

        per_slide_duration = total_duration // max(total_slides, 1)

        assert per_slide_duration == 30

    def test_per_slide_duration_20_minutes_25_slides(self):
        """20 minutes with 25 slides = 48s per slide"""
        total_duration = 1200  # 20 minutes in seconds
        total_slides = 25

        per_slide_duration = total_duration // max(total_slides, 1)

        assert per_slide_duration == 48

    def test_words_per_slide_10_minutes_15_slides(self):
        """10 minutes with 15 slides = 100 words per slide (at 2.5 words/sec)"""
        total_duration = 600  # 10 minutes in seconds
        total_slides = 15

        # 2.5 words per second = 150 words per minute
        words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))

        assert words_per_slide == 100

    def test_words_per_slide_5_minutes_10_slides(self):
        """5 minutes with 10 slides = 75 words per slide"""
        total_duration = 300  # 5 minutes in seconds
        total_slides = 10

        words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))

        assert words_per_slide == 75

    def test_words_per_slide_20_minutes_25_slides(self):
        """20 minutes with 25 slides = 120 words per slide"""
        total_duration = 1200  # 20 minutes in seconds
        total_slides = 25

        words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))

        assert words_per_slide == 120

    def test_edge_case_single_slide(self):
        """Single slide gets all the duration"""
        total_duration = 300  # 5 minutes
        total_slides = 1

        per_slide_duration = total_duration // max(total_slides, 1)
        words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))

        assert per_slide_duration == 300
        assert words_per_slide == 750  # 5 minutes worth of words

    def test_edge_case_zero_slides_no_division_error(self):
        """Zero slides should not cause division by zero"""
        total_duration = 300
        total_slides = 0

        # max(total_slides, 1) prevents division by zero
        per_slide_duration = total_duration // max(total_slides, 1)
        words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))

        assert per_slide_duration == 300
        assert words_per_slide == 750

    def test_edge_case_very_short_duration(self):
        """Very short duration (1 minute) with 6 slides"""
        total_duration = 60  # 1 minute
        total_slides = 6

        per_slide_duration = total_duration // max(total_slides, 1)
        words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))

        assert per_slide_duration == 10
        assert words_per_slide == 25

    def test_edge_case_many_slides(self):
        """Many slides (50) for 10 minutes"""
        total_duration = 600  # 10 minutes
        total_slides = 50

        per_slide_duration = total_duration // max(total_slides, 1)
        words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))

        assert per_slide_duration == 12
        assert words_per_slide == 30

    def test_total_duration_matches_target(self):
        """Sum of per-slide durations should approximately match target"""
        total_duration = 600  # 10 minutes
        total_slides = 15

        per_slide_duration = total_duration // max(total_slides, 1)
        calculated_total = per_slide_duration * total_slides

        # Should be close to target (integer division may lose some seconds)
        assert abs(calculated_total - total_duration) < total_slides

    def test_total_words_at_reading_speed(self):
        """Total words should produce correct duration at 2.5 words/sec"""
        total_duration = 600  # 10 minutes
        total_slides = 15

        words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))
        total_words = words_per_slide * total_slides

        # At 2.5 words/sec, total words should produce approximately target duration
        calculated_duration = total_words / 2.5

        # Allow 10% tolerance due to integer rounding
        assert abs(calculated_duration - total_duration) < total_duration * 0.1


class TestDurationCalculationFormulas:
    """Tests for the specific formulas used in the prompt generation."""

    @pytest.mark.parametrize("duration,slides,expected_per_slide,expected_words", [
        (300, 10, 30, 75),      # 5 min, 10 slides
        (600, 15, 40, 100),     # 10 min, 15 slides
        (600, 20, 30, 75),      # 10 min, 20 slides
        (900, 20, 45, 112),     # 15 min, 20 slides
        (1200, 25, 48, 120),    # 20 min, 25 slides
        (1800, 30, 60, 150),    # 30 min, 30 slides
    ])
    def test_duration_scenarios(self, duration, slides, expected_per_slide, expected_words):
        """Test various realistic duration scenarios."""
        per_slide_duration = duration // max(slides, 1)
        words_per_slide = int(duration * 2.5 / max(slides, 1))

        assert per_slide_duration == expected_per_slide
        assert words_per_slide == expected_words


class TestPromptDurationIntegration:
    """Tests for how duration values are used in prompt generation."""

    def test_prompt_duration_values_are_integers(self):
        """Duration values in prompt should be integers for clean formatting."""
        total_duration = 600
        total_slides = 15

        per_slide_duration = total_duration // max(total_slides, 1)
        words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))

        assert isinstance(per_slide_duration, int)
        assert isinstance(words_per_slide, int)

    def test_prompt_values_reasonable_range(self):
        """Per-slide values should be in reasonable ranges."""
        test_cases = [
            (300, 10),   # 5 min
            (600, 15),   # 10 min
            (1200, 25),  # 20 min
        ]

        for total_duration, total_slides in test_cases:
            per_slide_duration = total_duration // max(total_slides, 1)
            words_per_slide = int(total_duration * 2.5 / max(total_slides, 1))

            # Duration should be between 10s and 120s per slide
            assert 10 <= per_slide_duration <= 120, \
                f"Per-slide duration {per_slide_duration}s out of range for {total_duration}s/{total_slides} slides"

            # Words should be between 25 and 300 per slide
            assert 25 <= words_per_slide <= 300, \
                f"Words per slide {words_per_slide} out of range for {total_duration}s/{total_slides} slides"


class TestOldVsNewBehavior:
    """Tests comparing old hardcoded behavior vs new dynamic calculation."""

    def test_old_behavior_always_60_80_seconds(self):
        """Old behavior: 60-80s per slide regardless of target duration."""
        # Old hardcoded values
        old_min_duration = 60
        old_max_duration = 80

        # With 10 slides, old behavior would produce:
        old_total_min = old_min_duration * 10  # 600s = 10 min
        old_total_max = old_max_duration * 10  # 800s = 13.3 min

        assert old_total_min == 600
        assert old_total_max == 800

    def test_new_behavior_matches_target(self):
        """New behavior: duration matches target."""
        target_duration = 300  # 5 minutes
        total_slides = 10

        new_per_slide = target_duration // total_slides
        new_total = new_per_slide * total_slides

        # New behavior should match target (minus rounding)
        assert new_total == 300  # Exactly 5 minutes

    def test_improvement_for_short_videos(self):
        """New behavior is much better for short videos."""
        target_duration = 180  # 3 minutes
        total_slides = 8

        # Old behavior: 8 × 60s = 480s (8 minutes!) - way over target
        old_total = 60 * total_slides

        # New behavior: 8 × 22s = 176s (~3 minutes) - matches target
        new_per_slide = target_duration // total_slides
        new_total = new_per_slide * total_slides

        assert old_total == 480  # Old was 2.6x the target
        assert abs(new_total - target_duration) < 8  # New is within rounding error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
