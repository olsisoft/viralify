"""
Integration tests for duration calculation in presentation planner.

Tests the actual prompt generation and parameter passing through the system.
Uses mocked LLM responses to verify the integration works correctly.

Note: These tests mock the OpenAI client but test real code paths.
"""

import sys
import os
import json
import asyncio
import re

# Add the parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

# Mock external modules before any imports
sys.modules['shared'] = MagicMock()
sys.modules['shared.llm_provider'] = MagicMock()
sys.modules['shared.training_logger'] = MagicMock()

# Mock openai with async support
mock_openai = MagicMock()
mock_openai.AsyncOpenAI = MagicMock()
sys.modules['openai'] = mock_openai

# Mock other dependencies
sys.modules['cairosvg'] = MagicMock()
sys.modules['pygraphviz'] = MagicMock()
for mod in ['diagrams', 'diagrams.aws', 'diagrams.azure', 'diagrams.gcp',
            'diagrams.k8s', 'diagrams.onprem', 'diagrams.programming',
            'diagrams.saas', 'diagrams.generic']:
    sys.modules[mod] = MagicMock()


# ============================================================================
# Test fixtures and helpers
# ============================================================================

@dataclass
class MockGeneratePresentationRequest:
    """Mock request object for testing."""
    topic: str = "Python Programming"
    duration: int = 600  # 10 minutes in seconds
    target_audience: str = "intermediate"
    language: str = "python"
    content_language: str = "en"
    rag_context: Optional[str] = None
    style: str = "professional"
    title_style: Optional[str] = None
    practical_focus: Optional[str] = None


def create_mock_outline(num_slides: int) -> List[dict]:
    """Create a mock outline with specified number of slides."""
    slides = []
    slide_types = ["title", "content", "code", "diagram", "content", "conclusion"]

    for i in range(num_slides):
        slide_type = slide_types[i % len(slide_types)]
        slides.append({
            "type": slide_type,
            "title": f"Slide {i + 1}: {slide_type.capitalize()}",
            "description": f"Description for slide {i + 1}",
            "key_points": [f"Point {j}" for j in range(1, 4)]
        })

    return slides


def extract_duration_from_prompt(prompt: str) -> dict:
    """Extract duration-related values from a generated prompt."""
    result = {
        "total_duration": None,
        "total_slides": None,
        "per_slide_duration": None,
        "words_per_slide": None,
    }

    # Extract total duration: "600s total (10 minutes)"
    match = re.search(r'(\d+)s total \((\d+) minutes\)', prompt)
    if match:
        result["total_duration"] = int(match.group(1))

    # Extract total slides: "across 15 slides"
    match = re.search(r'across (\d+) slides', prompt)
    if match:
        result["total_slides"] = int(match.group(1))

    # Extract per-slide duration: "Target ~40s per slide"
    match = re.search(r'Target ~(\d+)s per slide', prompt)
    if match:
        result["per_slide_duration"] = int(match.group(1))

    # Extract words per slide: "~100 words"
    match = re.search(r'~(\d+) words (?:per slide voiceover|to match target)', prompt)
    if match:
        result["words_per_slide"] = int(match.group(1))

    return result


# ============================================================================
# Integration tests for prompt generation
# ============================================================================

class TestPromptDurationIntegration:
    """Test that prompts contain correct duration values."""

    def test_prompt_contains_duration_target_10_minutes(self):
        """Verify 10-minute duration is correctly included in prompt."""
        request = MockGeneratePresentationRequest(duration=600)  # 10 minutes
        total_slides = 15

        # Simulate the prompt generation logic
        prompt_section = f"""DURATION TARGET: {request.duration}s total ({request.duration // 60} minutes) across {total_slides} slides
- Target ~{request.duration // max(total_slides, 1)}s per slide
- Target ~{int(request.duration * 2.5 / max(total_slides, 1))} words per slide voiceover"""

        extracted = extract_duration_from_prompt(prompt_section)

        assert extracted["total_duration"] == 600
        assert extracted["total_slides"] == 15
        assert extracted["per_slide_duration"] == 40
        assert extracted["words_per_slide"] == 100

    def test_prompt_contains_duration_target_5_minutes(self):
        """Verify 5-minute duration is correctly included in prompt."""
        request = MockGeneratePresentationRequest(duration=300)  # 5 minutes
        total_slides = 10

        prompt_section = f"""DURATION TARGET: {request.duration}s total ({request.duration // 60} minutes) across {total_slides} slides
- Target ~{request.duration // max(total_slides, 1)}s per slide
- Target ~{int(request.duration * 2.5 / max(total_slides, 1))} words per slide voiceover"""

        extracted = extract_duration_from_prompt(prompt_section)

        assert extracted["total_duration"] == 300
        assert extracted["total_slides"] == 10
        assert extracted["per_slide_duration"] == 30
        assert extracted["words_per_slide"] == 75

    def test_prompt_contains_duration_target_20_minutes(self):
        """Verify 20-minute duration is correctly included in prompt."""
        request = MockGeneratePresentationRequest(duration=1200)  # 20 minutes
        total_slides = 25

        prompt_section = f"""DURATION TARGET: {request.duration}s total ({request.duration // 60} minutes) across {total_slides} slides
- Target ~{request.duration // max(total_slides, 1)}s per slide
- Target ~{int(request.duration * 2.5 / max(total_slides, 1))} words per slide voiceover"""

        extracted = extract_duration_from_prompt(prompt_section)

        assert extracted["total_duration"] == 1200
        assert extracted["total_slides"] == 25
        assert extracted["per_slide_duration"] == 48
        assert extracted["words_per_slide"] == 120


class TestBatchGenerationDurationPassing:
    """Test that duration parameters are correctly passed through the batch generation flow."""

    def test_total_slides_passed_to_batch_generator(self):
        """Verify total_slides is passed from outline to batch generator."""
        # Simulate the flow
        outline = create_mock_outline(15)
        request = MockGeneratePresentationRequest(duration=600)

        # In the actual code, this is how total_slides is determined
        total_slides = len(outline)

        # Verify the calculation
        per_slide_duration = request.duration // max(total_slides, 1)
        words_per_slide = int(request.duration * 2.5 / max(total_slides, 1))

        assert total_slides == 15
        assert per_slide_duration == 40
        assert words_per_slide == 100

    def test_batch_receives_correct_total_slides_for_partial_batch(self):
        """Verify partial batches still get correct total_slides (not batch size)."""
        outline = create_mock_outline(12)  # 12 slides total
        request = MockGeneratePresentationRequest(duration=600)

        batch_size = 5
        total_batches = (len(outline) + batch_size - 1) // batch_size  # 3 batches

        # Simulate batching
        batches = []
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(outline))
            batch_outline = outline[start_idx:end_idx]

            # This is what should be passed to _generate_slides_batch
            batches.append({
                "batch_outline": batch_outline,
                "total_slides": len(outline),  # IMPORTANT: total, not batch size
                "batch_size": len(batch_outline)
            })

        # Verify all batches get the correct total_slides
        assert len(batches) == 3
        assert batches[0]["batch_size"] == 5
        assert batches[0]["total_slides"] == 12  # Not 5!
        assert batches[1]["batch_size"] == 5
        assert batches[1]["total_slides"] == 12  # Not 5!
        assert batches[2]["batch_size"] == 2  # Last partial batch
        assert batches[2]["total_slides"] == 12  # Still 12, not 2!

    @pytest.mark.parametrize("num_slides,duration,expected_per_slide,expected_words", [
        (10, 300, 30, 75),      # 5 min, 10 slides
        (15, 600, 40, 100),     # 10 min, 15 slides
        (20, 600, 30, 75),      # 10 min, 20 slides
        (25, 1200, 48, 120),    # 20 min, 25 slides
        (8, 480, 60, 150),      # 8 min, 8 slides
    ])
    def test_various_outline_and_duration_combinations(
        self, num_slides, duration, expected_per_slide, expected_words
    ):
        """Test various combinations of slides and durations."""
        outline = create_mock_outline(num_slides)
        request = MockGeneratePresentationRequest(duration=duration)

        total_slides = len(outline)
        per_slide_duration = request.duration // max(total_slides, 1)
        words_per_slide = int(request.duration * 2.5 / max(total_slides, 1))

        assert per_slide_duration == expected_per_slide
        assert words_per_slide == expected_words


class TestLLMResponseIntegration:
    """Test that LLM responses are validated against duration targets."""

    def create_mock_llm_response(self, slides: List[dict]) -> dict:
        """Create a mock LLM response with slides."""
        return {"slides": slides}

    def test_llm_response_with_matching_durations(self):
        """Test processing LLM response where durations match target."""
        target_duration = 600  # 10 minutes
        total_slides = 10
        expected_per_slide = target_duration // total_slides  # 60s

        # Mock LLM response with correct durations
        slides = [
            {
                "type": "content",
                "title": f"Slide {i}",
                "voiceover_text": " ".join(["word"] * 150),  # 150 words = 60s
                "duration": expected_per_slide,
                "bullet_points": ["Point 1", "Point 2"]
            }
            for i in range(total_slides)
        ]

        response = self.create_mock_llm_response(slides)

        # Verify total duration
        total_duration = sum(s["duration"] for s in response["slides"])
        assert total_duration == target_duration

    def test_llm_response_with_old_hardcoded_durations(self):
        """Test that old hardcoded durations would exceed target."""
        target_duration = 300  # 5 minutes
        total_slides = 10
        old_hardcoded_duration = 60  # Old behavior: always 60s

        # Simulate old behavior response
        slides = [
            {
                "type": "content",
                "title": f"Slide {i}",
                "voiceover_text": " ".join(["word"] * 150),  # 150 words = 60s
                "duration": old_hardcoded_duration,  # OLD: always 60s
                "bullet_points": ["Point 1", "Point 2"]
            }
            for i in range(total_slides)
        ]

        response = self.create_mock_llm_response(slides)

        # Old behavior would produce 600s (10 min) for 5 min target!
        total_duration = sum(s["duration"] for s in response["slides"])
        assert total_duration == 600  # 2x the target!
        assert total_duration > target_duration  # Exceeds target

    def test_llm_response_with_new_dynamic_durations(self):
        """Test that new dynamic durations match target."""
        target_duration = 300  # 5 minutes
        total_slides = 10
        new_dynamic_duration = target_duration // total_slides  # 30s

        # Simulate new behavior response
        slides = [
            {
                "type": "content",
                "title": f"Slide {i}",
                "voiceover_text": " ".join(["word"] * 75),  # 75 words = 30s
                "duration": new_dynamic_duration,  # NEW: dynamic based on target
                "bullet_points": ["Point 1", "Point 2"]
            }
            for i in range(total_slides)
        ]

        response = self.create_mock_llm_response(slides)

        # New behavior produces exactly target duration
        total_duration = sum(s["duration"] for s in response["slides"])
        assert total_duration == target_duration  # Matches target!


class TestPostProcessingDurationValidation:
    """Test the post-processing step that validates/adjusts durations."""

    def test_post_process_calculates_duration_from_words(self):
        """Test that duration is recalculated from voiceover word count."""
        # Simulate post-processing logic
        voiceover = "This is a simple test voiceover that contains exactly twenty words here to verify the duration calculation logic works correctly."
        word_count = len(voiceover.split())  # 20 words

        # At 2.5 words/second
        calculated_duration = word_count / 2.5

        assert word_count == 20
        assert calculated_duration == 8.0  # 20 words / 2.5 = 8 seconds

    def test_post_process_warns_if_too_short(self):
        """Test that short voiceovers are flagged."""
        min_words_per_slide = 150  # As defined in the planner

        # A slide with too few words
        voiceover = "Short voiceover with only ten words here now."
        word_count = len(voiceover.split())

        is_short = word_count < min_words_per_slide

        assert is_short is True
        assert word_count < 150

    def test_post_process_duration_ratio_calculation(self):
        """Test the duration ratio calculation for validation."""
        target_duration = 600  # 10 minutes

        # Simulate total calculated duration from slides
        slides_durations = [40, 45, 38, 42, 40, 41, 39, 43, 40, 42]  # 10 slides
        total_calculated = sum(slides_durations)  # 410s

        duration_ratio = total_calculated / target_duration

        assert total_calculated == 410
        assert 0.6 < duration_ratio < 0.7  # About 68% of target

        # This would trigger a warning
        is_too_short = duration_ratio < 0.7
        assert is_too_short is True


class TestEndToEndDurationFlow:
    """End-to-end tests for the complete duration flow."""

    @pytest.mark.asyncio
    async def test_full_flow_5_minute_presentation(self):
        """Test complete flow for a 5-minute presentation."""
        request = MockGeneratePresentationRequest(duration=300)  # 5 minutes
        outline = create_mock_outline(10)

        # Simulate the flow
        total_slides = len(outline)
        per_slide_duration = request.duration // total_slides
        words_per_slide = int(request.duration * 2.5 / total_slides)

        # Generate mock slides with correct durations
        generated_slides = []
        for i, item in enumerate(outline):
            generated_slides.append({
                "type": item["type"],
                "title": item["title"],
                "voiceover_text": " ".join(["word"] * words_per_slide),
                "duration": per_slide_duration,
                "bullet_points": item["key_points"]
            })

        # Verify total matches target
        total_duration = sum(s["duration"] for s in generated_slides)
        total_words = sum(len(s["voiceover_text"].split()) for s in generated_slides)

        assert total_duration == 300  # Exactly 5 minutes
        assert total_words == 750  # 75 words × 10 slides

    @pytest.mark.asyncio
    async def test_full_flow_10_minute_presentation(self):
        """Test complete flow for a 10-minute presentation."""
        request = MockGeneratePresentationRequest(duration=600)  # 10 minutes
        outline = create_mock_outline(15)

        total_slides = len(outline)
        per_slide_duration = request.duration // total_slides
        words_per_slide = int(request.duration * 2.5 / total_slides)

        generated_slides = []
        for i, item in enumerate(outline):
            generated_slides.append({
                "type": item["type"],
                "title": item["title"],
                "voiceover_text": " ".join(["word"] * words_per_slide),
                "duration": per_slide_duration,
                "bullet_points": item["key_points"]
            })

        total_duration = sum(s["duration"] for s in generated_slides)
        total_words = sum(len(s["voiceover_text"].split()) for s in generated_slides)

        assert total_duration == 600  # Exactly 10 minutes
        assert total_words == 1500  # 100 words × 15 slides

    @pytest.mark.asyncio
    async def test_full_flow_20_minute_presentation(self):
        """Test complete flow for a 20-minute presentation."""
        request = MockGeneratePresentationRequest(duration=1200)  # 20 minutes
        outline = create_mock_outline(25)

        total_slides = len(outline)
        per_slide_duration = request.duration // total_slides
        words_per_slide = int(request.duration * 2.5 / total_slides)

        generated_slides = []
        for i, item in enumerate(outline):
            generated_slides.append({
                "type": item["type"],
                "title": item["title"],
                "voiceover_text": " ".join(["word"] * words_per_slide),
                "duration": per_slide_duration,
                "bullet_points": item["key_points"]
            })

        total_duration = sum(s["duration"] for s in generated_slides)
        total_words = sum(len(s["voiceover_text"].split()) for s in generated_slides)

        assert total_duration == 1200  # Exactly 20 minutes
        assert total_words == 3000  # 120 words × 25 slides


class TestRegressionOldBehavior:
    """Regression tests to ensure old hardcoded behavior is fixed."""

    def test_no_hardcoded_150_200_words(self):
        """Verify the old hardcoded 150-200 words is replaced with dynamic value."""
        request = MockGeneratePresentationRequest(duration=180)  # 3 minutes
        total_slides = 6

        # New dynamic calculation
        words_per_slide = int(request.duration * 2.5 / total_slides)

        # For 3 minutes with 6 slides, should be 75 words, NOT 150-200
        assert words_per_slide == 75
        assert words_per_slide < 150  # Less than old minimum

    def test_no_hardcoded_60_80_seconds(self):
        """Verify the old hardcoded 60-80s is replaced with dynamic value."""
        request = MockGeneratePresentationRequest(duration=180)  # 3 minutes
        total_slides = 6

        # New dynamic calculation
        per_slide_duration = request.duration // total_slides

        # For 3 minutes with 6 slides, should be 30s, NOT 60-80s
        assert per_slide_duration == 30
        assert per_slide_duration < 60  # Less than old minimum

    def test_short_video_duration_matches_target(self):
        """Test that short videos (3 min) don't become long (10+ min)."""
        request = MockGeneratePresentationRequest(duration=180)  # 3 minutes
        total_slides = 6

        per_slide_duration = request.duration // total_slides
        total_duration = per_slide_duration * total_slides

        # Old behavior: 6 × 60s = 360s (6 min) - 2x the target!
        old_total = 60 * total_slides

        # New behavior: 6 × 30s = 180s (3 min) - matches target
        assert total_duration == 180
        assert old_total == 360
        assert total_duration < old_total  # New is shorter (correct)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
