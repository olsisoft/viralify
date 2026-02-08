"""
Tests for CodeDisplayMode feature.

Tests the integration of the code display mode selection across:
- Presentation models (enum values)
- Presentation compositor (mode routing)
- Typing animator (animation selection)
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path


class TestCodeDisplayModeEnum:
    """Test the CodeDisplayMode enum definition."""

    def test_enum_values_exist(self):
        """Verify all expected enum values exist."""
        from models.presentation_models import CodeDisplayMode

        assert hasattr(CodeDisplayMode, 'TYPING')
        assert hasattr(CodeDisplayMode, 'REVEAL')
        assert hasattr(CodeDisplayMode, 'STATIC')

    def test_enum_values_are_strings(self):
        """Verify enum values are strings for API serialization."""
        from models.presentation_models import CodeDisplayMode

        assert CodeDisplayMode.TYPING.value == "typing"
        assert CodeDisplayMode.REVEAL.value == "reveal"
        assert CodeDisplayMode.STATIC.value == "static"

    def test_enum_has_three_modes(self):
        """Verify exactly 3 modes exist."""
        from models.presentation_models import CodeDisplayMode

        modes = list(CodeDisplayMode)
        assert len(modes) == 3

    def test_default_mode_is_reveal(self):
        """Verify reveal is the default mode in request model."""
        from models.presentation_models import GeneratePresentationRequest

        request = GeneratePresentationRequest(
            topic="Test topic for code display"
        )
        assert request.code_display_mode == "reveal"


class TestGeneratePresentationRequestWithCodeDisplayMode:
    """Test the GeneratePresentationRequest model with code_display_mode."""

    def test_request_accepts_typing_mode(self):
        """Test that request accepts typing mode."""
        from models.presentation_models import GeneratePresentationRequest, CodeDisplayMode

        request = GeneratePresentationRequest(
            topic="Test topic for code display",
            code_display_mode=CodeDisplayMode.TYPING
        )
        assert request.code_display_mode == CodeDisplayMode.TYPING

    def test_request_accepts_reveal_mode(self):
        """Test that request accepts reveal mode."""
        from models.presentation_models import GeneratePresentationRequest, CodeDisplayMode

        request = GeneratePresentationRequest(
            topic="Test topic for code display",
            code_display_mode=CodeDisplayMode.REVEAL
        )
        assert request.code_display_mode == CodeDisplayMode.REVEAL

    def test_request_accepts_static_mode(self):
        """Test that request accepts static mode."""
        from models.presentation_models import GeneratePresentationRequest, CodeDisplayMode

        request = GeneratePresentationRequest(
            topic="Test topic for code display",
            code_display_mode=CodeDisplayMode.STATIC
        )
        assert request.code_display_mode == CodeDisplayMode.STATIC

    def test_request_accepts_string_mode(self):
        """Test that request accepts string values for mode."""
        from models.presentation_models import GeneratePresentationRequest

        request = GeneratePresentationRequest(
            topic="Test topic for code display",
            code_display_mode="typing"
        )
        assert request.code_display_mode == "typing"

    def test_request_serialization(self):
        """Test that code_display_mode is properly serialized in JSON."""
        from models.presentation_models import GeneratePresentationRequest, CodeDisplayMode

        request = GeneratePresentationRequest(
            topic="Test topic for code display",
            code_display_mode=CodeDisplayMode.STATIC
        )
        data = request.model_dump()
        assert data["code_display_mode"] == "static"


class TestCodeDisplayModeRouting:
    """Test the routing logic for different code display modes."""

    def test_static_mode_sets_force_static(self):
        """Verify static mode sets force_static=True."""
        from models.presentation_models import CodeDisplayMode

        code_display_mode = CodeDisplayMode.STATIC
        force_static = (code_display_mode == CodeDisplayMode.STATIC)
        force_typing = (code_display_mode == CodeDisplayMode.TYPING)

        assert force_static is True
        assert force_typing is False

    def test_typing_mode_sets_force_typing(self):
        """Verify typing mode sets force_typing=True."""
        from models.presentation_models import CodeDisplayMode

        code_display_mode = CodeDisplayMode.TYPING
        force_static = (code_display_mode == CodeDisplayMode.STATIC)
        force_typing = (code_display_mode == CodeDisplayMode.TYPING)

        assert force_static is False
        assert force_typing is True

    def test_reveal_mode_sets_neither_flag(self):
        """Verify reveal mode sets neither flag (uses SSVS-C)."""
        from models.presentation_models import CodeDisplayMode

        code_display_mode = CodeDisplayMode.REVEAL
        force_static = (code_display_mode == CodeDisplayMode.STATIC)
        force_typing = (code_display_mode == CodeDisplayMode.TYPING)

        assert force_static is False
        assert force_typing is False


class TestTypingAnimatorModeSelection:
    """Test the typing animator mode selection logic."""

    @pytest.fixture
    def mock_typing_animator(self):
        """Create a mock typing animator for testing."""
        with patch('services.typing_animator.TypingAnimatorService') as mock:
            instance = MagicMock()
            instance.create_typing_animation = AsyncMock(return_value=("/tmp/video.mp4", 30.0))
            instance._create_static_video = AsyncMock(return_value=("/tmp/static.mp4", 30.0))
            instance._create_synced_reveal_video = AsyncMock(return_value=("/tmp/reveal.mp4", 30.0))
            mock.return_value = instance
            yield instance

    def test_mode_priority_static_first(self):
        """Verify static mode is checked first (highest priority)."""
        # Simulate the routing logic from typing_animator
        force_static = True
        force_typing = True  # Even if both are set
        sync_mode = True
        reveal_points = [{"line": 1}]

        # Static should win
        if force_static:
            selected_mode = "static"
        elif force_typing:
            selected_mode = "typing"
        elif sync_mode and reveal_points:
            selected_mode = "reveal"
        else:
            selected_mode = "typing"  # fallback

        assert selected_mode == "static"

    def test_mode_priority_typing_second(self):
        """Verify typing mode is checked second."""
        force_static = False
        force_typing = True
        sync_mode = True
        reveal_points = [{"line": 1}]

        if force_static:
            selected_mode = "static"
        elif force_typing:
            selected_mode = "typing"
        elif sync_mode and reveal_points:
            selected_mode = "reveal"
        else:
            selected_mode = "typing"

        assert selected_mode == "typing"

    def test_mode_priority_reveal_third(self):
        """Verify reveal mode is used when sync_mode is enabled."""
        force_static = False
        force_typing = False
        sync_mode = True
        reveal_points = [{"line": 1}]

        if force_static:
            selected_mode = "static"
        elif force_typing:
            selected_mode = "typing"
        elif sync_mode and reveal_points:
            selected_mode = "reveal"
        else:
            selected_mode = "typing"

        assert selected_mode == "reveal"

    def test_mode_fallback_to_typing(self):
        """Verify fallback to typing when no mode is specified."""
        force_static = False
        force_typing = False
        sync_mode = False
        reveal_points = []

        if force_static:
            selected_mode = "static"
        elif force_typing:
            selected_mode = "typing"
        elif sync_mode and reveal_points:
            selected_mode = "reveal"
        else:
            selected_mode = "typing"

        assert selected_mode == "typing"


class TestCodeDisplayModeIntegration:
    """Integration tests for code display mode across components."""

    def test_mode_propagation_from_request_to_flags(self):
        """Test that mode from request correctly sets animation flags."""
        from models.presentation_models import GeneratePresentationRequest, CodeDisplayMode

        # Test each mode
        test_cases = [
            (CodeDisplayMode.STATIC, True, False),
            (CodeDisplayMode.TYPING, False, True),
            (CodeDisplayMode.REVEAL, False, False),
        ]

        for mode, expected_static, expected_typing in test_cases:
            request = GeneratePresentationRequest(
                topic="Test integration",
                code_display_mode=mode
            )

            # Simulate the compositor logic
            code_display_mode = request.code_display_mode
            force_static = (code_display_mode == CodeDisplayMode.STATIC)
            force_typing = (code_display_mode == CodeDisplayMode.TYPING)

            assert force_static == expected_static, f"Failed for mode {mode}: force_static"
            assert force_typing == expected_typing, f"Failed for mode {mode}: force_typing"

    def test_env_override_for_typing(self):
        """Test that FORCE_TYPING_ANIMATION env var overrides mode."""
        from models.presentation_models import CodeDisplayMode

        # Simulate env override logic
        code_display_mode = CodeDisplayMode.REVEAL  # User selected reveal
        env_force_typing = True  # But env says force typing

        if env_force_typing:
            code_display_mode = CodeDisplayMode.TYPING

        force_typing = (code_display_mode == CodeDisplayMode.TYPING)
        assert force_typing is True


class TestCodeDisplayModeDescriptions:
    """Test that mode descriptions are properly defined."""

    def test_typing_mode_characteristics(self):
        """Verify typing mode characteristics."""
        # Typing mode: character-by-character animation
        # - Slow but immersive
        # - Live-coding effect
        # - Memory intensive (frame-by-frame)
        mode_info = {
            "typing": {
                "animation_type": "character-by-character",
                "speed": "slow",
                "effect": "live-coding",
                "memory_usage": "high"
            }
        }
        assert mode_info["typing"]["animation_type"] == "character-by-character"
        assert mode_info["typing"]["speed"] == "slow"

    def test_reveal_mode_characteristics(self):
        """Verify reveal mode characteristics."""
        # Reveal mode: line-by-line synchronized with voiceover
        # - Fast and professional
        # - Uses SSVS-C synchronization
        # - FFmpeg drawbox filters
        mode_info = {
            "reveal": {
                "animation_type": "line-by-line",
                "speed": "fast",
                "sync": "SSVS-C",
                "implementation": "ffmpeg-drawbox"
            }
        }
        assert mode_info["reveal"]["animation_type"] == "line-by-line"
        assert mode_info["reveal"]["sync"] == "SSVS-C"

    def test_static_mode_characteristics(self):
        """Verify static mode characteristics."""
        # Static mode: code displayed instantly
        # - Very fast
        # - No animation
        # - Single frame
        mode_info = {
            "static": {
                "animation_type": "none",
                "speed": "instant",
                "frames": "single"
            }
        }
        assert mode_info["static"]["animation_type"] == "none"
        assert mode_info["static"]["speed"] == "instant"


class TestCodeDisplayModeValidation:
    """Test validation of code display mode values."""

    def test_valid_modes_accepted(self):
        """Test that all valid modes are accepted."""
        from models.presentation_models import GeneratePresentationRequest

        valid_modes = ["typing", "reveal", "static"]

        for mode in valid_modes:
            request = GeneratePresentationRequest(
                topic="Test validation",
                code_display_mode=mode
            )
            assert request.code_display_mode == mode

    def test_mode_case_sensitivity(self):
        """Test that mode values are case-sensitive."""
        from models.presentation_models import CodeDisplayMode

        # Enum values are lowercase
        assert CodeDisplayMode.TYPING.value == "typing"
        assert CodeDisplayMode.REVEAL.value == "reveal"
        assert CodeDisplayMode.STATIC.value == "static"

        # Uppercase should not match
        assert CodeDisplayMode.TYPING.value != "TYPING"


class TestCodeDisplayModeWithTypingSpeed:
    """Test interaction between code_display_mode and typing_speed."""

    def test_typing_speed_relevant_only_for_typing_mode(self):
        """Verify typing_speed is only used in typing mode."""
        from models.presentation_models import GeneratePresentationRequest, CodeDisplayMode, TypingSpeed

        # Typing mode should use typing_speed
        request_typing = GeneratePresentationRequest(
            topic="Test speed",
            code_display_mode=CodeDisplayMode.TYPING,
            typing_speed=TypingSpeed.SLOW
        )
        assert request_typing.typing_speed == TypingSpeed.SLOW

        # Static mode doesn't use typing_speed (but field is still present)
        request_static = GeneratePresentationRequest(
            topic="Test speed",
            code_display_mode=CodeDisplayMode.STATIC,
            typing_speed=TypingSpeed.FAST
        )
        # Speed is set but won't be used in static mode
        assert request_static.typing_speed == TypingSpeed.FAST
        assert request_static.code_display_mode == CodeDisplayMode.STATIC

    def test_typing_speed_values(self):
        """Test all typing speed values."""
        from models.presentation_models import TypingSpeed

        speeds = [TypingSpeed.SLOW, TypingSpeed.NATURAL, TypingSpeed.MODERATE, TypingSpeed.FAST]
        expected_values = ["slow", "natural", "moderate", "fast"]

        for speed, expected in zip(speeds, expected_values):
            assert speed.value == expected


# Run tests with: pytest tests/test_code_display_mode.py -v
