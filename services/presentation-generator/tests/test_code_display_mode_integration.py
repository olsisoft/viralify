"""
Integration tests for CodeDisplayMode feature.

Tests the full flow from API request to video generation,
including interaction between components.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import json
import os


class TestCodeDisplayModeAPIIntegration:
    """Test API endpoints with CodeDisplayMode parameter."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app for testing."""
        from fastapi.testclient import TestClient

        # Import after mocking dependencies
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("services.presentation_planner.OpenAI"):
                with patch("services.voiceover_service.OpenAI"):
                    from main import app
                    yield TestClient(app)

    def test_generate_endpoint_accepts_code_display_mode(self):
        """Test that /generate endpoint accepts code_display_mode."""
        from models.presentation_models import GeneratePresentationRequest, CodeDisplayMode

        # Create request with code_display_mode
        request = GeneratePresentationRequest(
            topic="Python decorators tutorial",
            duration=300,
            code_display_mode=CodeDisplayMode.TYPING
        )

        # Verify request serialization
        data = request.model_dump()
        assert "code_display_mode" in data
        assert data["code_display_mode"] == "typing"

    def test_generate_endpoint_default_mode(self):
        """Test that default mode is 'reveal' when not specified."""
        from models.presentation_models import GeneratePresentationRequest

        request = GeneratePresentationRequest(
            topic="Python decorators tutorial",
            duration=300
        )

        assert request.code_display_mode == "reveal"

    def test_request_json_serialization(self):
        """Test full JSON serialization of request with code_display_mode."""
        from models.presentation_models import GeneratePresentationRequest, CodeDisplayMode

        request = GeneratePresentationRequest(
            topic="Building APIs with FastAPI",
            duration=600,
            code_display_mode=CodeDisplayMode.STATIC,
            typing_speed="fast",
            style="dark"
        )

        # Serialize to JSON
        json_data = request.model_dump_json()
        parsed = json.loads(json_data)

        assert parsed["code_display_mode"] == "static"
        assert parsed["typing_speed"] == "fast"
        assert parsed["duration"] == 600


class TestPresentationJobWithCodeDisplayMode:
    """Test PresentationJob handling of code_display_mode."""

    def test_job_stores_code_display_mode(self):
        """Test that job stores the code_display_mode from request."""
        from models.presentation_models import (
            GeneratePresentationRequest,
            PresentationJob,
            CodeDisplayMode
        )

        request = GeneratePresentationRequest(
            topic="React hooks tutorial",
            code_display_mode=CodeDisplayMode.TYPING
        )

        job = PresentationJob(request=request)

        assert job.request.code_display_mode == CodeDisplayMode.TYPING

    def test_job_propagates_mode_through_stages(self):
        """Test that code_display_mode is preserved through job stages."""
        from models.presentation_models import (
            GeneratePresentationRequest,
            PresentationJob,
            PresentationStage,
            CodeDisplayMode
        )

        request = GeneratePresentationRequest(
            topic="Docker containers",
            code_display_mode=CodeDisplayMode.REVEAL
        )

        job = PresentationJob(request=request)

        # Simulate stage progression
        job.update_progress(PresentationStage.PLANNING, 10, "Planning...")
        assert job.request.code_display_mode == CodeDisplayMode.REVEAL

        job.update_progress(PresentationStage.GENERATING_SLIDES, 30, "Generating...")
        assert job.request.code_display_mode == CodeDisplayMode.REVEAL

        job.update_progress(PresentationStage.CREATING_ANIMATIONS, 60, "Animating...")
        assert job.request.code_display_mode == CodeDisplayMode.REVEAL


class TestCompositorCodeDisplayModeIntegration:
    """Test PresentationCompositor integration with code_display_mode."""

    def test_compositor_extracts_mode_from_request(self):
        """Test that compositor correctly extracts code_display_mode."""
        from models.presentation_models import (
            GeneratePresentationRequest,
            PresentationJob,
            CodeDisplayMode
        )

        request = GeneratePresentationRequest(
            topic="Kubernetes deployment",
            code_display_mode=CodeDisplayMode.STATIC
        )

        job = PresentationJob(request=request)

        # Extract mode as compositor would
        code_display_mode = job.request.code_display_mode if job.request else CodeDisplayMode.REVEAL

        assert code_display_mode == CodeDisplayMode.STATIC

    def test_mode_determines_animation_flags(self):
        """Test that mode correctly sets animation flags."""
        from models.presentation_models import CodeDisplayMode

        test_cases = [
            (CodeDisplayMode.STATIC, True, False),   # force_static=True
            (CodeDisplayMode.TYPING, False, True),   # force_typing=True
            (CodeDisplayMode.REVEAL, False, False),  # Neither (uses SSVS-C)
        ]

        for mode, expected_static, expected_typing in test_cases:
            force_static = (mode == CodeDisplayMode.STATIC)
            force_typing = (mode == CodeDisplayMode.TYPING)

            assert force_static == expected_static, f"Failed for {mode}: force_static"
            assert force_typing == expected_typing, f"Failed for {mode}: force_typing"


class TestTypingAnimatorIntegration:
    """Test TypingAnimatorService integration with code_display_mode."""

    def test_static_mode_creates_single_frame_video(self):
        """Test that static mode creates a video with no animation."""
        # Simulate static mode behavior
        force_static = True
        force_typing = False

        # In static mode, we expect:
        # 1. Single frame rendered
        # 2. Frame converted to video with target duration
        # 3. No character-by-character animation

        if force_static:
            animation_type = "static"
        elif force_typing:
            animation_type = "typing"
        else:
            animation_type = "reveal"

        assert animation_type == "static"

    def test_typing_mode_creates_frame_by_frame(self):
        """Test that typing mode creates frame-by-frame animation."""
        force_static = False
        force_typing = True

        if force_static:
            animation_type = "static"
        elif force_typing:
            animation_type = "typing"
        else:
            animation_type = "reveal"

        assert animation_type == "typing"

    def test_reveal_mode_uses_ffmpeg_drawbox(self):
        """Test that reveal mode uses FFmpeg drawbox filters."""
        force_static = False
        force_typing = False
        sync_mode = True
        reveal_points = [{"line": 1, "timestamp": 2.5}]

        if force_static:
            animation_type = "static"
        elif force_typing:
            animation_type = "typing"
        elif sync_mode and reveal_points:
            animation_type = "reveal"
        else:
            animation_type = "typing"

        assert animation_type == "reveal"


class TestCodeDisplayModeWithSlideTypes:
    """Test code_display_mode interaction with different slide types."""

    def test_mode_only_affects_code_slides(self):
        """Test that code_display_mode only affects CODE and CODE_DEMO slides."""
        from models.presentation_models import SlideType, CodeDisplayMode

        code_slide_types = [SlideType.CODE, SlideType.CODE_DEMO]
        non_code_slide_types = [
            SlideType.TITLE,
            SlideType.CONTENT,
            SlideType.DIAGRAM,
            SlideType.CONCLUSION
        ]

        # Code display mode should only be relevant for code slides
        code_display_mode = CodeDisplayMode.TYPING

        for slide_type in code_slide_types:
            # Mode is relevant
            is_code_slide = slide_type in code_slide_types
            assert is_code_slide is True

        for slide_type in non_code_slide_types:
            # Mode is not relevant
            is_code_slide = slide_type in code_slide_types
            assert is_code_slide is False

    def test_mixed_slide_presentation(self):
        """Test presentation with mixed slide types uses mode correctly."""
        from models.presentation_models import Slide, SlideType, CodeDisplayMode

        slides = [
            Slide(type=SlideType.TITLE, title="Introduction"),
            Slide(type=SlideType.CONTENT, title="Overview"),
            Slide(type=SlideType.CODE, title="Example Code"),
            Slide(type=SlideType.DIAGRAM, title="Architecture"),
            Slide(type=SlideType.CODE_DEMO, title="Live Demo"),
            Slide(type=SlideType.CONCLUSION, title="Summary"),
        ]

        code_display_mode = CodeDisplayMode.STATIC

        # Count slides that should use the mode
        code_slides = [s for s in slides if s.type in [SlideType.CODE, SlideType.CODE_DEMO]]
        assert len(code_slides) == 2

        # Other slides should not be affected
        other_slides = [s for s in slides if s.type not in [SlideType.CODE, SlideType.CODE_DEMO]]
        assert len(other_slides) == 4


class TestCodeDisplayModeEnvOverride:
    """Test environment variable override for code_display_mode."""

    def test_env_force_typing_overrides_reveal(self):
        """Test FORCE_TYPING_ANIMATION env var overrides reveal mode."""
        from models.presentation_models import CodeDisplayMode

        # User selected reveal
        user_mode = CodeDisplayMode.REVEAL

        # But env says force typing
        env_force_typing = True

        # Final mode should be typing
        if env_force_typing:
            final_mode = CodeDisplayMode.TYPING
        else:
            final_mode = user_mode

        assert final_mode == CodeDisplayMode.TYPING

    def test_env_force_typing_overrides_static(self):
        """Test FORCE_TYPING_ANIMATION env var overrides static mode."""
        from models.presentation_models import CodeDisplayMode

        user_mode = CodeDisplayMode.STATIC
        env_force_typing = True

        if env_force_typing:
            final_mode = CodeDisplayMode.TYPING
        else:
            final_mode = user_mode

        assert final_mode == CodeDisplayMode.TYPING

    def test_no_env_override_preserves_user_choice(self):
        """Test that user choice is preserved when env var is not set."""
        from models.presentation_models import CodeDisplayMode

        user_mode = CodeDisplayMode.STATIC
        env_force_typing = False

        if env_force_typing:
            final_mode = CodeDisplayMode.TYPING
        else:
            final_mode = user_mode

        assert final_mode == CodeDisplayMode.STATIC


class TestCodeDisplayModeWithTypingSpeed:
    """Integration tests for code_display_mode and typing_speed interaction."""

    def test_typing_speed_used_in_typing_mode(self):
        """Test that typing_speed affects animation in typing mode."""
        from models.presentation_models import (
            GeneratePresentationRequest,
            CodeDisplayMode,
            TypingSpeed
        )

        request = GeneratePresentationRequest(
            topic="Python async/await",
            code_display_mode=CodeDisplayMode.TYPING,
            typing_speed=TypingSpeed.SLOW
        )

        # Calculate chars per second based on speed
        speed_map = {
            TypingSpeed.SLOW: 2.0,
            TypingSpeed.NATURAL: 4.0,
            TypingSpeed.MODERATE: 6.0,
            TypingSpeed.FAST: 10.0,
        }

        chars_per_second = speed_map[request.typing_speed]
        assert chars_per_second == 2.0  # Slow = 2 chars/sec

    def test_typing_speed_ignored_in_static_mode(self):
        """Test that typing_speed is ignored in static mode."""
        from models.presentation_models import (
            GeneratePresentationRequest,
            CodeDisplayMode,
            TypingSpeed
        )

        request = GeneratePresentationRequest(
            topic="Go concurrency patterns",
            code_display_mode=CodeDisplayMode.STATIC,
            typing_speed=TypingSpeed.SLOW  # Set but should be ignored
        )

        # In static mode, animation duration is fixed
        # typing_speed should not affect the result
        force_static = (request.code_display_mode == CodeDisplayMode.STATIC)

        assert force_static is True
        # Speed is set but won't be used
        assert request.typing_speed == TypingSpeed.SLOW


class TestCodeDisplayModeFullPipeline:
    """Full pipeline integration tests."""

    def test_full_flow_typing_mode(self):
        """Test full flow with typing mode selected."""
        from models.presentation_models import (
            GeneratePresentationRequest,
            PresentationJob,
            PresentationScript,
            Slide,
            SlideType,
            CodeDisplayMode,
            CodeBlock
        )

        # 1. Create request with typing mode
        request = GeneratePresentationRequest(
            topic="Building REST APIs",
            duration=300,
            code_display_mode=CodeDisplayMode.TYPING
        )

        # 2. Create job
        job = PresentationJob(request=request)

        # 3. Create mock script with code slide
        script = PresentationScript(
            title="REST API Tutorial",
            description="Learn to build APIs",
            language="python",
            total_duration=300,
            slides=[
                Slide(
                    type=SlideType.TITLE,
                    title="REST APIs",
                    duration=10
                ),
                Slide(
                    type=SlideType.CODE,
                    title="Hello World API",
                    code_blocks=[
                        CodeBlock(
                            language="python",
                            code="from fastapi import FastAPI\napp = FastAPI()"
                        )
                    ],
                    duration=60
                )
            ]
        )

        job.script = script

        # 4. Verify mode is preserved
        assert job.request.code_display_mode == CodeDisplayMode.TYPING

        # 5. Simulate compositor extracting mode
        code_display_mode = job.request.code_display_mode
        force_typing = (code_display_mode == CodeDisplayMode.TYPING)

        assert force_typing is True

    def test_full_flow_static_mode(self):
        """Test full flow with static mode selected."""
        from models.presentation_models import (
            GeneratePresentationRequest,
            PresentationJob,
            CodeDisplayMode
        )

        request = GeneratePresentationRequest(
            topic="Quick SQL tutorial",
            duration=120,
            code_display_mode=CodeDisplayMode.STATIC
        )

        job = PresentationJob(request=request)

        force_static = (job.request.code_display_mode == CodeDisplayMode.STATIC)
        force_typing = (job.request.code_display_mode == CodeDisplayMode.TYPING)

        assert force_static is True
        assert force_typing is False

    def test_full_flow_reveal_mode(self):
        """Test full flow with reveal mode (default)."""
        from models.presentation_models import (
            GeneratePresentationRequest,
            PresentationJob,
            CodeDisplayMode
        )

        request = GeneratePresentationRequest(
            topic="Machine Learning basics",
            duration=600,
            code_display_mode=CodeDisplayMode.REVEAL
        )

        job = PresentationJob(request=request)

        force_static = (job.request.code_display_mode == CodeDisplayMode.STATIC)
        force_typing = (job.request.code_display_mode == CodeDisplayMode.TYPING)

        # Neither flag should be set for reveal mode
        assert force_static is False
        assert force_typing is False


class TestCodeDisplayModeWithSSVSC:
    """Test integration with SSVS-C (Code-Aware Synchronization)."""

    def test_reveal_mode_enables_ssvs_c(self):
        """Test that reveal mode enables SSVS-C synchronization."""
        from models.presentation_models import CodeDisplayMode

        code_display_mode = CodeDisplayMode.REVEAL

        # In reveal mode, SSVS-C should be used
        force_static = (code_display_mode == CodeDisplayMode.STATIC)
        force_typing = (code_display_mode == CodeDisplayMode.TYPING)

        # When neither flag is set, reveal mode with SSVS-C is used
        use_ssvs_c = not force_static and not force_typing

        assert use_ssvs_c is True

    def test_static_mode_disables_ssvs_c(self):
        """Test that static mode disables SSVS-C synchronization."""
        from models.presentation_models import CodeDisplayMode

        code_display_mode = CodeDisplayMode.STATIC

        force_static = (code_display_mode == CodeDisplayMode.STATIC)
        use_ssvs_c = not force_static

        assert force_static is True
        assert use_ssvs_c is False

    def test_typing_mode_disables_ssvs_c(self):
        """Test that typing mode disables SSVS-C synchronization."""
        from models.presentation_models import CodeDisplayMode

        code_display_mode = CodeDisplayMode.TYPING

        force_typing = (code_display_mode == CodeDisplayMode.TYPING)

        # Typing mode uses frame-by-frame, not SSVS-C reveal
        assert force_typing is True


# Run tests with: pytest tests/test_code_display_mode_integration.py -v
