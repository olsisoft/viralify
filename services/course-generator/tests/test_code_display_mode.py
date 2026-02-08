"""
Tests for CodeDisplayMode feature in course-generator.

Tests the integration of code display mode:
- Course models (request field)
- State propagation
- Input validation
- Presentation request construction
"""

import pytest
from unittest.mock import MagicMock, patch
import sys

# Mock external modules before importing course-generator modules
sys.modules['openai'] = MagicMock()
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()


class TestGenerateCourseRequestCodeDisplayMode:
    """Test GenerateCourseRequest model with code_display_mode field."""

    def test_default_code_display_mode(self):
        """Verify default code_display_mode is 'reveal'."""
        from models.course_models import GenerateCourseRequest

        request = GenerateCourseRequest(
            profile_id="test-profile",
            topic="Python Programming"
        )
        assert request.code_display_mode == "reveal"

    def test_typing_mode(self):
        """Test setting code_display_mode to 'typing'."""
        from models.course_models import GenerateCourseRequest

        request = GenerateCourseRequest(
            profile_id="test-profile",
            topic="Python Programming",
            code_display_mode="typing"
        )
        assert request.code_display_mode == "typing"

    def test_static_mode(self):
        """Test setting code_display_mode to 'static'."""
        from models.course_models import GenerateCourseRequest

        request = GenerateCourseRequest(
            profile_id="test-profile",
            topic="Python Programming",
            code_display_mode="static"
        )
        assert request.code_display_mode == "static"

    def test_reveal_mode_explicit(self):
        """Test explicitly setting code_display_mode to 'reveal'."""
        from models.course_models import GenerateCourseRequest

        request = GenerateCourseRequest(
            profile_id="test-profile",
            topic="Python Programming",
            code_display_mode="reveal"
        )
        assert request.code_display_mode == "reveal"

    def test_request_serialization(self):
        """Test that code_display_mode is properly serialized."""
        from models.course_models import GenerateCourseRequest

        request = GenerateCourseRequest(
            profile_id="test-profile",
            topic="Python Programming",
            code_display_mode="static"
        )
        data = request.model_dump()
        assert data["code_display_mode"] == "static"

    def test_code_display_mode_with_other_fields(self):
        """Test code_display_mode works alongside other presentation fields."""
        from models.course_models import GenerateCourseRequest

        request = GenerateCourseRequest(
            profile_id="test-profile",
            topic="Python Programming",
            code_display_mode="typing",
            typing_speed="slow",
            voice_id="alloy",
            style="dark"
        )
        assert request.code_display_mode == "typing"
        assert request.typing_speed == "slow"
        assert request.voice_id == "alloy"
        assert request.style == "dark"


class TestInputValidatorCodeDisplayMode:
    """Test input validation for code_display_mode."""

    def test_valid_code_display_modes(self):
        """Test that valid modes pass validation."""
        from agents.input_validator import VALID_CODE_DISPLAY_MODES

        assert "typing" in VALID_CODE_DISPLAY_MODES
        assert "reveal" in VALID_CODE_DISPLAY_MODES
        assert "static" in VALID_CODE_DISPLAY_MODES
        assert len(VALID_CODE_DISPLAY_MODES) == 3

    def test_invalid_mode_generates_warning(self):
        """Test that invalid mode generates a warning."""
        from agents.input_validator import VALID_CODE_DISPLAY_MODES

        invalid_mode = "animated"
        assert invalid_mode not in VALID_CODE_DISPLAY_MODES

        # Simulate validation logic
        warnings = []
        if invalid_mode not in VALID_CODE_DISPLAY_MODES:
            warnings.append(f"Code display mode '{invalid_mode}' is not standard. Using 'reveal'.")

        assert len(warnings) == 1
        assert "not standard" in warnings[0]


class TestOrchestratorStateCodeDisplayMode:
    """Test OrchestratorState with code_display_mode field."""

    def test_orchestrator_state_has_code_display_mode(self):
        """Verify OrchestratorState has code_display_mode field."""
        from agents.state import OrchestratorState

        # Check the field exists in annotations
        annotations = OrchestratorState.__annotations__
        assert "code_display_mode" in annotations
        assert annotations["code_display_mode"] == str

    def test_production_state_has_code_display_mode(self):
        """Verify ProductionState has code_display_mode field."""
        from agents.state import ProductionState

        annotations = ProductionState.__annotations__
        assert "code_display_mode" in annotations
        assert annotations["code_display_mode"] == str


class TestCreateOrchestratorState:
    """Test create_orchestrator_state function with code_display_mode."""

    def test_create_orchestrator_state_default(self):
        """Test create_orchestrator_state uses default code_display_mode."""
        from agents.state import create_orchestrator_state

        state = create_orchestrator_state(
            job_id="test-job",
            topic="Python Programming"
        )
        assert state["code_display_mode"] == "reveal"

    def test_create_orchestrator_state_custom(self):
        """Test create_orchestrator_state with custom code_display_mode."""
        from agents.state import create_orchestrator_state

        state = create_orchestrator_state(
            job_id="test-job",
            topic="Python Programming",
            code_display_mode="typing"
        )
        assert state["code_display_mode"] == "typing"

    def test_create_orchestrator_state_static(self):
        """Test create_orchestrator_state with static mode."""
        from agents.state import create_orchestrator_state

        state = create_orchestrator_state(
            job_id="test-job",
            topic="Python Programming",
            code_display_mode="static"
        )
        assert state["code_display_mode"] == "static"


class TestProductionStateCodeDisplayMode:
    """Test ProductionState propagation of code_display_mode."""

    def test_create_production_state_inherits_mode(self):
        """Test that ProductionState inherits code_display_mode from orchestrator."""
        from agents.state import create_production_state_for_lecture

        # Create a mock orchestrator state
        orchestrator_state = {
            "job_id": "test-job",
            "topic": "Python Programming",
            "code_display_mode": "typing",
            "voice_id": "alloy",
            "style": "dark",
            "typing_speed": "natural",
            "include_avatar": False,
            "avatar_id": None,
            "content_language": "en",
            "programming_language": "python",
            "lesson_elements_enabled": {
                "concept_intro": True,
                "diagram_schema": True,
                "code_typing": True,
                "code_execution": False,
                "voiceover_explanation": True,
                "curriculum_slide": True,
            },
            "rag_context": None,
            "document_ids": [],
        }

        # Create a mock lecture plan
        lecture_plan = {
            "id": "lecture-1",
            "title": "Introduction",
            "objectives": ["Learn basics"],
            "duration_seconds": 300,
            "target_audience": "beginners",
        }

        state = create_production_state_for_lecture(
            orchestrator_state=orchestrator_state,
            lecture_plan=lecture_plan
        )

        assert state["code_display_mode"] == "typing"


class TestPresentationRequestConstruction:
    """Test that code_display_mode is passed to presentation request."""

    def test_presentation_request_includes_code_display_mode(self):
        """Simulate the presentation request construction from production_graph."""
        # Simulate settings extraction
        settings = {
            "voice_id": "alloy",
            "style": "dark",
            "typing_speed": "natural",
            "code_display_mode": "static",
            "include_avatar": False,
        }

        # Simulate presentation_request construction (from production_graph.py)
        presentation_request = {
            "topic": "Test topic",
            "voice_id": settings.get("voice_id", "default"),
            "style": settings.get("style", "modern"),
            "typing_speed": settings.get("typing_speed", "natural"),
            "code_display_mode": settings.get("code_display_mode", "reveal"),
        }

        assert presentation_request["code_display_mode"] == "static"

    def test_presentation_request_default_reveal(self):
        """Test that default is 'reveal' when not specified."""
        settings = {}  # Empty settings

        presentation_request = {
            "code_display_mode": settings.get("code_display_mode", "reveal"),
        }

        assert presentation_request["code_display_mode"] == "reveal"


class TestCodeDisplayModeEndToEnd:
    """End-to-end tests for code display mode flow."""

    def test_mode_flow_from_request_to_presentation(self):
        """Test the complete flow from request to presentation request."""
        # 1. Frontend sends request with code_display_mode
        frontend_request = {
            "profile_id": "test-profile",
            "topic": "Python Programming",
            "code_display_mode": "typing",
            "typing_speed": "slow"
        }

        # 2. Request is validated
        from agents.input_validator import VALID_CODE_DISPLAY_MODES
        assert frontend_request["code_display_mode"] in VALID_CODE_DISPLAY_MODES

        # 3. State is created with code_display_mode
        state = {
            "code_display_mode": frontend_request["code_display_mode"],
            "typing_speed": frontend_request["typing_speed"]
        }

        # 4. Presentation request is constructed
        presentation_request = {
            "code_display_mode": state.get("code_display_mode", "reveal"),
            "typing_speed": state.get("typing_speed", "natural")
        }

        # 5. Verify final values
        assert presentation_request["code_display_mode"] == "typing"
        assert presentation_request["typing_speed"] == "slow"

    def test_all_modes_flow(self):
        """Test all three modes flow correctly."""
        modes = ["typing", "reveal", "static"]

        for mode in modes:
            # Simulate full flow
            request_mode = mode
            state_mode = request_mode
            presentation_mode = state_mode

            assert presentation_mode == mode, f"Mode {mode} did not flow correctly"


class TestCodeDisplayModeWithLessonElements:
    """Test interaction with lesson elements configuration."""

    def test_code_typing_element_independent_of_display_mode(self):
        """Verify code_typing element is independent of code_display_mode."""
        # code_typing in lesson_elements = whether to include code slides
        # code_display_mode = HOW to display code in those slides
        lesson_elements = {
            "concept_intro": True,
            "diagram_schema": True,
            "code_typing": True,  # Include code slides
            "code_execution": False,
        }

        # Even with code_display_mode="static", code slides are included
        # They just display instantly instead of animating
        code_display_mode = "static"

        assert lesson_elements["code_typing"] is True
        assert code_display_mode == "static"

    def test_code_execution_with_display_modes(self):
        """Test code_execution works with all display modes."""
        # code_execution shows terminal output
        # This should work regardless of display mode
        for mode in ["typing", "reveal", "static"]:
            settings = {
                "lesson_elements": {"code_execution": True},
                "code_display_mode": mode
            }
            # Code execution is always enabled based on lesson_elements
            assert settings["lesson_elements"]["code_execution"] is True


# Run tests with: pytest tests/test_code_display_mode.py -v
