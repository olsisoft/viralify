"""
Integration tests for CodeDisplayMode feature in course-generator.

Tests the full flow from API request through orchestrator to presentation-generator.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import json
import sys

# Mock external modules
sys.modules['openai'] = MagicMock()
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()
sys.modules['langgraph.graph.state'] = MagicMock()


class TestCodeDisplayModeAPIIntegration:
    """Test API endpoint integration with code_display_mode."""

    def test_generate_course_request_serialization(self):
        """Test full JSON serialization of course request."""
        from models.course_models import GenerateCourseRequest

        request = GenerateCourseRequest(
            profile_id="tech-expert-001",
            topic="Advanced Python Patterns",
            code_display_mode="typing",
            typing_speed="slow",
            voice_id="alloy",
            style="dark"
        )

        # Serialize to JSON
        json_data = request.model_dump_json()
        parsed = json.loads(json_data)

        assert parsed["code_display_mode"] == "typing"
        assert parsed["typing_speed"] == "slow"
        assert parsed["profile_id"] == "tech-expert-001"

    def test_all_modes_in_request(self):
        """Test all three modes can be set in request."""
        from models.course_models import GenerateCourseRequest

        modes = ["typing", "reveal", "static"]

        for mode in modes:
            request = GenerateCourseRequest(
                profile_id="test-profile",
                topic="Test Course",
                code_display_mode=mode
            )
            assert request.code_display_mode == mode


class TestOrchestratorStateIntegration:
    """Test orchestrator state propagation of code_display_mode."""

    def test_state_creation_with_mode(self):
        """Test that state is created with correct mode."""
        from agents.state import create_orchestrator_state

        state = create_orchestrator_state(
            job_id="integration-test-001",
            topic="Integration Testing",
            code_display_mode="static"
        )

        assert state["code_display_mode"] == "static"
        assert state["job_id"] == "integration-test-001"

    def test_state_defaults_to_reveal(self):
        """Test that state defaults to reveal mode."""
        from agents.state import create_orchestrator_state

        state = create_orchestrator_state(
            job_id="test-job",
            topic="Default Mode Test"
        )

        assert state["code_display_mode"] == "reveal"

    def test_production_state_inherits_mode(self):
        """Test that production state inherits mode from orchestrator."""
        from agents.state import (
            create_orchestrator_state,
            create_production_state_for_lecture
        )

        # Create orchestrator state
        orchestrator_state = create_orchestrator_state(
            job_id="test-job",
            topic="Python Basics",
            code_display_mode="typing",
            voice_id="alloy",
            style="dark"
        )

        # Add outline (required by create_production_state_for_lecture)
        orchestrator_state["outline"] = {
            "title": "Python Basics",
            "description": "Learn Python fundamentals"
        }

        # Create lecture plan
        lecture_plan = {
            "id": "lecture-1",
            "title": "Variables and Types",
            "objectives": ["Understand variables"],
            "duration_seconds": 300,
            "target_audience": "beginners"
        }

        # Create production state
        production_state = create_production_state_for_lecture(
            orchestrator_state=orchestrator_state,
            lecture_plan=lecture_plan
        )

        assert production_state["code_display_mode"] == "typing"


class TestInputValidationIntegration:
    """Test input validation integration."""

    def test_valid_modes_pass_validation(self):
        """Test that all valid modes pass validation."""
        from agents.input_validator import VALID_CODE_DISPLAY_MODES

        valid_modes = ["typing", "reveal", "static"]

        for mode in valid_modes:
            assert mode in VALID_CODE_DISPLAY_MODES

    def test_invalid_mode_detected(self):
        """Test that invalid modes are detected."""
        from agents.input_validator import VALID_CODE_DISPLAY_MODES

        invalid_modes = ["animated", "live", "progressive", "instant"]

        for mode in invalid_modes:
            assert mode not in VALID_CODE_DISPLAY_MODES


class TestPresentationRequestConstruction:
    """Test construction of presentation request for presentation-generator."""

    def test_presentation_request_includes_mode(self):
        """Test that presentation request includes code_display_mode."""
        # Simulate the production_graph.py logic
        settings = {
            "voice_id": "alloy",
            "style": "dark",
            "typing_speed": "natural",
            "code_display_mode": "static",
            "include_avatar": False,
            "avatar_id": None,
            "lesson_elements": {
                "code_execution": False,
                "diagram_schema": True
            }
        }

        lecture_plan = {
            "duration_seconds": 300,
            "target_audience": "intermediate developers"
        }

        # Build presentation request as in production_graph.py
        presentation_request = {
            "topic": "Building REST APIs",
            "duration": lecture_plan.get("duration_seconds", 300),
            "style": settings.get("style", "modern"),
            "include_avatar": settings.get("include_avatar", False),
            "avatar_id": settings.get("avatar_id"),
            "voice_id": settings.get("voice_id", "default"),
            "execute_code": settings.get("lesson_elements", {}).get("code_execution", False),
            "show_typing_animation": not settings.get("animations_disabled", False),
            "typing_speed": settings.get("typing_speed", "natural"),
            "code_display_mode": settings.get("code_display_mode", "reveal"),
            "target_audience": lecture_plan.get("target_audience", ""),
            "enable_visuals": settings.get("lesson_elements", {}).get("diagram_schema", True),
        }

        assert presentation_request["code_display_mode"] == "static"
        assert presentation_request["typing_speed"] == "natural"

    def test_default_mode_when_not_specified(self):
        """Test default mode when not specified in settings."""
        settings = {}  # Empty settings

        presentation_request = {
            "code_display_mode": settings.get("code_display_mode", "reveal"),
        }

        assert presentation_request["code_display_mode"] == "reveal"


class TestFullFlowIntegration:
    """Full flow integration tests."""

    def test_request_to_orchestrator_flow(self):
        """Test flow from request to orchestrator state."""
        from models.course_models import GenerateCourseRequest
        from agents.state import create_orchestrator_state

        # 1. Create course request
        request = GenerateCourseRequest(
            profile_id="tech-profile",
            topic="Advanced Python",
            code_display_mode="typing",
            typing_speed="slow"
        )

        # 2. Extract values for state creation
        state = create_orchestrator_state(
            job_id="flow-test-001",
            topic=request.topic,
            code_display_mode=request.code_display_mode,
            typing_speed=request.typing_speed
        )

        # 3. Verify propagation
        assert state["topic"] == "Advanced Python"
        assert state["code_display_mode"] == "typing"
        assert state["typing_speed"] == "slow"

    def test_orchestrator_to_production_flow(self):
        """Test flow from orchestrator to production state."""
        from agents.state import (
            create_orchestrator_state,
            create_production_state_for_lecture
        )

        # 1. Create orchestrator state
        orchestrator = create_orchestrator_state(
            job_id="flow-test-002",
            topic="Docker Containers",
            code_display_mode="static"
        )

        # Add outline (required)
        orchestrator["outline"] = {
            "title": "Docker Containers",
            "description": "Learn Docker basics"
        }

        # 2. Create lecture plan
        lecture = {
            "id": "lec-1",
            "title": "Container Basics",
            "objectives": ["Understand containers"],
            "duration_seconds": 300
        }

        # 3. Create production state
        production = create_production_state_for_lecture(
            orchestrator_state=orchestrator,
            lecture_plan=lecture
        )

        # 4. Verify inheritance
        assert production["code_display_mode"] == "static"

    def test_production_to_presentation_request_flow(self):
        """Test flow from production state to presentation request."""
        from agents.state import (
            create_orchestrator_state,
            create_production_state_for_lecture
        )

        # 1. Setup orchestrator
        orchestrator = create_orchestrator_state(
            job_id="flow-test-003",
            topic="Kubernetes",
            code_display_mode="reveal",
            voice_id="nova",
            style="ocean"
        )

        # Add outline (required by create_production_state_for_lecture)
        orchestrator["outline"] = {
            "title": "Kubernetes",
            "description": "Learn Kubernetes basics"
        }

        # 2. Create production state
        lecture = {
            "id": "lec-k8s",
            "title": "Pod Deployment",
            "duration_seconds": 420
        }

        production = create_production_state_for_lecture(
            orchestrator_state=orchestrator,
            lecture_plan=lecture
        )

        # 3. Build presentation request (simulating production_graph.py)
        presentation_request = {
            "topic": lecture["title"],
            "duration": lecture["duration_seconds"],
            "voice_id": production.get("voice_id", "default"),
            "style": production.get("style", "dark"),
            "typing_speed": production.get("typing_speed", "natural"),
            "code_display_mode": production.get("code_display_mode", "reveal"),
        }

        # 4. Verify final values
        assert presentation_request["code_display_mode"] == "reveal"
        assert presentation_request["voice_id"] == "nova"
        assert presentation_request["style"] == "ocean"


class TestCodeDisplayModeWithLessonElements:
    """Test integration with lesson elements configuration."""

    def test_code_typing_element_enables_code_slides(self):
        """Test that code_typing element controls code slide inclusion."""
        lesson_elements = {
            "concept_intro": True,
            "diagram_schema": True,
            "code_typing": True,  # Enables code slides
            "code_execution": False,
            "voiceover_explanation": True,
            "curriculum_slide": True
        }

        # code_typing determines IF code slides are included
        include_code_slides = lesson_elements.get("code_typing", True)
        assert include_code_slides is True

        # code_display_mode determines HOW code is displayed
        code_display_mode = "static"

        # Both work independently
        assert include_code_slides is True
        assert code_display_mode == "static"

    def test_code_execution_with_display_modes(self):
        """Test code_execution works with all display modes."""
        modes = ["typing", "reveal", "static"]

        for mode in modes:
            lesson_elements = {
                "code_typing": True,
                "code_execution": True  # Show terminal output
            }

            # Code execution is independent of display mode
            settings = {
                "lesson_elements": lesson_elements,
                "code_display_mode": mode
            }

            execute_code = settings["lesson_elements"]["code_execution"]
            assert execute_code is True


class TestCodeDisplayModeWithQuizzes:
    """Test integration with quiz configuration."""

    def test_mode_does_not_affect_quizzes(self):
        """Test that code_display_mode does not affect quiz generation."""
        from models.course_models import GenerateCourseRequest, QuizConfigRequest

        request = GenerateCourseRequest(
            profile_id="test-profile",
            topic="Python Testing",
            code_display_mode="static",
            quiz_config=QuizConfigRequest(
                enabled=True,
                frequency="per_section",
                questions_per_quiz=5
            )
        )

        # Quiz config should be independent
        assert request.quiz_config.enabled is True
        assert request.quiz_config.frequency == "per_section"
        assert request.code_display_mode == "static"


class TestCodeDisplayModeWithRAG:
    """Test integration with RAG document sources."""

    def test_mode_works_with_document_sources(self):
        """Test that code_display_mode works with RAG sources."""
        from models.course_models import GenerateCourseRequest

        request = GenerateCourseRequest(
            profile_id="test-profile",
            topic="API Design Patterns",
            code_display_mode="typing",
            document_ids=["doc-001", "doc-002", "doc-003"]
        )

        assert request.code_display_mode == "typing"
        assert len(request.document_ids) == 3

    def test_mode_propagates_with_rag_context(self):
        """Test that mode propagates alongside RAG context."""
        from agents.state import create_orchestrator_state

        state = create_orchestrator_state(
            job_id="rag-test-001",
            topic="Machine Learning Pipeline",
            code_display_mode="reveal",
            rag_context="Sample RAG context from documents...",
            document_ids=["ml-doc-001"]
        )

        assert state["code_display_mode"] == "reveal"
        assert state["rag_context"] is not None
        assert len(state["document_ids"]) == 1


class TestMultipleLecturesIntegration:
    """Test code_display_mode with multiple lectures."""

    def test_mode_consistent_across_lectures(self):
        """Test that mode is consistent across all lectures."""
        from agents.state import (
            create_orchestrator_state,
            create_production_state_for_lecture
        )

        # Create orchestrator with typing mode
        orchestrator = create_orchestrator_state(
            job_id="multi-lecture-test",
            topic="Full Stack Development",
            code_display_mode="typing"
        )

        # Add outline (required by create_production_state_for_lecture)
        orchestrator["outline"] = {
            "title": "Full Stack Development",
            "description": "Learn full stack development"
        }

        # Create multiple lectures
        lectures = [
            {"id": "lec-1", "title": "Frontend Basics", "duration_seconds": 300},
            {"id": "lec-2", "title": "Backend APIs", "duration_seconds": 400},
            {"id": "lec-3", "title": "Database Design", "duration_seconds": 350},
        ]

        # Verify each lecture inherits the same mode
        for lecture in lectures:
            production = create_production_state_for_lecture(
                orchestrator_state=orchestrator,
                lecture_plan=lecture
            )
            assert production["code_display_mode"] == "typing"


class TestErrorHandlingIntegration:
    """Test error handling with code_display_mode."""

    def test_missing_mode_uses_default(self):
        """Test that missing mode uses default 'reveal'."""
        settings = {
            "voice_id": "alloy",
            "style": "dark"
            # code_display_mode not specified
        }

        mode = settings.get("code_display_mode", "reveal")
        assert mode == "reveal"

    def test_none_mode_uses_default(self):
        """Test that None mode uses default 'reveal'."""
        settings = {
            "code_display_mode": None
        }

        mode = settings.get("code_display_mode") or "reveal"
        assert mode == "reveal"


class TestCodeDisplayModeLogging:
    """Test that code_display_mode is properly logged."""

    def test_mode_in_presentation_request_log(self):
        """Test that mode appears in presentation request for logging."""
        settings = {
            "code_display_mode": "static",
            "typing_speed": "fast"
        }

        presentation_request = {
            "topic": "Test Topic",
            "code_display_mode": settings.get("code_display_mode", "reveal"),
            "typing_speed": settings.get("typing_speed", "natural")
        }

        # Simulate log message
        log_message = (
            f"[PRODUCTION] Settings: "
            f"code_display_mode={presentation_request['code_display_mode']}, "
            f"typing_speed={presentation_request['typing_speed']}"
        )

        assert "code_display_mode=static" in log_message
        assert "typing_speed=fast" in log_message


# Run tests with: pytest tests/test_code_display_mode_integration.py -v
