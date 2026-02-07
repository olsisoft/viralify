"""
Unit tests for presentation_models.py

Tests all enums, Pydantic models, and their methods.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import json

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.presentation_models import (
    SlideType,
    PresentationStyle,
    PresentationStage,
    ScriptSegmentType,
    ScriptSegment,
    CodeBlock,
    Slide,
    PresentationScript,
    TitleStyle,
    TypingSpeed,
    GeneratePresentationRequest,
    PresentationJob,
    SlidePreviewRequest,
    SlidePreviewResponse,
    LanguageInfo,
    StyleInfo,
)


# ============================================================================
# SlideType Enum Tests
# ============================================================================

class TestSlideType:
    """Tests for SlideType enum"""

    def test_all_slide_types_defined(self):
        """Verify all expected slide types exist"""
        expected_types = ["title", "content", "code", "code_demo", "diagram",
                         "split", "terminal", "conclusion"]
        for slide_type in expected_types:
            assert SlideType(slide_type) is not None

    def test_slide_type_values(self):
        """Test specific enum values"""
        assert SlideType.TITLE.value == "title"
        assert SlideType.CONTENT.value == "content"
        assert SlideType.CODE.value == "code"
        assert SlideType.CODE_DEMO.value == "code_demo"
        assert SlideType.DIAGRAM.value == "diagram"
        assert SlideType.SPLIT.value == "split"
        assert SlideType.TERMINAL.value == "terminal"
        assert SlideType.CONCLUSION.value == "conclusion"

    def test_slide_type_is_string_enum(self):
        """Verify SlideType is a string enum"""
        assert isinstance(SlideType.TITLE, str)
        assert SlideType.TITLE == "title"

    def test_invalid_slide_type_raises_error(self):
        """Test that invalid slide type raises ValueError"""
        with pytest.raises(ValueError):
            SlideType("invalid_type")


# ============================================================================
# PresentationStyle Enum Tests
# ============================================================================

class TestPresentationStyle:
    """Tests for PresentationStyle enum"""

    def test_all_styles_defined(self):
        """Verify all expected styles exist"""
        expected_styles = ["dark", "light", "gradient", "ocean"]
        for style in expected_styles:
            assert PresentationStyle(style) is not None

    def test_style_values(self):
        """Test specific enum values"""
        assert PresentationStyle.DARK.value == "dark"
        assert PresentationStyle.LIGHT.value == "light"
        assert PresentationStyle.GRADIENT.value == "gradient"
        assert PresentationStyle.OCEAN.value == "ocean"

    def test_style_is_string_enum(self):
        """Verify PresentationStyle is a string enum"""
        assert isinstance(PresentationStyle.DARK, str)
        assert PresentationStyle.DARK == "dark"


# ============================================================================
# PresentationStage Enum Tests
# ============================================================================

class TestPresentationStage:
    """Tests for PresentationStage enum"""

    def test_all_stages_defined(self):
        """Verify all expected stages exist"""
        expected_stages = [
            "queued", "planning", "generating_slides", "executing_code",
            "creating_animations", "generating_voiceover", "generating_avatar",
            "composing_video", "completed", "failed"
        ]
        for stage in expected_stages:
            assert PresentationStage(stage) is not None

    def test_stage_values(self):
        """Test specific enum values"""
        assert PresentationStage.QUEUED.value == "queued"
        assert PresentationStage.PLANNING.value == "planning"
        assert PresentationStage.COMPLETED.value == "completed"
        assert PresentationStage.FAILED.value == "failed"

    def test_stage_count(self):
        """Verify total number of stages"""
        assert len(PresentationStage) == 10


# ============================================================================
# ScriptSegmentType Enum Tests
# ============================================================================

class TestScriptSegmentType:
    """Tests for ScriptSegmentType enum"""

    def test_all_segment_types_defined(self):
        """Verify all expected segment types exist"""
        expected_types = ["intro", "explanation", "example", "summary", "transition"]
        for seg_type in expected_types:
            assert ScriptSegmentType(seg_type) is not None

    def test_segment_type_values(self):
        """Test specific enum values"""
        assert ScriptSegmentType.INTRO.value == "intro"
        assert ScriptSegmentType.EXPLANATION.value == "explanation"
        assert ScriptSegmentType.EXAMPLE.value == "example"
        assert ScriptSegmentType.SUMMARY.value == "summary"
        assert ScriptSegmentType.TRANSITION.value == "transition"


# ============================================================================
# TitleStyle Enum Tests
# ============================================================================

class TestTitleStyle:
    """Tests for TitleStyle enum"""

    def test_all_title_styles_defined(self):
        """Verify all expected title styles exist"""
        expected_styles = ["corporate", "engaging", "expert", "mentor",
                          "storyteller", "direct"]
        for style in expected_styles:
            assert TitleStyle(style) is not None

    def test_title_style_values(self):
        """Test specific enum values"""
        assert TitleStyle.CORPORATE.value == "corporate"
        assert TitleStyle.ENGAGING.value == "engaging"
        assert TitleStyle.EXPERT.value == "expert"
        assert TitleStyle.MENTOR.value == "mentor"
        assert TitleStyle.STORYTELLER.value == "storyteller"
        assert TitleStyle.DIRECT.value == "direct"


# ============================================================================
# TypingSpeed Enum Tests
# ============================================================================

class TestTypingSpeed:
    """Tests for TypingSpeed enum"""

    def test_all_typing_speeds_defined(self):
        """Verify all expected typing speeds exist"""
        expected_speeds = ["slow", "natural", "moderate", "fast"]
        for speed in expected_speeds:
            assert TypingSpeed(speed) is not None

    def test_typing_speed_values(self):
        """Test specific enum values"""
        assert TypingSpeed.SLOW.value == "slow"
        assert TypingSpeed.NATURAL.value == "natural"
        assert TypingSpeed.MODERATE.value == "moderate"
        assert TypingSpeed.FAST.value == "fast"


# ============================================================================
# ScriptSegment Model Tests
# ============================================================================

class TestScriptSegment:
    """Tests for ScriptSegment model"""

    def test_create_basic_segment(self):
        """Test creating a basic script segment"""
        segment = ScriptSegment(
            type=ScriptSegmentType.EXPLANATION,
            content="This is the explanation content",
            duration_seconds=30
        )
        assert segment.type == ScriptSegmentType.EXPLANATION
        assert segment.content == "This is the explanation content"
        assert segment.duration_seconds == 30
        assert segment.key_points == []

    def test_create_segment_with_key_points(self):
        """Test creating segment with key points"""
        segment = ScriptSegment(
            type=ScriptSegmentType.INTRO,
            content="Welcome to the tutorial",
            duration_seconds=15,
            key_points=["Point 1", "Point 2", "Point 3"]
        )
        assert len(segment.key_points) == 3
        assert "Point 1" in segment.key_points

    def test_segment_json_serialization(self):
        """Test that segment can be serialized to JSON"""
        segment = ScriptSegment(
            type=ScriptSegmentType.SUMMARY,
            content="In summary...",
            duration_seconds=10,
            key_points=["Key takeaway"]
        )
        json_data = segment.model_dump()
        assert json_data["type"] == "summary"
        assert json_data["content"] == "In summary..."

    def test_segment_type_validation(self):
        """Test that invalid type raises validation error"""
        with pytest.raises(ValueError):
            ScriptSegment(
                type="invalid_type",
                content="Content",
                duration_seconds=10
            )


# ============================================================================
# CodeBlock Model Tests
# ============================================================================

class TestCodeBlock:
    """Tests for CodeBlock model"""

    def test_create_basic_code_block(self):
        """Test creating a basic code block"""
        code = CodeBlock(
            language="python",
            code="print('Hello, World!')"
        )
        assert code.language == "python"
        assert code.code == "print('Hello, World!')"
        assert code.filename is None
        assert code.highlight_lines == []
        assert code.execution_order == 0
        assert code.expected_output is None
        assert code.show_line_numbers is True

    def test_create_full_code_block(self):
        """Test creating code block with all fields"""
        code = CodeBlock(
            language="python",
            code="def greet(name):\n    return f'Hello, {name}!'",
            filename="greet.py",
            highlight_lines=[1, 2],
            execution_order=1,
            expected_output="Hello, World!",
            show_line_numbers=False
        )
        assert code.filename == "greet.py"
        assert code.highlight_lines == [1, 2]
        assert code.execution_order == 1
        assert code.expected_output == "Hello, World!"
        assert code.show_line_numbers is False

    def test_code_block_json_serialization(self):
        """Test JSON serialization"""
        code = CodeBlock(
            language="javascript",
            code="console.log('test');"
        )
        json_data = code.model_dump()
        assert json_data["language"] == "javascript"
        assert json_data["code"] == "console.log('test');"


# ============================================================================
# Slide Model Tests
# ============================================================================

class TestSlide:
    """Tests for Slide model"""

    def test_create_basic_slide(self):
        """Test creating a basic slide"""
        slide = Slide(type=SlideType.CONTENT)
        assert slide.type == SlideType.CONTENT
        assert slide.id is not None
        assert len(slide.id) == 8
        assert slide.title is None
        assert slide.content is None
        assert slide.bullet_points == []
        assert slide.code_blocks == []
        assert slide.duration == 10.0
        assert slide.voiceover_text == ""
        assert slide.script_segments == []
        assert slide.transition == "fade"

    def test_create_title_slide(self):
        """Test creating a title slide"""
        slide = Slide(
            type=SlideType.TITLE,
            title="Introduction to Python",
            subtitle="A Beginner's Guide"
        )
        assert slide.type == SlideType.TITLE
        assert slide.title == "Introduction to Python"
        assert slide.subtitle == "A Beginner's Guide"

    def test_create_code_slide(self):
        """Test creating a code slide"""
        code_block = CodeBlock(language="python", code="print('test')")
        slide = Slide(
            type=SlideType.CODE,
            title="Example Code",
            code_blocks=[code_block]
        )
        assert slide.type == SlideType.CODE
        assert len(slide.code_blocks) == 1
        assert slide.code_blocks[0].language == "python"

    def test_create_diagram_slide(self):
        """Test creating a diagram slide"""
        slide = Slide(
            type=SlideType.DIAGRAM,
            title="Architecture Diagram",
            diagram_type="architecture"
        )
        assert slide.type == SlideType.DIAGRAM
        assert slide.diagram_type == "architecture"

    def test_slide_has_segments_property_false(self):
        """Test has_segments property when no segments"""
        slide = Slide(type=SlideType.CONTENT)
        assert slide.has_segments is False

    def test_slide_has_segments_property_true(self):
        """Test has_segments property with segments"""
        segment = ScriptSegment(
            type=ScriptSegmentType.EXPLANATION,
            content="Explanation",
            duration_seconds=20
        )
        slide = Slide(
            type=SlideType.CONTENT,
            script_segments=[segment]
        )
        assert slide.has_segments is True

    def test_slide_full_voiceover_without_segments(self):
        """Test full_voiceover returns voiceover_text when no segments"""
        slide = Slide(
            type=SlideType.CONTENT,
            voiceover_text="This is the voiceover text"
        )
        assert slide.full_voiceover == "This is the voiceover text"

    def test_slide_full_voiceover_with_segments(self):
        """Test full_voiceover concatenates segment content"""
        segments = [
            ScriptSegment(type=ScriptSegmentType.INTRO, content="Intro.", duration_seconds=5),
            ScriptSegment(type=ScriptSegmentType.EXPLANATION, content="Main part.", duration_seconds=20),
            ScriptSegment(type=ScriptSegmentType.SUMMARY, content="Summary.", duration_seconds=5),
        ]
        slide = Slide(
            type=SlideType.CONTENT,
            script_segments=segments,
            voiceover_text="This should be ignored"
        )
        assert slide.full_voiceover == "Intro. Main part. Summary."

    def test_slide_bloom_level(self):
        """Test slide with Bloom's taxonomy level"""
        slide = Slide(
            type=SlideType.CONTENT,
            bloom_level="apply",
            key_takeaways=["Key point 1", "Key point 2"]
        )
        assert slide.bloom_level == "apply"
        assert len(slide.key_takeaways) == 2

    def test_slide_json_serialization(self):
        """Test slide JSON serialization"""
        slide = Slide(
            type=SlideType.CONTENT,
            title="Test Slide",
            content="Some content"
        )
        json_data = slide.model_dump()
        assert json_data["type"] == "content"
        assert json_data["title"] == "Test Slide"


# ============================================================================
# PresentationScript Model Tests
# ============================================================================

class TestPresentationScript:
    """Tests for PresentationScript model"""

    def test_create_basic_script(self):
        """Test creating a basic presentation script"""
        slides = [
            Slide(type=SlideType.TITLE, title="Intro"),
            Slide(type=SlideType.CONTENT, content="Content"),
            Slide(type=SlideType.CONCLUSION, title="Conclusion"),
        ]
        script = PresentationScript(
            title="Test Presentation",
            description="A test presentation",
            language="python",
            total_duration=300,
            slides=slides
        )
        assert script.title == "Test Presentation"
        assert script.description == "A test presentation"
        assert script.language == "python"
        assert script.total_duration == 300
        assert len(script.slides) == 3

    def test_script_slide_count_property(self):
        """Test slide_count property"""
        slides = [Slide(type=SlideType.CONTENT) for _ in range(5)]
        script = PresentationScript(
            title="Test",
            description="Test",
            language="python",
            total_duration=300,
            slides=slides
        )
        assert script.slide_count == 5

    def test_script_code_slide_count_property(self):
        """Test code_slide_count property"""
        slides = [
            Slide(type=SlideType.TITLE),
            Slide(type=SlideType.CODE),
            Slide(type=SlideType.CODE_DEMO),
            Slide(type=SlideType.CONTENT),
            Slide(type=SlideType.CODE),
            Slide(type=SlideType.DIAGRAM),
        ]
        script = PresentationScript(
            title="Test",
            description="Test",
            language="python",
            total_duration=300,
            slides=slides
        )
        assert script.code_slide_count == 3  # CODE, CODE_DEMO, CODE

    def test_script_with_target_career(self):
        """Test script with target career"""
        script = PresentationScript(
            title="Test",
            description="Test",
            language="python",
            total_duration=300,
            slides=[Slide(type=SlideType.CONTENT)],
            target_career="data_engineer"
        )
        assert script.target_career == "data_engineer"

    def test_script_with_rag_verification(self):
        """Test script with RAG verification data"""
        rag_data = {
            "coverage": 0.92,
            "is_compliant": True,
            "summary": "RAG compliant"
        }
        script = PresentationScript(
            title="Test",
            description="Test",
            language="python",
            total_duration=300,
            slides=[Slide(type=SlideType.CONTENT)],
            rag_verification=rag_data
        )
        assert script.rag_verification["coverage"] == 0.92
        assert script.rag_verification["is_compliant"] is True

    def test_script_default_values(self):
        """Test script default values"""
        script = PresentationScript(
            title="Test",
            description="Test",
            language="python",
            total_duration=300,
            slides=[]
        )
        assert script.target_audience == "developers"
        assert script.target_career is None
        assert script.code_context == {}
        assert script.rag_verification is None


# ============================================================================
# GeneratePresentationRequest Model Tests
# ============================================================================

class TestGeneratePresentationRequest:
    """Tests for GeneratePresentationRequest model"""

    def test_create_basic_request(self):
        """Test creating a basic generation request"""
        request = GeneratePresentationRequest(
            topic="Introduction to Python decorators with examples"
        )
        assert request.topic == "Introduction to Python decorators with examples"
        assert request.language == "python"
        assert request.content_language == "en"
        assert request.duration == 300
        assert request.style == PresentationStyle.DARK
        assert request.include_avatar is False
        assert request.voice_id == "alloy"
        assert request.execute_code is False
        assert request.show_typing_animation is True
        assert request.typing_speed == TypingSpeed.NATURAL
        assert request.title_style == TitleStyle.ENGAGING

    def test_create_full_request(self):
        """Test creating request with all fields"""
        request = GeneratePresentationRequest(
            topic="Advanced Python Patterns for Data Engineering",
            language="python",
            content_language="fr",
            duration=600,
            style=PresentationStyle.OCEAN,
            include_avatar=True,
            avatar_id="avatar-123",
            voice_id="nova",
            execute_code=True,
            show_typing_animation=True,
            typing_speed=TypingSpeed.MODERATE,
            target_audience="senior developers",
            target_career="data_engineer",
            title_style=TitleStyle.EXPERT,
            document_ids=["doc-1", "doc-2"],
            practical_focus="practical",
            enable_visuals=True,
            visual_style="dark"
        )
        assert request.content_language == "fr"
        assert request.duration == 600
        assert request.style == PresentationStyle.OCEAN
        assert request.include_avatar is True
        assert request.avatar_id == "avatar-123"
        assert request.target_career == "data_engineer"
        assert len(request.document_ids) == 2

    def test_request_duration_validation_min(self):
        """Test that duration must be at least 60 seconds"""
        with pytest.raises(ValueError):
            GeneratePresentationRequest(
                topic="Short topic for testing purposes",
                duration=30
            )

    def test_request_duration_validation_max(self):
        """Test that duration must be at most 900 seconds"""
        with pytest.raises(ValueError):
            GeneratePresentationRequest(
                topic="Long topic for testing purposes",
                duration=1200
            )

    def test_request_topic_min_length(self):
        """Test that topic must be at least 10 characters"""
        with pytest.raises(ValueError):
            GeneratePresentationRequest(topic="Short")

    def test_request_json_serialization(self):
        """Test request JSON serialization"""
        request = GeneratePresentationRequest(
            topic="Python decorators tutorial for beginners"
        )
        json_data = request.model_dump()
        assert json_data["topic"] == "Python decorators tutorial for beginners"
        assert json_data["style"] == "dark"


# ============================================================================
# PresentationJob Model Tests
# ============================================================================

class TestPresentationJob:
    """Tests for PresentationJob model"""

    def test_create_default_job(self):
        """Test creating a job with defaults"""
        job = PresentationJob()
        assert job.job_id is not None
        assert len(job.job_id) == 36  # UUID format
        assert job.status == "queued"
        assert job.current_stage == PresentationStage.QUEUED
        assert job.progress == 0.0
        assert job.message == ""
        assert job.script is None
        assert job.slide_images == []
        assert job.error is None

    def test_job_update_progress_processing(self):
        """Test updating job progress during processing"""
        job = PresentationJob()
        job.update_progress(
            stage=PresentationStage.GENERATING_SLIDES,
            progress=25.0,
            message="Generating slide 1 of 4"
        )
        assert job.current_stage == PresentationStage.GENERATING_SLIDES
        assert job.progress == 25.0
        assert job.message == "Generating slide 1 of 4"
        assert job.status == "processing"
        assert job.completed_at is None

    def test_job_update_progress_completed(self):
        """Test updating job to completed status"""
        job = PresentationJob()
        job.update_progress(
            stage=PresentationStage.COMPLETED,
            progress=100.0,
            message="Generation complete"
        )
        assert job.status == "completed"
        assert job.progress == 100.0
        assert job.completed_at is not None

    def test_job_update_progress_failed(self):
        """Test updating job to failed status"""
        job = PresentationJob()
        job.update_progress(
            stage=PresentationStage.FAILED,
            progress=50.0,
            message="Error occurred"
        )
        assert job.status == "failed"
        assert job.current_stage == PresentationStage.FAILED

    def test_job_with_request(self):
        """Test job with attached request"""
        request = GeneratePresentationRequest(
            topic="Test topic for presentation generation"
        )
        job = PresentationJob(request=request)
        assert job.request is not None
        assert job.request.topic == "Test topic for presentation generation"

    def test_job_with_script(self):
        """Test job with attached script"""
        script = PresentationScript(
            title="Test",
            description="Test",
            language="python",
            total_duration=300,
            slides=[]
        )
        job = PresentationJob(script=script)
        assert job.script is not None
        assert job.script.title == "Test"

    def test_job_timestamps_updated(self):
        """Test that timestamps are updated correctly"""
        job = PresentationJob()
        original_updated_at = job.updated_at

        # Small delay to ensure time difference
        import time
        time.sleep(0.01)

        job.update_progress(
            stage=PresentationStage.PLANNING,
            progress=10.0
        )
        assert job.updated_at > original_updated_at

    def test_job_error_handling(self):
        """Test job error fields"""
        job = PresentationJob()
        job.error = "API rate limit exceeded"
        job.error_details = {"code": 429, "retry_after": 60}
        assert job.error == "API rate limit exceeded"
        assert job.error_details["code"] == 429


# ============================================================================
# SlidePreviewRequest/Response Tests
# ============================================================================

class TestSlidePreview:
    """Tests for SlidePreviewRequest and SlidePreviewResponse"""

    def test_create_preview_request(self):
        """Test creating a preview request"""
        slide = Slide(type=SlideType.CONTENT, title="Test Slide")
        request = SlidePreviewRequest(
            slide=slide,
            style=PresentationStyle.LIGHT
        )
        assert request.slide.title == "Test Slide"
        assert request.style == PresentationStyle.LIGHT

    def test_preview_request_default_style(self):
        """Test preview request default style"""
        slide = Slide(type=SlideType.CONTENT)
        request = SlidePreviewRequest(slide=slide)
        assert request.style == PresentationStyle.DARK

    def test_create_preview_response(self):
        """Test creating a preview response"""
        response = SlidePreviewResponse(
            image_url="http://example.com/slide.png"
        )
        assert response.image_url == "http://example.com/slide.png"
        assert response.width == 1920
        assert response.height == 1080

    def test_preview_response_custom_dimensions(self):
        """Test preview response with custom dimensions"""
        response = SlidePreviewResponse(
            image_url="http://example.com/slide.png",
            width=1280,
            height=720
        )
        assert response.width == 1280
        assert response.height == 720


# ============================================================================
# LanguageInfo and StyleInfo Tests
# ============================================================================

class TestInfoModels:
    """Tests for LanguageInfo and StyleInfo models"""

    def test_create_language_info(self):
        """Test creating LanguageInfo"""
        info = LanguageInfo(
            id="python",
            name="Python",
            file_extension=".py",
            supported=True,
            icon="🐍"
        )
        assert info.id == "python"
        assert info.name == "Python"
        assert info.file_extension == ".py"
        assert info.supported is True
        assert info.icon == "🐍"

    def test_create_style_info(self):
        """Test creating StyleInfo"""
        info = StyleInfo(
            id="dark",
            name="Dark Theme",
            preview_colors={
                "background": "#1a1a2e",
                "text": "#ffffff",
                "accent": "#00ff88"
            }
        )
        assert info.id == "dark"
        assert info.name == "Dark Theme"
        assert info.preview_colors["background"] == "#1a1a2e"


# ============================================================================
# Integration Tests
# ============================================================================

class TestModelIntegration:
    """Integration tests for model interactions"""

    def test_full_presentation_creation(self):
        """Test creating a complete presentation with all components"""
        # Create code blocks
        code_block = CodeBlock(
            language="python",
            code="def hello():\n    return 'Hello!'",
            filename="hello.py",
            highlight_lines=[2]
        )

        # Create script segments
        segments = [
            ScriptSegment(
                type=ScriptSegmentType.INTRO,
                content="Welcome to our tutorial.",
                duration_seconds=10,
                key_points=["Introduction"]
            ),
            ScriptSegment(
                type=ScriptSegmentType.EXPLANATION,
                content="Let's look at how functions work.",
                duration_seconds=30,
                key_points=["Function definition", "Return statement"]
            )
        ]

        # Create slides
        slides = [
            Slide(
                type=SlideType.TITLE,
                title="Python Functions",
                subtitle="A Quick Guide"
            ),
            Slide(
                type=SlideType.CODE,
                title="Hello Function",
                code_blocks=[code_block],
                script_segments=segments,
                bloom_level="understand"
            ),
            Slide(
                type=SlideType.CONCLUSION,
                title="Summary",
                bullet_points=["Functions are reusable", "Use return for output"]
            )
        ]

        # Create script
        script = PresentationScript(
            title="Python Functions Tutorial",
            description="Learn how to write Python functions",
            target_audience="beginner developers",
            target_career="software_developer",
            language="python",
            total_duration=120,
            slides=slides
        )

        # Create job
        request = GeneratePresentationRequest(
            topic="Python Functions Tutorial for Beginners",
            duration=120,
            target_audience="beginner developers"
        )

        job = PresentationJob(
            request=request,
            script=script
        )

        # Assertions
        assert job.script.slide_count == 3
        assert job.script.code_slide_count == 1
        assert job.script.slides[1].has_segments is True
        assert "Welcome to our tutorial. Let's look at how functions work." == job.script.slides[1].full_voiceover
        assert job.request.topic == "Python Functions Tutorial for Beginners"

    def test_job_lifecycle(self):
        """Test complete job lifecycle from creation to completion"""
        job = PresentationJob()

        # Initial state
        assert job.status == "queued"
        assert job.progress == 0.0

        # Planning
        job.update_progress(PresentationStage.PLANNING, 10.0, "Generating script...")
        assert job.status == "processing"

        # Generating slides
        job.update_progress(PresentationStage.GENERATING_SLIDES, 30.0, "Creating slides...")
        assert job.progress == 30.0

        # Generating voiceover
        job.update_progress(PresentationStage.GENERATING_VOICEOVER, 60.0, "Recording audio...")
        assert job.current_stage == PresentationStage.GENERATING_VOICEOVER

        # Composing video
        job.update_progress(PresentationStage.COMPOSING_VIDEO, 90.0, "Finalizing...")
        assert job.progress == 90.0

        # Completed
        job.update_progress(PresentationStage.COMPLETED, 100.0, "Done!")
        assert job.status == "completed"
        assert job.completed_at is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
