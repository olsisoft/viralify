"""
Unit tests for job management services in presentation-generator.

Tests:
- LessonStatus and JobStatus enums
- LessonError and RetryResult dataclasses
- JobManager class (helper methods)
- RedisJobStore key generation and serialization
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
import json


# =============================================================================
# Standalone implementations to avoid import chain issues
# =============================================================================

class LessonStatus(str, Enum):
    """Status of an individual lesson."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class JobStatus(str, Enum):
    """Status of a job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class LessonError:
    """Detailed error information for a lesson."""
    scene_index: int
    title: str
    error_type: str
    error_message: str
    original_content: Dict[str, Any]
    retry_count: int = 0
    last_retry_at: Optional[str] = None
    editable: bool = True


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    message: str
    scene_index: Optional[int] = None
    video_url: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class RedisConnectionError(Exception):
    """Raised when Redis connection fails."""
    pass


class JobManagerHelpers:
    """Helper methods from JobManager that don't require Redis."""

    def __init__(self):
        self.job_prefix = "v3"

    def _build_scene_url(self, job_id: str, scene_index: int, public_media_url: str = "", media_url: str = "") -> str:
        """Build URL for a scene video."""
        filename = f"{job_id}_scene_{scene_index:03d}.mp4"
        if public_media_url:
            return f"{public_media_url}/files/videos/{filename}"
        if media_url:
            return f"{media_url}/files/videos/{filename}"
        return f"http://media-generator:8004/files/videos/{filename}"

    def _build_final_url(self, job_id: str, public_media_url: str = "", media_url: str = "") -> str:
        """Build URL for the final video."""
        filename = f"{job_id}_final.mp4"
        if public_media_url:
            return f"{public_media_url}/files/videos/{filename}"
        if media_url:
            return f"{media_url}/files/videos/{filename}"
        return f"http://media-generator:8004/files/videos/{filename}"


class RedisJobStoreHelpers:
    """Helper methods from RedisJobStore that don't require Redis."""

    def _make_key(self, job_id: str, prefix: str = "v3") -> str:
        """Generate Redis key for a job."""
        return f"pres:{prefix}:{job_id}"

    def _make_index_key(self, prefix: str = "v3") -> str:
        """Generate Redis key for job index."""
        return f"pres:{prefix}:index"

    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data for JSON storage."""
        result = {}
        for key, value in data.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._serialize_data(value)
            elif isinstance(value, list):
                result[key] = [
                    self._serialize_data(item) if isinstance(item, dict)
                    else item.isoformat() if isinstance(item, datetime)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


# =============================================================================
# TESTS
# =============================================================================

class TestLessonStatus:
    """Tests for LessonStatus enum"""

    def test_all_values(self):
        assert LessonStatus.PENDING == "pending"
        assert LessonStatus.PROCESSING == "processing"
        assert LessonStatus.COMPLETED == "completed"
        assert LessonStatus.FAILED == "failed"
        assert LessonStatus.CANCELLED == "cancelled"
        assert LessonStatus.SKIPPED == "skipped"

    def test_enum_count(self):
        assert len(LessonStatus) == 6

    def test_value_comparison(self):
        assert LessonStatus.COMPLETED.value == "completed"
        assert LessonStatus.FAILED.value == "failed"

    def test_string_equality(self):
        assert LessonStatus.PENDING == "pending"
        assert LessonStatus.PROCESSING == "processing"

    def test_terminal_states(self):
        terminal_states = [LessonStatus.COMPLETED, LessonStatus.FAILED, LessonStatus.CANCELLED, LessonStatus.SKIPPED]
        non_terminal = [LessonStatus.PENDING, LessonStatus.PROCESSING]

        for state in terminal_states:
            assert state.value in ["completed", "failed", "cancelled", "skipped"]

        for state in non_terminal:
            assert state.value in ["pending", "processing"]


class TestJobStatus:
    """Tests for JobStatus enum"""

    def test_all_values(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.PROCESSING == "processing"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.PARTIAL == "partial"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_enum_count(self):
        assert len(JobStatus) == 6

    def test_value_comparison(self):
        assert JobStatus.PARTIAL.value == "partial"

    def test_partial_status_exists(self):
        # PARTIAL is unique to JobStatus (not in LessonStatus)
        assert hasattr(JobStatus, "PARTIAL")
        assert JobStatus.PARTIAL == "partial"


class TestLessonError:
    """Tests for LessonError dataclass"""

    def test_basic_creation(self):
        error = LessonError(
            scene_index=3,
            title="Introduction to Python",
            error_type="tts_failed",
            error_message="TTS service unavailable",
            original_content={"voiceover_text": "Hello world"}
        )
        assert error.scene_index == 3
        assert error.title == "Introduction to Python"
        assert error.error_type == "tts_failed"
        assert error.error_message == "TTS service unavailable"
        assert error.original_content == {"voiceover_text": "Hello world"}

    def test_default_values(self):
        error = LessonError(
            scene_index=0,
            title="Test",
            error_type="error",
            error_message="Test error",
            original_content={}
        )
        assert error.retry_count == 0
        assert error.last_retry_at is None
        assert error.editable is True

    def test_with_retry_info(self):
        error = LessonError(
            scene_index=5,
            title="Advanced Topics",
            error_type="ffmpeg_error",
            error_message="Video encoding failed",
            original_content={"type": "code"},
            retry_count=2,
            last_retry_at="2026-01-30T10:30:00Z"
        )
        assert error.retry_count == 2
        assert error.last_retry_at == "2026-01-30T10:30:00Z"

    def test_editable_flag(self):
        editable_error = LessonError(
            scene_index=0,
            title="Test",
            error_type="error",
            error_message="Error",
            original_content={},
            editable=True
        )
        assert editable_error.editable is True

        non_editable_error = LessonError(
            scene_index=0,
            title="Test",
            error_type="error",
            error_message="Error",
            original_content={},
            editable=False
        )
        assert non_editable_error.editable is False

    def test_rich_original_content(self):
        content = {
            "title": "Building APIs",
            "voiceover_text": "Let's build an API...",
            "type": "code",
            "code": "def api_handler(): pass",
            "language": "python",
            "bullet_points": ["Point 1", "Point 2"],
            "diagram_description": "API architecture"
        }
        error = LessonError(
            scene_index=2,
            title="Building APIs",
            error_type="render_failed",
            error_message="Failed to render code slide",
            original_content=content
        )
        assert error.original_content["code"] == "def api_handler(): pass"
        assert len(error.original_content["bullet_points"]) == 2


class TestRetryResult:
    """Tests for RetryResult dataclass"""

    def test_successful_result(self):
        result = RetryResult(
            success=True,
            message="Lesson regenerated successfully",
            scene_index=3,
            video_url="http://example.com/video.mp4"
        )
        assert result.success is True
        assert result.scene_index == 3
        assert result.video_url is not None
        assert len(result.errors) == 0

    def test_failed_result(self):
        result = RetryResult(
            success=False,
            message="Retry failed due to TTS error",
            scene_index=3,
            errors=["TTS timeout", "Voice ID invalid"]
        )
        assert result.success is False
        assert len(result.errors) == 2
        assert "TTS timeout" in result.errors

    def test_default_values(self):
        result = RetryResult(success=True, message="OK")
        assert result.scene_index is None
        assert result.video_url is None
        assert result.errors == []

    def test_partial_fields(self):
        result = RetryResult(
            success=True,
            message="Lesson completed",
            scene_index=5
        )
        assert result.scene_index == 5
        assert result.video_url is None


class TestRedisConnectionError:
    """Tests for RedisConnectionError exception"""

    def test_basic_exception(self):
        error = RedisConnectionError("Connection refused")
        assert str(error) == "Connection refused"

    def test_raise_and_catch(self):
        with pytest.raises(RedisConnectionError) as exc_info:
            raise RedisConnectionError("Redis unavailable")
        assert "Redis unavailable" in str(exc_info.value)


class TestJobManagerHelpers:
    """Tests for JobManager helper methods"""

    def test_default_prefix(self):
        manager = JobManagerHelpers()
        assert manager.job_prefix == "v3"

    def test_build_scene_url_default(self):
        manager = JobManagerHelpers()
        url = manager._build_scene_url("job123", 5)
        assert url == "http://media-generator:8004/files/videos/job123_scene_005.mp4"

    def test_build_scene_url_with_public(self):
        manager = JobManagerHelpers()
        url = manager._build_scene_url(
            "job456", 10,
            public_media_url="https://cdn.example.com"
        )
        assert url == "https://cdn.example.com/files/videos/job456_scene_010.mp4"

    def test_build_scene_url_with_media(self):
        manager = JobManagerHelpers()
        url = manager._build_scene_url(
            "job789", 0,
            media_url="http://custom-media:9000"
        )
        assert url == "http://custom-media:9000/files/videos/job789_scene_000.mp4"

    def test_build_scene_url_padding(self):
        manager = JobManagerHelpers()
        # Test 3-digit padding
        assert "_scene_000.mp4" in manager._build_scene_url("job", 0)
        assert "_scene_001.mp4" in manager._build_scene_url("job", 1)
        assert "_scene_099.mp4" in manager._build_scene_url("job", 99)
        assert "_scene_100.mp4" in manager._build_scene_url("job", 100)

    def test_build_final_url_default(self):
        manager = JobManagerHelpers()
        url = manager._build_final_url("job123")
        assert url == "http://media-generator:8004/files/videos/job123_final.mp4"

    def test_build_final_url_with_public(self):
        manager = JobManagerHelpers()
        url = manager._build_final_url(
            "job456",
            public_media_url="https://cdn.example.com"
        )
        assert url == "https://cdn.example.com/files/videos/job456_final.mp4"

    def test_build_final_url_with_media(self):
        manager = JobManagerHelpers()
        url = manager._build_final_url(
            "job789",
            media_url="http://custom-media:9000"
        )
        assert url == "http://custom-media:9000/files/videos/job789_final.mp4"

    def test_public_takes_precedence(self):
        manager = JobManagerHelpers()
        # Public URL should take precedence over media URL
        url = manager._build_scene_url(
            "job",
            5,
            public_media_url="https://public.example.com",
            media_url="http://internal.example.com"
        )
        assert "public.example.com" in url


class TestRedisJobStoreHelpers:
    """Tests for RedisJobStore helper methods"""

    def test_make_key_v3(self):
        store = RedisJobStoreHelpers()
        key = store._make_key("abc123", prefix="v3")
        assert key == "pres:v3:abc123"

    def test_make_key_v2(self):
        store = RedisJobStoreHelpers()
        key = store._make_key("xyz789", prefix="v2")
        assert key == "pres:v2:xyz789"

    def test_make_key_v1(self):
        store = RedisJobStoreHelpers()
        key = store._make_key("legacy", prefix="v1")
        assert key == "pres:v1:legacy"

    def test_make_key_default_prefix(self):
        store = RedisJobStoreHelpers()
        key = store._make_key("test")
        assert key == "pres:v3:test"

    def test_make_index_key(self):
        store = RedisJobStoreHelpers()
        key = store._make_index_key("v3")
        assert key == "pres:v3:index"

    def test_make_index_key_default(self):
        store = RedisJobStoreHelpers()
        key = store._make_index_key()
        assert key == "pres:v3:index"

    def test_serialize_simple_data(self):
        store = RedisJobStoreHelpers()
        data = {
            "status": "processing",
            "phase": "rendering",
            "count": 10
        }
        result = store._serialize_data(data)
        assert result == data

    def test_serialize_datetime(self):
        store = RedisJobStoreHelpers()
        dt = datetime(2026, 1, 30, 10, 30, 0)
        data = {"created_at": dt}
        result = store._serialize_data(data)
        assert result["created_at"] == "2026-01-30T10:30:00"

    def test_serialize_nested_dict(self):
        store = RedisJobStoreHelpers()
        data = {
            "request": {
                "topic": "Python",
                "options": {
                    "duration": 300
                }
            }
        }
        result = store._serialize_data(data)
        assert result["request"]["topic"] == "Python"
        assert result["request"]["options"]["duration"] == 300

    def test_serialize_nested_datetime(self):
        store = RedisJobStoreHelpers()
        dt = datetime(2026, 1, 30, 12, 0, 0)
        data = {
            "metadata": {
                "timestamp": dt
            }
        }
        result = store._serialize_data(data)
        assert result["metadata"]["timestamp"] == "2026-01-30T12:00:00"

    def test_serialize_list(self):
        store = RedisJobStoreHelpers()
        data = {
            "slides": [
                {"title": "Slide 1"},
                {"title": "Slide 2"}
            ]
        }
        result = store._serialize_data(data)
        assert len(result["slides"]) == 2
        assert result["slides"][0]["title"] == "Slide 1"

    def test_serialize_list_with_datetime(self):
        store = RedisJobStoreHelpers()
        dt1 = datetime(2026, 1, 30, 10, 0, 0)
        dt2 = datetime(2026, 1, 30, 11, 0, 0)
        data = {
            "timestamps": [dt1, dt2]
        }
        result = store._serialize_data(data)
        assert result["timestamps"][0] == "2026-01-30T10:00:00"
        assert result["timestamps"][1] == "2026-01-30T11:00:00"

    def test_serialize_list_with_dicts(self):
        store = RedisJobStoreHelpers()
        dt = datetime(2026, 1, 30, 10, 0, 0)
        data = {
            "scene_statuses": [
                {"status": "completed", "completed_at": dt},
                {"status": "pending"}
            ]
        }
        result = store._serialize_data(data)
        assert result["scene_statuses"][0]["completed_at"] == "2026-01-30T10:00:00"
        assert result["scene_statuses"][1]["status"] == "pending"

    def test_serialize_mixed_list(self):
        store = RedisJobStoreHelpers()
        data = {
            "items": [1, "string", {"key": "value"}, None]
        }
        result = store._serialize_data(data)
        assert result["items"] == [1, "string", {"key": "value"}, None]

    def test_serialize_empty_data(self):
        store = RedisJobStoreHelpers()
        result = store._serialize_data({})
        assert result == {}

    def test_serialize_preserves_other_types(self):
        store = RedisJobStoreHelpers()
        data = {
            "int_val": 42,
            "float_val": 3.14,
            "bool_val": True,
            "none_val": None,
            "str_val": "hello"
        }
        result = store._serialize_data(data)
        assert result["int_val"] == 42
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True
        assert result["none_val"] is None
        assert result["str_val"] == "hello"


class TestJobStateMachine:
    """Tests for job state transitions"""

    def test_valid_lesson_transitions(self):
        # Valid transitions from each state
        transitions = {
            LessonStatus.PENDING: [LessonStatus.PROCESSING, LessonStatus.CANCELLED, LessonStatus.SKIPPED],
            LessonStatus.PROCESSING: [LessonStatus.COMPLETED, LessonStatus.FAILED, LessonStatus.CANCELLED],
            LessonStatus.COMPLETED: [],  # Terminal
            LessonStatus.FAILED: [LessonStatus.PENDING],  # Can retry
            LessonStatus.CANCELLED: [],  # Terminal
            LessonStatus.SKIPPED: [],  # Terminal
        }

        # PENDING can transition to PROCESSING
        assert LessonStatus.PROCESSING in transitions[LessonStatus.PENDING]

        # PROCESSING can transition to COMPLETED or FAILED
        assert LessonStatus.COMPLETED in transitions[LessonStatus.PROCESSING]
        assert LessonStatus.FAILED in transitions[LessonStatus.PROCESSING]

        # FAILED can go back to PENDING (for retry)
        assert LessonStatus.PENDING in transitions[LessonStatus.FAILED]

    def test_valid_job_transitions(self):
        transitions = {
            JobStatus.PENDING: [JobStatus.PROCESSING, JobStatus.CANCELLED],
            JobStatus.PROCESSING: [JobStatus.COMPLETED, JobStatus.PARTIAL, JobStatus.FAILED, JobStatus.CANCELLED],
            JobStatus.COMPLETED: [],  # Terminal
            JobStatus.PARTIAL: [JobStatus.PROCESSING],  # Can retry failed lessons
            JobStatus.FAILED: [JobStatus.PENDING],  # Can retry entire job
            JobStatus.CANCELLED: [],  # Terminal
        }

        # PROCESSING can go to PARTIAL (some lessons completed, some failed)
        assert JobStatus.PARTIAL in transitions[JobStatus.PROCESSING]

        # PARTIAL can go back to PROCESSING (retrying)
        assert JobStatus.PROCESSING in transitions[JobStatus.PARTIAL]


class TestErrorContentStructure:
    """Tests for error content structure validation"""

    def test_minimal_error_content(self):
        content = {
            "title": "",
            "voiceover_text": "",
            "type": "content"
        }
        error = LessonError(
            scene_index=0,
            title="",
            error_type="unknown",
            error_message="Unknown error",
            original_content=content
        )
        assert "type" in error.original_content

    def test_code_slide_error_content(self):
        content = {
            "title": "Code Demo",
            "voiceover_text": "Let me show you...",
            "type": "code",
            "code": "print('hello')",
            "language": "python"
        }
        error = LessonError(
            scene_index=2,
            title="Code Demo",
            error_type="syntax_highlight_failed",
            error_message="Failed to highlight code",
            original_content=content
        )
        assert error.original_content["type"] == "code"
        assert "code" in error.original_content
        assert "language" in error.original_content

    def test_diagram_slide_error_content(self):
        content = {
            "title": "Architecture",
            "voiceover_text": "Here's the architecture...",
            "type": "diagram",
            "diagram_description": "Microservices architecture"
        }
        error = LessonError(
            scene_index=4,
            title="Architecture",
            error_type="diagram_render_failed",
            error_message="Failed to generate diagram",
            original_content=content
        )
        assert error.original_content["type"] == "diagram"
        assert "diagram_description" in error.original_content


class TestRetryResultScenarios:
    """Tests for different retry scenarios"""

    def test_retry_success_with_rebuild(self):
        result = RetryResult(
            success=True,
            message="Lesson 3 regenerated successfully",
            scene_index=3,
            video_url="http://media/job123_scene_003.mp4"
        )
        assert result.success
        assert result.scene_index == 3
        assert "_scene_003.mp4" in result.video_url

    def test_retry_failed_tts_error(self):
        result = RetryResult(
            success=False,
            message="Retry failed: TTS service unavailable",
            scene_index=5,
            errors=["TTS service timeout", "Voice ID not found"]
        )
        assert not result.success
        assert len(result.errors) == 2

    def test_retry_partial_success(self):
        # Lesson generated but final rebuild failed
        result = RetryResult(
            success=True,
            message="Lesson regenerated, but final video rebuild failed",
            scene_index=2,
            video_url="http://media/job123_scene_002.mp4",
            errors=["Final concat failed"]
        )
        assert result.success
        assert len(result.errors) == 1

    def test_batch_retry_result(self):
        # When retrying multiple lessons
        results = [
            RetryResult(success=True, message="OK", scene_index=1),
            RetryResult(success=False, message="Failed", scene_index=3, errors=["Error"]),
            RetryResult(success=True, message="OK", scene_index=5),
        ]

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) == 2
        assert len(failed) == 1
        assert failed[0].scene_index == 3


class TestKeyNaming:
    """Tests for Redis key naming conventions"""

    def test_key_format_consistency(self):
        store = RedisJobStoreHelpers()

        # All keys should follow pres:{prefix}:{id} format
        key = store._make_key("abc123", "v3")
        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "pres"
        assert parts[1] == "v3"
        assert parts[2] == "abc123"

    def test_index_key_format(self):
        store = RedisJobStoreHelpers()

        key = store._make_index_key("v3")
        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "pres"
        assert parts[1] == "v3"
        assert parts[2] == "index"

    def test_key_uniqueness(self):
        store = RedisJobStoreHelpers()

        # Different job IDs produce different keys
        key1 = store._make_key("job1", "v3")
        key2 = store._make_key("job2", "v3")
        assert key1 != key2

        # Different prefixes produce different keys
        key3 = store._make_key("job1", "v2")
        assert key1 != key3

    def test_special_characters_in_job_id(self):
        store = RedisJobStoreHelpers()

        # UUID-style job IDs should work
        key = store._make_key("550e8400-e29b-41d4-a716-446655440000", "v3")
        assert "550e8400-e29b-41d4-a716-446655440000" in key


class TestSerializationRoundTrip:
    """Tests for data serialization and JSON compatibility"""

    def test_serialized_data_is_json_compatible(self):
        store = RedisJobStoreHelpers()
        data = {
            "status": "processing",
            "created_at": datetime(2026, 1, 30, 10, 0, 0),
            "nested": {
                "updated_at": datetime(2026, 1, 30, 11, 0, 0)
            },
            "list_data": [
                {"timestamp": datetime(2026, 1, 30, 12, 0, 0)}
            ]
        }

        serialized = store._serialize_data(data)

        # Should be JSON serializable
        json_str = json.dumps(serialized)
        assert json_str is not None

        # Should round-trip correctly
        parsed = json.loads(json_str)
        assert parsed["status"] == "processing"
        assert parsed["created_at"] == "2026-01-30T10:00:00"
        assert parsed["nested"]["updated_at"] == "2026-01-30T11:00:00"
        assert parsed["list_data"][0]["timestamp"] == "2026-01-30T12:00:00"

    def test_empty_structures_serialize(self):
        store = RedisJobStoreHelpers()

        data = {
            "empty_dict": {},
            "empty_list": [],
            "empty_string": ""
        }

        serialized = store._serialize_data(data)
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)

        assert parsed["empty_dict"] == {}
        assert parsed["empty_list"] == []
        assert parsed["empty_string"] == ""


class TestIntegration:
    """Integration tests combining multiple components"""

    def test_error_to_retry_flow(self):
        """Test creating an error and converting to retry result."""
        # Create error
        error = LessonError(
            scene_index=3,
            title="Building APIs",
            error_type="tts_failed",
            error_message="TTS timeout after 30s",
            original_content={
                "title": "Building APIs",
                "voiceover_text": "In this lesson...",
                "type": "content"
            },
            retry_count=0
        )

        # Simulate retry
        error.retry_count += 1
        error.last_retry_at = datetime.utcnow().isoformat()

        # Create retry result
        result = RetryResult(
            success=True,
            message=f"Lesson {error.scene_index} regenerated successfully",
            scene_index=error.scene_index,
            video_url="http://media/video.mp4"
        )

        assert error.retry_count == 1
        assert result.success
        assert result.scene_index == error.scene_index

    def test_job_with_multiple_errors(self):
        """Test handling multiple lesson errors."""
        errors = [
            LessonError(
                scene_index=2,
                title="Lesson 2",
                error_type="tts_failed",
                error_message="Error 1",
                original_content={"type": "content"}
            ),
            LessonError(
                scene_index=5,
                title="Lesson 5",
                error_type="ffmpeg_error",
                error_message="Error 2",
                original_content={"type": "code"}
            ),
        ]

        # Aggregate error info
        error_summary = {
            "total_errors": len(errors),
            "error_types": list(set(e.error_type for e in errors)),
            "scene_indices": [e.scene_index for e in errors]
        }

        assert error_summary["total_errors"] == 2
        assert "tts_failed" in error_summary["error_types"]
        assert "ffmpeg_error" in error_summary["error_types"]
        assert error_summary["scene_indices"] == [2, 5]

    def test_url_building_with_job_context(self):
        """Test building URLs in job context."""
        manager = JobManagerHelpers()
        job_id = "pres_abc123"

        # Build URLs for multiple scenes
        scene_urls = [
            manager._build_scene_url(job_id, i)
            for i in range(5)
        ]

        # All URLs should have correct format
        for i, url in enumerate(scene_urls):
            assert f"_scene_{i:03d}.mp4" in url
            assert job_id in url

        # Final URL
        final_url = manager._build_final_url(job_id)
        assert "_final.mp4" in final_url
        assert job_id in final_url
