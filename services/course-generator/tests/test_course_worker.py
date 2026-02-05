"""
Unit tests for CourseWorker

Tests the background worker that processes course generation jobs.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import asdict

import sys
import os

# Add services directory directly to path (avoid services/__init__.py cascade)
_services_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "services"
)
sys.path.insert(0, _services_path)

# Mock heavy dependencies before importing
sys.modules['agents'] = MagicMock()
sys.modules['agents.pedagogical_graph'] = MagicMock()
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()
# Mock aio_pika and its submodules
mock_aio_pika = MagicMock()
mock_aio_pika.abc = MagicMock()
sys.modules['aio_pika'] = mock_aio_pika
sys.modules['aio_pika.abc'] = mock_aio_pika.abc

from course_worker import (
    CourseWorker,
    CourseJobStatus,
    get_worker,
)
from course_queue import QueuedCourseJob


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_queue_service():
    """Create a mock queue service"""
    service = AsyncMock()
    service.connect = AsyncMock()
    service.disconnect = AsyncMock()
    service.consume = AsyncMock()
    service.stop_consuming = MagicMock()
    return service


@pytest.fixture
def mock_redis():
    """Create a mock Redis client"""
    redis = AsyncMock()
    redis.hset = AsyncMock()
    redis.expire = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    return redis


@pytest.fixture
def sample_queued_job():
    """Create a sample queued job for testing"""
    return QueuedCourseJob(
        job_id="test-job-123",
        topic="Introduction to Python Programming",
        num_sections=3,
        lectures_per_section=2,
        user_id="user-456",
        difficulty_start="beginner",
        difficulty_end="intermediate",
        target_audience="beginners",
        language="en",
        category="education",
        domain="programming",
        priority=5,
        document_ids=[],
        source_ids=[]
    )


@pytest.fixture
def sample_outline():
    """Create a sample course outline"""
    return MagicMock(
        total_lectures=6,
        sections=[
            MagicMock(
                title="Section 1",
                lectures=[
                    MagicMock(title="Lecture 1", video_url="http://example.com/video1.mp4"),
                    MagicMock(title="Lecture 2", video_url="http://example.com/video2.mp4"),
                ]
            ),
            MagicMock(
                title="Section 2",
                lectures=[
                    MagicMock(title="Lecture 3", video_url="http://example.com/video3.mp4"),
                    MagicMock(title="Lecture 4", video_url="http://example.com/video4.mp4"),
                ]
            ),
        ],
        model_dump=MagicMock(return_value={
            "sections": [
                {"title": "Section 1", "lectures": [{"title": "Lecture 1"}, {"title": "Lecture 2"}]},
                {"title": "Section 2", "lectures": [{"title": "Lecture 3"}, {"title": "Lecture 4"}]},
            ],
            "total_lectures": 6
        })
    )


# ============================================================================
# CourseJobStatus Tests
# ============================================================================

class TestCourseJobStatus:
    """Test CourseJobStatus constants"""

    def test_status_values(self):
        """Test that all status values are defined"""
        assert CourseJobStatus.QUEUED == "queued"
        assert CourseJobStatus.GENERATING_OUTLINE == "generating_outline"
        assert CourseJobStatus.GENERATING_LECTURES == "generating_lectures"
        assert CourseJobStatus.CREATING_PACKAGE == "creating_package"
        assert CourseJobStatus.COMPLETED == "completed"
        assert CourseJobStatus.FAILED == "failed"


# ============================================================================
# CourseWorker Initialization Tests
# ============================================================================

class TestCourseWorkerInit:
    """Test CourseWorker initialization"""

    def test_default_initialization(self, mock_queue_service):
        """Test worker initializes with default values"""
        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor'):
                with patch('course_worker.CoursePlanner'):
                    worker = CourseWorker(queue_service=mock_queue_service)

                    assert worker.queue_service == mock_queue_service
                    assert worker.max_concurrent_jobs == 1
                    assert worker._running is False
                    assert worker._current_jobs == {}

    def test_custom_max_concurrent_jobs(self, mock_queue_service):
        """Test worker with custom max concurrent jobs"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                worker = CourseWorker(
                    queue_service=mock_queue_service,
                    max_concurrent_jobs=5
                )

                assert worker.max_concurrent_jobs == 5


# ============================================================================
# Redis Status Update Tests
# ============================================================================

class TestRedisStatusUpdate:
    """Test Redis status update functionality"""

    @pytest.mark.asyncio
    async def test_update_job_status_basic(self, mock_queue_service, mock_redis):
        """Test basic job status update"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis

                await worker._update_job_status(
                    job_id="test-123",
                    status=CourseJobStatus.GENERATING_OUTLINE,
                    progress=25
                )

                mock_redis.hset.assert_called_once()
                mock_redis.expire.assert_called_once()

                # Check the key format
                call_args = mock_redis.hset.call_args
                assert call_args[0][0] == "course_job:test-123"

    @pytest.mark.asyncio
    async def test_update_job_status_with_error(self, mock_queue_service, mock_redis):
        """Test job status update with error message"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis

                await worker._update_job_status(
                    job_id="test-123",
                    status=CourseJobStatus.FAILED,
                    error="Something went wrong"
                )

                call_args = mock_redis.hset.call_args
                mapping = call_args[1]['mapping']
                assert "error" in mapping
                assert mapping["error"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_update_job_status_with_output_urls(self, mock_queue_service, mock_redis):
        """Test job status update with output URLs"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis

                output_urls = {
                    "videos": ["http://example.com/video1.mp4"],
                    "zip": "http://example.com/course.zip"
                }

                await worker._update_job_status(
                    job_id="test-123",
                    status=CourseJobStatus.COMPLETED,
                    progress=100,
                    output_urls=output_urls
                )

                call_args = mock_redis.hset.call_args
                mapping = call_args[1]['mapping']
                assert "output_urls" in mapping

    @pytest.mark.asyncio
    async def test_update_job_status_with_outline(self, mock_queue_service, mock_redis):
        """Test job status update with outline data"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis

                outline = {"sections": [], "total_lectures": 6}

                await worker._update_job_status(
                    job_id="test-123",
                    status=CourseJobStatus.GENERATING_LECTURES,
                    progress=15,
                    outline=outline
                )

                call_args = mock_redis.hset.call_args
                mapping = call_args[1]['mapping']
                assert "outline" in mapping

    @pytest.mark.asyncio
    async def test_update_job_status_redis_error(self, mock_queue_service, mock_redis):
        """Test handling of Redis errors"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis
                mock_redis.hset.side_effect = Exception("Redis connection failed")

                with pytest.raises(Exception) as excinfo:
                    await worker._update_job_status(
                        job_id="test-123",
                        status=CourseJobStatus.GENERATING_OUTLINE,
                        progress=5
                    )

                assert "Redis connection failed" in str(excinfo.value)


# ============================================================================
# Job Processing Tests
# ============================================================================

class TestJobProcessing:
    """Test job processing logic"""

    @pytest.mark.asyncio
    async def test_process_job_success(
        self,
        mock_queue_service,
        mock_redis,
        sample_queued_job,
        sample_outline
    ):
        """Test successful job processing"""
        with patch('course_worker.CourseCompositor') as MockCompositor:
            with patch('course_worker.CoursePlanner') as MockPlanner:
                # Setup mocks
                mock_planner = MockPlanner.return_value
                mock_planner.generate_outline = AsyncMock(return_value=sample_outline)

                mock_compositor = MockCompositor.return_value
                mock_compositor.generate_all_lectures = AsyncMock()
                mock_compositor.create_course_zip = AsyncMock(return_value="/path/to/course.zip")

                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis

                await worker.process_job(sample_queued_job)

                # Verify outline was generated
                mock_planner.generate_outline.assert_called_once()

                # Verify lectures were generated
                mock_compositor.generate_all_lectures.assert_called_once()

                # Verify ZIP was created
                mock_compositor.create_course_zip.assert_called_once()

                # Verify job was removed from current jobs
                assert sample_queued_job.job_id not in worker._current_jobs

    @pytest.mark.asyncio
    async def test_process_job_tracks_current_job(
        self,
        mock_queue_service,
        mock_redis,
        sample_queued_job,
        sample_outline
    ):
        """Test that current job is tracked during processing"""
        with patch('course_worker.CourseCompositor') as MockCompositor:
            with patch('course_worker.CoursePlanner') as MockPlanner:
                mock_planner = MockPlanner.return_value
                mock_planner.generate_outline = AsyncMock(return_value=sample_outline)

                mock_compositor = MockCompositor.return_value
                mock_compositor.generate_all_lectures = AsyncMock()
                mock_compositor.create_course_zip = AsyncMock(return_value="/path/to/course.zip")

                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis

                # Check job is tracked during processing
                original_generate = mock_planner.generate_outline

                async def check_tracking(*args, **kwargs):
                    assert sample_queued_job.job_id in worker._current_jobs
                    return await original_generate(*args, **kwargs)

                mock_planner.generate_outline = check_tracking

                await worker.process_job(sample_queued_job)

    @pytest.mark.asyncio
    async def test_process_job_failure(
        self,
        mock_queue_service,
        mock_redis,
        sample_queued_job
    ):
        """Test job processing failure handling"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner') as MockPlanner:
                mock_planner = MockPlanner.return_value
                mock_planner.generate_outline = AsyncMock(
                    side_effect=Exception("LLM API error")
                )

                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis

                with pytest.raises(Exception) as excinfo:
                    await worker.process_job(sample_queued_job)

                assert "LLM API error" in str(excinfo.value)

                # Verify job was removed from current jobs even on failure
                assert sample_queued_job.job_id not in worker._current_jobs

    @pytest.mark.asyncio
    async def test_process_job_with_quiz_config(
        self,
        mock_queue_service,
        mock_redis,
        sample_outline
    ):
        """Test job processing with quiz configuration"""
        job_with_quiz = QueuedCourseJob(
            job_id="test-quiz-job",
            topic="Python Quiz Course",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-789",
            quiz_config={
                "enabled": True,
                "frequency": "per_section",
                "questions_per_quiz": 10,
                "passing_score": 80,
                "show_explanations": True
            },
            document_ids=[],
            source_ids=[]
        )

        with patch('course_worker.CourseCompositor') as MockCompositor:
            with patch('course_worker.CoursePlanner') as MockPlanner:
                mock_planner = MockPlanner.return_value
                mock_planner.generate_outline = AsyncMock(return_value=sample_outline)

                mock_compositor = MockCompositor.return_value
                mock_compositor.generate_all_lectures = AsyncMock()
                mock_compositor.create_course_zip = AsyncMock(return_value="/path/to/course.zip")

                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis

                await worker.process_job(job_with_quiz)

                # Verify outline was generated with quiz config
                mock_planner.generate_outline.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_job_with_document_ids(
        self,
        mock_queue_service,
        mock_redis,
        sample_outline
    ):
        """Test job processing with RAG document IDs"""
        job_with_docs = QueuedCourseJob(
            job_id="test-rag-job",
            topic="RAG-based Course",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-789",
            document_ids=["doc-1", "doc-2", "doc-3"],
            source_ids=[]
        )

        with patch('course_worker.CourseCompositor') as MockCompositor:
            with patch('course_worker.CoursePlanner') as MockPlanner:
                with patch('course_worker.RAGService') as MockRAG:
                    mock_rag = MockRAG.return_value
                    mock_rag.get_context_for_course_generation = AsyncMock(
                        return_value="RAG context content..."
                    )

                    mock_planner = MockPlanner.return_value
                    mock_planner.generate_outline = AsyncMock(return_value=sample_outline)

                    mock_compositor = MockCompositor.return_value
                    mock_compositor.generate_all_lectures = AsyncMock()
                    mock_compositor.create_course_zip = AsyncMock(return_value="/path/to/course.zip")

                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = mock_redis

                    await worker.process_job(job_with_docs)

                    # Verify RAG service was called
                    mock_rag.get_context_for_course_generation.assert_called_once()


# ============================================================================
# Worker Lifecycle Tests
# ============================================================================

class TestWorkerLifecycle:
    """Test worker start/stop lifecycle"""

    @pytest.mark.asyncio
    async def test_start_worker(self, mock_queue_service):
        """Test starting the worker"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                worker = CourseWorker(queue_service=mock_queue_service)

                # Don't actually start consuming, just verify setup
                mock_queue_service.consume = AsyncMock()

                # Start in background task
                start_task = asyncio.create_task(worker.start())

                # Give it a moment to start
                await asyncio.sleep(0.1)

                assert worker._running is True
                mock_queue_service.connect.assert_called_once()
                mock_queue_service.consume.assert_called_once()

                # Stop the worker
                worker.stop()
                start_task.cancel()
                try:
                    await start_task
                except asyncio.CancelledError:
                    pass

    def test_stop_worker(self, mock_queue_service):
        """Test stopping the worker"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                worker = CourseWorker(queue_service=mock_queue_service)
                worker._running = True

                worker.stop()

                assert worker._running is False
                mock_queue_service.stop_consuming.assert_called_once()

    def test_get_current_jobs(self, mock_queue_service):
        """Test getting current jobs"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                worker = CourseWorker(queue_service=mock_queue_service)
                worker._current_jobs = {
                    "job-1": "data1",
                    "job-2": "data2"
                }

                jobs = worker.get_current_jobs()

                assert jobs == {"job-1": "data1", "job-2": "data2"}
                # Should be a copy, not the original
                assert jobs is not worker._current_jobs


# ============================================================================
# Singleton Tests
# ============================================================================

class TestWorkerSingleton:
    """Test worker singleton pattern"""

    def test_get_worker_singleton(self):
        """Test that get_worker returns singleton"""
        with patch('course_worker.CourseCompositor'):
            with patch('course_worker.CoursePlanner'):
                with patch('course_worker.get_queue_service'):
                    # Reset singleton
                    import services.course_worker as worker_module
                    worker_module._worker = None

                    worker1 = get_worker()
                    worker2 = get_worker()

                    assert worker1 is worker2

                    # Cleanup
                    worker_module._worker = None


# ============================================================================
# Progress Callback Tests
# ============================================================================

class TestProgressCallback:
    """Test progress callback functionality"""

    @pytest.mark.asyncio
    async def test_progress_callback_updates_redis(
        self,
        mock_queue_service,
        mock_redis,
        sample_queued_job,
        sample_outline
    ):
        """Test that progress callback updates Redis"""
        with patch('course_worker.CourseCompositor') as MockCompositor:
            with patch('course_worker.CoursePlanner') as MockPlanner:
                mock_planner = MockPlanner.return_value
                mock_planner.generate_outline = AsyncMock(return_value=sample_outline)

                # Capture the progress callback
                captured_callback = None

                async def capture_callback(job, request, progress_callback):
                    nonlocal captured_callback
                    captured_callback = progress_callback
                    # Simulate calling the callback
                    progress_callback(3, 6, "Lecture 3")
                    await asyncio.sleep(0.1)  # Let the callback task run

                mock_compositor = MockCompositor.return_value
                mock_compositor.generate_all_lectures = capture_callback
                mock_compositor.create_course_zip = AsyncMock(return_value="/path/to/course.zip")

                worker = CourseWorker(queue_service=mock_queue_service)
                worker._redis = mock_redis

                await worker.process_job(sample_queued_job)

                # Verify Redis was updated multiple times (status updates)
                assert mock_redis.hset.call_count >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
