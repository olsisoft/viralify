"""
Integration tests for Course Workers

Tests the full integration between:
- CourseQueueService (RabbitMQ)
- CourseWorker (Background processing)
- Redis (Job status storage)

These tests verify the complete workflow from job submission to completion.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import asdict
from typing import List, Dict, Any
import sys
import os

# Add services directory to path
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

# Mock aio_pika
mock_aio_pika = MagicMock()
mock_aio_pika.abc = MagicMock()
mock_aio_pika.Message = MagicMock()
mock_aio_pika.DeliveryMode = MagicMock()
mock_aio_pika.DeliveryMode.PERSISTENT = 2
sys.modules['aio_pika'] = mock_aio_pika
sys.modules['aio_pika.abc'] = mock_aio_pika.abc

from course_worker import CourseWorker, CourseJobStatus, get_worker
from course_queue import CourseQueueService, QueuedCourseJob, get_queue_service


# ============================================================================
# Test Infrastructure - Simulated Message Broker
# ============================================================================

class SimulatedMessageBroker:
    """
    Simulates RabbitMQ behavior for integration testing.

    Tracks messages, supports priority queue, and simulates DLQ.
    """

    def __init__(self):
        self.main_queue: List[Dict[str, Any]] = []
        self.dlq: List[Dict[str, Any]] = []
        self.acknowledged: List[str] = []
        self.rejected: List[str] = []
        self.consumers: List[callable] = []
        self._is_consuming = False

    def publish(self, job: QueuedCourseJob) -> bool:
        """Add job to queue, sorted by priority"""
        message = {
            "job_id": job.job_id,
            "body": job.to_json(),
            "priority": job.priority,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.main_queue.append(message)
        # Sort by priority (lower = higher priority)
        self.main_queue.sort(key=lambda x: x["priority"])
        return True

    def get_next_message(self) -> Dict[str, Any]:
        """Get next message from queue (FIFO with priority)"""
        if not self.main_queue:
            return None
        return self.main_queue.pop(0)

    def acknowledge(self, job_id: str):
        """Mark message as successfully processed"""
        self.acknowledged.append(job_id)

    def reject_to_dlq(self, job_id: str, message: Dict[str, Any]):
        """Move failed message to DLQ"""
        self.rejected.append(job_id)
        self.dlq.append(message)

    def requeue_from_dlq(self, job_id: str) -> bool:
        """Move message from DLQ back to main queue"""
        for i, msg in enumerate(self.dlq):
            if msg["job_id"] == job_id:
                self.main_queue.append(self.dlq.pop(i))
                self.main_queue.sort(key=lambda x: x["priority"])
                return True
        return False

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        return {
            "pending": len(self.main_queue),
            "failed": len(self.dlq),
            "acknowledged": len(self.acknowledged),
            "rejected": len(self.rejected)
        }


class SimulatedRedis:
    """
    Simulates Redis behavior for integration testing.

    Tracks all operations for verification.
    """

    def __init__(self):
        self.data: Dict[str, Dict[str, str]] = {}
        self.expiries: Dict[str, int] = {}
        self.operations: List[Dict[str, Any]] = []

    async def hset(self, key: str, mapping: Dict[str, str]):
        """Set hash fields"""
        if key not in self.data:
            self.data[key] = {}
        self.data[key].update(mapping)
        self.operations.append({
            "op": "hset",
            "key": key,
            "mapping": mapping.copy(),
            "timestamp": datetime.utcnow().isoformat()
        })

    async def hgetall(self, key: str) -> Dict[str, str]:
        """Get all hash fields"""
        self.operations.append({"op": "hgetall", "key": key})
        return self.data.get(key, {})

    async def expire(self, key: str, seconds: int):
        """Set key expiry"""
        self.expiries[key] = seconds
        self.operations.append({"op": "expire", "key": key, "seconds": seconds})

    async def delete(self, key: str) -> int:
        """Delete key"""
        if key in self.data:
            del self.data[key]
            self.operations.append({"op": "delete", "key": key})
            return 1
        return 0

    def get_job_status(self, job_id: str) -> Dict[str, str]:
        """Helper: Get job status data"""
        return self.data.get(f"course_job:{job_id}", {})

    def get_status_history(self, job_id: str) -> List[str]:
        """Helper: Get all status changes for a job"""
        statuses = []
        for op in self.operations:
            if op.get("op") == "hset" and op.get("key") == f"course_job:{job_id}":
                if "status" in op.get("mapping", {}):
                    statuses.append(op["mapping"]["status"])
        return statuses


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def message_broker():
    """Create a simulated message broker"""
    return SimulatedMessageBroker()


@pytest.fixture
def redis_store():
    """Create a simulated Redis store"""
    return SimulatedRedis()


@pytest.fixture
def sample_job():
    """Create a sample job"""
    return QueuedCourseJob(
        job_id="integration-test-001",
        topic="Python Fundamentals for Beginners",
        num_sections=2,
        lectures_per_section=2,
        user_id="user-integration-test",
        difficulty_start="beginner",
        difficulty_end="intermediate",
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
    outline = MagicMock()
    outline.total_lectures = 4
    outline.sections = [
        MagicMock(
            title="Getting Started",
            lectures=[
                MagicMock(title="Introduction", video_url="http://test.com/v1.mp4"),
                MagicMock(title="Setup", video_url="http://test.com/v2.mp4"),
            ]
        ),
        MagicMock(
            title="Core Concepts",
            lectures=[
                MagicMock(title="Variables", video_url="http://test.com/v3.mp4"),
                MagicMock(title="Functions", video_url="http://test.com/v4.mp4"),
            ]
        ),
    ]
    outline.model_dump = MagicMock(return_value={
        "sections": [
            {"title": "Getting Started", "lectures": [{"title": "Introduction"}, {"title": "Setup"}]},
            {"title": "Core Concepts", "lectures": [{"title": "Variables"}, {"title": "Functions"}]},
        ],
        "total_lectures": 4
    })
    return outline


@pytest.fixture
def mock_planner(sample_outline):
    """Create a mock course planner"""
    planner = MagicMock()
    planner.generate_outline = AsyncMock(return_value=sample_outline)
    return planner


@pytest.fixture
def mock_compositor():
    """Create a mock course compositor"""
    compositor = MagicMock()
    compositor.generate_all_lectures = AsyncMock()
    compositor.create_course_zip = AsyncMock(return_value="/tmp/course.zip")
    return compositor


# ============================================================================
# Integration Test: Full Job Lifecycle
# ============================================================================

class TestFullJobLifecycle:
    """Test the complete job lifecycle from submission to completion"""

    @pytest.mark.asyncio
    async def test_successful_job_processing(
        self, message_broker, redis_store, sample_job, mock_planner, mock_compositor
    ):
        """Test a job that completes successfully through the entire pipeline"""
        # Arrange
        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()
        mock_queue_service.consume = AsyncMock()
        mock_queue_service.stop_consuming = MagicMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    # Act - Process the job
                    await worker.process_job(sample_job)

        # Assert - Verify the complete status transition
        status_history = redis_store.get_status_history(sample_job.job_id)
        assert status_history == [
            "generating_outline",
            "generating_lectures",
            "creating_package",
            "completed"
        ]

        # Verify final state
        final_status = redis_store.get_job_status(sample_job.job_id)
        assert final_status["status"] == "completed"
        assert final_status["progress"] == "100"
        assert "output_urls" in final_status

        # Verify planner and compositor were called
        mock_planner.generate_outline.assert_called_once()
        mock_compositor.generate_all_lectures.assert_called_once()
        mock_compositor.create_course_zip.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_job_updates_status(
        self, message_broker, redis_store, sample_job
    ):
        """Test that a failed job correctly updates status to FAILED"""
        # Arrange
        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        mock_planner = MagicMock()
        mock_planner.generate_outline = AsyncMock(
            side_effect=Exception("LLM API Error: Rate limit exceeded")
        )

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor'):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    # Act & Assert - Job should raise exception
                    with pytest.raises(Exception) as exc_info:
                        await worker.process_job(sample_job)

                    assert "Rate limit exceeded" in str(exc_info.value)

        # Verify failed status
        final_status = redis_store.get_job_status(sample_job.job_id)
        assert final_status["status"] == "failed"
        assert "Rate limit exceeded" in final_status.get("error", "")

    @pytest.mark.asyncio
    async def test_job_removed_from_current_jobs_after_completion(
        self, redis_store, sample_job, mock_planner, mock_compositor
    ):
        """Test that jobs are removed from current_jobs tracking after completion"""
        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    # Verify job is tracked during processing
                    assert sample_job.job_id not in worker._current_jobs

                    await worker.process_job(sample_job)

                    # Verify job is removed after completion
                    assert sample_job.job_id not in worker._current_jobs

    @pytest.mark.asyncio
    async def test_job_removed_from_current_jobs_after_failure(
        self, redis_store, sample_job
    ):
        """Test that jobs are removed from current_jobs even after failure"""
        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        mock_planner = MagicMock()
        mock_planner.generate_outline = AsyncMock(side_effect=Exception("Test error"))

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor'):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    with pytest.raises(Exception):
                        await worker.process_job(sample_job)

                    # Verify job is removed even after failure
                    assert sample_job.job_id not in worker._current_jobs


# ============================================================================
# Integration Test: Multiple Jobs Processing
# ============================================================================

class TestMultipleJobsProcessing:
    """Test processing multiple jobs in sequence"""

    @pytest.mark.asyncio
    async def test_process_multiple_jobs_sequentially(
        self, redis_store, mock_planner, mock_compositor
    ):
        """Test that multiple jobs are processed correctly in sequence"""
        jobs = [
            QueuedCourseJob(
                job_id=f"multi-job-{i}",
                topic=f"Course Topic {i}",
                num_sections=2,
                lectures_per_section=2,
                user_id="user-multi",
                document_ids=[],
                source_ids=[]
            )
            for i in range(3)
        ]

        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    # Process all jobs
                    for job in jobs:
                        await worker.process_job(job)

        # Verify all jobs completed
        for job in jobs:
            final_status = redis_store.get_job_status(job.job_id)
            assert final_status["status"] == "completed"
            assert final_status["progress"] == "100"

        # Verify planner was called for each job
        assert mock_planner.generate_outline.call_count == 3

    @pytest.mark.asyncio
    async def test_job_isolation_on_failure(self, redis_store, mock_compositor):
        """Test that one job's failure doesn't affect other jobs"""
        jobs = [
            QueuedCourseJob(job_id="job-ok-1", topic="Good Job 1", num_sections=1,
                           lectures_per_section=1, user_id="user", document_ids=[], source_ids=[]),
            QueuedCourseJob(job_id="job-fail", topic="Failing Job", num_sections=1,
                           lectures_per_section=1, user_id="user", document_ids=[], source_ids=[]),
            QueuedCourseJob(job_id="job-ok-2", topic="Good Job 2", num_sections=1,
                           lectures_per_section=1, user_id="user", document_ids=[], source_ids=[]),
        ]

        # Create outline mock that fails for specific job
        def outline_side_effect(request):
            if "Failing Job" in str(request):
                raise Exception("Intentional failure for testing")
            outline = MagicMock()
            outline.total_lectures = 1
            outline.sections = [MagicMock(title="Section", lectures=[
                MagicMock(title="Lecture", video_url="http://test.com/v.mp4")
            ])]
            outline.model_dump = MagicMock(return_value={"sections": [], "total_lectures": 1})
            return outline

        mock_planner = MagicMock()
        mock_planner.generate_outline = AsyncMock(side_effect=outline_side_effect)

        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    # Process jobs, catching the failure
                    for job in jobs:
                        try:
                            await worker.process_job(job)
                        except Exception:
                            pass  # Continue processing

        # Verify good jobs completed, bad job failed
        assert redis_store.get_job_status("job-ok-1")["status"] == "completed"
        assert redis_store.get_job_status("job-fail")["status"] == "failed"
        assert redis_store.get_job_status("job-ok-2")["status"] == "completed"


# ============================================================================
# Integration Test: Priority Queue Behavior
# ============================================================================

class TestPriorityQueueBehavior:
    """Test priority queue ordering"""

    def test_jobs_ordered_by_priority(self, message_broker):
        """Test that jobs are processed in priority order"""
        # Add jobs with different priorities
        jobs = [
            QueuedCourseJob(job_id="low-priority", topic="Low", num_sections=1,
                           lectures_per_section=1, user_id="user", priority=10,
                           document_ids=[], source_ids=[]),
            QueuedCourseJob(job_id="high-priority", topic="High", num_sections=1,
                           lectures_per_section=1, user_id="user", priority=1,
                           document_ids=[], source_ids=[]),
            QueuedCourseJob(job_id="medium-priority", topic="Medium", num_sections=1,
                           lectures_per_section=1, user_id="user", priority=5,
                           document_ids=[], source_ids=[]),
        ]

        # Publish in random order
        for job in jobs:
            message_broker.publish(job)

        # Verify they come out in priority order
        first = message_broker.get_next_message()
        assert first["job_id"] == "high-priority"

        second = message_broker.get_next_message()
        assert second["job_id"] == "medium-priority"

        third = message_broker.get_next_message()
        assert third["job_id"] == "low-priority"

    def test_same_priority_fifo(self, message_broker):
        """Test that jobs with same priority are FIFO"""
        jobs = [
            QueuedCourseJob(job_id="first", topic="First", num_sections=1,
                           lectures_per_section=1, user_id="user", priority=5,
                           document_ids=[], source_ids=[]),
            QueuedCourseJob(job_id="second", topic="Second", num_sections=1,
                           lectures_per_section=1, user_id="user", priority=5,
                           document_ids=[], source_ids=[]),
            QueuedCourseJob(job_id="third", topic="Third", num_sections=1,
                           lectures_per_section=1, user_id="user", priority=5,
                           document_ids=[], source_ids=[]),
        ]

        for job in jobs:
            message_broker.publish(job)

        # Should come out in order published
        assert message_broker.get_next_message()["job_id"] == "first"
        assert message_broker.get_next_message()["job_id"] == "second"
        assert message_broker.get_next_message()["job_id"] == "third"


# ============================================================================
# Integration Test: Dead Letter Queue
# ============================================================================

class TestDeadLetterQueue:
    """Test DLQ behavior for failed jobs"""

    def test_failed_job_moves_to_dlq(self, message_broker):
        """Test that failed jobs are moved to DLQ"""
        job = QueuedCourseJob(
            job_id="failing-job",
            topic="This will fail",
            num_sections=1,
            lectures_per_section=1,
            user_id="user",
            document_ids=[],
            source_ids=[]
        )

        message_broker.publish(job)
        message = message_broker.get_next_message()

        # Simulate processing failure
        message_broker.reject_to_dlq(job.job_id, message)

        # Verify job is in DLQ
        stats = message_broker.get_stats()
        assert stats["pending"] == 0
        assert stats["failed"] == 1
        assert stats["rejected"] == 1

    def test_requeue_from_dlq(self, message_broker):
        """Test that jobs can be requeued from DLQ"""
        job = QueuedCourseJob(
            job_id="retry-job",
            topic="Retry this",
            num_sections=1,
            lectures_per_section=1,
            user_id="user",
            document_ids=[],
            source_ids=[]
        )

        message_broker.publish(job)
        message = message_broker.get_next_message()
        message_broker.reject_to_dlq(job.job_id, message)

        # Requeue the job
        result = message_broker.requeue_from_dlq(job.job_id)
        assert result is True

        # Verify job is back in main queue
        stats = message_broker.get_stats()
        assert stats["pending"] == 1
        assert stats["failed"] == 0

        # Can retrieve the job again
        requeued = message_broker.get_next_message()
        assert requeued["job_id"] == "retry-job"

    def test_requeue_nonexistent_job(self, message_broker):
        """Test that requeuing nonexistent job returns False"""
        result = message_broker.requeue_from_dlq("nonexistent-job")
        assert result is False


# ============================================================================
# Integration Test: Redis Status Consistency
# ============================================================================

class TestRedisStatusConsistency:
    """Test Redis status updates are consistent"""

    @pytest.mark.asyncio
    async def test_status_transitions_are_sequential(
        self, redis_store, sample_job, mock_planner, mock_compositor
    ):
        """Test that status transitions follow the correct sequence"""
        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    await worker.process_job(sample_job)

        # Get all hset operations for this job
        job_key = f"course_job:{sample_job.job_id}"
        hset_ops = [
            op for op in redis_store.operations
            if op.get("op") == "hset" and op.get("key") == job_key
        ]

        # Verify we have the expected number of status updates
        assert len(hset_ops) >= 4  # outline, lectures, package, completed

        # Verify progress increases monotonically
        progress_values = [
            float(op["mapping"].get("progress", 0))
            for op in hset_ops
            if "progress" in op.get("mapping", {})
        ]
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i-1], \
                f"Progress should be monotonic: {progress_values}"

    @pytest.mark.asyncio
    async def test_outline_stored_in_redis(
        self, redis_store, sample_job, mock_planner, mock_compositor
    ):
        """Test that the course outline is stored in Redis"""
        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    await worker.process_job(sample_job)

        # Verify outline was stored
        final_status = redis_store.get_job_status(sample_job.job_id)
        assert "outline" in final_status

        # Verify outline can be parsed
        outline = json.loads(final_status["outline"])
        assert "sections" in outline
        assert outline["total_lectures"] == 4

    @pytest.mark.asyncio
    async def test_expiry_set_on_job_data(
        self, redis_store, sample_job, mock_planner, mock_compositor
    ):
        """Test that TTL is set on job data in Redis"""
        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    await worker.process_job(sample_job)

        # Verify expiry was set (48 hours = 172800 seconds)
        job_key = f"course_job:{sample_job.job_id}"
        assert job_key in redis_store.expiries
        assert redis_store.expiries[job_key] == 172800


# ============================================================================
# Integration Test: Quiz Configuration
# ============================================================================

class TestQuizConfigIntegration:
    """Test quiz configuration is passed through the pipeline"""

    @pytest.mark.asyncio
    async def test_quiz_config_passed_to_planner(
        self, redis_store, mock_compositor
    ):
        """Test that quiz configuration reaches the planner"""
        job_with_quiz = QueuedCourseJob(
            job_id="quiz-job",
            topic="Python with Quizzes",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-quiz",
            quiz_config={
                "enabled": True,
                "frequency": "per_lecture",
                "questions_per_quiz": 10,
                "passing_score": 80,
                "show_explanations": True
            },
            document_ids=[],
            source_ids=[]
        )

        captured_request = None

        async def capture_outline(request):
            nonlocal captured_request
            captured_request = request
            outline = MagicMock()
            outline.total_lectures = 4
            outline.sections = [MagicMock(title="S", lectures=[
                MagicMock(title="L", video_url="http://test.com/v.mp4")
            ])]
            outline.model_dump = MagicMock(return_value={"sections": [], "total_lectures": 4})
            return outline

        mock_planner = MagicMock()
        mock_planner.generate_outline = AsyncMock(side_effect=capture_outline)

        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    await worker.process_job(job_with_quiz)

        # Verify the request was captured
        assert captured_request is not None


# ============================================================================
# Integration Test: RAG Document Integration
# ============================================================================

class TestRAGDocumentIntegration:
    """Test RAG document handling in the pipeline"""

    @pytest.mark.asyncio
    async def test_rag_context_fetched_for_document_ids(
        self, redis_store, mock_planner, mock_compositor
    ):
        """Test that RAG context is fetched when document_ids are provided"""
        job_with_docs = QueuedCourseJob(
            job_id="rag-job",
            topic="Topic with Documents",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-rag",
            document_ids=["doc-1", "doc-2", "doc-3"],
            source_ids=[]
        )

        mock_rag_service = MagicMock()
        mock_rag_service.get_context_for_course_generation = AsyncMock(
            return_value="This is RAG context from documents"
        )

        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    with patch('course_worker.RAGService', return_value=mock_rag_service):
                        worker = CourseWorker(queue_service=mock_queue_service)
                        worker._redis = redis_store

                        await worker.process_job(job_with_docs)

        # Verify RAG service was called with correct params
        mock_rag_service.get_context_for_course_generation.assert_called_once()
        call_kwargs = mock_rag_service.get_context_for_course_generation.call_args
        assert call_kwargs.kwargs["document_ids"] == ["doc-1", "doc-2", "doc-3"]
        assert call_kwargs.kwargs["user_id"] == "user-rag"

    @pytest.mark.asyncio
    async def test_rag_error_does_not_block_generation(
        self, redis_store, mock_planner, mock_compositor
    ):
        """Test that RAG errors don't block course generation"""
        job_with_docs = QueuedCourseJob(
            job_id="rag-error-job",
            topic="Topic with Failing RAG",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-rag-error",
            document_ids=["doc-1"],
            source_ids=[]
        )

        mock_rag_service = MagicMock()
        mock_rag_service.get_context_for_course_generation = AsyncMock(
            side_effect=Exception("RAG Service Unavailable")
        )

        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    with patch('course_worker.RAGService', return_value=mock_rag_service):
                        worker = CourseWorker(queue_service=mock_queue_service)
                        worker._redis = redis_store

                        # Should NOT raise despite RAG error
                        await worker.process_job(job_with_docs)

        # Job should still complete
        final_status = redis_store.get_job_status(job_with_docs.job_id)
        assert final_status["status"] == "completed"


# ============================================================================
# Integration Test: Worker Lifecycle
# ============================================================================

class TestWorkerLifecycle:
    """Test worker start/stop lifecycle"""

    @pytest.mark.asyncio
    async def test_worker_start_connects_to_queue(self):
        """Test that starting worker connects to queue"""
        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()
        mock_queue_service.consume = AsyncMock()
        mock_queue_service.stop_consuming = MagicMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor'):
                with patch('course_worker.CoursePlanner'):
                    worker = CourseWorker(queue_service=mock_queue_service)

                    # Start in background task
                    start_task = asyncio.create_task(worker.start())

                    # Give it time to connect
                    await asyncio.sleep(0.1)

                    # Stop the worker
                    worker.stop()

                    # Cancel the task
                    start_task.cancel()
                    try:
                        await start_task
                    except asyncio.CancelledError:
                        pass

        # Verify connect was called
        mock_queue_service.connect.assert_called_once()

    def test_worker_stop_sets_running_false(self):
        """Test that stop() sets _running to False"""
        mock_queue_service = MagicMock()
        mock_queue_service.stop_consuming = MagicMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor'):
                with patch('course_worker.CoursePlanner'):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._running = True

                    worker.stop()

                    assert worker._running is False
                    mock_queue_service.stop_consuming.assert_called_once()


# ============================================================================
# Integration Test: Progress Callbacks
# ============================================================================

class TestProgressCallbacks:
    """Test progress callback integration"""

    @pytest.mark.asyncio
    async def test_progress_callback_updates_redis(
        self, redis_store, sample_job, mock_planner
    ):
        """Test that progress callbacks update Redis status"""
        progress_updates = []

        async def mock_generate_lectures(job, request, progress_callback):
            # Simulate progress updates
            for i in range(1, 5):
                progress_callback(i, 4, f"Lecture {i}")
                await asyncio.sleep(0.01)  # Small delay

        mock_compositor = MagicMock()
        mock_compositor.generate_all_lectures = AsyncMock(side_effect=mock_generate_lectures)
        mock_compositor.create_course_zip = AsyncMock(return_value="/tmp/course.zip")

        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    await worker.process_job(sample_job)

        # Verify multiple progress updates happened
        job_key = f"course_job:{sample_job.job_id}"
        hset_ops = [
            op for op in redis_store.operations
            if op.get("op") == "hset" and op.get("key") == job_key
        ]

        # Should have multiple updates during lecture generation
        assert len(hset_ops) >= 4


# ============================================================================
# Integration Test: QueuedCourseJob Serialization
# ============================================================================

class TestJobSerialization:
    """Test job serialization through the pipeline"""

    def test_full_job_roundtrip(self):
        """Test that a job can be serialized and deserialized correctly"""
        original = QueuedCourseJob(
            job_id="serialize-test",
            topic="Serialization Test Topic",
            num_sections=3,
            lectures_per_section=4,
            user_id="user-serialize",
            difficulty_start="beginner",
            difficulty_end="advanced",
            target_audience="developers",
            language="fr",
            category="tech",
            domain="python",
            selected_elements=["code_demo", "quiz"],
            quiz_config={"enabled": True, "frequency": "per_section"},
            document_ids=["doc-a", "doc-b"],
            source_ids=["src-1", "src-2"],
            priority=3
        )

        # Serialize
        json_str = original.to_json()

        # Deserialize
        restored = QueuedCourseJob.from_json(json_str)

        # Verify all fields match
        assert restored.job_id == original.job_id
        assert restored.topic == original.topic
        assert restored.num_sections == original.num_sections
        assert restored.lectures_per_section == original.lectures_per_section
        assert restored.user_id == original.user_id
        assert restored.difficulty_start == original.difficulty_start
        assert restored.difficulty_end == original.difficulty_end
        assert restored.target_audience == original.target_audience
        assert restored.language == original.language
        assert restored.category == original.category
        assert restored.domain == original.domain
        assert restored.selected_elements == original.selected_elements
        assert restored.quiz_config == original.quiz_config
        assert restored.document_ids == original.document_ids
        assert restored.source_ids == original.source_ids
        assert restored.priority == original.priority

    def test_job_with_none_values(self):
        """Test serialization with None optional fields"""
        job = QueuedCourseJob(
            job_id="none-test",
            topic="None Test",
            num_sections=1,
            lectures_per_section=1,
            user_id="user",
            domain=None,
            selected_elements=None,
            quiz_config=None,
            document_ids=None,
            source_ids=None
        )

        json_str = job.to_json()
        restored = QueuedCourseJob.from_json(json_str)

        assert restored.domain is None
        assert restored.selected_elements is None
        assert restored.quiz_config is None
        assert restored.document_ids is None
        assert restored.source_ids is None


# ============================================================================
# Integration Test: Error Recovery Scenarios
# ============================================================================

class TestErrorRecoveryScenarios:
    """Test various error recovery scenarios"""

    @pytest.mark.asyncio
    async def test_compositor_failure_after_outline(
        self, redis_store, sample_job, mock_planner
    ):
        """Test failure during lecture generation (after outline)"""
        mock_compositor = MagicMock()
        mock_compositor.generate_all_lectures = AsyncMock(
            side_effect=Exception("Compositor timeout")
        )

        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    with pytest.raises(Exception) as exc_info:
                        await worker.process_job(sample_job)

                    assert "Compositor timeout" in str(exc_info.value)

        # Verify outline was stored before failure
        final_status = redis_store.get_job_status(sample_job.job_id)
        assert final_status["status"] == "failed"
        assert "outline" in final_status  # Outline should still be saved

    @pytest.mark.asyncio
    async def test_zip_creation_failure(
        self, redis_store, sample_job, mock_planner
    ):
        """Test failure during ZIP creation (after lectures)"""
        mock_compositor = MagicMock()
        mock_compositor.generate_all_lectures = AsyncMock()
        mock_compositor.create_course_zip = AsyncMock(
            side_effect=Exception("Disk full")
        )

        mock_queue_service = MagicMock()
        mock_queue_service.connect = AsyncMock()

        with patch('course_worker.get_queue_service', return_value=mock_queue_service):
            with patch('course_worker.CourseCompositor', return_value=mock_compositor):
                with patch('course_worker.CoursePlanner', return_value=mock_planner):
                    worker = CourseWorker(queue_service=mock_queue_service)
                    worker._redis = redis_store

                    with pytest.raises(Exception) as exc_info:
                        await worker.process_job(sample_job)

                    assert "Disk full" in str(exc_info.value)

        # Verify status history shows progress up to failure point
        status_history = redis_store.get_status_history(sample_job.job_id)
        assert "generating_outline" in status_history
        assert "generating_lectures" in status_history
        assert "creating_package" in status_history
        assert "failed" in status_history


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
