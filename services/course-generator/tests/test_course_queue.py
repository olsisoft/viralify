"""
Unit tests for CourseQueueService

Tests the RabbitMQ queue service for course generation jobs.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
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
mock_aio_pika = MagicMock()
mock_aio_pika.abc = MagicMock()
sys.modules['aio_pika'] = mock_aio_pika
sys.modules['aio_pika.abc'] = mock_aio_pika.abc

from course_queue import (
    QueuedCourseJob,
    CourseQueueService,
    get_queue_service,
)


# ============================================================================
# QueuedCourseJob Tests
# ============================================================================

class TestQueuedCourseJob:
    """Test QueuedCourseJob dataclass"""

    def test_create_basic_job(self):
        """Test creating a basic job"""
        job = QueuedCourseJob(
            job_id="test-123",
            topic="Python Programming",
            num_sections=3,
            lectures_per_section=4,
            user_id="user-456"
        )

        assert job.job_id == "test-123"
        assert job.topic == "Python Programming"
        assert job.num_sections == 3
        assert job.lectures_per_section == 4
        assert job.user_id == "user-456"

    def test_default_values(self):
        """Test default values are set correctly"""
        job = QueuedCourseJob(
            job_id="test-123",
            topic="Test Topic",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-123"
        )

        assert job.difficulty_start == "beginner"
        assert job.difficulty_end == "intermediate"
        assert job.target_audience == "general"
        assert job.language == "en"
        assert job.category == "education"
        assert job.domain is None
        assert job.selected_elements is None
        assert job.quiz_config is None
        assert job.document_ids is None
        assert job.source_ids is None
        assert job.priority == 5

    def test_full_job_creation(self):
        """Test creating a job with all fields"""
        job = QueuedCourseJob(
            job_id="full-job-123",
            topic="Advanced Machine Learning",
            num_sections=5,
            lectures_per_section=3,
            user_id="user-ml",
            difficulty_start="intermediate",
            difficulty_end="expert",
            target_audience="data scientists",
            language="fr",
            category="tech",
            domain="machine learning",
            selected_elements=["code_demo", "quiz"],
            quiz_config={"enabled": True, "frequency": "per_section"},
            document_ids=["doc-1", "doc-2"],
            source_ids=["src-1"],
            created_at="2024-01-15T10:30:00",
            priority=2
        )

        assert job.difficulty_start == "intermediate"
        assert job.difficulty_end == "expert"
        assert job.target_audience == "data scientists"
        assert job.language == "fr"
        assert job.category == "tech"
        assert job.domain == "machine learning"
        assert job.selected_elements == ["code_demo", "quiz"]
        assert job.quiz_config == {"enabled": True, "frequency": "per_section"}
        assert job.document_ids == ["doc-1", "doc-2"]
        assert job.source_ids == ["src-1"]
        assert job.created_at == "2024-01-15T10:30:00"
        assert job.priority == 2

    def test_to_json(self):
        """Test JSON serialization"""
        job = QueuedCourseJob(
            job_id="json-test",
            topic="JSON Test Topic",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-json"
        )

        json_str = job.to_json()
        data = json.loads(json_str)

        assert data["job_id"] == "json-test"
        assert data["topic"] == "JSON Test Topic"
        assert data["num_sections"] == 2
        assert "created_at" in data  # Should be auto-set

    def test_to_json_preserves_created_at(self):
        """Test that to_json preserves existing created_at"""
        job = QueuedCourseJob(
            job_id="json-test",
            topic="JSON Test",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-json",
            created_at="2024-01-01T00:00:00"
        )

        json_str = job.to_json()
        data = json.loads(json_str)

        assert data["created_at"] == "2024-01-01T00:00:00"

    def test_from_json(self):
        """Test JSON deserialization"""
        json_str = json.dumps({
            "job_id": "from-json-test",
            "topic": "Deserialized Topic",
            "num_sections": 4,
            "lectures_per_section": 5,
            "user_id": "user-deser",
            "difficulty_start": "advanced",
            "difficulty_end": "expert",
            "language": "de"
        })

        job = QueuedCourseJob.from_json(json_str)

        assert job.job_id == "from-json-test"
        assert job.topic == "Deserialized Topic"
        assert job.num_sections == 4
        assert job.lectures_per_section == 5
        assert job.user_id == "user-deser"
        assert job.difficulty_start == "advanced"
        assert job.language == "de"

    def test_roundtrip_serialization(self):
        """Test that to_json and from_json are inverse operations"""
        original = QueuedCourseJob(
            job_id="roundtrip-test",
            topic="Roundtrip Topic",
            num_sections=3,
            lectures_per_section=3,
            user_id="user-round",
            difficulty_start="intermediate",
            difficulty_end="advanced",
            quiz_config={"enabled": True},
            document_ids=["doc-a", "doc-b"]
        )

        json_str = original.to_json()
        restored = QueuedCourseJob.from_json(json_str)

        assert restored.job_id == original.job_id
        assert restored.topic == original.topic
        assert restored.num_sections == original.num_sections
        assert restored.difficulty_start == original.difficulty_start
        assert restored.quiz_config == original.quiz_config
        assert restored.document_ids == original.document_ids


# ============================================================================
# CourseQueueService Tests
# ============================================================================

class TestCourseQueueService:
    """Test CourseQueueService"""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock RabbitMQ connection"""
        connection = AsyncMock()
        connection.is_closed = False
        connection.close = AsyncMock()
        return connection

    @pytest.fixture
    def mock_channel(self):
        """Create a mock RabbitMQ channel"""
        channel = AsyncMock()
        channel.set_qos = AsyncMock()
        channel.declare_queue = AsyncMock()
        channel.default_exchange = AsyncMock()
        channel.default_exchange.publish = AsyncMock()
        return channel

    @pytest.fixture
    def mock_queue(self):
        """Create a mock RabbitMQ queue"""
        queue = AsyncMock()
        queue.consume = AsyncMock()
        queue.get = AsyncMock(return_value=None)
        queue.declaration_result = MagicMock()
        queue.declaration_result.message_count = 5
        queue.declaration_result.consumer_count = 2
        return queue

    def test_initialization_default(self):
        """Test service initialization with defaults"""
        service = CourseQueueService()

        assert service.QUEUE_NAME == "course_generation_queue"
        assert service.DLQ_NAME == "course_generation_dlq"
        assert service._connection is None
        assert service._channel is None
        assert service._is_consuming is False

    def test_initialization_custom_url(self):
        """Test service initialization with custom URL"""
        custom_url = "amqp://custom:password@host:5672/"
        service = CourseQueueService(rabbitmq_url=custom_url)

        assert service.rabbitmq_url == custom_url

    @pytest.mark.asyncio
    async def test_connect(self, mock_connection, mock_channel, mock_queue):
        """Test connecting to RabbitMQ"""
        mock_connect = AsyncMock(return_value=mock_connection)
        with patch('course_queue.aio_pika.connect_robust', mock_connect):
            mock_connection.channel = AsyncMock(return_value=mock_channel)
            mock_channel.declare_queue = AsyncMock(return_value=mock_queue)

            service = CourseQueueService()
            await service.connect()

            assert service._connection == mock_connection
            assert service._channel == mock_channel

            # Verify QoS was set
            mock_channel.set_qos.assert_called_once_with(prefetch_count=1, global_=False)

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mock_connection, mock_channel, mock_queue):
        """Test that connect doesn't reconnect if already connected"""
        mock_connect = AsyncMock(return_value=mock_connection)
        with patch('course_queue.aio_pika.connect_robust', mock_connect):
            mock_connection.channel = AsyncMock(return_value=mock_channel)
            mock_channel.declare_queue = AsyncMock(return_value=mock_queue)

            service = CourseQueueService()
            await service.connect()
            await service.connect()  # Second call

            # Should only connect once
            assert mock_connect.call_count == 1

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_connection, mock_channel, mock_queue):
        """Test disconnecting from RabbitMQ"""
        mock_connect = AsyncMock(return_value=mock_connection)
        with patch('course_queue.aio_pika.connect_robust', mock_connect):
            mock_connection.channel = AsyncMock(return_value=mock_channel)
            mock_channel.declare_queue = AsyncMock(return_value=mock_queue)

            service = CourseQueueService()
            await service.connect()
            await service.disconnect()

            mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_success(self, mock_connection, mock_channel, mock_queue):
        """Test publishing a job to the queue"""
        mock_connect = AsyncMock(return_value=mock_connection)
        with patch('course_queue.aio_pika.connect_robust', mock_connect):
            mock_connection.channel = AsyncMock(return_value=mock_channel)
            mock_channel.declare_queue = AsyncMock(return_value=mock_queue)

            service = CourseQueueService()

            job = QueuedCourseJob(
                job_id="pub-test",
                topic="Publish Test",
                num_sections=2,
                lectures_per_section=2,
                user_id="user-pub"
            )

            result = await service.publish(job)

            assert result is True
            mock_channel.default_exchange.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_failure(self, mock_connection, mock_channel, mock_queue):
        """Test handling publish failure"""
        mock_connect = AsyncMock(return_value=mock_connection)
        with patch('course_queue.aio_pika.connect_robust', mock_connect):
            mock_connection.channel = AsyncMock(return_value=mock_channel)
            mock_channel.declare_queue = AsyncMock(return_value=mock_queue)
            mock_channel.default_exchange.publish = AsyncMock(
                side_effect=Exception("Publish failed")
            )

            service = CourseQueueService()

            job = QueuedCourseJob(
                job_id="fail-test",
                topic="Fail Test",
                num_sections=2,
                lectures_per_section=2,
                user_id="user-fail"
            )

            result = await service.publish(job)

            assert result is False

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, mock_connection, mock_channel, mock_queue):
        """Test getting queue statistics"""
        mock_connect = AsyncMock(return_value=mock_connection)
        with patch('course_queue.aio_pika.connect_robust', mock_connect):
            mock_connection.channel = AsyncMock(return_value=mock_channel)
            mock_channel.declare_queue = AsyncMock(return_value=mock_queue)

            service = CourseQueueService()

            stats = await service.get_queue_stats()

            assert stats["queue_name"] == "course_generation_queue"
            assert stats["pending_jobs"] == 5
            assert stats["consumers"] == 2

    @pytest.mark.asyncio
    async def test_consume_starts_consuming(self, mock_connection, mock_channel, mock_queue):
        """Test that consume starts the consumer"""
        mock_connect = AsyncMock(return_value=mock_connection)
        with patch('course_queue.aio_pika.connect_robust', mock_connect):
            mock_connection.channel = AsyncMock(return_value=mock_channel)
            mock_channel.declare_queue = AsyncMock(return_value=mock_queue)

            service = CourseQueueService()

            callback = AsyncMock()

            # Start consuming in background
            async def start_and_stop():
                consume_task = asyncio.create_task(service.consume(callback))
                await asyncio.sleep(0.1)
                service.stop_consuming()
                try:
                    await consume_task
                except asyncio.CancelledError:
                    pass

            await start_and_stop()

            assert service._is_consuming is False
            mock_queue.consume.assert_called_once()

    def test_stop_consuming(self):
        """Test stopping the consumer"""
        service = CourseQueueService()
        service._is_consuming = True

        service.stop_consuming()

        assert service._is_consuming is False


# ============================================================================
# Singleton Tests
# ============================================================================

class TestQueueServiceSingleton:
    """Test queue service singleton pattern"""

    def test_get_queue_service_singleton(self):
        """Test that get_queue_service returns singleton"""
        # Reset singleton
        import course_queue as queue_module
        queue_module._queue_service = None

        service1 = get_queue_service()
        service2 = get_queue_service()

        assert service1 is service2

        # Cleanup
        queue_module._queue_service = None


# ============================================================================
# Message Processing Tests
# ============================================================================

class TestMessageProcessing:
    """Test message processing in the consumer"""

    @pytest.fixture
    def mock_message(self):
        """Create a mock incoming message"""
        message = AsyncMock()
        message.headers = {"job_id": "msg-test"}
        message.body = json.dumps({
            "job_id": "msg-test",
            "topic": "Message Test",
            "num_sections": 2,
            "lectures_per_section": 2,
            "user_id": "user-msg"
        }).encode()

        # Mock context manager for process()
        process_cm = AsyncMock()
        process_cm.__aenter__ = AsyncMock(return_value=None)
        process_cm.__aexit__ = AsyncMock(return_value=None)
        message.process = MagicMock(return_value=process_cm)

        message.ack = AsyncMock()
        message.nack = AsyncMock()

        return message

    @pytest.mark.asyncio
    async def test_message_deserialization(self, mock_message):
        """Test that messages are correctly deserialized"""
        job = QueuedCourseJob.from_json(mock_message.body.decode())

        assert job.job_id == "msg-test"
        assert job.topic == "Message Test"
        assert job.num_sections == 2


# ============================================================================
# Priority Queue Tests
# ============================================================================

class TestPriorityQueue:
    """Test priority queue functionality"""

    def test_job_priority_default(self):
        """Test default priority value"""
        job = QueuedCourseJob(
            job_id="priority-test",
            topic="Priority Test",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-prio"
        )

        assert job.priority == 5

    def test_job_priority_custom(self):
        """Test custom priority value"""
        high_priority_job = QueuedCourseJob(
            job_id="high-priority",
            topic="High Priority",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-high",
            priority=1
        )

        low_priority_job = QueuedCourseJob(
            job_id="low-priority",
            topic="Low Priority",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-low",
            priority=10
        )

        assert high_priority_job.priority < low_priority_job.priority

    def test_priority_in_json(self):
        """Test that priority is preserved in JSON serialization"""
        job = QueuedCourseJob(
            job_id="priority-json",
            topic="Priority JSON",
            num_sections=2,
            lectures_per_section=2,
            user_id="user-prio-json",
            priority=3
        )

        json_str = job.to_json()
        restored = QueuedCourseJob.from_json(json_str)

        assert restored.priority == 3


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling in queue service"""

    @pytest.mark.asyncio
    async def test_connect_error(self):
        """Test handling connection errors"""
        with patch('course_queue.aio_pika.connect_robust') as mock_connect:
            mock_connect.side_effect = Exception("Connection refused")

            service = CourseQueueService()

            with pytest.raises(Exception) as excinfo:
                await service.connect()

            assert "Connection refused" in str(excinfo.value)

    def test_from_json_invalid_json(self):
        """Test handling invalid JSON"""
        with pytest.raises(json.JSONDecodeError):
            QueuedCourseJob.from_json("invalid json {")

    def test_from_json_missing_required_fields(self):
        """Test handling missing required fields"""
        with pytest.raises(TypeError):
            QueuedCourseJob.from_json(json.dumps({"job_id": "only-id"}))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
