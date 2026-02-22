"""
Lecture Queue Service

Redis-based queue for distributing lecture generation jobs to workers.
Uses Redis Streams for reliable message delivery with consumer groups.
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Callable, Optional, List, Dict, Any
import redis.asyncio as aioredis
from redis.exceptions import ResponseError as RedisResponseError

from models.queue_models import (
    QueuedLectureJob,
    LectureResult,
    LectureJobStatus,
    CourseProgress,
    CourseJobStatus,
    get_course_key,
    get_course_lectures_key,
)


class LectureQueueService:
    """
    Redis Streams-based queue service for lecture generation.

    Features:
    - Reliable message delivery with consumer groups
    - Automatic acknowledgment
    - Dead letter handling for failed jobs
    - Priority support via multiple streams
    """

    STREAM_NAME = "lecture_jobs"
    CONSUMER_GROUP = "lecture_workers"
    DLQ_STREAM = "lecture_jobs_dlq"
    PENDING_TIMEOUT_MS = 300000  # 5 minutes
    STREAM_MAXLEN = 10000  # Trim streams to prevent unbounded memory growth

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/7"
        )
        self._redis: Optional[aioredis.Redis] = None
        self._is_consuming = False
        self._consumer_name: Optional[str] = None

    async def connect(self) -> None:
        """Establish connection to Redis"""
        if self._redis:
            return

        print(f"[LECTURE_QUEUE] Connecting to Redis...", flush=True)

        try:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self._redis.ping()
            print(f"[LECTURE_QUEUE] Connected to Redis", flush=True)

            # Create consumer group if it doesn't exist
            try:
                await self._redis.xgroup_create(
                    self.STREAM_NAME,
                    self.CONSUMER_GROUP,
                    id='0',
                    mkstream=True
                )
                print(f"[LECTURE_QUEUE] Created consumer group: {self.CONSUMER_GROUP}", flush=True)
            except RedisResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
                # Group already exists, that's fine

        except Exception as e:
            print(f"[LECTURE_QUEUE] Failed to connect: {e}", flush=True)
            raise

    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None
            print("[LECTURE_QUEUE] Disconnected from Redis", flush=True)

    async def publish(self, job: QueuedLectureJob) -> str:
        """
        Publish a lecture job to the queue.

        Returns the message ID.
        """
        if not self._redis:
            await self.connect()

        message_data = {
            'job_data': job.to_json(),
            'priority': str(job.priority),
            'created_at': job.created_at or datetime.utcnow().isoformat()
        }

        message_id = await self._redis.xadd(
            self.STREAM_NAME,
            message_data,
            maxlen=self.STREAM_MAXLEN,
            approximate=True,
        )

        print(f"[LECTURE_QUEUE] Published job {job.job_id} (msg: {message_id})", flush=True)
        return message_id

    async def publish_batch(self, jobs: List[QueuedLectureJob]) -> List[str]:
        """Publish multiple lecture jobs efficiently"""
        if not self._redis:
            await self.connect()

        message_ids = []
        pipe = self._redis.pipeline()

        for job in jobs:
            message_data = {
                'job_data': job.to_json(),
                'priority': str(job.priority),
                'created_at': job.created_at or datetime.utcnow().isoformat()
            }
            pipe.xadd(self.STREAM_NAME, message_data, maxlen=self.STREAM_MAXLEN, approximate=True)

        results = await pipe.execute()
        message_ids = [str(r) for r in results if r]

        print(f"[LECTURE_QUEUE] Published {len(message_ids)} jobs in batch", flush=True)
        return message_ids

    async def consume(
        self,
        handler: Callable[[QueuedLectureJob], Any],
        consumer_name: str = None,
        batch_size: int = 1,
        block_ms: int = 5000
    ) -> None:
        """
        Start consuming jobs from the queue.

        Args:
            handler: Async function to process each job
            consumer_name: Unique name for this consumer
            batch_size: Number of messages to fetch at once
            block_ms: How long to block waiting for messages
        """
        if not self._redis:
            await self.connect()

        self._consumer_name = consumer_name or f"worker_{os.getpid()}"
        self._is_consuming = True

        print(f"[LECTURE_QUEUE] Starting consumer: {self._consumer_name}", flush=True)

        while self._is_consuming:
            try:
                # Read from consumer group
                messages = await self._redis.xreadgroup(
                    self.CONSUMER_GROUP,
                    self._consumer_name,
                    {self.STREAM_NAME: '>'},
                    count=batch_size,
                    block=block_ms
                )

                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        await self._process_message(message_id, data, handler)

            except asyncio.CancelledError:
                print(f"[LECTURE_QUEUE] Consumer cancelled", flush=True)
                break
            except Exception as e:
                print(f"[LECTURE_QUEUE] Consumer error: {e}", flush=True)
                await asyncio.sleep(1)

    async def _process_message(
        self,
        message_id: str,
        data: Dict[str, str],
        handler: Callable[[QueuedLectureJob], Any]
    ) -> None:
        """Process a single message from the queue"""
        job = None
        try:
            job_data = data.get('job_data')
            if not job_data:
                print(f"[LECTURE_QUEUE] Invalid message format: {message_id}", flush=True)
                await self._redis.xack(self.STREAM_NAME, self.CONSUMER_GROUP, message_id)
                return

            job = QueuedLectureJob.from_json(job_data)
            print(f"[LECTURE_QUEUE] Processing job {job.job_id}", flush=True)

            # Call the handler
            await handler(job)

            # Acknowledge successful processing
            await self._redis.xack(self.STREAM_NAME, self.CONSUMER_GROUP, message_id)
            print(f"[LECTURE_QUEUE] Completed job {job.job_id}", flush=True)

        except Exception as e:
            print(f"[LECTURE_QUEUE] Failed to process job: {e}", flush=True)

            if job and job.retry_count < job.max_retries:
                # Retry: increment count and republish
                job.retry_count += 1
                await self.publish(job)
                await self._redis.xack(self.STREAM_NAME, self.CONSUMER_GROUP, message_id)
                print(f"[LECTURE_QUEUE] Retrying job {job.job_id} (attempt {job.retry_count})", flush=True)
            else:
                # Move to DLQ
                await self._move_to_dlq(message_id, data, str(e))
                await self._redis.xack(self.STREAM_NAME, self.CONSUMER_GROUP, message_id)

    async def _move_to_dlq(self, message_id: str, data: Dict, error: str) -> None:
        """Move failed message to dead letter queue"""
        dlq_data = {
            **data,
            'original_message_id': message_id,
            'error': error,
            'failed_at': datetime.utcnow().isoformat()
        }
        await self._redis.xadd(self.DLQ_STREAM, dlq_data, maxlen=self.STREAM_MAXLEN, approximate=True)
        print(f"[LECTURE_QUEUE] Moved to DLQ: {message_id}", flush=True)

    async def stop_consuming(self) -> None:
        """Stop the consumer loop"""
        self._is_consuming = False

    async def get_queue_length(self) -> int:
        """Get the number of pending messages"""
        if not self._redis:
            await self.connect()

        info = await self._redis.xinfo_stream(self.STREAM_NAME)
        return info.get('length', 0)

    async def get_pending_count(self) -> int:
        """Get the number of messages being processed"""
        if not self._redis:
            await self.connect()

        pending = await self._redis.xpending(self.STREAM_NAME, self.CONSUMER_GROUP)
        return pending.get('pending', 0) if pending else 0


class CourseProgressService:
    """
    Service for tracking course generation progress in Redis.
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/7"
        )
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Establish connection to Redis"""
        if self._redis:
            return

        self._redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )

    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def create_course_progress(self, progress: CourseProgress) -> None:
        """Create or update course progress in Redis"""
        if not self._redis:
            await self.connect()

        key = get_course_key(progress.course_job_id)
        await self._redis.hset(key, mapping=progress.to_dict())

        # Set expiry (7 days)
        await self._redis.expire(key, 60 * 60 * 24 * 7)

    async def get_course_progress(self, course_job_id: str) -> Optional[CourseProgress]:
        """Get course progress from Redis"""
        if not self._redis:
            await self.connect()

        key = get_course_key(course_job_id)
        data = await self._redis.hgetall(key)

        if not data:
            return None

        return CourseProgress.from_redis_hash(data)

    async def update_course_status(
        self,
        course_job_id: str,
        status: CourseJobStatus,
        error: str = None
    ) -> None:
        """Update course status"""
        if not self._redis:
            await self.connect()

        key = get_course_key(course_job_id)
        updates = {'status': status.value}

        if error:
            updates['error'] = error

        if status == CourseJobStatus.COMPLETED:
            updates['completed_at'] = datetime.utcnow().isoformat()

        await self._redis.hset(key, mapping=updates)

    async def increment_completed_lectures(self, course_job_id: str) -> int:
        """Increment completed lecture count, returns new value"""
        if not self._redis:
            await self.connect()

        key = get_course_key(course_job_id)
        return await self._redis.hincrby(key, 'completed_lectures', 1)

    # Lua script for atomic failed lecture tracking
    # Atomically: HINCRBY failed_lectures, append to failed_lecture_ids list,
    # add to failed_lecture_errors dict - all in a single Redis call
    _INCR_FAILED_LUA = """
    local key = KEYS[1]
    local lecture_id = ARGV[1]
    local error_msg = ARGV[2]

    -- Increment counter
    local count = redis.call('HINCRBY', key, 'failed_lectures', 1)

    -- Append to failed IDs list
    local ids_str = redis.call('HGET', key, 'failed_lecture_ids') or '[]'
    local ids = cjson.decode(ids_str)
    table.insert(ids, lecture_id)
    redis.call('HSET', key, 'failed_lecture_ids', cjson.encode(ids))

    -- Add to failed errors dict
    local errors_str = redis.call('HGET', key, 'failed_lecture_errors') or '{}'
    local errors = cjson.decode(errors_str)
    errors[lecture_id] = error_msg
    redis.call('HSET', key, 'failed_lecture_errors', cjson.encode(errors))

    return count
    """

    async def increment_failed_lectures(
        self,
        course_job_id: str,
        lecture_id: str,
        error: str
    ) -> int:
        """Increment failed lecture count and record error atomically via Lua script"""
        if not self._redis:
            await self.connect()

        key = get_course_key(course_job_id)

        result = await self._redis.eval(
            self._INCR_FAILED_LUA,
            1,  # number of keys
            key,  # KEYS[1]
            lecture_id,  # ARGV[1]
            error,  # ARGV[2]
        )

        return int(result)

    async def save_lecture_result(
        self,
        course_job_id: str,
        result: LectureResult
    ) -> None:
        """Save a lecture result"""
        if not self._redis:
            await self.connect()

        key = get_course_lectures_key(course_job_id)
        await self._redis.hset(key, result.lecture_id, result.to_json())

        # Set expiry (7 days)
        await self._redis.expire(key, 60 * 60 * 24 * 7)

    async def get_lecture_result(
        self,
        course_job_id: str,
        lecture_id: str
    ) -> Optional[LectureResult]:
        """Get a lecture result"""
        if not self._redis:
            await self.connect()

        key = get_course_lectures_key(course_job_id)
        data = await self._redis.hget(key, lecture_id)

        if not data:
            return None

        return LectureResult.from_json(data)

    async def get_all_lecture_results(
        self,
        course_job_id: str
    ) -> Dict[str, LectureResult]:
        """Get all lecture results for a course"""
        if not self._redis:
            await self.connect()

        key = get_course_lectures_key(course_job_id)
        data = await self._redis.hgetall(key)

        results = {}
        for lecture_id, result_json in data.items():
            results[lecture_id] = LectureResult.from_json(result_json)

        return results

    async def check_all_lectures_complete(self, course_job_id: str) -> bool:
        """Check if all lectures are complete (success or failure)"""
        progress = await self.get_course_progress(course_job_id)
        if not progress:
            return False

        return progress.is_complete

    async def save_outline(self, course_job_id: str, outline_json: str) -> None:
        """Save course outline to Redis"""
        if not self._redis:
            await self.connect()

        key = get_course_key(course_job_id)
        await self._redis.hset(key, 'outline_json', outline_json)

    async def get_outline(self, course_job_id: str) -> Optional[str]:
        """Get course outline from Redis"""
        if not self._redis:
            await self.connect()

        key = get_course_key(course_job_id)
        return await self._redis.hget(key, 'outline_json')

    async def set_final_urls(
        self,
        course_job_id: str,
        zip_url: str = None,
        final_video_url: str = None
    ) -> None:
        """Set final output URLs"""
        if not self._redis:
            await self.connect()

        key = get_course_key(course_job_id)
        updates = {}

        if zip_url:
            updates['zip_url'] = zip_url
        if final_video_url:
            updates['final_video_url'] = final_video_url

        if updates:
            await self._redis.hset(key, mapping=updates)


# Global instances
_lecture_queue: Optional[LectureQueueService] = None
_progress_service: Optional[CourseProgressService] = None


async def get_lecture_queue() -> LectureQueueService:
    """Get or create the global lecture queue instance"""
    global _lecture_queue
    if _lecture_queue is None:
        _lecture_queue = LectureQueueService()
        await _lecture_queue.connect()
    return _lecture_queue


async def get_progress_service() -> CourseProgressService:
    """Get or create the global progress service instance"""
    global _progress_service
    if _progress_service is None:
        _progress_service = CourseProgressService()
        await _progress_service.connect()
    return _progress_service
