#!/usr/bin/env python3
"""
Dead Letter Queue Worker

This worker processes jobs from the DLQ (Dead Letter Queue) and retries them
with exponential backoff. After MAX_RETRY_COUNT failures, jobs are moved to
the manual intervention queue.
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.course_queue import CourseQueueService, QueuedCourseJob
from services.course_worker import CourseWorker


# Singleton worker instance for DLQ processing
_dlq_worker: CourseWorker = None


def get_dlq_worker() -> CourseWorker:
    """Get or create the DLQ worker instance."""
    global _dlq_worker
    if _dlq_worker is None:
        _dlq_worker = CourseWorker()
    return _dlq_worker


async def process_job(job: QueuedCourseJob) -> None:
    """
    Process a job from the DLQ - uses the same CourseWorker logic as main worker.
    """
    print(f"[DLQ-WORKER] Processing job {job.job_id}: {job.topic}", flush=True)

    worker = get_dlq_worker()
    await worker.process_job(job)

    print(f"[DLQ-WORKER] Job {job.job_id} completed successfully", flush=True)


async def main():
    """
    Main entry point for the DLQ worker.
    """
    print("[DLQ-WORKER] Starting Dead Letter Queue Worker...", flush=True)

    # Get retry delay from environment (default: 30 seconds)
    retry_delay = int(os.getenv("DLQ_RETRY_DELAY_SECONDS", "30"))

    queue_service = CourseQueueService()

    try:
        await queue_service.consume_dlq(
            callback=process_job,
            retry_delay_seconds=retry_delay
        )
    except KeyboardInterrupt:
        print("[DLQ-WORKER] Shutting down...", flush=True)
    finally:
        # Clean up worker resources (Redis connections, etc.)
        global _dlq_worker
        if _dlq_worker is not None:
            await _dlq_worker.close()
        await queue_service.close()


if __name__ == "__main__":
    asyncio.run(main())
