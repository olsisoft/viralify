"""
FFmpeg Resource Manager

Manages FFmpeg process concurrency and memory to prevent OOM kills.

Key Features:
- Global semaphore limits concurrent FFmpeg processes
- Memory cleanup after each process
- Process tracking for debugging
- Timeout management

Usage:
    from services.ffmpeg_resource_manager import ffmpeg_manager

    async with ffmpeg_manager.acquire("job_123", "segment_1"):
        # FFmpeg work here - only N processes can run concurrently
        await run_ffmpeg(...)
    # Resources automatically released and cleaned up
"""

import os
import gc
import asyncio
import psutil
from datetime import datetime
from typing import Dict, Optional, Set
from contextlib import asynccontextmanager


class FFmpegResourceManager:
    """
    Manages FFmpeg resources across all workers.

    Prevents memory exhaustion by:
    1. Limiting concurrent FFmpeg processes via semaphore
    2. Tracking active processes
    3. Forcing garbage collection after each process
    4. Monitoring memory usage
    """

    def __init__(self):
        # Maximum concurrent FFmpeg processes (configurable via env)
        max_concurrent = int(os.getenv("FFMPEG_MAX_CONCURRENT", "2"))
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Track active processes for debugging
        self._active_processes: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

        # Memory thresholds
        self._memory_warning_percent = 80
        self._memory_critical_percent = 90

        print(f"[FFMPEG_MANAGER] Initialized with max {max_concurrent} concurrent processes", flush=True)

    @asynccontextmanager
    async def acquire(self, job_id: str, segment_id: str = "unknown"):
        """
        Acquire a slot for FFmpeg processing.

        Usage:
            async with ffmpeg_manager.acquire("job_123", "segment_1"):
                await run_ffmpeg(...)
        """
        process_key = f"{job_id}:{segment_id}"

        # Wait for available slot
        await self._wait_for_slot(process_key)

        try:
            yield
        finally:
            # Release slot and cleanup
            await self._release_slot(process_key)

    async def _wait_for_slot(self, process_key: str):
        """Wait for a semaphore slot and register the process."""
        # Check memory before acquiring
        await self._check_memory_pressure()

        # Log queue position if waiting
        if self._semaphore.locked():
            async with self._lock:
                queue_position = len(self._active_processes) + 1
            print(f"[FFMPEG_MANAGER] {process_key} waiting in queue (position ~{queue_position})", flush=True)

        # Acquire semaphore (blocks if all slots used)
        await self._semaphore.acquire()

        # Register active process
        async with self._lock:
            self._active_processes[process_key] = {
                "started_at": datetime.utcnow().isoformat(),
                "memory_at_start": self._get_memory_usage_percent()
            }

        active_count = len(self._active_processes)
        print(f"[FFMPEG_MANAGER] {process_key} acquired slot ({active_count} active)", flush=True)

    async def _release_slot(self, process_key: str):
        """Release semaphore slot and cleanup resources."""
        # Unregister process
        async with self._lock:
            process_info = self._active_processes.pop(process_key, {})

        # Force garbage collection
        self._cleanup_memory()

        # Release semaphore
        self._semaphore.release()

        # Log completion
        memory_now = self._get_memory_usage_percent()
        memory_start = process_info.get("memory_at_start", 0)
        memory_delta = memory_now - memory_start

        active_count = len(self._active_processes)
        print(f"[FFMPEG_MANAGER] {process_key} released slot ({active_count} active, "
              f"memory: {memory_now:.1f}% [{memory_delta:+.1f}%])", flush=True)

    async def _check_memory_pressure(self):
        """Check memory and wait if under pressure."""
        memory_percent = self._get_memory_usage_percent()

        if memory_percent > self._memory_critical_percent:
            print(f"[FFMPEG_MANAGER] CRITICAL memory pressure ({memory_percent:.1f}%), "
                  f"forcing GC and waiting...", flush=True)
            self._cleanup_memory()
            await asyncio.sleep(5)  # Wait for other processes to finish

        elif memory_percent > self._memory_warning_percent:
            print(f"[FFMPEG_MANAGER] WARNING: memory at {memory_percent:.1f}%", flush=True)
            self._cleanup_memory()

    def _cleanup_memory(self):
        """Force garbage collection to free memory."""
        # Run garbage collection
        collected = gc.collect()

        # Try to free unreferenced objects
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)

        if collected > 0:
            print(f"[FFMPEG_MANAGER] GC collected {collected} objects", flush=True)

    def _get_memory_usage_percent(self) -> float:
        """Get current memory usage percentage."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception:
            return 0.0

    def get_status(self) -> dict:
        """Get current manager status for debugging."""
        return {
            "active_processes": len(self._active_processes),
            "active_keys": list(self._active_processes.keys()),
            "memory_percent": self._get_memory_usage_percent(),
            "semaphore_locked": self._semaphore.locked(),
        }


# Global singleton instance
ffmpeg_manager = FFmpegResourceManager()
