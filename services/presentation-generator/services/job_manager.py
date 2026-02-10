"""
Job Manager Service

Handles job retry, cancellation, and error queue management.

Features:
- Retry individual lessons or entire jobs
- Edit lesson content before retry
- Graceful cancellation (keep completed lessons)
- Error queue with detailed context
- Rebuild final video after retry
"""

import os
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .redis_job_store import job_store, RedisConnectionError
from .video_sync import sync_final_video, sync_scene_video
from .url_config import url_config


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
    PARTIAL = "partial"  # Some lessons completed, some failed/cancelled
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


class JobManager:
    """
    Manages job lifecycle: retry, cancel, error handling.
    """

    def __init__(self):
        self.job_prefix = "v3"

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data from Redis."""
        return await job_store.get(job_id, prefix=self.job_prefix)

    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job fields."""
        return await job_store.update_fields(job_id, updates, prefix=self.job_prefix)

    # =========================================================================
    # ERROR QUEUE
    # =========================================================================

    async def get_errors(self, job_id: str) -> Dict[str, Any]:
        """
        Get all errors for a job with editable content.

        Returns:
            {
                "job_id": "...",
                "status": "...",
                "total_lessons": 10,
                "failed_count": 2,
                "errors": [
                    {
                        "scene_index": 3,
                        "title": "...",
                        "error_type": "tts_failed",
                        "error_message": "...",
                        "original_content": { ... },
                        "editable": true,
                        "retry_count": 0
                    }
                ]
            }
        """
        job = await self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}

        errors = []
        scene_statuses = job.get("scene_statuses", [])
        request_data = job.get("request", {})
        original_slides = request_data.get("slides", [])

        for i, status in enumerate(scene_statuses):
            if status.get("status") in ("failed", "error"):
                # Get original content for editing
                original_content = {}
                if i < len(original_slides):
                    slide = original_slides[i]
                    original_content = {
                        "title": slide.get("title", ""),
                        "voiceover_text": slide.get("voiceover_text", ""),
                        "type": slide.get("type", "content"),
                        "code": slide.get("code", ""),
                        "language": slide.get("language", ""),
                        "bullet_points": slide.get("bullet_points", []),
                        "diagram_description": slide.get("diagram_description", ""),
                    }

                # Check for edited content
                edited_content = job.get("edited_lessons", {}).get(str(i))
                if edited_content:
                    original_content.update(edited_content)

                error_info = {
                    "scene_index": i,
                    "title": original_content.get("title", f"Lesson {i + 1}"),
                    "error_type": status.get("error_type", "unknown"),
                    "error_message": status.get("error", status.get("error_message", "Unknown error")),
                    "original_content": original_content,
                    "editable": True,
                    "retry_count": status.get("retry_count", 0),
                    "last_retry_at": status.get("last_retry_at"),
                }
                errors.append(error_info)

        return {
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "phase": job.get("phase", "unknown"),
            "total_lessons": len(scene_statuses),
            "failed_count": len(errors),
            "errors": errors,
            "can_retry": len(errors) > 0,
        }

    async def update_lesson_content(
        self,
        job_id: str,
        scene_index: int,
        content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update lesson content before retry.

        Args:
            job_id: Job identifier
            scene_index: Lesson index (0-based)
            content: New content {title, voiceover_text, code, ...}

        Returns:
            {"success": True/False, "message": "..."}
        """
        job = await self.get_job(job_id)
        if not job:
            return {"success": False, "message": "Job not found"}

        scene_statuses = job.get("scene_statuses", [])
        if scene_index >= len(scene_statuses):
            return {"success": False, "message": f"Invalid scene index: {scene_index}"}

        # Store edited content separately
        edited_lessons = job.get("edited_lessons", {})
        edited_lessons[str(scene_index)] = {
            **content,
            "edited_at": datetime.utcnow().isoformat()
        }

        # Also update the original slides in the request
        request_data = job.get("request", {})
        slides = request_data.get("slides", [])
        if scene_index < len(slides):
            # Update the slide with new content
            for key in ["title", "voiceover_text", "code", "language", "bullet_points", "diagram_description"]:
                if key in content:
                    slides[scene_index][key] = content[key]
            request_data["slides"] = slides

        success = await self.update_job(job_id, {
            "edited_lessons": edited_lessons,
            "request": request_data,
        })

        if success:
            print(f"[JOB_MANAGER] Updated lesson {scene_index} content for job {job_id}", flush=True)
            return {
                "success": True,
                "message": f"Lesson {scene_index} content updated. Ready for retry.",
                "scene_index": scene_index
            }
        else:
            return {"success": False, "message": "Failed to update job"}

    # =========================================================================
    # RETRY
    # =========================================================================

    async def retry_lesson(
        self,
        job_id: str,
        scene_index: int,
        rebuild_final: bool = True
    ) -> Dict[str, Any]:
        """
        Retry a single failed lesson.

        Args:
            job_id: Job identifier
            scene_index: Lesson index to retry
            rebuild_final: If True, rebuild the final video after retry

        Returns:
            {"success": True/False, "message": "...", "video_url": "..."}
        """
        job = await self.get_job(job_id)
        if not job:
            return {"success": False, "message": "Job not found"}

        scene_statuses = job.get("scene_statuses", [])
        if scene_index >= len(scene_statuses):
            return {"success": False, "message": f"Invalid scene index: {scene_index}"}

        # Get the slide content (possibly edited)
        request_data = job.get("request", {})
        slides = request_data.get("slides", [])
        if scene_index >= len(slides):
            return {"success": False, "message": "Slide data not found"}

        slide_data = slides[scene_index]

        # Update scene status to pending for retry
        scene_statuses[scene_index]["status"] = "pending"
        scene_statuses[scene_index]["retry_count"] = scene_statuses[scene_index].get("retry_count", 0) + 1
        scene_statuses[scene_index]["last_retry_at"] = datetime.utcnow().isoformat()

        await self.update_job(job_id, {
            "scene_statuses": scene_statuses,
            "phase": f"retrying_lesson_{scene_index}",
        })

        print(f"[JOB_MANAGER] Retrying lesson {scene_index} for job {job_id}", flush=True)

        # Process the single scene
        try:
            from .agents.scene_graph import compiled_scene_graph, create_initial_scene_state

            initial_state = create_initial_scene_state(
                slide_data=slide_data,
                scene_index=scene_index,
                job_id=job_id,
                style=request_data.get("style", "modern"),
                content_language=request_data.get("content_language", "en"),
                voice_id=request_data.get("voice_id"),
            )

            # Run the scene graph
            final_state = await compiled_scene_graph.ainvoke(initial_state)
            scene_package = final_state.get("scene_package", {})

            # Update scene status
            scene_statuses[scene_index] = {
                "scene_index": scene_index,
                "status": scene_package.get("sync_status", "completed"),
                "sync_score": scene_package.get("sync_score", 0),
                "retry_count": scene_statuses[scene_index].get("retry_count", 1),
                "last_retry_at": datetime.utcnow().isoformat(),
            }

            # Render the scene video using compositor
            from .agents.compositor_agent import CompositorAgent
            compositor = CompositorAgent()

            scene_path = os.path.join(
                compositor.output_dir,
                f"{job_id}_scene_{scene_index:03d}.mp4"
            )

            render_success = await compositor._render_single_scene(scene_package, scene_path)

            if render_success:
                # Extract lecture metadata for explicit naming
                lecture_title = scene_package.get("title") or slide_data.get("title")
                section_index = slide_data.get("section_index")
                lecture_index = slide_data.get("lecture_index")

                # Sync to storage
                sync_success, sync_result = await sync_scene_video(
                    video_path=scene_path,
                    job_id=job_id,
                    scene_index=scene_index,
                    lecture_title=lecture_title,
                    section_index=section_index,
                    lecture_index=lecture_index,
                )

                # Determine video URL - prefer storage URL if available
                if sync_success and sync_result and sync_result.startswith("http"):
                    video_url = sync_result
                    print(f"[JOB_MANAGER] Scene {scene_index} uploaded to storage: {video_url}", flush=True)
                else:
                    video_url = self._build_scene_url(job_id, scene_index)
                    if not sync_success:
                        print(f"[JOB_MANAGER] Scene {scene_index} sync warning: {sync_result}", flush=True)
            else:
                print(f"[JOB_MANAGER] Scene {scene_index} render failed", flush=True)
                video_url = self._build_scene_url(job_id, scene_index)

            # Update scene_videos with the new video
            scene_videos = job.get("scene_videos", [])
            # Remove old entry for this scene
            scene_videos = [sv for sv in scene_videos if sv.get("scene_index") != scene_index]

            scene_videos.append({
                "scene_index": scene_index,
                "video_url": video_url,
                "status": "ready",
                "duration": scene_package.get("audio_duration", 0),
                "title": slide_data.get("title", f"Lesson {scene_index + 1}"),
                "ready_at": datetime.utcnow().isoformat(),
                "retry_count": scene_statuses[scene_index].get("retry_count", 1),
            })

            # Sort by scene_index
            scene_videos.sort(key=lambda x: x.get("scene_index", 0))

            await self.update_job(job_id, {
                "scene_statuses": scene_statuses,
                "scene_videos": scene_videos,
                "phase": "lesson_retry_complete",
            })

            print(f"[JOB_MANAGER] Lesson {scene_index} retry successful", flush=True)

            # Rebuild final video if requested
            if rebuild_final:
                rebuild_result = await self.rebuild_final_video(job_id)
                if not rebuild_result.get("success"):
                    return {
                        "success": True,
                        "message": f"Lesson {scene_index} regenerated, but final video rebuild failed",
                        "scene_index": scene_index,
                        "video_url": video_url,
                        "rebuild_error": rebuild_result.get("message")
                    }

            return {
                "success": True,
                "message": f"Lesson {scene_index} regenerated successfully",
                "scene_index": scene_index,
                "video_url": video_url,
                "final_video_url": job.get("output_url") if rebuild_final else None
            }

        except Exception as e:
            print(f"[JOB_MANAGER] Lesson {scene_index} retry failed: {e}", flush=True)
            scene_statuses[scene_index]["status"] = "failed"
            scene_statuses[scene_index]["error"] = str(e)
            scene_statuses[scene_index]["error_type"] = type(e).__name__

            await self.update_job(job_id, {
                "scene_statuses": scene_statuses,
                "phase": "retry_failed",
            })

            return {
                "success": False,
                "message": f"Retry failed: {str(e)}",
                "scene_index": scene_index
            }

    async def retry_failed_lessons(self, job_id: str) -> Dict[str, Any]:
        """
        Retry all failed lessons in a job.

        Returns:
            {"success": True/False, "retried": [...], "failed": [...]}
        """
        job = await self.get_job(job_id)
        if not job:
            return {"success": False, "message": "Job not found"}

        scene_statuses = job.get("scene_statuses", [])
        failed_indices = [
            i for i, s in enumerate(scene_statuses)
            if s.get("status") in ("failed", "error")
        ]

        if not failed_indices:
            return {"success": True, "message": "No failed lessons to retry", "retried": []}

        print(f"[JOB_MANAGER] Retrying {len(failed_indices)} failed lessons for job {job_id}", flush=True)

        retried = []
        still_failed = []

        for scene_index in failed_indices:
            result = await self.retry_lesson(job_id, scene_index, rebuild_final=False)
            if result.get("success"):
                retried.append(scene_index)
            else:
                still_failed.append({
                    "scene_index": scene_index,
                    "error": result.get("message")
                })

        # Rebuild final video after all retries
        rebuild_result = await self.rebuild_final_video(job_id)

        return {
            "success": len(still_failed) == 0,
            "message": f"Retried {len(retried)} lessons, {len(still_failed)} still failed",
            "retried": retried,
            "failed": still_failed,
            "final_video_url": rebuild_result.get("output_url") if rebuild_result.get("success") else None
        }

    async def rebuild_final_video(self, job_id: str) -> Dict[str, Any]:
        """
        Rebuild the final concatenated video from all scene videos.

        This is called after retrying lessons to create a new final video.
        """
        job = await self.get_job(job_id)
        if not job:
            return {"success": False, "message": "Job not found"}

        scene_videos = job.get("scene_videos", [])
        if not scene_videos:
            return {"success": False, "message": "No scene videos to concatenate"}

        print(f"[JOB_MANAGER] Rebuilding final video for job {job_id}", flush=True)

        try:
            from .agents.compositor_agent import CompositorAgent

            compositor = CompositorAgent()

            # Build scene packages from scene_videos
            scene_packages = []
            for sv in sorted(scene_videos, key=lambda x: x.get("scene_index", 0)):
                scene_packages.append({
                    "scene_index": sv.get("scene_index"),
                    "title": sv.get("title", ""),
                    "audio_duration": sv.get("duration", 10),
                    "total_duration": sv.get("duration", 10),
                    "primary_visual_url": "",  # Already rendered to video
                    "audio_url": "",
                    "sync_status": "synced",
                })

            # Create concat file and rebuild
            output_dir = compositor.output_dir
            concat_file = os.path.join(output_dir, f"{job_id}_rebuild_concat.txt")
            output_path = os.path.join(output_dir, f"{job_id}_final.mp4")

            # Build concat file from scene video files
            with open(concat_file, "w") as f:
                for sv in sorted(scene_videos, key=lambda x: x.get("scene_index", 0)):
                    scene_path = os.path.join(output_dir, f"{job_id}_scene_{sv.get('scene_index', 0):03d}.mp4")
                    if os.path.exists(scene_path):
                        escaped_path = scene_path.replace("'", "'\\''")
                        f.write(f"file '{escaped_path}'\n")

            # Concatenate
            await compositor._concatenate_scenes(concat_file, output_path)

            # Sync to production server (if configured)
            # Note: sync_result contains public URL on success with object storage
            sync_success, sync_result = await sync_final_video(output_path)

            # Determine output URL - prefer URL from storage if available
            if sync_success and sync_result and sync_result.startswith("http"):
                # Object storage returned a URL
                output_url = sync_result
                print(f"[JOB_MANAGER] Video uploaded to storage: {output_url}", flush=True)
            elif not sync_success:
                print(f"[JOB_MANAGER] Warning: Video sync failed: {sync_result}", flush=True)
                # Fall back to url_config
                output_url = self._build_final_url(job_id)
            else:
                # rsync succeeded, use url_config
                output_url = self._build_final_url(job_id)

            # Update job
            await self.update_job(job_id, {
                "output_url": output_url,
                "phase": "rebuild_complete",
                "status": "completed" if all(sv.get("status") == "ready" for sv in scene_videos) else "partial"
            })

            print(f"[JOB_MANAGER] Final video rebuilt: {output_url}", flush=True)

            return {
                "success": True,
                "message": "Final video rebuilt",
                "output_url": output_url
            }

        except Exception as e:
            print(f"[JOB_MANAGER] Failed to rebuild final video: {e}", flush=True)
            return {"success": False, "message": str(e)}

    # =========================================================================
    # CANCEL
    # =========================================================================

    async def cancel_job(
        self,
        job_id: str,
        keep_completed: bool = True
    ) -> Dict[str, Any]:
        """
        Cancel a job in progress.

        Args:
            job_id: Job identifier
            keep_completed: If True, keep completed lessons (status="partial")
                           If False, mark entire job as cancelled

        Returns:
            {"success": True/False, "message": "...", "completed_lessons": [...]}
        """
        job = await self.get_job(job_id)
        if not job:
            return {"success": False, "message": "Job not found"}

        current_status = job.get("status")
        if current_status in ("completed", "cancelled"):
            return {
                "success": False,
                "message": f"Job already {current_status}, cannot cancel"
            }

        print(f"[JOB_MANAGER] Cancelling job {job_id} (keep_completed={keep_completed})", flush=True)

        # Set cancellation flag
        await self.update_job(job_id, {
            "cancel_requested": True,
            "cancel_requested_at": datetime.utcnow().isoformat(),
            "cancel_keep_completed": keep_completed,
        })

        # Update scene statuses
        scene_statuses = job.get("scene_statuses", [])
        completed_lessons = []
        cancelled_lessons = []

        for i, status in enumerate(scene_statuses):
            if status.get("status") == "completed":
                completed_lessons.append(i)
            elif status.get("status") in ("pending", "processing"):
                scene_statuses[i]["status"] = "cancelled"
                scene_statuses[i]["cancelled_at"] = datetime.utcnow().isoformat()
                cancelled_lessons.append(i)

        # Determine final status
        if keep_completed and completed_lessons:
            final_status = "partial"
            message = f"Job cancelled. {len(completed_lessons)} lessons completed and available."
        else:
            final_status = "cancelled"
            message = "Job cancelled."

        await self.update_job(job_id, {
            "status": final_status,
            "phase": "cancelled",
            "scene_statuses": scene_statuses,
            "cancelled_at": datetime.utcnow().isoformat(),
        })

        # If keeping completed, rebuild partial video
        output_url = None
        if keep_completed and completed_lessons:
            rebuild_result = await self.rebuild_final_video(job_id)
            output_url = rebuild_result.get("output_url")

        return {
            "success": True,
            "message": message,
            "status": final_status,
            "completed_lessons": completed_lessons,
            "cancelled_lessons": cancelled_lessons,
            "output_url": output_url
        }

    async def is_cancelled(self, job_id: str) -> bool:
        """
        Check if a job has been requested to cancel.

        Called by workers to check if they should stop processing.
        """
        job = await self.get_job(job_id)
        if not job:
            return False
        return job.get("cancel_requested", False)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _build_scene_url(self, job_id: str, scene_index: int) -> str:
        """Build URL for a scene video using centralized URL config."""
        return url_config.build_scene_video_url(job_id, scene_index)

    def _build_final_url(self, job_id: str) -> str:
        """Build URL for the final video using centralized URL config."""
        return url_config.build_final_video_url(job_id)


# Global instance
job_manager = JobManager()
