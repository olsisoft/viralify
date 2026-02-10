"""
Video Sync Module

Handles video upload to object storage (MinIO/S3) or legacy rsync to production server.

Priority:
1. Object Storage (MinIO/S3) if STORAGE_ENABLED=true
2. Legacy rsync/scp if VIDEO_SYNC_ENABLED=true
3. No sync (local only) if neither is enabled

Environment Variables:
    STORAGE_ENABLED: Use object storage (recommended for production)
    VIDEO_SYNC_ENABLED: Use legacy rsync/scp (deprecated)
"""

import os
import subprocess
import asyncio
from typing import Optional, Tuple

# Object storage is optional - graceful fallback if not available
try:
    from .object_storage import storage_client
    OBJECT_STORAGE_AVAILABLE = True
except ImportError:
    OBJECT_STORAGE_AVAILABLE = False
    storage_client = None


class VideoSyncConfig:
    """Configuration for video sync to production server"""

    def __init__(self):
        # Production server configuration
        self.enabled = os.getenv("VIDEO_SYNC_ENABLED", "false").lower() == "true"
        self.host = os.getenv("VIDEO_SYNC_HOST", "")  # e.g., "51.79.65.199" or "olsitec.com"
        self.user = os.getenv("VIDEO_SYNC_USER", "ubuntu")
        self.remote_path = os.getenv("VIDEO_SYNC_PATH", "/var/lib/docker/volumes/repo_media_generator_videos/_data")
        self.ssh_key = os.getenv("VIDEO_SYNC_SSH_KEY", "")  # Optional: path to SSH key
        self.timeout = int(os.getenv("VIDEO_SYNC_TIMEOUT", "300"))  # 5 minutes default

    def is_configured(self) -> bool:
        """Check if sync is properly configured"""
        return self.enabled and bool(self.host)

    def get_ssh_options(self) -> list:
        """Get SSH options for rsync/scp"""
        options = [
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes"
        ]
        if self.ssh_key:
            options.extend(["-i", self.ssh_key])
        return options


class VideoSyncer:
    """Handles syncing videos to production server"""

    def __init__(self):
        self.config = VideoSyncConfig()
        self._log_prefix = "[VIDEO_SYNC]"

    def log(self, message: str):
        """Log a message"""
        print(f"{self._log_prefix} {message}", flush=True)

    async def sync_video(self, local_path: str) -> Tuple[bool, Optional[str]]:
        """
        Sync a video file to the production server.

        Args:
            local_path: Full path to the local video file

        Returns:
            Tuple of (success, error_message)
        """
        if not self.config.is_configured():
            self.log("Sync disabled or not configured, skipping")
            return True, None

        if not os.path.exists(local_path):
            error = f"Video file not found: {local_path}"
            self.log(error)
            return False, error

        filename = os.path.basename(local_path)
        remote_dest = f"{self.config.user}@{self.config.host}:{self.config.remote_path}/{filename}"

        self.log(f"Syncing {filename} to {self.config.host}")

        try:
            # Build rsync command
            ssh_opts = " ".join(self.config.get_ssh_options())
            cmd = [
                "rsync",
                "-avz",
                "--progress",
                "-e", f"ssh {ssh_opts}",
                local_path,
                remote_dest
            ]

            # Run rsync asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                error = f"Sync timed out after {self.config.timeout}s"
                self.log(error)
                return False, error

            if process.returncode == 0:
                self.log(f"Successfully synced {filename}")
                return True, None
            else:
                error = f"Sync failed: {stderr.decode()}"
                self.log(error)
                return False, error

        except Exception as e:
            error = f"Sync error: {str(e)}"
            self.log(error)
            return False, error

    async def sync_video_scp(self, local_path: str) -> Tuple[bool, Optional[str]]:
        """
        Fallback: Sync using scp if rsync is not available.

        Args:
            local_path: Full path to the local video file

        Returns:
            Tuple of (success, error_message)
        """
        if not self.config.is_configured():
            return True, None

        if not os.path.exists(local_path):
            return False, f"Video file not found: {local_path}"

        filename = os.path.basename(local_path)
        remote_dest = f"{self.config.user}@{self.config.host}:{self.config.remote_path}/{filename}"

        self.log(f"Syncing {filename} via scp to {self.config.host}")

        try:
            cmd = ["scp"] + self.config.get_ssh_options() + [local_path, remote_dest]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return False, f"SCP timed out after {self.config.timeout}s"

            if process.returncode == 0:
                self.log(f"Successfully synced {filename} via scp")
                return True, None
            else:
                return False, f"SCP failed: {stderr.decode()}"

        except Exception as e:
            return False, f"SCP error: {str(e)}"


# Global syncer instance
_syncer: Optional[VideoSyncer] = None


def get_syncer() -> VideoSyncer:
    """Get or create the global video syncer instance"""
    global _syncer
    if _syncer is None:
        _syncer = VideoSyncer()
    return _syncer


async def sync_final_video(
    video_path: str,
    job_id: Optional[str] = None,
    course_title: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Sync a final video to production storage.

    Priority:
    1. Object Storage (MinIO/S3) if STORAGE_ENABLED=true
    2. Legacy rsync/scp if VIDEO_SYNC_ENABLED=true
    3. Skip sync if neither is enabled

    Args:
        video_path: Full path to the video file
        job_id: Optional job ID for organizing in object storage
        course_title: Optional course title for explicit naming

    Returns:
        Tuple of (success, error_message_or_url)
        On success with object storage, error_message contains the public URL
    """
    # Check if object storage is enabled (recommended)
    storage_enabled = os.getenv("STORAGE_ENABLED", "false").lower() == "true"

    if storage_enabled and OBJECT_STORAGE_AVAILABLE:
        return await sync_to_object_storage(video_path, job_id, course_title)

    # Fall back to legacy rsync/scp
    syncer = get_syncer()

    if not syncer.config.is_configured():
        print("[VIDEO_SYNC] Neither object storage nor rsync configured, skipping sync", flush=True)
        return True, None

    # Try rsync first, fall back to scp
    success, error = await syncer.sync_video(video_path)

    if not success and "rsync" in str(error).lower():
        # rsync not available, try scp
        syncer.log("rsync failed, trying scp fallback")
        success, error = await syncer.sync_video_scp(video_path)

    return success, error


async def sync_to_object_storage(
    video_path: str,
    job_id: Optional[str] = None,
    course_title: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Upload a video to object storage (MinIO/S3).

    Args:
        video_path: Full path to the video file
        job_id: Optional job ID for organizing files
        course_title: Optional course title for explicit naming of final video

    Returns:
        Tuple of (success, public_url_or_error)
    """
    if not OBJECT_STORAGE_AVAILABLE or storage_client is None:
        return False, "Object storage not available (boto3 not installed)"

    try:
        filename = os.path.basename(video_path)

        # Extract job_id from filename if not provided
        # Format: {job_id}_final.mp4 or {job_id}_scene_001.mp4
        if job_id is None:
            parts = filename.rsplit("_", 1)
            if len(parts) > 1:
                job_id = parts[0]
            else:
                job_id = "unknown"

        # Upload to object storage
        if "_final" in filename:
            url = await storage_client.upload_final_video(video_path, job_id, course_title)
        elif "_scene_" in filename:
            # Extract scene index from filename
            import re
            match = re.search(r"_scene_(\d+)", filename)
            if match:
                scene_index = int(match.group(1))
                url = await storage_client.upload_scene_video(
                    file_path=video_path,
                    job_id=job_id,
                    scene_index=scene_index,
                )
            else:
                url = await storage_client.upload_video(video_path, job_id, filename)
        else:
            url = await storage_client.upload_video(video_path, job_id, filename)

        print(f"[VIDEO_SYNC] Uploaded to object storage: {url}", flush=True)
        return True, url

    except Exception as e:
        error_msg = f"Object storage upload failed: {str(e)}"
        print(f"[VIDEO_SYNC] {error_msg}", flush=True)
        return False, error_msg


async def sync_scene_video(
    video_path: str,
    job_id: str,
    scene_index: int,
    lecture_title: Optional[str] = None,
    section_index: Optional[int] = None,
    lecture_index: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Sync a scene/lecture video to production storage.

    When lecture metadata is provided, the video is stored with explicit naming:
        videos/{job_id}/{section_idx:02d}_{lecture_idx:02d}_{lecture_slug}/lecture.mp4

    Otherwise, fallback naming is used:
        videos/{job_id}/scene_{scene_index:03d}.mp4

    Args:
        video_path: Full path to the scene video
        job_id: Job ID (also course ID)
        scene_index: Scene index (0-based, used as fallback)
        lecture_title: Optional lecture title for explicit naming
        section_index: Optional section index (1-based)
        lecture_index: Optional lecture index within section (1-based)

    Returns:
        Tuple of (success, public_url_or_error)
    """
    storage_enabled = os.getenv("STORAGE_ENABLED", "false").lower() == "true"

    if storage_enabled and OBJECT_STORAGE_AVAILABLE and storage_client is not None:
        try:
            url = await storage_client.upload_scene_video(
                file_path=video_path,
                job_id=job_id,
                scene_index=scene_index,
                lecture_title=lecture_title,
                section_index=section_index,
                lecture_index=lecture_index,
            )
            if lecture_title:
                print(f"[VIDEO_SYNC] Lecture '{lecture_title}' uploaded: {url}", flush=True)
            else:
                print(f"[VIDEO_SYNC] Scene {scene_index} uploaded: {url}", flush=True)
            return True, url
        except Exception as e:
            return False, f"Scene upload failed: {str(e)}"

    # Fall back to legacy sync
    return await sync_final_video(video_path, job_id)
