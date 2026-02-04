"""
Video Sync Module

Automatically syncs final videos to the production server.
Configured via environment variables for easy deployment on new workers.
"""

import os
import subprocess
import asyncio
from typing import Optional, Tuple


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


async def sync_final_video(video_path: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to sync a final video to production.

    Args:
        video_path: Full path to the video file

    Returns:
        Tuple of (success, error_message)
    """
    syncer = get_syncer()

    # Try rsync first, fall back to scp
    success, error = await syncer.sync_video(video_path)

    if not success and "rsync" in str(error).lower():
        # rsync not available, try scp
        syncer.log("rsync failed, trying scp fallback")
        success, error = await syncer.sync_video_scp(video_path)

    return success, error
