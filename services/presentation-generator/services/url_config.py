"""
Centralized URL Configuration Module

This module provides a single source of truth for all URL configurations
used in the presentation-generator service.

IMPORTANT: This module uses production-safe defaults. If environment variables
are not set, URLs will default to the production domain (olsitec.com) rather
than localhost to prevent broken URLs in production.

Environment Variables:
    PUBLIC_BASE_URL: Base URL for presentation files (e.g., https://olsitec.com/presentations)
    PUBLIC_MEDIA_URL: Base URL for media files (e.g., https://olsitec.com/media)
    STORAGE_PUBLIC_URL: Base URL for MinIO storage (e.g., https://olsitec.com/storage)
    SERVICE_URL: Internal Docker URL for presentation-generator
    MEDIA_GENERATOR_URL: Internal Docker URL for media-generator
"""

import os
import warnings
from typing import Optional
from functools import lru_cache


# =============================================================================
# PRODUCTION DOMAIN - CHANGE THIS IF YOUR DOMAIN IS DIFFERENT
# =============================================================================
PRODUCTION_DOMAIN = "https://olsitec.com"


# =============================================================================
# URL Configuration Class
# =============================================================================

class URLConfig:
    """
    Centralized URL configuration with production-safe defaults.

    Usage:
        from services.url_config import url_config

        video_url = url_config.build_video_url("my_video.mp4")
        presentation_url = url_config.build_presentation_url("slides/my_slides.mp4")
    """

    def __init__(self):
        # Public URLs (for browser access via nginx)
        self.public_media_url = os.getenv("PUBLIC_MEDIA_URL", "").strip()
        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip()
        self.storage_public_url = os.getenv("STORAGE_PUBLIC_URL", "").strip()

        # Internal Docker URLs (for inter-service communication)
        self.internal_media_url = os.getenv("MEDIA_GENERATOR_URL", "http://media-generator:8004")
        self.internal_service_url = os.getenv("SERVICE_URL", "http://presentation-generator:8006")

        # Apply production defaults if public URLs are not set
        if not self.public_media_url:
            self.public_media_url = f"{PRODUCTION_DOMAIN}/media"
            self._warn_missing_env("PUBLIC_MEDIA_URL", self.public_media_url)

        if not self.public_base_url:
            self.public_base_url = f"{PRODUCTION_DOMAIN}/presentations"
            self._warn_missing_env("PUBLIC_BASE_URL", self.public_base_url)

        if not self.storage_public_url:
            self.storage_public_url = f"{PRODUCTION_DOMAIN}/storage"
            self._warn_missing_env("STORAGE_PUBLIC_URL", self.storage_public_url)

        # Remove trailing slashes for consistency
        self.public_media_url = self.public_media_url.rstrip("/")
        self.public_base_url = self.public_base_url.rstrip("/")
        self.storage_public_url = self.storage_public_url.rstrip("/")

        # Log configuration at startup
        self._log_config()

    def _warn_missing_env(self, var_name: str, default_value: str):
        """Log a warning about missing environment variable."""
        print(
            f"[URL_CONFIG] WARNING: {var_name} not set, using default: {default_value}",
            flush=True
        )

    def _log_config(self):
        """Log the current URL configuration."""
        print(f"[URL_CONFIG] PUBLIC_MEDIA_URL = {self.public_media_url}", flush=True)
        print(f"[URL_CONFIG] PUBLIC_BASE_URL = {self.public_base_url}", flush=True)
        print(f"[URL_CONFIG] STORAGE_PUBLIC_URL = {self.storage_public_url}", flush=True)
        print(f"[URL_CONFIG] INTERNAL_MEDIA_URL = {self.internal_media_url}", flush=True)
        print(f"[URL_CONFIG] INTERNAL_SERVICE_URL = {self.internal_service_url}", flush=True)

    # =========================================================================
    # Video URL Builders (MinIO Storage)
    # =========================================================================

    def build_video_url(self, filename: str, job_id: Optional[str] = None) -> str:
        """
        Build a public URL for a video file in MinIO storage.

        Args:
            filename: Video filename (e.g., "scene_001.mp4" or "course_xxx_final.mp4")
            job_id: Optional job ID for path construction

        Returns:
            Public URL like https://olsitec.com/storage/videos/{job_id}/{filename}
        """
        # Clean the filename
        filename = filename.lstrip("/")

        # Remove old path prefixes if present
        if filename.startswith("files/videos/"):
            filename = filename.replace("files/videos/", "")
        if filename.startswith("videos/"):
            filename = filename.replace("videos/", "")

        if job_id:
            return f"{self.storage_public_url}/videos/{job_id}/{filename}"
        else:
            # Fallback without job_id (legacy)
            return f"{self.storage_public_url}/videos/{filename}"

    def build_scene_video_url(self, job_id: str, scene_index: int) -> str:
        """
        Build a public URL for a scene video.

        Args:
            job_id: Job identifier
            scene_index: Scene index (0-based)

        Returns:
            Public URL for the scene video in MinIO
        """
        filename = f"scene_{scene_index:03d}.mp4"
        return self.build_video_url(filename, job_id)

    def build_final_video_url(self, job_id: str, title_slug: Optional[str] = None) -> str:
        """
        Build a public URL for the final composed video.

        Args:
            job_id: Job identifier
            title_slug: Optional slugified title for filename

        Returns:
            Public URL for the final video in MinIO
        """
        if title_slug:
            filename = f"course_{title_slug}_final.mp4"
        else:
            filename = f"{job_id}_final.mp4"
        return self.build_video_url(filename, job_id)

    # =========================================================================
    # Presentation URL Builders
    # =========================================================================

    def build_presentation_url(self, relative_path: str) -> str:
        """
        Build a public URL for a presentation file.

        Args:
            relative_path: Path relative to /tmp/presentations/

        Returns:
            Public URL like https://olsitec.com/presentations/files/presentations/output/xxx.mp4
        """
        # Clean the path
        relative_path = relative_path.lstrip("/")
        if relative_path.startswith("tmp/presentations/"):
            relative_path = relative_path.replace("tmp/presentations/", "")

        return f"{self.public_base_url}/files/presentations/{relative_path}"

    # =========================================================================
    # URL Conversion (Internal -> Public)
    # =========================================================================

    def convert_to_public_url(self, url: Optional[str]) -> Optional[str]:
        """
        Convert an internal Docker URL or file path to a public URL.

        This handles various input formats:
            - http://media-generator:8004/files/videos/xxx.mp4
            - http://localhost:8004/files/videos/xxx.mp4
            - http://presentation-generator:8006/files/presentations/xxx.mp4
            - /tmp/viralify/videos/xxx.mp4
            - /tmp/presentations/xxx.mp4

        Args:
            url: Internal URL or file path

        Returns:
            Public URL accessible from browser, or None if input is None
        """
        if not url:
            return None

        # Handle media-generator URLs
        internal_media_patterns = [
            "http://media-generator:8004",
            "http://localhost:8004",
            "http://127.0.0.1:8004",
        ]
        for pattern in internal_media_patterns:
            if pattern in url:
                return url.replace(pattern, self.public_media_url)

        # Handle presentation-generator URLs
        internal_pres_patterns = [
            "http://presentation-generator:8006",
            "http://localhost:8006",
            "http://127.0.0.1:8006",
        ]
        for pattern in internal_pres_patterns:
            if pattern in url:
                return url.replace(pattern, self.public_base_url)

        # Handle file paths
        if url.startswith("/tmp/viralify/videos/"):
            filename = url.replace("/tmp/viralify/videos/", "")
            return self.build_video_url(filename)

        if url.startswith("/tmp/presentations/"):
            relative_path = url.replace("/tmp/presentations/", "")
            return self.build_presentation_url(relative_path)

        # Already a public URL or unknown format - return as-is
        return url

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def is_internal_url(self, url: str) -> bool:
        """Check if a URL is an internal Docker URL that needs conversion."""
        if not url:
            return False

        internal_patterns = [
            "media-generator:",
            "presentation-generator:",
            "localhost:",
            "127.0.0.1:",
            "/tmp/viralify/",
            "/tmp/presentations/",
        ]
        return any(pattern in url for pattern in internal_patterns)


# =============================================================================
# Global Singleton Instance
# =============================================================================

# Create singleton instance at module load
url_config = URLConfig()


# =============================================================================
# Convenience Functions
# =============================================================================

def get_video_url(filename: str, job_id: Optional[str] = None) -> str:
    """Convenience function to build a video URL."""
    return url_config.build_video_url(filename, job_id)


def get_scene_url(job_id: str, scene_index: int) -> str:
    """Convenience function to build a scene video URL."""
    return url_config.build_scene_video_url(job_id, scene_index)


def get_final_url(job_id: str, title_slug: Optional[str] = None) -> str:
    """Convenience function to build a final video URL."""
    return url_config.build_final_video_url(job_id, title_slug)


def convert_url(url: Optional[str]) -> Optional[str]:
    """Convenience function to convert internal URL to public URL."""
    return url_config.convert_to_public_url(url)
