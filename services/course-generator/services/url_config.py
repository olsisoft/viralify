"""
Centralized URL Configuration Module for Course Generator

This module provides a single source of truth for all URL configurations
used in the course-generator service.

IMPORTANT: This module uses production-safe defaults. If environment variables
are not set, URLs will default to the production domain (olsitec.com) rather
than localhost to prevent broken URLs in production.

Environment Variables:
    EXTERNAL_MEDIA_URL: Public URL for media files (e.g., https://olsitec.com/media)
    EXTERNAL_PRESENTATION_URL: Public URL for presentation files (e.g., https://olsitec.com/presentations)
"""

import os
from typing import Optional


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

        video_url = url_config.convert_to_public_url("http://media-generator:8004/files/videos/xxx.mp4")
    """

    def __init__(self):
        # Public URLs (for browser access via nginx)
        self.external_media_url = os.getenv("EXTERNAL_MEDIA_URL", "").strip()
        self.external_presentation_url = os.getenv("EXTERNAL_PRESENTATION_URL", "").strip()

        # Apply production defaults if URLs are not set
        if not self.external_media_url:
            self.external_media_url = f"{PRODUCTION_DOMAIN}/media"
            self._warn_missing_env("EXTERNAL_MEDIA_URL", self.external_media_url)

        if not self.external_presentation_url:
            self.external_presentation_url = f"{PRODUCTION_DOMAIN}/presentations"
            self._warn_missing_env("EXTERNAL_PRESENTATION_URL", self.external_presentation_url)

        # Remove trailing slashes for consistency
        self.external_media_url = self.external_media_url.rstrip("/")
        self.external_presentation_url = self.external_presentation_url.rstrip("/")

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
        print(f"[URL_CONFIG] EXTERNAL_MEDIA_URL = {self.external_media_url}", flush=True)
        print(f"[URL_CONFIG] EXTERNAL_PRESENTATION_URL = {self.external_presentation_url}", flush=True)

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
                return url.replace(pattern, self.external_media_url)

        # Handle presentation-generator URLs
        internal_pres_patterns = [
            "http://presentation-generator:8006",
            "http://localhost:8006",
            "http://127.0.0.1:8006",
        ]
        for pattern in internal_pres_patterns:
            if pattern in url:
                return url.replace(pattern, self.external_presentation_url)

        # Handle file paths
        if url.startswith("/tmp/viralify/videos/"):
            filename = url.replace("/tmp/viralify/videos/", "")
            return f"{self.external_media_url}/files/videos/{filename}"

        if url.startswith("/tmp/presentations/"):
            relative_path = url.replace("/tmp/presentations/", "")
            return f"{self.external_presentation_url}/files/presentations/{relative_path}"

        # Already a public URL or unknown format - return as-is
        return url

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
# Convenience Function (drop-in replacement for convert_internal_url_to_external)
# =============================================================================

def convert_internal_url_to_external(url: Optional[str]) -> Optional[str]:
    """
    Convert internal URL to external URL.

    This is a drop-in replacement for the old convert_internal_url_to_external function.
    """
    return url_config.convert_to_public_url(url)
