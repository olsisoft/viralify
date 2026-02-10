"""
Object Storage Module - S3-Compatible Interface

This module provides a unified interface for object storage that works with:
- MinIO (self-hosted, default for development/staging)
- AWS S3 (production option)
- Cloudflare R2 (cost-effective production option)
- Backblaze B2 (budget production option)

The interface is 100% S3-compatible, so switching providers requires only
changing environment variables - no code changes needed.

Environment Variables:
    STORAGE_ENDPOINT: S3/MinIO endpoint (e.g., http://minio:9000 or https://s3.amazonaws.com)
    STORAGE_ACCESS_KEY: Access key ID
    STORAGE_SECRET_KEY: Secret access key
    STORAGE_REGION: AWS region (default: us-east-1)
    STORAGE_USE_SSL: Use SSL/TLS (default: true for S3, auto-detect for MinIO)
    STORAGE_PUBLIC_URL: Public URL for accessing files (e.g., https://olsitec.com/storage)

Buckets:
    - videos: Video files (.mp4)
    - presentations: Presentation files
    - thumbnails: Thumbnail images

Usage:
    from services.shared.object_storage import storage_client

    # Upload a file
    url = await storage_client.upload_file(
        bucket="videos",
        key="job123/scene_001.mp4",
        file_path="/tmp/output.mp4"
    )

    # Upload bytes directly
    url = await storage_client.upload_bytes(
        bucket="videos",
        key="job123/scene_001.mp4",
        data=video_bytes,
        content_type="video/mp4"
    )

    # Get public URL
    url = storage_client.get_public_url("videos", "job123/scene_001.mp4")

    # Delete a file
    await storage_client.delete_file("videos", "job123/scene_001.mp4")
"""

import os
import asyncio
from typing import Optional, BinaryIO
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import hashlib

# boto3 is optional - graceful fallback if not installed
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    Config = None
    ClientError = Exception


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StorageConfig:
    """Configuration for object storage."""
    endpoint: str
    access_key: str
    secret_key: str
    region: str
    use_ssl: bool
    public_url: str

    # Bucket names
    videos_bucket: str = "videos"
    presentations_bucket: str = "presentations"
    thumbnails_bucket: str = "thumbnails"


def load_storage_config() -> StorageConfig:
    """Load storage configuration from environment variables."""
    endpoint = os.getenv("STORAGE_ENDPOINT", "http://minio:9000")

    # Auto-detect SSL based on endpoint
    use_ssl_default = "true" if endpoint.startswith("https://") else "false"
    use_ssl = os.getenv("STORAGE_USE_SSL", use_ssl_default).lower() == "true"

    # Public URL for accessing files
    # Default to production domain if not set
    public_url = os.getenv("STORAGE_PUBLIC_URL", "").strip()
    if not public_url:
        public_url = "https://olsitec.com/storage"
        print(f"[STORAGE] WARNING: STORAGE_PUBLIC_URL not set, using default: {public_url}", flush=True)

    config = StorageConfig(
        endpoint=endpoint,
        access_key=os.getenv("STORAGE_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "viralify")),
        secret_key=os.getenv("STORAGE_SECRET_KEY", os.getenv("MINIO_ROOT_PASSWORD", "viralify_secret_key_change_me")),
        region=os.getenv("STORAGE_REGION", "us-east-1"),
        use_ssl=use_ssl,
        public_url=public_url.rstrip("/"),
    )

    print(f"[STORAGE] Endpoint: {config.endpoint}", flush=True)
    print(f"[STORAGE] Public URL: {config.public_url}", flush=True)
    print(f"[STORAGE] SSL: {config.use_ssl}", flush=True)

    return config


# =============================================================================
# STORAGE CLIENT
# =============================================================================

class ObjectStorageClient:
    """
    S3-compatible object storage client.

    Works with MinIO, AWS S3, Cloudflare R2, and Backblaze B2.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize the storage client."""
        self.config = config or load_storage_config()
        self._client = None
        self._initialized = False

    def _get_client(self):
        """Get or create the S3 client (lazy initialization)."""
        if not BOTO3_AVAILABLE:
            raise RuntimeError(
                "boto3 is not installed. Install it with: pip install boto3"
            )

        if self._client is None:
            # Configure client for MinIO compatibility
            s3_config = Config(
                signature_version='s3v4',
                s3={'addressing_style': 'path'},  # Required for MinIO
                retries={'max_attempts': 3, 'mode': 'standard'}
            )

            self._client = boto3.client(
                's3',
                endpoint_url=self.config.endpoint,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
                use_ssl=self.config.use_ssl,
                config=s3_config,
            )

        return self._client

    # =========================================================================
    # PUBLIC URL GENERATION
    # =========================================================================

    def get_public_url(self, bucket: str, key: str) -> str:
        """
        Get the public URL for an object.

        Args:
            bucket: Bucket name (videos, presentations, thumbnails)
            key: Object key (path within bucket)

        Returns:
            Public URL accessible from browser
        """
        # Clean the key
        key = key.lstrip("/")
        return f"{self.config.public_url}/{bucket}/{key}"

    def get_video_url(self, key: str) -> str:
        """Convenience method to get a video URL."""
        return self.get_public_url(self.config.videos_bucket, key)

    def get_presentation_url(self, key: str) -> str:
        """Convenience method to get a presentation URL."""
        return self.get_public_url(self.config.presentations_bucket, key)

    # =========================================================================
    # UPLOAD OPERATIONS
    # =========================================================================

    async def upload_file(
        self,
        bucket: str,
        key: str,
        file_path: str,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload a file to object storage.

        Args:
            bucket: Target bucket name
            key: Object key (path within bucket)
            file_path: Local file path to upload
            content_type: MIME type (auto-detected if not provided)

        Returns:
            Public URL of the uploaded file
        """
        if not content_type:
            content_type = self._guess_content_type(file_path)

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._upload_file_sync,
            bucket,
            key,
            file_path,
            content_type,
        )

        url = self.get_public_url(bucket, key)
        print(f"[STORAGE] Uploaded {file_path} -> {url}", flush=True)
        return url

    def _upload_file_sync(
        self,
        bucket: str,
        key: str,
        file_path: str,
        content_type: str,
    ):
        """Synchronous file upload."""
        client = self._get_client()

        extra_args = {
            'ContentType': content_type,
        }

        client.upload_file(file_path, bucket, key, ExtraArgs=extra_args)

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload bytes directly to object storage.

        Args:
            bucket: Target bucket name
            key: Object key
            data: Bytes to upload
            content_type: MIME type

        Returns:
            Public URL of the uploaded file
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._upload_bytes_sync,
            bucket,
            key,
            data,
            content_type,
        )

        return self.get_public_url(bucket, key)

    def _upload_bytes_sync(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str,
    ):
        """Synchronous bytes upload."""
        import io
        client = self._get_client()

        client.upload_fileobj(
            io.BytesIO(data),
            bucket,
            key,
            ExtraArgs={'ContentType': content_type},
        )

    # =========================================================================
    # UPLOAD HELPERS FOR VIDEOS
    # =========================================================================

    async def upload_video(
        self,
        file_path: str,
        job_id: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        Upload a video file to the videos bucket.

        Args:
            file_path: Local path to video file
            job_id: Job ID (used as prefix/folder)
            filename: Optional filename (defaults to basename of file_path)

        Returns:
            Public URL of the uploaded video
        """
        if filename is None:
            filename = os.path.basename(file_path)

        key = f"{job_id}/{filename}"
        return await self.upload_file(
            bucket=self.config.videos_bucket,
            key=key,
            file_path=file_path,
            content_type="video/mp4",
        )

    async def upload_scene_video(
        self,
        file_path: str,
        job_id: str,
        scene_index: int,
    ) -> str:
        """
        Upload a scene video with standardized naming.

        Args:
            file_path: Local path to scene video
            job_id: Job ID
            scene_index: Scene index (0-based)

        Returns:
            Public URL of the uploaded scene video
        """
        filename = f"scene_{scene_index:03d}.mp4"
        return await self.upload_video(file_path, job_id, filename)

    async def upload_final_video(
        self,
        file_path: str,
        job_id: str,
    ) -> str:
        """
        Upload the final composed video.

        Args:
            file_path: Local path to final video
            job_id: Job ID

        Returns:
            Public URL of the uploaded final video
        """
        return await self.upload_video(file_path, job_id, "final.mp4")

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================

    async def delete_file(self, bucket: str, key: str) -> bool:
        """
        Delete a file from object storage.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            True if deleted, False if not found
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._delete_file_sync,
            bucket,
            key,
        )

    def _delete_file_sync(self, bucket: str, key: str) -> bool:
        """Synchronous file deletion."""
        client = self._get_client()
        try:
            client.delete_object(Bucket=bucket, Key=key)
            print(f"[STORAGE] Deleted {bucket}/{key}", flush=True)
            return True
        except ClientError:
            return False

    async def delete_job_files(self, job_id: str) -> int:
        """
        Delete all files for a job (cleanup).

        Args:
            job_id: Job ID

        Returns:
            Number of files deleted
        """
        count = 0
        for bucket in [self.config.videos_bucket, self.config.presentations_bucket]:
            deleted = await self._delete_prefix(bucket, f"{job_id}/")
            count += deleted
        return count

    async def _delete_prefix(self, bucket: str, prefix: str) -> int:
        """Delete all objects with a given prefix."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._delete_prefix_sync,
            bucket,
            prefix,
        )

    def _delete_prefix_sync(self, bucket: str, prefix: str) -> int:
        """Synchronous prefix deletion."""
        client = self._get_client()
        count = 0

        try:
            # List all objects with prefix
            paginator = client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' not in page:
                    continue

                # Delete in batches
                objects = [{'Key': obj['Key']} for obj in page['Contents']]
                if objects:
                    client.delete_objects(
                        Bucket=bucket,
                        Delete={'Objects': objects}
                    )
                    count += len(objects)

        except ClientError as e:
            print(f"[STORAGE] Error deleting prefix {prefix}: {e}", flush=True)

        return count

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def file_exists(self, bucket: str, key: str) -> bool:
        """Check if a file exists in storage."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._file_exists_sync,
            bucket,
            key,
        )

    def _file_exists_sync(self, bucket: str, key: str) -> bool:
        """Synchronous existence check."""
        client = self._get_client()
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def _guess_content_type(self, file_path: str) -> str:
        """Guess content type from file extension."""
        ext = Path(file_path).suffix.lower()
        content_types = {
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.json': 'application/json',
            '.pdf': 'application/pdf',
        }
        return content_types.get(ext, 'application/octet-stream')


# =============================================================================
# FALLBACK CLIENT (No boto3)
# =============================================================================

class FallbackStorageClient:
    """
    Fallback client when boto3 is not available.

    Uses local file storage instead of object storage.
    This allows the application to run without MinIO for development.
    """

    def __init__(self):
        self.base_path = Path("/tmp/viralify/storage")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create bucket directories
        (self.base_path / "videos").mkdir(exist_ok=True)
        (self.base_path / "presentations").mkdir(exist_ok=True)
        (self.base_path / "thumbnails").mkdir(exist_ok=True)

        self.public_url = os.getenv("STORAGE_PUBLIC_URL", "https://olsitec.com/storage")

        print("[STORAGE] WARNING: Using fallback local storage (boto3 not installed)", flush=True)

    def get_public_url(self, bucket: str, key: str) -> str:
        """Get public URL for a file."""
        return f"{self.public_url}/{bucket}/{key}"

    def get_video_url(self, key: str) -> str:
        return self.get_public_url("videos", key)

    def get_presentation_url(self, key: str) -> str:
        return self.get_public_url("presentations", key)

    async def upload_file(
        self,
        bucket: str,
        key: str,
        file_path: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Copy file to local storage."""
        import shutil
        dest = self.base_path / bucket / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        return self.get_public_url(bucket, key)

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Write bytes to local storage."""
        dest = self.base_path / bucket / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return self.get_public_url(bucket, key)

    async def upload_video(self, file_path: str, job_id: str, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = os.path.basename(file_path)
        return await self.upload_file("videos", f"{job_id}/{filename}", file_path)

    async def upload_scene_video(self, file_path: str, job_id: str, scene_index: int) -> str:
        return await self.upload_video(file_path, job_id, f"scene_{scene_index:03d}.mp4")

    async def upload_final_video(self, file_path: str, job_id: str) -> str:
        return await self.upload_video(file_path, job_id, "final.mp4")

    async def delete_file(self, bucket: str, key: str) -> bool:
        path = self.base_path / bucket / key
        if path.exists():
            path.unlink()
            return True
        return False

    async def delete_job_files(self, job_id: str) -> int:
        count = 0
        for bucket in ["videos", "presentations"]:
            job_dir = self.base_path / bucket / job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir)
                count += 1
        return count

    async def file_exists(self, bucket: str, key: str) -> bool:
        return (self.base_path / bucket / key).exists()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

def create_storage_client() -> ObjectStorageClient:
    """Create the appropriate storage client based on available dependencies."""
    if BOTO3_AVAILABLE:
        return ObjectStorageClient()
    else:
        return FallbackStorageClient()


# Global singleton instance
storage_client = create_storage_client()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def upload_video(file_path: str, job_id: str, filename: Optional[str] = None) -> str:
    """Upload a video to object storage."""
    return await storage_client.upload_video(file_path, job_id, filename)


async def upload_scene_video(file_path: str, job_id: str, scene_index: int) -> str:
    """Upload a scene video to object storage."""
    return await storage_client.upload_scene_video(file_path, job_id, scene_index)


async def upload_final_video(file_path: str, job_id: str) -> str:
    """Upload the final video to object storage."""
    return await storage_client.upload_final_video(file_path, job_id)


def get_video_url(job_id: str, filename: str) -> str:
    """Get the public URL for a video."""
    return storage_client.get_video_url(f"{job_id}/{filename}")


def get_scene_url(job_id: str, scene_index: int) -> str:
    """Get the public URL for a scene video."""
    return storage_client.get_video_url(f"{job_id}/scene_{scene_index:03d}.mp4")


def get_final_url(job_id: str) -> str:
    """Get the public URL for the final video."""
    return storage_client.get_video_url(f"{job_id}/final.mp4")
