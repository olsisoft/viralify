"""
Local Avatar Service - Unified avatar animation using open-source models.

Pipeline:
1. FOMM: Add body/gesture animation to static image
2. Wav2Lip: Add accurate lip-sync to the animated video
3. D-ID: Fallback if local processing fails

This provides a cost-effective alternative to D-ID while maintaining quality.
"""

import os
import asyncio
import logging
import tempfile
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from enum import Enum
import httpx
from PIL import Image
import io

logger = logging.getLogger(__name__)


class AnimationProvider(str, Enum):
    """Available animation providers."""
    LOCAL = "local"           # Wav2Lip + FOMM (requires GPU)
    REPLICATE = "replicate"   # Replicate API (serverless GPU)
    HUNYUAN = "hunyuan"       # HunyuanVideo-Avatar on RunPod (full-body, cheap)
    DID = "d-id"              # D-ID API (expensive fallback)
    HYBRID = "hybrid"         # Try Hunyuan/Replicate first, then D-ID


class AvatarQuality(str, Enum):
    """Avatar animation quality modes for cost optimization.

    Cost comparison per 15s video:
    - DRAFT: ~$0.002 (SadTalker - fast preview, head motion only)
    - PREVIEW: ~$0.01 (SadTalker + GFPGAN face enhancement)
    - STANDARD: ~$0.10 (HunyuanVideo-Avatar - full body, good quality)
    - FINAL: ~$0.20 (HunyuanVideo-Avatar HIGH - full body, best quality)
    """
    DRAFT = "draft"       # SadTalker basic - fast, cheap
    PREVIEW = "preview"   # SadTalker + enhancement - good quality
    STANDARD = "standard" # HunyuanVideo-Avatar - full body
    FINAL = "final"       # HunyuanVideo-Avatar HIGH - premium full-body


class LocalAvatarService:
    """
    Unified avatar animation service with local processing and cloud fallback.

    Processing Pipeline:
    1. Pre-process: Remove background from avatar image (rembg)
    2. Body Animation: Apply gesture/movement with FOMM
    3. Lip-Sync: Synchronize lips with audio using Wav2Lip
    4. Post-process: Combine with background, add effects

    Fallback: If local processing fails, use D-ID API
    """

    def __init__(
        self,
        did_api_key: Optional[str] = None,
        replicate_api_key: Optional[str] = None,
        prefer_local: Optional[bool] = None,
        output_dir: str = "/tmp/viralify/avatars",
        cache_ttl_hours: int = 24
    ):
        self.did_api_key = did_api_key or os.getenv("DID_API_KEY")
        self.replicate_api_key = replicate_api_key or os.getenv("REPLICATE_API_KEY")
        # Check environment variable for local preference
        if prefer_local is None:
            env_pref = os.getenv("USE_LOCAL_AVATAR", "true").lower()
            self.prefer_local = env_pref in ("true", "1", "yes")
        else:
            self.prefer_local = prefer_local
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache settings
        self.cache_dir = Path(output_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self._cache_index: Dict[str, Dict] = {}
        self._load_cache_index()

        # Lazy load services
        self._wav2lip = None
        self._fomm = None
        self._did = None
        self._replicate = None
        self._bg_remover = None
        self._face_swap = None
        self._hunyuan = None

    @property
    def wav2lip(self):
        if self._wav2lip is None:
            from services.wav2lip_service import get_wav2lip_service
            self._wav2lip = get_wav2lip_service()
        return self._wav2lip

    @property
    def fomm(self):
        if self._fomm is None:
            from services.fomm_service import get_fomm_service
            self._fomm = get_fomm_service()
        return self._fomm

    @property
    def did_provider(self):
        if self._did is None and self.did_api_key:
            from providers.did_provider import DIDProvider
            self._did = DIDProvider(self.did_api_key)
        return self._did

    @property
    def replicate(self):
        if self._replicate is None and self.replicate_api_key:
            from services.replicate_service import get_replicate_service
            self._replicate = get_replicate_service()
        return self._replicate

    @property
    def bg_remover(self):
        if self._bg_remover is None:
            from services.background_remover import get_background_remover
            self._bg_remover = get_background_remover()
        return self._bg_remover

    @property
    def face_swap(self):
        if self._face_swap is None:
            from services.face_swap_service import get_face_swap_service
            self._face_swap = get_face_swap_service()
        return self._face_swap

    @property
    def hunyuan(self):
        if self._hunyuan is None:
            from services.hunyuan_avatar_service import get_hunyuan_avatar_service
            self._hunyuan = get_hunyuan_avatar_service()
        return self._hunyuan

    # === Cache Management ===

    def _load_cache_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r") as f:
                    self._cache_index = json.load(f)
                logger.info(f"[Cache] Loaded {len(self._cache_index)} cached entries")
            except Exception as e:
                logger.warning(f"[Cache] Failed to load index: {e}")
                self._cache_index = {}

    def _save_cache_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, "w") as f:
                json.dump(self._cache_index, f, indent=2)
        except Exception as e:
            logger.warning(f"[Cache] Failed to save index: {e}")

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of file content."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""

    def _compute_cache_key(
        self,
        image_path: str,
        audio_path: str,
        quality: str,
        face_swap_image: Optional[str] = None
    ) -> str:
        """Generate cache key from input parameters."""
        image_hash = self._compute_file_hash(image_path)
        audio_hash = self._compute_file_hash(audio_path)
        face_hash = self._compute_file_hash(face_swap_image) if face_swap_image else ""

        key_data = f"{image_hash}:{audio_hash}:{quality}:{face_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """Check if valid cached result exists."""
        if cache_key not in self._cache_index:
            return None

        entry = self._cache_index[cache_key]
        video_path = entry.get("video_path")
        created_at = entry.get("created_at", 0)

        # Check TTL
        if time.time() - created_at > self.cache_ttl_seconds:
            logger.info(f"[Cache] Entry {cache_key} expired")
            del self._cache_index[cache_key]
            self._save_cache_index()
            return None

        # Check file exists
        if video_path and os.path.exists(video_path):
            logger.info(f"[Cache] HIT - returning cached video for {cache_key}")
            return video_path

        logger.info(f"[Cache] Entry {cache_key} file missing")
        del self._cache_index[cache_key]
        self._save_cache_index()
        return None

    def _cache_result(self, cache_key: str, video_path: str, quality: str):
        """Store result in cache."""
        # Copy video to cache directory with permanent name
        cache_video_path = str(self.cache_dir / f"{cache_key}.mp4")
        try:
            import shutil
            shutil.copy2(video_path, cache_video_path)

            self._cache_index[cache_key] = {
                "video_path": cache_video_path,
                "created_at": time.time(),
                "quality": quality
            }
            self._save_cache_index()
            logger.info(f"[Cache] Stored result for {cache_key}")
        except Exception as e:
            logger.warning(f"[Cache] Failed to store result: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache_index)
        total_size = 0
        valid_entries = 0

        for key, entry in list(self._cache_index.items()):
            video_path = entry.get("video_path")
            if video_path and os.path.exists(video_path):
                total_size += os.path.getsize(video_path)
                valid_entries += 1

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_ttl_hours": self.cache_ttl_seconds / 3600
        }

    def clear_cache(self) -> Dict[str, int]:
        """Clear all cached videos."""
        cleared = 0
        for key, entry in list(self._cache_index.items()):
            video_path = entry.get("video_path")
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    cleared += 1
                except Exception:
                    pass
        self._cache_index = {}
        self._save_cache_index()
        logger.info(f"[Cache] Cleared {cleared} cached videos")
        return {"cleared": cleared}

    def get_available_providers(self) -> Dict[str, bool]:
        """Check which providers are available."""
        return {
            "wav2lip": self.wav2lip.is_available(),
            "fomm": self.fomm.is_available(),
            "replicate": self.replicate_api_key is not None,
            "hunyuan": self.hunyuan.is_available() if self.hunyuan else False,
            "did": self.did_api_key is not None,
            "face_swap": self.face_swap.is_available() if self.face_swap else False
        }

    async def generate_avatar_video(
        self,
        source_image: str,
        audio_path: str,
        provider: AnimationProvider = AnimationProvider.HYBRID,
        gesture_type: str = "talking",
        remove_background: bool = True,
        output_path: Optional[str] = None,
        face_swap_image: Optional[str] = None,
        face_swap_hair_source: str = "user",
        quality: str = "final"
    ) -> Dict[str, Any]:
        """
        Generate animated avatar video with lip-sync.

        Args:
            source_image: Path/URL to avatar image
            audio_path: Path to audio file for lip-sync
            provider: Which provider to use (local, d-id, hybrid)
            gesture_type: Type of body gesture (talking, presenting, nodding)
            remove_background: Whether to remove image background
            output_path: Optional output video path
            face_swap_image: Optional URL/path to user's face for face swap
            face_swap_hair_source: Hair source for face swap ('user' or 'target')
            quality: Quality mode - 'draft' (~$0.002), 'preview' (~$0.01), 'final' (~$2.80)

        Returns:
            Dict with video_url, provider_used, duration, status, cost_estimate
        """
        import uuid

        # Map quality mode to cost estimates
        cost_estimates = {
            "draft": 0.002,    # SadTalker basic
            "preview": 0.01,   # SadTalker + enhancement
            "standard": 0.10,  # HunyuanVideo-Avatar (full-body)
            "final": 0.20      # HunyuanVideo-Avatar HIGH (full-body, best)
        }

        result = {
            "video_url": None,
            "provider_used": None,
            "duration": 0,
            "status": "pending",
            "error": None,
            "quality": quality,
            "cost_estimate": cost_estimates.get(quality, 2.80)
        }

        logger.info(f"[LocalAvatar] Generating with quality={quality} (est. ${result['cost_estimate']:.3f})")

        try:
            # Generate output path
            if not output_path:
                output_path = str(self.output_dir / f"avatar_{uuid.uuid4().hex[:8]}.mp4")

            # Download source image if URL
            local_image = await self._ensure_local_file(source_image, "image")
            local_audio = await self._ensure_local_file(audio_path, "audio")

            if not local_image or not local_audio:
                raise ValueError("Failed to access source files")

            # Download face swap image early for cache key computation
            local_face_image = None
            if face_swap_image:
                local_face_image = await self._ensure_local_file(face_swap_image, "face")

            # Check cache BEFORE expensive processing
            cache_key = self._compute_cache_key(
                local_image, local_audio, quality, local_face_image
            )
            cached_video = self._get_cached_result(cache_key)
            if cached_video:
                result["video_url"] = cached_video
                result["provider_used"] = "cache"
                result["status"] = "completed"
                result["cost_estimate"] = 0.0  # No cost for cached result!
                logger.info(f"[LocalAvatar] Cache HIT - saved ${cost_estimates.get(quality, 2.80):.3f}")
                return result

            logger.info(f"[LocalAvatar] Cache MISS - generating new video")

            # Step 0.5: Optimize image (resize/compress) to reduce API costs
            # Use smaller max_size for draft/preview, larger for final
            max_image_size = 768 if quality in ["draft", "preview"] else 1024
            local_image = await self._optimize_image(local_image, max_size=max_image_size)

            # Step 0: Apply face swap if requested (BEFORE background removal)
            if face_swap_image and local_face_image:
                try:
                    from services.face_swap_service import HairSource
                    logger.info(f"[LocalAvatar] Applying face swap (hair_source: {face_swap_hair_source})...")

                    if self.face_swap and self.face_swap.is_available():
                        hair_source = HairSource.USER if face_swap_hair_source == "user" else HairSource.TARGET

                        if local_face_image:
                            swapped_image = await self.face_swap.swap_face(
                                target_image=local_image,
                                swap_image=local_face_image,
                                gender="auto",
                                hair_source=hair_source,
                                upscale=True
                            )

                            if swapped_image:
                                local_image = swapped_image
                                logger.info(f"[LocalAvatar] Face swap completed: {local_image}")
                            else:
                                logger.warning("[LocalAvatar] Face swap failed, using original image")
                        else:
                            logger.warning("[LocalAvatar] Failed to download face image, skipping face swap")
                    else:
                        logger.warning("[LocalAvatar] Face swap service not available, skipping")
                except Exception as e:
                    logger.warning(f"[LocalAvatar] Face swap error: {e}, using original image")

            # Step 1: Remove background if requested
            if remove_background:
                try:
                    logger.info("[LocalAvatar] Attempting background removal...")
                    processed_image = await self.bg_remover.remove_background(local_image)
                    if processed_image:
                        local_image = processed_image
                        logger.info("[LocalAvatar] Background removed successfully")
                except Exception as e:
                    logger.warning(f"[LocalAvatar] Background removal failed: {e}, using original image")

            # Step 2: Try LOCAL processing (Wav2Lip + FOMM - requires GPU)
            if provider == AnimationProvider.LOCAL:
                local_result = await self._process_local(
                    local_image, local_audio, gesture_type, output_path
                )
                if local_result:
                    result["video_url"] = local_result
                    result["provider_used"] = "local"
                    result["status"] = "completed"
                    self._cache_result(cache_key, local_result, quality)
                    logger.info("[LocalAvatar] Local processing succeeded")
                    return result
                raise RuntimeError("Local processing failed")

            # Step 3: For STANDARD/FINAL quality, use HunyuanVideo-Avatar (full-body, cheap)
            if quality in ["standard", "final"] and provider in [AnimationProvider.HUNYUAN, AnimationProvider.HYBRID]:
                if self.hunyuan and self.hunyuan.is_available():
                    from services.hunyuan_avatar_service import HunyuanQuality
                    hunyuan_quality = HunyuanQuality.HIGH if quality == "final" else HunyuanQuality.STANDARD
                    logger.info(f"[LocalAvatar] Trying HunyuanVideo-Avatar (quality={hunyuan_quality.value})...")

                    hunyuan_result = await self.hunyuan.generate_avatar_video(
                        image_path=local_image,
                        audio_path=local_audio,
                        quality=hunyuan_quality,
                        output_path=output_path
                    )

                    if hunyuan_result.get("status") == "completed":
                        result["video_url"] = hunyuan_result["video_url"]
                        result["provider_used"] = "hunyuan-avatar"
                        result["status"] = "completed"
                        result["duration"] = hunyuan_result.get("duration", 0)
                        result["cost_estimate"] = hunyuan_result.get("cost_estimate", 0.10)
                        self._cache_result(cache_key, result["video_url"], quality)
                        logger.info(f"[LocalAvatar] HunyuanVideo-Avatar succeeded (quality={quality})")
                        return result

                    logger.warning(f"[LocalAvatar] HunyuanVideo-Avatar failed: {hunyuan_result.get('error')}")
                else:
                    logger.info("[LocalAvatar] HunyuanVideo-Avatar not available, trying Replicate...")

            # Step 4: For DRAFT/PREVIEW or as fallback, use Replicate (SadTalker)
            if provider in [AnimationProvider.REPLICATE, AnimationProvider.HYBRID]:
                if self.replicate and self.replicate.is_available():
                    # Use SadTalker for draft/preview, OmniHuman as fallback for standard/final
                    model_name = "SadTalker" if quality in ["draft", "preview"] else "OmniHuman"
                    logger.info(f"[LocalAvatar] Trying Replicate ({model_name}, quality={quality})...")
                    replicate_result = await self._process_replicate(
                        local_image, local_audio, output_path, quality=quality
                    )
                    if replicate_result:
                        result["video_url"] = replicate_result
                        result["provider_used"] = f"replicate-{model_name.lower()}"
                        result["status"] = "completed"
                        self._cache_result(cache_key, replicate_result, quality)
                        logger.info(f"[LocalAvatar] Replicate {model_name} succeeded (quality={quality})")
                        return result
                    logger.warning("[LocalAvatar] Replicate failed, trying fallback...")
                else:
                    logger.info("[LocalAvatar] Replicate not available")

                if provider == AnimationProvider.REPLICATE:
                    raise RuntimeError("Replicate processing failed and fallback disabled")

            # Step 5: Fallback to D-ID (HYBRID or DID - expensive but reliable)
            if provider in [AnimationProvider.DID, AnimationProvider.HYBRID]:
                if self.did_provider:
                    logger.info("[LocalAvatar] Using D-ID fallback...")
                    did_result = await self._process_did(
                        local_image, local_audio, output_path
                    )
                    if did_result:
                        result["video_url"] = did_result
                        result["provider_used"] = "d-id"
                        result["status"] = "completed"
                        self._cache_result(cache_key, did_result, quality)
                        logger.info("[LocalAvatar] D-ID processing succeeded")
                        return result
                else:
                    logger.warning("[LocalAvatar] D-ID not configured")

            raise RuntimeError("All providers failed")

        except Exception as e:
            logger.error(f"[LocalAvatar] Error: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result

    async def _process_local(
        self,
        image_path: str,
        audio_path: str,
        gesture_type: str,
        output_path: str
    ) -> Optional[str]:
        """
        Process locally using Wav2Lip + FOMM.

        Pipeline:
        1. FOMM: Apply body animation (if available)
        2. Wav2Lip: Apply lip-sync
        """
        try:
            animated_video = image_path  # Start with static image

            # Step 1: Apply body animation with FOMM (if available)
            if self.fomm.is_available():
                logger.info("[LocalAvatar] Applying body animation with FOMM...")
                fomm_output = await self.fomm.animate(
                    source_image=image_path,
                    gesture_type=gesture_type
                )
                if fomm_output:
                    animated_video = fomm_output
                    logger.info("[LocalAvatar] FOMM animation complete")
                else:
                    logger.warning("[LocalAvatar] FOMM failed, continuing with static image")
            else:
                logger.info("[LocalAvatar] FOMM not available, skipping body animation")

            # Step 2: Apply lip-sync with Wav2Lip
            if self.wav2lip.is_available():
                logger.info("[LocalAvatar] Applying lip-sync with Wav2Lip...")
                lipsync_output = await self.wav2lip.generate_lipsync(
                    face_path=animated_video,
                    audio_path=audio_path,
                    output_path=output_path
                )
                if lipsync_output:
                    logger.info("[LocalAvatar] Wav2Lip lip-sync complete")
                    return lipsync_output
                else:
                    logger.warning("[LocalAvatar] Wav2Lip failed")
            else:
                logger.info("[LocalAvatar] Wav2Lip not available, using simple method")

            # Fallback: Simple video creation (image + audio)
            simple_output = await self.wav2lip.generate_lipsync_simple(
                face_path=animated_video,
                audio_path=audio_path,
                output_path=output_path
            )

            return simple_output

        except Exception as e:
            logger.error(f"[LocalAvatar] Local processing error: {e}")
            return None

    async def _process_replicate(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        quality: str = "final"
    ) -> Optional[str]:
        """
        Process using Replicate API with quality-based model selection.

        Quality modes:
        - draft: SadTalker only (~$0.002) - fast preview
        - preview: SadTalker + GFPGAN enhancement (~$0.01) - good quality
        - final: OmniHuman (~$2.80) - full body animation, best quality
        """
        if not self.replicate or not self.replicate.is_available():
            logger.error("[LocalAvatar] Replicate API not configured")
            return None

        try:
            from services.replicate_service import ReplicateModel

            # Quality-based model selection
            if quality in ["draft", "preview"]:
                # Use SadTalker for draft/preview (much cheaper)
                use_enhancer = "gfpgan" if quality == "preview" else None
                logger.info(f"[LocalAvatar] Using SadTalker (quality={quality}, enhancer={use_enhancer})")

                result = await self.replicate.generate_avatar(
                    source_image=image_path,
                    audio_path=audio_path,
                    output_path=output_path,
                    model=ReplicateModel.SADTALKER,
                    preprocess="crop",
                    still_mode=False,
                    expression_scale=1.0
                )

                if result:
                    logger.info(f"[LocalAvatar] SadTalker output (quality={quality}): {result}")
                    return result

                # Fallback to Wav2Lip if SadTalker fails
                logger.info("[LocalAvatar] SadTalker failed, trying Wav2Lip...")
                return await self.replicate.generate_avatar(
                    source_image=image_path,
                    audio_path=audio_path,
                    output_path=output_path,
                    model=ReplicateModel.WAV2LIP
                )

            else:
                # FINAL quality: Use OmniHuman for full body animation
                logger.info("[LocalAvatar] Generating with OmniHuman (quality=final, full body)...")
                result = await self.replicate.generate_avatar(
                    source_image=image_path,
                    audio_path=audio_path,
                    output_path=output_path,
                    model=ReplicateModel.OMNI_HUMAN
                )

                if result:
                    logger.info(f"[LocalAvatar] OmniHuman output: {result}")
                    return result

                # Fallback to SadTalker if OmniHuman fails
                logger.info("[LocalAvatar] OmniHuman failed, falling back to SadTalker...")
                result = await self.replicate.generate_avatar(
                    source_image=image_path,
                    audio_path=audio_path,
                    output_path=output_path,
                    model=ReplicateModel.SADTALKER,
                    preprocess="crop",
                    still_mode=False,
                    expression_scale=1.0
                )

                if result:
                    logger.info(f"[LocalAvatar] SadTalker fallback output: {result}")
                    return result

                # Final fallback to Wav2Lip
                logger.info("[LocalAvatar] SadTalker failed, trying Wav2Lip...")
                return await self.replicate.generate_avatar(
                    source_image=image_path,
                    audio_path=audio_path,
                    output_path=output_path,
                    model=ReplicateModel.WAV2LIP
                )

        except Exception as e:
            logger.error(f"[LocalAvatar] Replicate error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _process_did(
        self,
        image_path: str,
        audio_path: str,
        output_path: str
    ) -> Optional[str]:
        """Process using D-ID API."""
        if not self.did_provider:
            logger.error("[LocalAvatar] D-ID API key not configured")
            return None

        try:
            # Upload image to D-ID
            logger.info("[LocalAvatar] Uploading image to D-ID...")
            source_url = await self.did_provider.upload_source_image(image_path)

            # Upload audio to D-ID
            logger.info("[LocalAvatar] Uploading audio to D-ID...")
            audio_url = await self.did_provider.upload_audio(audio_path)

            # Create talk
            logger.info("[LocalAvatar] Creating D-ID talk...")
            talk_id = await self.did_provider.create_talk(
                source_url=source_url,
                audio_url=audio_url,
                enable_body_motion=True
            )

            # Poll for completion
            result = await self.did_provider.poll_until_complete(talk_id)

            if result and result.get("result_url"):
                # Download result
                video_url = result["result_url"]
                local_path = await self.did_provider._download_video(video_url, talk_id)
                return local_path

            return None

        except Exception as e:
            logger.error(f"[LocalAvatar] D-ID error: {e}")
            return None

    async def _ensure_local_file(
        self,
        file_path: str,
        file_type: str = "file"
    ) -> Optional[str]:
        """Ensure file is local, download if URL."""
        if file_path.startswith(("http://", "https://")):
            try:
                import uuid
                ext = Path(file_path).suffix or (".png" if file_type == "image" else ".mp3")
                local_path = self.output_dir / f"{file_type}_{uuid.uuid4().hex[:8]}{ext}"

                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.get(file_path, follow_redirects=True)
                    if response.status_code == 200:
                        with open(local_path, "wb") as f:
                            f.write(response.content)
                        return str(local_path)

                return None
            except Exception as e:
                logger.error(f"Failed to download {file_type}: {e}")
                return None

        return file_path if os.path.exists(file_path) else None

    async def _optimize_image(
        self,
        image_path: str,
        max_size: int = 1024,
        quality: int = 85
    ) -> str:
        """
        Optimize image for avatar processing.

        - Resize to max_size on longest side (OmniHuman doesn't need >1024px)
        - Compress to JPEG/PNG with reasonable quality
        - This reduces API processing time and cost

        Args:
            image_path: Path to input image
            max_size: Max dimension (width or height)
            quality: JPEG quality (1-100)

        Returns:
            Path to optimized image
        """
        try:
            img = Image.open(image_path)
            original_size = os.path.getsize(image_path)
            original_dims = img.size

            # Check if resize needed
            width, height = img.size
            if max(width, height) > max_size:
                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.info(f"[Optimize] Resized: {original_dims} -> {img.size}")

            # Handle transparency - convert RGBA to RGB for JPEG
            has_transparency = img.mode in ('RGBA', 'LA') or (
                img.mode == 'P' and 'transparency' in img.info
            )

            # Save optimized image
            import uuid
            if has_transparency:
                # Keep PNG for transparent images
                output_path = str(self.output_dir / f"opt_{uuid.uuid4().hex[:8]}.png")
                img.save(output_path, "PNG", optimize=True)
            else:
                # Convert to JPEG for smaller size
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                output_path = str(self.output_dir / f"opt_{uuid.uuid4().hex[:8]}.jpg")
                img.save(output_path, "JPEG", quality=quality, optimize=True)

            new_size = os.path.getsize(output_path)
            savings = (1 - new_size / original_size) * 100 if original_size > 0 else 0

            logger.info(
                f"[Optimize] Image: {original_size/1024:.1f}KB -> {new_size/1024:.1f}KB "
                f"({savings:.1f}% reduction)"
            )

            return output_path

        except Exception as e:
            logger.warning(f"[Optimize] Image optimization failed: {e}, using original")
            return image_path

    async def download_models(self) -> Dict[str, bool]:
        """Download all required models."""
        results = {}

        logger.info("Downloading Wav2Lip model...")
        results["wav2lip"] = await self.wav2lip.download_model()

        logger.info("Downloading FOMM model...")
        results["fomm"] = await self.fomm.download_models()

        return results


# Singleton
_local_avatar_service = None

def get_local_avatar_service() -> LocalAvatarService:
    global _local_avatar_service
    if _local_avatar_service is None:
        _local_avatar_service = LocalAvatarService()
    return _local_avatar_service
