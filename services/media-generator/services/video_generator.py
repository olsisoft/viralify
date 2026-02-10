"""
AI Video Generator Orchestrator
Coordinates all services to generate a complete video from a prompt
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from pydantic import BaseModel, Field
from enum import Enum

from .ai_video_planner import AIVideoPlannerService, VideoProject, Scene, SceneType
from .asset_fetcher import AssetFetcherService, MediaType, FetchedAsset
from .music_service import MusicService, MusicTrack
from .video_compositor import VideoCompositorService, CompositionRequest, CompositionScene, WordTimestamp


class GenerationStage(str, Enum):
    PLANNING = "planning"
    FETCHING_ASSETS = "fetching_assets"
    GENERATING_VOICEOVER = "generating_voiceover"
    FETCHING_MUSIC = "fetching_music"
    COMPOSING = "composing"
    COMPLETED = "completed"
    FAILED = "failed"


class StageProgress(BaseModel):
    stage: GenerationStage
    progress: int = 0  # 0-100
    message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class VideoGenerationJob(BaseModel):
    id: str
    user_id: str
    prompt: str
    status: GenerationStage = GenerationStage.PLANNING
    stages: Dict[str, StageProgress] = {}
    project: Optional[VideoProject] = None
    output_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True


class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=1000)
    duration: int = Field(default=30, ge=15, le=2700)  # Max 45 minutes
    style: str = Field(default="cinematic")
    format: str = Field(default="9:16", pattern=r"^(9:16|16:9|1:1)$")
    voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM")  # Rachel
    voice_provider: str = Field(default="elevenlabs")
    include_music: bool = True
    music_style: Optional[str] = None
    prefer_ai_images: bool = False
    caption_style: Optional[str] = None  # classic, bold, neon, minimal, karaoke, boxed, gradient
    caption_config: Optional[Dict[str, Any]] = None
    # Lip-sync options
    enable_lipsync: bool = False  # Enable lip-sync animation for presenter scenes
    avatar_id: Optional[str] = None  # Avatar ID for lip-sync
    lipsync_expression: str = "neutral"  # Avatar expression: neutral, happy, serious
    # Body motion options
    enable_body_motion: bool = True  # Enable natural body movements during speech
    use_presenter: bool = False  # Use D-ID presenter (real actor with natural movements)
    # Face swap options
    face_swap_image: Optional[str] = None  # URL or base64 of user's face to swap onto avatar
    face_swap_hair_source: str = "user"  # Hair source: 'user' or 'target'
    # PIP (Picture-in-Picture) options
    pip_position: str = "bottom-right"  # Position: bottom-right, bottom-left, top-right, top-left
    pip_size: float = 0.35  # Size as fraction of screen width (0.2-0.5)
    pip_shadow: bool = True  # Add drop shadow
    pip_remove_background: bool = True  # Remove avatar background
    # Quality/Cost optimization
    avatar_quality: str = "final"  # 'draft' (~$0.002), 'preview' (~$0.01), 'final' (~$2.80)


class AIVideoGenerator:
    """
    Main orchestrator for AI-powered video generation
    """

    def __init__(
        self,
        openai_api_key: str,
        elevenlabs_api_key: str = "",
        pexels_api_key: str = "",
        unsplash_api_key: str = "",
        pixabay_api_key: str = "",
        did_api_key: str = ""
    ):
        self.planner = AIVideoPlannerService(openai_api_key)
        self.asset_fetcher = AssetFetcherService(
            pexels_api_key=pexels_api_key,
            unsplash_api_key=unsplash_api_key,
            pixabay_api_key=pixabay_api_key,
            openai_api_key=openai_api_key
        )
        self.music_service = MusicService(pixabay_api_key=pixabay_api_key)
        self.compositor = VideoCompositorService()

        self.openai_key = openai_api_key
        self.elevenlabs_key = elevenlabs_api_key
        self.did_api_key = did_api_key

        # D-ID provider for lip-sync (lazy loaded)
        self._did_provider = None

        # Job storage (in production, use Redis/DB)
        self.jobs: Dict[str, VideoGenerationJob] = {}

    def _get_did_provider(self):
        """Lazy load D-ID provider"""
        if self._did_provider is None and self.did_api_key:
            from providers.did_provider import DIDProvider
            self._did_provider = DIDProvider(
                api_key=self.did_api_key,
                output_dir="/tmp/viralify/lipsync"
            )
        return self._did_provider

    async def generate_video(
        self,
        request: VideoGenerationRequest,
        user_id: str = "demo-user",
        progress_callback: Optional[Callable] = None
    ) -> VideoGenerationJob:
        """
        Generate a complete video from a prompt
        Returns immediately with job ID, processing happens in background
        """

        job_id = str(uuid.uuid4())
        job = VideoGenerationJob(
            id=job_id,
            user_id=user_id,
            prompt=request.prompt,
            status=GenerationStage.PLANNING,
            stages={
                GenerationStage.PLANNING.value: StageProgress(
                    stage=GenerationStage.PLANNING,
                    progress=0,
                    message="Initializing..."
                ),
                GenerationStage.FETCHING_ASSETS.value: StageProgress(
                    stage=GenerationStage.FETCHING_ASSETS
                ),
                GenerationStage.GENERATING_VOICEOVER.value: StageProgress(
                    stage=GenerationStage.GENERATING_VOICEOVER
                ),
                GenerationStage.FETCHING_MUSIC.value: StageProgress(
                    stage=GenerationStage.FETCHING_MUSIC
                ),
                GenerationStage.COMPOSING.value: StageProgress(
                    stage=GenerationStage.COMPOSING
                )
            }
        )

        self.jobs[job_id] = job

        # Start async generation
        asyncio.create_task(self._run_generation(job, request, progress_callback))

        return job

    async def _run_generation(
        self,
        job: VideoGenerationJob,
        request: VideoGenerationRequest,
        progress_callback: Optional[Callable] = None
    ):
        """Run the full generation pipeline"""

        try:
            # Stage 1: Planning
            await self._update_stage(job, GenerationStage.PLANNING, 0, "Analyzing prompt...")

            project = await self.planner.plan_video(
                prompt=request.prompt,
                duration=request.duration,
                style=request.style,
                format=request.format
            )
            project.voice_id = request.voice_id
            job.project = project

            await self._update_stage(job, GenerationStage.PLANNING, 100, "Plan created!")

            # Stage 2: Fetch Assets
            await self._update_stage(job, GenerationStage.FETCHING_ASSETS, 0, "Searching for media...")
            job.status = GenerationStage.FETCHING_ASSETS

            await self._fetch_scene_assets(job, project, request)

            await self._update_stage(job, GenerationStage.FETCHING_ASSETS, 100, "Assets ready!")

            # Stage 3: Generate Voiceover with word timestamps
            await self._update_stage(job, GenerationStage.GENERATING_VOICEOVER, 0, "Generating voiceover...")
            job.status = GenerationStage.GENERATING_VOICEOVER

            voiceover_url, word_timestamps = await self._generate_voiceover(
                project.voiceover_text,
                request.voice_id,
                request.voice_provider
            )

            await self._update_stage(job, GenerationStage.GENERATING_VOICEOVER, 100, "Voiceover ready!")

            # Stage 3.5: Process lip-sync if enabled
            lipsync_video_url = None
            if request.enable_lipsync and voiceover_url:
                print(f"[LIP-SYNC] Processing lip-sync with avatar: {request.avatar_id}", flush=True)
                await self._update_stage(job, GenerationStage.GENERATING_VOICEOVER, 60, "Generating lip-sync animation...")

                lipsync_video_url = await self._process_lipsync_scenes(
                    job=job,
                    project=project,
                    voiceover_url=voiceover_url,
                    avatar_id=request.avatar_id,
                    expression=request.lipsync_expression,
                    enable_body_motion=request.enable_body_motion,
                    use_presenter=request.use_presenter,
                    face_swap_image=request.face_swap_image,
                    face_swap_hair_source=request.face_swap_hair_source,
                    avatar_quality=request.avatar_quality
                )
                print(f"[LIP-SYNC] Avatar video URL for PIP: {lipsync_video_url}", flush=True)
                await self._update_stage(job, GenerationStage.GENERATING_VOICEOVER, 100, "Voiceover + lip-sync ready!")

            # Stage 4: Get Music - match to video style
            music_url = None
            if request.include_music:
                await self._update_stage(job, GenerationStage.FETCHING_MUSIC, 0, "Finding matching music...")
                job.status = GenerationStage.FETCHING_MUSIC

                # Map video style to music mood for better matching
                style_to_mood = {
                    "energetic": "energetic",
                    "cinematic": "cinematic",
                    "calm": "calm",
                    "professional": "corporate",
                    "fun": "happy",
                    "epic": "epic",
                    "emotional": "emotional",
                    "inspiring": "inspirational",
                    "motivational": "upbeat",
                    "relaxing": "calm",
                    "dramatic": "epic",
                    "upbeat": "upbeat",
                }

                # Determine best mood from style and music_style
                target_mood = style_to_mood.get(request.style, "cinematic")
                if request.music_style:
                    # User specified music style takes priority
                    target_mood = request.music_style.lower().split()[0]  # First word
                elif project.music_style:
                    target_mood = style_to_mood.get(project.music_style.split()[0].lower(), target_mood)

                music_track = await self.music_service.get_best_track_for_mood(
                    target_mood,
                    min_duration=request.duration
                )
                if music_track:
                    music_url = music_track.url
                    project.music_url = music_url
                    print(f"Selected music: {music_track.title} (mood: {music_track.mood})")

                await self._update_stage(job, GenerationStage.FETCHING_MUSIC, 100, "Music ready!")

            # Stage 5: Compose Video
            await self._update_stage(job, GenerationStage.COMPOSING, 0, "Composing video...")
            job.status = GenerationStage.COMPOSING

            # Build composition request
            composition_scenes = []
            for scene in project.scenes:
                if scene.media_url:
                    composition_scenes.append(CompositionScene(
                        id=scene.id,
                        order=scene.order,
                        media_url=scene.media_url,
                        media_type="video" if scene.scene_type == SceneType.VIDEO else "image",
                        duration=scene.duration,
                        start_time=scene.start_time,
                        text_overlay=scene.text_overlay,
                        transition=scene.transition
                    ))

            if not composition_scenes:
                raise Exception("No scenes with media to compose")

            composition_request = CompositionRequest(
                project_id=project.id,
                scenes=composition_scenes,
                voiceover_url=voiceover_url,
                voiceover_text=project.voiceover_text,  # For captions
                word_timestamps=word_timestamps,  # For synchronized animated captions
                music_url=music_url,
                music_volume=project.music_volume,
                format=request.format,
                quality="1080p",
                fps=30,
                caption_style=request.caption_style,
                caption_config=request.caption_config,
                # PIP Avatar overlay settings
                pip_avatar_url=lipsync_video_url,  # Lip-sync avatar video for Picture-in-Picture
                pip_position=getattr(request, 'pip_position', 'bottom-right'),
                pip_size=getattr(request, 'pip_size', 0.35),
                pip_margin=getattr(request, 'pip_margin', 20),
                pip_border_radius=getattr(request, 'pip_border_radius', 15),
                pip_shadow=getattr(request, 'pip_shadow', True),
                pip_remove_background=getattr(request, 'pip_remove_background', True)
            )

            def composition_progress(percent, message):
                asyncio.create_task(
                    self._update_stage(job, GenerationStage.COMPOSING, percent, message)
                )

            result = await self.compositor.compose_video(
                composition_request,
                progress_callback=composition_progress
            )

            if not result.success:
                raise Exception(result.error_message or "Composition failed")

            # Complete!
            job.output_url = result.output_url
            job.status = GenerationStage.COMPLETED
            job.completed_at = datetime.utcnow()

            await self._update_stage(job, GenerationStage.COMPOSING, 100, "Video ready!")

        except Exception as e:
            job.status = GenerationStage.FAILED
            job.error_message = str(e)
            print(f"Video generation error: {e}")

    async def generate_video_from_project(
        self,
        project: VideoProject,
        request: VideoGenerationRequest,
        user_id: str = "demo-user"
    ) -> VideoGenerationJob:
        """
        Generate a video from an existing project (script-based).
        Skips the planning phase since project is already created.
        """
        import asyncio

        job = VideoGenerationJob(
            id=project.id,
            user_id=user_id,
            prompt=f"Script-based video: {project.title}",
            status=GenerationStage.FETCHING_ASSETS,
            project=project,
            created_at=datetime.utcnow(),
            stages={
                GenerationStage.PLANNING.value: StageProgress(
                    stage=GenerationStage.PLANNING,
                    progress=100,
                    message="Script provided - skipped"
                ),
                GenerationStage.FETCHING_ASSETS.value: StageProgress(
                    stage=GenerationStage.FETCHING_ASSETS,
                    progress=0,
                    message="Starting..."
                ),
                GenerationStage.GENERATING_VOICEOVER.value: StageProgress(
                    stage=GenerationStage.GENERATING_VOICEOVER
                ),
                GenerationStage.FETCHING_MUSIC.value: StageProgress(
                    stage=GenerationStage.FETCHING_MUSIC
                ),
                GenerationStage.COMPOSING.value: StageProgress(
                    stage=GenerationStage.COMPOSING
                )
            }
        )
        self.jobs[job.id] = job

        # Start generation in background (skip planning)
        print(f"Creating background task for job {job.id}...")
        task = asyncio.create_task(self._generate_from_existing_project(job, project, request))
        print(f"Background task created: {task}")
        # Add callback to handle errors
        def handle_task_result(t):
            try:
                exc = t.exception()
                if exc:
                    print(f"Background task failed with exception: {exc}")
                    import traceback
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
            except asyncio.CancelledError:
                print(f"Background task was cancelled")
            except asyncio.InvalidStateError:
                pass  # Task still running
        task.add_done_callback(handle_task_result)

        return job

    async def _generate_from_existing_project(
        self,
        job: VideoGenerationJob,
        project: VideoProject,
        request: VideoGenerationRequest
    ):
        """Generate video from an existing project (skips planning)"""
        import sys
        import traceback
        print(f"=== STARTING VIDEO GENERATION ===", flush=True)
        print(f"Job ID: {job.id}", flush=True)
        print(f"Project: {project.title}", flush=True)
        print(f"Lip-sync enabled: {request.enable_lipsync}, Avatar: {request.avatar_id}", flush=True)
        print(f"Project has {len(project.scenes)} scenes", flush=True)
        sys.stdout.flush()
        try:
            # Stage 1: Fetch Assets
            print("Stage 1: Fetching assets...", flush=True)
            await self._update_stage(job, GenerationStage.FETCHING_ASSETS, 0, "Fetching media assets...")
            print("Calling _fetch_scene_assets...", flush=True)
            await self._fetch_scene_assets(job, project, request)
            print("Assets fetched successfully!", flush=True)
            await self._update_stage(job, GenerationStage.FETCHING_ASSETS, 100, "Assets ready!")

            # Stage 2: Generate Voiceover with word timestamps
            print(f"Stage 2: Generating voiceover (text length: {len(project.voiceover_text) if project.voiceover_text else 0})...", flush=True)
            voiceover_url = None
            word_timestamps = []
            if project.voiceover_text:
                await self._update_stage(job, GenerationStage.GENERATING_VOICEOVER, 0, "Generating voiceover...")
                job.status = GenerationStage.GENERATING_VOICEOVER

                voiceover_url, word_timestamps = await self._generate_voiceover(
                    text=project.voiceover_text,
                    voice_id=request.voice_id,
                    provider=request.voice_provider
                )
                await self._update_stage(job, GenerationStage.GENERATING_VOICEOVER, 100, "Voiceover ready!")

            # Stage 2.5: Lip-sync processing (if enabled)
            # Returns the lip-sync video URL for PIP overlay (doesn't replace scene media)
            lipsync_video_url = None
            if request.enable_lipsync and voiceover_url:
                lipsync_video_url = await self._process_lipsync_scenes(
                    job=job,
                    project=project,
                    voiceover_url=voiceover_url,
                    avatar_id=request.avatar_id,
                    expression=request.lipsync_expression,
                    enable_body_motion=request.enable_body_motion,
                    use_presenter=request.use_presenter,
                    face_swap_image=request.face_swap_image,
                    face_swap_hair_source=request.face_swap_hair_source,
                    avatar_quality=request.avatar_quality
                )
                print(f"[LIP-SYNC] Video URL for PIP: {lipsync_video_url}", flush=True)

            # Stage 3: Get Music
            music_url = None
            if request.include_music:
                await self._update_stage(job, GenerationStage.FETCHING_MUSIC, 0, "Finding matching music...")
                job.status = GenerationStage.FETCHING_MUSIC

                music_track = await self.music_service.get_best_track_for_mood(
                    request.music_style or project.music_style or "cinematic",
                    min_duration=request.duration
                )
                if music_track:
                    music_url = music_track.url
                    project.music_url = music_url
                    print(f"Selected music: {music_track.title} (mood: {music_track.mood})")

                await self._update_stage(job, GenerationStage.FETCHING_MUSIC, 100, "Music ready!")

            # Stage 4: Compose Video
            await self._update_stage(job, GenerationStage.COMPOSING, 0, "Composing video...")
            job.status = GenerationStage.COMPOSING

            # Count scenes with and without media
            scenes_with_media = sum(1 for s in project.scenes if s.media_url)
            scenes_without_media = len(project.scenes) - scenes_with_media
            print(f"[COMPOSITION] Starting composition...", flush=True)
            print(f"[COMPOSITION] {scenes_with_media}/{len(project.scenes)} scenes have media ({scenes_without_media} missing)", flush=True)

            # Log all scene details before composition
            print(f"[COMPOSITION] Scene details:", flush=True)
            for idx, s in enumerate(project.scenes):
                print(f"[COMPOSITION]   Scene {idx+1}: type={s.scene_type}, duration={s.duration}s, media={s.media_url[:60] if s.media_url else 'None'}...", flush=True)

            composition_scenes = []
            for scene in project.scenes:
                if scene.media_url:
                    composition_scenes.append(CompositionScene(
                        id=scene.id,
                        order=scene.order,
                        media_url=scene.media_url,
                        media_type="video" if scene.scene_type == SceneType.VIDEO else "image",
                        duration=scene.duration,
                        start_time=scene.start_time,
                        text_overlay=scene.text_overlay,
                        transition=scene.transition
                    ))
                else:
                    # Generate a placeholder for scenes without media
                    print(f"Scene {scene.id} missing media, generating placeholder...")
                    try:
                        placeholder = await self.asset_fetcher.generate_ai_image(
                            prompt=scene.description or "Abstract colorful background",
                            style="cinematic",
                            aspect_ratio=project.format
                        )
                        if placeholder:
                            composition_scenes.append(CompositionScene(
                                id=scene.id,
                                order=scene.order,
                                media_url=placeholder.url,
                                media_type="image",
                                duration=scene.duration,
                                start_time=scene.start_time,
                                text_overlay=scene.text_overlay,
                                transition=scene.transition
                            ))
                    except Exception as e:
                        print(f"Failed to generate placeholder for scene {scene.id}: {e}")

            total_duration = sum(s.duration for s in composition_scenes)
            print(f"Total composition duration: {total_duration}s from {len(composition_scenes)} scenes")

            if not composition_scenes:
                raise Exception("No scenes with media to compose")

            composition_request = CompositionRequest(
                project_id=project.id,
                scenes=composition_scenes,
                voiceover_url=voiceover_url,
                voiceover_text=project.voiceover_text,
                word_timestamps=word_timestamps,  # For synchronized animated captions
                music_url=music_url,
                music_volume=project.music_volume,
                format=request.format,
                quality="1080p",
                fps=30,
                caption_style=request.caption_style,
                caption_config=request.caption_config,
                # PIP Avatar overlay settings
                pip_avatar_url=lipsync_video_url,  # Lip-sync avatar video for Picture-in-Picture
                pip_position=getattr(request, 'pip_position', 'bottom-right'),
                pip_size=getattr(request, 'pip_size', 0.35),
                pip_margin=getattr(request, 'pip_margin', 20),
                pip_border_radius=getattr(request, 'pip_border_radius', 15),
                pip_shadow=getattr(request, 'pip_shadow', True),
                # Background removal for seamless avatar blending
                pip_remove_background=getattr(request, 'pip_remove_background', True),
                pip_bg_color=getattr(request, 'pip_bg_color', None),
                pip_bg_similarity=getattr(request, 'pip_bg_similarity', 0.3)
            )

            def composition_progress(percent, message):
                asyncio.create_task(
                    self._update_stage(job, GenerationStage.COMPOSING, percent, message)
                )

            result = await self.compositor.compose_video(
                composition_request,
                progress_callback=composition_progress
            )

            if not result.success:
                raise Exception(result.error_message or "Composition failed")

            # Complete!
            job.output_url = result.output_url
            job.status = GenerationStage.COMPLETED
            job.completed_at = datetime.utcnow()
            await self._update_stage(job, GenerationStage.COMPOSING, 100, "Video ready!")

        except Exception as e:
            job.status = GenerationStage.FAILED
            job.error_message = str(e)
            print(f"Video generation error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    async def _fetch_scene_assets(
        self,
        job: VideoGenerationJob,
        project: VideoProject,
        request: VideoGenerationRequest
    ):
        """Fetch media assets for all scenes in parallel"""

        total_scenes = len(project.scenes)
        print(f"Fetching assets for {total_scenes} scenes, format: {project.format}")

        async def fetch_scene_asset(scene: Scene, index: int):
            try:
                # Determine orientation and aspect ratio based on format
                if project.format == "9:16":
                    orientation = "portrait"
                    aspect_ratio = "9:16"
                elif project.format == "1:1":
                    orientation = "square"
                    aspect_ratio = "1:1"
                else:  # 16:9
                    orientation = "landscape"
                    aspect_ratio = "16:9"

                # Determine media type
                if scene.scene_type == SceneType.AI_IMAGE:
                    # Generate with DALL-E
                    asset = await self.asset_fetcher.generate_ai_image(
                        prompt=scene.description,
                        style=project.style,
                        aspect_ratio=aspect_ratio
                    )
                elif scene.scene_type == SceneType.IMAGE:
                    # Search for image with visual description and DALL-E fallback
                    asset = await self.asset_fetcher.fetch_best_asset(
                        keywords=scene.search_keywords,
                        media_type=MediaType.IMAGE,
                        orientation=orientation,
                        prefer_ai=request.prefer_ai_images,
                        visual_description=scene.description,
                        style=project.style,
                        fallback_to_ai=True,
                        aspect_ratio=aspect_ratio
                    )
                else:
                    # Search for video with visual description and DALL-E fallback for images
                    asset = await self.asset_fetcher.fetch_best_asset(
                        keywords=scene.search_keywords,
                        media_type=MediaType.VIDEO,
                        orientation=orientation,
                        visual_description=scene.description,
                        style=project.style,
                        fallback_to_ai=True,  # Will generate image if no video found
                        aspect_ratio=aspect_ratio
                    )

                if asset:
                    scene.media_url = asset.url
                    scene.thumbnail_url = asset.preview_url

                # Update progress
                progress = int((index + 1) / total_scenes * 100)
                await self._update_stage(
                    job,
                    GenerationStage.FETCHING_ASSETS,
                    progress,
                    f"Fetched {index + 1}/{total_scenes} assets"
                )

            except Exception as e:
                print(f"Error fetching asset for scene {scene.id}: {e}")

        # Fetch all scenes in parallel (with limit)
        tasks = [
            fetch_scene_asset(scene, i)
            for i, scene in enumerate(project.scenes)
        ]
        await asyncio.gather(*tasks)

    async def _generate_voiceover(
        self,
        text: str,
        voice_id: str,
        provider: str
    ) -> tuple[Optional[str], Optional[List[WordTimestamp]]]:
        """Generate voiceover audio with word-level timestamps"""

        import httpx
        import json
        import os

        # Ensure temp directories exist
        os.makedirs("/tmp/viralify/audio", exist_ok=True)
        os.makedirs("/tmp/viralify/video", exist_ok=True)
        os.makedirs("/tmp/viralify/output", exist_ok=True)

        word_timestamps = []

        try:
            if provider == "elevenlabs" and self.elevenlabs_key:
                # Use the streaming endpoint with timestamps
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
                        headers={
                            "xi-api-key": self.elevenlabs_key,
                            "Content-Type": "application/json"
                        },
                        json={
                            "text": text,
                            "model_id": "eleven_multilingual_v2",
                            "voice_settings": {
                                "stability": 0.5,
                                "similarity_boost": 0.75
                            }
                        },
                        timeout=180.0
                    )

                    if response.status_code == 200:
                        data = response.json()

                        # Extract audio (base64 encoded)
                        import base64
                        audio_base64 = data.get("audio_base64", "")

                        if audio_base64:
                            import tempfile
                            with tempfile.NamedTemporaryFile(
                                suffix=".mp3",
                                delete=False,
                                dir="/tmp/viralify/audio"
                            ) as f:
                                f.write(base64.b64decode(audio_base64))
                                audio_path = f.name

                            # Extract word timestamps from alignment
                            alignment = data.get("alignment", {})
                            characters = alignment.get("characters", [])
                            char_start_times = alignment.get("character_start_times_seconds", [])
                            char_end_times = alignment.get("character_end_times_seconds", [])

                            if characters and char_start_times:
                                # Build words from characters
                                current_word = ""
                                word_start = 0.0

                                for i, char in enumerate(characters):
                                    if char == " " or i == len(characters) - 1:
                                        if i == len(characters) - 1 and char != " ":
                                            current_word += char

                                        if current_word.strip():
                                            word_end = char_end_times[i-1] if i > 0 else char_end_times[i]
                                            word_timestamps.append(WordTimestamp(
                                                word=current_word.strip(),
                                                start=word_start,
                                                end=word_end
                                            ))

                                        current_word = ""
                                        if i < len(char_start_times) - 1:
                                            word_start = char_start_times[i + 1]
                                    else:
                                        if not current_word:
                                            word_start = char_start_times[i] if i < len(char_start_times) else 0
                                        current_word += char

                            print(f"Generated voiceover with {len(word_timestamps)} word timestamps")
                            return audio_path, word_timestamps

                    # Fallback to regular endpoint if with-timestamps fails
                    print(f"ElevenLabs with-timestamps failed ({response.status_code}), trying regular endpoint")
                    response = await client.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                        headers={
                            "xi-api-key": self.elevenlabs_key,
                            "Content-Type": "application/json"
                        },
                        json={
                            "text": text,
                            "model_id": "eleven_multilingual_v2",
                            "voice_settings": {
                                "stability": 0.5,
                                "similarity_boost": 0.75
                            }
                        },
                        timeout=120.0
                    )

                    if response.status_code == 200:
                        import tempfile
                        with tempfile.NamedTemporaryFile(
                            suffix=".mp3",
                            delete=False,
                            dir="/tmp/viralify/audio"
                        ) as f:
                            f.write(response.content)
                            # Generate estimated timestamps based on text length
                            word_timestamps = self._estimate_word_timestamps(text, f.name)
                            return f.name, word_timestamps

            # Fallback to OpenAI TTS (no native timestamps, will estimate)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/audio/speech",
                    headers={
                        "Authorization": f"Bearer {self.openai_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "tts-1-hd",
                        "input": text,
                        "voice": "nova"  # Default OpenAI voice
                    },
                    timeout=120.0
                )

                if response.status_code == 200:
                    import tempfile
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp3",
                        delete=False,
                        dir="/tmp/viralify/audio"
                    ) as f:
                        f.write(response.content)
                        # Estimate timestamps for OpenAI TTS
                        word_timestamps = self._estimate_word_timestamps(text, f.name)
                        return f.name, word_timestamps

        except Exception as e:
            print(f"Voiceover generation error: {e}")
            import traceback
            traceback.print_exc()

        return None, []

    def _estimate_word_timestamps(self, text: str, audio_path: str) -> List[WordTimestamp]:
        """Estimate word timestamps based on audio duration and text"""
        import subprocess

        # Get audio duration using ffprobe
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True, text=True
            )
            duration = float(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError, OSError):
            # Estimate duration based on text length (average speaking rate)
            words = text.split()
            duration = len(words) * 0.4  # ~150 words per minute

        words = text.split()
        if not words:
            return []

        # Distribute time evenly with slight variation for natural feel
        time_per_word = duration / len(words)
        timestamps = []
        current_time = 0.0

        for word in words:
            # Longer words take slightly more time
            word_duration = time_per_word * (0.8 + 0.4 * min(len(word) / 8, 1))
            timestamps.append(WordTimestamp(
                word=word,
                start=current_time,
                end=current_time + word_duration
            ))
            current_time += word_duration

        # Adjust last word to end at exact duration
        if timestamps:
            timestamps[-1].end = duration

        print(f"Estimated {len(timestamps)} word timestamps for {duration:.1f}s audio")
        return timestamps

    async def _update_stage(
        self,
        job: VideoGenerationJob,
        stage: GenerationStage,
        progress: int,
        message: str
    ):
        """Update stage progress"""
        if stage.value in job.stages:
            job.stages[stage.value].progress = progress
            job.stages[stage.value].message = message
            if progress == 0:
                job.stages[stage.value].started_at = datetime.utcnow()
            elif progress == 100:
                job.stages[stage.value].completed_at = datetime.utcnow()

    def get_job(self, job_id: str) -> Optional[VideoGenerationJob]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    async def update_scene(
        self,
        job_id: str,
        scene_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Scene]:
        """Update a scene in a project (for user editing)"""
        job = self.jobs.get(job_id)
        if not job or not job.project:
            return None

        for scene in job.project.scenes:
            if scene.id == scene_id:
                for key, value in updates.items():
                    if hasattr(scene, key):
                        setattr(scene, key, value)
                # Reset media URL if description changed
                if "description" in updates or "search_keywords" in updates:
                    scene.media_url = None
                    scene.thumbnail_url = None
                return scene

        return None

    async def add_scene(
        self,
        job_id: str,
        scene_data: Dict[str, Any],
        after_scene_id: Optional[str] = None
    ) -> Optional[Scene]:
        """Add a new scene to the project"""
        job = self.jobs.get(job_id)
        if not job or not job.project:
            return None

        new_scene = Scene(
            id=f"{job.project.id}-scene-{len(job.project.scenes) + 1}",
            order=len(job.project.scenes) + 1,
            start_time=0,  # Will be recalculated
            duration=scene_data.get("duration", 5),
            scene_type=SceneType(scene_data.get("scene_type", "video")),
            description=scene_data.get("description", ""),
            search_keywords=scene_data.get("search_keywords", []),
            text_overlay=scene_data.get("text_overlay")
        )

        # Insert at correct position
        if after_scene_id:
            for i, scene in enumerate(job.project.scenes):
                if scene.id == after_scene_id:
                    job.project.scenes.insert(i + 1, new_scene)
                    break
        else:
            job.project.scenes.append(new_scene)

        # Recalculate order and timing
        self._recalculate_scene_timing(job.project)

        return new_scene

    async def remove_scene(self, job_id: str, scene_id: str) -> bool:
        """Remove a scene from the project"""
        job = self.jobs.get(job_id)
        if not job or not job.project:
            return False

        job.project.scenes = [s for s in job.project.scenes if s.id != scene_id]
        self._recalculate_scene_timing(job.project)

        return True

    def _recalculate_scene_timing(self, project: VideoProject):
        """Recalculate scene order and start times"""
        current_time = 0.0
        for i, scene in enumerate(project.scenes):
            scene.order = i + 1
            scene.start_time = current_time
            current_time += scene.duration

    async def _process_lipsync_scenes(
        self,
        job: VideoGenerationJob,
        project: VideoProject,
        voiceover_url: str,
        avatar_id: Optional[str] = None,
        expression: str = "neutral",
        enable_body_motion: bool = True,
        use_presenter: bool = False,
        face_swap_image: Optional[str] = None,
        face_swap_hair_source: str = "user",
        avatar_quality: str = "final"
    ) -> Optional[str]:
        """
        Process D-ID lip-sync to create avatar talking video.
        Returns the lip-sync video URL for PIP overlay (doesn't replace scene media).
        Uploads our voiceover to D-ID so lip movements sync with our actual audio.

        Args:
            enable_body_motion: Add natural body/head movements during speech
            use_presenter: Use D-ID presenter (real actor) for more natural movements
            face_swap_image: Optional URL/path to user's face for face swap
            face_swap_hair_source: Hair source for face swap ('user' or 'target')
        """
        print("=" * 60, flush=True)
        print("=== LIP-SYNC PROCESSING STARTED ===", flush=True)
        print(f"Avatar ID: {avatar_id}", flush=True)
        print(f"Voiceover URL: {voiceover_url}", flush=True)
        print(f"Expression: {expression}", flush=True)
        print(f"Body motion: {enable_body_motion}", flush=True)
        print("=" * 60, flush=True)

        did_provider = self._get_did_provider()
        if not did_provider:
            print("ERROR: Lip-sync requested but D-ID API key not configured. Skipping...", flush=True)
            return None

        print(f"D-ID provider initialized, processing {len(project.scenes)} scenes...", flush=True)
        await self._update_stage(job, GenerationStage.GENERATING_VOICEOVER, 50, "Uploading audio to D-ID...")

        # First, upload our voiceover audio to D-ID
        # voiceover_url is a local file path like /tmp/viralify/audio/xxx.mp3
        did_audio_url = None
        if voiceover_url and voiceover_url.startswith('/'):
            try:
                print(f"[LIP-SYNC] Uploading voiceover to D-ID: {voiceover_url}", flush=True)
                # Check if file exists
                if os.path.exists(voiceover_url):
                    print(f"[LIP-SYNC] Audio file exists, size: {os.path.getsize(voiceover_url)} bytes", flush=True)
                else:
                    print(f"[LIP-SYNC] WARNING: Audio file does not exist at {voiceover_url}", flush=True)
                did_audio_url = await did_provider.upload_audio(voiceover_url)
                print(f"[LIP-SYNC] Audio uploaded to D-ID successfully: {did_audio_url}", flush=True)
            except Exception as e:
                print(f"[LIP-SYNC] ERROR: Failed to upload audio to D-ID: {e}", flush=True)
                import traceback
                traceback.print_exc()
                # Fall back to TTS if upload fails
                did_audio_url = None
        else:
            print(f"[LIP-SYNC] Voiceover URL not a local path: {voiceover_url}", flush=True)

        await self._update_stage(job, GenerationStage.GENERATING_VOICEOVER, 60, "Generating lip-sync animation...")

        # If avatar_id is specified, look up the D-ID presenter URL from our gallery
        avatar_source_url = None
        original_avatar_url = None  # Keep original for reference
        avatar = None  # Avatar object for accessing clip presenter ID
        print(f"[LIP-SYNC] Looking up avatar: {avatar_id}", flush=True)
        if avatar_id:
            from services.avatar_service import AvatarService
            avatar_service = AvatarService(
                did_api_key=os.getenv("DID_API_KEY"),
                heygen_api_key=os.getenv("HEYGEN_API_KEY")
            )
            avatar = avatar_service.get_avatar_by_id(avatar_id)
            if avatar:
                avatar_source_url = avatar.did_presenter_id
                original_avatar_url = avatar_source_url
                print(f"[LIP-SYNC] Found avatar '{avatar.name}' (id: {avatar.id})", flush=True)
                print(f"[LIP-SYNC] D-ID presenter URL: {avatar_source_url}", flush=True)
            else:
                # Try to use default avatar
                print(f"[LIP-SYNC] Avatar '{avatar_id}' not found in gallery, trying default avatar...", flush=True)
                default_avatar = avatar_service.get_default_avatar()
                if default_avatar and default_avatar.did_presenter_id:
                    avatar_source_url = default_avatar.did_presenter_id
                    original_avatar_url = avatar_source_url
                    print(f"[LIP-SYNC] Using default avatar '{default_avatar.name}': {avatar_source_url}", flush=True)
                elif avatar_id.startswith("http"):
                    # If it looks like a URL, use it directly
                    avatar_source_url = avatar_id
                    original_avatar_url = avatar_source_url
                    print(f"[LIP-SYNC] Using direct URL: {avatar_source_url}", flush=True)
                else:
                    print(f"[LIP-SYNC] ERROR: Could not find avatar or fallback", flush=True)

        # Skip early background removal - let the avatar service handle it
        # The rembg model download is unreliable inside containers
        # Replicate/D-ID will handle the avatar as-is
        print(f"[LIP-SYNC] Skipping early background removal (handled by avatar service)", flush=True)

        processed_count = 0
        lipsync_video_path = None  # Will store the lip-sync video URL for PIP overlay
        print(f"[LIP-SYNC] Starting scene loop, avatar_source_url: {avatar_source_url}", flush=True)

        for i, scene in enumerate(project.scenes):
            # When avatar_id is specified, apply lip-sync to first scene always
            # Otherwise, check if scene is suitable for lip-sync
            force_lipsync = avatar_source_url and i == 0
            print(f"[LIP-SYNC] Scene {i+1}: force_lipsync={force_lipsync}", flush=True)

            scene_desc = (scene.description or "").lower()
            is_presenter_scene = any(keyword in scene_desc for keyword in [
                "person", "presenter", "speaker", "host", "talking", "explaining",
                "professional", "expert", "teacher", "instructor", "avatar", "face"
            ])
            print(f"[LIP-SYNC] Scene {i+1}: is_presenter_scene={is_presenter_scene}, desc[:50]={scene_desc[:50]}", flush=True)

            # Skip if not a presenter scene and no forced lipsync
            if not force_lipsync and not is_presenter_scene:
                print(f"[LIP-SYNC] Scene {i+1}: Skipping (not presenter, not forced)", flush=True)
                continue
            # Skip if no media and no avatar specified
            if not scene.media_url and not avatar_source_url:
                print(f"[LIP-SYNC] Scene {i+1}: Skipping (no media and no avatar)", flush=True)
                continue

            try:
                print(f"[LIP-SYNC] Processing scene {i + 1}: {scene.description[:50] if scene.description else 'no desc'}...", flush=True)

                # Use avatar if specified, otherwise use scene's image
                source_image = avatar_source_url if avatar_source_url else scene.media_url
                print(f"[LIP-SYNC] Source image: {source_image}", flush=True)

                # If we have uploaded audio, use it for perfect sync
                if did_audio_url:
                    print(f"[LIP-SYNC] Using uploaded audio for lip-sync: {did_audio_url[:80]}...", flush=True)
                    print(f"[LIP-SYNC] Body motion: {enable_body_motion}, Use presenter: {use_presenter}", flush=True)

                    # HYBRID APPROACH: Replicate (cheap GPU) first, D-ID fallback
                    # Configure via AVATAR_PROVIDER env var: "hybrid", "replicate", "d-id"
                    avatar_provider_env = os.getenv("AVATAR_PROVIDER", "hybrid").lower()
                    use_hybrid = avatar_provider_env in ("hybrid", "replicate", "true", "1")

                    if use_hybrid:
                        print(f"[LIP-SYNC] Using HYBRID provider (Replicate → D-ID fallback)...", flush=True)
                        try:
                            from services.local_avatar_service import get_local_avatar_service, AnimationProvider

                            # Map env var to AnimationProvider
                            provider_map = {
                                "hybrid": AnimationProvider.HYBRID,
                                "replicate": AnimationProvider.REPLICATE,
                                "d-id": AnimationProvider.DID,
                                "did": AnimationProvider.DID,
                            }
                            selected_provider = provider_map.get(avatar_provider_env, AnimationProvider.HYBRID)

                            local_avatar = get_local_avatar_service()
                            providers = local_avatar.get_available_providers()
                            print(f"[LIP-SYNC] Available providers: {providers}", flush=True)
                            print(f"[LIP-SYNC] Selected provider: {selected_provider.value}", flush=True)

                            # Download audio locally if it's a D-ID URL
                            local_audio_path = voiceover_url
                            if did_audio_url and did_audio_url.startswith("s3://"):
                                # Audio was uploaded to D-ID, need to use local file
                                local_audio_path = voiceover_url

                            # Log face swap if enabled
                            if face_swap_image:
                                print(f"[LIP-SYNC] Face swap enabled (hair_source: {face_swap_hair_source})", flush=True)

                            # Try Replicate first (SadTalker/OmniHuman based on quality), then D-ID fallback
                            local_result = await local_avatar.generate_avatar_video(
                                source_image=original_avatar_url or source_image,
                                audio_path=local_audio_path,
                                provider=selected_provider,
                                gesture_type="talking" if not use_presenter else "presenting",
                                remove_background=True,
                                face_swap_image=face_swap_image,
                                face_swap_hair_source=face_swap_hair_source,
                                quality=avatar_quality
                            )

                            if local_result.get("status") == "completed" and local_result.get("video_url"):
                                local_path = local_result["video_url"]
                                provider_used = local_result.get("provider_used", "unknown")
                                print(f"[LIP-SYNC] {provider_used.upper()} processing succeeded!", flush=True)
                                print(f"[LIP-SYNC] Video ready: {local_path}", flush=True)

                                lipsync_video_path = local_path
                                processed_count += 1
                                break

                            print(f"[LIP-SYNC] Hybrid processing failed: {local_result.get('error')}", flush=True)

                        except Exception as local_error:
                            print(f"[LIP-SYNC] Hybrid processing error: {local_error}", flush=True)

                    # Use Clips API for full body movement with presenters
                    if use_presenter and avatar and hasattr(avatar, 'did_clip_presenter_id') and avatar.did_clip_presenter_id:
                        print(f"[LIP-SYNC] Using D-ID Clips API for FULL BODY movement", flush=True)
                        print(f"[LIP-SYNC] Presenter ID: {avatar.did_clip_presenter_id}", flush=True)

                        # Create clip with presenter (has full body movements)
                        talk_id = await did_provider.create_clip_with_presenter(
                            presenter_id=avatar.did_clip_presenter_id,
                            audio_url=did_audio_url,
                            background_color="#00FF00"  # Green screen for easy removal
                        )
                        print(f"[LIP-SYNC] Clip created with ID: {talk_id}", flush=True)

                        # Poll for clip completion (different endpoint)
                        result = await did_provider.poll_clip_until_complete(talk_id)
                        if result and result.get("result_url"):
                            video_url = result["result_url"]
                            print(f"[LIP-SYNC] Clip video URL: {video_url}", flush=True)
                            local_path = await did_provider._download_video(video_url, talk_id)
                            print(f"[LIP-SYNC] Video downloaded to: {local_path}", flush=True)

                            # Store for PIP overlay
                            print(f"[LIP-SYNC] Clip video ready for PIP overlay: {local_path}", flush=True)
                            print(f"[LIP-SYNC] SUCCESS: Full body presenter video ready: {local_path}", flush=True)
                            lipsync_video_path = local_path
                            processed_count += 1
                            break  # Only process first scene for presenter

                        continue  # Skip to next scene if clip failed

                    # Direct D-ID Talks API (without local attempt)
                    print(f"[LIP-SYNC] Calling D-ID create_talk...", flush=True)
                    talk_id = await did_provider.create_talk(
                        source_url=source_image,
                        audio_url=did_audio_url,
                        driver_type="microsoft",
                        expression=expression,
                        enable_body_motion=enable_body_motion
                    )
                    print(f"[LIP-SYNC] Talk created with ID: {talk_id}", flush=True)
                else:
                    # Fallback to D-ID TTS (won't be perfectly synced)
                    print("Warning: Using D-ID TTS fallback - audio may not sync perfectly")
                    script_text = project.voiceover_text or "Hello, welcome to this video."
                    voice_map = {
                        "happy": "en-US-JennyNeural",
                        "serious": "en-US-GuyNeural",
                        "neutral": "en-US-AriaNeural"
                    }
                    voice_id = voice_map.get(expression, "en-US-AriaNeural")

                    talk_id = await did_provider.create_talk_with_text(
                        source_url=source_image,
                        script_text=script_text,
                        voice_id=voice_id,
                        driver_type="microsoft",
                        expression=expression
                    )

                # Poll for completion
                print(f"[LIP-SYNC] Polling for talk completion...", flush=True)
                result = await did_provider.poll_until_complete(talk_id)
                print(f"[LIP-SYNC] Poll complete, result: {result}", flush=True)

                if result and result.get("result_url"):
                    # Download the video locally
                    video_url = result["result_url"]
                    print(f"[LIP-SYNC] Downloading video from: {video_url}", flush=True)
                    local_path = await did_provider._download_video(video_url, talk_id)
                    print(f"[LIP-SYNC] Video downloaded to: {local_path}", flush=True)

                    # Verify file exists
                    if os.path.exists(local_path):
                        print(f"[LIP-SYNC] Video file exists, size: {os.path.getsize(local_path)} bytes", flush=True)
                    else:
                        print(f"[LIP-SYNC] WARNING: Video file does not exist at {local_path}", flush=True)

                    # Store the lip-sync video URL for PIP overlay
                    # DON'T replace scene media - keep original B-roll for background
                    print(f"[LIP-SYNC] Lip-sync video ready for PIP overlay: {local_path}", flush=True)
                    print(f"[LIP-SYNC] Original scene media preserved: {scene.media_url}", flush=True)
                    processed_count += 1
                    lipsync_video_path = local_path  # Store for return
                    print(f"[LIP-SYNC] SUCCESS: Lip-sync video ready: {local_path}", flush=True)
                else:
                    print(f"[LIP-SYNC] ERROR: No result_url in D-ID response", flush=True)
                    lipsync_video_path = None

                # Update progress
                progress = 60 + int((processed_count / max(1, len(project.scenes))) * 40)
                await self._update_stage(
                    job,
                    GenerationStage.GENERATING_VOICEOVER,
                    progress,
                    f"Lip-sync: avatar video ready"
                )

                # Only process once - we only need one lip-sync video for PIP
                break

            except Exception as e:
                print(f"Lip-sync failed for scene {i + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if processed_count > 0:
            print(f"[LIP-SYNC] COMPLETE: Lip-sync video generated", flush=True)
            print(f"[LIP-SYNC] Returning video URL for PIP: {lipsync_video_path}", flush=True)
        else:
            print("[LIP-SYNC] COMPLETE: No lip-sync video generated", flush=True)
            lipsync_video_path = None
        print("=" * 60, flush=True)

        return lipsync_video_path

    async def regenerate_from_edit(self, job_id: str) -> Optional[VideoGenerationJob]:
        """Re-run composition after user edits scenes"""
        job = self.jobs.get(job_id)
        if not job or not job.project:
            return None

        # Fetch assets for scenes without media
        scenes_needing_assets = [s for s in job.project.scenes if not s.media_url]

        if scenes_needing_assets:
            job.status = GenerationStage.FETCHING_ASSETS
            # This would trigger re-fetching...

        # Then re-compose
        # ...

        return job
