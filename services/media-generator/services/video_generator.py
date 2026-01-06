"""
AI Video Generator Orchestrator
Coordinates all services to generate a complete video from a prompt
"""

import asyncio
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
        pixabay_api_key: str = ""
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

        # Job storage (in production, use Redis/DB)
        self.jobs: Dict[str, VideoGenerationJob] = {}

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
                caption_config=request.caption_config
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
        asyncio.create_task(self._generate_from_existing_project(job, project, request))

        return job

    async def _generate_from_existing_project(
        self,
        job: VideoGenerationJob,
        project: VideoProject,
        request: VideoGenerationRequest
    ):
        """Generate video from an existing project (skips planning)"""
        try:
            # Stage 1: Fetch Assets
            await self._update_stage(job, GenerationStage.FETCHING_ASSETS, 0, "Fetching media assets...")
            await self._fetch_scene_assets(job, project, request)
            await self._update_stage(job, GenerationStage.FETCHING_ASSETS, 100, "Assets ready!")

            # Stage 2: Generate Voiceover with word timestamps
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
            print(f"Composing: {scenes_with_media}/{len(project.scenes)} scenes have media ({scenes_without_media} missing)")

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
                caption_config=request.caption_config
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
        except:
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
