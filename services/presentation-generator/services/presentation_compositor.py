"""
Presentation Compositor Service

Main orchestrator that coordinates all services to generate the final presentation video.
"""
import asyncio
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import httpx


def clean_voiceover_text(text: str) -> str:
    """
    Clean voiceover text before sending to TTS.
    Removes sync markers and technical artifacts that shouldn't be read aloud.
    """
    if not text:
        return ""

    # Remove [SYNC:slide_XXX] markers
    text = re.sub(r'\[SYNC:slide_\d+\]', '', text)

    # Remove other common technical markers
    text = re.sub(r'\[SLIDE[:\s]*\d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[PAUSE[:\s]*\d*m?s?\]', '', text, flags=re.IGNORECASE)

    # Remove markdown artifacts
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code

    # Remove multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()

    return text

from models.presentation_models import (
    CodeDisplayMode,
    GeneratePresentationRequest,
    PresentationJob,
    PresentationScript,
    PresentationStage,
    Slide,
    SlideType,
)
from services.presentation_planner import PresentationPlannerService
from services.slide_generator import SlideGeneratorService
from services.code_executor import CodeExecutorService
from services.typing_animator import TypingAnimatorService
from services.timeline_builder import TimelineBuilder, Timeline
from services.ffmpeg_timeline_compositor import FFmpegTimelineCompositor, SimpleTimelineCompositor

# Option B+: Direct sync (TTS per slide + crossfade)
from services.slide_audio_generator import SlideAudioGenerator, SlideAudioBatch
from services.audio_concatenator import AudioConcatenator, ConcatenatedAudio
from services.direct_timeline_builder import DirectTimelineBuilder, DirectTimeline

# Hybrid sync: Direct Sync + SSVS-D for diagram focus animations (Option A)
from services.sync.hybrid_synchronizer import (
    HybridSynchronizer,
    HybridSyncConfig,
    HybridSyncResult,
)

# SSVS-C: Code-aware synchronization for typing animations
from services.sync.code_synchronizer import CodeAwareSynchronizer, VoiceSegment


class PresentationCompositorService:
    """Main orchestrator for presentation generation"""

    # Typing speed presets (must match TypingAnimatorService)
    SPEED_PRESETS = {
        "slow": 2.0,
        "natural": 4.0,
        "moderate": 6.0,
        "fast": 10.0
    }

    def __init__(self):
        self.planner = PresentationPlannerService()
        self.slide_generator = SlideGeneratorService()
        self.code_executor = CodeExecutorService()
        self.typing_animator = TypingAnimatorService()

        # Media generator service URL (use Docker hostname for container communication)
        # In development, set MEDIA_GENERATOR_URL env var to override
        self.media_generator_url = os.getenv(
            "MEDIA_GENERATOR_URL",
            "http://media-generator:8004"
        )

        # Internal service URL (for Docker-to-Docker communication)
        self.service_url = os.getenv(
            "SERVICE_URL",
            "http://presentation-generator:8006"
        )

        # Public base URL for user-facing URLs (via nginx proxy)
        # e.g., https://olsitec.com -> URLs will be https://olsitec.com/presentations/files/...
        self.public_base_url = os.getenv("PUBLIC_BASE_URL", "")

        # Job storage (in production, use Redis)
        self.jobs: Dict[str, PresentationJob] = {}

        # Timeline-based composition (for precise sync)
        self.timeline_builder = TimelineBuilder()
        self.timeline_compositor = SimpleTimelineCompositor()

        # Output directory for generated videos
        self.output_dir = Path("/tmp/presentations/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use timeline composition for non-English languages (better sync)
        self.use_timeline_composition = os.getenv("USE_TIMELINE_COMPOSITION", "true").lower() == "true"

        # OPTION B+: Direct sync (TTS per slide + crossfade)
        # This provides PERFECT synchronization by construction
        # Set USE_DIRECT_SYNC=true to enable (recommended)
        self.use_direct_sync = os.getenv("USE_DIRECT_SYNC", "true").lower() == "true"

        # Direct sync components
        self.slide_audio_generator = SlideAudioGenerator()
        self.audio_concatenator = AudioConcatenator(crossfade_ms=100)
        self.direct_timeline_builder = DirectTimelineBuilder()

        if self.use_direct_sync:
            print("[COMPOSITOR] Using DIRECT SYNC mode (TTS per slide + crossfade)", flush=True)
        else:
            print("[COMPOSITOR] Using SSVS mode (post-hoc matching)", flush=True)

        # Hybrid sync: SSVS-D for diagram focus animations (can be disabled easily)
        # Set ENABLE_DIAGRAM_FOCUS=false to disable
        self.use_diagram_focus = os.getenv("ENABLE_DIAGRAM_FOCUS", "true").lower() == "true"
        self.hybrid_synchronizer = HybridSynchronizer(
            enable_diagram_focus=self.use_diagram_focus
        )
        if self.use_diagram_focus:
            print("[COMPOSITOR] Diagram focus animations: ENABLED (set ENABLE_DIAGRAM_FOCUS=false to disable)", flush=True)

    def _get_public_url(self, internal_path: str) -> str:
        """
        Convert an internal file path to a public URL.

        If PUBLIC_BASE_URL is set (e.g., https://olsitec.com/presentation),
        generates URLs like:
            https://olsitec.com/presentation/files/presentations/output/xxx.mp4

        Otherwise falls back to internal SERVICE_URL.
        """
        if self.public_base_url:
            # Extract relative path from /tmp/presentations/...
            if internal_path.startswith("/tmp/presentations/"):
                relative_path = internal_path.replace("/tmp/presentations/", "")
                # PUBLIC_BASE_URL already includes path prefix (e.g., /presentation)
                return f"{self.public_base_url}/files/presentations/{relative_path}"
            # Already a relative path
            return f"{self.public_base_url}/files/presentations/{internal_path}"
        else:
            # Fallback to internal URL (for development)
            if internal_path.startswith("/tmp/presentations/"):
                relative_path = internal_path.replace("/tmp/presentations/", "")
                return f"{self.service_url}/files/presentations/{relative_path}"
            return f"{self.service_url}/files/presentations/{internal_path}"

    def _estimate_animation_duration(
        self,
        code_length: int,
        typing_speed: str,
        has_execution_output: bool
    ) -> float:
        """
        Estimate the duration needed for a typing animation.

        This is called BEFORE creating animations to set proper slide durations
        so voiceover generation can account for the time needed.
        """
        chars_per_second = self.SPEED_PRESETS.get(typing_speed, 4.0)

        # Base typing time
        base_typing = code_length / chars_per_second

        # Add pause overhead (~50% for human-like pauses)
        with_pauses = base_typing * 1.5

        # Add execution output display time
        output_time = 4.0 if has_execution_output else 0

        # Add intro (0.5s) + outro hold for comprehension (3.0s)
        buffer = 3.5

        total = with_pauses + output_time + buffer

        return total

    async def _set_minimum_durations_for_animations(
        self,
        job: PresentationJob
    ):
        """
        Set minimum durations for code slides based on animation requirements.

        Called BEFORE voiceover generation so the voiceover can be paced correctly.
        """
        if not job.request.show_typing_animation:
            return

        typing_speed = job.request.typing_speed.value if job.request.typing_speed else "natural"
        has_execution = job.request.execute_code

        print(f"[DURATION] Setting minimum durations for animations (speed: {typing_speed}, execute: {has_execution})", flush=True)

        for slide in job.script.slides:
            if slide.type in [SlideType.CODE, SlideType.CODE_DEMO] and slide.code_blocks:
                # Get the longest code block
                max_code_length = max(len(cb.code) for cb in slide.code_blocks)

                # Check if this slide has execution output
                slide_has_execution = has_execution and slide.type == SlideType.CODE_DEMO

                # Estimate animation duration
                min_duration = self._estimate_animation_duration(
                    code_length=max_code_length,
                    typing_speed=typing_speed,
                    has_execution_output=slide_has_execution
                )

                # Set minimum duration (don't reduce if already longer)
                old_duration = slide.duration
                slide.duration = max(slide.duration, min_duration)

                print(f"[DURATION] Slide '{slide.title}': {max_code_length} chars -> min {min_duration:.1f}s (was {old_duration:.1f}s, now {slide.duration:.1f}s)", flush=True)

    async def generate_presentation(
        self,
        request: GeneratePresentationRequest,
        on_progress: Optional[Callable] = None
    ) -> PresentationJob:
        """
        Generate a complete presentation from a topic prompt.

        Args:
            request: The presentation generation request
            on_progress: Optional callback for progress updates

        Returns:
            PresentationJob with status and results
        """
        # Create job
        job = PresentationJob(request=request)
        self.jobs[job.job_id] = job

        # Start async generation
        asyncio.create_task(
            self._generate_async(job, on_progress)
        )

        return job

    async def _generate_async(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable] = None
    ):
        """Async generation pipeline"""
        # Initialize word timestamps for timeline-based composition
        job_word_timestamps = []

        try:
            # Stage 1: Planning (0-15%)
            job.update_progress(PresentationStage.PLANNING, 0, "Analyzing topic...")
            await self._notify_progress(job, on_progress)

            script = await self.planner.generate_script(
                job.request,
                on_progress=lambda p, m: self._update_stage_progress(
                    job, PresentationStage.PLANNING, p * 0.15, m, on_progress
                )
            )
            job.script = script
            job.update_progress(PresentationStage.PLANNING, 15, "Script generated")
            await self._notify_progress(job, on_progress)

            # Stage 2: Generating Slides (15-30%)
            job.update_progress(
                PresentationStage.GENERATING_SLIDES,
                15,
                "Generating slide images..."
            )
            await self._notify_progress(job, on_progress)

            slide_images = await self._generate_slides(job, on_progress)
            job.slide_images = slide_images
            job.update_progress(
                PresentationStage.GENERATING_SLIDES,
                30,
                f"Generated {len(slide_images)} slides"
            )
            await self._notify_progress(job, on_progress)

            # Stage 2.5: Execute Code (30-35%) - Phase 2
            if job.request.execute_code:
                job.update_progress(
                    PresentationStage.EXECUTING_CODE,
                    30,
                    "Executing code demos..."
                )
                await self._notify_progress(job, on_progress)

                execution_results = await self._execute_code_demos(job, on_progress)
                job.code_execution_results = execution_results
                job.update_progress(
                    PresentationStage.EXECUTING_CODE,
                    35,
                    f"Executed {len(execution_results)} code blocks"
                )
                await self._notify_progress(job, on_progress)

            # Stage 3: Generating Voiceover (35-50%)
            # Voiceover dictates the pace - animations will adapt to fit
            job.update_progress(
                PresentationStage.GENERATING_VOICEOVER,
                35,
                "Generating voiceover..."
            )
            await self._notify_progress(job, on_progress)

            # Store direct timeline for Option B+ mode
            direct_timeline_result = None
            slide_audio_batch = None  # For hybrid sync (SSVS-D diagram focus)
            hybrid_sync_result = None  # Diagram focus animations

            if self.use_direct_sync:
                # OPTION B+: Direct sync - TTS per slide with perfect synchronization
                print("[VOICEOVER] Using DIRECT SYNC mode (TTS per slide)", flush=True)
                voiceover_url, voiceover_duration, direct_timeline_result, slide_audio_batch = await self._generate_voiceover_direct(
                    job, on_progress
                )
                job.voiceover_url = voiceover_url
                # No word timestamps needed - sync is perfect by construction
                job_word_timestamps = []

                # Update slide durations from direct timeline
                if direct_timeline_result and direct_timeline_result.slide_timings:
                    for timing in direct_timeline_result.slide_timings:
                        slide_idx = timing.get("slide_index", 0)
                        if slide_idx < len(job.script.slides):
                            job.script.slides[slide_idx].duration = timing.get("duration", 1.0)

                # HYBRID SYNC: Process diagram slides with SSVS-D for focus animations
                # This adds focus animations while keeping Direct Sync's perfect timing
                # Can be disabled via ENABLE_DIAGRAM_FOCUS=false
                if self.use_diagram_focus and slide_audio_batch and slide_audio_batch.slide_audios:
                    try:
                        hybrid_sync_result = await self.hybrid_synchronizer.process_diagram_slides(
                            slides=job.script.slides,
                            slide_audios=slide_audio_batch.slide_audios,
                            diagram_metadata=None  # Could be populated from diagram generator
                        )
                        if hybrid_sync_result.diagrams_processed > 0:
                            print(f"[HYBRID_SYNC] Processed {hybrid_sync_result.diagrams_processed} diagram slides for focus animations", flush=True)
                    except Exception as e:
                        print(f"[HYBRID_SYNC] Error processing diagram focus (continuing without): {e}", flush=True)
                        hybrid_sync_result = None
            else:
                # Legacy mode: Single TTS + SSVS matching
                voiceover_url, voiceover_duration, word_timestamps = await self._generate_voiceover(job, on_progress)
                job.voiceover_url = voiceover_url
                job_word_timestamps = word_timestamps

                # Adjust slide durations to match actual voiceover duration
                if voiceover_duration and voiceover_duration > 0:
                    await self._adjust_slide_durations(job, voiceover_duration)

            job.update_progress(
                PresentationStage.GENERATING_VOICEOVER,
                50,
                "Voiceover generated"
            )
            await self._notify_progress(job, on_progress)

            # Stage 3.5: Create Typing Animations (50-60%) - AFTER voiceover (for correct durations)
            animation_map = {}
            print(f"[ANIMATION] show_typing_animation={job.request.show_typing_animation}", flush=True)
            if job.request.show_typing_animation:
                job.update_progress(
                    PresentationStage.CREATING_ANIMATIONS,
                    50,
                    "Creating typing animations..."
                )
                await self._notify_progress(job, on_progress)

                animation_map = await self._create_typing_animations(job, on_progress)
                job.animation_videos = [v["url"] for v in animation_map.values()]  # Store URLs for API response
                job.update_progress(
                    PresentationStage.CREATING_ANIMATIONS,
                    60,
                    f"Created {len(animation_map)} animations"
                )
                await self._notify_progress(job, on_progress)

            # Stage 3.5: Generate Avatar (55-65%) - Phase 2
            if job.request.include_avatar and job.request.avatar_id:
                job.update_progress(
                    PresentationStage.GENERATING_AVATAR,
                    55,
                    "Generating avatar video..."
                )
                await self._notify_progress(job, on_progress)

                avatar_url = await self._generate_avatar(job, on_progress)
                job.avatar_video_url = avatar_url
                job.update_progress(
                    PresentationStage.GENERATING_AVATAR,
                    65,
                    "Avatar video generated"
                )
                await self._notify_progress(job, on_progress)

            # Stage 4: Composing Video (65-95%)
            job.update_progress(
                PresentationStage.COMPOSING_VIDEO,
                65,
                "Composing final video..."
            )
            await self._notify_progress(job, on_progress)

            # Use timeline-based composition for precise audio-video sync
            # This is especially important for non-English languages
            output_url = None
            content_language = getattr(job.request, 'content_language', 'en') or 'en'

            # OPTION B+: Use direct timeline if available (perfect sync)
            if self.use_direct_sync and direct_timeline_result:
                print(f"[COMPOSE] Using DIRECT TIMELINE composition (perfect sync)", flush=True)
                output_url = await self._compose_video_with_direct_timeline(
                    job, on_progress, animation_map, direct_timeline_result,
                    hybrid_sync_result=hybrid_sync_result  # Pass diagram focus animations
                )

            # Legacy: Use SSVS-based timeline composition
            elif self.use_timeline_composition and job_word_timestamps:
                print(f"[COMPOSE] Using SSVS timeline-based composition ({len(job_word_timestamps)} word timestamps)", flush=True)
                output_url = await self._compose_video_with_timeline(
                    job, on_progress, animation_map, job_word_timestamps
                )

            # Fallback to traditional composition if timeline fails
            if not output_url:
                if self.use_direct_sync or self.use_timeline_composition:
                    print("[COMPOSE] Timeline composition failed, falling back to traditional method", flush=True)
                output_url = await self._compose_video(job, on_progress, animation_map)

            job.output_url = output_url

            # Stage 5: Complete
            job.update_progress(
                PresentationStage.COMPLETED,
                100,
                "Presentation ready!"
            )
            await self._notify_progress(job, on_progress)

        except Exception as e:
            job.error = str(e)
            job.error_details = {"type": type(e).__name__}
            job.update_progress(
                PresentationStage.FAILED,
                job.progress,
                f"Error: {str(e)}"
            )
            await self._notify_progress(job, on_progress)
            raise

    async def _update_stage_progress(
        self,
        job: PresentationJob,
        stage: PresentationStage,
        progress: float,
        message: str,
        on_progress: Optional[Callable]
    ):
        """Update progress for a specific stage"""
        job.update_progress(stage, progress, message)
        await self._notify_progress(job, on_progress)

    async def _notify_progress(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable]
    ):
        """Notify progress callback if provided"""
        if on_progress:
            if asyncio.iscoroutinefunction(on_progress):
                await on_progress(job)
            else:
                on_progress(job)

    async def _generate_slides(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable]
    ) -> list:
        """Generate all slide images with full context for accurate diagrams"""
        slide_images = []
        total_slides = len(job.script.slides)

        # Extract RAG context from request (for diagram accuracy)
        rag_context = getattr(job.request, 'rag_context', None)

        # Build course context for diagram generation
        course_context = {
            'topic': job.request.topic if job.request else '',
            'description': getattr(job.request, 'description', ''),
            'target_audience': job.script.target_audience if job.script else 'intermediate developers',
            'objectives': getattr(job.script, 'learning_objectives', []) if job.script else [],
        }

        if rag_context:
            print(f"[SLIDES] Using RAG context for diagram generation: {len(rag_context)} chars", flush=True)

        for i, slide in enumerate(job.script.slides):
            progress = 15 + (i / total_slides) * 25
            job.update_progress(
                PresentationStage.GENERATING_SLIDES,
                progress,
                f"Generating slide {i + 1}/{total_slides}..."
            )
            await self._notify_progress(job, on_progress)

            # Generate slide image with audience-based diagram complexity, career-based focus,
            # RAG context for accurate diagram generation, and RAG images for real diagrams
            rag_images = None
            if job.request and hasattr(job.request, 'rag_images'):
                rag_images = [img.model_dump() if hasattr(img, 'model_dump') else img for img in job.request.rag_images]

            image_bytes = await self.slide_generator.generate_slide_image(
                slide,
                job.request.style,
                target_audience=job.script.target_audience if job.script else "intermediate developers",
                target_career=job.script.target_career if job.script else None,
                rag_context=rag_context,
                course_context=course_context,
                rag_images=rag_images,
                job_id=job.job_id
            )

            # Upload to storage
            filename = f"{job.job_id}_slide_{i:03d}.png"
            image_url = await self.slide_generator.upload_to_cloudinary(
                image_bytes,
                filename
            )

            slide.image_url = image_url
            slide_images.append(image_url)

        return slide_images

    async def _generate_voiceover(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable]
    ) -> tuple:
        """Generate voiceover audio via media-generator service

        Returns:
            tuple: (audio_url, duration_seconds, word_timestamps) or (None, 0, []) on failure
            word_timestamps: List of {"word": str, "start": float, "end": float}
        """
        # Combine all voiceover texts and clean them (remove [SYNC:slide_XXX] markers, etc.)
        raw_voiceover_text = " ".join([
            slide.voiceover_text
            for slide in job.script.slides
            if slide.voiceover_text
        ])
        voiceover_text = clean_voiceover_text(raw_voiceover_text)

        if not voiceover_text.strip():
            print("[VOICEOVER] No voiceover text found in slides after cleaning", flush=True)
            return None, 0, []

        print(f"[VOICEOVER] Cleaned voiceover text: {len(raw_voiceover_text)} -> {len(voiceover_text)} chars", flush=True)

        # Truncate if too long (max 5000 chars for API)
        if len(voiceover_text) > 4900:
            voiceover_text = voiceover_text[:4900] + "..."

        # Natural speaking pace for teaching context (0.95 speed)
        # Slightly slower than normal for clarity, but not robotic
        speech_speed = 0.95

        print(f"[VOICEOVER] Generating voiceover for {len(voiceover_text)} characters (speed: {speech_speed})", flush=True)

        # Get voice_id from user request
        voice_id = job.request.voice_id
        content_language = getattr(job.request, 'content_language', 'en') or 'en'

        # DEBUG: Log the original voice_id from request
        print(f"[VOICEOVER] Original voice_id from request: '{voice_id}'", flush=True)
        print(f"[VOICEOVER] Content language: '{content_language}'", flush=True)

        # OpenAI voice IDs (for detection)
        openai_voices = ['nova', 'shimmer', 'echo', 'onyx', 'fable', 'alloy', 'ash', 'sage', 'coral']

        # ElevenLabs is the PRIMARY provider for best quality
        # User can select any ElevenLabs voice from the frontend
        # If user selected an OpenAI voice, map to equivalent ElevenLabs voice
        # Using universally available ElevenLabs default voices
        openai_to_elevenlabs = {
            'onyx': 'pNInz6obpgDQGcFmaJgB',   # Adam - deep male multilingual (available on all accounts)
            'echo': 'VR6AewLTigWG4xSOukaG',   # Arnold - warm male
            'alloy': 'pNInz6obpgDQGcFmaJgB',  # Adam - neutral multilingual
            'nova': '21m00Tcm4TlvDq8ikWAM',   # Rachel - female calm
            'shimmer': 'EXAVITQu4vr4xnSDxMaL', # Bella - soft female
            'fable': 'ErXwobaYiN019PkySvjV',  # Antoni - expressive male
        }

        # Default ElevenLabs voice: Adam (pNInz6obpgDQGcFmaJgB)
        # Adam is available on ALL ElevenLabs accounts and supports multilingual
        DEFAULT_ELEVENLABS_VOICE = "pNInz6obpgDQGcFmaJgB"

        # Determine provider and voice
        if voice_id and voice_id not in openai_voices:
            # User selected an ElevenLabs voice ID directly
            provider = "elevenlabs"
            print(f"[VOICEOVER] Using user-selected ElevenLabs voice: {voice_id}", flush=True)
        elif voice_id in openai_voices:
            # User selected OpenAI voice - map to ElevenLabs equivalent
            provider = "elevenlabs"
            voice_id = openai_to_elevenlabs.get(voice_id, DEFAULT_ELEVENLABS_VOICE)
            print(f"[VOICEOVER] Mapped OpenAI voice to ElevenLabs: {voice_id}", flush=True)
        else:
            # No voice selected - use default ElevenLabs voice (Adam - multilingual)
            provider = "elevenlabs"
            voice_id = DEFAULT_ELEVENLABS_VOICE
            print(f"[VOICEOVER] Using default ElevenLabs voice (Adam): {voice_id}", flush=True)

        print(f"[VOICEOVER] Provider: {provider}, Voice: {voice_id}, Language: {content_language}", flush=True)

        # Build provider fallback chain
        # ElevenLabs first, then OpenAI as fallback
        providers_to_try = [(provider, voice_id)]
        # Add OpenAI fallback with onyx voice (best OpenAI voice)
        providers_to_try.append(("openai", "onyx"))

        # Try each provider in order
        for current_provider, current_voice_id in providers_to_try:
            print(f"[VOICEOVER] Trying provider: {current_provider}, voice: {current_voice_id}", flush=True)

            result = await self._try_voiceover_provider(
                voiceover_text=voiceover_text,
                provider=current_provider,
                voice_id=current_voice_id,
                speed=speech_speed,
                language=content_language
            )

            if result[0] is not None:  # Success - got audio URL
                return result

            # Check if we should try fallback
            if current_provider == "elevenlabs":
                print(f"[VOICEOVER] ElevenLabs failed, trying OpenAI fallback...", flush=True)

        # All providers failed
        print(f"[VOICEOVER] All providers failed", flush=True)
        return None, 0, []

    async def _generate_voiceover_direct(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable]
    ) -> tuple:
        """
        OPTION B+: Generate voiceover using TTS per slide with perfect synchronization.

        This method:
        1. Generates TTS for each slide in parallel
        2. Concatenates with crossfade for seamless transitions
        3. Returns a direct timeline (no SSVS needed!)

        Returns:
            tuple: (audio_url, duration_seconds, DirectTimeline) or (None, 0, None) on failure
        """
        # Prepare slides data for the generator
        slides_data = []
        for i, slide in enumerate(job.script.slides):
            # Clean voiceover text
            raw_text = slide.voiceover_text or ""
            clean_text = clean_voiceover_text(raw_text)

            # IMPORTANT: Use actual slide.id to match animation_map keys
            # animation_map is keyed by slide.id (UUID like "a1b2c3d4")
            slides_data.append({
                "id": slide.id,  # Use actual slide ID, not generated format
                "voiceover_text": clean_text,
                "type": slide.type.value if slide.type else "content",
                "title": slide.title or "",
                "image_url": slide.image_url,
            })

        if not slides_data:
            print("[DIRECT_VOICEOVER] No slides with voiceover text", flush=True)
            return None, 0, None

        # Get voice and language configuration
        voice_id = job.request.voice_id or "alloy"
        content_language = getattr(job.request, 'content_language', 'en') or 'en'

        # ElevenLabs voice mapping for OpenAI voices
        openai_voices = ['nova', 'shimmer', 'echo', 'onyx', 'fable', 'alloy', 'ash', 'sage', 'coral']

        # Language-specific default voices (use native speakers for non-English)
        # These are ElevenLabs voice IDs for each language
        language_default_voices = {
            'fr': 'IKne3meq5aSn9XLyUdCD',      # Thomas - French male professional
            'fr-CA': 'ZQe5CZNOzWyzPSCn5a3c',   # Jean-Pierre - Quebec French
            'fr-AF': 't0jbNlBVZ17f02VDIeMI',   # Mamadou - African French
            'es': 'pNInz6obpgDQGcFmaJgB',      # Adam (ES) - Spanish
            'de': 'pNInz6obpgDQGcFmaJgB',      # Adam (DE) - German
            'pt': 'pNInz6obpgDQGcFmaJgB',      # Adam (PT) - Portuguese
            'en': 'pNInz6obpgDQGcFmaJgB',      # Adam - English
        }

        openai_to_elevenlabs = {
            'onyx': 'pNInz6obpgDQGcFmaJgB',
            'echo': 'VR6AewLTigWG4xSOukaG',
            'alloy': 'pNInz6obpgDQGcFmaJgB',
            'nova': '21m00Tcm4TlvDq8ikWAM',
            'shimmer': 'EXAVITQu4vr4xnSDxMaL',
            'fable': 'ErXwobaYiN019PkySvjV',
        }

        # If user selected an OpenAI voice name AND language is not English,
        # use language-specific default voice instead
        if voice_id in openai_voices:
            if content_language != 'en' and content_language in language_default_voices:
                # Use native speaker for non-English content
                voice_id = language_default_voices[content_language]
                print(f"[DIRECT_VOICEOVER] Switched to native voice for {content_language}: {voice_id}", flush=True)
            else:
                # Map OpenAI voice name to ElevenLabs ID
                voice_id = openai_to_elevenlabs.get(voice_id, 'pNInz6obpgDQGcFmaJgB')

        print(f"[DIRECT_VOICEOVER] Generating TTS for {len(slides_data)} slides (voice: {voice_id}, lang: {content_language})", flush=True)

        try:
            # Step 1: Generate audio for each slide in parallel
            batch = await self.slide_audio_generator.generate_batch(
                slides_data,
                voice_id=voice_id,
                language=content_language,  # CRITICAL: Pass language for correct TTS pronunciation
                job_id=job.job_id
            )

            if not batch.slide_audios:
                print("[DIRECT_VOICEOVER] No audio generated", flush=True)
                return None, 0, None

            # Step 2: Concatenate with crossfade
            concat_result = await self.audio_concatenator.concatenate(
                batch,
                job_id=job.job_id
            )

            # Step 3: Build direct timeline
            direct_timeline = self.direct_timeline_builder.build(
                slides_data,
                concat_result
            )

            # Upload concatenated audio to storage (if needed)
            audio_url = None
            if concat_result.audio_path and os.path.exists(concat_result.audio_path):
                # For now, use local path converted to URL
                # In production, upload to cloud storage
                audio_url = self._get_public_url(concat_result.audio_path)

            print(f"[DIRECT_VOICEOVER] Complete: {concat_result.total_duration:.2f}s, {len(batch.slide_audios)} slides", flush=True)
            print(f"[DIRECT_VOICEOVER] Timeline sync quality: PERFECT (by construction)", flush=True)

            # Return batch as well for hybrid sync (SSVS-D diagram focus)
            return audio_url, concat_result.total_duration, direct_timeline, batch

        except Exception as e:
            print(f"[DIRECT_VOICEOVER] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None, 0, None, None

    async def _compose_video_with_direct_timeline(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable],
        animation_map: Dict[str, Dict[str, Any]],
        direct_timeline: DirectTimeline,
        hybrid_sync_result: Optional[HybridSyncResult] = None
    ) -> Optional[str]:
        """
        Compose video using the direct timeline (perfect sync).

        This uses the timeline built from actual audio durations,
        so synchronization is guaranteed to be perfect.

        If hybrid_sync_result is provided, diagram slides will have
        focus animations (highlight, zoom) applied via FFmpeg filters.
        """
        from services.timeline_builder import VisualEvent as TLVisualEvent, VisualEventType as TLVisualEventType

        try:
            print(f"[DIRECT_COMPOSE] Starting composition with {len(direct_timeline.visual_events)} events", flush=True)

            # Convert DirectTimeline visual events to Timeline format
            visual_events = []
            for event in direct_timeline.visual_events:
                slide_idx = event.slide_index
                slide = job.script.slides[slide_idx] if slide_idx < len(job.script.slides) else None

                if not slide:
                    continue

                # Map event type
                event_type_map = {
                    "slide": TLVisualEventType.SLIDE,
                    "code_animation": TLVisualEventType.CODE_ANIMATION,
                    "diagram": TLVisualEventType.DIAGRAM,
                    "freeze_frame": TLVisualEventType.FREEZE_FRAME,
                }
                tl_event_type = event_type_map.get(event.event_type.value, TLVisualEventType.SLIDE)

                # Check for animation - use actual slide.id (UUID) to match animation_map keys
                actual_slide_id = slide.id  # UUID like "a1b2c3d4"
                if actual_slide_id in animation_map and tl_event_type == TLVisualEventType.CODE_ANIMATION:
                    anim = animation_map[actual_slide_id]
                    asset_url = anim.get("url")
                    asset_path = anim.get("file_path")
                    print(f"[DIRECT_COMPOSE] Using animation for slide {actual_slide_id}: {asset_url}", flush=True)
                else:
                    asset_url = slide.image_url
                    asset_path = None

                # Build metadata with optional diagram focus info
                event_metadata = {
                    "slide_id": actual_slide_id,
                    "slide_index": slide_idx,
                    "title": slide.title or ""
                }

                # Add diagram focus animations if available (from SSVS-D hybrid sync)
                if hybrid_sync_result and slide.id in hybrid_sync_result.diagram_focus:
                    focus_result = hybrid_sync_result.diagram_focus[slide.id]
                    event_metadata["diagram_focus"] = {
                        "enabled": True,
                        "ffmpeg_filter": focus_result.ffmpeg_filter,
                        "animation_timeline": focus_result.animation_timeline,
                        "semantic_score": focus_result.semantic_score,
                        "coverage_score": focus_result.coverage_score,
                        "focus_points": len(focus_result.focus_sequence)
                    }
                    print(f"[DIRECT_COMPOSE] Slide {actual_slide_id}: Adding diagram focus ({len(focus_result.focus_sequence)} focus points)", flush=True)

                visual_events.append(TLVisualEvent(
                    event_type=tl_event_type,
                    time_start=event.time_start,
                    time_end=event.time_end,
                    duration=event.duration,
                    asset_path=asset_path,
                    asset_url=asset_url,
                    layer=0,
                    metadata=event_metadata
                ))

            # Create a Timeline object compatible with SimpleTimelineCompositor
            timeline = Timeline(
                total_duration=direct_timeline.total_duration,
                audio_track_path=direct_timeline.audio_path,
                audio_track_url=None,
                visual_events=visual_events,
                word_timestamps=[],
                sync_anchors=[],
                sync_method="direct",
                metadata={"sync_quality": "perfect"}
            )

            # Log summary
            diagrams_with_focus = sum(1 for e in visual_events if e.metadata.get("diagram_focus", {}).get("enabled"))
            print(f"[DIRECT_COMPOSE] Composing {len(visual_events)} events, duration: {direct_timeline.total_duration:.2f}s", flush=True)
            print(f"[DIRECT_COMPOSE] Audio path: {direct_timeline.audio_path}", flush=True)
            if diagrams_with_focus > 0:
                print(f"[DIRECT_COMPOSE] Diagram focus animations: {diagrams_with_focus} slides (SSVS-D hybrid sync)", flush=True)

            # Use SimpleTimelineCompositor's compose method
            output_filename = f"{job.job_id}_final.mp4"
            result = await self.timeline_compositor.compose(
                timeline=timeline,
                output_filename=output_filename,
                resolution=(1920, 1080),
                fps=30
            )

            if result.success and result.output_path and os.path.exists(result.output_path):
                output_url = self._get_public_url(result.output_path)
                print(f"[DIRECT_COMPOSE] Success! Output: {output_url}", flush=True)
                return output_url

            print(f"[DIRECT_COMPOSE] Composition failed: {result.error}", flush=True)
            return None

        except Exception as e:
            print(f"[DIRECT_COMPOSE] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    async def _try_voiceover_provider(
        self,
        voiceover_text: str,
        provider: str,
        voice_id: str,
        speed: float,
        language: str
    ) -> tuple:
        """
        Try generating voiceover with a specific provider.

        Returns:
            tuple: (audio_url, duration_seconds, word_timestamps) or (None, 0, []) on failure
        """
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Submit voiceover job
            response = await client.post(
                f"{self.media_generator_url}/api/v1/media/voiceover",
                json={
                    "text": voiceover_text,
                    "provider": provider,
                    "voice_id": voice_id,
                    "speed": speed,
                    "language": language
                }
            )

            if response.status_code != 200:
                print(f"[VOICEOVER] Error submitting job: {response.status_code} - {response.text}", flush=True)
                return None, 0, []

            result = response.json()
            voiceover_job_id = result.get("job_id")

            if not voiceover_job_id:
                print(f"[VOICEOVER] No job_id in response: {result}", flush=True)
                return None, 0, []

            print(f"[VOICEOVER] Job submitted: {voiceover_job_id}", flush=True)

            # Poll for job completion with retry on connection errors
            max_attempts = 60  # 5 minutes max
            connection_errors = 0
            max_connection_errors = 5

            for attempt in range(max_attempts):
                await asyncio.sleep(5)  # Wait 5 seconds between polls

                try:
                    status_response = await client.get(
                        f"{self.media_generator_url}/api/v1/media/jobs/{voiceover_job_id}"
                    )
                except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as e:
                    connection_errors += 1
                    print(f"[VOICEOVER] Connection error ({connection_errors}/{max_connection_errors}): {type(e).__name__}: {e}", flush=True)
                    if connection_errors >= max_connection_errors:
                        print(f"[VOICEOVER] Too many connection errors, aborting", flush=True)
                        return None, 0, []
                    await asyncio.sleep(2)  # Extra wait before retry
                    continue
                except Exception as e:
                    print(f"[VOICEOVER] Unexpected error polling job: {type(e).__name__}: {e}", flush=True)
                    continue

                # Reset connection error count on success
                connection_errors = 0

                if status_response.status_code != 200:
                    print(f"[VOICEOVER] Error polling job: {status_response.status_code}", flush=True)
                    continue

                status_data = status_response.json()
                status = status_data.get("status")

                if status == "completed":
                    output_data = status_data.get("output_data", {})
                    # Try different possible field names
                    audio_url = output_data.get("url") or output_data.get("audio_url")
                    duration = output_data.get("duration_seconds", 0)
                    print(f"[VOICEOVER] Completed! Audio URL: {audio_url}, Duration: {duration}s", flush=True)

                    # Extract word-level timestamps using Whisper
                    word_timestamps = []
                    if audio_url:
                        word_timestamps = await self._extract_word_timestamps(
                            audio_url, voiceover_text, language
                        )
                        print(f"[VOICEOVER] Extracted {len(word_timestamps)} word timestamps", flush=True)

                    return audio_url, duration, word_timestamps

                elif status == "failed":
                    error = status_data.get("error_message", "Unknown error")
                    print(f"[VOICEOVER] Job failed with {provider}: {error}", flush=True)
                    return None, 0, []

                # Still processing
                progress = status_data.get("progress_percent", 0)
                print(f"[VOICEOVER] Progress: {progress}%", flush=True)

            print("[VOICEOVER] Timeout waiting for job completion", flush=True)
            return None, 0, []

    async def _extract_word_timestamps(
        self,
        audio_url: str,
        original_text: str,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Extract word-level timestamps from audio using Whisper.

        Args:
            audio_url: URL to the audio file
            original_text: Original text for reference
            language: Audio language code

        Returns:
            List of {"word": str, "start": float, "end": float}
        """
        import tempfile
        from openai import AsyncOpenAI

        temp_path = None
        try:
            # Download audio to temp file
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(audio_url)
                if response.status_code != 200:
                    print(f"[TIMESTAMPS] Failed to download audio: {response.status_code}", flush=True)
                    return self._estimate_word_timestamps(original_text, language)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(response.content)
                    temp_path = f.name

            # Use OpenAI Whisper for transcription with word timestamps
            openai_client = AsyncOpenAI()

            with open(temp_path, "rb") as audio_file:
                transcript = await openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"]
                )

            # Parse word timestamps from response
            word_timestamps = []

            if hasattr(transcript, 'words') and transcript.words:
                for word_data in transcript.words:
                    word = word_data.word if hasattr(word_data, 'word') else word_data.get("word", "")
                    start = float(word_data.start if hasattr(word_data, 'start') else word_data.get("start", 0))
                    end = float(word_data.end if hasattr(word_data, 'end') else word_data.get("end", 0))

                    word_timestamps.append({
                        "word": word,
                        "start": start,
                        "end": end
                    })

                print(f"[TIMESTAMPS] Extracted {len(word_timestamps)} timestamps from Whisper", flush=True)
                return word_timestamps

            # Fallback to estimates if Whisper didn't return word-level data
            print("[TIMESTAMPS] Whisper did not return word-level timestamps, using estimates", flush=True)
            return self._estimate_word_timestamps(original_text, language)

        except Exception as e:
            print(f"[TIMESTAMPS] Error extracting timestamps: {e}, using estimates", flush=True)
            return self._estimate_word_timestamps(original_text, language)
        finally:
            # Guaranteed cleanup of temp file
            if temp_path:
                try:
                    import os
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception:
                    pass

    def _estimate_word_timestamps(
        self,
        text: str,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Estimate word timestamps when Whisper fails.

        Uses language-specific speech rates for more accurate estimates.
        """
        words = text.split()
        timestamps = []

        # Language-specific speech rates (words per second)
        language_wps = {
            "en": 2.5,   # ~150 WPM
            "fr": 3.0,   # ~180 WPM - French spoken faster
            "es": 3.2,   # ~190 WPM - Spanish spoken faster
            "de": 2.4,   # ~145 WPM - German slightly slower
            "it": 3.0,   # ~180 WPM
            "pt": 2.8,   # ~170 WPM
            "nl": 2.6,   # ~155 WPM
            "pl": 2.7,   # ~160 WPM
            "ru": 2.5,   # ~150 WPM
            "zh": 3.5,   # Chinese has shorter "words"
        }

        words_per_second = language_wps.get(language, 2.5)
        current_time = 0.0

        for word in words:
            # Longer words take slightly longer
            char_factor = 0.04 if language in ["fr", "es", "it"] else 0.05
            word_duration = max(0.15, len(word) * char_factor + 0.15)

            timestamps.append({
                "word": word,
                "start": current_time,
                "end": current_time + word_duration
            })

            current_time += word_duration + 0.08  # Gap between words

        return timestamps

    async def _adjust_slide_durations(
        self,
        job: PresentationJob,
        voiceover_duration: float
    ):
        """Adjust slide durations to match actual voiceover duration EXACTLY.

        IMPROVED ALGORITHM (v2):
        - Uses CUMULATIVE calculation to prevent rounding drift
        - Cleans voiceover text before calculating lengths
        - Rounds to milliseconds for FFmpeg precision
        - Last slide is adjusted to fill remaining time exactly

        CRITICAL: Total slide duration MUST equal voiceover duration for sync.
        """
        import re
        slides = job.script.slides

        # Clean voiceover text and calculate character counts
        # Remove [SYNC:...] markers before counting
        sync_pattern = re.compile(r'\[SYNC:[\w_]+\]', re.IGNORECASE)
        char_counts = []
        for slide in slides:
            raw_text = slide.voiceover_text or ""
            clean_text = sync_pattern.sub("", raw_text).strip()
            char_counts.append(len(clean_text))

        total_chars = sum(char_counts)
        original_total = sum(slide.duration for slide in slides)

        if total_chars == 0:
            # Fallback: distribute evenly
            duration_per_slide = round(voiceover_duration / len(slides), 3)
            for slide in slides:
                slide.duration = duration_per_slide
            # Adjust last slide to fill remaining time
            slides[-1].duration = round(voiceover_duration - duration_per_slide * (len(slides) - 1), 3)
            print(f"[SYNC] Distributed {voiceover_duration}s evenly across {len(slides)} slides", flush=True)
            return

        # Calculate CUMULATIVE durations to prevent drift
        # Each slide's end time is calculated from total_duration directly
        cumulative_time = 0.0

        for i, slide in enumerate(slides):
            char_count = char_counts[i]

            if i == len(slides) - 1:
                # Last slide: fill remaining time exactly
                slide.duration = round(voiceover_duration - cumulative_time, 3)
            else:
                # Calculate this slide's proportion of total
                if char_count > 0:
                    proportion = char_count / total_chars
                    slide.duration = round(voiceover_duration * proportion, 3)
                else:
                    slide.duration = 0.5  # Minimal duration for empty slides

            # Ensure minimum duration
            slide.duration = max(slide.duration, 0.5)

            cumulative_time += slide.duration

        # Final adjustment: ensure total exactly matches voiceover duration
        # This corrects any accumulated rounding errors
        new_total = sum(slide.duration for slide in slides)
        if abs(new_total - voiceover_duration) > 0.001:
            # Adjust last slide to compensate
            correction = voiceover_duration - new_total
            slides[-1].duration = round(slides[-1].duration + correction, 3)
            slides[-1].duration = max(slides[-1].duration, 0.5)

        new_total = sum(slide.duration for slide in slides)
        print(f"[SYNC] Adjusted slide durations: {original_total:.3f}s -> {new_total:.3f}s (voiceover: {voiceover_duration:.3f}s)", flush=True)

        # Debug: print each slide's duration
        for i, slide in enumerate(slides):
            print(f"[SYNC]   Slide {i}: {slide.duration:.3f}s ({char_counts[i]} chars)", flush=True)

        # Verify sync
        if abs(new_total - voiceover_duration) > 0.01:
            print(f"[SYNC] WARNING: Duration mismatch! Slides={new_total:.3f}s, Audio={voiceover_duration:.3f}s", flush=True)

    async def _compose_video(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable],
        animation_map: Dict[str, str] = None
    ) -> str:
        """Compose final video via media-generator slideshow endpoint

        Args:
            job: The presentation job
            on_progress: Progress callback
            animation_map: Dict mapping slide_id to animation video URL
        """
        print(f"[COMPOSE] Starting video composition...", flush=True)
        animation_map = animation_map or {}

        # Build slideshow scenes
        # Use slide.duration for all scenes to stay synchronized with voiceover
        scenes = []
        for i, slide in enumerate(job.script.slides):
            # Check if this slide has a typing animation video
            if slide.id in animation_map:
                anim_info = animation_map[slide.id]
                # Use animation video but with slide duration (synced to voiceover)
                # The video compositor will handle trimming/looping as needed
                scenes.append({
                    "video_url": anim_info["url"],
                    "duration": slide.duration,  # Use voiceover-synced duration
                    "transition": slide.transition or "fade"
                })
                print(f"[COMPOSE] Slide {i}: animation video, duration {slide.duration:.1f}s", flush=True)
            else:
                # Use static image
                scenes.append({
                    "image_url": slide.image_url,
                    "duration": slide.duration,
                    "transition": slide.transition or "fade"
                })

        print(f"[COMPOSE] {len(scenes)} scenes ({len(animation_map)} animations), voiceover: {bool(job.voiceover_url)}", flush=True)

        # Call media-generator slideshow endpoint
        async with httpx.AsyncClient(timeout=600.0) as client:
            slideshow_request = {
                "scenes": scenes,
                "voiceover_url": job.voiceover_url,
                "output_format": "16:9",
                "quality": "1080p",
                "fps": 30
            }

            # Add PIP avatar (medallion) if avatar video was generated
            if job.avatar_video_url:
                slideshow_request["pip_avatar_url"] = job.avatar_video_url
                slideshow_request["pip_position"] = "bottom-right"
                slideshow_request["pip_size"] = 0.20  # 20% of video width
                slideshow_request["pip_circular"] = True  # Medallion style
                print(f"[COMPOSE] Adding PIP avatar: {job.avatar_video_url}", flush=True)

            response = await client.post(
                f"{self.media_generator_url}/api/v1/media/slideshow/compose",
                json=slideshow_request
            )

            if response.status_code != 200:
                print(f"[COMPOSE] Error submitting job: {response.status_code} - {response.text}", flush=True)
                return None

            result = response.json()
            compose_job_id = result.get("job_id")

            if not compose_job_id:
                print(f"[COMPOSE] No job_id in response: {result}", flush=True)
                return None

            print(f"[COMPOSE] Job submitted: {compose_job_id}", flush=True)

            # Poll for job completion with retry logic
            max_attempts = 120  # 10 minutes max for video composition
            consecutive_errors = 0
            max_consecutive_errors = 5

            for attempt in range(max_attempts):
                await asyncio.sleep(5)

                try:
                    status_response = await client.get(
                        f"{self.media_generator_url}/api/v1/media/jobs/{compose_job_id}",
                        timeout=30.0
                    )

                    if status_response.status_code != 200:
                        print(f"[COMPOSE] Error polling job: {status_response.status_code}", flush=True)
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"[COMPOSE] Too many consecutive errors, aborting", flush=True)
                            return None
                        continue

                    consecutive_errors = 0  # Reset on success

                except Exception as poll_error:
                    consecutive_errors += 1
                    print(f"[COMPOSE] Poll error ({consecutive_errors}/{max_consecutive_errors}): {poll_error}", flush=True)
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[COMPOSE] Too many consecutive errors, aborting", flush=True)
                        return None
                    await asyncio.sleep(2)  # Extra wait on error
                    continue

                status_data = status_response.json()
                status = status_data.get("status")
                progress = status_data.get("progress_percent", 0)

                # Update presentation job progress (60-95% range)
                scaled_progress = 60 + int(progress * 0.35)
                job.update_progress(
                    PresentationStage.COMPOSING_VIDEO,
                    scaled_progress,
                    f"Composing video... {progress}%"
                )
                await self._notify_progress(job, on_progress)

                if status == "completed":
                    output_data = status_data.get("output_data", {})
                    video_url = output_data.get("video_url")
                    duration = output_data.get("duration_seconds", 0)
                    print(f"[COMPOSE] Completed! Video URL: {video_url}, Duration: {duration}s", flush=True)
                    return video_url

                elif status == "failed":
                    error = status_data.get("error_message", "Unknown error")
                    print(f"[COMPOSE] Job failed: {error}", flush=True)
                    return None

                print(f"[COMPOSE] Progress: {progress}%", flush=True)

            print("[COMPOSE] Timeout waiting for job completion", flush=True)
            return None

    async def _compose_video_with_timeline(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable],
        animation_map: Dict[str, Dict[str, Any]],
        word_timestamps: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Compose video using timeline-based synchronization.

        This method provides millisecond-precision sync between audio and visuals
        by using word-level timestamps to drive the timeline.

        Args:
            job: The presentation job
            on_progress: Progress callback
            animation_map: Dict mapping slide_id to animation info
            word_timestamps: List of {"word": str, "start": float, "end": float}

        Returns:
            URL to the composed video
        """
        print(f"[TIMELINE_COMPOSE] Starting timeline-based composition...", flush=True)

        try:
            # Build slides data for timeline
            slides_data = []
            for slide in job.script.slides:
                slide_data = {
                    "id": slide.id,
                    "type": slide.type.value if hasattr(slide.type, 'value') else str(slide.type),
                    "title": slide.title,
                    "voiceover_text": slide.voiceover_text or "",
                    "image_url": slide.image_url,
                    "duration": slide.duration,
                    "language": job.script.language
                }

                if slide.code_blocks:
                    slide_data["code_blocks"] = [
                        {"code": cb.code, "language": cb.language}
                        for cb in slide.code_blocks
                    ]

                slides_data.append(slide_data)

            # Get audio duration
            audio_duration = sum(s.duration for s in job.script.slides)
            if word_timestamps:
                audio_duration = max(audio_duration, word_timestamps[-1].get("end", 0) + 0.5)

            # Build timeline
            timeline = self.timeline_builder.build(
                word_timestamps=word_timestamps,
                slides=slides_data,
                audio_duration=audio_duration,
                audio_url=job.voiceover_url,
                animations=animation_map
            )

            print(f"[TIMELINE_COMPOSE] Timeline built: {len(timeline.visual_events)} events", flush=True)

            # Update progress
            job.update_progress(
                PresentationStage.COMPOSING_VIDEO,
                65,
                "Composing video with timeline sync..."
            )
            await self._notify_progress(job, on_progress)

            # Compose using timeline compositor
            output_filename = f"{job.job_id}_timeline.mp4"
            result = await self.timeline_compositor.compose(
                timeline=timeline,
                output_filename=output_filename,
                resolution=(1920, 1080),
                fps=30
            )

            if result.success and result.output_path:
                final_output_path = result.output_path

                # Add PIP avatar overlay if available
                if job.avatar_video_url:
                    print(f"[TIMELINE_COMPOSE] Adding PIP avatar overlay...", flush=True)
                    pip_output_filename = f"{job.job_id}_timeline_pip.mp4"
                    pip_output_path = str(self.output_dir / pip_output_filename)

                    pip_result = await self._add_pip_overlay_ffmpeg(
                        input_video_path=result.output_path,
                        avatar_video_url=job.avatar_video_url,
                        output_path=pip_output_path,
                        position="bottom-right",
                        size=0.20,
                        circular=True
                    )

                    if pip_result and pip_result != result.output_path:
                        final_output_path = pip_result
                        output_filename = pip_output_filename
                        print(f"[TIMELINE_COMPOSE] PIP overlay added!", flush=True)

                # Generate public URL for the video
                video_url = self._get_public_url(f"output/{output_filename}")

                print(f"[TIMELINE_COMPOSE] Completed! Video URL: {video_url}", flush=True)

                job.update_progress(
                    PresentationStage.COMPOSING_VIDEO,
                    95,
                    "Video composition complete!"
                )
                await self._notify_progress(job, on_progress)

                return video_url
            else:
                print(f"[TIMELINE_COMPOSE] Composition failed: {result.error}", flush=True)
                return None

        except Exception as e:
            print(f"[TIMELINE_COMPOSE] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None

    async def _execute_code_demos(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable]
    ) -> Dict[str, dict]:
        """Execute code in CODE_DEMO slides and capture output"""
        print("[EXECUTE] Starting code execution...", flush=True)

        results = {}
        demo_slides = [
            s for s in job.script.slides
            if s.type == SlideType.CODE_DEMO and s.code_blocks
        ]

        if not demo_slides:
            print("[EXECUTE] No CODE_DEMO slides found", flush=True)
            return results

        for i, slide in enumerate(demo_slides):
            print(f"[EXECUTE] Processing slide {i + 1}/{len(demo_slides)}: {slide.title}", flush=True)

            for j, code_block in enumerate(slide.code_blocks):
                key = f"{slide.id}_{j}"

                result = await self.code_executor.execute(
                    code=code_block.code,
                    language=code_block.language,
                    timeout=30
                )

                results[key] = {
                    "slide_id": slide.id,
                    "code_block_index": j,
                    "success": result.success,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "execution_time": result.execution_time,
                    "error": result.error
                }

                if result.success:
                    # Store output in the code block for later use
                    code_block.expected_output = result.stdout
                    print(f"[EXECUTE] Block {key}: SUCCESS - {result.stdout[:50] if result.stdout else '(no output)'}", flush=True)
                else:
                    print(f"[EXECUTE] Block {key}: FAILED - {result.error or result.stderr}", flush=True)

        print(f"[EXECUTE] Completed {len(results)} code executions", flush=True)
        return results

    async def _create_typing_animations(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable]
    ) -> Dict[str, dict]:
        """Create typing animation videos for code slides with SSVS-C synchronization.

        Animations are synchronized with the voiceover using SSVS-C (Code-Aware Synchronizer).
        This ensures that when the narrator describes a function, that function is being typed.

        Returns:
            Dict mapping slide_id to {"url": str}
        """
        print("[ANIMATION] Starting typing animation creation with SSVS-C sync...", flush=True)

        animation_map = {}  # slide_id -> {"url": ...}

        # Diagnostic logging for code slides
        all_slides_info = [(s.type, s.title, bool(s.code_blocks), len(s.code_blocks) if s.code_blocks else 0) for s in job.script.slides]
        print(f"[ANIMATION] All slides: {len(job.script.slides)}", flush=True)
        for i, (stype, title, has_cb, cb_count) in enumerate(all_slides_info):
            if stype in [SlideType.CODE, SlideType.CODE_DEMO]:
                print(f"[ANIMATION] Slide {i}: type={stype.value}, title='{title}', code_blocks={cb_count}", flush=True)

        code_slides = [
            s for s in job.script.slides
            if s.type in [SlideType.CODE, SlideType.CODE_DEMO] and s.code_blocks
        ]

        if not code_slides:
            print("[ANIMATION] No code slides with code_blocks found - check if LLM generated code_blocks", flush=True)
            # Additional diagnostic: show CODE slides without code_blocks
            code_type_slides = [s for s in job.script.slides if s.type in [SlideType.CODE, SlideType.CODE_DEMO]]
            if code_type_slides:
                print(f"[ANIMATION] WARNING: Found {len(code_type_slides)} CODE slides but none have code_blocks!", flush=True)
            return animation_map

        # Create temp directory for animation outputs
        temp_dir = Path(tempfile.gettempdir()) / "presentations" / "animations"
        temp_dir.mkdir(parents=True, exist_ok=True)

        colors = self.slide_generator.get_style_colors(job.request.style)

        # Initialize SSVS-C synchronizer
        code_synchronizer = CodeAwareSynchronizer()

        for i, slide in enumerate(code_slides):
            print(f"[ANIMATION] Processing slide {i + 1}/{len(code_slides)}: {slide.title}", flush=True)

            # Use first code block for the animation
            if slide.code_blocks:
                code_block = slide.code_blocks[0]
                output_filename = f"{job.job_id}_typing_{slide.id}.mp4"
                output_path = str(temp_dir / output_filename)

                # Get execution output if available
                execution_output = None
                if job.code_execution_results:
                    result_key = f"{slide.id}_0"
                    exec_result = job.code_execution_results.get(result_key)
                    if exec_result and exec_result.get("success"):
                        execution_output = exec_result.get("stdout")

                try:
                    # Get typing speed from request (default: natural)
                    typing_speed = job.request.typing_speed.value if job.request.typing_speed else "natural"

                    # Unescape literal \n to actual newlines (GPT sometimes double-escapes)
                    code_content = code_block.code
                    if '\\n' in code_content:
                        code_content = code_content.replace('\\n', '\n')
                    if '\\t' in code_content:
                        code_content = code_content.replace('\\t', '\t')

                    # Get code display mode from request (default: reveal for fast generation)
                    code_display_mode = job.request.code_display_mode if job.request.code_display_mode else CodeDisplayMode.REVEAL

                    # Override with env var for backward compatibility
                    env_force_typing = os.getenv("FORCE_TYPING_ANIMATION", "false").lower() == "true"
                    if env_force_typing:
                        code_display_mode = CodeDisplayMode.TYPING

                    print(f"[ANIMATION] Code display mode: {code_display_mode.value}", flush=True)

                    # Initialize sync variables
                    reveal_points = None
                    use_sync_mode = False
                    force_typing = (code_display_mode == CodeDisplayMode.TYPING)
                    force_static = (code_display_mode == CodeDisplayMode.STATIC)

                    voiceover_text = slide.voiceover_text or ""

                    # REVEAL mode: Use SSVS-C for line-by-line reveal synced with voiceover
                    if code_display_mode == CodeDisplayMode.REVEAL and voiceover_text and slide.duration > 0:
                        try:
                            # Estimate word timestamps from voiceover text and duration
                            voice_segments = self._estimate_voice_segments(voiceover_text, slide.duration)

                            if voice_segments:
                                # Run SSVS-C synchronization
                                sync_result = code_synchronizer.synchronize(
                                    code=code_content,
                                    language=code_block.language,
                                    segments=voice_segments
                                )

                                if sync_result and sync_result.reveal_sequence:
                                    reveal_points = [
                                        {
                                            "element_id": rp.element_id,
                                            "start_line": rp.start_line,
                                            "end_line": rp.end_line,
                                            "reveal_time": rp.reveal_time,
                                            "hold_time": rp.hold_time,
                                            "reveal_type": rp.reveal_type,
                                            "confidence": rp.confidence
                                        }
                                        for rp in sync_result.reveal_sequence
                                    ]
                                    use_sync_mode = True
                                    print(f"[ANIMATION] SSVS-C: {len(reveal_points)} reveal points for {code_block.language} code", flush=True)
                        except Exception as sync_error:
                            print(f"[ANIMATION] SSVS-C sync failed (using fallback): {sync_error}", flush=True)

                    # TYPING mode: Character-by-character animation (slower generation)
                    elif code_display_mode == CodeDisplayMode.TYPING:
                        print(f"[ANIMATION] TYPING mode: Character-by-character animation (may take longer)", flush=True)

                    # STATIC mode: Instant display, no animation
                    elif code_display_mode == CodeDisplayMode.STATIC:
                        print(f"[ANIMATION] STATIC mode: Code will appear instantly", flush=True)

                    video_path, actual_duration = await self.typing_animator.create_typing_animation(
                        code=code_content,
                        language=code_block.language,
                        output_path=output_path,
                        title=slide.title,
                        typing_speed=typing_speed,
                        target_duration=slide.duration,
                        execution_output=execution_output,
                        fps=30,
                        background_color=colors["background"] if "linear" not in colors["background"] else "#1e1e2e",
                        text_color=colors["text"],
                        accent_color=colors["accent"],
                        pygments_style=colors["pygments_style"],
                        # Code display mode parameters
                        reveal_points=reveal_points,
                        sync_mode=use_sync_mode,
                        force_static=force_static,
                        force_typing=force_typing
                    )

                    # Get URL for the animation (internal URL for composition)
                    animation_url = f"{self.service_url}/files/presentations/animations/{output_filename}"

                    # Store in map
                    animation_map[slide.id] = {"url": animation_url}

                    mode_info = ""
                    if use_sync_mode:
                        mode_info = f" [REVEAL: {len(reveal_points)} sync points]"
                    elif force_typing:
                        mode_info = " [TYPING: char-by-char]"
                    elif force_static:
                        mode_info = " [STATIC]"
                    print(f"[ANIMATION] Created: {animation_url} (target: {slide.duration:.1f}s, actual: {actual_duration:.1f}s){mode_info}", flush=True)

                except Exception as e:
                    print(f"[ANIMATION] Error creating animation: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

        print(f"[ANIMATION] Completed {len(animation_map)} animations", flush=True)
        return animation_map

    def _estimate_voice_segments(
        self,
        voiceover_text: str,
        duration: float
    ) -> List[VoiceSegment]:
        """Estimate voice segments from voiceover text and duration.
        
        Splits text into sentence-like segments and estimates timing based on
        word count and total duration.
        
        Args:
            voiceover_text: The voiceover narration text
            duration: Total audio duration in seconds
            
        Returns:
            List of VoiceSegment objects with estimated timing
        """
        import re
        
        if not voiceover_text or duration <= 0:
            return []
        
        # Split into sentences (on . ! ? : and newlines)
        sentences = re.split(r'(?<=[.!?:])\s+|\n+', voiceover_text.strip())
        
        if not sentences:
            return []
        
        # Calculate total word count for proportional timing
        total_words = sum(len(s.split()) for s in sentences)
        if total_words == 0:
            return []
        
        # Generate segments with proportional timing
        segments = []
        current_time = 0.0
        
        for i, sentence in enumerate(sentences):
            word_count = len(sentence.split())
            # Duration proportional to word count
            segment_duration = (word_count / total_words) * duration
            
            segments.append(VoiceSegment(
                id=i,
                text=sentence,
                start_time=current_time,
                end_time=current_time + segment_duration
            ))
            
            current_time += segment_duration
        
        return segments

    async def _generate_avatar(
        self,
        job: PresentationJob,
        on_progress: Optional[Callable]
    ) -> Optional[str]:
        """Generate avatar video via media-generator"""
        print(f"[AVATAR] Generating avatar video...", flush=True)

        if not job.voiceover_url:
            print("[AVATAR] No voiceover URL available", flush=True)
            return None

        async with httpx.AsyncClient(timeout=600.0) as client:
            # Submit avatar generation job
            response = await client.post(
                f"{self.media_generator_url}/api/v1/media/avatars/generate",
                json={
                    "avatar_id": job.request.avatar_id,
                    "audio_url": job.voiceover_url,
                    "output_format": "16:9"
                }
            )

            if response.status_code != 200:
                print(f"[AVATAR] Error submitting job: {response.status_code} - {response.text}", flush=True)
                return None

            result = response.json()
            avatar_job_id = result.get("job_id")

            if not avatar_job_id:
                print(f"[AVATAR] No job_id in response: {result}", flush=True)
                return None

            print(f"[AVATAR] Job submitted: {avatar_job_id}", flush=True)

            # Poll for completion
            max_attempts = 120
            for attempt in range(max_attempts):
                await asyncio.sleep(5)

                status_response = await client.get(
                    f"{self.media_generator_url}/api/v1/media/jobs/{avatar_job_id}"
                )

                if status_response.status_code != 200:
                    continue

                status_data = status_response.json()
                status = status_data.get("status")

                if status == "completed":
                    output_data = status_data.get("output_data", {})
                    video_url = output_data.get("video_url")
                    print(f"[AVATAR] Completed! Video URL: {video_url}", flush=True)
                    return video_url

                elif status == "failed":
                    error = status_data.get("error_message", "Unknown error")
                    print(f"[AVATAR] Job failed: {error}", flush=True)
                    return None

            print("[AVATAR] Timeout waiting for job completion", flush=True)
            return None

    async def _add_pip_overlay_ffmpeg(
        self,
        input_video_path: str,
        avatar_video_url: str,
        output_path: str,
        position: str = "bottom-right",
        size: float = 0.20,
        circular: bool = True
    ) -> Optional[str]:
        """
        Add PIP avatar overlay using FFmpeg directly.

        Args:
            input_video_path: Path to the input video
            avatar_video_url: URL to the avatar video
            output_path: Path for the output video
            position: PIP position (bottom-right, bottom-left, top-right, top-left)
            size: PIP size as fraction of video width (0.1-0.35)
            circular: Use circular mask for medallion style

        Returns:
            Path to the output video with PIP overlay
        """
        import subprocess

        print(f"[PIP_OVERLAY] Adding avatar overlay to {input_video_path}", flush=True)

        # Download avatar video to temp file
        avatar_temp = Path(tempfile.gettempdir()) / f"pip_avatar_{os.path.basename(input_video_path)}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.get(avatar_video_url)
                if response.status_code == 200:
                    with open(avatar_temp, "wb") as f:
                        f.write(response.content)
                else:
                    print(f"[PIP_OVERLAY] Failed to download avatar: {response.status_code}", flush=True)
                    return input_video_path
        except Exception as e:
            print(f"[PIP_OVERLAY] Error downloading avatar: {e}", flush=True)
            return input_video_path

        # Get video dimensions (assume 1920x1080 for 16:9)
        width, height = 1920, 1080

        # Calculate PIP dimensions
        pip_width = int(width * size)
        pip_height = int(pip_width * 9 / 16)  # Maintain 16:9 aspect ratio
        margin = 20

        # Calculate position
        if position == "bottom-right":
            x_pos = width - pip_width - margin
            y_pos = height - pip_height - margin - 100  # Extra margin for captions
        elif position == "bottom-left":
            x_pos = margin
            y_pos = height - pip_height - margin - 100
        elif position == "top-right":
            x_pos = width - pip_width - margin
            y_pos = margin
        else:  # top-left
            x_pos = margin
            y_pos = margin

        # Build filter
        if circular:
            circle_radius = min(pip_width, pip_height) // 2
            cx = pip_width // 2
            cy = pip_height // 2
            mask_filter = f"geq=lum='lum(X,Y)':a='if(lte(hypot(X-{cx},Y-{cy}),{circle_radius}),255,0)'"
            filter_complex = (
                f"[1:v]scale={pip_width}:{pip_height},format=rgba,{mask_filter}[pip];"
                f"[0:v][pip]overlay={x_pos}:{y_pos}:shortest=1"
            )
        else:
            filter_complex = (
                f"[1:v]scale={pip_width}:{pip_height}[pip];"
                f"[0:v][pip]overlay={x_pos}:{y_pos}:shortest=1"
            )

        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video_path,
            "-stream_loop", "-1",
            "-i", str(avatar_temp),
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "copy",
            "-shortest",
            output_path
        ]

        print(f"[PIP_OVERLAY] Running FFmpeg for medallion overlay...", flush=True)
        print(f"[PIP_OVERLAY] Position: {position}, Size: {pip_width}x{pip_height}, Circular: {circular}", flush=True)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode != 0:
                print(f"[PIP_OVERLAY] FFmpeg error: {stderr.decode()[:500]}", flush=True)
                return input_video_path

            print(f"[PIP_OVERLAY] Avatar overlay added successfully", flush=True)
            return output_path

        except asyncio.TimeoutError:
            print("[PIP_OVERLAY] FFmpeg timeout, returning original video", flush=True)
            return input_video_path
        except Exception as e:
            print(f"[PIP_OVERLAY] Error: {e}", flush=True)
            return input_video_path
        finally:
            # Clean up temp avatar file
            if avatar_temp.exists():
                avatar_temp.unlink()

    def get_job(self, job_id: str) -> Optional[PresentationJob]:
        """Get a job by ID"""
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list:
        """List recent jobs"""
        jobs = sorted(
            self.jobs.values(),
            key=lambda j: j.created_at,
            reverse=True
        )
        return jobs[:limit]
