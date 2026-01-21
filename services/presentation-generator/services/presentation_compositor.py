"""
Presentation Compositor Service

Main orchestrator that coordinates all services to generate the final presentation video.
"""
import asyncio
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
import httpx

from models.presentation_models import (
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

        # Job storage (in production, use Redis)
        self.jobs: Dict[str, PresentationJob] = {}

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

            voiceover_url, voiceover_duration = await self._generate_voiceover(job, on_progress)
            job.voiceover_url = voiceover_url

            # Adjust slide durations to match actual voiceover duration
            # BUT preserve minimum durations for code slides with animations
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
        """Generate all slide images"""
        slide_images = []
        total_slides = len(job.script.slides)

        for i, slide in enumerate(job.script.slides):
            progress = 15 + (i / total_slides) * 25
            job.update_progress(
                PresentationStage.GENERATING_SLIDES,
                progress,
                f"Generating slide {i + 1}/{total_slides}..."
            )
            await self._notify_progress(job, on_progress)

            # Generate slide image
            image_bytes = await self.slide_generator.generate_slide_image(
                slide,
                job.request.style
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
            tuple: (audio_url, duration_seconds) or (None, 0) on failure
        """
        # Combine all voiceover texts
        voiceover_text = " ".join([
            slide.voiceover_text
            for slide in job.script.slides
            if slide.voiceover_text
        ])

        if not voiceover_text.strip():
            print("[VOICEOVER] No voiceover text found in slides", flush=True)
            return None, 0

        # Truncate if too long (max 5000 chars for API)
        if len(voiceover_text) > 4900:
            voiceover_text = voiceover_text[:4900] + "..."

        # Use slightly slower speech for teaching context (0.9 speed)
        # This gives more time for code typing while keeping natural narration
        speech_speed = 0.9

        print(f"[VOICEOVER] Generating voiceover for {len(voiceover_text)} characters (speed: {speech_speed})", flush=True)

        # Call media-generator voiceover endpoint
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Submit voiceover job
            response = await client.post(
                f"{self.media_generator_url}/api/v1/media/voiceover",
                json={
                    "text": voiceover_text,
                    "provider": "openai",
                    "voice_id": job.request.voice_id or "alloy",
                    "speed": speech_speed
                }
            )

            if response.status_code != 200:
                print(f"[VOICEOVER] Error submitting job: {response.status_code} - {response.text}", flush=True)
                return None, 0

            result = response.json()
            voiceover_job_id = result.get("job_id")

            if not voiceover_job_id:
                print(f"[VOICEOVER] No job_id in response: {result}", flush=True)
                return None, 0

            print(f"[VOICEOVER] Job submitted: {voiceover_job_id}", flush=True)

            # Poll for job completion
            max_attempts = 60  # 5 minutes max
            for attempt in range(max_attempts):
                await asyncio.sleep(5)  # Wait 5 seconds between polls

                status_response = await client.get(
                    f"{self.media_generator_url}/api/v1/media/jobs/{voiceover_job_id}"
                )

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
                    return audio_url, duration

                elif status == "failed":
                    error = status_data.get("error_message", "Unknown error")
                    print(f"[VOICEOVER] Job failed: {error}", flush=True)
                    return None, 0

                # Still processing
                progress = status_data.get("progress_percent", 0)
                print(f"[VOICEOVER] Progress: {progress}%", flush=True)

            print("[VOICEOVER] Timeout waiting for job completion", flush=True)
            return None, 0

    async def _adjust_slide_durations(
        self,
        job: PresentationJob,
        voiceover_duration: float
    ):
        """Adjust slide durations to match actual voiceover duration

        This ensures the video and audio are perfectly synchronized.
        Distributes time proportionally based on each slide's voiceover text length.
        Animations will adapt their speed to fit these durations.

        IMPORTANT: Code slides get extra time for comprehension pause (3 seconds)
        after the typing animation finishes.
        """
        slides = job.script.slides

        # Calculate total characters in voiceover text
        total_chars = sum(len(slide.voiceover_text or "") for slide in slides)

        if total_chars == 0:
            # Fallback: distribute evenly
            duration_per_slide = voiceover_duration / len(slides)
            for slide in slides:
                slide.duration = duration_per_slide
            print(f"[SYNC] Distributed {voiceover_duration}s evenly across {len(slides)} slides", flush=True)
            return

        original_total = sum(slide.duration for slide in slides)
        buffer_per_slide = 0.3

        # Comprehension pause for code slides (3 seconds after typing)
        CODE_COMPREHENSION_PAUSE = 3.0

        for slide in slides:
            char_count = len(slide.voiceover_text or "")
            if char_count > 0:
                # Proportional duration based on text length + buffer
                proportion = char_count / total_chars
                base_duration = (voiceover_duration * proportion) + buffer_per_slide

                # Add comprehension pause for code slides
                # This ensures the typed code stays visible for learners to absorb
                if slide.type in [SlideType.CODE, SlideType.CODE_DEMO] and slide.code_blocks:
                    slide.duration = base_duration + CODE_COMPREHENSION_PAUSE
                    print(f"[SYNC] Code slide '{slide.title}': {base_duration:.1f}s + {CODE_COMPREHENSION_PAUSE}s comprehension = {slide.duration:.1f}s", flush=True)
                else:
                    slide.duration = base_duration
            else:
                # Slides without voiceover get minimum duration
                slide.duration = 2.0

        new_total = sum(slide.duration for slide in slides)
        print(f"[SYNC] Adjusted slide durations: {original_total:.1f}s -> {new_total:.1f}s (voiceover: {voiceover_duration:.1f}s)", flush=True)

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

            # Poll for job completion
            max_attempts = 120  # 10 minutes max for video composition
            for attempt in range(max_attempts):
                await asyncio.sleep(5)

                status_response = await client.get(
                    f"{self.media_generator_url}/api/v1/media/jobs/{compose_job_id}"
                )

                if status_response.status_code != 200:
                    print(f"[COMPOSE] Error polling job: {status_response.status_code}", flush=True)
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
        """Create typing animation videos for code slides

        Animations adapt their typing speed to fit within slide.duration
        (which is synced to voiceover). This ensures audio/video synchronization.

        Returns:
            Dict mapping slide_id to {"url": str}
        """
        print("[ANIMATION] Starting typing animation creation...", flush=True)

        animation_map = {}  # slide_id -> {"url": ...}
        code_slides = [
            s for s in job.script.slides
            if s.type in [SlideType.CODE, SlideType.CODE_DEMO] and s.code_blocks
        ]

        if not code_slides:
            print("[ANIMATION] No code slides found", flush=True)
            return animation_map

        # Create temp directory for animation outputs
        temp_dir = Path(tempfile.gettempdir()) / "presentations" / "animations"
        temp_dir.mkdir(parents=True, exist_ok=True)

        colors = self.slide_generator.get_style_colors(job.request.style)

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

                    video_path, actual_duration = await self.typing_animator.create_typing_animation(
                        code=code_block.code,
                        language=code_block.language,
                        output_path=output_path,
                        title=slide.title,
                        typing_speed=typing_speed,  # Human-like speed preset
                        target_duration=slide.duration,  # Match voiceover duration
                        execution_output=execution_output,  # Show output if available
                        fps=30,
                        background_color=colors["background"] if "linear" not in colors["background"] else "#1e1e2e",
                        text_color=colors["text"],
                        accent_color=colors["accent"],
                        pygments_style=colors["pygments_style"]
                    )

                    # Get URL for the animation
                    service_url = os.getenv("SERVICE_URL", "http://127.0.0.1:8006")
                    animation_url = f"{service_url}/files/presentations/animations/{output_filename}"

                    # Store in map - animation adapts to slide.duration for voiceover sync
                    animation_map[slide.id] = {"url": animation_url}

                    print(f"[ANIMATION] Created: {animation_url} (target: {slide.duration:.1f}s, actual: {actual_duration:.1f}s)", flush=True)

                except Exception as e:
                    print(f"[ANIMATION] Error creating animation: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

        print(f"[ANIMATION] Completed {len(animation_map)} animations", flush=True)
        return animation_map

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
