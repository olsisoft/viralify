"""
Lecture Editor Service

Handles lecture component storage, editing, and regeneration.
Enables:
- Storing components after generation (for later editing)
- Editing slide content, voiceover, diagrams
- Regenerating individual slides or entire lectures
- Recomposing video from edited components
"""
import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

import asyncpg

from models.lecture_components import (
    CodeBlockComponent,
    ComponentStatus,
    LectureComponents,
    LectureComponentsDB,
    MediaType,
    SlideComponent,
    SlideType,
    VoiceoverComponent,
    UpdateSlideRequest,
    RegenerateSlideRequest,
    RegenerateLectureRequest,
    RegenerateVoiceoverRequest,
    RecomposeVideoRequest,
    ReorderSlideRequest,
    InsertMediaRequest,
)
from models.course_models import Lecture
from services.http_client import ResilientHTTPClient, RetryConfig


class LectureComponentsRepository:
    """Repository for storing and retrieving lecture components from PostgreSQL"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/viralify"
        )
        self._pool: Optional[asyncpg.Pool] = None

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool"""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10
            )
        return self._pool

    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def save(self, components: LectureComponents) -> str:
        """Save lecture components to database"""
        pool = await self.get_pool()
        db_model = LectureComponentsDB.from_lecture_components(components)

        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO lecture_components (
                    id, lecture_id, job_id, slides_json, voiceover_json,
                    generation_params_json, total_duration, video_url,
                    presentation_job_id, status, is_edited, error,
                    created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (id) DO UPDATE SET
                    slides_json = EXCLUDED.slides_json,
                    voiceover_json = EXCLUDED.voiceover_json,
                    total_duration = EXCLUDED.total_duration,
                    video_url = EXCLUDED.video_url,
                    status = EXCLUDED.status,
                    is_edited = EXCLUDED.is_edited,
                    error = EXCLUDED.error,
                    updated_at = EXCLUDED.updated_at
            """,
                db_model.id,
                db_model.lecture_id,
                db_model.job_id,
                db_model.slides_json,
                db_model.voiceover_json,
                db_model.generation_params_json,
                db_model.total_duration,
                db_model.video_url,
                db_model.presentation_job_id,
                db_model.status,
                db_model.is_edited,
                db_model.error,
                db_model.created_at,
                db_model.updated_at
            )

        return db_model.id

    async def get_by_id(self, components_id: str) -> Optional[LectureComponents]:
        """Get lecture components by ID"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM lecture_components WHERE id = $1",
                components_id
            )

        if not row:
            return None

        db_model = LectureComponentsDB(
            id=row["id"],
            lecture_id=row["lecture_id"],
            job_id=row["job_id"],
            slides_json=row["slides_json"],
            voiceover_json=row["voiceover_json"],
            generation_params_json=row["generation_params_json"],
            total_duration=row["total_duration"],
            video_url=row["video_url"],
            presentation_job_id=row["presentation_job_id"],
            status=row["status"],
            is_edited=row["is_edited"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

        return db_model.to_lecture_components()

    async def get_by_lecture_id(self, lecture_id: str) -> Optional[LectureComponents]:
        """Get lecture components by lecture ID"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM lecture_components WHERE lecture_id = $1",
                lecture_id
            )

        if not row:
            return None

        db_model = LectureComponentsDB(
            id=row["id"],
            lecture_id=row["lecture_id"],
            job_id=row["job_id"],
            slides_json=row["slides_json"],
            voiceover_json=row["voiceover_json"],
            generation_params_json=row["generation_params_json"],
            total_duration=row["total_duration"],
            video_url=row["video_url"],
            presentation_job_id=row["presentation_job_id"],
            status=row["status"],
            is_edited=row["is_edited"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )

        return db_model.to_lecture_components()

    async def delete(self, components_id: str) -> bool:
        """Delete lecture components"""
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM lecture_components WHERE id = $1",
                components_id
            )

        return "DELETE 1" in result


class LectureEditorService:
    """Service for editing lecture components and regenerating content"""

    def __init__(
        self,
        presentation_generator_url: str = None,
        media_generator_url: str = None
    ):
        self.presentation_generator_url = presentation_generator_url or os.getenv(
            "PRESENTATION_GENERATOR_URL", "http://127.0.0.1:8006"
        )
        self.media_generator_url = media_generator_url or os.getenv(
            "MEDIA_GENERATOR_URL", "http://127.0.0.1:8004"
        )

        # Repository for persistence
        self.repository = LectureComponentsRepository()

        # HTTP clients for service calls
        retry_config = RetryConfig(
            max_retries=3,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
        )
        self.presentation_client = ResilientHTTPClient(
            self.presentation_generator_url,
            timeout=90.0,
            retry_config=retry_config,
        )
        self.media_client = ResilientHTTPClient(
            self.media_generator_url,
            timeout=120.0,
            retry_config=retry_config,
        )

    async def close(self):
        """Cleanup resources"""
        await self.repository.close()

    # =========================================================================
    # Component Storage (called after lecture generation)
    # =========================================================================

    async def store_components_from_presentation_job(
        self,
        presentation_job_id: str,
        lecture_id: str,
        job_id: str,
        generation_params: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Fetch presentation job details and store components for editing.
        Called after a lecture is successfully generated.

        Returns the components_id for lazy loading reference.
        """
        try:
            # Fetch the presentation job details
            response = await self.presentation_client.get(
                f"/api/v1/presentations/jobs/{presentation_job_id}"
            )

            if response.status_code != 200:
                print(f"[EDITOR] Failed to fetch presentation job {presentation_job_id}: {response.status_code}", flush=True)
                return None

            job_data = response.json()

            # Extract script with slides
            script = job_data.get("script", {})
            slides_data = script.get("slides", [])

            # Convert to SlideComponent models
            slides = []
            for idx, slide_data in enumerate(slides_data):
                # Convert code blocks
                code_blocks = []
                for cb in slide_data.get("code_blocks", []):
                    code_blocks.append(CodeBlockComponent(
                        id=cb.get("id", str(uuid.uuid4())[:8]),
                        language=cb.get("language", "python"),
                        code=cb.get("code", ""),
                        filename=cb.get("filename"),
                        highlight_lines=cb.get("highlight_lines", []),
                        execution_order=cb.get("execution_order", 0),
                        expected_output=cb.get("expected_output"),
                        actual_output=cb.get("actual_output"),
                        show_line_numbers=cb.get("show_line_numbers", True)
                    ))

                # Get animation URL if available
                animation_url = None
                animation_videos = job_data.get("animation_videos", [])
                if idx < len(animation_videos):
                    animation_url = animation_videos[idx]

                slide = SlideComponent(
                    id=slide_data.get("id", str(uuid.uuid4())[:8]),
                    index=idx,
                    type=SlideType(slide_data.get("type", "content")),
                    status=ComponentStatus.COMPLETED,
                    title=slide_data.get("title"),
                    subtitle=slide_data.get("subtitle"),
                    content=slide_data.get("content"),
                    bullet_points=slide_data.get("bullet_points", []),
                    code_blocks=code_blocks,
                    voiceover_text=slide_data.get("voiceover_text", ""),
                    duration=float(slide_data.get("duration", 10.0)),
                    transition=slide_data.get("transition", "fade"),
                    diagram_type=slide_data.get("diagram_type"),
                    image_url=slide_data.get("image_url"),
                    animation_url=animation_url
                )
                slides.append(slide)

            # Create voiceover component
            voiceover = None
            voiceover_url = job_data.get("voiceover_url")
            if voiceover_url:
                voiceover = VoiceoverComponent(
                    audio_url=voiceover_url,
                    duration_seconds=sum(s.duration for s in slides),
                    full_text=" ".join(s.voiceover_text for s in slides if s.voiceover_text)
                )

            # Create LectureComponents
            components = LectureComponents(
                id=str(uuid.uuid4()),
                lecture_id=lecture_id,
                job_id=job_id,
                slides=slides,
                voiceover=voiceover,
                total_duration=sum(s.duration for s in slides),
                generation_params=generation_params or {},
                presentation_job_id=presentation_job_id,
                video_url=job_data.get("output_url"),
                status=ComponentStatus.COMPLETED
            )

            # Save to database
            components_id = await self.repository.save(components)
            print(f"[EDITOR] Stored components for lecture {lecture_id}: {components_id}", flush=True)

            return components_id

        except Exception as e:
            print(f"[EDITOR] Failed to store components for lecture {lecture_id}: {str(e)}", flush=True)
            return None

    # =========================================================================
    # Component Retrieval
    # =========================================================================

    async def get_components(self, lecture_id: str) -> Optional[LectureComponents]:
        """Get lecture components by lecture ID (lazy loading)"""
        return await self.repository.get_by_lecture_id(lecture_id)

    async def get_components_by_id(self, components_id: str) -> Optional[LectureComponents]:
        """Get lecture components by components ID"""
        return await self.repository.get_by_id(components_id)

    # =========================================================================
    # Slide Editing
    # =========================================================================

    async def update_slide(
        self,
        lecture_id: str,
        slide_id: str,
        updates: UpdateSlideRequest
    ) -> Optional[SlideComponent]:
        """Update a slide's content"""
        components = await self.get_components(lecture_id)
        if not components:
            return None

        slide = components.get_slide(slide_id)
        if not slide:
            return None

        # Apply updates
        update_dict = updates.model_dump(exclude_unset=True, exclude_none=True)
        components.update_slide(slide_id, update_dict)

        # Recalculate duration if slide duration changed
        if "duration" in update_dict:
            components.recalculate_duration()

        # Update voiceover text if slide voiceover changed
        if "voiceover_text" in update_dict and components.voiceover:
            components.voiceover.full_text = components.get_combined_voiceover_text()
            components.voiceover.is_edited = True
            components.voiceover.edited_at = datetime.utcnow()

        # Save changes
        await self.repository.save(components)

        return slide

    async def reorder_slide(
        self,
        lecture_id: str,
        slide_id: str,
        new_index: int
    ) -> Optional[SlideComponent]:
        """Reorder a slide to a new position"""
        components = await self.get_components(lecture_id)
        if not components:
            return None

        slide = components.get_slide(slide_id)
        if not slide:
            return None

        # Perform reorder
        success = components.reorder_slide(slide_id, new_index)
        if not success:
            return None

        # Save changes
        await self.repository.save(components)

        # Return the updated slide (with new index)
        return components.get_slide(slide_id)

    async def delete_slide(
        self,
        lecture_id: str,
        slide_id: str
    ) -> Optional[SlideComponent]:
        """Delete a slide from the lecture"""
        components = await self.get_components(lecture_id)
        if not components:
            return None

        # Cannot delete if only one slide
        if len(components.slides) <= 1:
            raise ValueError("Cannot delete the last slide. A lecture must have at least one slide.")

        # Perform delete
        deleted_slide = components.delete_slide(slide_id)
        if not deleted_slide:
            return None

        # Update voiceover text
        if components.voiceover:
            components.voiceover.full_text = components.get_combined_voiceover_text()
            components.voiceover.is_edited = True
            components.voiceover.edited_at = datetime.utcnow()

        # Save changes
        await self.repository.save(components)

        return deleted_slide

    async def insert_media_slide(
        self,
        lecture_id: str,
        request: InsertMediaRequest,
        media_url: str,
        media_thumbnail_url: Optional[str] = None,
        original_filename: Optional[str] = None
    ) -> Optional[SlideComponent]:
        """Insert a new media slide (image or video)"""
        components = await self.get_components(lecture_id)
        if not components:
            return None

        # Create new media slide
        new_slide = SlideComponent(
            id=str(uuid.uuid4())[:8],
            index=0,  # Will be set by insert_slide
            type=SlideType.MEDIA,
            status=ComponentStatus.COMPLETED,
            title=request.title,
            voiceover_text=request.voiceover_text or "",
            duration=request.duration,
            media_type=request.media_type,
            media_url=media_url,
            media_thumbnail_url=media_thumbnail_url,
            media_original_filename=original_filename,
            is_edited=False,
        )

        # Insert the slide
        inserted_slide = components.insert_slide(new_slide, request.insert_after_slide_id)

        # Update voiceover text if voiceover exists
        if components.voiceover and request.voiceover_text:
            components.voiceover.full_text = components.get_combined_voiceover_text()
            components.voiceover.is_edited = True
            components.voiceover.edited_at = datetime.utcnow()

        # Save changes
        await self.repository.save(components)

        return inserted_slide

    async def upload_media_to_slide(
        self,
        lecture_id: str,
        slide_id: str,
        media_data: bytes,
        filename: str,
        media_type: MediaType
    ) -> Optional[SlideComponent]:
        """Upload media (image/video) to an existing slide"""
        components = await self.get_components(lecture_id)
        if not components:
            return None

        slide = components.get_slide(slide_id)
        if not slide:
            return None

        try:
            # Upload to media-generator
            files = {"file": (filename, media_data)}
            endpoint = "/api/v1/media/upload/image" if media_type == MediaType.IMAGE else "/api/v1/media/upload/video"

            response = await self.media_client.post(endpoint, files=files)

            if response.status_code == 200:
                result = response.json()

                # Update slide with media info
                slide.media_type = media_type
                slide.media_url = result.get("url")
                slide.media_original_filename = filename

                # For images, use the URL as the slide image
                if media_type == MediaType.IMAGE:
                    slide.image_url = result.get("url")
                else:
                    # For videos, get thumbnail if available
                    slide.media_thumbnail_url = result.get("thumbnail_url")

                slide.mark_edited(["media_url", "media_type"])
            else:
                raise Exception(f"Failed to upload media: {response.text}")

            await self.repository.save(components)
            return slide

        except Exception as e:
            slide.error = str(e)
            await self.repository.save(components)
            raise

    # =========================================================================
    # Regeneration
    # =========================================================================

    async def regenerate_slide(
        self,
        lecture_id: str,
        slide_id: str,
        options: RegenerateSlideRequest
    ) -> Optional[SlideComponent]:
        """Regenerate a single slide (image, animation, or both)"""
        components = await self.get_components(lecture_id)
        if not components:
            return None

        slide = components.get_slide(slide_id)
        if not slide:
            return None

        slide.status = ComponentStatus.GENERATING

        try:
            # Regenerate slide image if requested
            if options.regenerate_image:
                slide_data = {
                    "id": slide.id,
                    "type": slide.type.value,
                    "title": slide.title,
                    "subtitle": slide.subtitle,
                    "content": slide.content,
                    "bullet_points": slide.bullet_points,
                    "code_blocks": [cb.model_dump() for cb in slide.code_blocks],
                    "duration": slide.duration,
                    "voiceover_text": slide.voiceover_text,
                    "diagram_type": slide.diagram_type
                }

                # Call presentation-generator to regenerate slide
                response = await self.presentation_client.post(
                    "/api/v1/presentations/slides/regenerate",
                    json={
                        "slide": slide_data,
                        "style": components.generation_params.get("style", "dark")
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    slide.image_url = result.get("image_url")
                else:
                    raise Exception(f"Failed to regenerate slide image: {response.text}")

            # Regenerate animation if requested (for code slides)
            if options.regenerate_animation and slide.type in [SlideType.CODE, SlideType.CODE_DEMO]:
                response = await self.presentation_client.post(
                    "/api/v1/presentations/slides/regenerate-animation",
                    json={
                        "slide_id": slide.id,
                        "code_blocks": [cb.model_dump() for cb in slide.code_blocks],
                        "duration": slide.duration,
                        "typing_speed": components.generation_params.get("typing_speed", "natural")
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    slide.animation_url = result.get("animation_url")
                else:
                    raise Exception(f"Failed to regenerate animation: {response.text}")

            slide.status = ComponentStatus.COMPLETED
            await self.repository.save(components)
            return slide

        except Exception as e:
            slide.status = ComponentStatus.FAILED
            slide.error = str(e)
            await self.repository.save(components)
            raise

    async def regenerate_voiceover(
        self,
        lecture_id: str,
        options: RegenerateVoiceoverRequest
    ) -> Optional[VoiceoverComponent]:
        """Regenerate voiceover audio from slide texts"""
        components = await self.get_components(lecture_id)
        if not components:
            return None

        if not components.voiceover:
            components.voiceover = VoiceoverComponent()

        components.voiceover.status = ComponentStatus.GENERATING

        try:
            # Combine all slide voiceover texts
            full_text = components.get_combined_voiceover_text()

            # Call media-generator to generate voiceover
            voice_id = options.voice_id or components.voiceover.voice_id
            response = await self.media_client.post(
                "/api/v1/media/voiceover",
                json={
                    "text": full_text,
                    "voice_id": voice_id,
                    "settings": options.voice_settings or components.voiceover.voice_settings
                }
            )

            if response.status_code == 200:
                result = response.json()
                components.voiceover.audio_url = result.get("audio_url")
                components.voiceover.duration_seconds = result.get("duration", 0)
                components.voiceover.voice_id = voice_id
                components.voiceover.full_text = full_text
                components.voiceover.is_custom_audio = False
                components.voiceover.status = ComponentStatus.COMPLETED
            else:
                raise Exception(f"Failed to generate voiceover: {response.text}")

            await self.repository.save(components)
            return components.voiceover

        except Exception as e:
            components.voiceover.status = ComponentStatus.FAILED
            components.voiceover.error = str(e)
            await self.repository.save(components)
            raise

    async def upload_custom_audio(
        self,
        lecture_id: str,
        audio_data: bytes,
        filename: str
    ) -> Optional[VoiceoverComponent]:
        """Upload custom audio to replace generated voiceover"""
        components = await self.get_components(lecture_id)
        if not components:
            return None

        if not components.voiceover:
            components.voiceover = VoiceoverComponent()

        try:
            # Upload to media-generator
            files = {"file": (filename, audio_data)}
            response = await self.media_client.post(
                "/api/v1/media/upload/audio",
                files=files
            )

            if response.status_code == 200:
                result = response.json()
                components.voiceover.audio_url = result.get("url")
                components.voiceover.duration_seconds = result.get("duration", 0)
                components.voiceover.is_custom_audio = True
                components.voiceover.original_filename = filename
                components.voiceover.is_edited = True
                components.voiceover.edited_at = datetime.utcnow()
                components.voiceover.status = ComponentStatus.COMPLETED
            else:
                raise Exception(f"Failed to upload audio: {response.text}")

            await self.repository.save(components)
            return components.voiceover

        except Exception as e:
            components.voiceover.status = ComponentStatus.FAILED
            components.voiceover.error = str(e)
            await self.repository.save(components)
            raise

    async def regenerate_lecture(
        self,
        lecture_id: str,
        options: RegenerateLectureRequest,
        lecture: Lecture
    ) -> Optional[str]:
        """
        Regenerate entire lecture.

        If use_edited_components=True, keeps edited slides and regenerates others.
        If use_edited_components=False, regenerates everything from scratch.

        Returns the new video URL.
        """
        components = await self.get_components(lecture_id)

        if options.use_edited_components and components:
            # Keep edited slides, regenerate non-edited ones
            for slide in components.slides:
                if not slide.is_edited:
                    # Regenerate this slide
                    await self.regenerate_slide(
                        lecture_id,
                        slide.id,
                        RegenerateSlideRequest(regenerate_image=True, regenerate_animation=True)
                    )

            # Regenerate voiceover if requested
            if options.regenerate_voiceover:
                await self.regenerate_voiceover(
                    lecture_id,
                    RegenerateVoiceoverRequest(voice_id=options.voice_id)
                )

            # Recompose video
            return await self.recompose_video(lecture_id, RecomposeVideoRequest())

        else:
            # Full regeneration - call presentation-generator
            # This requires the original generation parameters
            if not components or not components.generation_params:
                raise Exception("No generation parameters available for full regeneration")

            params = components.generation_params
            # Use v3 endpoint which includes VoiceoverEnforcer for proper video duration
            response = await self.presentation_client.post(
                "/api/v1/presentations/generate/v3",
                json=params
            )

            if response.status_code != 200:
                raise Exception(f"Failed to start regeneration: {response.text}")

            job_data = response.json()
            new_job_id = job_data.get("job_id")

            # Poll for completion
            video_url = await self._poll_regeneration_job(new_job_id)

            # Store new components
            await self.store_components_from_presentation_job(
                new_job_id,
                lecture_id,
                components.job_id,
                params
            )

            return video_url

    async def recompose_video(
        self,
        lecture_id: str,
        options: RecomposeVideoRequest
    ) -> Optional[str]:
        """Recompose video from current components (after editing)"""
        components = await self.get_components(lecture_id)
        if not components:
            return None

        try:
            # Build scenes from slides
            scenes = []
            for slide in components.slides:
                scene = {
                    "duration": slide.duration,
                    "transition": slide.transition
                }

                # Use animation URL for code slides, image URL for others
                if slide.animation_url and slide.type in [SlideType.CODE, SlideType.CODE_DEMO]:
                    scene["video_url"] = slide.animation_url
                else:
                    scene["image_url"] = slide.image_url

                scenes.append(scene)

            # Call media-generator to compose video
            compose_request = {
                "scenes": scenes,
                "voiceover_url": components.voiceover.audio_url if components.voiceover else None,
                "quality": options.quality,
                "include_transitions": options.include_transitions
            }

            response = await self.media_client.post(
                "/api/v1/media/slideshow/compose",
                json=compose_request
            )

            if response.status_code == 200:
                result = response.json()
                job_id = result.get("job_id")

                # Poll for completion
                video_url = await self._poll_media_job(job_id)

                # Update components
                components.video_url = video_url
                components.updated_at = datetime.utcnow()
                await self.repository.save(components)

                return video_url
            else:
                raise Exception(f"Failed to start video composition: {response.text}")

        except Exception as e:
            components.error = str(e)
            await self.repository.save(components)
            raise

    # =========================================================================
    # Polling Helpers
    # =========================================================================

    async def _poll_regeneration_job(
        self,
        job_id: str,
        max_wait: float = 600.0,
        poll_interval: float = 5.0
    ) -> str:
        """Poll presentation-generator job until completion"""
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Regeneration timed out after {max_wait}s")

            response = await self.presentation_client.get(
                f"/api/v1/presentations/jobs/{job_id}"
            )

            if response.status_code != 200:
                await asyncio.sleep(poll_interval)
                continue

            job_data = response.json()
            status = job_data.get("status")

            if status == "completed":
                return job_data.get("output_url")
            elif status == "failed":
                raise Exception(f"Regeneration failed: {job_data.get('error')}")

            await asyncio.sleep(poll_interval)

    async def _poll_media_job(
        self,
        job_id: str,
        max_wait: float = 300.0,
        poll_interval: float = 3.0
    ) -> str:
        """Poll media-generator job until completion"""
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_wait:
                raise TimeoutError(f"Media job timed out after {max_wait}s")

            response = await self.media_client.get(
                f"/api/v1/media/jobs/{job_id}"
            )

            if response.status_code != 200:
                await asyncio.sleep(poll_interval)
                continue

            job_data = response.json()
            status = job_data.get("status")

            if status == "completed":
                return job_data.get("output_url")
            elif status == "failed":
                raise Exception(f"Media job failed: {job_data.get('error')}")

            await asyncio.sleep(poll_interval)
