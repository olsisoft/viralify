"""
Timeline Service

Manages video project timelines, segment ordering, and time calculations.
Phase 3: Video Editor feature.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from models.video_editor_models import (
    VideoProject,
    VideoSegment,
    SegmentType,
    SegmentStatus,
    ProjectStatus,
    TextOverlay,
    ImageOverlay,
    CreateProjectRequest,
    AddSegmentRequest,
    UpdateSegmentRequest,
)


class ProjectRepository:
    """
    In-memory project repository.
    In production, use PostgreSQL or similar.
    """

    def __init__(self):
        self.projects: Dict[str, VideoProject] = {}
        self.user_projects: Dict[str, List[str]] = {}  # user_id -> [project_ids]

    async def save(self, project: VideoProject) -> None:
        """Save project"""
        self.projects[project.id] = project

        # Index by user
        if project.user_id not in self.user_projects:
            self.user_projects[project.user_id] = []
        if project.id not in self.user_projects[project.user_id]:
            self.user_projects[project.user_id].append(project.id)

    async def get(self, project_id: str) -> Optional[VideoProject]:
        """Get project by ID"""
        return self.projects.get(project_id)

    async def get_by_user(self, user_id: str) -> List[VideoProject]:
        """Get all projects for a user"""
        project_ids = self.user_projects.get(user_id, [])
        return [self.projects[pid] for pid in project_ids if pid in self.projects]

    async def delete(self, project_id: str) -> bool:
        """Delete project"""
        project = self.projects.pop(project_id, None)
        if project:
            if project.user_id in self.user_projects:
                self.user_projects[project.user_id] = [
                    p for p in self.user_projects[project.user_id] if p != project_id
                ]
            return True
        return False


class TimelineService:
    """
    Service for managing video project timelines.
    Handles segment ordering, time calculations, and project state.
    """

    def __init__(self):
        self.repository = ProjectRepository()
        print("[TIMELINE] Timeline service initialized", flush=True)

    async def create_project(
        self,
        request: CreateProjectRequest,
        course_videos: Optional[List[dict]] = None,
    ) -> VideoProject:
        """
        Create a new video editing project.

        Args:
            request: Project creation request
            course_videos: Optional list of course videos to import
                          [{lecture_id, video_url, title, duration}, ...]

        Returns:
            Created VideoProject
        """
        print(f"[TIMELINE] Creating project: {request.title}", flush=True)

        project = VideoProject(
            id=str(uuid.uuid4()),
            user_id=request.user_id,
            course_id=request.course_id,
            course_job_id=request.course_job_id,
            title=request.title,
            description=request.description,
            status=ProjectStatus.DRAFT,
        )

        # Import course videos as segments
        if request.import_course_videos and course_videos:
            print(f"[TIMELINE] Importing {len(course_videos)} course videos", flush=True)
            current_time = 0.0

            for idx, video in enumerate(course_videos):
                segment = VideoSegment(
                    project_id=project.id,
                    segment_type=SegmentType.GENERATED,
                    source_url=video.get("video_url"),
                    source_lecture_id=video.get("lecture_id"),
                    order=idx,
                    start_time=current_time,
                    duration=video.get("duration", 60.0),
                    title=video.get("title"),
                    status=SegmentStatus.READY,
                )
                project.segments.append(segment)
                current_time += segment.duration

            project.total_duration = current_time

        await self.repository.save(project)

        print(f"[TIMELINE] Project created: {project.id} with {len(project.segments)} segments", flush=True)

        return project

    async def get_project(self, project_id: str, user_id: str) -> Optional[VideoProject]:
        """Get project with access control"""
        project = await self.repository.get(project_id)
        if project and project.user_id == user_id:
            return project
        return None

    async def list_projects(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[VideoProject], int]:
        """List projects for a user with pagination"""
        all_projects = await self.repository.get_by_user(user_id)

        # Sort by updated_at descending
        all_projects.sort(key=lambda p: p.updated_at, reverse=True)

        # Paginate
        total = len(all_projects)
        start = (page - 1) * page_size
        end = start + page_size
        paginated = all_projects[start:end]

        return paginated, total

    async def delete_project(self, project_id: str, user_id: str) -> bool:
        """Delete project with access control"""
        project = await self.repository.get(project_id)
        if not project or project.user_id != user_id:
            return False

        return await self.repository.delete(project_id)

    async def add_segment(
        self,
        project_id: str,
        user_id: str,
        request: AddSegmentRequest,
    ) -> Optional[VideoSegment]:
        """
        Add a new segment to the timeline.

        Args:
            project_id: Project ID
            user_id: User ID for access control
            request: Segment details

        Returns:
            Created VideoSegment or None
        """
        project = await self.get_project(project_id, user_id)
        if not project:
            return None

        print(f"[TIMELINE] Adding segment to project {project_id}", flush=True)

        # Determine position
        if request.insert_after_segment_id:
            # Find insert position
            insert_idx = 0
            for idx, seg in enumerate(project.segments):
                if seg.id == request.insert_after_segment_id:
                    insert_idx = idx + 1
                    break
        else:
            # Append to end
            insert_idx = len(project.segments)

        # Calculate start time
        if insert_idx > 0 and project.segments:
            prev_segment = project.segments[insert_idx - 1]
            start_time = prev_segment.start_time + prev_segment.duration
        else:
            start_time = 0.0

        # Determine duration
        if request.segment_type == SegmentType.SLIDE:
            duration = request.slide_duration
        else:
            duration = 60.0  # Will be updated when video is processed

        # Create segment
        segment = VideoSegment(
            project_id=project_id,
            segment_type=request.segment_type,
            source_url=request.source_url or request.slide_image_url,
            source_lecture_id=request.source_lecture_id,
            order=insert_idx,
            start_time=start_time,
            duration=duration,
            trim_start=request.trim_start,
            trim_end=request.trim_end,
            title=request.title,
            status=SegmentStatus.PENDING if not request.source_url else SegmentStatus.READY,
        )

        # Insert at position
        project.segments.insert(insert_idx, segment)

        # Update order and times for subsequent segments
        self._recalculate_timeline(project, from_index=insert_idx + 1)

        project.updated_at = datetime.utcnow()
        await self.repository.save(project)

        print(f"[TIMELINE] Segment {segment.id} added at position {insert_idx}", flush=True)

        return segment

    async def update_segment(
        self,
        project_id: str,
        segment_id: str,
        user_id: str,
        request: UpdateSegmentRequest,
    ) -> Optional[VideoSegment]:
        """Update segment properties"""
        project = await self.get_project(project_id, user_id)
        if not project:
            return None

        # Find segment
        segment = None
        segment_idx = -1
        for idx, seg in enumerate(project.segments):
            if seg.id == segment_id:
                segment = seg
                segment_idx = idx
                break

        if not segment:
            return None

        print(f"[TIMELINE] Updating segment {segment_id}", flush=True)

        # Update fields
        if request.trim_start is not None:
            segment.trim_start = request.trim_start
        if request.trim_end is not None:
            segment.trim_end = request.trim_end
        if request.original_audio_volume is not None:
            segment.original_audio_volume = request.original_audio_volume
        if request.is_audio_muted is not None:
            segment.is_audio_muted = request.is_audio_muted
        if request.opacity is not None:
            segment.opacity = request.opacity
        if request.scale is not None:
            segment.scale = request.scale
        if request.position_x is not None:
            segment.position_x = request.position_x
        if request.position_y is not None:
            segment.position_y = request.position_y
        if request.transition_in is not None:
            segment.transition_in = request.transition_in
        if request.transition_in_duration is not None:
            segment.transition_in_duration = request.transition_in_duration
        if request.transition_out is not None:
            segment.transition_out = request.transition_out
        if request.transition_out_duration is not None:
            segment.transition_out_duration = request.transition_out_duration
        if request.title is not None:
            segment.title = request.title

        segment.updated_at = datetime.utcnow()

        # Recalculate if trim changed duration
        if request.trim_start is not None or request.trim_end is not None:
            self._recalculate_timeline(project, from_index=segment_idx)

        project.updated_at = datetime.utcnow()
        await self.repository.save(project)

        return segment

    async def remove_segment(
        self,
        project_id: str,
        segment_id: str,
        user_id: str,
    ) -> bool:
        """Remove segment from timeline"""
        project = await self.get_project(project_id, user_id)
        if not project:
            return False

        # Find and remove segment
        segment_idx = -1
        for idx, seg in enumerate(project.segments):
            if seg.id == segment_id:
                segment_idx = idx
                break

        if segment_idx < 0:
            return False

        print(f"[TIMELINE] Removing segment {segment_id} from position {segment_idx}", flush=True)

        project.segments.pop(segment_idx)

        # Recalculate timeline
        self._recalculate_timeline(project, from_index=segment_idx)

        project.updated_at = datetime.utcnow()
        await self.repository.save(project)

        return True

    async def reorder_segments(
        self,
        project_id: str,
        user_id: str,
        segment_ids: List[str],
    ) -> bool:
        """Reorder segments according to provided ID list"""
        project = await self.get_project(project_id, user_id)
        if not project:
            return False

        # Create segment map
        segment_map = {seg.id: seg for seg in project.segments}

        # Verify all IDs exist
        for seg_id in segment_ids:
            if seg_id not in segment_map:
                return False

        print(f"[TIMELINE] Reordering {len(segment_ids)} segments in project {project_id}", flush=True)

        # Reorder
        new_segments = [segment_map[seg_id] for seg_id in segment_ids]
        project.segments = new_segments

        # Recalculate all times
        self._recalculate_timeline(project, from_index=0)

        project.updated_at = datetime.utcnow()
        await self.repository.save(project)

        return True

    async def split_segment(
        self,
        project_id: str,
        segment_id: str,
        user_id: str,
        split_time: float,
    ) -> Optional[Tuple[VideoSegment, VideoSegment]]:
        """
        Split a segment at a specific time.

        Args:
            project_id: Project ID
            segment_id: Segment to split
            user_id: User ID
            split_time: Time within segment to split (seconds)

        Returns:
            Tuple of (first_half, second_half) segments
        """
        project = await self.get_project(project_id, user_id)
        if not project:
            return None

        # Find segment
        segment = None
        segment_idx = -1
        for idx, seg in enumerate(project.segments):
            if seg.id == segment_id:
                segment = seg
                segment_idx = idx
                break

        if not segment or split_time <= 0 or split_time >= segment.duration:
            return None

        print(f"[TIMELINE] Splitting segment {segment_id} at {split_time}s", flush=True)

        # Create first half (original segment with modified duration)
        first_half = segment
        first_half.duration = split_time
        if first_half.trim_end:
            first_half.trim_end = first_half.trim_start + split_time

        # Create second half
        second_half = VideoSegment(
            project_id=project_id,
            segment_type=segment.segment_type,
            source_url=segment.source_url,
            source_lecture_id=segment.source_lecture_id,
            order=segment_idx + 1,
            start_time=segment.start_time + split_time,
            duration=segment.duration - split_time,
            trim_start=segment.trim_start + split_time,
            trim_end=segment.trim_end,
            title=f"{segment.title} (Part 2)" if segment.title else None,
            status=segment.status,
            original_audio_volume=segment.original_audio_volume,
            is_audio_muted=segment.is_audio_muted,
        )

        # Update first half title
        if first_half.title:
            first_half.title = f"{first_half.title} (Part 1)"

        # Insert second half
        project.segments.insert(segment_idx + 1, second_half)

        # Recalculate timeline
        self._recalculate_timeline(project, from_index=segment_idx)

        project.updated_at = datetime.utcnow()
        await self.repository.save(project)

        return (first_half, second_half)

    async def add_text_overlay(
        self,
        project_id: str,
        user_id: str,
        overlay: TextOverlay,
    ) -> bool:
        """Add text overlay to project"""
        project = await self.get_project(project_id, user_id)
        if not project:
            return False

        project.text_overlays.append(overlay)
        project.updated_at = datetime.utcnow()
        await self.repository.save(project)

        return True

    async def add_image_overlay(
        self,
        project_id: str,
        user_id: str,
        overlay: ImageOverlay,
    ) -> bool:
        """Add image overlay (logo, watermark) to project"""
        project = await self.get_project(project_id, user_id)
        if not project:
            return False

        project.image_overlays.append(overlay)
        project.updated_at = datetime.utcnow()
        await self.repository.save(project)

        return True

    async def update_project_settings(
        self,
        project_id: str,
        user_id: str,
        settings: dict,
    ) -> Optional[VideoProject]:
        """Update project output settings"""
        project = await self.get_project(project_id, user_id)
        if not project:
            return None

        if "title" in settings:
            project.title = settings["title"]
        if "description" in settings:
            project.description = settings["description"]
        if "output_resolution" in settings:
            project.output_resolution = settings["output_resolution"]
        if "output_fps" in settings:
            project.output_fps = settings["output_fps"]
        if "output_format" in settings:
            project.output_format = settings["output_format"]
        if "output_quality" in settings:
            project.output_quality = settings["output_quality"]
        if "background_music_url" in settings:
            project.background_music_url = settings["background_music_url"]
        if "background_music_volume" in settings:
            project.background_music_volume = settings["background_music_volume"]

        project.updated_at = datetime.utcnow()
        await self.repository.save(project)

        return project

    def _recalculate_timeline(self, project: VideoProject, from_index: int = 0) -> None:
        """Recalculate segment order and start times"""
        current_time = 0.0

        # Get start time from previous segment if exists
        if from_index > 0 and project.segments:
            prev_seg = project.segments[from_index - 1]
            current_time = prev_seg.start_time + prev_seg.duration

        # Update from from_index onwards
        for idx in range(from_index, len(project.segments)):
            segment = project.segments[idx]
            segment.order = idx
            segment.start_time = current_time
            current_time += segment.duration

        # Update total duration
        if project.segments:
            last_segment = project.segments[-1]
            project.total_duration = last_segment.start_time + last_segment.duration
        else:
            project.total_duration = 0.0

    async def update_segment_duration(
        self,
        project_id: str,
        segment_id: str,
        duration: float,
        thumbnail_url: Optional[str] = None,
    ) -> None:
        """
        Update segment duration (called after video upload processing).
        Internal method - no access control.
        """
        project = await self.repository.get(project_id)
        if not project:
            return

        for idx, segment in enumerate(project.segments):
            if segment.id == segment_id:
                segment.duration = duration
                segment.status = SegmentStatus.READY
                if thumbnail_url:
                    segment.thumbnail_url = thumbnail_url
                self._recalculate_timeline(project, from_index=idx)
                break

        await self.repository.save(project)
