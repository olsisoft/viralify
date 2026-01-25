"""
Direct Timeline Builder - Perfect synchronization without SSVS

This module builds timelines directly from slide audio durations,
eliminating the need for post-hoc semantic matching (SSVS).

The key insight: if we generate audio per slide, the audio duration
IS the slide duration. Synchronization is perfect by construction.

Flow:
1. SlideAudioGenerator produces audio for each slide
2. AudioConcatenator merges them with crossfade
3. DirectTimelineBuilder creates timeline from known durations
4. No SSVS, no Whisper, no calibration needed!
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .slide_audio_generator import SlideAudioBatch
from .audio_concatenator import ConcatenatedAudio


class VisualEventType(Enum):
    """Types of visual events on the timeline"""
    SLIDE = "slide"
    CODE_ANIMATION = "code_animation"
    DIAGRAM = "diagram"
    TRANSITION = "transition"
    FREEZE_FRAME = "freeze_frame"


@dataclass
class DirectVisualEvent:
    """A visual event with precise timing from audio duration"""
    event_type: VisualEventType
    slide_index: int
    slide_id: str
    time_start: float
    time_end: float
    duration: float
    asset_path: Optional[str] = None
    asset_url: Optional[str] = None
    layer: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "slide_index": self.slide_index,
            "slide_id": self.slide_id,
            "time_start": round(self.time_start, 3),
            "time_end": round(self.time_end, 3),
            "duration": round(self.duration, 3),
            "asset_path": self.asset_path,
            "asset_url": self.asset_url,
            "layer": self.layer,
            "metadata": self.metadata
        }


@dataclass
class DirectTimeline:
    """
    Timeline built directly from audio durations.

    Unlike SSVS-based timelines, this has PERFECT synchronization
    because the timeline is derived from the audio, not matched to it.
    """
    total_duration: float
    audio_path: str
    audio_url: Optional[str]
    visual_events: List[DirectVisualEvent]
    slide_timings: List[Dict[str, Any]]
    sync_method: str = "direct"  # Not SSVS
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_duration": round(self.total_duration, 3),
            "audio_path": self.audio_path,
            "audio_url": self.audio_url,
            "visual_events": [e.to_dict() for e in self.visual_events],
            "slide_timings": self.slide_timings,
            "sync_method": self.sync_method,
            "metadata": self.metadata
        }


class DirectTimelineBuilder:
    """
    Builds timeline directly from slide audio durations.

    This is the recommended approach when using per-slide TTS generation.
    SSVS is no longer needed because synchronization is built-in.

    Usage:
        # Generate audio per slide
        batch = await slide_audio_generator.generate_batch(slides)

        # Concatenate with crossfade
        concat_result = await audio_concatenator.concatenate(batch)

        # Build timeline (perfect sync!)
        builder = DirectTimelineBuilder()
        timeline = builder.build(slides, concat_result)
    """

    # Transition duration for crossfade effect
    TRANSITION_DURATION = 0.1  # Match crossfade duration

    def __init__(self):
        print("[DIRECT_TIMELINE] Initialized (no SSVS needed)", flush=True)

    def build(
        self,
        slides: List[Dict[str, Any]],
        concat_result: ConcatenatedAudio,
        animations: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> DirectTimeline:
        """
        Build timeline from concatenated audio result.

        The timeline is derived directly from the audio durations,
        so synchronization is perfect by construction.

        Args:
            slides: Original slide data with images, types, etc.
            concat_result: Result from AudioConcatenator
            animations: Optional animation data per slide

        Returns:
            DirectTimeline with perfect synchronization
        """
        animations = animations or {}

        print(f"[DIRECT_TIMELINE] Building timeline for {len(slides)} slides", flush=True)
        print(f"[DIRECT_TIMELINE] Total duration: {concat_result.total_duration:.2f}s", flush=True)

        # Build visual events from the concatenated timeline
        visual_events = []

        for timing in concat_result.timeline:
            slide_idx = timing["slide_index"]
            slide_id = timing["slide_id"]

            # Get slide data
            slide = slides[slide_idx] if slide_idx < len(slides) else {}
            slide_type = slide.get("type", "content")

            # Create visual event
            event_type = self._get_event_type(slide_type)

            # Check for animation
            animation_info = animations.get(slide_id)

            if animation_info and slide_type in ["code", "code_demo"]:
                # Code animation event
                animation_duration = animation_info.get("duration", timing["duration"])

                visual_events.append(DirectVisualEvent(
                    event_type=VisualEventType.CODE_ANIMATION,
                    slide_index=slide_idx,
                    slide_id=slide_id,
                    time_start=timing["start"],
                    time_end=timing["start"] + animation_duration,
                    duration=animation_duration,
                    asset_path=animation_info.get("file_path"),
                    asset_url=animation_info.get("url"),
                    layer=0,
                    metadata={
                        "language": slide.get("language", "python"),
                        "has_animation": True
                    }
                ))

                # Freeze frame if animation is shorter than voiceover
                if animation_duration < timing["duration"]:
                    freeze_start = timing["start"] + animation_duration
                    visual_events.append(DirectVisualEvent(
                        event_type=VisualEventType.FREEZE_FRAME,
                        slide_index=slide_idx,
                        slide_id=slide_id,
                        time_start=freeze_start,
                        time_end=timing["end"],
                        duration=timing["end"] - freeze_start,
                        asset_path=animation_info.get("file_path"),
                        asset_url=animation_info.get("url"),
                        layer=0,
                        metadata={"freeze_from": "last_frame"}
                    ))

            else:
                # Regular slide or diagram
                visual_events.append(DirectVisualEvent(
                    event_type=event_type,
                    slide_index=slide_idx,
                    slide_id=slide_id,
                    time_start=timing["start"],
                    time_end=timing["end"],
                    duration=timing["duration"],
                    asset_path=slide.get("image_path"),
                    asset_url=slide.get("image_url"),
                    layer=0,
                    metadata={
                        "title": slide.get("title", ""),
                        "slide_type": slide_type
                    }
                ))

            print(f"[DIRECT_TIMELINE] Slide {slide_idx}: {timing['start']:.3f}s - {timing['end']:.3f}s ({event_type.value})", flush=True)

        # Sort events by time
        visual_events.sort(key=lambda e: (e.time_start, e.layer))

        timeline = DirectTimeline(
            total_duration=concat_result.total_duration,
            audio_path=concat_result.audio_path,
            audio_url=concat_result.audio_url,
            visual_events=visual_events,
            slide_timings=concat_result.timeline,
            sync_method="direct",
            metadata={
                "slide_count": len(slides),
                "event_count": len(visual_events),
                "crossfade_duration": concat_result.crossfade_duration,
                "sync_quality": "perfect"  # By construction!
            }
        )

        print(f"[DIRECT_TIMELINE] Timeline built: {len(visual_events)} events, perfect sync", flush=True)

        return timeline

    def build_from_batch(
        self,
        slides: List[Dict[str, Any]],
        batch: SlideAudioBatch,
        animations: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> DirectTimeline:
        """
        Build timeline directly from SlideAudioBatch (without concatenation).

        Use this when you want the raw timeline without crossfade adjustments.
        The individual audio files can be concatenated later.

        Args:
            slides: Original slide data
            batch: SlideAudioBatch from SlideAudioGenerator
            animations: Optional animation data

        Returns:
            DirectTimeline with per-slide timing
        """
        animations = animations or {}

        print(f"[DIRECT_TIMELINE] Building from batch: {len(slides)} slides", flush=True)

        visual_events = []
        current_time = 0.0

        for audio in batch.slide_audios:
            slide_idx = audio.slide_index
            slide_id = audio.slide_id
            duration = audio.duration

            slide = slides[slide_idx] if slide_idx < len(slides) else {}
            slide_type = slide.get("type", "content")
            event_type = self._get_event_type(slide_type)

            visual_events.append(DirectVisualEvent(
                event_type=event_type,
                slide_index=slide_idx,
                slide_id=slide_id,
                time_start=current_time,
                time_end=current_time + duration,
                duration=duration,
                asset_path=slide.get("image_path"),
                asset_url=slide.get("image_url"),
                layer=0,
                metadata={
                    "title": slide.get("title", ""),
                    "slide_type": slide_type,
                    "audio_path": audio.audio_path
                }
            ))

            current_time += duration

        return DirectTimeline(
            total_duration=batch.total_duration,
            audio_path="",  # Not concatenated yet
            audio_url=None,
            visual_events=visual_events,
            slide_timings=batch.timeline,
            sync_method="direct_batch",
            metadata={
                "slide_count": len(slides),
                "requires_concat": True
            }
        )

    def _get_event_type(self, slide_type: str) -> VisualEventType:
        """Map slide type to visual event type"""
        type_map = {
            "code": VisualEventType.CODE_ANIMATION,
            "code_demo": VisualEventType.CODE_ANIMATION,
            "diagram": VisualEventType.DIAGRAM,
            "content": VisualEventType.SLIDE,
            "title": VisualEventType.SLIDE,
            "conclusion": VisualEventType.SLIDE,
        }
        return type_map.get(slide_type, VisualEventType.SLIDE)


# Convenience function for the complete flow
async def build_direct_timeline(
    slides: List[Dict[str, Any]],
    voice_id: str = "alloy",
    language: str = "en",
    crossfade_ms: float = 100,
    job_id: Optional[str] = None
) -> DirectTimeline:
    """
    Complete flow: Generate audio per slide, concatenate, build timeline.

    This is the recommended way to create synchronized presentations.
    No SSVS, no Whisper, no calibration - just perfect sync.

    Example:
        timeline = await build_direct_timeline(
            slides=presentation_slides,
            voice_id="nova",
            language="fr",
            crossfade_ms=100
        )

        # Timeline is perfectly synchronized!
        for event in timeline.visual_events:
            print(f"Slide {event.slide_index}: {event.time_start}s - {event.time_end}s")
    """
    from .slide_audio_generator import SlideAudioGenerator
    from .audio_concatenator import AudioConcatenator

    # Step 1: Generate audio per slide (parallel)
    generator = SlideAudioGenerator(voice_id=voice_id, language=language)
    batch = await generator.generate_batch(slides, language=language, job_id=job_id)

    # Step 2: Concatenate with crossfade
    concatenator = AudioConcatenator(crossfade_ms=crossfade_ms)
    concat_result = await concatenator.concatenate(batch, job_id=job_id)

    # Step 3: Build timeline
    builder = DirectTimelineBuilder()
    timeline = builder.build(slides, concat_result)

    return timeline
