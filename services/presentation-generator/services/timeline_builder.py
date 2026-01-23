"""
Timeline Builder Service

Builds an event-driven timeline for precise audio-video synchronization.
This is the core of professional-grade video generation.

The approach:
1. Audio (voiceover) is the source of truth
2. Word-level timestamps drive all visual events
3. Visual assets are rendered to match the timeline
4. FFmpeg filtercomplex assembles with millisecond precision
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class VisualEventType(Enum):
    """Types of visual events on the timeline"""
    SLIDE = "slide"                    # Static slide/image
    CODE_ANIMATION = "code_animation"  # Typing animation
    DIAGRAM = "diagram"                # Diagram reveal
    AVATAR = "avatar"                  # Avatar/presenter overlay
    TRANSITION = "transition"          # Transition effect
    HIGHLIGHT = "highlight"            # Code highlight
    BULLET_REVEAL = "bullet_reveal"    # Bullet point animation
    FREEZE_FRAME = "freeze_frame"      # Freeze last frame


@dataclass
class WordTimestamp:
    """Word-level timestamp from Whisper"""
    word: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class SyncAnchor:
    """
    Sync anchor linking script text to visual events.
    Format in script: [SYNC:SLIDE_2] or [SYNC:CODE_1] or [SYNC:DIAGRAM]
    """
    anchor_type: str      # SLIDE, CODE, DIAGRAM, etc.
    anchor_id: str        # Full anchor like "SLIDE_2"
    word_index: int       # Index in word_timestamps where this anchor triggers
    timestamp: float      # Actual time in seconds
    slide_index: int      # Which slide this refers to


@dataclass
class VisualEvent:
    """A visual event on the timeline"""
    event_type: VisualEventType
    time_start: float           # When to start showing (seconds)
    time_end: float             # When to stop showing (seconds)
    duration: float             # Calculated duration
    asset_path: Optional[str]   # Path to video/image asset
    asset_url: Optional[str]    # URL to asset
    layer: int = 0              # Z-order (0=background, higher=foreground)
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0})
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "time_start": round(self.time_start, 3),
            "time_end": round(self.time_end, 3),
            "duration": round(self.duration, 3),
            "asset_path": self.asset_path,
            "asset_url": self.asset_url,
            "layer": self.layer,
            "position": self.position,
            "metadata": self.metadata
        }


@dataclass
class Timeline:
    """Complete timeline for video composition"""
    total_duration: float
    audio_track_path: Optional[str]
    audio_track_url: Optional[str]
    visual_events: List[VisualEvent]
    word_timestamps: List[WordTimestamp]
    sync_anchors: List[SyncAnchor]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_duration": round(self.total_duration, 3),
            "audio_track_path": self.audio_track_path,
            "audio_track_url": self.audio_track_url,
            "visual_events": [e.to_dict() for e in self.visual_events],
            "word_timestamps": [
                {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in self.word_timestamps
            ],
            "sync_anchors": [asdict(a) for a in self.sync_anchors],
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class TimelineBuilder:
    """
    Builds an event-driven timeline from audio timestamps and script.

    Usage:
        builder = TimelineBuilder()
        timeline = builder.build(
            word_timestamps=whisper_timestamps,
            slides=presentation_slides,
            audio_duration=total_audio_duration,
            audio_url=voiceover_url
        )
    """

    # Regex to detect sync anchors in text: [SYNC:SLIDE_1], [SYNC:CODE_2], etc.
    SYNC_ANCHOR_PATTERN = re.compile(r'\[SYNC:([A-Z]+_?\d*)\]', re.IGNORECASE)

    # Minimum duration for any visual event (prevents flicker)
    MIN_EVENT_DURATION = 0.5

    # Default transition duration
    TRANSITION_DURATION = 0.3

    def __init__(self):
        self.debug = True

    def log(self, message: str):
        if self.debug:
            print(f"[TIMELINE] {message}", flush=True)

    def build(
        self,
        word_timestamps: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
        audio_duration: float,
        audio_url: Optional[str] = None,
        audio_path: Optional[str] = None,
        animations: Dict[str, Dict[str, Any]] = None
    ) -> Timeline:
        """
        Build a complete timeline from word timestamps and slides.

        Args:
            word_timestamps: List of {"word": str, "start": float, "end": float}
            slides: List of slide data with voiceover_text
            audio_duration: Total audio duration in seconds
            audio_url: URL to the audio file
            audio_path: Local path to audio file
            animations: Dict mapping slide_id to animation info

        Returns:
            Timeline object with all visual events
        """
        self.log(f"Building timeline: {len(slides)} slides, {len(word_timestamps)} words, {audio_duration:.2f}s")

        # Convert to WordTimestamp objects
        words = [
            WordTimestamp(
                word=w.get("word", ""),
                start=float(w.get("start", 0)),
                end=float(w.get("end", 0))
            )
            for w in word_timestamps
        ]

        # Find sync anchors in voiceover text
        sync_anchors = self._find_sync_anchors(slides, words)

        # Calculate slide timings based on voiceover
        slide_timings = self._calculate_slide_timings(slides, words, audio_duration)

        # Build visual events
        visual_events = self._build_visual_events(
            slides, slide_timings, animations or {}
        )

        # Add transitions between slides
        visual_events = self._add_transitions(visual_events)

        # Sort events by start time and layer
        visual_events.sort(key=lambda e: (e.time_start, e.layer))

        timeline = Timeline(
            total_duration=audio_duration,
            audio_track_path=audio_path,
            audio_track_url=audio_url,
            visual_events=visual_events,
            word_timestamps=words,
            sync_anchors=sync_anchors,
            metadata={
                "slides_count": len(slides),
                "events_count": len(visual_events),
                "anchors_count": len(sync_anchors)
            }
        )

        self.log(f"Timeline built: {len(visual_events)} events, {len(sync_anchors)} anchors")
        return timeline

    def _find_sync_anchors(
        self,
        slides: List[Dict[str, Any]],
        words: List[WordTimestamp]
    ) -> List[SyncAnchor]:
        """
        Find sync anchors in slide voiceover text and map to word timestamps.

        Anchors can be explicit [SYNC:SLIDE_2] or implicit (start of each slide's voiceover).
        """
        anchors = []
        word_index = 0

        for slide_idx, slide in enumerate(slides):
            voiceover_text = slide.get("voiceover_text", "") or ""

            # Check for explicit anchors
            explicit_anchors = self.SYNC_ANCHOR_PATTERN.findall(voiceover_text)

            # Always clean voiceover text first (remove SYNC markers)
            clean_text = self.SYNC_ANCHOR_PATTERN.sub("", voiceover_text).strip()
            slide["voiceover_text"] = clean_text

            if explicit_anchors:
                for anchor_id in explicit_anchors:
                    anchor_type = anchor_id.split("_")[0] if "_" in anchor_id else anchor_id
                    # Find the word index for this anchor
                    # For now, use the start of this slide's voiceover
                    if word_index < len(words):
                        anchors.append(SyncAnchor(
                            anchor_type=anchor_type,
                            anchor_id=anchor_id,
                            word_index=word_index,
                            timestamp=words[word_index].start,
                            slide_index=slide_idx
                        ))
            else:
                # Create implicit anchor at slide start
                if word_index < len(words):
                    anchors.append(SyncAnchor(
                        anchor_type="SLIDE",
                        anchor_id=f"SLIDE_{slide_idx}",
                        word_index=word_index,
                        timestamp=words[word_index].start if word_index < len(words) else 0,
                        slide_index=slide_idx
                    ))

            # CRITICAL: Use CLEANED text for word count to match Whisper timestamps
            voiceover_word_count = len(clean_text.split())
            word_index += voiceover_word_count

        return anchors

    def _calculate_slide_timings(
        self,
        slides: List[Dict[str, Any]],
        words: List[WordTimestamp],
        total_duration: float
    ) -> List[Dict[str, float]]:
        """
        Calculate exact start/end times for each slide using PROPORTIONAL DISTRIBUTION.

        IMPROVED ALGORITHM (v2):
        - Uses character count as proportion (more accurate than word count)
        - Uses CUMULATIVE positioning to prevent drift accumulation
        - Audio duration is the absolute source of truth
        - Final adjustment ensures perfect alignment

        This fixes the sync drift problem where word count mismatches between
        script text and Whisper transcription caused accumulating delays.
        """
        timings = []

        # Calculate total characters across all slides (for proportional distribution)
        char_counts = []
        for slide in slides:
            voiceover_text = slide.get("voiceover_text", "") or ""
            # Clean any remaining markers
            clean_text = self.SYNC_ANCHOR_PATTERN.sub("", voiceover_text).strip()
            char_counts.append(len(clean_text))

        total_chars = sum(char_counts)

        if total_chars == 0:
            # Edge case: no voiceover text - distribute evenly
            duration_per_slide = max(self.MIN_EVENT_DURATION, total_duration / len(slides))
            current_time = 0.0
            for slide_idx in range(len(slides)):
                end_time = min(current_time + duration_per_slide, total_duration)
                timings.append({
                    "slide_index": slide_idx,
                    "start": round(current_time, 3),
                    "end": round(end_time, 3),
                    "duration": round(end_time - current_time, 3),
                    "word_start_idx": 0,
                    "word_end_idx": 0
                })
                current_time = end_time
            return timings

        # Calculate cumulative proportions for precise boundary calculation
        cumulative_proportions = []
        cumulative = 0.0
        for char_count in char_counts:
            cumulative += char_count / total_chars
            cumulative_proportions.append(cumulative)

        # Convert proportions to absolute timestamps
        # Using CUMULATIVE approach prevents drift accumulation
        previous_end = 0.0

        for slide_idx, proportion in enumerate(cumulative_proportions):
            # Calculate end time from cumulative proportion
            # This ensures no drift because each boundary is calculated from total_duration
            end_time = total_duration * proportion

            # Start time is previous slide's end (seamless transitions)
            start_time = previous_end

            # Apply minimum duration constraint
            if end_time - start_time < self.MIN_EVENT_DURATION:
                end_time = start_time + self.MIN_EVENT_DURATION
                # Don't let it exceed total duration
                end_time = min(end_time, total_duration)

            # Round to 3 decimal places for FFmpeg precision
            start_time = round(start_time, 3)
            end_time = round(end_time, 3)
            duration = round(end_time - start_time, 3)

            timings.append({
                "slide_index": slide_idx,
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "word_start_idx": 0,  # Word indices not used in proportional mode
                "word_end_idx": 0
            })

            self.log(f"Slide {slide_idx}: {start_time:.3f}s - {end_time:.3f}s ({duration:.3f}s) [{char_counts[slide_idx]} chars]")
            previous_end = end_time

        # CRITICAL: Ensure last slide ends EXACTLY at total_duration
        # This prevents any cumulative rounding errors
        if timings:
            timings[-1]["end"] = round(total_duration, 3)
            timings[-1]["duration"] = round(total_duration - timings[-1]["start"], 3)

        # Validation: check for gaps or overlaps
        for i in range(1, len(timings)):
            if abs(timings[i]["start"] - timings[i-1]["end"]) > 0.001:
                gap = timings[i]["start"] - timings[i-1]["end"]
                self.log(f"WARNING: Gap/overlap detected between slide {i-1} and {i}: {gap:.3f}s")
                # Fix it
                timings[i]["start"] = timings[i-1]["end"]
                timings[i]["duration"] = timings[i]["end"] - timings[i]["start"]

        total_calculated = sum(t["duration"] for t in timings)
        self.log(f"Total calculated duration: {total_calculated:.3f}s (audio: {total_duration:.3f}s)")

        return timings

    def _build_visual_events(
        self,
        slides: List[Dict[str, Any]],
        timings: List[Dict[str, float]],
        animations: Dict[str, Dict[str, Any]]
    ) -> List[VisualEvent]:
        """
        Build visual events for each slide.
        """
        events = []

        for timing in timings:
            slide_idx = timing["slide_index"]
            slide = slides[slide_idx]
            slide_id = slide.get("id", f"slide_{slide_idx}")
            slide_type = slide.get("type", "content")

            start_time = timing["start"]
            end_time = timing["end"]
            duration = timing["duration"]

            # Check if this slide has an animation
            animation_info = animations.get(slide_id)

            if animation_info and slide_type in ["code", "code_demo"]:
                # Code animation event
                animation_duration = animation_info.get("duration", duration)
                animation_url = animation_info.get("url")
                animation_path = animation_info.get("file_path")

                events.append(VisualEvent(
                    event_type=VisualEventType.CODE_ANIMATION,
                    time_start=start_time,
                    time_end=start_time + animation_duration,
                    duration=animation_duration,
                    asset_path=animation_path,
                    asset_url=animation_url,
                    layer=0,
                    metadata={
                        "slide_id": slide_id,
                        "slide_index": slide_idx,
                        "language": slide.get("language", "python")
                    }
                ))

                # If animation is shorter than voiceover, add freeze frame
                if animation_duration < duration:
                    freeze_start = start_time + animation_duration
                    events.append(VisualEvent(
                        event_type=VisualEventType.FREEZE_FRAME,
                        time_start=freeze_start,
                        time_end=end_time,
                        duration=end_time - freeze_start,
                        asset_path=animation_path,
                        asset_url=animation_url,
                        layer=0,
                        metadata={
                            "slide_id": slide_id,
                            "freeze_from": "last_frame"
                        }
                    ))
                    self.log(f"Slide {slide_idx}: freeze frame {freeze_start:.2f}s - {end_time:.2f}s")

            elif slide_type == "diagram":
                # Diagram event
                events.append(VisualEvent(
                    event_type=VisualEventType.DIAGRAM,
                    time_start=start_time,
                    time_end=end_time,
                    duration=duration,
                    asset_path=slide.get("diagram_path"),
                    asset_url=slide.get("diagram_url") or slide.get("image_url"),
                    layer=0,
                    metadata={
                        "slide_id": slide_id,
                        "diagram_type": slide.get("diagram_type")
                    }
                ))

            else:
                # Regular slide (image)
                events.append(VisualEvent(
                    event_type=VisualEventType.SLIDE,
                    time_start=start_time,
                    time_end=end_time,
                    duration=duration,
                    asset_path=slide.get("image_path"),
                    asset_url=slide.get("image_url"),
                    layer=0,
                    metadata={
                        "slide_id": slide_id,
                        "slide_type": slide_type,
                        "title": slide.get("title")
                    }
                ))

            # Add bullet point reveals if present
            bullet_points = slide.get("bullet_points", [])
            if bullet_points and duration > 1.0:
                self._add_bullet_reveals(events, bullet_points, start_time, end_time, slide_id)

        return events

    def _add_bullet_reveals(
        self,
        events: List[VisualEvent],
        bullets: List[str],
        start_time: float,
        end_time: float,
        slide_id: str
    ):
        """Add timed bullet point reveal events"""
        if not bullets:
            return

        duration = end_time - start_time
        interval = duration / (len(bullets) + 1)  # +1 for initial pause

        for i, bullet in enumerate(bullets):
            reveal_time = start_time + (interval * (i + 1))
            events.append(VisualEvent(
                event_type=VisualEventType.BULLET_REVEAL,
                time_start=reveal_time,
                time_end=end_time,
                duration=end_time - reveal_time,
                asset_path=None,
                asset_url=None,
                layer=1,  # Above slide
                metadata={
                    "slide_id": slide_id,
                    "bullet_index": i,
                    "bullet_text": bullet
                }
            ))

    def _add_transitions(self, events: List[VisualEvent]) -> List[VisualEvent]:
        """Add transition events between slides"""
        # For now, transitions are handled by FFmpeg crossfade
        # This is a placeholder for more complex transitions
        return events

    def build_ffmpeg_filtercomplex(self, timeline: Timeline) -> str:
        """
        Generate FFmpeg filtercomplex string for precise composition.

        This creates overlay commands with enable='between(t,start,end)' for each event.
        """
        filters = []
        inputs = []

        # Group events by layer
        layer_events = {}
        for event in timeline.visual_events:
            layer = event.layer
            if layer not in layer_events:
                layer_events[layer] = []
            layer_events[layer].append(event)

        # Build filter for each layer
        current_output = "0:v"  # Start with first input (background)

        for layer in sorted(layer_events.keys()):
            events = layer_events[layer]

            for i, event in enumerate(events):
                input_idx = len(inputs) + 1
                inputs.append(event.asset_path or event.asset_url)

                # Create overlay with time-based enable
                enable_expr = f"between(t,{event.time_start:.3f},{event.time_end:.3f})"

                if event.event_type == VisualEventType.FREEZE_FRAME:
                    # For freeze frame, loop the last frame
                    filter_str = f"[{input_idx}:v]loop=loop=-1:size=1:start=999999[frozen{i}];"
                    filter_str += f"[{current_output}][frozen{i}]overlay=enable='{enable_expr}'[out{layer}_{i}]"
                else:
                    filter_str = f"[{current_output}][{input_idx}:v]overlay=enable='{enable_expr}'[out{layer}_{i}]"

                filters.append(filter_str)
                current_output = f"out{layer}_{i}"

        filtercomplex = ";".join(filters)

        self.log(f"Generated filtercomplex with {len(inputs)} inputs")
        return filtercomplex, inputs, current_output


# Convenience function for quick timeline building
def build_timeline(
    word_timestamps: List[Dict[str, Any]],
    slides: List[Dict[str, Any]],
    audio_duration: float,
    audio_url: Optional[str] = None,
    animations: Dict[str, Dict[str, Any]] = None
) -> Timeline:
    """
    Quick function to build a timeline.

    Example:
        timeline = build_timeline(
            word_timestamps=[{"word": "Hello", "start": 0.0, "end": 0.5}, ...],
            slides=[{"id": "slide_1", "voiceover_text": "Hello world", ...}],
            audio_duration=30.5,
            audio_url="https://example.com/audio.mp3"
        )
    """
    builder = TimelineBuilder()
    return builder.build(
        word_timestamps=word_timestamps,
        slides=slides,
        audio_duration=audio_duration,
        audio_url=audio_url,
        animations=animations
    )
