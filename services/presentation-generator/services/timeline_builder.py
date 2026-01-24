"""
Timeline Builder Service

Builds an event-driven timeline for precise audio-video synchronization.
This is the core of professional-grade video generation.

The approach:
1. Audio (voiceover) is the source of truth
2. SSVS (Semantic Slide-Voiceover Synchronization) aligns slides to narration
3. Word-level timestamps enable sub-sentence precision
4. FFmpeg filtercomplex assembles with millisecond precision

SYNC METHODS:
- SSVS: Semantic alignment using TF-IDF embeddings and dynamic programming
- PROPORTIONAL: Fallback using character count distribution
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Import SSVS algorithms
from .sync import (
    SSVSSynchronizer,
    VoiceSegment,
    Slide as SSVSSlide,
    SynchronizationResult,
    DiagramAwareSynchronizer,
    Diagram,
    DiagramElement,
    DiagramElementType,
    BoundingBox,
    DiagramSyncResult,
    FocusAnimationGenerator,
    # Calibration
    SSVSCalibrator,
    CalibrationConfig,
    CalibrationPresets,
    SyncDiagnostic,
)


class SyncMethod(Enum):
    """Synchronization methods available"""
    SSVS = "ssvs"              # Semantic Slide-Voiceover Synchronization (recommended)
    PROPORTIONAL = "proportional"  # Character-based proportional distribution (fallback)


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
    DIAGRAM_FOCUS = "diagram_focus"    # Focus/highlight on diagram element


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
class DiagramFocusEvent:
    """Focus event for diagram element highlighting"""
    element_id: str
    element_label: str
    start_time: float
    end_time: float
    focus_type: str  # "highlight", "pointer", "outline"
    intensity: float
    bbox: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "element_label": self.element_label,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "focus_type": self.focus_type,
            "intensity": round(self.intensity, 2),
            "bbox": self.bbox
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
    diagram_focus_events: List[DiagramFocusEvent] = field(default_factory=list)
    sync_method: str = "ssvs"  # Track which method was used
    semantic_scores: Dict[str, float] = field(default_factory=dict)  # Slide ID -> semantic score
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
            "diagram_focus_events": [e.to_dict() for e in self.diagram_focus_events],
            "sync_method": self.sync_method,
            "semantic_scores": self.semantic_scores,
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class TimelineBuilder:
    """
    Builds an event-driven timeline from audio timestamps and script.

    Supports two synchronization methods:
    - SSVS: Semantic Slide-Voiceover Synchronization (default, recommended)
    - PROPORTIONAL: Character-based distribution (fallback)

    Usage:
        builder = TimelineBuilder(sync_method=SyncMethod.SSVS)
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

    def __init__(self,
                 sync_method: SyncMethod = SyncMethod.SSVS,
                 calibration_preset: str = "training_course"):
        """
        Initialize TimelineBuilder with SSVS and calibration.

        Args:
            sync_method: SyncMethod.SSVS (recommended) or SyncMethod.PROPORTIONAL
            calibration_preset: One of:
                - "default": Balanced configuration
                - "fast_speech": For fast speakers
                - "slow_speech": For slow speakers
                - "technical_content": For diagrams/code (more anticipation)
                - "simple_slides": For simple text slides
                - "training_course": Optimized for Viralify training videos
        """
        self.debug = True
        self.sync_method = sync_method
        self.ssvs_synchronizer = SSVSSynchronizer(
            alpha=0.6,   # Semantic weight
            beta=0.3,    # Temporal weight
            gamma=0.1    # Transition weight
        )
        self.diagram_synchronizer = DiagramAwareSynchronizer()

        # Initialize calibrator with preset
        preset_map = {
            "default": CalibrationPresets.default,
            "fast_speech": CalibrationPresets.fast_speech,
            "slow_speech": CalibrationPresets.slow_speech,
            "technical_content": CalibrationPresets.technical_content,
            "simple_slides": CalibrationPresets.simple_slides,
            "live_presentation": CalibrationPresets.live_presentation,
            "training_course": CalibrationPresets.training_course,
        }
        preset_fn = preset_map.get(calibration_preset, CalibrationPresets.training_course)
        self.calibration_config = preset_fn()
        self.calibrator = SSVSCalibrator(self.calibration_config)

        self.log(f"Initialized with sync_method={sync_method.value}, calibration_preset={calibration_preset}")

    def log(self, message: str):
        if self.debug:
            print(f"[TIMELINE] {message}", flush=True)

    def _convert_to_voice_segments(
        self,
        slides: List[Dict[str, Any]],
        words: List[WordTimestamp],
        total_duration: float
    ) -> List[VoiceSegment]:
        """
        Convert slide voiceover texts and word timestamps into VoiceSegment objects.

        Groups words by their corresponding slide voiceover text for SSVS processing.
        """
        segments = []
        word_idx = 0

        for slide_idx, slide in enumerate(slides):
            voiceover_text = slide.get("voiceover_text", "") or ""
            # Clean SYNC markers
            clean_text = self.SYNC_ANCHOR_PATTERN.sub("", voiceover_text).strip()

            if not clean_text:
                continue

            # Count words in this slide's voiceover
            word_count = len(clean_text.split())

            # Get the word timestamps for this segment
            segment_words = words[word_idx:word_idx + word_count] if word_idx < len(words) else []

            if segment_words:
                start_time = segment_words[0].start
                end_time = segment_words[-1].end
            elif segments:
                # No words, continue from last segment
                start_time = segments[-1].end_time
                end_time = start_time + 1.0  # Minimum duration
            else:
                start_time = 0.0
                end_time = 1.0

            segment = VoiceSegment(
                id=slide_idx,
                text=clean_text,
                start_time=start_time,
                end_time=end_time,
                word_timestamps=[
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in segment_words
                ]
            )
            segments.append(segment)
            word_idx += word_count

        self.log(f"Created {len(segments)} voice segments from {len(slides)} slides")
        return segments

    def _convert_to_ssvs_slides(self, slides: List[Dict[str, Any]]) -> List[SSVSSlide]:
        """Convert slide dictionaries to SSVS Slide objects."""
        ssvs_slides = []

        for idx, slide in enumerate(slides):
            voiceover_text = slide.get("voiceover_text", "") or ""
            clean_text = self.SYNC_ANCHOR_PATTERN.sub("", voiceover_text).strip()

            # Extract keywords from various slide fields
            keywords = []
            if slide.get("keywords"):
                keywords.extend(slide.get("keywords", []))
            if slide.get("bullet_points"):
                keywords.extend(slide.get("bullet_points", []))

            ssvs_slide = SSVSSlide(
                id=slide.get("id", f"slide_{idx}"),
                index=idx,
                title=slide.get("title", ""),
                content=slide.get("content", "") or slide.get("text", "") or "",
                voiceover_text=clean_text,
                keywords=keywords,
                slide_type=slide.get("type", "content")
            )
            ssvs_slides.append(ssvs_slide)

        return ssvs_slides

    def _apply_ssvs_results(
        self,
        slides: List[Dict[str, Any]],
        ssvs_results: List[SynchronizationResult],
        total_duration: float
    ) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        """
        Convert SSVS synchronization results to slide timings format.

        Returns:
            Tuple of (timings list, semantic_scores dict)
        """
        timings = []
        semantic_scores = {}

        for result in ssvs_results:
            slide_idx = result.slide_index

            timing = {
                "slide_index": slide_idx,
                "start": round(result.start_time, 3),
                "end": round(result.end_time, 3),
                "duration": round(result.end_time - result.start_time, 3),
                "word_start_idx": 0,
                "word_end_idx": 0,
                "semantic_score": result.semantic_score,
                "combined_score": result.combined_score
            }
            timings.append(timing)
            semantic_scores[slides[slide_idx].get("id", f"slide_{slide_idx}")] = result.semantic_score

            self.log(f"SSVS Slide {slide_idx}: {timing['start']:.3f}s - {timing['end']:.3f}s "
                    f"(semantic: {result.semantic_score:.3f})")

        # Ensure last slide ends at total_duration
        if timings:
            timings[-1]["end"] = round(total_duration, 3)
            timings[-1]["duration"] = round(total_duration - timings[-1]["start"], 3)

        return timings, semantic_scores

    def build(
        self,
        word_timestamps: List[Dict[str, Any]],
        slides: List[Dict[str, Any]],
        audio_duration: float,
        audio_url: Optional[str] = None,
        audio_path: Optional[str] = None,
        animations: Dict[str, Dict[str, Any]] = None,
        diagrams: Dict[str, Dict[str, Any]] = None
    ) -> Timeline:
        """
        Build a complete timeline from word timestamps and slides.

        Uses SSVS (Semantic Slide-Voiceover Synchronization) by default for
        optimal alignment between slides and narration.

        Args:
            word_timestamps: List of {"word": str, "start": float, "end": float}
            slides: List of slide data with voiceover_text
            audio_duration: Total audio duration in seconds
            audio_url: URL to the audio file
            audio_path: Local path to audio file
            animations: Dict mapping slide_id to animation info
            diagrams: Dict mapping slide_id to diagram structure for SSVS-D

        Returns:
            Timeline object with all visual events
        """
        self.log(f"Building timeline: {len(slides)} slides, {len(word_timestamps)} words, "
                f"{audio_duration:.2f}s, method={self.sync_method.value}")

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

        # Calculate slide timings based on sync method
        semantic_scores = {}
        diagram_focus_events = []

        if self.sync_method == SyncMethod.SSVS and len(slides) > 0 and len(words) > 0:
            # Use SSVS semantic synchronization
            self.log("Using SSVS semantic synchronization")
            try:
                slide_timings, semantic_scores = self._calculate_slide_timings_ssvs(
                    slides, words, audio_duration
                )

                # Apply SSVS-D for diagram slides
                if diagrams:
                    diagram_focus_events = self._synchronize_diagrams(
                        slides, slide_timings, diagrams, words
                    )

            except Exception as e:
                self.log(f"SSVS failed, falling back to proportional: {e}")
                slide_timings = self._calculate_slide_timings_proportional(slides, words, audio_duration)
        else:
            # Fallback to proportional distribution
            self.log("Using proportional synchronization (fallback)")
            slide_timings = self._calculate_slide_timings_proportional(slides, words, audio_duration)

        # Build visual events
        visual_events = self._build_visual_events(
            slides, slide_timings, animations or {}
        )

        # Add diagram focus events as visual events
        for focus_event in diagram_focus_events:
            visual_events.append(VisualEvent(
                event_type=VisualEventType.DIAGRAM_FOCUS,
                time_start=focus_event.start_time,
                time_end=focus_event.end_time,
                duration=focus_event.end_time - focus_event.start_time,
                asset_path=None,
                asset_url=None,
                layer=2,  # Above diagram
                metadata={
                    "element_id": focus_event.element_id,
                    "element_label": focus_event.element_label,
                    "focus_type": focus_event.focus_type,
                    "intensity": focus_event.intensity,
                    "bbox": focus_event.bbox
                }
            ))

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
            diagram_focus_events=diagram_focus_events,
            sync_method=self.sync_method.value,
            semantic_scores=semantic_scores,
            metadata={
                "slides_count": len(slides),
                "events_count": len(visual_events),
                "anchors_count": len(sync_anchors),
                "diagram_focus_count": len(diagram_focus_events),
                "avg_semantic_score": (
                    sum(semantic_scores.values()) / len(semantic_scores)
                    if semantic_scores else 0.0
                )
            }
        )

        self.log(f"Timeline built: {len(visual_events)} events, {len(sync_anchors)} anchors, "
                f"{len(diagram_focus_events)} focus events")
        return timeline

    def _calculate_slide_timings_ssvs(
        self,
        slides: List[Dict[str, Any]],
        words: List[WordTimestamp],
        total_duration: float
    ) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        """
        Calculate slide timings using SSVS semantic synchronization + calibration.

        Pipeline:
        1. Convert to SSVS format
        2. Run SSVS synchronization
        3. Apply calibration to fix audio-video offset
        4. Convert to timings format

        Returns:
            Tuple of (timings list, semantic_scores dict)
        """
        # Convert to SSVS format
        ssvs_slides = self._convert_to_ssvs_slides(slides)
        voice_segments = self._convert_to_voice_segments(slides, words, total_duration)

        if not ssvs_slides or not voice_segments:
            self.log("No slides or segments for SSVS, using fallback")
            return self._calculate_slide_timings_proportional(slides, words, total_duration), {}

        # Run SSVS synchronization with word-level refinement
        ssvs_results = self.ssvs_synchronizer.synchronize_with_word_timestamps(
            ssvs_slides,
            voice_segments,
            [{"word": w.word, "start": w.start, "end": w.end} for w in words]
        )

        if not ssvs_results:
            self.log("SSVS returned no results, using fallback")
            return self._calculate_slide_timings_proportional(slides, words, total_duration), {}

        # ═══════════════════════════════════════════════════════════════════
        # CALIBRATION: Fix audio-video offset issues
        # ═══════════════════════════════════════════════════════════════════
        self.log("Applying SSVS calibration to fix sync offset...")

        # Run diagnostic first
        diagnostic = SyncDiagnostic.analyze_timing(ssvs_results, voice_segments)
        self.log(f"Pre-calibration: speech_rate={diagnostic['stats']['speech_rate']:.0f} words/min")

        if diagnostic['issues']:
            for issue in diagnostic['issues'][:3]:  # Log first 3 issues
                self.log(f"  Issue: {issue}")

        # Apply calibration
        calibrated_results = self.calibrator.calibrate(ssvs_results, voice_segments)

        self.log(f"Calibration applied: global_offset={self.calibration_config.global_offset_ms}ms, "
                f"anticipation={self.calibration_config.semantic_anticipation_ms}ms")

        # Convert calibrated results to timings format
        return self._apply_ssvs_results(slides, calibrated_results, total_duration)

    def _synchronize_diagrams(
        self,
        slides: List[Dict[str, Any]],
        timings: List[Dict[str, float]],
        diagrams: Dict[str, Dict[str, Any]],
        words: List[WordTimestamp]
    ) -> List[DiagramFocusEvent]:
        """
        Apply SSVS-D diagram-aware synchronization for diagram slides.

        Returns:
            List of DiagramFocusEvent for rendering
        """
        focus_events = []

        for timing in timings:
            slide_idx = timing["slide_index"]
            slide = slides[slide_idx]
            slide_id = slide.get("id", f"slide_{slide_idx}")
            slide_type = slide.get("type", "content")

            # Only process diagram slides with diagram data
            if slide_type != "diagram" or slide_id not in diagrams:
                continue

            diagram_data = diagrams[slide_id]

            # Convert diagram data to SSVS-D format
            diagram = self._convert_to_diagram(diagram_data, slide_id)
            if not diagram or not diagram.elements:
                continue

            # Create voice segment for this slide
            voiceover_text = slide.get("voiceover_text", "") or ""
            clean_text = self.SYNC_ANCHOR_PATTERN.sub("", voiceover_text).strip()

            if not clean_text:
                continue

            segment = VoiceSegment(
                id=slide_idx,
                text=clean_text,
                start_time=timing["start"],
                end_time=timing["end"]
            )

            # Run SSVS-D synchronization
            try:
                sync_result = self.diagram_synchronizer.synchronize(diagram, [segment])

                # Convert focus sequence to DiagramFocusEvent objects
                for focus in sync_result.focus_sequence:
                    element = diagram.get_element_by_id(focus.element_id)
                    bbox_dict = None
                    if element:
                        bbox_dict = {
                            "x": element.bbox.x,
                            "y": element.bbox.y,
                            "width": element.bbox.width,
                            "height": element.bbox.height
                        }

                    focus_events.append(DiagramFocusEvent(
                        element_id=focus.element_id,
                        element_label=element.label if element else focus.element_id,
                        start_time=focus.start_time,
                        end_time=focus.end_time,
                        focus_type=focus.focus_type,
                        intensity=focus.intensity,
                        bbox=bbox_dict
                    ))

                self.log(f"Diagram {slide_id}: {len(sync_result.focus_sequence)} focus points, "
                        f"semantic={sync_result.semantic_score:.3f}")

            except Exception as e:
                self.log(f"SSVS-D failed for diagram {slide_id}: {e}")

        return focus_events

    def _convert_to_diagram(self, diagram_data: Dict[str, Any], slide_id: str) -> Optional[Diagram]:
        """Convert diagram data dict to SSVS-D Diagram object."""
        try:
            elements = []

            for elem_data in diagram_data.get("elements", []):
                bbox_data = elem_data.get("bbox", {})
                bbox = BoundingBox(
                    x=bbox_data.get("x", 0.0),
                    y=bbox_data.get("y", 0.0),
                    width=bbox_data.get("width", 0.1),
                    height=bbox_data.get("height", 0.1)
                )

                elem_type_str = elem_data.get("type", "node").upper()
                try:
                    elem_type = DiagramElementType[elem_type_str]
                except KeyError:
                    elem_type = DiagramElementType.NODE

                element = DiagramElement(
                    id=elem_data.get("id", f"elem_{len(elements)}"),
                    element_type=elem_type,
                    label=elem_data.get("label", ""),
                    bbox=bbox,
                    keywords=elem_data.get("keywords", []),
                    connected_to=elem_data.get("connected_to", []),
                    importance=elem_data.get("importance", 1.0)
                )
                elements.append(element)

            return Diagram(
                id=slide_id,
                title=diagram_data.get("title", ""),
                elements=elements,
                diagram_type=diagram_data.get("diagram_type", "flowchart"),
                reading_order=diagram_data.get("reading_order")
            )

        except Exception as e:
            self.log(f"Failed to convert diagram data: {e}")
            return None

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

    def _calculate_slide_timings_proportional(
        self,
        slides: List[Dict[str, Any]],
        words: List[WordTimestamp],
        total_duration: float
    ) -> List[Dict[str, float]]:
        """
        Calculate exact start/end times for each slide using PROPORTIONAL DISTRIBUTION.

        This is the FALLBACK method when SSVS is not available or fails.

        Algorithm:
        - Uses character count as proportion (more accurate than word count)
        - Uses CUMULATIVE positioning to prevent drift accumulation
        - Audio duration is the absolute source of truth
        - Final adjustment ensures perfect alignment
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
    audio_path: Optional[str] = None,
    animations: Dict[str, Dict[str, Any]] = None,
    diagrams: Dict[str, Dict[str, Any]] = None,
    sync_method: SyncMethod = SyncMethod.SSVS
) -> Timeline:
    """
    Quick function to build a timeline with SSVS synchronization.

    Uses semantic alignment by default for optimal slide-voiceover sync.

    Example:
        timeline = build_timeline(
            word_timestamps=[{"word": "Hello", "start": 0.0, "end": 0.5}, ...],
            slides=[{"id": "slide_1", "voiceover_text": "Hello world", ...}],
            audio_duration=30.5,
            audio_url="https://example.com/audio.mp3"
        )

    Args:
        word_timestamps: List of {"word": str, "start": float, "end": float}
        slides: List of slide data with voiceover_text
        audio_duration: Total audio duration in seconds
        audio_url: URL to the audio file
        audio_path: Local path to audio file
        animations: Dict mapping slide_id to animation info
        diagrams: Dict mapping slide_id to diagram structure for SSVS-D
        sync_method: SyncMethod.SSVS (default) or SyncMethod.PROPORTIONAL

    Returns:
        Timeline object with synchronized visual events
    """
    builder = TimelineBuilder(sync_method=sync_method)
    return builder.build(
        word_timestamps=word_timestamps,
        slides=slides,
        audio_duration=audio_duration,
        audio_url=audio_url,
        audio_path=audio_path,
        animations=animations,
        diagrams=diagrams
    )
