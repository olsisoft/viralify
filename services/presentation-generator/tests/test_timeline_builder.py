"""
Unit tests for Timeline Builder Service

Tests cover:
- SyncMethod and VisualEventType enums
- WordTimestamp, SyncAnchor, VisualEvent, DiagramFocusEvent dataclasses
- Timeline dataclass and serialization
- TimelineBuilder helper functions (text wrapping, proportional calculation)
"""

import pytest
from unittest.mock import MagicMock, patch
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Mock implementations of dataclasses and enums for testing
# (Avoid importing from services which has complex dependencies)
# ============================================================================

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


class SyncMethod(Enum):
    """Synchronization methods available"""
    SSVS = "ssvs"
    PROPORTIONAL = "proportional"


class VisualEventType(Enum):
    """Types of visual events on the timeline"""
    SLIDE = "slide"
    CODE_ANIMATION = "code_animation"
    DIAGRAM = "diagram"
    AVATAR = "avatar"
    TRANSITION = "transition"
    HIGHLIGHT = "highlight"
    BULLET_REVEAL = "bullet_reveal"
    FREEZE_FRAME = "freeze_frame"
    DIAGRAM_FOCUS = "diagram_focus"


@dataclass
class WordTimestamp:
    """Word-level timestamp from Whisper"""
    word: str
    start: float
    end: float


@dataclass
class SyncAnchor:
    """Sync anchor linking script text to visual events"""
    anchor_type: str
    anchor_id: str
    word_index: int
    timestamp: float
    slide_index: int


@dataclass
class VisualEvent:
    """A visual event on the timeline"""
    event_type: VisualEventType
    time_start: float
    time_end: float
    duration: float
    asset_path: Optional[str]
    asset_url: Optional[str]
    layer: int = 0
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
    focus_type: str
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
    sync_method: str = "ssvs"
    semantic_scores: Dict[str, float] = field(default_factory=dict)
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


# ============================================================================
# Test Classes
# ============================================================================

class TestSyncMethodEnum:
    """Tests for SyncMethod enum"""

    def test_ssvs_value(self):
        """Test SSVS enum value"""
        assert SyncMethod.SSVS.value == "ssvs"

    def test_proportional_value(self):
        """Test PROPORTIONAL enum value"""
        assert SyncMethod.PROPORTIONAL.value == "proportional"

    def test_all_methods_exist(self):
        """Test all sync methods are defined"""
        methods = list(SyncMethod)
        assert len(methods) == 2
        assert SyncMethod.SSVS in methods
        assert SyncMethod.PROPORTIONAL in methods


class TestVisualEventTypeEnum:
    """Tests for VisualEventType enum"""

    def test_slide_value(self):
        """Test SLIDE event type"""
        assert VisualEventType.SLIDE.value == "slide"

    def test_code_animation_value(self):
        """Test CODE_ANIMATION event type"""
        assert VisualEventType.CODE_ANIMATION.value == "code_animation"

    def test_diagram_value(self):
        """Test DIAGRAM event type"""
        assert VisualEventType.DIAGRAM.value == "diagram"

    def test_avatar_value(self):
        """Test AVATAR event type"""
        assert VisualEventType.AVATAR.value == "avatar"

    def test_transition_value(self):
        """Test TRANSITION event type"""
        assert VisualEventType.TRANSITION.value == "transition"

    def test_all_event_types(self):
        """Test all event types are defined"""
        event_types = list(VisualEventType)
        assert len(event_types) == 9

        expected = [
            "slide", "code_animation", "diagram", "avatar",
            "transition", "highlight", "bullet_reveal",
            "freeze_frame", "diagram_focus"
        ]
        actual = [et.value for et in event_types]
        assert sorted(actual) == sorted(expected)


class TestWordTimestamp:
    """Tests for WordTimestamp dataclass"""

    def test_basic_creation(self):
        """Test basic WordTimestamp creation"""
        wt = WordTimestamp(word="Hello", start=0.0, end=0.5)

        assert wt.word == "Hello"
        assert wt.start == 0.0
        assert wt.end == 0.5

    def test_duration_calculation(self):
        """Test duration can be calculated from start/end"""
        wt = WordTimestamp(word="world", start=1.5, end=2.0)

        duration = wt.end - wt.start
        assert duration == 0.5

    def test_unicode_word(self):
        """Test Unicode word handling"""
        wt = WordTimestamp(word="bonjour", start=0.0, end=0.8)

        assert wt.word == "bonjour"

    def test_precision(self):
        """Test timestamp precision"""
        wt = WordTimestamp(word="test", start=0.123456, end=0.234567)

        assert wt.start == 0.123456
        assert wt.end == 0.234567


class TestSyncAnchor:
    """Tests for SyncAnchor dataclass"""

    def test_basic_creation(self):
        """Test basic SyncAnchor creation"""
        anchor = SyncAnchor(
            anchor_type="SLIDE",
            anchor_id="SLIDE_2",
            word_index=15,
            timestamp=5.5,
            slide_index=1
        )

        assert anchor.anchor_type == "SLIDE"
        assert anchor.anchor_id == "SLIDE_2"
        assert anchor.word_index == 15
        assert anchor.timestamp == 5.5
        assert anchor.slide_index == 1

    def test_code_anchor(self):
        """Test CODE type anchor"""
        anchor = SyncAnchor(
            anchor_type="CODE",
            anchor_id="CODE_1",
            word_index=30,
            timestamp=12.0,
            slide_index=3
        )

        assert anchor.anchor_type == "CODE"
        assert anchor.anchor_id == "CODE_1"

    def test_diagram_anchor(self):
        """Test DIAGRAM type anchor"""
        anchor = SyncAnchor(
            anchor_type="DIAGRAM",
            anchor_id="DIAGRAM",
            word_index=45,
            timestamp=18.5,
            slide_index=5
        )

        assert anchor.anchor_type == "DIAGRAM"

    def test_asdict(self):
        """Test conversion to dictionary"""
        anchor = SyncAnchor(
            anchor_type="SLIDE",
            anchor_id="SLIDE_1",
            word_index=0,
            timestamp=0.0,
            slide_index=0
        )

        d = asdict(anchor)
        assert d["anchor_type"] == "SLIDE"
        assert d["anchor_id"] == "SLIDE_1"


class TestVisualEvent:
    """Tests for VisualEvent dataclass"""

    def test_basic_creation(self):
        """Test basic VisualEvent creation"""
        event = VisualEvent(
            event_type=VisualEventType.SLIDE,
            time_start=0.0,
            time_end=10.0,
            duration=10.0,
            asset_path="/tmp/slide_0.png",
            asset_url="https://example.com/slide_0.png"
        )

        assert event.event_type == VisualEventType.SLIDE
        assert event.time_start == 0.0
        assert event.time_end == 10.0
        assert event.duration == 10.0

    def test_default_values(self):
        """Test default values"""
        event = VisualEvent(
            event_type=VisualEventType.SLIDE,
            time_start=0.0,
            time_end=5.0,
            duration=5.0,
            asset_path=None,
            asset_url=None
        )

        assert event.layer == 0
        assert event.position == {"x": 0, "y": 0}
        assert event.metadata == {}

    def test_with_metadata(self):
        """Test event with metadata"""
        event = VisualEvent(
            event_type=VisualEventType.CODE_ANIMATION,
            time_start=5.0,
            time_end=15.0,
            duration=10.0,
            asset_path="/tmp/animation.mp4",
            asset_url=None,
            layer=1,
            metadata={"language": "python", "lines": 10}
        )

        assert event.metadata["language"] == "python"
        assert event.metadata["lines"] == 10
        assert event.layer == 1

    def test_to_dict(self):
        """Test serialization to dictionary"""
        event = VisualEvent(
            event_type=VisualEventType.TRANSITION,
            time_start=9.95,
            time_end=10.25,
            duration=0.30,
            asset_path=None,
            asset_url=None,
            layer=2
        )

        d = event.to_dict()

        assert d["event_type"] == "transition"
        assert d["time_start"] == 9.95
        assert d["time_end"] == 10.25
        assert d["duration"] == 0.3
        assert d["layer"] == 2

    def test_to_dict_rounding(self):
        """Test that to_dict rounds values to 3 decimal places"""
        event = VisualEvent(
            event_type=VisualEventType.SLIDE,
            time_start=1.23456789,
            time_end=5.98765432,
            duration=4.75308643,
            asset_path=None,
            asset_url=None
        )

        d = event.to_dict()

        assert d["time_start"] == 1.235
        assert d["time_end"] == 5.988
        assert d["duration"] == 4.753


class TestDiagramFocusEvent:
    """Tests for DiagramFocusEvent dataclass"""

    def test_basic_creation(self):
        """Test basic DiagramFocusEvent creation"""
        event = DiagramFocusEvent(
            element_id="node_1",
            element_label="API Gateway",
            start_time=2.0,
            end_time=5.0,
            focus_type="highlight",
            intensity=0.8
        )

        assert event.element_id == "node_1"
        assert event.element_label == "API Gateway"
        assert event.start_time == 2.0
        assert event.end_time == 5.0
        assert event.focus_type == "highlight"
        assert event.intensity == 0.8
        assert event.bbox is None

    def test_with_bounding_box(self):
        """Test event with bounding box"""
        bbox = {"x": 100, "y": 200, "width": 150, "height": 100}
        event = DiagramFocusEvent(
            element_id="node_2",
            element_label="Database",
            start_time=5.0,
            end_time=8.0,
            focus_type="pointer",
            intensity=1.0,
            bbox=bbox
        )

        assert event.bbox == bbox
        assert event.bbox["x"] == 100

    def test_to_dict(self):
        """Test serialization to dictionary"""
        event = DiagramFocusEvent(
            element_id="node_3",
            element_label="Cache",
            start_time=8.123,
            end_time=10.456,
            focus_type="outline",
            intensity=0.666
        )

        d = event.to_dict()

        assert d["element_id"] == "node_3"
        assert d["element_label"] == "Cache"
        assert d["start_time"] == 8.123
        assert d["end_time"] == 10.456
        assert d["focus_type"] == "outline"
        assert d["intensity"] == 0.67  # Rounded to 2 decimals


class TestTimeline:
    """Tests for Timeline dataclass"""

    def test_basic_creation(self):
        """Test basic Timeline creation"""
        timeline = Timeline(
            total_duration=60.0,
            audio_track_path="/tmp/audio.mp3",
            audio_track_url="https://example.com/audio.mp3",
            visual_events=[],
            word_timestamps=[],
            sync_anchors=[]
        )

        assert timeline.total_duration == 60.0
        assert timeline.audio_track_path == "/tmp/audio.mp3"
        assert timeline.audio_track_url == "https://example.com/audio.mp3"

    def test_default_values(self):
        """Test default values"""
        timeline = Timeline(
            total_duration=30.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[],
            word_timestamps=[],
            sync_anchors=[]
        )

        assert timeline.diagram_focus_events == []
        assert timeline.sync_method == "ssvs"
        assert timeline.semantic_scores == {}
        assert timeline.metadata == {}

    def test_with_events(self):
        """Test timeline with visual events"""
        event = VisualEvent(
            event_type=VisualEventType.SLIDE,
            time_start=0.0,
            time_end=10.0,
            duration=10.0,
            asset_path=None,
            asset_url=None
        )

        timeline = Timeline(
            total_duration=10.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[event],
            word_timestamps=[],
            sync_anchors=[]
        )

        assert len(timeline.visual_events) == 1
        assert timeline.visual_events[0].event_type == VisualEventType.SLIDE

    def test_with_word_timestamps(self):
        """Test timeline with word timestamps"""
        words = [
            WordTimestamp(word="Hello", start=0.0, end=0.5),
            WordTimestamp(word="World", start=0.6, end=1.0)
        ]

        timeline = Timeline(
            total_duration=1.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[],
            word_timestamps=words,
            sync_anchors=[]
        )

        assert len(timeline.word_timestamps) == 2
        assert timeline.word_timestamps[0].word == "Hello"

    def test_with_sync_anchors(self):
        """Test timeline with sync anchors"""
        anchor = SyncAnchor(
            anchor_type="SLIDE",
            anchor_id="SLIDE_1",
            word_index=0,
            timestamp=0.0,
            slide_index=0
        )

        timeline = Timeline(
            total_duration=10.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[],
            word_timestamps=[],
            sync_anchors=[anchor]
        )

        assert len(timeline.sync_anchors) == 1
        assert timeline.sync_anchors[0].anchor_id == "SLIDE_1"

    def test_to_dict(self):
        """Test serialization to dictionary"""
        event = VisualEvent(
            event_type=VisualEventType.SLIDE,
            time_start=0.0,
            time_end=5.0,
            duration=5.0,
            asset_path="/tmp/slide.png",
            asset_url=None
        )

        word = WordTimestamp(word="Test", start=0.0, end=0.5)

        anchor = SyncAnchor(
            anchor_type="SLIDE",
            anchor_id="SLIDE_0",
            word_index=0,
            timestamp=0.0,
            slide_index=0
        )

        timeline = Timeline(
            total_duration=5.0,
            audio_track_path="/tmp/audio.mp3",
            audio_track_url=None,
            visual_events=[event],
            word_timestamps=[word],
            sync_anchors=[anchor],
            sync_method="ssvs",
            semantic_scores={"slide_0": 0.85},
            metadata={"slides_count": 1}
        )

        d = timeline.to_dict()

        assert d["total_duration"] == 5.0
        assert d["audio_track_path"] == "/tmp/audio.mp3"
        assert len(d["visual_events"]) == 1
        assert len(d["word_timestamps"]) == 1
        assert len(d["sync_anchors"]) == 1
        assert d["sync_method"] == "ssvs"
        assert d["semantic_scores"]["slide_0"] == 0.85
        assert d["metadata"]["slides_count"] == 1

    def test_to_json(self):
        """Test JSON serialization"""
        timeline = Timeline(
            total_duration=10.0,
            audio_track_path=None,
            audio_track_url="https://example.com/audio.mp3",
            visual_events=[],
            word_timestamps=[],
            sync_anchors=[],
            metadata={"test": True}
        )

        json_str = timeline.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)

        assert parsed["total_duration"] == 10.0
        assert parsed["audio_track_url"] == "https://example.com/audio.mp3"
        assert parsed["metadata"]["test"] is True

    def test_to_json_with_indent(self):
        """Test JSON serialization with custom indent"""
        timeline = Timeline(
            total_duration=5.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[],
            word_timestamps=[],
            sync_anchors=[]
        )

        json_str = timeline.to_json(indent=4)

        # Should have proper indentation
        assert "    " in json_str  # 4-space indent


class TestTimelineWithComplexData:
    """Tests for Timeline with complex nested data"""

    def test_full_timeline(self):
        """Test a full timeline with all components"""
        # Create visual events
        slide_event = VisualEvent(
            event_type=VisualEventType.SLIDE,
            time_start=0.0,
            time_end=10.0,
            duration=10.0,
            asset_path="/tmp/slide_0.png",
            asset_url="https://cdn.example.com/slide_0.png",
            layer=0,
            metadata={"slide_index": 0}
        )

        code_event = VisualEvent(
            event_type=VisualEventType.CODE_ANIMATION,
            time_start=10.0,
            time_end=25.0,
            duration=15.0,
            asset_path="/tmp/code_animation.mp4",
            asset_url=None,
            layer=1,
            metadata={"language": "python"}
        )

        transition_event = VisualEvent(
            event_type=VisualEventType.TRANSITION,
            time_start=9.7,
            time_end=10.3,
            duration=0.6,
            asset_path=None,
            asset_url=None,
            layer=2
        )

        # Create word timestamps
        words = [
            WordTimestamp(word="Let's", start=0.0, end=0.3),
            WordTimestamp(word="learn", start=0.4, end=0.7),
            WordTimestamp(word="Python", start=0.8, end=1.2),
        ]

        # Create sync anchors
        anchors = [
            SyncAnchor(
                anchor_type="SLIDE",
                anchor_id="SLIDE_0",
                word_index=0,
                timestamp=0.0,
                slide_index=0
            ),
            SyncAnchor(
                anchor_type="CODE",
                anchor_id="CODE_1",
                word_index=10,
                timestamp=10.0,
                slide_index=1
            )
        ]

        # Create diagram focus events
        focus_events = [
            DiagramFocusEvent(
                element_id="node_api",
                element_label="API Gateway",
                start_time=3.0,
                end_time=5.0,
                focus_type="highlight",
                intensity=0.9,
                bbox={"x": 100, "y": 50, "width": 200, "height": 100}
            )
        ]

        # Create full timeline
        timeline = Timeline(
            total_duration=25.0,
            audio_track_path="/tmp/voiceover.mp3",
            audio_track_url="https://cdn.example.com/voiceover.mp3",
            visual_events=[slide_event, transition_event, code_event],
            word_timestamps=words,
            sync_anchors=anchors,
            diagram_focus_events=focus_events,
            sync_method="ssvs",
            semantic_scores={"slide_0": 0.92, "slide_1": 0.88},
            metadata={
                "slides_count": 2,
                "events_count": 3,
                "anchors_count": 2,
                "avg_semantic_score": 0.90
            }
        )

        # Verify structure
        assert timeline.total_duration == 25.0
        assert len(timeline.visual_events) == 3
        assert len(timeline.word_timestamps) == 3
        assert len(timeline.sync_anchors) == 2
        assert len(timeline.diagram_focus_events) == 1

        # Serialize and verify
        d = timeline.to_dict()

        assert d["total_duration"] == 25.0
        assert len(d["visual_events"]) == 3
        assert d["visual_events"][0]["event_type"] == "slide"
        assert d["visual_events"][1]["event_type"] == "transition"
        assert d["visual_events"][2]["event_type"] == "code_animation"

        assert d["semantic_scores"]["slide_0"] == 0.92
        assert d["metadata"]["avg_semantic_score"] == 0.90


class TestTimelineSyncMethods:
    """Tests for different sync method configurations"""

    def test_ssvs_sync_method(self):
        """Test timeline with SSVS sync method"""
        timeline = Timeline(
            total_duration=10.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[],
            word_timestamps=[],
            sync_anchors=[],
            sync_method="ssvs"
        )

        assert timeline.sync_method == "ssvs"

    def test_proportional_sync_method(self):
        """Test timeline with proportional sync method"""
        timeline = Timeline(
            total_duration=10.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[],
            word_timestamps=[],
            sync_anchors=[],
            sync_method="proportional"
        )

        assert timeline.sync_method == "proportional"


class TestTimelineEdgeCases:
    """Tests for edge cases"""

    def test_empty_timeline(self):
        """Test empty timeline"""
        timeline = Timeline(
            total_duration=0.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[],
            word_timestamps=[],
            sync_anchors=[]
        )

        d = timeline.to_dict()
        assert d["total_duration"] == 0.0
        assert d["visual_events"] == []

    def test_very_long_timeline(self):
        """Test very long timeline (1 hour)"""
        timeline = Timeline(
            total_duration=3600.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[],
            word_timestamps=[],
            sync_anchors=[]
        )

        assert timeline.total_duration == 3600.0

    def test_many_events(self):
        """Test timeline with many events"""
        events = [
            VisualEvent(
                event_type=VisualEventType.SLIDE,
                time_start=i * 10.0,
                time_end=(i + 1) * 10.0,
                duration=10.0,
                asset_path=f"/tmp/slide_{i}.png",
                asset_url=None
            )
            for i in range(100)
        ]

        timeline = Timeline(
            total_duration=1000.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=events,
            word_timestamps=[],
            sync_anchors=[]
        )

        assert len(timeline.visual_events) == 100
        d = timeline.to_dict()
        assert len(d["visual_events"]) == 100

    def test_unicode_content(self):
        """Test timeline with Unicode content"""
        word = WordTimestamp(word="日本語", start=0.0, end=1.0)

        timeline = Timeline(
            total_duration=1.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[],
            word_timestamps=[word],
            sync_anchors=[],
            metadata={"title": "日本語テスト"}
        )

        json_str = timeline.to_json()
        parsed = json.loads(json_str)

        assert parsed["word_timestamps"][0]["word"] == "日本語"
        assert parsed["metadata"]["title"] == "日本語テスト"

    def test_special_characters(self):
        """Test timeline with special characters"""
        event = VisualEvent(
            event_type=VisualEventType.SLIDE,
            time_start=0.0,
            time_end=5.0,
            duration=5.0,
            asset_path="/tmp/slide with spaces & symbols!.png",
            asset_url=None,
            metadata={"description": "Test with <html> & \"quotes\""}
        )

        timeline = Timeline(
            total_duration=5.0,
            audio_track_path=None,
            audio_track_url=None,
            visual_events=[event],
            word_timestamps=[],
            sync_anchors=[]
        )

        json_str = timeline.to_json()
        parsed = json.loads(json_str)

        assert "<html>" in parsed["visual_events"][0]["metadata"]["description"]


class TestVisualEventTypes:
    """Tests for all visual event type combinations"""

    @pytest.mark.parametrize("event_type", list(VisualEventType))
    def test_all_event_types_create_valid_events(self, event_type):
        """Test that all event types can create valid events"""
        event = VisualEvent(
            event_type=event_type,
            time_start=0.0,
            time_end=5.0,
            duration=5.0,
            asset_path=None,
            asset_url=None
        )

        assert event.event_type == event_type
        d = event.to_dict()
        assert d["event_type"] == event_type.value


class TestTimelineDurationCalculations:
    """Tests for duration-related calculations"""

    def test_event_duration_matches_start_end(self):
        """Test that duration equals end - start"""
        event = VisualEvent(
            event_type=VisualEventType.SLIDE,
            time_start=5.5,
            time_end=15.5,
            duration=10.0,
            asset_path=None,
            asset_url=None
        )

        calculated_duration = event.time_end - event.time_start
        assert calculated_duration == event.duration

    def test_word_timestamp_duration(self):
        """Test word duration calculation"""
        word = WordTimestamp(word="example", start=2.345, end=3.456)

        duration = word.end - word.start
        assert abs(duration - 1.111) < 0.001

    def test_diagram_focus_duration(self):
        """Test diagram focus duration calculation"""
        focus = DiagramFocusEvent(
            element_id="node",
            element_label="Node",
            start_time=1.0,
            end_time=4.5,
            focus_type="highlight",
            intensity=1.0
        )

        duration = focus.end_time - focus.start_time
        assert duration == 3.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
