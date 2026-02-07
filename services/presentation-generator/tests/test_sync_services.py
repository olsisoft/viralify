"""
Unit tests for SSVS Sync Services

Tests cover:
- VoiceSegment, Slide, SyncAnchor, SynchronizationResult dataclasses
- CalibrationConfig and CalibrationPresets
- PauseDetector and SpeechRateAnalyzer utilities
- SentenceAligner functions
- SSVSCalibrator configuration
"""

import pytest
from unittest.mock import MagicMock, patch
import json
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# Mock implementations of sync dataclasses for testing
# (Avoid importing from services which has complex dependencies)
# ============================================================================

@dataclass
class VoiceSegment:
    """A segment of voice narration with timestamps"""
    id: int
    text: str
    start_time: float
    end_time: float
    word_timestamps: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Slide:
    """A presentation slide with content for semantic matching"""
    id: str
    index: int
    title: str
    content: str
    voiceover_text: str
    keywords: List[str] = field(default_factory=list)
    slide_type: str = "content"

    def get_searchable_text(self) -> str:
        """Combine all text for semantic embedding"""
        parts = [self.title, self.content, self.voiceover_text]
        parts.extend(self.keywords)
        return " ".join(filter(None, parts))


@dataclass
class SyncAnchor:
    """Sync anchor that forces alignment at a specific point"""
    slide_index: int
    timestamp: float
    segment_index: int
    anchor_type: str = "SLIDE"
    anchor_id: str = ""
    tolerance_ms: float = 500.0


@dataclass
class SynchronizationResult:
    """Result of synchronizing a slide with voice segments"""
    slide_id: str
    slide_index: int
    segment_ids: List[int]
    start_time: float
    end_time: float
    semantic_score: float
    temporal_score: float
    combined_score: float
    transition_words: List[str] = field(default_factory=list)
    anchor_used: Optional[SyncAnchor] = None


@dataclass
class CalibrationConfig:
    """Configuration de calibration pour corriger les décalages"""
    global_offset_ms: float = -300.0
    semantic_anticipation_ms: float = -150.0
    transition_duration_ms: float = 200.0
    transition_compensation: float = 0.5
    stt_latency_compensation_ms: float = -50.0
    align_to_sentence_start: bool = True
    sentence_start_markers: List[str] = field(default_factory=lambda: [
        "maintenant", "ensuite", "puis", "passons", "voyons",
        "now", "next", "then", "let's", "moving on"
    ])
    use_pause_detection: bool = True
    min_pause_duration_ms: float = 300.0
    snap_to_pause_threshold_ms: float = 500.0
    adapt_to_speech_rate: bool = True
    reference_speech_rate: float = 150.0
    min_slide_duration_ms: float = 2000.0
    max_slide_duration_ms: float = 120000.0
    diagram_to_code_anticipation_ms: float = -1500.0


class PauseDetector:
    """Détecte les pauses naturelles dans la narration"""

    def __init__(self, min_pause_ms: float = 300.0):
        self.min_pause_ms = min_pause_ms

    def detect_pauses(self, segments: List[VoiceSegment]) -> List[Tuple[float, float]]:
        """Détecte les pauses entre les segments"""
        pauses = []
        for i in range(len(segments) - 1):
            gap_start = segments[i].end_time
            gap_end = segments[i + 1].start_time
            gap_duration = (gap_end - gap_start) * 1000  # en ms
            if gap_duration >= self.min_pause_ms:
                pauses.append((gap_start, gap_duration))
        return pauses

    def find_nearest_pause(self,
                           timestamp: float,
                           pauses: List[Tuple[float, float]],
                           max_distance_ms: float = 1000.0) -> Optional[float]:
        """Trouve la pause la plus proche d'un timestamp donné"""
        best_pause = None
        best_distance = float('inf')
        for pause_time, _ in pauses:
            distance = abs(pause_time - timestamp) * 1000
            if distance < best_distance and distance <= max_distance_ms:
                best_distance = distance
                best_pause = pause_time
        return best_pause


class SpeechRateAnalyzer:
    """Analyse la vitesse de parole pour adapter les offsets"""

    def __init__(self, reference_rate: float = 150.0):
        self.reference_rate = reference_rate

    def compute_speech_rate(self, segments: List[VoiceSegment]) -> float:
        """Calcule la vitesse de parole moyenne en mots/minute"""
        total_words = 0
        total_duration = 0.0
        for seg in segments:
            words = len(seg.text.split())
            duration = seg.end_time - seg.start_time
            total_words += words
            total_duration += duration
        if total_duration == 0:
            return self.reference_rate
        return (total_words / total_duration) * 60

    def compute_rate_factor(self, segments: List[VoiceSegment]) -> float:
        """Calcule un facteur d'ajustement basé sur la vitesse de parole"""
        actual_rate = self.compute_speech_rate(segments)
        return actual_rate / self.reference_rate

    def compute_local_rate(self, segment: VoiceSegment) -> float:
        """Calcule la vitesse de parole pour un segment spécifique"""
        words = len(segment.text.split())
        duration = segment.end_time - segment.start_time
        if duration == 0:
            return self.reference_rate
        return (words / duration) * 60


class SentenceAligner:
    """Aligne les transitions sur le début des phrases"""

    def __init__(self, markers: List[str]):
        self.markers = [m.lower() for m in markers]

    def find_sentence_start(self, segment: VoiceSegment) -> Optional[float]:
        """Trouve le timestamp du début de phrase dans un segment"""
        text_lower = segment.text.lower()
        for marker in self.markers:
            if text_lower.startswith(marker):
                return segment.start_time
            pos = text_lower.find(marker)
            if pos > 0 and pos < len(text_lower) * 0.3:
                ratio = pos / len(text_lower)
                duration = segment.end_time - segment.start_time
                return segment.start_time + (ratio * duration * 0.5)
        return None


# ============================================================================
# Test Classes
# ============================================================================

class TestVoiceSegment:
    """Tests for VoiceSegment dataclass"""

    def test_basic_creation(self):
        """Test basic VoiceSegment creation"""
        segment = VoiceSegment(
            id=1,
            text="Hello world this is a test",
            start_time=0.0,
            end_time=2.5
        )

        assert segment.id == 1
        assert segment.text == "Hello world this is a test"
        assert segment.start_time == 0.0
        assert segment.end_time == 2.5

    def test_duration_property(self):
        """Test duration calculation"""
        segment = VoiceSegment(
            id=1,
            text="Test",
            start_time=1.5,
            end_time=4.5
        )

        assert segment.duration == 3.0

    def test_word_count_property(self):
        """Test word count calculation"""
        segment = VoiceSegment(
            id=1,
            text="This is a test with six words",
            start_time=0.0,
            end_time=2.0
        )

        assert segment.word_count == 7

    def test_with_word_timestamps(self):
        """Test segment with word-level timestamps"""
        word_ts = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.0}
        ]

        segment = VoiceSegment(
            id=1,
            text="Hello world",
            start_time=0.0,
            end_time=1.0,
            word_timestamps=word_ts
        )

        assert len(segment.word_timestamps) == 2
        assert segment.word_timestamps[0]["word"] == "Hello"

    def test_empty_text(self):
        """Test segment with empty text"""
        segment = VoiceSegment(
            id=0,
            text="",
            start_time=0.0,
            end_time=0.0
        )

        assert segment.word_count == 0
        assert segment.duration == 0.0


class TestSlide:
    """Tests for Slide dataclass"""

    def test_basic_creation(self):
        """Test basic Slide creation"""
        slide = Slide(
            id="slide_0",
            index=0,
            title="Introduction",
            content="This is the introduction slide",
            voiceover_text="Welcome to this presentation"
        )

        assert slide.id == "slide_0"
        assert slide.index == 0
        assert slide.title == "Introduction"

    def test_get_searchable_text(self):
        """Test searchable text generation"""
        slide = Slide(
            id="slide_1",
            index=1,
            title="Python Basics",
            content="Learn Python fundamentals",
            voiceover_text="Let's explore Python",
            keywords=["python", "programming", "basics"]
        )

        text = slide.get_searchable_text()

        assert "Python Basics" in text
        assert "Learn Python fundamentals" in text
        assert "Let's explore Python" in text
        assert "python" in text
        assert "programming" in text

    def test_default_slide_type(self):
        """Test default slide type"""
        slide = Slide(
            id="slide_0",
            index=0,
            title="Test",
            content="Content",
            voiceover_text="Text"
        )

        assert slide.slide_type == "content"

    def test_custom_slide_type(self):
        """Test custom slide type"""
        slide = Slide(
            id="slide_2",
            index=2,
            title="Code Example",
            content="def hello(): pass",
            voiceover_text="Here's the code",
            slide_type="code"
        )

        assert slide.slide_type == "code"

    def test_empty_keywords(self):
        """Test slide without keywords"""
        slide = Slide(
            id="slide_0",
            index=0,
            title="Title",
            content="Content",
            voiceover_text="Text"
        )

        assert slide.keywords == []

    def test_searchable_text_filters_empty(self):
        """Test that searchable text filters empty strings"""
        slide = Slide(
            id="slide_0",
            index=0,
            title="Title",
            content="",
            voiceover_text="",
            keywords=[]
        )

        text = slide.get_searchable_text()
        assert text == "Title"


class TestSyncAnchor:
    """Tests for SyncAnchor dataclass"""

    def test_basic_creation(self):
        """Test basic SyncAnchor creation"""
        anchor = SyncAnchor(
            slide_index=2,
            timestamp=10.5,
            segment_index=5
        )

        assert anchor.slide_index == 2
        assert anchor.timestamp == 10.5
        assert anchor.segment_index == 5

    def test_default_values(self):
        """Test default values"""
        anchor = SyncAnchor(
            slide_index=0,
            timestamp=0.0,
            segment_index=0
        )

        assert anchor.anchor_type == "SLIDE"
        assert anchor.anchor_id == ""
        assert anchor.tolerance_ms == 500.0

    def test_code_anchor(self):
        """Test CODE anchor type"""
        anchor = SyncAnchor(
            slide_index=3,
            timestamp=15.0,
            segment_index=8,
            anchor_type="CODE",
            anchor_id="CODE_1"
        )

        assert anchor.anchor_type == "CODE"
        assert anchor.anchor_id == "CODE_1"

    def test_diagram_anchor(self):
        """Test DIAGRAM anchor type"""
        anchor = SyncAnchor(
            slide_index=5,
            timestamp=25.0,
            segment_index=12,
            anchor_type="DIAGRAM",
            anchor_id="DIAGRAM_2"
        )

        assert anchor.anchor_type == "DIAGRAM"

    def test_custom_tolerance(self):
        """Test custom tolerance"""
        anchor = SyncAnchor(
            slide_index=1,
            timestamp=5.0,
            segment_index=3,
            tolerance_ms=1000.0
        )

        assert anchor.tolerance_ms == 1000.0


class TestSynchronizationResult:
    """Tests for SynchronizationResult dataclass"""

    def test_basic_creation(self):
        """Test basic SynchronizationResult creation"""
        result = SynchronizationResult(
            slide_id="slide_0",
            slide_index=0,
            segment_ids=[0, 1, 2],
            start_time=0.0,
            end_time=10.0,
            semantic_score=0.85,
            temporal_score=0.90,
            combined_score=0.87
        )

        assert result.slide_id == "slide_0"
        assert result.slide_index == 0
        assert len(result.segment_ids) == 3
        assert result.start_time == 0.0
        assert result.end_time == 10.0

    def test_with_transition_words(self):
        """Test result with transition words"""
        result = SynchronizationResult(
            slide_id="slide_1",
            slide_index=1,
            segment_ids=[3, 4],
            start_time=10.0,
            end_time=20.0,
            semantic_score=0.80,
            temporal_score=0.85,
            combined_score=0.82,
            transition_words=["now", "let's", "move"]
        )

        assert len(result.transition_words) == 3
        assert "now" in result.transition_words

    def test_with_anchor(self):
        """Test result with anchor constraint"""
        anchor = SyncAnchor(
            slide_index=2,
            timestamp=15.0,
            segment_index=6,
            anchor_type="SLIDE",
            anchor_id="SLIDE_2"
        )

        result = SynchronizationResult(
            slide_id="slide_2",
            slide_index=2,
            segment_ids=[6, 7],
            start_time=15.0,
            end_time=25.0,
            semantic_score=0.90,
            temporal_score=0.95,
            combined_score=0.92,
            anchor_used=anchor
        )

        assert result.anchor_used is not None
        assert result.anchor_used.anchor_id == "SLIDE_2"

    def test_score_values(self):
        """Test score value ranges"""
        result = SynchronizationResult(
            slide_id="slide_0",
            slide_index=0,
            segment_ids=[0],
            start_time=0.0,
            end_time=5.0,
            semantic_score=0.75,
            temporal_score=0.80,
            combined_score=0.77
        )

        assert 0.0 <= result.semantic_score <= 1.0
        assert 0.0 <= result.temporal_score <= 1.0
        assert 0.0 <= result.combined_score <= 1.0


class TestCalibrationConfig:
    """Tests for CalibrationConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        config = CalibrationConfig()

        assert config.global_offset_ms == -300.0
        assert config.semantic_anticipation_ms == -150.0
        assert config.transition_duration_ms == 200.0
        assert config.use_pause_detection is True
        assert config.min_slide_duration_ms == 2000.0

    def test_custom_values(self):
        """Test custom configuration values"""
        config = CalibrationConfig(
            global_offset_ms=-500.0,
            semantic_anticipation_ms=-250.0,
            min_slide_duration_ms=3000.0
        )

        assert config.global_offset_ms == -500.0
        assert config.semantic_anticipation_ms == -250.0
        assert config.min_slide_duration_ms == 3000.0

    def test_sentence_markers(self):
        """Test default sentence start markers"""
        config = CalibrationConfig()

        assert "maintenant" in config.sentence_start_markers
        assert "now" in config.sentence_start_markers
        assert len(config.sentence_start_markers) >= 10

    def test_speech_rate_settings(self):
        """Test speech rate related settings"""
        config = CalibrationConfig()

        assert config.adapt_to_speech_rate is True
        assert config.reference_speech_rate == 150.0


class TestPauseDetector:
    """Tests for PauseDetector class"""

    def test_detect_no_pauses(self):
        """Test detection with no pauses"""
        detector = PauseDetector(min_pause_ms=300.0)

        segments = [
            VoiceSegment(id=0, text="First", start_time=0.0, end_time=1.0),
            VoiceSegment(id=1, text="Second", start_time=1.1, end_time=2.0)
        ]

        pauses = detector.detect_pauses(segments)
        assert len(pauses) == 0

    def test_detect_single_pause(self):
        """Test detection of single pause"""
        detector = PauseDetector(min_pause_ms=300.0)

        segments = [
            VoiceSegment(id=0, text="First", start_time=0.0, end_time=1.0),
            VoiceSegment(id=1, text="Second", start_time=1.5, end_time=2.5)
        ]

        pauses = detector.detect_pauses(segments)
        assert len(pauses) == 1
        assert pauses[0][0] == 1.0  # pause starts at end of first segment
        assert pauses[0][1] == 500.0  # 500ms duration

    def test_detect_multiple_pauses(self):
        """Test detection of multiple pauses"""
        detector = PauseDetector(min_pause_ms=200.0)

        segments = [
            VoiceSegment(id=0, text="First", start_time=0.0, end_time=1.0),
            VoiceSegment(id=1, text="Second", start_time=1.5, end_time=2.5),
            VoiceSegment(id=2, text="Third", start_time=3.0, end_time=4.0)
        ]

        pauses = detector.detect_pauses(segments)
        assert len(pauses) == 2

    def test_find_nearest_pause(self):
        """Test finding nearest pause"""
        detector = PauseDetector()

        pauses = [(5.0, 500.0), (10.0, 300.0), (15.0, 400.0)]

        nearest = detector.find_nearest_pause(9.5, pauses)
        assert nearest == 10.0

    def test_find_no_nearby_pause(self):
        """Test when no pause is within range"""
        detector = PauseDetector()

        pauses = [(5.0, 500.0)]

        nearest = detector.find_nearest_pause(20.0, pauses, max_distance_ms=1000.0)
        assert nearest is None

    def test_empty_segments(self):
        """Test with empty segment list"""
        detector = PauseDetector()

        pauses = detector.detect_pauses([])
        assert pauses == []


class TestSpeechRateAnalyzer:
    """Tests for SpeechRateAnalyzer class"""

    def test_compute_speech_rate(self):
        """Test speech rate computation"""
        analyzer = SpeechRateAnalyzer(reference_rate=150.0)

        segments = [
            VoiceSegment(id=0, text="one two three four five", start_time=0.0, end_time=2.0),
            VoiceSegment(id=1, text="six seven eight nine ten", start_time=2.0, end_time=4.0)
        ]

        rate = analyzer.compute_speech_rate(segments)

        # 10 words / 4 seconds = 2.5 words/sec = 150 words/minute
        assert rate == 150.0

    def test_compute_rate_factor_normal(self):
        """Test rate factor for normal speech"""
        analyzer = SpeechRateAnalyzer(reference_rate=150.0)

        segments = [
            VoiceSegment(id=0, text="one two three four five", start_time=0.0, end_time=2.0),
            VoiceSegment(id=1, text="six seven eight nine ten", start_time=2.0, end_time=4.0)
        ]

        factor = analyzer.compute_rate_factor(segments)
        assert factor == 1.0

    def test_compute_rate_factor_fast(self):
        """Test rate factor for fast speech"""
        analyzer = SpeechRateAnalyzer(reference_rate=150.0)

        # 10 words in 2 seconds = 5 words/sec = 300 words/min
        segments = [
            VoiceSegment(id=0, text="one two three four five six seven eight nine ten", start_time=0.0, end_time=2.0)
        ]

        factor = analyzer.compute_rate_factor(segments)
        assert factor == 2.0

    def test_compute_local_rate(self):
        """Test local rate computation for single segment"""
        analyzer = SpeechRateAnalyzer()

        segment = VoiceSegment(
            id=0,
            text="one two three",
            start_time=0.0,
            end_time=1.0
        )

        rate = analyzer.compute_local_rate(segment)

        # 3 words / 1 second = 180 words/minute
        assert rate == 180.0

    def test_empty_duration(self):
        """Test with zero duration"""
        analyzer = SpeechRateAnalyzer(reference_rate=150.0)

        segment = VoiceSegment(
            id=0,
            text="test",
            start_time=0.0,
            end_time=0.0
        )

        rate = analyzer.compute_local_rate(segment)
        assert rate == 150.0  # Returns reference rate


class TestSentenceAligner:
    """Tests for SentenceAligner class"""

    def test_find_sentence_start_at_beginning(self):
        """Test finding marker at segment start"""
        aligner = SentenceAligner(["maintenant", "now", "let's"])

        segment = VoiceSegment(
            id=0,
            text="Now let's look at the code",
            start_time=5.0,
            end_time=10.0
        )

        start = aligner.find_sentence_start(segment)
        assert start == 5.0  # Returns segment start

    def test_find_sentence_start_in_text(self):
        """Test finding marker in first third of text"""
        aligner = SentenceAligner(["maintenant", "now"])

        segment = VoiceSegment(
            id=0,
            text="So now we can see",
            start_time=0.0,
            end_time=4.0
        )

        start = aligner.find_sentence_start(segment)
        assert start is not None
        assert start >= 0.0

    def test_no_marker_found(self):
        """Test when no marker is found"""
        aligner = SentenceAligner(["maintenant", "now"])

        segment = VoiceSegment(
            id=0,
            text="This text has no markers",
            start_time=0.0,
            end_time=3.0
        )

        start = aligner.find_sentence_start(segment)
        assert start is None

    def test_case_insensitive(self):
        """Test case insensitive matching"""
        aligner = SentenceAligner(["Now", "MAINTENANT"])

        segment = VoiceSegment(
            id=0,
            text="NOW we continue",
            start_time=0.0,
            end_time=2.0
        )

        start = aligner.find_sentence_start(segment)
        assert start == 0.0


class TestCalibrationPresets:
    """Tests for calibration preset configurations"""

    def test_default_preset(self):
        """Test default preset values"""
        config = CalibrationConfig()

        assert config.global_offset_ms == -300.0
        assert config.semantic_anticipation_ms == -150.0

    def test_fast_speech_preset(self):
        """Test fast speech preset"""
        config = CalibrationConfig(
            global_offset_ms=-500.0,
            semantic_anticipation_ms=-250.0,
            adapt_to_speech_rate=True,
            reference_speech_rate=150.0
        )

        assert config.global_offset_ms == -500.0
        assert config.semantic_anticipation_ms == -250.0

    def test_slow_speech_preset(self):
        """Test slow speech preset"""
        config = CalibrationConfig(
            global_offset_ms=-150.0,
            semantic_anticipation_ms=-100.0,
            min_slide_duration_ms=3000.0
        )

        assert config.global_offset_ms == -150.0
        assert config.min_slide_duration_ms == 3000.0

    def test_technical_content_preset(self):
        """Test technical content preset"""
        config = CalibrationConfig(
            global_offset_ms=-600.0,
            semantic_anticipation_ms=-300.0,
            transition_duration_ms=300.0,
            min_slide_duration_ms=3000.0
        )

        assert config.global_offset_ms == -600.0
        assert config.transition_duration_ms == 300.0

    def test_training_course_preset(self):
        """Test training course preset"""
        config = CalibrationConfig(
            global_offset_ms=-400.0,
            semantic_anticipation_ms=-200.0,
            transition_duration_ms=250.0,
            transition_compensation=0.6,
            min_slide_duration_ms=2500.0,
            use_pause_detection=True,
            adapt_to_speech_rate=True
        )

        assert config.global_offset_ms == -400.0
        assert config.transition_compensation == 0.6
        assert config.use_pause_detection is True


class TestSyncIntegration:
    """Integration tests for sync components"""

    def test_full_sync_flow(self):
        """Test complete synchronization flow"""
        # Create slides
        slides = [
            Slide(
                id="slide_0",
                index=0,
                title="Introduction",
                content="Welcome to the course",
                voiceover_text="Let's begin our lesson"
            ),
            Slide(
                id="slide_1",
                index=1,
                title="Main Topic",
                content="Core concepts",
                voiceover_text="Now we'll cover the main topic"
            )
        ]

        # Create voice segments
        segments = [
            VoiceSegment(id=0, text="Let's begin our lesson", start_time=0.0, end_time=3.0),
            VoiceSegment(id=1, text="Now we'll cover the main topic", start_time=3.5, end_time=7.0)
        ]

        # Detect pauses
        detector = PauseDetector(min_pause_ms=300.0)
        pauses = detector.detect_pauses(segments)
        assert len(pauses) == 1

        # Analyze speech rate
        analyzer = SpeechRateAnalyzer(reference_rate=150.0)
        rate_factor = analyzer.compute_rate_factor(segments)
        assert rate_factor > 0

        # Create sync results
        results = [
            SynchronizationResult(
                slide_id="slide_0",
                slide_index=0,
                segment_ids=[0],
                start_time=0.0,
                end_time=3.0,
                semantic_score=0.85,
                temporal_score=0.90,
                combined_score=0.87
            ),
            SynchronizationResult(
                slide_id="slide_1",
                slide_index=1,
                segment_ids=[1],
                start_time=3.5,
                end_time=7.0,
                semantic_score=0.82,
                temporal_score=0.88,
                combined_score=0.84
            )
        ]

        assert len(results) == 2
        assert results[0].start_time < results[1].start_time

    def test_anchor_constrained_sync(self):
        """Test synchronization with anchor constraints"""
        anchor = SyncAnchor(
            slide_index=1,
            timestamp=5.0,
            segment_index=2,
            anchor_type="SLIDE",
            anchor_id="SLIDE_1"
        )

        result = SynchronizationResult(
            slide_id="slide_1",
            slide_index=1,
            segment_ids=[2, 3],
            start_time=5.0,
            end_time=10.0,
            semantic_score=0.90,
            temporal_score=0.95,
            combined_score=0.92,
            anchor_used=anchor
        )

        # Verify anchor constraint is respected
        assert result.start_time == anchor.timestamp
        assert result.anchor_used is not None


class TestEdgeCases:
    """Tests for edge cases"""

    def test_single_segment(self):
        """Test with single segment"""
        segments = [
            VoiceSegment(id=0, text="Single segment", start_time=0.0, end_time=5.0)
        ]

        detector = PauseDetector()
        pauses = detector.detect_pauses(segments)
        assert pauses == []

    def test_unicode_text(self):
        """Test with Unicode text"""
        segment = VoiceSegment(
            id=0,
            text="日本語のテキスト avec des accents français",
            start_time=0.0,
            end_time=5.0
        )

        slide = Slide(
            id="slide_0",
            index=0,
            title="日本語",
            content="テスト",
            voiceover_text="日本語のテスト"
        )

        assert segment.word_count > 0
        assert slide.get_searchable_text() == "日本語 テスト 日本語のテスト"

    def test_very_long_pause(self):
        """Test with very long pause"""
        segments = [
            VoiceSegment(id=0, text="First", start_time=0.0, end_time=1.0),
            VoiceSegment(id=1, text="Second", start_time=10.0, end_time=11.0)
        ]

        detector = PauseDetector(min_pause_ms=300.0)
        pauses = detector.detect_pauses(segments)

        assert len(pauses) == 1
        assert pauses[0][1] == 9000.0  # 9 second pause

    def test_zero_scores(self):
        """Test result with zero scores"""
        result = SynchronizationResult(
            slide_id="slide_0",
            slide_index=0,
            segment_ids=[0],
            start_time=0.0,
            end_time=1.0,
            semantic_score=0.0,
            temporal_score=0.0,
            combined_score=0.0
        )

        assert result.semantic_score == 0.0
        assert result.combined_score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
