"""
SSVS-D - Diagram-Aware Synchronization Extension

Extends SSVS to handle diagram synchronization with:
- Element mention detection
- Spatial reference resolution
- Focus animation generation

APPROACH:
1. Extract diagram structure (nodes, connections, positions)
2. Detect element mentions in narration
3. Resolve spatial references ("en haut", "à gauche")
4. Generate focus animation sequence
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set, Any
from enum import Enum
import re

from .ssvs_algorithm import SemanticEmbeddingEngine, VoiceSegment


# ==============================================================================
# DATA STRUCTURES FOR DIAGRAMS
# ==============================================================================

class DiagramElementType(Enum):
    """Types of diagram elements"""
    NODE = "node"
    CONNECTOR = "connector"
    LABEL = "label"
    GROUP = "group"
    ICON = "icon"


@dataclass
class BoundingBox:
    """Bounding box for spatial positioning (normalized 0-1)"""
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def distance_to(self, other: 'BoundingBox') -> float:
        """Euclidean distance between centers"""
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


@dataclass
class DiagramElement:
    """Individual diagram element"""
    id: str
    element_type: DiagramElementType
    label: str
    bbox: BoundingBox
    keywords: List[str] = field(default_factory=list)
    connected_to: List[str] = field(default_factory=list)
    importance: float = 1.0

    def get_searchable_text(self) -> str:
        """Text for semantic embedding"""
        parts = [self.label]
        parts.extend(self.keywords)
        return " ".join(filter(None, parts))


@dataclass
class Diagram:
    """Complete diagram representation"""
    id: str
    title: str
    elements: List[DiagramElement]
    diagram_type: str = "flowchart"
    reading_order: Optional[List[str]] = None

    def get_element_by_id(self, element_id: str) -> Optional[DiagramElement]:
        for elem in self.elements:
            if elem.id == element_id:
                return elem
        return None

    def get_nodes(self) -> List[DiagramElement]:
        """Return only nodes (not connectors)"""
        return [e for e in self.elements if e.element_type in
                (DiagramElementType.NODE, DiagramElementType.GROUP)]

    def infer_reading_order(self) -> List[str]:
        """Infer logical reading order based on position and connections"""
        nodes = self.get_nodes()
        if not nodes:
            return []

        def position_score(elem: DiagramElement) -> float:
            cx, cy = elem.bbox.center
            return cy * 2 + cx  # Top-to-bottom priority

        sorted_nodes = sorted(nodes, key=lambda e: (position_score(e), -e.importance))
        return [n.id for n in sorted_nodes]


@dataclass
class DiagramFocusPoint:
    """Focus point for visual animation"""
    element_id: str
    start_time: float
    end_time: float
    focus_type: str  # "highlight", "zoom", "pointer", "outline"
    intensity: float = 1.0

    def to_animation_keyframe(self) -> Dict:
        return {
            "element": self.element_id,
            "start": self.start_time,
            "end": self.end_time,
            "effect": self.focus_type,
            "intensity": self.intensity
        }


@dataclass
class DiagramSyncResult:
    """Synchronization result for a diagram"""
    diagram_id: str
    voice_segment_ids: List[int]
    start_time: float
    end_time: float
    focus_sequence: List[DiagramFocusPoint]
    element_mentions: Dict[str, List[Tuple[float, float]]]
    semantic_score: float
    coverage_score: float


# ==============================================================================
# MENTION DETECTOR
# ==============================================================================

class DiagramMentionDetector:
    """
    Detects when narrator mentions specific diagram elements.

    Uses:
    1. Exact/fuzzy label matching
    2. Semantic embedding similarity
    3. Spatial reference detection
    4. Flow reference detection
    """

    SPATIAL_MARKERS = {
        "top": ["en haut", "au sommet", "au-dessus", "top", "upper", "above"],
        "bottom": ["en bas", "au fond", "en dessous", "bottom", "lower", "below"],
        "left": ["à gauche", "sur la gauche", "left", "côté gauche"],
        "right": ["à droite", "sur la droite", "right", "côté droit"],
        "center": ["au centre", "au milieu", "central", "center", "middle"],
        "first": ["premier", "première", "first", "initial", "début", "commence"],
        "last": ["dernier", "dernière", "last", "final", "fin", "termine"],
    }

    FLOW_MARKERS = {
        "next": ["ensuite", "puis", "après", "next", "then", "suivant"],
        "leads_to": ["mène à", "conduit à", "leads to", "goes to", "vers"],
        "from": ["depuis", "à partir de", "from", "venant de"],
        "between": ["entre", "between", "connecte", "relie"],
    }

    def __init__(self, embedding_engine: SemanticEmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.similarity_threshold = 0.4

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return ' '.join(text.split())

    def _fuzzy_match(self, text: str, target: str, threshold: float = 0.7) -> bool:
        text_words = set(self._normalize_text(text).split())
        target_words = set(self._normalize_text(target).split())

        if not target_words:
            return False

        intersection = len(text_words & target_words)
        union = len(text_words | target_words)

        return (intersection / union) >= threshold if union > 0 else False

    def detect_element_mentions(self,
                                 segment: VoiceSegment,
                                 diagram: Diagram) -> List[Tuple[str, float]]:
        """
        Detect which diagram elements are mentioned in this segment.

        Returns:
            List of (element_id, confidence_score)
        """
        mentions = []
        segment_text = self._normalize_text(segment.text)
        segment_embedding = self.embedding_engine.embed(segment.text)

        for element in diagram.elements:
            confidence = 0.0

            # 1. Exact label match
            element_label_norm = self._normalize_text(element.label)
            if element_label_norm and element_label_norm in segment_text:
                confidence = max(confidence, 0.95)

            # 2. Fuzzy match
            elif element_label_norm and self._fuzzy_match(segment_text, element.label, 0.6):
                confidence = max(confidence, 0.7)

            # 3. Keyword match
            for keyword in element.keywords:
                if self._normalize_text(keyword) in segment_text:
                    confidence = max(confidence, 0.8)
                    break

            # 4. Semantic similarity
            if confidence < 0.5:
                element_embedding = self.embedding_engine.embed(element.get_searchable_text())
                similarity = self.embedding_engine.similarity(segment_embedding, element_embedding)
                if similarity > self.similarity_threshold:
                    confidence = max(confidence, similarity * 0.8)

            if confidence > 0.3:
                mentions.append((element.id, confidence))

        mentions.sort(key=lambda x: -x[1])
        return mentions

    def detect_spatial_reference(self, segment: VoiceSegment) -> Optional[str]:
        text_lower = segment.text.lower()

        for position, markers in self.SPATIAL_MARKERS.items():
            for marker in markers:
                if marker in text_lower:
                    return position

        return None

    def detect_flow_reference(self, segment: VoiceSegment) -> Optional[str]:
        text_lower = segment.text.lower()

        for flow_type, markers in self.FLOW_MARKERS.items():
            for marker in markers:
                if marker in text_lower:
                    return flow_type

        return None


# ==============================================================================
# SPATIAL RESOLVER
# ==============================================================================

class SpatialResolver:
    """Resolves spatial references to concrete diagram elements"""

    def __init__(self, diagram: Diagram):
        self.diagram = diagram
        self.nodes = diagram.get_nodes()

    def resolve_spatial_reference(self,
                                   position: str,
                                   current_focus: Optional[str] = None) -> Optional[str]:
        if not self.nodes:
            return None

        candidates = self.nodes.copy()

        if position == "top":
            candidates = sorted(candidates, key=lambda e: e.bbox.y)[:3]
        elif position == "bottom":
            candidates = sorted(candidates, key=lambda e: -e.bbox.y)[:3]
        elif position == "left":
            candidates = sorted(candidates, key=lambda e: e.bbox.x)[:3]
        elif position == "right":
            candidates = sorted(candidates, key=lambda e: -e.bbox.x)[:3]
        elif position == "center":
            candidates = sorted(candidates,
                              key=lambda e: abs(e.bbox.center[0] - 0.5) + abs(e.bbox.center[1] - 0.5))[:3]
        elif position == "first":
            reading_order = self.diagram.reading_order or self.diagram.infer_reading_order()
            if reading_order:
                return reading_order[0]
        elif position == "last":
            reading_order = self.diagram.reading_order or self.diagram.infer_reading_order()
            if reading_order:
                return reading_order[-1]

        if candidates:
            candidates.sort(key=lambda e: -e.importance)
            return candidates[0].id

        return None

    def resolve_flow_reference(self,
                                flow_type: str,
                                current_focus: str) -> Optional[str]:
        current_elem = self.diagram.get_element_by_id(current_focus)
        if not current_elem:
            return None

        if flow_type == "next":
            if current_elem.connected_to:
                return current_elem.connected_to[0]

            reading_order = self.diagram.reading_order or self.diagram.infer_reading_order()
            if current_focus in reading_order:
                idx = reading_order.index(current_focus)
                if idx + 1 < len(reading_order):
                    return reading_order[idx + 1]

        elif flow_type == "from":
            for elem in self.nodes:
                if current_focus in elem.connected_to:
                    return elem.id

        return None


# ==============================================================================
# DIAGRAM-AWARE SYNCHRONIZER
# ==============================================================================

class DiagramAwareSynchronizer:
    """
    Extension of SSVS for diagram synchronization.

    Algorithm:
    1. MAPPING: Associate each segment with mentioned elements
    2. SEQUENCING: Order focus according to diagram logic
    3. TIMING: Calculate optimal focus durations
    """

    def __init__(self):
        self.embedding_engine = SemanticEmbeddingEngine()
        self.mention_detector: Optional[DiagramMentionDetector] = None
        self.min_focus_duration = 1.5
        self.transition_duration = 0.3

    def _initialize(self, diagram: Diagram, segments: List[VoiceSegment]):
        documents = [diagram.title]
        for elem in diagram.elements:
            documents.append(elem.get_searchable_text())
        for seg in segments:
            documents.append(seg.text)

        self.embedding_engine.build_vocabulary(documents)
        self.mention_detector = DiagramMentionDetector(self.embedding_engine)

    def synchronize(self,
                    diagram: Diagram,
                    segments: List[VoiceSegment]) -> DiagramSyncResult:
        """
        Synchronize narration segments with diagram.

        Returns:
            DiagramSyncResult with optimal focus sequence
        """
        if not segments:
            return DiagramSyncResult(
                diagram_id=diagram.id,
                voice_segment_ids=[],
                start_time=0.0,
                end_time=0.0,
                focus_sequence=[],
                element_mentions={},
                semantic_score=0.0,
                coverage_score=0.0
            )

        print(f"[SSVS-D] Synchronizing diagram '{diagram.title}' with {len(segments)} segments", flush=True)

        # Initialize
        self._initialize(diagram, segments)
        spatial_resolver = SpatialResolver(diagram)

        # Phase 1: MAPPING
        segment_mentions: Dict[int, List[Tuple[str, float]]] = {}
        element_mentions: Dict[str, List[Tuple[float, float]]] = {
            elem.id: [] for elem in diagram.elements
        }

        for segment in segments:
            mentions = self.mention_detector.detect_element_mentions(segment, diagram)
            segment_mentions[segment.id] = mentions

            for elem_id, confidence in mentions:
                if confidence > 0.5:
                    element_mentions[elem_id].append((segment.start_time, segment.end_time))

        # Phase 2: SEQUENCING
        focus_sequence: List[DiagramFocusPoint] = []
        current_focus: Optional[str] = None
        reading_order = diagram.reading_order or diagram.infer_reading_order()

        for segment in segments:
            mentions = segment_mentions.get(segment.id, [])

            # Check spatial references
            spatial_ref = self.mention_detector.detect_spatial_reference(segment)
            if spatial_ref:
                resolved = spatial_resolver.resolve_spatial_reference(spatial_ref, current_focus)
                if resolved:
                    mentions.insert(0, (resolved, 0.85))

            # Check flow references
            if current_focus:
                flow_ref = self.mention_detector.detect_flow_reference(segment)
                if flow_ref:
                    resolved = spatial_resolver.resolve_flow_reference(flow_ref, current_focus)
                    if resolved:
                        mentions.insert(0, (resolved, 0.8))

            # Select primary focus
            if mentions:
                best_elem_id, confidence = mentions[0]

                if best_elem_id != current_focus or confidence > 0.8:
                    focus_point = DiagramFocusPoint(
                        element_id=best_elem_id,
                        start_time=segment.start_time,
                        end_time=segment.end_time,
                        focus_type="highlight" if confidence > 0.7 else "pointer",
                        intensity=min(1.0, confidence)
                    )
                    focus_sequence.append(focus_point)
                    current_focus = best_elem_id
                else:
                    if focus_sequence:
                        focus_sequence[-1].end_time = segment.end_time

            elif current_focus is None and reading_order:
                focus_point = DiagramFocusPoint(
                    element_id=reading_order[0],
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    focus_type="outline",
                    intensity=0.5
                )
                focus_sequence.append(focus_point)
                current_focus = reading_order[0]

        # Phase 3: TIMING optimization
        focus_sequence = self._optimize_focus_timing(focus_sequence)

        # Compute scores
        total_elements = len(diagram.get_nodes())
        mentioned_elements = sum(1 for mentions in element_mentions.values() if mentions)
        coverage_score = mentioned_elements / total_elements if total_elements > 0 else 0.0

        semantic_scores = []
        for segment in segments:
            for elem_id, confidence in segment_mentions.get(segment.id, []):
                semantic_scores.append(confidence)
        semantic_score = np.mean(semantic_scores) if semantic_scores else 0.0

        print(f"[SSVS-D] Sync complete. Semantic: {semantic_score:.3f}, Coverage: {coverage_score:.1%}", flush=True)

        return DiagramSyncResult(
            diagram_id=diagram.id,
            voice_segment_ids=[s.id for s in segments],
            start_time=segments[0].start_time,
            end_time=segments[-1].end_time,
            focus_sequence=focus_sequence,
            element_mentions=element_mentions,
            semantic_score=semantic_score,
            coverage_score=coverage_score
        )

    def _optimize_focus_timing(self,
                                focus_sequence: List[DiagramFocusPoint]) -> List[DiagramFocusPoint]:
        if not focus_sequence:
            return focus_sequence

        optimized = []

        for i, focus in enumerate(focus_sequence):
            duration = focus.end_time - focus.start_time

            if duration < self.min_focus_duration:
                if i + 1 < len(focus_sequence):
                    gap = focus_sequence[i + 1].start_time - focus.end_time
                    if gap > 0:
                        extension = min(gap, self.min_focus_duration - duration)
                        focus.end_time += extension

            optimized.append(focus)

        # Merge consecutive focus on same element
        merged = []
        for focus in optimized:
            if merged and merged[-1].element_id == focus.element_id:
                merged[-1].end_time = focus.end_time
                merged[-1].intensity = max(merged[-1].intensity, focus.intensity)
            else:
                merged.append(focus)

        return merged


# ==============================================================================
# ANIMATION GENERATOR
# ==============================================================================

class FocusAnimationGenerator:
    """Generates animation instructions for video rendering"""

    def __init__(self, diagram: Diagram):
        self.diagram = diagram

    def generate_json_timeline(self, sync_result: DiagramSyncResult) -> Dict:
        """Generate JSON timeline for rendering"""
        timeline = {
            "diagram_id": sync_result.diagram_id,
            "duration": sync_result.end_time - sync_result.start_time,
            "keyframes": []
        }

        for focus in sync_result.focus_sequence:
            element = self.diagram.get_element_by_id(focus.element_id)
            if not element:
                continue

            keyframe = {
                "time": focus.start_time - sync_result.start_time,
                "duration": focus.end_time - focus.start_time,
                "target": {
                    "id": element.id,
                    "label": element.label,
                    "bbox": {
                        "x": element.bbox.x,
                        "y": element.bbox.y,
                        "width": element.bbox.width,
                        "height": element.bbox.height
                    }
                },
                "effect": focus.focus_type,
                "intensity": focus.intensity,
                "easing": "easeInOutCubic"
            }
            timeline["keyframes"].append(keyframe)

        return timeline

    def generate_ffmpeg_drawbox_filter(self, sync_result: DiagramSyncResult,
                                        video_width: int = 1920,
                                        video_height: int = 1080) -> str:
        """
        Generate FFmpeg filter string for highlighting elements.

        This can be used to overlay highlight boxes on the diagram.
        """
        filters = []

        for i, focus in enumerate(sync_result.focus_sequence):
            element = self.diagram.get_element_by_id(focus.element_id)
            if not element:
                continue

            # Convert normalized bbox to pixel coordinates
            x = int(element.bbox.x * video_width)
            y = int(element.bbox.y * video_height)
            w = int(element.bbox.width * video_width)
            h = int(element.bbox.height * video_height)

            # Color based on focus type
            color = "yellow" if focus.focus_type == "highlight" else "white"
            thickness = 4 if focus.focus_type == "highlight" else 2

            # FFmpeg drawbox with time enable
            start = focus.start_time
            end = focus.end_time
            filter_str = (
                f"drawbox=x={x}:y={y}:w={w}:h={h}:color={color}@0.8:"
                f"t={thickness}:enable='between(t,{start},{end})'"
            )
            filters.append(filter_str)

        return ",".join(filters) if filters else ""
