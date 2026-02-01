"""
Hybrid Synchronizer - Combines Direct Sync + SSVS-D for Diagrams

This module implements a hybrid approach:
1. Direct Sync provides PERFECT timing (TTS per slide)
2. SSVS-D adds focus animations for diagram slides ONLY

The SSVS-D post-processing can be easily disabled via:
- Environment variable: ENABLE_DIAGRAM_FOCUS=false
- Or programmatically: HybridSynchronizer(enable_diagram_focus=False)

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID SYNC FLOW                         │
├─────────────────────────────────────────────────────────────┤
│  1. Direct Sync generates timings (PERFECT by construction) │
│  2. For DIAGRAM slides only:                                │
│     - SSVS-D analyzes the narration                         │
│     - Detects element mentions                              │
│     - Generates focus_sequence (highlight, zoom, pointer)   │
│  3. FFmpeg can apply animations via drawbox/overlay         │
└─────────────────────────────────────────────────────────────┘

USAGE:
    hybrid_sync = HybridSynchronizer()

    # Process slides after direct sync timeline is built
    focus_animations = await hybrid_sync.process_diagram_slides(
        slides=job.script.slides,
        slide_audios=slide_audio_batch.slide_audios,
        diagram_metadata=diagram_info  # Optional: element positions from diagram generator
    )

    # Apply animations during video composition
    for slide_id, focus_seq in focus_animations.items():
        # Add drawbox filters or overlays
"""

import os
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .diagram_synchronizer import (
    DiagramAwareSynchronizer,
    DiagramSyncResult,
    DiagramFocusPoint,
    Diagram,
    DiagramElement,
    DiagramElementType,
    BoundingBox,
    FocusAnimationGenerator,
)
from .code_synchronizer import (
    CodeAwareSynchronizer,
    CodeSyncResult,
    CodeRevealPoint,
    CodeRevealAnimationGenerator,
)
from .ssvs_algorithm import VoiceSegment


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class HybridSyncMode(str, Enum):
    """Modes for hybrid synchronization"""
    DISABLED = "disabled"           # No SSVS-D, just direct sync
    DIAGRAMS_ONLY = "diagrams_only" # SSVS-D for diagrams only (default)
    ALL_SLIDES = "all_slides"       # SSVS-D for all slides (experimental)


@dataclass
class HybridSyncConfig:
    """
    Configuration for hybrid synchronization.

    Easy to disable via environment variable or programmatically.
    """
    # Master switch - can disable all SSVS-D processing
    enable_diagram_focus: bool = True

    # Mode selection
    mode: HybridSyncMode = HybridSyncMode.DIAGRAMS_ONLY

    # SSVS-D parameters for diagrams
    min_focus_duration_sec: float = 1.5
    transition_duration_sec: float = 0.3
    similarity_threshold: float = 0.4

    # Focus animation settings
    default_focus_type: str = "highlight"  # highlight, zoom, pointer, outline
    focus_intensity: float = 0.8

    # Logging
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "HybridSyncConfig":
        """Create config from environment variables"""
        return cls(
            enable_diagram_focus=os.getenv("ENABLE_DIAGRAM_FOCUS", "true").lower() == "true",
            mode=HybridSyncMode(os.getenv("HYBRID_SYNC_MODE", "diagrams_only")),
            min_focus_duration_sec=float(os.getenv("DIAGRAM_MIN_FOCUS_SEC", "1.5")),
            similarity_threshold=float(os.getenv("DIAGRAM_SIMILARITY_THRESHOLD", "0.4")),
            verbose=os.getenv("HYBRID_SYNC_VERBOSE", "false").lower() == "true",
        )


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class SlideAudioInfo:
    """Information about a slide's audio for sync processing"""
    slide_id: str
    slide_index: int
    slide_type: str
    start_time: float
    end_time: float
    voiceover_text: str
    audio_path: Optional[str] = None


@dataclass
class DiagramFocusResult:
    """Result of SSVS-D processing for a diagram slide"""
    slide_id: str
    slide_index: int
    focus_sequence: List[DiagramFocusPoint]
    element_mentions: Dict[str, List[Tuple[float, float]]]
    semantic_score: float
    coverage_score: float

    # FFmpeg filter string (ready to use)
    ffmpeg_filter: Optional[str] = None

    # JSON timeline for custom renderers
    animation_timeline: Optional[Dict] = None


@dataclass
class HybridSyncResult:
    """Complete result of hybrid synchronization"""
    # All slides with their timing (from direct sync)
    slide_timings: List[SlideAudioInfo]

    # Diagram-specific focus animations (from SSVS-D)
    diagram_focus: Dict[str, DiagramFocusResult] = field(default_factory=dict)

    # Processing metadata
    diagrams_processed: int = 0
    ssvs_d_enabled: bool = True
    processing_time_ms: float = 0.0


# ==============================================================================
# DIAGRAM METADATA EXTRACTOR
# ==============================================================================

class DiagramMetadataExtractor:
    """
    Extracts diagram structure from various sources.

    Can work with:
    - Mermaid diagram code (parse nodes/edges)
    - Python Diagrams code (parse clusters/nodes)
    - Pre-computed metadata from diagram generator
    """

    def extract_from_slide(
        self,
        slide: Any,
        precomputed_metadata: Optional[Dict] = None
    ) -> Optional[Diagram]:
        """
        Extract diagram structure from a slide.

        Args:
            slide: The slide object (must have content, diagram_type)
            precomputed_metadata: Optional pre-computed element positions

        Returns:
            Diagram object or None if extraction fails
        """
        if precomputed_metadata:
            return self._from_precomputed(slide, precomputed_metadata)

        # Try to infer from slide content
        content = getattr(slide, 'content', '') or ''
        diagram_type = getattr(slide, 'diagram_type', 'flowchart') or 'flowchart'

        # Extract keywords from title and content
        title = getattr(slide, 'title', '') or ''
        voiceover = getattr(slide, 'voiceover_text', '') or ''

        # Create a basic diagram with inferred elements
        elements = self._infer_elements_from_text(content, voiceover)

        if not elements:
            return None

        return Diagram(
            id=getattr(slide, 'id', 'unknown'),
            title=title,
            elements=elements,
            diagram_type=diagram_type
        )

    def _from_precomputed(
        self,
        slide: Any,
        metadata: Dict
    ) -> Optional[Diagram]:
        """Create Diagram from precomputed metadata"""
        elements = []

        for elem_data in metadata.get('elements', []):
            bbox = elem_data.get('bbox', {})
            element = DiagramElement(
                id=elem_data.get('id', f"elem_{len(elements)}"),
                element_type=DiagramElementType(elem_data.get('type', 'node')),
                label=elem_data.get('label', ''),
                bbox=BoundingBox(
                    x=bbox.get('x', 0),
                    y=bbox.get('y', 0),
                    width=bbox.get('width', 0.1),
                    height=bbox.get('height', 0.1)
                ),
                keywords=elem_data.get('keywords', []),
                connected_to=elem_data.get('connected_to', []),
                importance=elem_data.get('importance', 1.0)
            )
            elements.append(element)

        if not elements:
            return None

        return Diagram(
            id=getattr(slide, 'id', metadata.get('id', 'unknown')),
            title=getattr(slide, 'title', metadata.get('title', '')),
            elements=elements,
            diagram_type=metadata.get('diagram_type', 'flowchart'),
            reading_order=metadata.get('reading_order')
        )

    def _infer_elements_from_text(
        self,
        content: str,
        voiceover: str
    ) -> List[DiagramElement]:
        """
        Infer diagram elements from text content.

        This is a heuristic approach - works better with precomputed metadata.
        """
        import re

        elements = []
        combined_text = f"{content} {voiceover}"

        # Look for capitalized terms (likely component names)
        # Pattern: Words with 2+ caps, or quoted terms
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # PascalCase
            r'\b([A-Z]{2,}[a-z]*)\b',                   # ACRONYMS
            r'"([^"]+)"',                               # "Quoted terms"
            r'`([^`]+)`',                               # `Code terms`
        ]

        found_terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, combined_text)
            for match in matches:
                term = match.strip()
                if len(term) > 2 and term.lower() not in ['the', 'and', 'for', 'this']:
                    found_terms.add(term)

        # Create elements with estimated positions (grid layout)
        n_terms = len(found_terms)
        if n_terms == 0:
            return []

        cols = min(3, n_terms)
        for i, term in enumerate(list(found_terms)[:12]):  # Max 12 elements
            row = i // cols
            col = i % cols

            element = DiagramElement(
                id=f"elem_{i}",
                element_type=DiagramElementType.NODE,
                label=term,
                bbox=BoundingBox(
                    x=0.1 + (col * 0.3),
                    y=0.1 + (row * 0.25),
                    width=0.2,
                    height=0.15
                ),
                keywords=[term.lower()],
                importance=1.0 - (i * 0.05)  # First elements more important
            )
            elements.append(element)

        return elements


# ==============================================================================
# HYBRID SYNCHRONIZER - MAIN CLASS
# ==============================================================================

class HybridSynchronizer:
    """
    Hybrid synchronizer that combines Direct Sync timing with SSVS-D focus animations.

    IMPORTANT: This does NOT replace Direct Sync for timing.
    It only ADDS focus animations for diagram slides.

    Easy to disable:
        - Set ENABLE_DIAGRAM_FOCUS=false environment variable
        - Or pass enable_diagram_focus=False to constructor
    """

    # Slide types that should get SSVS-D processing
    DIAGRAM_SLIDE_TYPES = {'diagram', 'architecture', 'flowchart', 'process'}

    # Slide types that should get SSVS-C processing (code reveal)
    CODE_SLIDE_TYPES = {'code', 'code_demo', 'terminal'}

    def __init__(
        self,
        config: Optional[HybridSyncConfig] = None,
        enable_diagram_focus: Optional[bool] = None
    ):
        """
        Initialize hybrid synchronizer.

        Args:
            config: Full configuration (optional)
            enable_diagram_focus: Quick toggle to disable (overrides config)
        """
        self.config = config or HybridSyncConfig.from_env()

        # Allow quick disable via constructor param
        if enable_diagram_focus is not None:
            self.config.enable_diagram_focus = enable_diagram_focus

        # SSVS-D synchronizer (lazy init)
        self._diagram_sync: Optional[DiagramAwareSynchronizer] = None
        self._metadata_extractor = DiagramMetadataExtractor()

        # SSVS-C synchronizer (lazy init)
        self._code_sync: Optional[CodeAwareSynchronizer] = None

        # Log configuration
        status = "ENABLED" if self.config.enable_diagram_focus else "DISABLED"
        print(f"[HYBRID_SYNC] Diagram focus animations: {status}", flush=True)
        if self.config.enable_diagram_focus:
            print(f"[HYBRID_SYNC] Mode: {self.config.mode.value}", flush=True)

    @property
    def diagram_sync(self) -> DiagramAwareSynchronizer:
        """Lazy init SSVS-D synchronizer"""
        if self._diagram_sync is None:
            self._diagram_sync = DiagramAwareSynchronizer()
            self._diagram_sync.min_focus_duration = self.config.min_focus_duration_sec
            self._diagram_sync.transition_duration = self.config.transition_duration_sec
        return self._diagram_sync

    def is_diagram_slide(self, slide: Any) -> bool:
        """Check if a slide should get SSVS-D processing"""
        slide_type = getattr(slide, 'type', None)
        if slide_type is None:
            return False

        # Handle enum or string
        type_str = slide_type.value if hasattr(slide_type, 'value') else str(slide_type)
        return type_str.lower() in self.DIAGRAM_SLIDE_TYPES

    @property
    def code_sync(self) -> CodeAwareSynchronizer:
        """Lazy init SSVS-C synchronizer"""
        if self._code_sync is None:
            self._code_sync = CodeAwareSynchronizer()
        return self._code_sync

    def is_code_slide(self, slide: Any) -> bool:
        """Check if a slide should get SSVS-C processing"""
        slide_type = getattr(slide, 'type', None)
        if slide_type is None:
            return False

        # Handle enum or string
        type_str = slide_type.value if hasattr(slide_type, 'value') else str(slide_type)
        return type_str.lower() in self.CODE_SLIDE_TYPES

    async def process_diagram_slides(
        self,
        slides: List[Any],
        slide_audios: List[Any],
        diagram_metadata: Optional[Dict[str, Dict]] = None,
        video_width: int = 1920,
        video_height: int = 1080
    ) -> HybridSyncResult:
        """
        Process diagram slides with SSVS-D to generate focus animations.

        This is called AFTER direct sync has established perfect timing.
        It only ADDS focus information for diagram slides.

        Args:
            slides: List of Slide objects from the presentation script
            slide_audios: List of SlideAudio objects from direct sync
            diagram_metadata: Optional pre-computed diagram element positions
                              Format: {slide_id: {elements: [...], ...}}
            video_width: Video width for FFmpeg filter generation
            video_height: Video height for FFmpeg filter generation

        Returns:
            HybridSyncResult with focus animations for diagram slides
        """
        import time
        start_time = time.time()

        # Build slide timing info
        slide_timings = []
        current_time = 0.0

        for i, (slide, audio) in enumerate(zip(slides, slide_audios)):
            duration = getattr(audio, 'duration', 0) or getattr(slide, 'duration', 10.0)

            slide_info = SlideAudioInfo(
                slide_id=getattr(slide, 'id', f"slide_{i}"),
                slide_index=i,
                slide_type=str(getattr(slide, 'type', 'content')),
                start_time=current_time,
                end_time=current_time + duration,
                voiceover_text=getattr(slide, 'voiceover_text', ''),
                audio_path=getattr(audio, 'audio_path', None)
            )
            slide_timings.append(slide_info)
            current_time += duration

        # If SSVS-D is disabled, return just the timings
        if not self.config.enable_diagram_focus:
            return HybridSyncResult(
                slide_timings=slide_timings,
                diagram_focus={},
                diagrams_processed=0,
                ssvs_d_enabled=False,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Process diagram slides with SSVS-D
        diagram_focus = {}
        diagrams_processed = 0

        for slide_info in slide_timings:
            slide = slides[slide_info.slide_index]

            # Check if this is a diagram slide
            if not self.is_diagram_slide(slide):
                continue

            if self.config.verbose:
                print(f"[HYBRID_SYNC] Processing diagram slide: {slide_info.slide_id}", flush=True)

            # Get diagram metadata if available
            metadata = None
            if diagram_metadata:
                metadata = diagram_metadata.get(slide_info.slide_id)

            # Extract diagram structure
            diagram = self._metadata_extractor.extract_from_slide(slide, metadata)

            if diagram is None:
                if self.config.verbose:
                    print(f"[HYBRID_SYNC] Could not extract diagram structure for {slide_info.slide_id}", flush=True)
                continue

            # Create voice segment for SSVS-D
            segment = VoiceSegment(
                id=slide_info.slide_index,
                text=slide_info.voiceover_text,
                start_time=slide_info.start_time,
                end_time=slide_info.end_time
            )

            # Run SSVS-D synchronization
            try:
                sync_result = self.diagram_sync.synchronize(diagram, [segment])

                # Generate FFmpeg filter
                animation_gen = FocusAnimationGenerator(diagram)
                ffmpeg_filter = animation_gen.generate_ffmpeg_drawbox_filter(
                    sync_result, video_width, video_height
                )

                # Generate JSON timeline
                animation_timeline = animation_gen.generate_json_timeline(sync_result)

                # Store result
                focus_result = DiagramFocusResult(
                    slide_id=slide_info.slide_id,
                    slide_index=slide_info.slide_index,
                    focus_sequence=sync_result.focus_sequence,
                    element_mentions=sync_result.element_mentions,
                    semantic_score=sync_result.semantic_score,
                    coverage_score=sync_result.coverage_score,
                    ffmpeg_filter=ffmpeg_filter if ffmpeg_filter else None,
                    animation_timeline=animation_timeline
                )

                diagram_focus[slide_info.slide_id] = focus_result
                diagrams_processed += 1

                if self.config.verbose:
                    print(f"[HYBRID_SYNC] Diagram {slide_info.slide_id}: "
                          f"{len(sync_result.focus_sequence)} focus points, "
                          f"semantic={sync_result.semantic_score:.2f}", flush=True)

            except Exception as e:
                print(f"[HYBRID_SYNC] Error processing diagram {slide_info.slide_id}: {e}", flush=True)
                continue

        processing_time = (time.time() - start_time) * 1000

        print(f"[HYBRID_SYNC] Processed {diagrams_processed} diagram slides in {processing_time:.1f}ms", flush=True)

        return HybridSyncResult(
            slide_timings=slide_timings,
            diagram_focus=diagram_focus,
            diagrams_processed=diagrams_processed,
            ssvs_d_enabled=True,
            processing_time_ms=processing_time
        )

    def get_ffmpeg_filters_for_slide(
        self,
        slide_id: str,
        hybrid_result: HybridSyncResult
    ) -> Optional[str]:
        """
        Get FFmpeg filter string for a specific slide.

        Returns None if the slide has no focus animations.

        Usage:
            filters = hybrid_sync.get_ffmpeg_filters_for_slide("slide_3", result)
            if filters:
                # Add to FFmpeg command: -vf "{filters}"
        """
        focus = hybrid_result.diagram_focus.get(slide_id)
        if focus and focus.ffmpeg_filter:
            return focus.ffmpeg_filter
        return None

    def get_animation_timeline(
        self,
        slide_id: str,
        hybrid_result: HybridSyncResult
    ) -> Optional[Dict]:
        """
        Get animation timeline JSON for a specific slide.

        Returns None if the slide has no focus animations.

        Usage:
            timeline = hybrid_sync.get_animation_timeline("slide_3", result)
            if timeline:
                # Use for custom rendering
        """
        focus = hybrid_result.diagram_focus.get(slide_id)
        if focus and focus.animation_timeline:
            return focus.animation_timeline
        return None

    async def process_code_slides(
        self,
        slides: List[Any],
        slide_audios: List[Any],
        word_timestamps: Optional[Dict[str, List[Dict]]] = None,
        video_width: int = 1920,
        video_height: int = 1080
    ) -> Dict[str, CodeSyncResult]:
        """
        Process code slides with SSVS-C for line-by-line reveal.

        This is called AFTER direct sync has established perfect timing.
        It generates reveal points synchronized with the voiceover.

        Args:
            slides: List of Slide objects from the presentation script
            slide_audios: List of SlideAudio objects from direct sync
            word_timestamps: Optional dict mapping slide_id -> word timestamps
            video_width: Video width for FFmpeg filter generation
            video_height: Video height for FFmpeg filter generation

        Returns:
            Dict mapping slide_id -> CodeSyncResult with reveal sequence
        """
        import time
        start_time = time.time()

        code_sync_results: Dict[str, CodeSyncResult] = {}
        codes_processed = 0

        # Build slide timing info
        current_time = 0.0

        for i, (slide, audio) in enumerate(zip(slides, slide_audios)):
            # Check if this is a code slide
            if not self.is_code_slide(slide):
                duration = getattr(audio, 'duration', 0) or getattr(slide, 'duration', 10.0)
                current_time += duration
                continue

            slide_id = getattr(slide, 'id', f"slide_{i}")

            # Extract code and language
            code = self._extract_code(slide)
            language = self._extract_language(slide)

            if not code:
                duration = getattr(audio, 'duration', 0) or getattr(slide, 'duration', 10.0)
                current_time += duration
                continue

            duration = getattr(audio, 'duration', 0) or getattr(slide, 'duration', 10.0)
            start = current_time
            end = current_time + duration

            # Get word timestamps for this slide
            wts = word_timestamps.get(slide_id, []) if word_timestamps else []

            # Create voice segment for SSVS-C
            voiceover_text = getattr(slide, 'voiceover_text', '') or ''
            segment = VoiceSegment(
                id=i,
                text=voiceover_text,
                start_time=0,  # Relative to slide start
                end_time=duration
            )

            # Run SSVS-C synchronization
            try:
                sync_result = self.code_sync.synchronize(code, language, [segment])

                # Generate FFmpeg filter
                anim_gen = CodeRevealAnimationGenerator(video_width, video_height)
                sync_result.ffmpeg_filter = anim_gen.generate_drawbox_filter(sync_result)
                sync_result.animation_timeline = anim_gen.generate_json_timeline(sync_result)

                code_sync_results[slide_id] = sync_result
                codes_processed += 1

                if self.config.verbose:
                    print(f"[HYBRID_SYNC] Code {slide_id}: "
                          f"{len(sync_result.reveal_sequence)} reveal points, "
                          f"semantic={sync_result.semantic_score:.2f}", flush=True)

            except Exception as e:
                print(f"[HYBRID_SYNC] Error processing code slide {slide_id}: {e}",
                      flush=True)

            current_time += duration

        processing_time = (time.time() - start_time) * 1000
        print(f"[HYBRID_SYNC] Processed {codes_processed} code slides in {processing_time:.1f}ms", flush=True)

        return code_sync_results

    def _extract_code(self, slide: Any) -> Optional[str]:
        """Extract code content from a slide"""
        # Try different attributes
        code = getattr(slide, 'code', None)
        if code:
            return code

        # Check code_blocks
        code_blocks = getattr(slide, 'code_blocks', [])
        if code_blocks:
            first_block = code_blocks[0]
            return getattr(first_block, 'code', None) or first_block.get('code') if isinstance(first_block, dict) else None

        return None

    def _extract_language(self, slide: Any) -> str:
        """Extract programming language from a slide"""
        language = getattr(slide, 'language', None)
        if language:
            return language

        # Check code_blocks
        code_blocks = getattr(slide, 'code_blocks', [])
        if code_blocks:
            first_block = code_blocks[0]
            lang = getattr(first_block, 'language', None) or (first_block.get('language') if isinstance(first_block, dict) else None)
            if lang:
                return lang

        return "python"  # Default

    def get_code_sync_result(
        self,
        slide_id: str,
        code_sync_results: Dict[str, CodeSyncResult]
    ) -> Optional[CodeSyncResult]:
        """Get code sync result for a specific slide"""
        return code_sync_results.get(slide_id)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_hybrid_synchronizer(
    enable: bool = True,
    verbose: bool = False
) -> HybridSynchronizer:
    """
    Create a hybrid synchronizer with simple options.

    Args:
        enable: Enable SSVS-D for diagrams (default True)
        verbose: Enable verbose logging

    Returns:
        Configured HybridSynchronizer instance
    """
    config = HybridSyncConfig(
        enable_diagram_focus=enable,
        verbose=verbose
    )
    return HybridSynchronizer(config=config)


async def process_presentation_diagrams(
    slides: List[Any],
    slide_audios: List[Any],
    diagram_metadata: Optional[Dict] = None,
    enable_focus: bool = True
) -> HybridSyncResult:
    """
    Convenience function to process diagram focus for a presentation.

    Args:
        slides: Presentation slides
        slide_audios: Audio info from direct sync
        diagram_metadata: Optional pre-computed diagram metadata
        enable_focus: Enable/disable SSVS-D processing

    Returns:
        HybridSyncResult with focus animations
    """
    sync = create_hybrid_synchronizer(enable=enable_focus)
    return await sync.process_diagram_slides(
        slides=slides,
        slide_audios=slide_audios,
        diagram_metadata=diagram_metadata
    )
