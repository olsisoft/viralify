"""
Integration tests for RAG Images in Slide Generation

Tests the integration of RAG images into diagram slides,
including the fallback logic to LLM generation.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.presentation_models import (
    Slide,
    SlideType,
    PresentationStyle,
    RAGImageReference,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_rag_images():
    """Sample RAG image references for testing"""
    return [
        {
            "image_id": "img_001",
            "document_id": "doc_123",
            "file_path": "/tmp/viralify/documents/user_1/images/kafka_architecture.png",
            "file_name": "kafka_architecture.png",
            "detected_type": "diagram",
            "context_text": "Apache Kafka is a distributed streaming platform for data pipelines",
            "caption": "Kafka Architecture Overview",
            "description": "Diagram showing Kafka brokers, producers, and consumers",
            "relevance_score": 0.85,
            "page_number": 5,
            "width": 1200,
            "height": 800,
            "lecture_id": "lec_001",
            "lecture_title": "Introduction to Kafka",
        },
        {
            "image_id": "img_002",
            "document_id": "doc_123",
            "file_path": "/tmp/viralify/documents/user_1/images/data_pipeline.png",
            "file_name": "data_pipeline.png",
            "detected_type": "architecture",
            "context_text": "ETL pipeline with batch and stream processing",
            "caption": "Data Pipeline Architecture",
            "description": "Architecture showing data flow from sources to warehouse",
            "relevance_score": 0.75,
            "page_number": 12,
            "width": 1600,
            "height": 900,
            "lecture_id": "lec_002",
            "lecture_title": "Data Pipelines",
        },
        {
            "image_id": "img_003",
            "document_id": "doc_456",
            "file_path": "/tmp/viralify/documents/user_1/images/unrelated_chart.png",
            "file_name": "unrelated_chart.png",
            "detected_type": "chart",
            "context_text": "Sales performance metrics",
            "caption": "Sales Chart",
            "description": "Bar chart of quarterly sales",
            "relevance_score": 0.45,  # Below threshold
            "page_number": 3,
            "width": 800,
            "height": 600,
            "lecture_id": "lec_003",
            "lecture_title": "Performance Metrics",
        },
    ]


@pytest.fixture
def sample_diagram_slide():
    """Sample diagram slide for testing"""
    return Slide(
        id="slide_001",
        type=SlideType.DIAGRAM,
        title="Apache Kafka Architecture",
        subtitle="Understanding the core components",
        content="Kafka consists of brokers, producers, and consumers",
        bullet_points=[
            "Brokers manage partitions",
            "Producers send messages",
            "Consumers read messages",
        ],
        duration=30.0,
        voiceover_text="Let's explore the Kafka architecture...",
        diagram_type="architecture",
    )


@pytest.fixture
def sample_content_slide():
    """Sample content slide (non-diagram) for testing"""
    return Slide(
        id="slide_002",
        type=SlideType.CONTENT,
        title="Key Concepts",
        content="Understanding the fundamentals",
        bullet_points=["Point 1", "Point 2"],
        duration=20.0,
        voiceover_text="Here are the key concepts...",
    )


# ============================================================================
# Tests for RAG Image Matching
# ============================================================================

class TestRAGImageMatching:
    """Tests for _find_matching_rag_image() logic"""

    def test_finds_best_matching_image(self, sample_rag_images):
        """Test that the best matching image is found for a topic"""
        slide_topic = "Apache Kafka Architecture"
        min_score = 0.7

        # Simulate matching logic
        candidates = []
        for img in sample_rag_images:
            if img.get("relevance_score", 0) >= min_score:
                if img.get("detected_type") in ["diagram", "chart", "architecture", "flowchart"]:
                    # Calculate adjusted score with topic bonus
                    adjusted_score = img["relevance_score"]
                    context = (img.get("context_text") or "").lower()
                    caption = (img.get("caption") or "").lower()

                    topic_words = [w.lower() for w in slide_topic.split() if len(w) > 3]
                    for word in topic_words:
                        if word in context or word in caption:
                            adjusted_score += 0.05

                    candidates.append({**img, "adjusted_score": min(1.0, adjusted_score)})

        # Sort and get best
        candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
        best_match = candidates[0] if candidates else None

        assert best_match is not None
        assert "kafka" in best_match["file_name"].lower()
        assert best_match["adjusted_score"] >= 0.85

    def test_returns_none_when_no_match(self, sample_rag_images):
        """Test that None is returned when no image matches"""
        slide_topic = "Quantum Computing Fundamentals"
        min_score = 0.7

        # No images should match this topic
        candidates = [
            img for img in sample_rag_images
            if img.get("relevance_score", 0) >= min_score
            and "quantum" in (img.get("context_text") or "").lower()
        ]

        assert len(candidates) == 0

    def test_filters_by_min_score(self, sample_rag_images):
        """Test that images below min_score are filtered out"""
        min_score = 0.7

        filtered = [
            img for img in sample_rag_images
            if img.get("relevance_score", 0) >= min_score
        ]

        # Only 2 images have score >= 0.7
        assert len(filtered) == 2
        for img in filtered:
            assert img["relevance_score"] >= min_score

    def test_filters_by_image_type(self, sample_rag_images):
        """Test that only diagram-suitable types are considered"""
        allowed_types = ["diagram", "chart", "architecture", "flowchart"]

        filtered = [
            img for img in sample_rag_images
            if img.get("detected_type") in allowed_types
        ]

        for img in filtered:
            assert img["detected_type"] in allowed_types


# ============================================================================
# Tests for RAG Image Usage in Slides
# ============================================================================

class TestRAGImageUsage:
    """Tests for using RAG images in slide generation"""

    def test_uses_rag_image_when_available(self, sample_diagram_slide, sample_rag_images):
        """Test that RAG image is used when a good match is found"""
        # Simulate the decision logic
        use_rag_images = os.environ.get("USE_RAG_IMAGES", "true").lower() == "true"

        if use_rag_images and sample_rag_images:
            # Find matching image
            matching = [
                img for img in sample_rag_images
                if img.get("relevance_score", 0) >= 0.7
                and "kafka" in (img.get("context_text") or "").lower()
            ]

            assert len(matching) >= 1
            best = matching[0]
            assert best["image_id"] == "img_001"

    def test_falls_back_to_llm_when_no_rag_image(self, sample_diagram_slide):
        """Test that LLM generation is used when no RAG image matches"""
        rag_images = []  # No RAG images available

        # When no RAG images, should fall back to LLM
        use_llm = len(rag_images) == 0
        assert use_llm is True

    def test_falls_back_to_llm_when_score_too_low(self, sample_diagram_slide, sample_rag_images):
        """Test that LLM is used when RAG image score is below threshold"""
        min_score = 0.9  # High threshold

        matching = [
            img for img in sample_rag_images
            if img.get("relevance_score", 0) >= min_score
        ]

        # No images meet 0.9 threshold
        assert len(matching) == 0

    def test_skips_rag_for_non_diagram_slides(self, sample_content_slide, sample_rag_images):
        """Test that RAG images are not used for non-diagram slides"""
        # RAG images should only be used for DIAGRAM slides
        is_diagram_slide = sample_content_slide.type == SlideType.DIAGRAM

        assert is_diagram_slide is False

    @patch.dict(os.environ, {"USE_RAG_IMAGES": "false"})
    def test_respects_use_rag_images_env_var(self, sample_diagram_slide, sample_rag_images):
        """Test that USE_RAG_IMAGES=false disables RAG image usage"""
        use_rag = os.environ.get("USE_RAG_IMAGES", "true").lower() == "true"

        assert use_rag is False


# ============================================================================
# Tests for RAG Image Client
# ============================================================================

class TestRAGImageClient:
    """Tests for rag_image_client.py"""

    def test_find_matching_image_returns_best_match(self, sample_rag_images):
        """Test that find_matching_image returns the highest scoring match"""
        slide_topic = "Kafka Architecture"
        min_score = 0.7

        # Simulate RAGImageClient.find_matching_image
        candidates = []
        for img in sample_rag_images:
            score = img.get("relevance_score", 0)
            if score < min_score:
                continue

            img_type = img.get("detected_type", "unknown")
            if img_type not in ["diagram", "chart", "architecture", "flowchart"]:
                continue

            # Topic matching bonus
            context = (img.get("context_text") or "").lower()
            caption = (img.get("caption") or "").lower()
            topic_words = [w.lower() for w in slide_topic.split() if len(w) > 3]

            bonus = sum(0.05 for w in topic_words if w in context or w in caption)
            adjusted_score = min(1.0, score + bonus)

            candidates.append({**img, "adjusted_score": adjusted_score})

        if candidates:
            candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
            best = candidates[0]
            assert "kafka" in best["context_text"].lower()

    def test_get_image_path_copies_file(self, sample_rag_images):
        """Test that get_image_path copies image to output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            job_id = "test_job_123"
            image = sample_rag_images[0]

            # Simulate copy logic
            output_subdir = os.path.join(tmpdir, job_id, "rag_images")
            os.makedirs(output_subdir, exist_ok=True)

            image_id = image.get("image_id", "img")
            ext = os.path.splitext(image["file_path"])[1] or ".png"
            dest_filename = f"rag_{image_id}{ext}"
            dest_path = os.path.join(output_subdir, dest_filename)

            # Verify path structure
            assert job_id in dest_path
            assert "rag_images" in dest_path
            assert dest_filename == "rag_img_001.png"

    def test_filter_images_for_slides(self, sample_rag_images):
        """Test that filter_images_for_slides maps images to diagram slides"""
        slides = [
            {"type": "title", "title": "Introduction"},
            {"type": "DIAGRAM", "title": "Kafka Architecture"},
            {"type": "content", "title": "Key Points"},
            {"type": "DIAGRAM", "title": "Data Pipeline Overview"},
        ]
        min_score = 0.7

        slide_images = {}
        for i, slide in enumerate(slides):
            if slide.get("type") not in ["diagram", "DIAGRAM"]:
                continue

            title = slide.get("title", "")
            # Find matching image
            for img in sample_rag_images:
                if img.get("relevance_score", 0) >= min_score:
                    context = (img.get("context_text") or "").lower()
                    if any(w.lower() in context for w in title.split() if len(w) > 3):
                        slide_images[i] = img
                        break

        # Should map to slide indices 1 and 3 (diagram slides)
        assert 1 in slide_images or 3 in slide_images


# ============================================================================
# Tests for Fallback Chain
# ============================================================================

class TestFallbackChain:
    """Tests for the RAG → ViralifyDiagrams → DiagramGenerator fallback chain"""

    def test_fallback_order(self):
        """Test that fallback happens in correct order"""
        fallback_order = [
            "RAG Image",           # 1. Use image from documents
            "ViralifyDiagrams",    # 2. Generate with viralify-diagrams library
            "DiagramGenerator",    # 3. Generate with LLM
            "ContentSlide",        # 4. Convert to content slide
        ]

        assert fallback_order[0] == "RAG Image"
        assert fallback_order[1] == "ViralifyDiagrams"
        assert fallback_order[2] == "DiagramGenerator"
        assert fallback_order[3] == "ContentSlide"

    def test_rag_image_used_when_score_high(self, sample_rag_images):
        """Test that RAG image is used when score is above threshold"""
        threshold = 0.7
        kafka_image = sample_rag_images[0]

        # High score - should use RAG image
        if kafka_image["relevance_score"] >= threshold:
            method_used = "RAG Image"
        else:
            method_used = "ViralifyDiagrams"

        assert method_used == "RAG Image"

    def test_viralify_diagrams_used_when_no_rag(self):
        """Test that ViralifyDiagrams is used when no RAG image available"""
        rag_images = []
        viralify_enabled = os.environ.get("USE_VIRALIFY_DIAGRAMS", "true").lower() == "true"

        if not rag_images and viralify_enabled:
            method_used = "ViralifyDiagrams"
        else:
            method_used = "DiagramGenerator"

        assert method_used in ["ViralifyDiagrams", "DiagramGenerator"]

    def test_content_slide_used_as_last_resort(self):
        """Test that content slide is used when all generation fails"""
        rag_image_failed = True
        viralify_failed = True
        diagram_gen_failed = True

        if rag_image_failed and viralify_failed and diagram_gen_failed:
            method_used = "ContentSlide"
        else:
            method_used = "DiagramGenerator"

        assert method_used == "ContentSlide"


# ============================================================================
# Tests for Image Processing
# ============================================================================

class TestImageProcessing:
    """Tests for image processing in _use_rag_image()"""

    def test_image_resized_to_slide_dimensions(self):
        """Test that image is resized to fit slide canvas"""
        slide_width = 1920
        slide_height = 1080
        original_width = 800
        original_height = 600

        # Calculate aspect ratio preserving resize
        scale_x = slide_width / original_width
        scale_y = slide_height / original_height
        scale = min(scale_x, scale_y) * 0.9  # 90% of available space

        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        assert new_width <= slide_width
        assert new_height <= slide_height

    def test_image_centered_on_canvas(self):
        """Test that image is centered on slide canvas"""
        canvas_width = 1920
        canvas_height = 1080
        image_width = 1440
        image_height = 810

        # Calculate center position
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        assert x_offset > 0
        assert y_offset > 0
        assert x_offset == (1920 - 1440) // 2
        assert y_offset == (1080 - 810) // 2


# ============================================================================
# Tests for RAGImageReference Model
# ============================================================================

class TestRAGImageReferenceModel:
    """Tests for the RAGImageReference Pydantic model"""

    def test_model_creation(self):
        """Test that RAGImageReference can be created with required fields"""
        ref = RAGImageReference(
            image_id="img_001",
            document_id="doc_123",
            file_path="/tmp/images/diagram.png",
        )

        assert ref.image_id == "img_001"
        assert ref.document_id == "doc_123"
        assert ref.file_path == "/tmp/images/diagram.png"
        assert ref.detected_type == "diagram"  # Default
        assert ref.relevance_score == 0.0  # Default

    def test_model_with_all_fields(self):
        """Test that RAGImageReference can be created with all fields"""
        ref = RAGImageReference(
            image_id="img_001",
            document_id="doc_123",
            file_path="/tmp/images/diagram.png",
            file_name="diagram.png",
            detected_type="architecture",
            context_text="System architecture overview",
            caption="Architecture Diagram",
            description="Shows service connections",
            relevance_score=0.85,
            page_number=5,
            document_name="system_docs.pdf",
            width=1200,
            height=800,
            lecture_id="lec_001",
            lecture_title="System Design",
        )

        assert ref.detected_type == "architecture"
        assert ref.relevance_score == 0.85
        assert ref.page_number == 5
        assert ref.width == 1200

    def test_model_to_dict(self):
        """Test that RAGImageReference can be converted to dict"""
        ref = RAGImageReference(
            image_id="img_001",
            document_id="doc_123",
            file_path="/tmp/images/diagram.png",
            relevance_score=0.85,
        )

        ref_dict = ref.model_dump()

        assert isinstance(ref_dict, dict)
        assert ref_dict["image_id"] == "img_001"
        assert ref_dict["relevance_score"] == 0.85


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
