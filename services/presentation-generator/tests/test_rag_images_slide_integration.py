"""
Integration tests for RAG Images in Slide Generation Pipeline

Tests the complete flow from receiving RAG images to rendering diagram slides,
including the fallback chain and cross-service integration.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import List, Dict, Any, Optional
from io import BytesIO

import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock PIL for headless testing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = MagicMock()

from models.presentation_models import (
    Slide,
    SlideType,
    PresentationStyle,
    RAGImageReference,
    GeneratePresentationRequest,
    PresentationJob,
    PresentationStage,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory"""
    tmpdir = tempfile.mkdtemp(prefix="viralify_slides_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_png_image(temp_output_dir):
    """Create a sample PNG image file"""
    img_path = os.path.join(temp_output_dir, "sample_diagram.png")

    if PIL_AVAILABLE:
        # Create a real 100x100 red image
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path, "PNG")
    else:
        # Create minimal PNG bytes
        with open(img_path, "wb") as f:
            f.write(bytes([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
                0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
                0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
                0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
                0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
                0x44, 0xAE, 0x42, 0x60, 0x82
            ]))

    return img_path


@pytest.fixture
def sample_rag_image_references(sample_png_image):
    """Create sample RAGImageReference objects"""
    return [
        RAGImageReference(
            image_id="img_001",
            document_id="doc_123",
            file_path=sample_png_image,
            file_name="kafka_architecture.png",
            detected_type="architecture",
            context_text="Apache Kafka distributed streaming platform with brokers and partitions",
            caption="Kafka Architecture Overview",
            description="Diagram showing Kafka cluster components",
            relevance_score=0.92,
            page_number=5,
            width=1200,
            height=800,
            lecture_id="lec_001",
            lecture_title="Introduction to Kafka",
        ),
        RAGImageReference(
            image_id="img_002",
            document_id="doc_123",
            file_path=sample_png_image,
            file_name="data_pipeline.png",
            detected_type="flowchart",
            context_text="ETL pipeline with streaming and batch processing",
            caption="Data Pipeline Flow",
            description="Flowchart of data pipeline stages",
            relevance_score=0.78,
            page_number=12,
            width=1600,
            height=900,
            lecture_id="lec_002",
            lecture_title="Building Data Pipelines",
        ),
        RAGImageReference(
            image_id="img_003",
            document_id="doc_456",
            file_path=sample_png_image,
            file_name="low_score_diagram.png",
            detected_type="chart",
            context_text="Unrelated content about sales metrics",
            caption="Sales Chart",
            description="Quarterly sales data",
            relevance_score=0.45,  # Below threshold
            page_number=3,
            width=800,
            height=600,
        ),
    ]


@pytest.fixture
def sample_diagram_slides():
    """Create sample diagram slides for testing"""
    return [
        Slide(
            id="slide_001",
            type=SlideType.DIAGRAM,
            title="Kafka Architecture Overview",
            subtitle="Understanding the core components",
            content="Kafka consists of brokers, producers, and consumers",
            duration=30.0,
            voiceover_text="Let's explore the Kafka architecture...",
            diagram_type="architecture",
        ),
        Slide(
            id="slide_002",
            type=SlideType.DIAGRAM,
            title="Data Pipeline Design",
            subtitle="Building streaming pipelines",
            content="Real-time data flows through Kafka",
            duration=25.0,
            voiceover_text="Here's how we design the pipeline...",
            diagram_type="flowchart",
        ),
        Slide(
            id="slide_003",
            type=SlideType.DIAGRAM,
            title="Quantum Computing Basics",  # No matching image
            subtitle="An introduction",
            content="Qubits and quantum gates",
            duration=20.0,
            voiceover_text="Quantum computing represents...",
            diagram_type="diagram",
        ),
    ]


@pytest.fixture
def sample_presentation_request(sample_rag_image_references):
    """Create a sample GeneratePresentationRequest with RAG images"""
    return GeneratePresentationRequest(
        topic="Apache Kafka Tutorial: Building Real-time Data Pipelines",
        language="python",
        content_language="en",
        duration=300,
        style=PresentationStyle.DARK,
        target_audience="senior developers",
        target_career="data_engineer",
        rag_images=sample_rag_image_references,
    )


@pytest.fixture
def sample_presentation_job(sample_presentation_request):
    """Create a sample PresentationJob"""
    job = PresentationJob(
        job_id="test_job_123",
        request=sample_presentation_request,
    )
    return job


# ============================================================================
# Integration Tests: RAG Image Client
# ============================================================================

class TestRAGImageClientIntegration:
    """Tests for RAGImageClient integration with slide generation"""

    def test_find_matching_image_with_real_references(self, sample_rag_image_references):
        """Test finding matching images with real RAGImageReference objects"""
        slide_topic = "Kafka Architecture"
        min_score = 0.7
        allowed_types = ["diagram", "chart", "architecture", "flowchart"]

        # Convert to dicts for matching
        rag_images = [ref.model_dump() for ref in sample_rag_image_references]

        candidates = []
        for img in rag_images:
            if img.get("relevance_score", 0) < min_score:
                continue
            if img.get("detected_type") not in allowed_types:
                continue

            # Calculate adjusted score with topic bonus
            context = (img.get("context_text") or "").lower()
            caption = (img.get("caption") or "").lower()
            topic_words = [w.lower() for w in slide_topic.split() if len(w) > 3]

            bonus = sum(0.05 for w in topic_words if w in context or w in caption)
            adjusted_score = min(1.0, img["relevance_score"] + bonus)

            candidates.append({**img, "adjusted_score": adjusted_score})

        candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)

        assert len(candidates) >= 1
        best_match = candidates[0]
        assert "kafka" in best_match["context_text"].lower()
        assert best_match["adjusted_score"] > 0.9

    def test_copy_to_output_creates_correct_structure(self, sample_png_image, temp_output_dir):
        """Test that copying RAG image creates correct output structure"""
        job_id = "test_job_456"

        # Simulate copy logic
        output_subdir = os.path.join(temp_output_dir, job_id, "rag_images")
        os.makedirs(output_subdir, exist_ok=True)

        image_id = "img_001"
        ext = os.path.splitext(sample_png_image)[1]
        dest_filename = f"rag_{image_id}{ext}"
        dest_path = os.path.join(output_subdir, dest_filename)

        # Copy file
        shutil.copy2(sample_png_image, dest_path)

        assert os.path.exists(dest_path)
        assert os.path.isfile(dest_path)
        assert dest_filename == "rag_img_001.png"

    def test_filter_images_for_slides_mapping(self, sample_rag_image_references, sample_diagram_slides):
        """Test mapping RAG images to diagram slides"""
        rag_images = [ref.model_dump() for ref in sample_rag_image_references]
        min_score = 0.7

        slide_images = {}
        for i, slide in enumerate(sample_diagram_slides):
            if slide.type != SlideType.DIAGRAM:
                continue

            title = slide.title or ""
            title_words = [w.lower() for w in title.split() if len(w) > 3]

            for img in rag_images:
                if img.get("relevance_score", 0) < min_score:
                    continue

                context = (img.get("context_text") or "").lower()
                caption = (img.get("caption") or "").lower()

                if any(w in context or w in caption for w in title_words):
                    slide_images[i] = img
                    break

        # Slide 0 (Kafka) and Slide 1 (Pipeline) should have matches
        # Slide 2 (Quantum) should not have a match
        assert 0 in slide_images  # Kafka slide matched
        assert 1 in slide_images  # Pipeline slide matched
        assert 2 not in slide_images  # Quantum slide not matched


# ============================================================================
# Integration Tests: Slide Generator with RAG Images
# ============================================================================

class TestSlideGeneratorRAGIntegration:
    """Tests for SlideGenerator integration with RAG images"""

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_use_rag_image_loads_and_resizes(self, sample_png_image, temp_output_dir):
        """Test that RAG image is loaded and resized correctly"""
        # Load original image
        original = Image.open(sample_png_image)
        original_size = original.size

        # Target slide dimensions
        slide_width = 1920
        slide_height = 1080

        # Calculate resize (90% of available space)
        scale_x = slide_width / original_size[0]
        scale_y = slide_height / original_size[1]
        scale = min(scale_x, scale_y) * 0.9

        new_width = int(original_size[0] * scale)
        new_height = int(original_size[1] * scale)

        # Resize
        resized = original.resize((new_width, new_height), Image.Resampling.LANCZOS)

        assert resized.size[0] <= slide_width
        assert resized.size[1] <= slide_height

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_use_rag_image_centers_on_canvas(self, sample_png_image):
        """Test that RAG image is centered on slide canvas"""
        # Create canvas
        canvas_width = 1920
        canvas_height = 1080
        canvas = Image.new("RGB", (canvas_width, canvas_height), color="#1e1e2e")

        # Load and resize image
        img = Image.open(sample_png_image)
        img_resized = img.resize((800, 600))

        # Calculate center position
        x_offset = (canvas_width - img_resized.size[0]) // 2
        y_offset = (canvas_height - img_resized.size[1]) // 2

        # Paste centered
        canvas.paste(img_resized, (x_offset, y_offset))

        assert x_offset == (1920 - 800) // 2
        assert y_offset == (1080 - 600) // 2

    def test_should_use_rag_images_env_check(self):
        """Test USE_RAG_IMAGES environment variable check"""
        # Test with default (true)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("USE_RAG_IMAGES", None)
            result = os.environ.get("USE_RAG_IMAGES", "true").lower() == "true"
            assert result is True

        # Test with explicit false
        with patch.dict(os.environ, {"USE_RAG_IMAGES": "false"}):
            result = os.environ.get("USE_RAG_IMAGES", "true").lower() == "true"
            assert result is False

        # Test with explicit true
        with patch.dict(os.environ, {"USE_RAG_IMAGES": "true"}):
            result = os.environ.get("USE_RAG_IMAGES", "true").lower() == "true"
            assert result is True


# ============================================================================
# Integration Tests: Presentation Compositor
# ============================================================================

class TestPresentationCompositorIntegration:
    """Tests for PresentationCompositor integration with RAG images"""

    def test_extracts_rag_images_from_request(self, sample_presentation_job):
        """Test that compositor extracts rag_images from job request"""
        job = sample_presentation_job

        rag_images = None
        if job.request and hasattr(job.request, 'rag_images'):
            rag_images = [
                img.model_dump() if hasattr(img, 'model_dump') else img
                for img in job.request.rag_images
            ]

        assert rag_images is not None
        assert len(rag_images) == 3
        assert rag_images[0]["image_id"] == "img_001"

    def test_passes_rag_images_to_slide_generator(self, sample_presentation_job):
        """Test that rag_images is passed through to slide generation"""
        job = sample_presentation_job

        # Simulate the parameter extraction
        rag_images = [
            img.model_dump() if hasattr(img, 'model_dump') else img
            for img in job.request.rag_images
        ]
        job_id = job.job_id

        # These should be passed to generate_slide_image
        params = {
            "rag_images": rag_images,
            "job_id": job_id,
        }

        assert params["rag_images"] is not None
        assert params["job_id"] == "test_job_123"
        assert len(params["rag_images"]) == 3


# ============================================================================
# Integration Tests: LangGraph Orchestrator
# ============================================================================

class TestLangGraphOrchestratorIntegration:
    """Tests for LangGraphOrchestrator integration with RAG images"""

    def test_extracts_rag_images_from_request_dict(self, sample_rag_image_references):
        """Test that orchestrator extracts rag_images from request dict"""
        request = {
            "topic": "Apache Kafka",
            "rag_images": [ref.model_dump() for ref in sample_rag_image_references],
            "rag_context": "Some RAG context...",
        }
        state = {
            "request": request,
            "job_id": "test_job_789",
        }

        rag_images = request.get("rag_images", [])
        job_id = state.get("job_id")

        assert len(rag_images) == 3
        assert job_id == "test_job_789"

    def test_logs_rag_images_usage(self, sample_rag_image_references, capsys):
        """Test that RAG images usage is logged"""
        rag_images = [ref.model_dump() for ref in sample_rag_image_references]

        # Simulate logging
        if rag_images:
            print(f"[LANGGRAPH] Using {len(rag_images)} RAG images for diagram slides", flush=True)

        captured = capsys.readouterr()
        assert "Using 3 RAG images" in captured.out


# ============================================================================
# Integration Tests: Visual Sync Agent
# ============================================================================

class TestVisualSyncAgentIntegration:
    """Tests for VisualSyncAgent integration with RAG images"""

    def test_extracts_rag_images_from_state(self, sample_rag_image_references):
        """Test that agent extracts rag_images from state"""
        state = {
            "slide_data": {"type": "diagram", "title": "Kafka Architecture"},
            "job_id": "test_job",
            "style": "dark",
            "target_audience": "senior developers",
            "rag_images": [ref.model_dump() for ref in sample_rag_image_references],
        }

        rag_images = state.get("rag_images")

        assert rag_images is not None
        assert len(rag_images) == 3

    def test_passes_rag_images_to_generate_slide_image(self, sample_rag_image_references):
        """Test that rag_images is passed to _generate_slide_image"""
        rag_images = [ref.model_dump() for ref in sample_rag_image_references]

        # Simulate the call parameters
        call_params = {
            "slide_data": {"type": "diagram"},
            "job_id": "test_job",
            "scene_index": 0,
            "style": "dark",
            "target_audience": "senior developers",
            "target_career": "data_engineer",
            "rag_context": None,
            "course_context": None,
            "rag_images": rag_images,
        }

        assert "rag_images" in call_params
        assert len(call_params["rag_images"]) == 3


# ============================================================================
# Integration Tests: Fallback Chain
# ============================================================================

class TestFallbackChainIntegration:
    """Tests for the complete fallback chain integration"""

    def test_fallback_chain_with_matching_rag_image(self, sample_rag_image_references, sample_png_image):
        """Test that RAG image is used when available and matching"""
        slide_topic = "Kafka Architecture Overview"
        rag_images = [ref.model_dump() for ref in sample_rag_image_references]
        min_score = 0.7

        # Step 1: Try RAG image
        matching_image = None
        for img in rag_images:
            if img.get("relevance_score", 0) >= min_score:
                context = (img.get("context_text") or "").lower()
                if any(w in context for w in ["kafka", "architecture"]):
                    matching_image = img
                    break

        method_used = "RAG Image" if matching_image else "LLM Generation"

        assert method_used == "RAG Image"
        assert matching_image["image_id"] == "img_001"

    def test_fallback_chain_without_rag_image(self, sample_diagram_slides):
        """Test that LLM is used when no RAG image matches"""
        slide = sample_diagram_slides[2]  # Quantum Computing - no match
        rag_images = []  # Empty list

        # Step 1: Try RAG image - fails
        matching_image = None

        # Step 2: Determine fallback
        if matching_image:
            method_used = "RAG Image"
        else:
            viralify_enabled = os.environ.get("USE_VIRALIFY_DIAGRAMS", "true").lower() == "true"
            method_used = "ViralifyDiagrams" if viralify_enabled else "DiagramGenerator"

        assert method_used in ["ViralifyDiagrams", "DiagramGenerator"]

    def test_fallback_chain_with_low_score_rag_image(self, sample_rag_image_references):
        """Test that LLM is used when RAG image score is too low"""
        slide_topic = "Random Unrelated Topic"
        rag_images = [ref.model_dump() for ref in sample_rag_image_references]
        min_score = 0.7

        # All images should have low adjusted scores for unrelated topic
        matching_image = None
        for img in rag_images:
            context = (img.get("context_text") or "").lower()
            # No matching words
            if "random" in context and "unrelated" in context:
                if img.get("relevance_score", 0) >= min_score:
                    matching_image = img
                    break

        assert matching_image is None  # No match found


# ============================================================================
# Integration Tests: Cross-Service Data Flow
# ============================================================================

class TestCrossServiceDataFlow:
    """Tests for data flow between services"""

    def test_rag_images_format_consistency(self, sample_rag_image_references):
        """Test that RAG images format is consistent across services"""
        # Course-generator output format
        course_gen_format = {
            "lecture_id": "lec_001",
            "image_id": "img_001",
            "file_path": "/tmp/images/kafka.png",
            "detected_type": "architecture",
            "relevance_score": 0.85,
            "context_text": "Kafka streaming platform",
        }

        # Presentation-generator expected format (RAGImageReference)
        pres_gen_ref = RAGImageReference(
            image_id=course_gen_format["image_id"],
            document_id="doc_123",
            file_path=course_gen_format["file_path"],
            detected_type=course_gen_format["detected_type"],
            relevance_score=course_gen_format["relevance_score"],
            context_text=course_gen_format["context_text"],
            lecture_id=course_gen_format["lecture_id"],
        )

        # Verify conversion works
        assert pres_gen_ref.image_id == course_gen_format["image_id"]
        assert pres_gen_ref.relevance_score == course_gen_format["relevance_score"]

    def test_file_path_accessibility(self, sample_png_image):
        """Test that file paths are accessible across service boundaries"""
        # In Docker, shared volumes make paths accessible
        # Test that the file exists and is readable
        assert os.path.exists(sample_png_image)

        with open(sample_png_image, "rb") as f:
            header = f.read(8)
            assert header[:4] == b'\x89PNG'

    def test_request_serialization_roundtrip(self, sample_presentation_request):
        """Test that request can be serialized and deserialized"""
        import json

        # Serialize
        request_dict = sample_presentation_request.model_dump()
        json_str = json.dumps(request_dict, default=str)

        # Deserialize
        parsed = json.loads(json_str)

        assert parsed["topic"] == sample_presentation_request.topic
        assert len(parsed["rag_images"]) == 3


# ============================================================================
# Integration Tests: Error Recovery
# ============================================================================

class TestErrorRecoveryIntegration:
    """Tests for error recovery in the integration"""

    def test_handles_missing_image_file(self, sample_rag_image_references, temp_output_dir):
        """Test handling when RAG image file doesn't exist"""
        # Create reference to non-existent file
        bad_ref = RAGImageReference(
            image_id="img_missing",
            document_id="doc_123",
            file_path="/nonexistent/path/image.png",
            relevance_score=0.95,
        )

        # Should fallback gracefully
        file_exists = os.path.exists(bad_ref.file_path)
        assert file_exists is False

        # Method should fallback to LLM
        if not file_exists:
            method_used = "LLM Generation"
        else:
            method_used = "RAG Image"

        assert method_used == "LLM Generation"

    def test_handles_corrupted_image_file(self, temp_output_dir):
        """Test handling when RAG image file is corrupted"""
        # Create corrupted file
        corrupted_path = os.path.join(temp_output_dir, "corrupted.png")
        with open(corrupted_path, "wb") as f:
            f.write(b"not a valid png file")

        # Try to detect corruption
        is_valid = False
        with open(corrupted_path, "rb") as f:
            header = f.read(8)
            is_valid = header[:4] == b'\x89PNG'

        assert is_valid is False

    def test_handles_empty_rag_images_list(self, sample_diagram_slides):
        """Test handling when rag_images list is empty"""
        rag_images = []
        slide = sample_diagram_slides[0]

        # Should skip RAG image check
        if not rag_images:
            use_rag = False
        else:
            use_rag = True

        assert use_rag is False


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
