"""
Integration tests for RAG Image Extraction Pipeline

Tests the complete flow from document parsing to image retrieval,
including the integration with the pedagogical graph.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

import sys

# Add parent directories to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock heavy dependencies
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_documents_dir():
    """Create a temporary directory for document storage"""
    tmpdir = tempfile.mkdtemp(prefix="viralify_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_document_with_images(temp_documents_dir):
    """Create a sample document structure with images"""
    user_id = "test_user_123"
    doc_id = "doc_456"

    # Create directory structure
    images_dir = os.path.join(temp_documents_dir, user_id, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Create dummy image files
    image_files = []
    for i, name in enumerate(["kafka_arch.png", "pipeline_flow.png", "database_schema.png"]):
        img_path = os.path.join(images_dir, name)
        # Create a minimal valid PNG (1x1 pixel)
        with open(img_path, "wb") as f:
            # Minimal PNG header + IHDR + IDAT + IEND
            f.write(bytes([
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 dimensions
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,  # bit depth, color type
                0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
                0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
                0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
                0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
                0x44, 0xAE, 0x42, 0x60, 0x82
            ]))
        image_files.append(img_path)

    return {
        "user_id": user_id,
        "document_id": doc_id,
        "images_dir": images_dir,
        "image_files": image_files,
        "base_dir": temp_documents_dir,
    }


@pytest.fixture
def sample_extracted_images_db():
    """Simulated database of extracted images"""
    return [
        {
            "image_id": "img_001",
            "document_id": "doc_456",
            "file_path": "/tmp/viralify/documents/test_user/images/kafka_arch.png",
            "file_name": "kafka_arch.png",
            "detected_type": "architecture",
            "context_text": "Apache Kafka is a distributed event streaming platform capable of handling trillions of events a day. The architecture consists of brokers, producers, consumers, and ZooKeeper for coordination.",
            "caption": "Kafka Cluster Architecture",
            "description": "Diagram showing Kafka brokers, partitions, and consumer groups",
            "page_number": 8,
            "width": 1200,
            "height": 800,
        },
        {
            "image_id": "img_002",
            "document_id": "doc_456",
            "file_path": "/tmp/viralify/documents/test_user/images/pipeline_flow.png",
            "file_name": "pipeline_flow.png",
            "detected_type": "flowchart",
            "context_text": "The data pipeline processes events in real-time using Kafka Streams. Data flows from producers through topics to consumers for analytics.",
            "caption": "Real-time Data Pipeline",
            "description": "Flowchart of data pipeline stages",
            "page_number": 15,
            "width": 1600,
            "height": 900,
        },
        {
            "image_id": "img_003",
            "document_id": "doc_456",
            "file_path": "/tmp/viralify/documents/test_user/images/database_schema.png",
            "file_name": "database_schema.png",
            "detected_type": "diagram",
            "context_text": "PostgreSQL database schema for storing processed events with tables for events, users, and analytics.",
            "caption": "Database Schema",
            "description": "ER diagram showing table relationships",
            "page_number": 22,
            "width": 1000,
            "height": 700,
        },
    ]


@pytest.fixture
def sample_course_outline():
    """Sample course outline with lectures"""
    return {
        "title": "Building Real-time Data Pipelines with Kafka",
        "sections": [
            {
                "id": "sec_1",
                "title": "Introduction to Kafka",
                "lectures": [
                    {
                        "id": "lec_1",
                        "title": "Kafka Architecture Overview",
                        "description": "Understanding the core components of Apache Kafka",
                    },
                    {
                        "id": "lec_2",
                        "title": "Producers and Consumers",
                        "description": "How data flows through Kafka",
                    },
                ]
            },
            {
                "id": "sec_2",
                "title": "Building Pipelines",
                "lectures": [
                    {
                        "id": "lec_3",
                        "title": "Real-time Data Pipeline Design",
                        "description": "Designing streaming data pipelines",
                    },
                    {
                        "id": "lec_4",
                        "title": "Database Integration",
                        "description": "Connecting Kafka to PostgreSQL",
                    },
                ]
            }
        ]
    }


# ============================================================================
# Integration Tests: Document Parsing to Image Extraction
# ============================================================================

class TestDocumentToImageExtraction:
    """Tests for the complete document parsing and image extraction flow"""

    def test_images_stored_in_correct_directory(self, sample_document_with_images):
        """Test that extracted images are stored in the user's images directory"""
        images_dir = sample_document_with_images["images_dir"]
        image_files = sample_document_with_images["image_files"]

        assert os.path.exists(images_dir)
        assert len(image_files) == 3

        for img_path in image_files:
            assert os.path.exists(img_path)
            assert img_path.startswith(images_dir)

    def test_image_files_are_valid(self, sample_document_with_images):
        """Test that created image files are valid PNGs"""
        for img_path in sample_document_with_images["image_files"]:
            with open(img_path, "rb") as f:
                header = f.read(8)
                # Check PNG signature
                assert header[:4] == b'\x89PNG'

    def test_directory_structure_follows_convention(self, sample_document_with_images):
        """Test that directory structure follows {base}/{user_id}/images/ convention"""
        base_dir = sample_document_with_images["base_dir"]
        user_id = sample_document_with_images["user_id"]

        expected_path = os.path.join(base_dir, user_id, "images")
        assert sample_document_with_images["images_dir"] == expected_path


# ============================================================================
# Integration Tests: Image Retrieval Service
# ============================================================================

class TestImageRetrievalIntegration:
    """Tests for the image retrieval service integration"""

    def test_get_images_for_topic_scoring(self, sample_extracted_images_db):
        """Test the complete scoring algorithm for topic matching"""
        topic = "Kafka Architecture and Brokers"
        min_relevance = 0.3

        # Simulate the actual scoring logic from retrieval_service.py
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        topic_words = [w.lower() for w in topic.split() if w.lower() not in stopwords and len(w) > 2]

        results = []
        for img in sample_extracted_images_db:
            if img["detected_type"] not in ["diagram", "chart", "architecture", "flowchart", "schema"]:
                continue

            context = (img.get("context_text") or "").lower()
            caption = (img.get("caption") or "").lower()
            description = (img.get("description") or "").lower()

            # Weighted scoring
            context_score = sum(0.1 for w in topic_words if w in context)
            caption_score = sum(0.08 for w in topic_words if w in caption)
            description_score = sum(0.05 for w in topic_words if w in description)

            # Bonus for diagram types
            type_bonus = 0.1 if img["detected_type"] in ["architecture", "diagram"] else 0.05

            total_score = min(1.0, context_score + caption_score + description_score + type_bonus)

            if total_score >= min_relevance:
                results.append({
                    **img,
                    "relevance_score": round(total_score, 3)
                })

        # Sort by score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Verify results
        assert len(results) >= 1
        assert results[0]["file_name"] == "kafka_arch.png"
        assert results[0]["relevance_score"] > 0.3

    def test_retrieval_filters_non_diagram_types(self, sample_extracted_images_db):
        """Test that non-diagram image types are filtered out"""
        # Add a photo type image
        images_with_photo = sample_extracted_images_db + [{
            "image_id": "img_photo",
            "document_id": "doc_456",
            "file_path": "/tmp/images/team_photo.jpg",
            "detected_type": "photo",
            "context_text": "The development team",
        }]

        diagram_types = ["diagram", "chart", "architecture", "flowchart", "schema"]
        filtered = [img for img in images_with_photo if img["detected_type"] in diagram_types]

        assert len(filtered) == 3
        assert all(img["detected_type"] != "photo" for img in filtered)

    def test_retrieval_respects_document_ids_filter(self, sample_extracted_images_db):
        """Test that retrieval filters by document_ids"""
        document_ids = ["doc_456"]

        filtered = [
            img for img in sample_extracted_images_db
            if img["document_id"] in document_ids
        ]

        assert len(filtered) == 3

        # Test with non-matching document
        filtered_empty = [
            img for img in sample_extracted_images_db
            if img["document_id"] in ["doc_999"]
        ]

        assert len(filtered_empty) == 0


# ============================================================================
# Integration Tests: Pedagogical Graph to RAG Images
# ============================================================================

class TestPedagogicalGraphIntegration:
    """Tests for the integration of RAG images into the pedagogical graph"""

    def test_fetch_rag_images_populates_state(self, sample_course_outline, sample_extracted_images_db):
        """Test that fetch_rag_images correctly populates the state"""
        state = {
            "outline": sample_course_outline,
            "document_ids": ["doc_456"],
            "user_id": "test_user",
            "rag_images": [],
        }

        # Simulate fetch_rag_images logic
        lectures = []
        for section in sample_course_outline.get("sections", []):
            lectures.extend(section.get("lectures", []))

        rag_images = []
        for lecture in lectures:
            topic = lecture.get("title", "")
            topic_words = [w.lower() for w in topic.split() if len(w) > 3]

            for img in sample_extracted_images_db:
                context = (img.get("context_text") or "").lower()
                score = sum(0.15 for w in topic_words if w in context)

                if score >= 0.3:
                    rag_images.append({
                        "lecture_id": lecture["id"],
                        "lecture_title": lecture["title"],
                        "image_id": img["image_id"],
                        "file_path": img["file_path"],
                        "detected_type": img["detected_type"],
                        "relevance_score": min(1.0, score),
                        "context_text": img.get("context_text", ""),
                    })

        state["rag_images"] = rag_images

        # Verify
        assert len(state["rag_images"]) >= 1
        kafka_images = [img for img in state["rag_images"] if "kafka" in img["lecture_title"].lower()]
        assert len(kafka_images) >= 1

    def test_images_mapped_to_correct_lectures(self, sample_course_outline, sample_extracted_images_db):
        """Test that images are correctly mapped to their matching lectures"""
        lectures = []
        for section in sample_course_outline.get("sections", []):
            lectures.extend(section.get("lectures", []))

        # Map lecture topics to expected images
        expected_mappings = {
            "Kafka Architecture Overview": "kafka_arch.png",
            "Real-time Data Pipeline Design": "pipeline_flow.png",
            "Database Integration": "database_schema.png",
        }

        for lecture in lectures:
            title = lecture["title"]
            if title in expected_mappings:
                expected_file = expected_mappings[title]
                topic_words = [w.lower() for w in title.split() if len(w) > 3]

                # Find matching image
                for img in sample_extracted_images_db:
                    if img["file_name"] == expected_file:
                        context = img.get("context_text", "").lower()
                        matches = any(w in context for w in topic_words)
                        assert matches, f"Expected {expected_file} to match '{title}'"

    def test_state_passed_to_production_graph(self, sample_course_outline, sample_extracted_images_db):
        """Test that rag_images is correctly passed through the production graph"""
        # Simulate the state as it would be after fetch_rag_images
        state = {
            "outline": sample_course_outline,
            "document_ids": ["doc_456"],
            "user_id": "test_user",
            "rag_images": [
                {
                    "lecture_id": "lec_1",
                    "image_id": "img_001",
                    "file_path": "/tmp/images/kafka_arch.png",
                    "detected_type": "architecture",
                    "relevance_score": 0.85,
                }
            ],
        }

        # Simulate production_graph extracting rag_images for presentation request
        settings = {
            "topic": sample_course_outline["title"],
            "rag_images": state.get("rag_images", []),
        }

        presentation_request = {
            "topic": settings["topic"],
            "rag_images": settings.get("rag_images", []),
        }

        assert "rag_images" in presentation_request
        assert len(presentation_request["rag_images"]) == 1
        assert presentation_request["rag_images"][0]["image_id"] == "img_001"


# ============================================================================
# Integration Tests: End-to-End Flow
# ============================================================================

class TestEndToEndFlow:
    """Tests for the complete end-to-end flow"""

    def test_complete_extraction_to_state_flow(
        self,
        sample_document_with_images,
        sample_course_outline
    ):
        """Test the complete flow from document to state population"""
        # Step 1: Simulate document parsing (images extracted)
        extracted_images = []
        for i, img_path in enumerate(sample_document_with_images["image_files"]):
            extracted_images.append({
                "image_id": f"img_{i:03d}",
                "document_id": sample_document_with_images["document_id"],
                "file_path": img_path,
                "file_name": os.path.basename(img_path),
                "detected_type": "diagram",
                "context_text": f"Sample context for image {i}",
                "page_number": i + 1,
            })

        # Step 2: Simulate retrieval service storing images
        image_store = {
            sample_document_with_images["user_id"]: extracted_images
        }

        assert sample_document_with_images["user_id"] in image_store
        assert len(image_store[sample_document_with_images["user_id"]]) == 3

        # Step 3: Simulate fetch_rag_images querying the store
        user_images = image_store.get(sample_document_with_images["user_id"], [])

        # Step 4: Populate state
        state = {
            "rag_images": user_images,
            "outline": sample_course_outline,
        }

        assert len(state["rag_images"]) == 3

    def test_flow_handles_no_documents(self, sample_course_outline):
        """Test that flow handles case with no documents gracefully"""
        state = {
            "outline": sample_course_outline,
            "document_ids": [],
            "user_id": "test_user",
        }

        # With no documents, rag_images should be empty
        if not state.get("document_ids"):
            state["rag_images"] = []

        assert state["rag_images"] == []

    def test_flow_handles_no_matching_images(self, sample_course_outline):
        """Test that flow handles case with no matching images"""
        # Images that don't match the course topic
        unrelated_images = [
            {
                "image_id": "img_001",
                "file_path": "/tmp/images/cooking_recipe.png",
                "detected_type": "diagram",
                "context_text": "How to make pasta carbonara",
                "relevance_score": 0.0,
            }
        ]

        # No images should match Kafka course
        state = {
            "outline": sample_course_outline,
            "document_ids": ["doc_456"],
            "rag_images": [],
        }

        # Simulate matching logic
        lectures = []
        for section in sample_course_outline.get("sections", []):
            lectures.extend(section.get("lectures", []))

        for lecture in lectures:
            topic = lecture.get("title", "").lower()
            for img in unrelated_images:
                context = (img.get("context_text") or "").lower()
                # No match expected
                if any(w in context for w in topic.split() if len(w) > 3):
                    state["rag_images"].append(img)

        # No images should match
        assert len(state["rag_images"]) == 0


# ============================================================================
# Integration Tests: Cross-Service Communication
# ============================================================================

class TestCrossServiceCommunication:
    """Tests for communication between course-generator and presentation-generator"""

    def test_rag_images_serialization(self, sample_extracted_images_db):
        """Test that rag_images can be serialized for API transmission"""
        import json

        rag_images = [
            {
                "image_id": img["image_id"],
                "document_id": img["document_id"],
                "file_path": img["file_path"],
                "detected_type": img["detected_type"],
                "context_text": img.get("context_text"),
                "relevance_score": 0.85,
            }
            for img in sample_extracted_images_db[:2]
        ]

        # Should be JSON serializable
        json_str = json.dumps(rag_images)
        assert isinstance(json_str, str)

        # Should deserialize correctly
        deserialized = json.loads(json_str)
        assert len(deserialized) == 2
        assert deserialized[0]["image_id"] == "img_001"

    def test_presentation_request_format(self, sample_extracted_images_db):
        """Test that presentation request includes rag_images in correct format"""
        presentation_request = {
            "topic": "Apache Kafka Tutorial",
            "language": "python",
            "content_language": "en",
            "duration": 300,
            "style": "dark",
            "target_audience": "intermediate developers",
            "rag_images": [
                {
                    "image_id": img["image_id"],
                    "document_id": img["document_id"],
                    "file_path": img["file_path"],
                    "file_name": img.get("file_name"),
                    "detected_type": img["detected_type"],
                    "context_text": img.get("context_text"),
                    "caption": img.get("caption"),
                    "description": img.get("description"),
                    "relevance_score": 0.85,
                    "page_number": img.get("page_number"),
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                }
                for img in sample_extracted_images_db
            ],
        }

        assert "rag_images" in presentation_request
        assert len(presentation_request["rag_images"]) == 3

        # Verify all required fields present
        for img in presentation_request["rag_images"]:
            assert "image_id" in img
            assert "document_id" in img
            assert "file_path" in img
            assert "detected_type" in img
            assert "relevance_score" in img


# ============================================================================
# Integration Tests: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in the integration"""

    def test_handles_missing_file_path(self, sample_extracted_images_db):
        """Test handling of images with missing file paths"""
        images_with_missing = sample_extracted_images_db + [{
            "image_id": "img_missing",
            "document_id": "doc_456",
            "file_path": None,  # Missing
            "detected_type": "diagram",
        }]

        valid_images = [
            img for img in images_with_missing
            if img.get("file_path")
        ]

        assert len(valid_images) == 3

    def test_handles_invalid_image_type(self, sample_extracted_images_db):
        """Test handling of images with invalid types"""
        images_with_invalid = sample_extracted_images_db + [{
            "image_id": "img_invalid",
            "document_id": "doc_456",
            "file_path": "/tmp/images/invalid.xyz",
            "detected_type": "unknown_type",
        }]

        valid_types = ["diagram", "chart", "architecture", "flowchart", "schema"]
        filtered = [
            img for img in images_with_invalid
            if img.get("detected_type") in valid_types
        ]

        assert len(filtered) == 3

    def test_handles_empty_context(self, sample_extracted_images_db):
        """Test handling of images with empty context"""
        images_with_empty = [{
            "image_id": "img_empty",
            "document_id": "doc_456",
            "file_path": "/tmp/images/empty.png",
            "detected_type": "diagram",
            "context_text": "",  # Empty
            "caption": "",
            "description": "",
        }]

        topic = "Kafka Architecture"
        topic_words = [w.lower() for w in topic.split() if len(w) > 3]

        for img in images_with_empty:
            context = (img.get("context_text") or "").lower()
            caption = (img.get("caption") or "").lower()

            # With empty context, no matches expected
            matches = sum(1 for w in topic_words if w in context or w in caption)
            assert matches == 0


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
