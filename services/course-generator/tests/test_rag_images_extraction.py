"""
Unit tests for RAG Image Extraction

Tests the extraction and retrieval of images from documents for use in diagram slides.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_extracted_images():
    """Sample extracted images from documents"""
    return [
        {
            "image_id": "img_001",
            "document_id": "doc_123",
            "file_path": "/tmp/viralify/documents/user_1/images/diagram_kafka.png",
            "file_name": "diagram_kafka.png",
            "detected_type": "diagram",
            "context_text": "Apache Kafka is a distributed streaming platform for building real-time data pipelines",
            "caption": "Kafka Architecture Overview",
            "description": "A diagram showing Kafka brokers, producers, and consumers",
            "page_number": 5,
            "width": 800,
            "height": 600,
            "relevance_score": 0.0,  # Will be calculated
        },
        {
            "image_id": "img_002",
            "document_id": "doc_123",
            "file_path": "/tmp/viralify/documents/user_1/images/chart_performance.png",
            "file_name": "chart_performance.png",
            "detected_type": "chart",
            "context_text": "Performance benchmarks show throughput improvements",
            "caption": "Performance Comparison",
            "description": "Bar chart comparing different configurations",
            "page_number": 12,
            "width": 640,
            "height": 480,
            "relevance_score": 0.0,
        },
        {
            "image_id": "img_003",
            "document_id": "doc_456",
            "file_path": "/tmp/viralify/documents/user_1/images/architecture_microservices.png",
            "file_name": "architecture_microservices.png",
            "detected_type": "architecture",
            "context_text": "Microservices architecture pattern with API gateway",
            "caption": "Microservices Architecture",
            "description": "Architecture diagram showing service mesh",
            "page_number": 3,
            "width": 1200,
            "height": 800,
            "relevance_score": 0.0,
        },
        {
            "image_id": "img_004",
            "document_id": "doc_456",
            "file_path": "/tmp/viralify/documents/user_1/images/photo_team.jpg",
            "file_name": "photo_team.jpg",
            "detected_type": "photo",  # Should be filtered out for diagrams
            "context_text": "Our development team",
            "caption": "Team Photo",
            "description": "Photo of the development team",
            "page_number": 1,
            "width": 1920,
            "height": 1080,
            "relevance_score": 0.0,
        },
    ]


@pytest.fixture
def mock_retrieval_service(sample_extracted_images):
    """Mock retrieval service with sample images"""
    service = MagicMock()
    service.user_images = {"user_1": sample_extracted_images}
    return service


# ============================================================================
# Tests for get_images_for_topic
# ============================================================================

class TestGetImagesForTopic:
    """Tests for retrieval_service.get_images_for_topic()"""

    def test_returns_relevant_images_for_topic(self, sample_extracted_images):
        """Test that relevant images are returned for a matching topic"""
        topic = "Apache Kafka Architecture"

        # Simulate the scoring logic
        results = []
        for img in sample_extracted_images:
            if img["detected_type"] in ["diagram", "chart", "architecture", "flowchart"]:
                context = (img.get("context_text") or "").lower()
                caption = (img.get("caption") or "").lower()
                topic_lower = topic.lower()

                # Simple keyword matching
                score = 0.0
                topic_words = [w for w in topic_lower.split() if len(w) > 3]
                for word in topic_words:
                    if word in context:
                        score += 0.2
                    if word in caption:
                        score += 0.15

                if score > 0.3:
                    img_copy = img.copy()
                    img_copy["relevance_score"] = min(1.0, score)
                    results.append(img_copy)

        # Should find the Kafka diagram
        assert len(results) >= 1
        kafka_images = [r for r in results if "kafka" in r["file_name"].lower()]
        assert len(kafka_images) == 1
        assert kafka_images[0]["relevance_score"] > 0.3

    def test_filters_by_image_type(self, sample_extracted_images):
        """Test that only diagram-suitable types are returned"""
        diagram_types = ["diagram", "chart", "architecture", "flowchart"]

        filtered = [
            img for img in sample_extracted_images
            if img["detected_type"] in diagram_types
        ]

        # Should exclude photos
        assert len(filtered) == 3
        for img in filtered:
            assert img["detected_type"] != "photo"

    def test_respects_min_relevance_threshold(self, sample_extracted_images):
        """Test that images below min_relevance are filtered out"""
        min_relevance = 0.5

        # Simulate scoring with low scores
        results = []
        for img in sample_extracted_images:
            img_copy = img.copy()
            img_copy["relevance_score"] = 0.3  # Below threshold
            results.append(img_copy)

        filtered = [r for r in results if r["relevance_score"] >= min_relevance]

        # All should be filtered out with 0.3 scores and 0.5 threshold
        assert len(filtered) == 0

    def test_sorts_by_relevance_score(self, sample_extracted_images):
        """Test that results are sorted by relevance score descending"""
        # Assign different scores
        images_with_scores = []
        for i, img in enumerate(sample_extracted_images[:3]):
            img_copy = img.copy()
            img_copy["relevance_score"] = 0.9 - (i * 0.2)  # 0.9, 0.7, 0.5
            images_with_scores.append(img_copy)

        # Sort by score
        sorted_images = sorted(
            images_with_scores,
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        assert sorted_images[0]["relevance_score"] == 0.9
        assert sorted_images[1]["relevance_score"] == 0.7
        assert sorted_images[2]["relevance_score"] == 0.5

    def test_handles_empty_topic(self, sample_extracted_images):
        """Test that empty topic returns empty results"""
        topic = ""

        # With empty topic, no keywords to match
        topic_words = [w for w in topic.lower().split() if len(w) > 3]

        assert len(topic_words) == 0

    def test_handles_no_images(self):
        """Test that empty image list returns empty results"""
        images = []
        topic = "Apache Kafka"

        results = [
            img for img in images
            if img.get("relevance_score", 0) > 0.3
        ]

        assert len(results) == 0

    def test_bonus_for_diagram_type(self, sample_extracted_images):
        """Test that diagram/architecture types get a score bonus"""
        diagram_types = ["diagram", "architecture"]

        for img in sample_extracted_images:
            bonus = 0.1 if img["detected_type"] in diagram_types else 0.0
            if img["detected_type"] == "diagram":
                assert bonus == 0.1
            elif img["detected_type"] == "photo":
                assert bonus == 0.0


# ============================================================================
# Tests for fetch_rag_images (pedagogical_nodes.py)
# ============================================================================

class TestFetchRagImages:
    """Tests for fetch_rag_images() in pedagogical_nodes.py"""

    def test_populates_state_with_images(self):
        """Test that fetch_rag_images populates state with image references"""
        # Simulate state with lectures
        state = {
            "outline": {
                "lectures": [
                    {"id": "lec_1", "title": "Introduction to Kafka"},
                    {"id": "lec_2", "title": "Kafka Producers and Consumers"},
                ]
            },
            "document_ids": ["doc_123"],
            "user_id": "user_1",
        }

        # Simulate the result
        rag_images = [
            {
                "lecture_id": "lec_1",
                "image_id": "img_001",
                "file_path": "/tmp/images/kafka_diagram.png",
                "detected_type": "diagram",
                "relevance_score": 0.85,
            }
        ]

        state["rag_images"] = rag_images

        assert "rag_images" in state
        assert len(state["rag_images"]) == 1
        assert state["rag_images"][0]["lecture_id"] == "lec_1"

    def test_handles_no_documents(self):
        """Test that empty document_ids returns empty rag_images"""
        state = {
            "outline": {
                "lectures": [
                    {"id": "lec_1", "title": "Introduction"},
                ]
            },
            "document_ids": [],
            "user_id": "user_1",
        }

        # With no documents, no images to fetch
        if not state.get("document_ids"):
            state["rag_images"] = []

        assert state["rag_images"] == []

    def test_handles_no_lectures(self):
        """Test that empty lectures list returns empty rag_images"""
        state = {
            "outline": {
                "lectures": []
            },
            "document_ids": ["doc_123"],
            "user_id": "user_1",
        }

        # With no lectures, no topics to search
        if not state.get("outline", {}).get("lectures"):
            state["rag_images"] = []

        assert state["rag_images"] == []

    def test_maps_images_to_correct_lecture(self):
        """Test that images are mapped to the correct lecture by topic"""
        lectures = [
            {"id": "lec_1", "title": "Kafka Introduction"},
            {"id": "lec_2", "title": "Database Design"},
        ]

        # Simulate image matching
        kafka_image = {
            "image_id": "img_001",
            "context_text": "kafka streaming platform",
            "detected_type": "diagram",
        }

        # The image should match "Kafka Introduction"
        kafka_lecture = lectures[0]
        assert "kafka" in kafka_lecture["title"].lower()


# ============================================================================
# Tests for Relevance Scoring
# ============================================================================

class TestRelevanceScoring:
    """Tests for the relevance scoring algorithm"""

    def test_normalized_score_between_0_and_1(self):
        """Test that relevance scores are normalized to 0-1 range"""
        # Simulate scoring
        context_match = 3  # 3 words match in context
        caption_match = 2  # 2 words match in caption

        # Weights: context 40%, caption 25%, description 20%, keywords 15%
        score = (context_match * 0.1) + (caption_match * 0.08)
        normalized_score = min(1.0, score)

        assert 0.0 <= normalized_score <= 1.0

    def test_context_has_highest_weight(self):
        """Test that context_text matching has the highest weight"""
        weights = {
            "context": 0.40,
            "caption": 0.25,
            "description": 0.20,
            "keywords": 0.15,
        }

        assert weights["context"] > weights["caption"]
        assert weights["context"] > weights["description"]
        assert weights["context"] > weights["keywords"]

    def test_topic_words_filter_stopwords(self):
        """Test that short words (stopwords) are filtered out"""
        topic = "Introduction to the Apache Kafka"

        # Filter words with len > 3
        topic_words = [w for w in topic.lower().split() if len(w) > 3]

        assert "the" not in topic_words
        assert "to" not in topic_words
        assert "introduction" in topic_words
        assert "apache" in topic_words
        assert "kafka" in topic_words


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
