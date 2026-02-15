"""
Unit Tests for StructureExtractor

Tests document structure extraction from various sources.
"""

import pytest
from dataclasses import dataclass
from typing import Optional, Dict, Any
from ..processors.structure_extractor import (
    StructureExtractor,
    DocumentStructure,
    HeadingInfo,
    extract_structure,
    get_structure_extractor,
)


@dataclass
class MockDocument:
    """Mock document for testing."""
    raw_content: str = ""
    extracted_metadata: Dict[str, Any] = None
    document_type: Optional[str] = None
    source_url: str = ""
    filename: str = "test.pdf"
    content_summary: str = ""
    page_count: int = 0
    word_count: int = 0

    def __post_init__(self):
        if self.extracted_metadata is None:
            self.extracted_metadata = {}


class TestHeadingInfo:
    """Tests for HeadingInfo dataclass."""

    def test_create_basic(self):
        """Test creating basic heading."""
        heading = HeadingInfo(
            text="Introduction",
            level=1,
        )

        assert heading.text == "Introduction"
        assert heading.level == 1
        assert heading.start_time is None
        assert heading.page_number is None

    def test_create_youtube_heading(self):
        """Test creating YouTube chapter heading."""
        heading = HeadingInfo(
            text="Kafka Basics",
            level=1,
            start_time=120.5,
        )

        assert heading.start_time == 120.5

    def test_create_pdf_heading(self):
        """Test creating PDF heading with page number."""
        heading = HeadingInfo(
            text="Chapter 1",
            level=1,
            page_number=5,
        )

        assert heading.page_number == 5


class TestDocumentStructure:
    """Tests for DocumentStructure dataclass."""

    def test_create_empty(self):
        """Test creating empty structure."""
        structure = DocumentStructure()

        assert structure.headings == []
        assert structure.has_toc is False
        assert structure.is_youtube is False

    def test_create_with_headings(self):
        """Test creating structure with headings."""
        headings = [
            HeadingInfo(text="Intro", level=1),
            HeadingInfo(text="Details", level=2),
        ]
        structure = DocumentStructure(
            headings=headings,
            has_toc=True,
        )

        assert len(structure.headings) == 2
        assert structure.has_toc is True


class TestStructureExtractor:
    """Tests for StructureExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create default extractor."""
        return StructureExtractor()

    # ==========================================================================
    # Metadata Headings Tests
    # ==========================================================================

    def test_extract_from_metadata(self, extractor):
        """Test extraction from metadata headings."""
        doc = MockDocument(
            raw_content="Some content",
            extracted_metadata={
                "headings": [
                    {"text": "Chapter 1", "level": 1},
                    {"text": "Section 1.1", "level": 2},
                ],
            },
        )

        structure = extractor.extract(doc)

        assert len(structure.headings) == 2
        assert structure.has_toc is True
        assert structure.headings[0].text == "Chapter 1"
        assert structure.headings[1].level == 2

    def test_extract_youtube_chapters(self, extractor):
        """Test extraction of YouTube chapters from metadata."""
        doc = MockDocument(
            raw_content="Transcript content",
            source_url="https://youtube.com/watch?v=123",
            extracted_metadata={
                "title": "Kafka Tutorial",
                "duration_seconds": 1800,
                "headings": [
                    {"text": "Introduction", "level": 1, "start_time": 0},
                    {"text": "Setup", "level": 1, "start_time": 120},
                    {"text": "Demo", "level": 1, "start_time": 600},
                ],
            },
        )
        doc.document_type = type("DT", (), {"value": "url"})()

        structure = extractor.extract(doc)

        assert structure.is_youtube is True
        assert len(structure.headings) == 3
        assert structure.headings[1].start_time == 120

    # ==========================================================================
    # Markdown Header Detection Tests
    # ==========================================================================

    def test_detect_markdown_headers(self, extractor):
        """Test detection of markdown headers."""
        doc = MockDocument(
            raw_content="""
# Introduction to Kafka

Some intro text.

## What is Kafka

Kafka is a streaming platform.

### Key Features

Features include...

## Architecture

The architecture consists of...
""",
        )

        structure = extractor.extract(doc)

        assert len(structure.headings) >= 4
        assert any(h.text == "Introduction to Kafka" for h in structure.headings)
        assert any(h.text == "What is Kafka" for h in structure.headings)

    def test_markdown_header_levels(self, extractor):
        """Test that markdown header levels are correct."""
        doc = MockDocument(
            raw_content="""
# Level 1
## Level 2
### Level 3
""",
        )

        structure = extractor.extract(doc)

        levels = [h.level for h in structure.headings]
        assert 1 in levels
        assert 2 in levels
        assert 3 in levels

    # ==========================================================================
    # Numbered Section Detection Tests
    # ==========================================================================

    def test_detect_numbered_sections(self, extractor):
        """Test detection of numbered sections."""
        doc = MockDocument(
            raw_content="""
1. Introduction
This is the intro.

2. Background
Background info here.

2.1 History
Historical context.

3. Conclusion
Final thoughts.
""",
        )

        structure = extractor.extract(doc)

        assert len(structure.headings) >= 3
        # Check levels from numbering
        intro = next((h for h in structure.headings if "Introduction" in h.text), None)
        assert intro is not None

    # ==========================================================================
    # ALL CAPS Detection Tests
    # ==========================================================================

    def test_detect_all_caps_sections(self, extractor):
        """Test detection of ALL CAPS section titles."""
        doc = MockDocument(
            raw_content="""
INTRODUCTION TO THE SYSTEM

This is the introduction.

SYSTEM ARCHITECTURE

Details about the architecture.

DEPLOYMENT GUIDE

How to deploy.
""",
        )

        structure = extractor.extract(doc)

        # Should detect caps lines and convert to title case
        assert len(structure.headings) >= 3
        texts = [h.text.lower() for h in structure.headings]
        assert any("introduction" in t for t in texts)
        assert any("architecture" in t for t in texts)

    # ==========================================================================
    # Edge Cases Tests
    # ==========================================================================

    def test_extract_empty_content(self, extractor):
        """Test extraction from empty content."""
        doc = MockDocument(raw_content="")

        structure = extractor.extract(doc)

        assert structure.headings == []

    def test_extract_no_structure(self, extractor):
        """Test extraction from unstructured content."""
        doc = MockDocument(
            raw_content="Just some plain text without any structure or headings whatsoever."
        )

        structure = extractor.extract(doc)

        assert structure.headings == []

    def test_max_headings_limit(self):
        """Test that max_headings limit is respected."""
        extractor = StructureExtractor(max_headings=3)

        doc = MockDocument(
            raw_content="""
# One
## Two
### Three
#### Four
##### Five
""",
        )

        structure = extractor.extract(doc)

        assert len(structure.headings) <= 3

    # ==========================================================================
    # Format Structure Tests
    # ==========================================================================

    def test_format_structure_pdf(self, extractor):
        """Test formatting structure for PDF."""
        structure = DocumentStructure(
            headings=[
                HeadingInfo(text="Chapter 1", level=1),
                HeadingInfo(text="Section 1.1", level=2),
            ],
            has_toc=True,
            title="Test Document",
            page_count=10,
            word_count=5000,
        )

        formatted = extractor.format_structure(structure)

        assert "📄" in formatted
        assert "Test Document" in formatted
        assert "Chapter 1" in formatted
        assert "┌──" in formatted or "├──" in formatted
        assert "10 pages" in formatted

    def test_format_structure_youtube(self, extractor):
        """Test formatting structure for YouTube."""
        structure = DocumentStructure(
            headings=[
                HeadingInfo(text="Introduction", level=1, start_time=0),
                HeadingInfo(text="Main Content", level=1, start_time=120),
            ],
            is_youtube=True,
            title="Kafka Tutorial",
            duration_seconds=1800,
            word_count=3000,
        )

        formatted = extractor.format_structure(structure)

        assert "🎬" in formatted
        assert "Kafka Tutorial" in formatted
        assert "[00:00]" in formatted
        assert "[02:00]" in formatted
        assert "mots" in formatted  # French word count

    def test_format_structure_with_summary(self, extractor):
        """Test formatting includes summary."""
        structure = DocumentStructure(
            headings=[HeadingInfo(text="Test", level=1)],
            summary="This is a test summary.",
        )

        formatted = extractor.format_structure(structure)

        assert "SUMMARY" in formatted
        assert "This is a test summary" in formatted

    def test_format_structure_no_stats(self, extractor):
        """Test formatting without stats."""
        structure = DocumentStructure(
            headings=[HeadingInfo(text="Test", level=1)],
            title="Test Doc",
        )

        formatted = extractor.format_structure(structure, include_stats=False)

        assert "pages" not in formatted
        assert "words" not in formatted

    # ==========================================================================
    # Multiple Structures Tests
    # ==========================================================================

    def test_format_multiple_structures(self, extractor):
        """Test formatting multiple document structures."""
        structures = [
            ("doc1", DocumentStructure(
                headings=[HeadingInfo(text="Doc 1 Chapter", level=1)],
                title="Document 1",
            )),
            ("doc2", DocumentStructure(
                headings=[HeadingInfo(text="Doc 2 Chapter", level=1)],
                title="Document 2",
            )),
        ]

        formatted = extractor.format_multiple_structures(structures)

        assert "Document 1" in formatted
        assert "Document 2" in formatted
        assert "MUST FOLLOW" in formatted  # Header instruction

    def test_format_multiple_empty(self, extractor):
        """Test formatting empty structures list."""
        formatted = extractor.format_multiple_structures([])
        assert formatted == ""

    # ==========================================================================
    # Module-level Functions Tests
    # ==========================================================================

    def test_extract_structure_function(self):
        """Test module-level extract_structure function."""
        doc = MockDocument(
            raw_content="# Test Header\nSome content",
        )

        structure = extract_structure(doc)

        assert isinstance(structure, DocumentStructure)

    def test_get_structure_extractor_singleton(self):
        """Test singleton behavior."""
        ext1 = get_structure_extractor()
        ext2 = get_structure_extractor()
        assert ext1 is ext2


class TestStructureExtractorYouTube:
    """Tests specifically for YouTube transcript handling."""

    @pytest.fixture
    def extractor(self):
        return StructureExtractor()

    def test_youtube_detection_youtube_com(self, extractor):
        """Test YouTube detection for youtube.com URLs."""
        doc = MockDocument(
            source_url="https://www.youtube.com/watch?v=abc123",
        )
        doc.document_type = type("DT", (), {"value": "url"})()

        structure = extractor.extract(doc)

        assert structure.is_youtube is True

    def test_youtube_detection_youtu_be(self, extractor):
        """Test YouTube detection for youtu.be URLs."""
        doc = MockDocument(
            source_url="https://youtu.be/abc123",
        )
        doc.document_type = type("DT", (), {"value": "url"})()

        structure = extractor.extract(doc)

        assert structure.is_youtube is True

    def test_youtube_detection_non_youtube_url(self, extractor):
        """Test that non-YouTube URLs are not marked as YouTube."""
        doc = MockDocument(
            source_url="https://example.com/video",
        )
        doc.document_type = type("DT", (), {"value": "url"})()

        structure = extractor.extract(doc)

        assert structure.is_youtube is False
