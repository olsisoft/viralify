"""
Document Structure Extractor

Extracts document structure (headings, TOC, chapters) from various document types.
Supports PDF, DOCX, YouTube videos, and plain text.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class HeadingInfo:
    """Information about a detected heading."""
    text: str
    level: int  # 1 = main heading, 2 = subheading, etc.
    start_time: Optional[float] = None  # For video chapters
    page_number: Optional[int] = None


@dataclass
class DocumentStructure:
    """Extracted structure from a document."""
    headings: List[HeadingInfo] = field(default_factory=list)
    has_toc: bool = False
    is_youtube: bool = False
    title: Optional[str] = None
    summary: Optional[str] = None
    page_count: int = 0
    word_count: int = 0
    duration_seconds: int = 0  # For videos


class StructureExtractor:
    """
    Extract document structure from various sources.

    Handles:
    - PDF/DOCX with explicit headings
    - Markdown with # headers
    - Numbered sections (1. Introduction, 1.1 Overview)
    - YouTube videos with chapters
    - Plain text with ALL CAPS section titles

    Usage:
        extractor = StructureExtractor()
        structure = extractor.extract(document)
        formatted = extractor.format_structure(structure)
    """

    # Patterns for detecting section headers
    MARKDOWN_HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    NUMBERED_SECTION_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\s+([A-Z].+)$', re.MULTILINE)
    ALL_CAPS_PATTERN = re.compile(r'^([A-Z][A-Z\s]{4,78})$', re.MULTILINE)

    def __init__(self, max_headings: int = 50):
        """
        Initialize the structure extractor.

        Args:
            max_headings: Maximum number of headings to extract
        """
        self.max_headings = max_headings

    def extract(
        self,
        document,  # Document object
    ) -> DocumentStructure:
        """
        Extract structure from a document.

        Args:
            document: Document object with raw_content and metadata

        Returns:
            DocumentStructure with extracted headings
        """
        raw_content = getattr(document, 'raw_content', '') or ''
        metadata = getattr(document, 'extracted_metadata', {}) or {}
        doc_type = getattr(document, 'document_type', None)
        source_url = getattr(document, 'source_url', '')

        # Check if YouTube video
        is_youtube = self._is_youtube(source_url, doc_type)

        structure = DocumentStructure(
            is_youtube=is_youtube,
            title=metadata.get('title', getattr(document, 'filename', '')),
            summary=getattr(document, 'content_summary', ''),
            page_count=getattr(document, 'page_count', 0) or 0,
            word_count=getattr(document, 'word_count', 0) or len(raw_content.split()),
            duration_seconds=metadata.get('duration_seconds', 0),
        )

        # First check for explicit headings in metadata
        if 'headings' in metadata and metadata['headings']:
            structure.has_toc = True
            structure.headings = self._parse_metadata_headings(metadata['headings'], is_youtube)
        else:
            # Try to detect headings from content
            structure.headings = self._detect_headings_from_content(raw_content)

        return structure

    def _is_youtube(self, source_url: str, doc_type) -> bool:
        """Check if document is a YouTube video."""
        if not source_url:
            return False
        doc_type_str = doc_type.value if hasattr(doc_type, 'value') else str(doc_type or '')
        return (
            doc_type_str == 'url' and
            ('youtube.com' in source_url or 'youtu.be' in source_url)
        )

    def _parse_metadata_headings(
        self,
        headings: List[Dict],
        is_youtube: bool,
    ) -> List[HeadingInfo]:
        """Parse headings from document metadata."""
        result = []
        for heading in headings[:self.max_headings]:
            text = heading.get('text', '').strip()
            if not text:
                continue

            result.append(HeadingInfo(
                text=text,
                level=heading.get('level', 1),
                start_time=heading.get('start_time') if is_youtube else None,
                page_number=heading.get('page_number'),
            ))
        return result

    def _detect_headings_from_content(
        self,
        content: str,
    ) -> List[HeadingInfo]:
        """Detect headings from raw content using patterns."""
        headings = []

        # Limit to first portion of content for efficiency
        content_sample = content[:10000]

        # 1. Try markdown headers
        md_headings = self._detect_markdown_headers(content_sample)
        if md_headings:
            return md_headings[:self.max_headings]

        # 2. Try numbered sections
        numbered_headings = self._detect_numbered_sections(content_sample)
        if numbered_headings:
            return numbered_headings[:self.max_headings]

        # 3. Try ALL CAPS lines
        caps_headings = self._detect_all_caps_sections(content_sample)
        if caps_headings:
            return caps_headings[:self.max_headings]

        return []

    def _detect_markdown_headers(self, content: str) -> List[HeadingInfo]:
        """Detect markdown-style headers (# Header)."""
        headings = []
        for match in self.MARKDOWN_HEADER_PATTERN.finditer(content):
            hashes = match.group(1)
            text = match.group(2).strip()
            if text and len(text) > 2:
                headings.append(HeadingInfo(
                    text=text,
                    level=len(hashes),
                ))
        return headings

    def _detect_numbered_sections(self, content: str) -> List[HeadingInfo]:
        """Detect numbered sections (1. Introduction, 1.1 Overview)."""
        headings = []
        lines = content.split('\n')

        for line in lines[:200]:  # Check first 200 lines
            line = line.strip()
            match = re.match(r'^(\d+(?:\.\d+)*)\s+([A-Z].{2,})$', line)
            if match:
                number = match.group(1)
                text = match.group(2).strip()
                if len(text) > 3:
                    level = number.count('.') + 1
                    headings.append(HeadingInfo(
                        text=text,
                        level=level,
                    ))

        return headings

    def _detect_all_caps_sections(self, content: str) -> List[HeadingInfo]:
        """Detect ALL CAPS section titles."""
        headings = []
        lines = content.split('\n')

        for line in lines[:200]:
            line = line.strip()
            if line.isupper() and 5 < len(line) < 80 and ' ' in line:
                # Likely a section title
                headings.append(HeadingInfo(
                    text=line.title(),  # Convert to title case
                    level=1,
                ))

        return headings

    def format_structure(
        self,
        structure: DocumentStructure,
        include_stats: bool = True,
    ) -> str:
        """
        Format extracted structure for LLM consumption.

        Args:
            structure: Extracted document structure
            include_stats: Include page/word count statistics

        Returns:
            Formatted structure string
        """
        parts = []

        # Header based on document type
        if structure.is_youtube:
            parts.append(f"\n🎬 VIDEO YOUTUBE: {structure.title}")
            if structure.duration_seconds:
                minutes = structure.duration_seconds // 60
                parts.append(f"   DURÉE: {minutes} minutes")
        else:
            parts.append(f"\n📄 DOCUMENT: {structure.title}")

        # Headings
        if structure.headings:
            if structure.is_youtube and structure.has_toc:
                parts.append("   CHAPITRES YOUTUBE (structure obligatoire):")
            elif structure.has_toc:
                parts.append("   TABLE OF CONTENTS:")
            else:
                parts.append("   DETECTED SECTIONS:")

            for heading in structure.headings[:20]:  # Limit display
                if structure.is_youtube and heading.start_time is not None:
                    mins = int(heading.start_time // 60)
                    secs = int(heading.start_time % 60)
                    timestamp = f"{mins:02d}:{secs:02d}"
                    parts.append(f"   ┌── [{timestamp}] {heading.text}")
                else:
                    indent = "   " * heading.level
                    prefix = "├──" if heading.level > 1 else "┌──"
                    parts.append(f"   {indent}{prefix} {heading.text}")

        # Summary
        if structure.summary:
            parts.append(f"\n   SUMMARY: {structure.summary}")

        # Statistics
        if include_stats:
            if structure.is_youtube:
                parts.append(f"   STATS: {structure.word_count} mots dans la transcription")
            else:
                parts.append(f"   STATS: {structure.page_count} pages, {structure.word_count} words")

        return "\n".join(parts)

    def format_multiple_structures(
        self,
        structures: List[Tuple[str, DocumentStructure]],  # (doc_id, structure)
    ) -> str:
        """
        Format multiple document structures with header.

        Args:
            structures: List of (document_id, structure) tuples

        Returns:
            Combined formatted structure string
        """
        if not structures:
            return ""

        parts = []

        # Header
        header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           DOCUMENT STRUCTURE - YOUR COURSE MUST FOLLOW THIS OUTLINE          ║
╚══════════════════════════════════════════════════════════════════════════════╝

The following is the EXACT structure extracted from the source documents.
Your course sections and lectures MUST map directly to this structure.

DO NOT invent new topics. DO NOT reorganize. FOLLOW THIS STRUCTURE.
"""
        parts.append(header)

        # Each document's structure
        for doc_id, structure in structures:
            formatted = self.format_structure(structure)
            parts.append(formatted)

        return "\n".join(parts)


# Module-level instance for convenience
_default_extractor = None


def get_structure_extractor() -> StructureExtractor:
    """Get the default structure extractor instance."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = StructureExtractor()
    return _default_extractor


def extract_structure(document) -> DocumentStructure:
    """
    Convenience function to extract structure from a document.

    Args:
        document: Document object

    Returns:
        DocumentStructure
    """
    return get_structure_extractor().extract(document)
