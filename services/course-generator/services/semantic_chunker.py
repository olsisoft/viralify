"""
Semantic Chunking Service

Advanced chunking strategy for RAG that:
1. Uses token-based chunking (not characters)
2. Respects document structure (headers, sections)
3. Associates images with their context
4. Enriches metadata for better LLM understanding
5. Adapts strategy based on content type (PDF, video, URL)
"""
import re
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import tiktoken


class ContentType(Enum):
    """Type of content being chunked"""
    TEXT = "text"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    IMAGE_CONTEXT = "image_context"
    VIDEO_SEGMENT = "video_segment"
    DEFINITION = "definition"
    EXAMPLE = "example"
    KEY_POINT = "key_point"


@dataclass
class SemanticChunk:
    """
    A semantically meaningful chunk of content with rich metadata.
    Designed to give the LLM maximum context about the content.
    """
    # Core content
    content: str
    chunk_index: int

    # Token information
    token_count: int

    # Document context
    document_id: str = ""
    document_name: str = ""
    document_type: str = ""  # pdf, video, url, etc.

    # Position in document
    page_number: Optional[int] = None
    section_hierarchy: List[str] = field(default_factory=list)  # ["Chapter 1", "Section 1.2", "Subsection"]
    section_title: Optional[str] = None
    position_percent: float = 0.0  # 0-100, position in document

    # Content classification
    content_type: ContentType = ContentType.TEXT
    is_key_content: bool = False  # Definitions, important concepts, etc.

    # Related media
    associated_images: List[Dict] = field(default_factory=list)  # Images in this chunk's context

    # For video content
    timestamp_start: Optional[float] = None  # seconds
    timestamp_end: Optional[float] = None

    # Semantic markers found in content
    contains_definition: bool = False
    contains_example: bool = False
    contains_code: bool = False
    contains_list: bool = False
    contains_table: bool = False

    # Keywords extracted from this chunk
    keywords: List[str] = field(default_factory=list)

    # For LLM context - a summary prompt of what this chunk contains
    context_hint: str = ""

    # Embedding (set during vectorization)
    embedding: Optional[List[float]] = None

    def to_prompt_format(self) -> str:
        """
        Format this chunk for inclusion in LLM prompts.
        Includes rich context to help the LLM understand the content.
        """
        lines = []

        # Header with source information
        source_info = f"[SOURCE: {self.document_name}"
        if self.page_number:
            source_info += f" | Page {self.page_number}"
        if self.section_title:
            source_info += f" | Section: {self.section_title}"
        if self.timestamp_start is not None:
            mins = int(self.timestamp_start // 60)
            secs = int(self.timestamp_start % 60)
            source_info += f" | Timestamp: {mins}:{secs:02d}"
        source_info += "]"
        lines.append(source_info)

        # Content type indicator
        if self.content_type != ContentType.TEXT:
            lines.append(f"[Content Type: {self.content_type.value}]")

        # Semantic markers
        markers = []
        if self.contains_definition:
            markers.append("DEFINITION")
        if self.contains_example:
            markers.append("EXAMPLE")
        if self.contains_code:
            markers.append("CODE")
        if self.is_key_content:
            markers.append("KEY CONCEPT")
        if markers:
            lines.append(f"[Contains: {', '.join(markers)}]")

        # The actual content
        lines.append("")
        lines.append(self.content)

        # Associated images description
        if self.associated_images:
            lines.append("")
            lines.append("[ASSOCIATED VISUALS:]")
            for img in self.associated_images:
                img_desc = f"  - Image: {img.get('description', 'No description')}"
                if img.get('detected_type'):
                    img_desc += f" (Type: {img['detected_type']})"
                lines.append(img_desc)

        return "\n".join(lines)


class SemanticChunker:
    """
    Advanced semantic chunking service.

    Features:
    - Token-based chunking using tiktoken
    - Respects document structure (headers, sections)
    - Associates images with their text context
    - Classifies content types (definitions, examples, code, etc.)
    - Adapts strategy based on document type
    """

    # Chunk size configuration (in tokens)
    DEFAULT_CHUNK_SIZE = 600  # tokens - optimal for embeddings
    DEFAULT_CHUNK_OVERLAP = 100  # tokens
    MAX_CHUNK_SIZE = 1000  # Never exceed this
    MIN_CHUNK_SIZE = 100  # Don't create tiny chunks

    # Patterns for semantic detection
    HEADING_PATTERNS = [
        r'^#{1,6}\s+(.+)$',  # Markdown headers
        r'^(.+)\n[=]+$',  # Underlined headers (=)
        r'^(.+)\n[-]+$',  # Underlined headers (-)
        r'^\d+\.\s+(.+)$',  # Numbered sections
        r'^[A-Z][A-Z\s]+:?\s*$',  # ALL CAPS HEADERS
        r'^Chapter\s+\d+[:\s]+(.+)$',  # Chapter headers
        r'^Section\s+\d+[.\d]*[:\s]+(.+)$',  # Section headers
    ]

    DEFINITION_PATTERNS = [
        r'(?:is defined as|means|refers to|is called|is known as)',
        r'(?:Definition:|Définition:)',
        r'(?:^|\n)\s*(?:•|►|▸)\s*\*\*[^*]+\*\*\s*[:\-]',  # Bold term followed by definition
    ]

    EXAMPLE_PATTERNS = [
        r'(?:for example|e\.g\.|such as|for instance|consider)',
        r'(?:Example:|Exemple:|Ex:)',
        r'(?:Let\'s say|Suppose|Imagine)',
    ]

    CODE_PATTERNS = [
        r'```[\s\S]*?```',  # Markdown code blocks
        r'`[^`]+`',  # Inline code
        r'(?:def |class |function |import |from |const |let |var )',  # Code keywords
        r'(?:\(\)\s*{|\(\)\s*=>)',  # Function syntax
    ]

    KEY_CONTENT_PATTERNS = [
        r'(?:Important:|Note:|Warning:|Attention:|Key point:|Remember:)',
        r'(?:IMPORTANT|NOTE|WARNING|KEY)',
        r'(?:must|should|always|never)\s+(?:be|have|use)',
    ]

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = min(chunk_size, self.MAX_CHUNK_SIZE)
        self.chunk_overlap = chunk_overlap

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        print(f"[CHUNKER] Initialized: {self.chunk_size} tokens/chunk, {self.chunk_overlap} overlap", flush=True)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def chunk_document(
        self,
        text: str,
        document_id: str,
        document_name: str,
        document_type: str,
        metadata: Dict[str, Any] = None,
        images: List[Dict] = None,
    ) -> List[SemanticChunk]:
        """
        Chunk a document using semantic-aware strategy.

        Args:
            text: Raw document text
            document_id: Document ID
            document_name: Document filename
            document_type: Type (pdf, video, url, etc.)
            metadata: Document metadata (page_breaks, sections, etc.)
            images: List of extracted images with their context

        Returns:
            List of SemanticChunk objects
        """
        print(f"[CHUNKER] Processing: {document_name} ({document_type})", flush=True)

        metadata = metadata or {}
        images = images or []

        # Choose strategy based on document type
        if document_type in ['youtube', 'video']:
            chunks = self._chunk_video_transcript(text, metadata)
        elif document_type == 'url':
            chunks = self._chunk_web_content(text, metadata)
        elif document_type in ['pdf', 'docx', 'pptx']:
            chunks = self._chunk_structured_document(text, metadata)
        else:
            chunks = self._chunk_generic_text(text, metadata)

        # Enrich all chunks with document info and images
        total_tokens = sum(c.token_count for c in chunks)

        for i, chunk in enumerate(chunks):
            chunk.document_id = document_id
            chunk.document_name = document_name
            chunk.document_type = document_type
            chunk.chunk_index = i
            chunk.position_percent = (i / len(chunks)) * 100 if chunks else 0

            # Associate images with chunks based on context
            chunk.associated_images = self._find_associated_images(chunk, images)

            # Classify content and detect semantic markers
            self._classify_chunk_content(chunk)

            # Generate context hint for LLM
            chunk.context_hint = self._generate_context_hint(chunk)

        print(f"[CHUNKER] Created {len(chunks)} chunks, {total_tokens} total tokens", flush=True)

        return chunks

    def _chunk_structured_document(
        self,
        text: str,
        metadata: Dict,
    ) -> List[SemanticChunk]:
        """
        Chunk structured documents (PDF, DOCX, PPTX).
        Respects section hierarchy and page breaks.
        """
        chunks = []

        # Extract section structure if available
        sections = self._extract_sections(text, metadata)

        if sections:
            # Chunk by sections first
            for section in sections:
                section_chunks = self._chunk_section(
                    section['content'],
                    section['title'],
                    section['hierarchy'],
                    section.get('page_number'),
                )
                chunks.extend(section_chunks)
        else:
            # Fall back to paragraph-based chunking with header detection
            chunks = self._chunk_by_paragraphs_with_headers(text, metadata)

        return chunks

    def _chunk_video_transcript(
        self,
        text: str,
        metadata: Dict,
    ) -> List[SemanticChunk]:
        """
        Chunk video transcripts.
        Preserves timestamps and creates topic-based segments.
        """
        chunks = []

        # Check if we have timestamped segments
        timestamps = metadata.get('timestamps', [])

        if timestamps:
            # Chunk by timestamp segments, grouping related content
            current_chunk_text = ""
            current_start = 0
            chunk_timestamps = []

            for ts in timestamps:
                segment_text = ts.get('text', '')
                segment_start = ts.get('start', 0)

                potential_text = current_chunk_text + " " + segment_text
                potential_tokens = self.count_tokens(potential_text)

                if potential_tokens > self.chunk_size and current_chunk_text:
                    # Save current chunk
                    chunk = SemanticChunk(
                        content=current_chunk_text.strip(),
                        chunk_index=len(chunks),
                        token_count=self.count_tokens(current_chunk_text),
                        content_type=ContentType.VIDEO_SEGMENT,
                        timestamp_start=current_start,
                        timestamp_end=segment_start,
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap_text = self._get_token_overlap(current_chunk_text)
                    current_chunk_text = overlap_text + " " + segment_text
                    current_start = segment_start
                else:
                    current_chunk_text = potential_text

            # Don't forget last chunk
            if current_chunk_text.strip():
                chunk = SemanticChunk(
                    content=current_chunk_text.strip(),
                    chunk_index=len(chunks),
                    token_count=self.count_tokens(current_chunk_text),
                    content_type=ContentType.VIDEO_SEGMENT,
                    timestamp_start=current_start,
                    timestamp_end=metadata.get('duration_seconds', current_start),
                )
                chunks.append(chunk)
        else:
            # No timestamps, chunk as regular text but mark as video
            chunks = self._chunk_by_paragraphs_with_headers(text, metadata)
            for chunk in chunks:
                chunk.content_type = ContentType.VIDEO_SEGMENT

        return chunks

    def _chunk_web_content(
        self,
        text: str,
        metadata: Dict,
    ) -> List[SemanticChunk]:
        """
        Chunk web content.
        Preserves article structure and extracts key sections.
        """
        # Web content often has clear sections, use header-based chunking
        return self._chunk_by_paragraphs_with_headers(text, metadata)

    def _chunk_generic_text(
        self,
        text: str,
        metadata: Dict,
    ) -> List[SemanticChunk]:
        """
        Generic text chunking with semantic awareness.
        """
        return self._chunk_by_paragraphs_with_headers(text, metadata)

    def _chunk_by_paragraphs_with_headers(
        self,
        text: str,
        metadata: Dict,
    ) -> List[SemanticChunk]:
        """
        Chunk text by paragraphs while respecting headers.
        Headers start new chunks to preserve context.
        """
        chunks = []

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk_text = ""
        current_section = None
        section_hierarchy = []
        current_page = 1

        page_breaks = metadata.get('page_breaks', {})

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this is a header
            header_match = self._detect_header(para)

            if header_match:
                # Save current chunk before starting new section
                if current_chunk_text.strip():
                    chunk = SemanticChunk(
                        content=current_chunk_text.strip(),
                        chunk_index=len(chunks),
                        token_count=self.count_tokens(current_chunk_text),
                        section_title=current_section,
                        section_hierarchy=section_hierarchy.copy(),
                        page_number=current_page,
                    )
                    chunks.append(chunk)
                    current_chunk_text = ""

                # Update section tracking
                current_section = header_match
                section_hierarchy = self._update_hierarchy(section_hierarchy, header_match, para)

                # Start new chunk with header
                current_chunk_text = para
            else:
                # Check if adding this paragraph exceeds chunk size
                potential_text = current_chunk_text + "\n\n" + para if current_chunk_text else para
                potential_tokens = self.count_tokens(potential_text)

                if potential_tokens > self.chunk_size:
                    # Save current chunk
                    if current_chunk_text.strip():
                        chunk = SemanticChunk(
                            content=current_chunk_text.strip(),
                            chunk_index=len(chunks),
                            token_count=self.count_tokens(current_chunk_text),
                            section_title=current_section,
                            section_hierarchy=section_hierarchy.copy(),
                            page_number=current_page,
                        )
                        chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap = self._get_token_overlap(current_chunk_text)
                    current_chunk_text = overlap + "\n\n" + para if overlap else para
                else:
                    current_chunk_text = potential_text

            # Update page number from metadata
            char_pos = text.find(para)
            for pos, page in page_breaks.items():
                if char_pos >= int(pos):
                    current_page = page

        # Don't forget last chunk
        if current_chunk_text.strip():
            chunk = SemanticChunk(
                content=current_chunk_text.strip(),
                chunk_index=len(chunks),
                token_count=self.count_tokens(current_chunk_text),
                section_title=current_section,
                section_hierarchy=section_hierarchy.copy(),
                page_number=current_page,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_section(
        self,
        content: str,
        title: str,
        hierarchy: List[str],
        page_number: Optional[int],
    ) -> List[SemanticChunk]:
        """Chunk a single section, respecting token limits."""
        chunks = []

        tokens = self.count_tokens(content)

        if tokens <= self.chunk_size:
            # Section fits in one chunk
            chunk = SemanticChunk(
                content=content,
                chunk_index=0,
                token_count=tokens,
                section_title=title,
                section_hierarchy=hierarchy,
                page_number=page_number,
            )
            chunks.append(chunk)
        else:
            # Need to split section
            paragraphs = re.split(r'\n\s*\n', content)
            current_text = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                potential = current_text + "\n\n" + para if current_text else para

                if self.count_tokens(potential) > self.chunk_size and current_text:
                    chunk = SemanticChunk(
                        content=current_text.strip(),
                        chunk_index=len(chunks),
                        token_count=self.count_tokens(current_text),
                        section_title=title,
                        section_hierarchy=hierarchy,
                        page_number=page_number,
                    )
                    chunks.append(chunk)

                    overlap = self._get_token_overlap(current_text)
                    current_text = overlap + "\n\n" + para if overlap else para
                else:
                    current_text = potential

            if current_text.strip():
                chunk = SemanticChunk(
                    content=current_text.strip(),
                    chunk_index=len(chunks),
                    token_count=self.count_tokens(current_text),
                    section_title=title,
                    section_hierarchy=hierarchy,
                    page_number=page_number,
                )
                chunks.append(chunk)

        return chunks

    def _extract_sections(self, text: str, metadata: Dict) -> List[Dict]:
        """Extract sections from document based on headers."""
        sections = []

        # Use metadata sections if available (from PDF/DOCX parsing)
        if 'sections' in metadata:
            return metadata['sections']

        # Otherwise, detect sections from headers
        lines = text.split('\n')
        current_section = {'title': 'Introduction', 'hierarchy': [], 'content': '', 'page_number': 1}

        for line in lines:
            header = self._detect_header(line)
            if header:
                # Save previous section
                if current_section['content'].strip():
                    sections.append(current_section)

                # Start new section
                current_section = {
                    'title': header,
                    'hierarchy': [header],
                    'content': line + '\n',
                    'page_number': current_section.get('page_number', 1),
                }
            else:
                current_section['content'] += line + '\n'

        # Don't forget last section
        if current_section['content'].strip():
            sections.append(current_section)

        return sections if len(sections) > 1 else []

    def _detect_header(self, text: str) -> Optional[str]:
        """Detect if text is a header and extract the title."""
        text = text.strip()

        for pattern in self.HEADING_PATTERNS:
            match = re.match(pattern, text, re.MULTILINE)
            if match:
                return match.group(1) if match.groups() else text

        # Check for short lines that look like headers
        if len(text) < 100 and text.endswith(':') and '\n' not in text:
            return text.rstrip(':')

        return None

    def _update_hierarchy(
        self,
        hierarchy: List[str],
        new_header: str,
        raw_text: str,
    ) -> List[str]:
        """Update section hierarchy based on header level."""
        # Detect header level
        level = 1
        if raw_text.startswith('#'):
            level = len(re.match(r'^#+', raw_text).group())
        elif re.match(r'^\d+\.\d+\.', raw_text):
            level = 3
        elif re.match(r'^\d+\.', raw_text):
            level = 2

        # Trim hierarchy to appropriate level and add new header
        new_hierarchy = hierarchy[:level-1]
        new_hierarchy.append(new_header)

        return new_hierarchy

    def _get_token_overlap(self, text: str) -> str:
        """Get overlap text based on tokens."""
        if not text:
            return ""

        tokens = self.tokenizer.encode(text)

        if len(tokens) <= self.chunk_overlap:
            return text

        # Get last N tokens
        overlap_tokens = tokens[-self.chunk_overlap:]
        overlap_text = self.tokenizer.decode(overlap_tokens)

        # Try to start at a sentence boundary
        sentence_match = re.search(r'[.!?]\s+', overlap_text)
        if sentence_match:
            overlap_text = overlap_text[sentence_match.end():]

        return overlap_text.strip()

    def _find_associated_images(
        self,
        chunk: SemanticChunk,
        images: List[Dict],
    ) -> List[Dict]:
        """Find images that are contextually related to this chunk."""
        associated = []

        chunk_text_lower = chunk.content.lower()
        chunk_keywords = set(re.findall(r'\b\w{4,}\b', chunk_text_lower))

        for img in images:
            relevance_score = 0

            # Check page match
            if chunk.page_number and img.get('page_number') == chunk.page_number:
                relevance_score += 0.5

            # Check context text overlap
            if img.get('context_text'):
                img_keywords = set(re.findall(r'\b\w{4,}\b', img['context_text'].lower()))
                overlap = len(chunk_keywords & img_keywords)
                relevance_score += min(overlap * 0.1, 0.5)

            # Check caption overlap
            if img.get('caption'):
                caption_lower = img['caption'].lower()
                if any(kw in caption_lower for kw in chunk_keywords):
                    relevance_score += 0.3

            # Check if image is referenced in text
            if img.get('file_name'):
                img_name = img['file_name'].lower()
                if img_name in chunk_text_lower or any(
                    ref in chunk_text_lower
                    for ref in ['figure', 'image', 'diagram', 'chart', 'illustration']
                ):
                    relevance_score += 0.2

            if relevance_score > 0.3:
                associated.append({
                    'image_id': img.get('id'),
                    'file_path': img.get('file_path'),
                    'description': img.get('description', img.get('caption', '')),
                    'detected_type': img.get('detected_type'),
                    'relevance_score': relevance_score,
                })

        # Sort by relevance and limit
        associated.sort(key=lambda x: x['relevance_score'], reverse=True)
        return associated[:3]  # Max 3 images per chunk

    def _classify_chunk_content(self, chunk: SemanticChunk) -> None:
        """Classify chunk content type and detect semantic markers."""
        content = chunk.content
        content_lower = content.lower()

        # Check for definitions
        for pattern in self.DEFINITION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                chunk.contains_definition = True
                chunk.is_key_content = True
                break

        # Check for examples
        for pattern in self.EXAMPLE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                chunk.contains_example = True
                break

        # Check for code
        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, content):
                chunk.contains_code = True
                chunk.content_type = ContentType.CODE
                break

        # Check for lists
        if re.search(r'(?:^|\n)\s*(?:[-•►▸*]|\d+[.)]\s)', content):
            chunk.contains_list = True
            if not chunk.contains_code:
                chunk.content_type = ContentType.LIST

        # Check for tables
        if '|' in content and content.count('|') > 4:
            chunk.contains_table = True
            chunk.content_type = ContentType.TABLE

        # Check for key content markers
        for pattern in self.KEY_CONTENT_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                chunk.is_key_content = True
                break

        # Extract keywords (simple approach - could use NLP for better results)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        chunk.keywords = list(set(words))[:10]

    def _generate_context_hint(self, chunk: SemanticChunk) -> str:
        """Generate a context hint to help LLM understand this chunk."""
        hints = []

        if chunk.section_title:
            hints.append(f"From section: {chunk.section_title}")

        if chunk.content_type != ContentType.TEXT:
            hints.append(f"Contains: {chunk.content_type.value}")

        if chunk.is_key_content:
            hints.append("Key content")

        if chunk.contains_definition:
            hints.append("Has definitions")

        if chunk.contains_example:
            hints.append("Has examples")

        if chunk.associated_images:
            hints.append(f"{len(chunk.associated_images)} related image(s)")

        if chunk.timestamp_start is not None:
            mins = int(chunk.timestamp_start // 60)
            secs = int(chunk.timestamp_start % 60)
            hints.append(f"Video segment starting at {mins}:{secs:02d}")

        return " | ".join(hints) if hints else "General content"


# Singleton instance
_chunker_instance: Optional[SemanticChunker] = None


def get_semantic_chunker() -> SemanticChunker:
    """Get or create the semantic chunker instance."""
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = SemanticChunker()
    return _chunker_instance
