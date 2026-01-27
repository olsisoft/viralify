"""
Document Structure Extractor

Pre-processes documents to extract hierarchical structure BEFORE sending to LLM.
This ensures the LLM knows exactly how many sections/lectures to create.

Supports:
- Markdown headings (# ## ###)
- Numbered sections (1. 1.1 1.1.1)
- Roman numerals (I. II. III.)
- Named sections (Chapter, Section, Part, Chapitre, Partie)
- PDF-style headings (ALL CAPS, Bold markers)

Author: Viralify Team
Version: 1.0
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum


# =============================================================================
#                              DATA MODELS
# =============================================================================

class HeadingLevel(Enum):
    """Heading hierarchy levels."""
    DOCUMENT_TITLE = 0
    LEVEL_1 = 1  # Section / Chapter
    LEVEL_2 = 2  # Lecture / Subsection
    LEVEL_3 = 3  # Sub-subsection (usually ignored for course structure)
    UNKNOWN = 99


@dataclass
class ExtractedHeading:
    """A single extracted heading from the document."""
    level: HeadingLevel
    text: str
    original_line: str
    line_number: int
    pattern_matched: str  # Which pattern detected this heading

    def __str__(self):
        indent = "  " * (self.level.value - 1) if self.level.value > 0 else ""
        return f"{indent}[L{self.level.value}] {self.text}"


@dataclass
class DocumentSection:
    """A section with its subsections (for hierarchical structure)."""
    title: str
    level: HeadingLevel
    line_number: int
    children: List['DocumentSection'] = field(default_factory=list)

    @property
    def lecture_count(self) -> int:
        """Count direct children (lectures)."""
        return len(self.children)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "level": self.level.value,
            "children": [c.to_dict() for c in self.children]
        }


@dataclass
class DocumentStructure:
    """Complete extracted document structure."""
    title: Optional[str]
    sections: List[DocumentSection]
    all_headings: List[ExtractedHeading]
    detection_confidence: float  # 0.0 to 1.0
    patterns_used: List[str]
    warnings: List[str] = field(default_factory=list)

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def total_lectures(self) -> int:
        return sum(s.lecture_count for s in self.sections)

    @property
    def lectures_per_section(self) -> List[int]:
        return [s.lecture_count for s in self.sections]

    def get_summary(self) -> dict:
        return {
            "title": self.title,
            "section_count": self.section_count,
            "total_lectures": self.total_lectures,
            "lectures_per_section": self.lectures_per_section,
            "confidence": self.detection_confidence,
            "patterns_used": self.patterns_used,
            "warnings": self.warnings
        }


# =============================================================================
#                           HEADING PATTERNS
# =============================================================================

class HeadingPatterns:
    """
    Regex patterns for detecting headings in various formats.
    Ordered by specificity (most specific first).
    """

    # Markdown patterns
    MARKDOWN_H1 = (r'^#\s+(.+)$', HeadingLevel.LEVEL_1, "markdown_h1")
    MARKDOWN_H2 = (r'^##\s+(.+)$', HeadingLevel.LEVEL_2, "markdown_h2")
    MARKDOWN_H3 = (r'^###\s+(.+)$', HeadingLevel.LEVEL_3, "markdown_h3")

    # Numbered patterns: 1. 2. 3.
    NUMBERED_L1 = (r'^(\d+)\.\s+([A-ZÀ-Ü].+)$', HeadingLevel.LEVEL_1, "numbered_l1")
    # Numbered patterns: 1.1 1.2 or 1.1. 1.2.
    NUMBERED_L2 = (r'^(\d+\.\d+)\.?\s+(.+)$', HeadingLevel.LEVEL_2, "numbered_l2")
    # Numbered patterns: 1.1.1
    NUMBERED_L3 = (r'^(\d+\.\d+\.\d+)\.?\s+(.+)$', HeadingLevel.LEVEL_3, "numbered_l3")

    # Roman numerals: I. II. III.
    ROMAN_L1 = (r'^(I{1,3}|IV|VI{0,3}|IX|X{1,3})\.\s+(.+)$', HeadingLevel.LEVEL_1, "roman_l1")

    # Named sections (French)
    CHAPITRE = (r'^Chapitre\s+(\d+|[IVX]+)\s*[:\-–]?\s*(.+)$', HeadingLevel.LEVEL_1, "chapitre")
    PARTIE = (r'^Partie\s+(\d+|[IVX]+)\s*[:\-–]?\s*(.+)$', HeadingLevel.LEVEL_1, "partie")
    SECTION_FR = (r'^Section\s+(\d+|[IVX]+)\s*[:\-–]?\s*(.+)$', HeadingLevel.LEVEL_1, "section_fr")

    # Named sections (English)
    CHAPTER = (r'^Chapter\s+(\d+|[IVX]+)\s*[:\-–]?\s*(.+)$', HeadingLevel.LEVEL_1, "chapter")
    PART = (r'^Part\s+(\d+|[IVX]+)\s*[:\-–]?\s*(.+)$', HeadingLevel.LEVEL_1, "part")
    SECTION_EN = (r'^Section\s+(\d+|[IVX]+)\s*[:\-–]?\s*(.+)$', HeadingLevel.LEVEL_1, "section_en")

    # Letter enumeration: a) b) c) or a. b. c.
    LETTER_ENUM = (r'^([a-z])[)\.]\s+(.+)$', HeadingLevel.LEVEL_2, "letter_enum")

    # Bullet with capitalized content (potential heading)
    BULLET_CAPS = (r'^[-•]\s+([A-ZÀ-Ü][^.]+)$', HeadingLevel.LEVEL_2, "bullet_caps")

    # ALL CAPS line (PDF-style heading)
    ALL_CAPS = (r'^([A-ZÀ-Ü\s]{10,})$', HeadingLevel.LEVEL_1, "all_caps")

    # Colon-terminated (e.g., "Introduction:")
    COLON_TERM = (r'^([A-ZÀ-Ü][^:]{3,50}):$', HeadingLevel.LEVEL_1, "colon_term")

    @classmethod
    def get_all_patterns(cls) -> List[Tuple[str, HeadingLevel, str]]:
        """Get all patterns in priority order."""
        return [
            # Most specific first
            cls.NUMBERED_L3,
            cls.NUMBERED_L2,
            cls.NUMBERED_L1,
            cls.MARKDOWN_H3,
            cls.MARKDOWN_H2,
            cls.MARKDOWN_H1,
            cls.CHAPITRE,
            cls.PARTIE,
            cls.SECTION_FR,
            cls.CHAPTER,
            cls.PART,
            cls.SECTION_EN,
            cls.ROMAN_L1,
            cls.LETTER_ENUM,
            cls.BULLET_CAPS,
            cls.ALL_CAPS,
            cls.COLON_TERM,
        ]


# =============================================================================
#                           MAIN EXTRACTOR
# =============================================================================

class DocumentStructureExtractor:
    """
    Extracts hierarchical structure from document text.

    Usage:
        extractor = DocumentStructureExtractor()
        structure = extractor.extract(document_text)

        print(f"Found {structure.section_count} sections")
        print(f"Lectures per section: {structure.lectures_per_section}")
    """

    # Minimum confidence to consider structure valid
    MIN_CONFIDENCE = 0.3

    # Minimum headings to consider document has structure
    MIN_HEADINGS = 2

    def __init__(self):
        self.patterns = HeadingPatterns.get_all_patterns()

    def extract(self, text: str) -> DocumentStructure:
        """
        Extract structure from document text.

        Args:
            text: Raw document text

        Returns:
            DocumentStructure with sections, lectures, and metadata
        """
        if not text or len(text.strip()) < 50:
            return self._empty_structure("Document too short")

        # Step 1: Extract all headings
        headings = self._extract_all_headings(text)

        if len(headings) < self.MIN_HEADINGS:
            return self._empty_structure(
                f"Only {len(headings)} headings found (minimum: {self.MIN_HEADINGS})"
            )

        # Step 2: Normalize heading levels
        headings = self._normalize_levels(headings)

        # Step 3: Build hierarchical structure
        sections = self._build_hierarchy(headings)

        # Step 4: Extract document title
        title = self._extract_title(text, headings)

        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(headings, sections, text)

        # Step 6: Collect patterns used
        patterns_used = list(set(h.pattern_matched for h in headings))

        # Step 7: Generate warnings
        warnings = self._generate_warnings(sections, headings)

        return DocumentStructure(
            title=title,
            sections=sections,
            all_headings=headings,
            detection_confidence=confidence,
            patterns_used=patterns_used,
            warnings=warnings
        )

    def _extract_all_headings(self, text: str) -> List[ExtractedHeading]:
        """Extract all headings from text using patterns."""
        headings = []
        lines = text.split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            if not line or len(line) < 3:
                continue

            # Skip lines that look like code
            if self._is_code_line(line):
                continue

            # Try each pattern
            for pattern, level, pattern_name in self.patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Extract heading text
                    groups = match.groups()
                    if len(groups) >= 2:
                        # Pattern has number + text (e.g., "1. Introduction")
                        heading_text = groups[-1].strip()
                    else:
                        # Pattern has just text
                        heading_text = groups[0].strip()

                    # Clean up heading text
                    heading_text = self._clean_heading_text(heading_text)

                    if heading_text and len(heading_text) >= 3:
                        headings.append(ExtractedHeading(
                            level=level,
                            text=heading_text,
                            original_line=line,
                            line_number=line_num,
                            pattern_matched=pattern_name
                        ))
                    break  # Use first matching pattern

        return headings

    def _normalize_levels(self, headings: List[ExtractedHeading]) -> List[ExtractedHeading]:
        """
        Normalize heading levels to ensure proper hierarchy.

        If document only uses L2 and L3, promote them to L1 and L2.
        """
        if not headings:
            return headings

        # Find minimum level used
        levels_used = set(h.level for h in headings if h.level != HeadingLevel.UNKNOWN)

        if not levels_used:
            return headings

        min_level = min(l.value for l in levels_used if l.value > 0)

        # If minimum is not L1, shift all levels down
        if min_level > 1:
            shift = min_level - 1
            for heading in headings:
                if heading.level != HeadingLevel.UNKNOWN:
                    new_value = heading.level.value - shift
                    if new_value >= 1:
                        heading.level = HeadingLevel(new_value)

        return headings

    def _build_hierarchy(self, headings: List[ExtractedHeading]) -> List[DocumentSection]:
        """Build hierarchical section structure from flat heading list."""
        sections = []
        current_section: Optional[DocumentSection] = None

        for heading in headings:
            if heading.level == HeadingLevel.LEVEL_1:
                # New section
                current_section = DocumentSection(
                    title=heading.text,
                    level=heading.level,
                    line_number=heading.line_number
                )
                sections.append(current_section)

            elif heading.level == HeadingLevel.LEVEL_2:
                # Lecture/subsection
                if current_section is None:
                    # No parent section, create one
                    current_section = DocumentSection(
                        title="(Untitled Section)",
                        level=HeadingLevel.LEVEL_1,
                        line_number=heading.line_number
                    )
                    sections.append(current_section)

                # Add as child
                child = DocumentSection(
                    title=heading.text,
                    level=heading.level,
                    line_number=heading.line_number
                )
                current_section.children.append(child)

            # Ignore L3 and below for course structure

        return sections

    def _extract_title(
        self,
        text: str,
        headings: List[ExtractedHeading]
    ) -> Optional[str]:
        """Extract document title."""
        # Try first line if it looks like a title
        first_lines = text.strip().split('\n')[:5]

        for line in first_lines:
            line = line.strip()

            # Markdown title
            if line.startswith('# '):
                return line[2:].strip()

            # ALL CAPS title
            if line.isupper() and 5 < len(line) < 100:
                return line.title()

            # Short line that could be title
            if 5 < len(line) < 80 and not line.endswith('.'):
                return line

        # Use first L1 heading if no clear title
        l1_headings = [h for h in headings if h.level == HeadingLevel.LEVEL_1]
        if l1_headings:
            return l1_headings[0].text

        return None

    def _calculate_confidence(
        self,
        headings: List[ExtractedHeading],
        sections: List[DocumentSection],
        text: str
    ) -> float:
        """
        Calculate confidence score for extracted structure.

        Factors:
        - Consistency of patterns used
        - Ratio of headings to text length
        - Hierarchical structure validity
        """
        if not headings or not sections:
            return 0.0

        score = 0.0

        # Factor 1: Pattern consistency (40%)
        patterns = [h.pattern_matched for h in headings]
        unique_patterns = len(set(patterns))
        pattern_score = 1.0 / unique_patterns if unique_patterns > 0 else 0
        score += pattern_score * 0.4

        # Factor 2: Heading density (30%)
        text_length = len(text)
        heading_count = len(headings)
        # Ideal: ~1 heading per 500-1000 chars
        ideal_density = text_length / 750
        density_ratio = min(heading_count / max(ideal_density, 1), 2.0) / 2.0
        score += density_ratio * 0.3

        # Factor 3: Structure validity (30%)
        # Check if we have L1 with L2 children
        sections_with_lectures = sum(1 for s in sections if s.lecture_count > 0)
        structure_ratio = sections_with_lectures / len(sections) if sections else 0
        score += structure_ratio * 0.3

        return min(score, 1.0)

    def _generate_warnings(
        self,
        sections: List[DocumentSection],
        headings: List[ExtractedHeading]
    ) -> List[str]:
        """Generate warnings about potential issues."""
        warnings = []

        # Check for sections without lectures
        empty_sections = [s for s in sections if s.lecture_count == 0]
        if empty_sections:
            titles = [s.title[:30] for s in empty_sections[:3]]
            warnings.append(
                f"{len(empty_sections)} sections have no subsections: {titles}"
            )

        # Check for very unbalanced sections
        if sections:
            counts = [s.lecture_count for s in sections]
            if counts:
                avg = sum(counts) / len(counts)
                max_diff = max(abs(c - avg) for c in counts)
                if max_diff > avg * 2:
                    warnings.append(
                        f"Sections are unbalanced: {counts} lectures each"
                    )

        # Check for mixed patterns
        patterns = set(h.pattern_matched for h in headings)
        if len(patterns) > 3:
            warnings.append(
                f"Multiple heading styles detected: {patterns}"
            )

        return warnings

    def _is_code_line(self, line: str) -> bool:
        """Check if line looks like code."""
        code_indicators = [
            line.startswith('```'),
            line.startswith('    ') and any(c in line for c in '=()[]{}'),
            line.startswith('def '),
            line.startswith('class '),
            line.startswith('import '),
            line.startswith('from '),
            line.startswith('const '),
            line.startswith('let '),
            line.startswith('var '),
            line.startswith('function '),
            line.startswith('//'),
            line.startswith('/*'),
            line.startswith('*'),
            '=>' in line,
            '&&' in line,
            '||' in line,
        ]
        return any(code_indicators)

    def _clean_heading_text(self, text: str) -> str:
        """Clean up heading text."""
        # Remove trailing punctuation
        text = text.rstrip(':.-–')
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove markdown formatting
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'_+', '', text)
        return text.strip()

    def _empty_structure(self, reason: str) -> DocumentStructure:
        """Return empty structure with warning."""
        return DocumentStructure(
            title=None,
            sections=[],
            all_headings=[],
            detection_confidence=0.0,
            patterns_used=[],
            warnings=[f"No structure detected: {reason}"]
        )


# =============================================================================
#                           PROMPT FORMATTER
# =============================================================================

class StructurePromptFormatter:
    """
    Formats extracted structure for injection into LLM prompt.
    """

    @staticmethod
    def format_for_prompt(structure: DocumentStructure) -> str:
        """
        Format structure as a clear prompt section.

        Returns string to inject into the curriculum prompt.
        """
        if structure.section_count == 0:
            return StructurePromptFormatter._format_no_structure(structure)

        return StructurePromptFormatter._format_with_structure(structure)

    @staticmethod
    def _format_with_structure(structure: DocumentStructure) -> str:
        """Format when structure was detected."""

        # Build structure tree
        tree_lines = []
        for i, section in enumerate(structure.sections, 1):
            tree_lines.append(f"  {i}. {section.title}")
            for j, lecture in enumerate(section.children, 1):
                tree_lines.append(f"      {i}.{j} {lecture.title}")

        tree = "\n".join(tree_lines)

        # Build lecture distribution
        distribution = ", ".join(
            f"Section {i+1}: {count}"
            for i, count in enumerate(structure.lectures_per_section)
        )

        # Confidence indicator
        if structure.detection_confidence >= 0.7:
            confidence_text = "HIGH - Follow this structure exactly"
        elif structure.detection_confidence >= 0.4:
            confidence_text = "MEDIUM - Structure is likely correct"
        else:
            confidence_text = "LOW - Verify structure manually"

        # Warnings
        warnings_text = ""
        if structure.warnings:
            warnings_text = "\n⚠️ WARNINGS:\n" + "\n".join(f"  - {w}" for w in structure.warnings)

        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              📄 DETECTED DOCUMENT STRUCTURE (PRE-ANALYZED)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

DOCUMENT TITLE: {structure.title or "(Not detected)"}

STRUCTURE TREE:
{tree}

STATISTICS:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Total Sections:    {structure.section_count:<5}                                             │
│  Total Lectures:    {structure.total_lectures:<5}                                             │
│  Distribution:      {distribution:<40} │
│  Confidence:        {structure.detection_confidence:.0%} ({confidence_text})│
└─────────────────────────────────────────────────────────────────────────────┘
{warnings_text}

═══════════════════════════════════════════════════════════════════════════════
                    YOUR TASK: CONVERT THIS STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

Create EXACTLY:
• {structure.section_count} sections (one per Level-1 heading above)
• Lectures per section: {structure.lectures_per_section}

Section-by-section mapping:
{StructurePromptFormatter._format_mapping_instructions(structure)}

DO NOT deviate from this structure. It was extracted directly from the documents.
"""

    @staticmethod
    def _format_mapping_instructions(structure: DocumentStructure) -> str:
        """Format explicit mapping instructions."""
        lines = []
        for i, section in enumerate(structure.sections, 1):
            lines.append(f"  Section {i}: \"{section.title}\"")
            lines.append(f"            → Create section with title: \"{section.title}\"")
            lines.append(f"            → Include {section.lecture_count} lectures:")
            for j, lecture in enumerate(section.children, 1):
                lines.append(f"               {j}. \"{lecture.title}\"")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_no_structure(structure: DocumentStructure) -> str:
        """Format when no clear structure was detected."""
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ⚠️ NO CLEAR DOCUMENT STRUCTURE DETECTED                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

The document does not have clearly identifiable sections/subsections.

WARNINGS:
{chr(10).join(f"  - {w}" for w in structure.warnings) if structure.warnings else "  - Document may be unstructured text"}

FALLBACK INSTRUCTIONS:
1. Read the document content carefully
2. Identify natural topic boundaries
3. Group related content into sections
4. Each distinct concept = one lecture
5. Still DO NOT invent topics not covered in the document

Use the user's target counts as guidance:
• Target sections will be provided below
• Target lectures per section will be provided below
"""


# =============================================================================
#                           CONVENIENCE FUNCTIONS
# =============================================================================

def extract_document_structure(text: str) -> DocumentStructure:
    """
    Convenience function to extract structure.

    Usage:
        structure = extract_document_structure(document_text)
    """
    extractor = DocumentStructureExtractor()
    return extractor.extract(text)


def format_structure_for_prompt(structure: DocumentStructure) -> str:
    """
    Convenience function to format structure for prompt.

    Usage:
        prompt_section = format_structure_for_prompt(structure)
    """
    return StructurePromptFormatter.format_for_prompt(structure)


def analyze_and_format(text: str) -> Tuple[DocumentStructure, str]:
    """
    Extract structure and format for prompt in one call.

    Usage:
        structure, prompt_section = analyze_and_format(document_text)
    """
    structure = extract_document_structure(text)
    prompt_section = format_structure_for_prompt(structure)
    return structure, prompt_section
