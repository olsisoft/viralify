"""
Post-Generation Validation System

Validates LLM output AFTER generation to ensure:
1. Structure matches document structure
2. Source references are valid
3. No hallucinated content
4. Constraints are respected (section count, lecture count)
5. Content quality meets standards

Author: Viralify Team
Version: 1.0
"""

import re
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from difflib import SequenceMatcher


# =============================================================================
#                              ENUMS & TYPES
# =============================================================================

class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    ERROR = "error"        # Must fix - invalid output
    WARNING = "warning"    # Should review - potential issue
    INFO = "info"          # FYI - minor deviation


class ValidationCategory(Enum):
    """Categories of validation checks."""
    STRUCTURE = "structure"           # Section/lecture counts
    SOURCE_REFERENCE = "source_ref"   # RAG traceability
    CONTENT = "content"               # Quality checks
    HALLUCINATION = "hallucination"   # Invented content
    CONSTRAINT = "constraint"         # User requirements


# =============================================================================
#                              DATA MODELS
# =============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    location: str  # e.g., "sections[0].lectures[2]"
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    suggestion: Optional[str] = None

    def __str__(self):
        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[self.severity.value]
        return f"{icon} [{self.category.value}] {self.location}: {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    # Counts
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    # Scores (0.0 to 1.0)
    structure_score: float = 1.0
    source_reference_score: float = 1.0
    content_score: float = 1.0
    overall_score: float = 1.0

    # Detected issues
    hallucinated_sections: List[str] = field(default_factory=list)
    hallucinated_lectures: List[str] = field(default_factory=list)
    missing_source_refs: List[str] = field(default_factory=list)
    invalid_source_refs: List[str] = field(default_factory=list)

    def add_issue(self, issue: ValidationIssue):
        """Add an issue and update counts."""
        self.issues.append(issue)

        if issue.severity == ValidationSeverity.ERROR:
            self.error_count += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
        else:
            self.info_count += 1

    def get_summary(self) -> str:
        """Get human-readable summary."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"

        lines = [
            f"═══════════════════════════════════════════════════════════════",
            f"              VALIDATION REPORT: {status}",
            f"═══════════════════════════════════════════════════════════════",
            f"",
            f"📊 SCORES:",
            f"   Overall:          {self.overall_score:.0%}",
            f"   Structure:        {self.structure_score:.0%}",
            f"   Source References:{self.source_reference_score:.0%}",
            f"   Content Quality:  {self.content_score:.0%}",
            f"",
            f"📋 ISSUES: {self.error_count} errors, {self.warning_count} warnings, {self.info_count} info",
        ]

        if self.hallucinated_sections:
            lines.append(f"")
            lines.append(f"🚨 HALLUCINATED SECTIONS ({len(self.hallucinated_sections)}):")
            for s in self.hallucinated_sections[:5]:
                lines.append(f"   - {s}")

        if self.hallucinated_lectures:
            lines.append(f"")
            lines.append(f"🚨 HALLUCINATED LECTURES ({len(self.hallucinated_lectures)}):")
            for l in self.hallucinated_lectures[:5]:
                lines.append(f"   - {l}")

        if self.issues:
            lines.append(f"")
            lines.append(f"📝 ISSUES DETAIL:")
            for issue in self.issues[:10]:
                lines.append(f"   {issue}")
            if len(self.issues) > 10:
                lines.append(f"   ... and {len(self.issues) - 10} more")

        return "\n".join(lines)


# =============================================================================
#                         MAIN VALIDATOR CLASS
# =============================================================================

class PostGenerationValidator:
    """
    Validates LLM-generated curriculum against source documents and constraints.

    Usage:
        validator = PostGenerationValidator(
            document_text=rag_context,
            expected_sections=4,
            expected_lectures_per_section=[3, 4, 2, 3]
        )

        report = validator.validate(curriculum_json)

        if not report.is_valid:
            print(report.get_summary())
            # Handle invalid output...
    """

    # Thresholds
    MIN_SIMILARITY_HEADING = 0.7
    MIN_SIMILARITY_QUOTE = 0.5
    MIN_DESCRIPTION_LENGTH = 20
    MIN_OBJECTIVES_COUNT = 1
    MAX_OBJECTIVES_COUNT = 5
    MIN_VOICEOVER_WORDS = 50

    def __init__(
        self,
        document_text: str,
        expected_sections: Optional[int] = None,
        expected_lectures_per_section: Optional[List[int]] = None,
        strict_mode: bool = True,
        language: str = "fr"
    ):
        """
        Initialize validator.

        Args:
            document_text: Source document text (RAG context)
            expected_sections: Expected number of sections (from structure extraction)
            expected_lectures_per_section: Expected lectures per section
            strict_mode: If True, source_reference is mandatory
            language: Content language for quality checks
        """
        self.document_text = document_text
        self.document_lower = document_text.lower()
        self.expected_sections = expected_sections
        self.expected_lectures = expected_lectures_per_section or []
        self.strict_mode = strict_mode
        self.language = language

        # Pre-process document
        self.document_headings = self._extract_headings()
        self.document_words = set(re.findall(r'\b[a-zà-ÿ]{4,}\b', self.document_lower))

    def validate(self, curriculum: dict) -> ValidationReport:
        """
        Run all validations on the curriculum.

        Args:
            curriculum: The generated curriculum JSON

        Returns:
            ValidationReport with all issues found
        """
        report = ValidationReport(is_valid=True)

        # 1. Structure validation
        self._validate_structure(curriculum, report)

        # 2. Source reference validation
        self._validate_source_references(curriculum, report)

        # 3. Hallucination detection
        self._detect_hallucinations(curriculum, report)

        # 4. Content quality validation
        self._validate_content_quality(curriculum, report)

        # 5. Constraint validation
        self._validate_constraints(curriculum, report)

        # Calculate scores
        self._calculate_scores(report, curriculum)

        return report

    # =========================================================================
    #                      1. STRUCTURE VALIDATION
    # =========================================================================

    def _validate_structure(self, curriculum: dict, report: ValidationReport):
        """Validate structural requirements."""

        sections = curriculum.get("sections", [])

        # Check if sections exist
        if not sections:
            report.add_issue(ValidationIssue(
                category=ValidationCategory.STRUCTURE,
                severity=ValidationSeverity.ERROR,
                message="No sections found in curriculum",
                location="curriculum.sections",
                expected="At least 1 section",
                actual="0 sections"
            ))
            return

        # Check section count
        if self.expected_sections is not None:
            if len(sections) != self.expected_sections:
                severity = ValidationSeverity.ERROR if self.strict_mode else ValidationSeverity.WARNING
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.STRUCTURE,
                    severity=severity,
                    message=f"Section count mismatch",
                    location="curriculum.sections",
                    expected=self.expected_sections,
                    actual=len(sections),
                    suggestion=f"Adjust to {self.expected_sections} sections based on document structure"
                ))

        # Check lecture counts per section
        for i, section in enumerate(sections):
            lectures = section.get("lectures", [])

            if not lectures:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.STRUCTURE,
                    severity=ValidationSeverity.ERROR,
                    message="Section has no lectures",
                    location=f"sections[{i}]",
                    expected="At least 1 lecture",
                    actual="0 lectures"
                ))

            # Check against expected
            if i < len(self.expected_lectures):
                expected_count = self.expected_lectures[i]
                if len(lectures) != expected_count:
                    severity = ValidationSeverity.ERROR if self.strict_mode else ValidationSeverity.WARNING
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.STRUCTURE,
                        severity=severity,
                        message=f"Lecture count mismatch in section '{section.get('title', 'Unknown')}'",
                        location=f"sections[{i}].lectures",
                        expected=expected_count,
                        actual=len(lectures)
                    ))

    # =========================================================================
    #                  2. SOURCE REFERENCE VALIDATION
    # =========================================================================

    def _validate_source_references(self, curriculum: dict, report: ValidationReport):
        """Validate source_reference fields."""

        sections = curriculum.get("sections", [])

        for i, section in enumerate(sections):
            section_title = section.get("title", "Unknown")
            section_ref = section.get("source_reference", "")

            # Check section source_reference
            ref_result = self._check_source_reference(
                source_ref=section_ref,
                item_title=section_title,
                location=f"sections[{i}]",
                item_type="Section"
            )

            if ref_result:
                report.add_issue(ref_result)
                if ref_result.severity == ValidationSeverity.ERROR:
                    if "missing" in ref_result.message.lower():
                        report.missing_source_refs.append(section_title)
                    else:
                        report.invalid_source_refs.append(section_title)

            # Check lecture source_references
            for j, lecture in enumerate(section.get("lectures", [])):
                lecture_title = lecture.get("title", "Unknown")
                lecture_ref = lecture.get("source_reference", "")

                ref_result = self._check_source_reference(
                    source_ref=lecture_ref,
                    item_title=lecture_title,
                    location=f"sections[{i}].lectures[{j}]",
                    item_type="Lecture"
                )

                if ref_result:
                    report.add_issue(ref_result)
                    if ref_result.severity == ValidationSeverity.ERROR:
                        if "missing" in ref_result.message.lower():
                            report.missing_source_refs.append(lecture_title)
                        else:
                            report.invalid_source_refs.append(lecture_title)

    def _check_source_reference(
        self,
        source_ref: str,
        item_title: str,
        location: str,
        item_type: str
    ) -> Optional[ValidationIssue]:
        """Check a single source_reference."""

        # Check if missing
        if not source_ref or not source_ref.strip():
            if self.strict_mode:
                return ValidationIssue(
                    category=ValidationCategory.SOURCE_REFERENCE,
                    severity=ValidationSeverity.ERROR,
                    message=f"{item_type} '{item_title}' missing source_reference",
                    location=location,
                    suggestion="Add source_reference with exact document heading"
                )
            else:
                return ValidationIssue(
                    category=ValidationCategory.SOURCE_REFERENCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"{item_type} '{item_title}' has no source_reference",
                    location=location
                )

        # Check for placeholder values
        placeholder_values = ["n/a", "na", "none", "null", "undefined",
                              "see document", "from document", "document"]
        if source_ref.lower().strip() in placeholder_values:
            return ValidationIssue(
                category=ValidationCategory.SOURCE_REFERENCE,
                severity=ValidationSeverity.ERROR,
                message=f"{item_type} '{item_title}' has placeholder source_reference: '{source_ref}'",
                location=location,
                suggestion="Replace with actual document heading"
            )

        # Check if reference exists in document
        if not self._reference_exists_in_document(source_ref):
            return ValidationIssue(
                category=ValidationCategory.SOURCE_REFERENCE,
                severity=ValidationSeverity.ERROR,
                message=f"{item_type} '{item_title}' source_reference not found in document",
                location=location,
                actual=source_ref[:50] + "..." if len(source_ref) > 50 else source_ref,
                suggestion="Verify this heading exists in the source document"
            )

        return None  # Valid

    def _reference_exists_in_document(self, reference: str) -> bool:
        """Check if a reference exists in the document."""
        ref_lower = reference.lower().strip()

        # Direct match
        if ref_lower in self.document_lower:
            return True

        # Check against extracted headings
        for heading in self.document_headings:
            # Exact match
            if heading.lower() == ref_lower:
                return True
            # Substring match
            if ref_lower in heading.lower() or heading.lower() in ref_lower:
                return True
            # Similarity match
            similarity = SequenceMatcher(None, ref_lower, heading.lower()).ratio()
            if similarity >= self.MIN_SIMILARITY_HEADING:
                return True

        # Extract key words and check
        words = re.findall(r'\b[a-zà-ÿ]{4,}\b', ref_lower)
        if words:
            matches = sum(1 for w in words if w in self.document_lower)
            if matches / len(words) >= 0.6:
                return True

        return False

    # =========================================================================
    #                    3. HALLUCINATION DETECTION
    # =========================================================================

    def _detect_hallucinations(self, curriculum: dict, report: ValidationReport):
        """Detect potentially hallucinated content."""

        sections = curriculum.get("sections", [])

        for i, section in enumerate(sections):
            section_title = section.get("title", "")

            # Check if section title appears in document
            if not self._title_in_document(section_title):
                report.hallucinated_sections.append(section_title)
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.HALLUCINATION,
                    severity=ValidationSeverity.ERROR,
                    message=f"Section title not found in documents (possible hallucination)",
                    location=f"sections[{i}]",
                    actual=section_title,
                    suggestion="Remove this section or verify against source documents"
                ))

            # Check lectures
            for j, lecture in enumerate(section.get("lectures", [])):
                lecture_title = lecture.get("title", "")

                if not self._title_in_document(lecture_title):
                    report.hallucinated_lectures.append(lecture_title)
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.HALLUCINATION,
                        severity=ValidationSeverity.ERROR,
                        message=f"Lecture title not found in documents (possible hallucination)",
                        location=f"sections[{i}].lectures[{j}]",
                        actual=lecture_title,
                        suggestion="Remove this lecture or verify against source documents"
                    ))

                # Check key concepts
                for concept in lecture.get("key_concepts", []):
                    if not self._concept_in_document(concept):
                        report.add_issue(ValidationIssue(
                            category=ValidationCategory.HALLUCINATION,
                            severity=ValidationSeverity.WARNING,
                            message=f"Key concept '{concept}' not found in documents",
                            location=f"sections[{i}].lectures[{j}].key_concepts",
                            suggestion="Remove or verify this concept"
                        ))

    def _title_in_document(self, title: str) -> bool:
        """Check if a title appears in the document."""
        if not title:
            return False

        title_lower = title.lower().strip()

        # Direct match
        if title_lower in self.document_lower:
            return True

        # Check against headings
        for heading in self.document_headings:
            if title_lower in heading.lower() or heading.lower() in title_lower:
                return True
            similarity = SequenceMatcher(None, title_lower, heading.lower()).ratio()
            if similarity >= 0.7:
                return True

        # Word overlap
        words = set(re.findall(r'\b[a-zà-ÿ]{4,}\b', title_lower))
        if words:
            overlap = words.intersection(self.document_words)
            if len(overlap) / len(words) >= 0.5:
                return True

        return False

    def _concept_in_document(self, concept: str) -> bool:
        """Check if a concept appears in the document."""
        concept_lower = concept.lower().strip()
        return concept_lower in self.document_lower

    # =========================================================================
    #                    4. CONTENT QUALITY VALIDATION
    # =========================================================================

    def _validate_content_quality(self, curriculum: dict, report: ValidationReport):
        """Validate content quality."""

        sections = curriculum.get("sections", [])

        for i, section in enumerate(sections):
            # Check section description
            desc = section.get("description", "")
            if len(desc) < self.MIN_DESCRIPTION_LENGTH:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.CONTENT,
                    severity=ValidationSeverity.WARNING,
                    message=f"Section description too short ({len(desc)} chars)",
                    location=f"sections[{i}].description",
                    expected=f"At least {self.MIN_DESCRIPTION_LENGTH} chars",
                    actual=f"{len(desc)} chars"
                ))

            for j, lecture in enumerate(section.get("lectures", [])):
                # Check lecture description
                lecture_desc = lecture.get("description", "")
                if len(lecture_desc) < self.MIN_DESCRIPTION_LENGTH:
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=ValidationSeverity.WARNING,
                        message=f"Lecture description too short",
                        location=f"sections[{i}].lectures[{j}].description",
                        expected=f"At least {self.MIN_DESCRIPTION_LENGTH} chars",
                        actual=f"{len(lecture_desc)} chars"
                    ))

                # Check objectives
                objectives = lecture.get("objectives", [])
                if len(objectives) < self.MIN_OBJECTIVES_COUNT:
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=ValidationSeverity.WARNING,
                        message=f"Not enough learning objectives",
                        location=f"sections[{i}].lectures[{j}].objectives",
                        expected=f"At least {self.MIN_OBJECTIVES_COUNT}",
                        actual=f"{len(objectives)}"
                    ))
                elif len(objectives) > self.MAX_OBJECTIVES_COUNT:
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=ValidationSeverity.INFO,
                        message=f"Many learning objectives ({len(objectives)})",
                        location=f"sections[{i}].lectures[{j}].objectives",
                        suggestion="Consider reducing to 3-5 key objectives"
                    ))

                # Check difficulty value
                difficulty = lecture.get("difficulty", "")
                valid_difficulties = ["beginner", "intermediate", "advanced",
                                      "very_advanced", "expert"]
                if difficulty not in valid_difficulties:
                    report.add_issue(ValidationIssue(
                        category=ValidationCategory.CONTENT,
                        severity=ValidationSeverity.WARNING,
                        message=f"Invalid difficulty value: '{difficulty}'",
                        location=f"sections[{i}].lectures[{j}].difficulty",
                        expected="beginner, intermediate, advanced, very_advanced, or expert",
                        actual=difficulty
                    ))

    # =========================================================================
    #                    5. CONSTRAINT VALIDATION
    # =========================================================================

    def _validate_constraints(self, curriculum: dict, report: ValidationReport):
        """Validate against user constraints."""

        # Check required fields
        required_fields = ["title", "description", "sections"]
        for field_name in required_fields:
            if field_name not in curriculum or not curriculum[field_name]:
                report.add_issue(ValidationIssue(
                    category=ValidationCategory.CONSTRAINT,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing required field: {field_name}",
                    location=f"curriculum.{field_name}"
                ))

        # Check title length
        title = curriculum.get("title", "")
        if len(title) < 5:
            report.add_issue(ValidationIssue(
                category=ValidationCategory.CONSTRAINT,
                severity=ValidationSeverity.ERROR,
                message="Course title too short",
                location="curriculum.title",
                expected="At least 5 characters",
                actual=f"{len(title)} characters"
            ))

    # =========================================================================
    #                        SCORE CALCULATION
    # =========================================================================

    def _calculate_scores(self, report: ValidationReport, curriculum: dict):
        """Calculate validation scores."""

        sections = curriculum.get("sections", [])
        total_items = len(sections) + sum(len(s.get("lectures", [])) for s in sections)

        if total_items == 0:
            report.overall_score = 0.0
            return

        # Structure score
        structure_issues = [i for i in report.issues if i.category == ValidationCategory.STRUCTURE]
        structure_errors = sum(1 for i in structure_issues if i.severity == ValidationSeverity.ERROR)
        report.structure_score = max(0, 1.0 - (structure_errors * 0.2))

        # Source reference score
        missing_refs = len(report.missing_source_refs) + len(report.invalid_source_refs)
        report.source_reference_score = max(0, 1.0 - (missing_refs / max(total_items, 1)))

        # Content score
        content_issues = [i for i in report.issues if i.category == ValidationCategory.CONTENT]
        content_errors = sum(1 for i in content_issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING])
        report.content_score = max(0, 1.0 - (content_errors * 0.1))

        # Hallucination penalty
        hallucination_count = len(report.hallucinated_sections) + len(report.hallucinated_lectures)
        hallucination_penalty = hallucination_count / max(total_items, 1)

        # Overall score
        report.overall_score = (
            report.structure_score * 0.3 +
            report.source_reference_score * 0.4 +
            report.content_score * 0.2 +
            (1.0 - hallucination_penalty) * 0.1
        )

    # =========================================================================
    #                        HELPER METHODS
    # =========================================================================

    def _extract_headings(self) -> List[str]:
        """Extract all headings from document."""
        headings = []

        for line in self.document_text.split('\n'):
            line = line.strip()

            # Markdown headings
            if line.startswith('#'):
                heading = line.lstrip('#').strip()
                if heading:
                    headings.append(heading)
                    headings.append(line)  # Also keep with marker

            # Numbered headings
            elif re.match(r'^\d+\.(\d+\.)*\s+', line):
                headings.append(line)
                # Also without number
                text = re.sub(r'^\d+\.(\d+\.)*\s+', '', line)
                if text:
                    headings.append(text)

            # Chapter markers
            elif re.match(r'^(chapitre|chapter|section|partie|part)\s+\d+', line, re.I):
                headings.append(line)

        return headings


# =============================================================================
#                        CURRICULUM CORRECTOR
# =============================================================================

class CurriculumCorrector:
    """
    Attempts to auto-correct validation issues.

    Actions:
    - Remove hallucinated sections/lectures
    - Fill missing source_references (if possible)
    - Fix constraint violations
    """

    def __init__(self, document_text: str):
        self.document_text = document_text
        self.document_headings = self._extract_headings()

    def correct(
        self,
        curriculum: dict,
        report: ValidationReport
    ) -> Tuple[dict, List[str]]:
        """
        Attempt to correct the curriculum.

        Returns:
            Tuple of (corrected_curriculum, list_of_corrections_made)
        """
        corrections = []
        corrected = json.loads(json.dumps(curriculum))  # Deep copy

        # 1. Remove hallucinated sections
        if report.hallucinated_sections:
            original_count = len(corrected.get("sections", []))
            corrected["sections"] = [
                s for s in corrected.get("sections", [])
                if s.get("title", "") not in report.hallucinated_sections
            ]
            removed = original_count - len(corrected.get("sections", []))
            if removed > 0:
                corrections.append(f"Removed {removed} hallucinated sections")

        # 2. Remove hallucinated lectures
        if report.hallucinated_lectures:
            for section in corrected.get("sections", []):
                original_count = len(section.get("lectures", []))
                section["lectures"] = [
                    l for l in section.get("lectures", [])
                    if l.get("title", "") not in report.hallucinated_lectures
                ]
                removed = original_count - len(section.get("lectures", []))
                if removed > 0:
                    corrections.append(f"Removed {removed} hallucinated lectures from '{section.get('title', '')}'")

        # 3. Try to fill missing source_references
        for section in corrected.get("sections", []):
            if not section.get("source_reference"):
                match = self._find_best_heading_match(section.get("title", ""))
                if match:
                    section["source_reference"] = match
                    corrections.append(f"Added source_reference for section '{section.get('title', '')}'")

            for lecture in section.get("lectures", []):
                if not lecture.get("source_reference"):
                    match = self._find_best_heading_match(lecture.get("title", ""))
                    if match:
                        lecture["source_reference"] = match
                        corrections.append(f"Added source_reference for lecture '{lecture.get('title', '')}'")

        # 4. Remove empty sections
        original_count = len(corrected.get("sections", []))
        corrected["sections"] = [
            s for s in corrected.get("sections", [])
            if s.get("lectures")
        ]
        removed = original_count - len(corrected.get("sections", []))
        if removed > 0:
            corrections.append(f"Removed {removed} empty sections")

        return corrected, corrections

    def _extract_headings(self) -> List[str]:
        """Extract headings from document."""
        headings = []
        for line in self.document_text.split('\n'):
            line = line.strip()
            if line.startswith('#') or re.match(r'^\d+\.', line):
                headings.append(line)
        return headings

    def _find_best_heading_match(self, title: str) -> Optional[str]:
        """Find best matching heading for a title."""
        if not title:
            return None

        title_lower = title.lower()
        best_match = None
        best_score = 0

        for heading in self.document_headings:
            score = SequenceMatcher(None, title_lower, heading.lower()).ratio()
            if score > best_score and score >= 0.5:
                best_score = score
                best_match = heading

        return best_match


# =============================================================================
#                        CONVENIENCE FUNCTIONS
# =============================================================================

def validate_curriculum(
    curriculum: dict,
    document_text: str,
    expected_sections: Optional[int] = None,
    expected_lectures: Optional[List[int]] = None,
    strict: bool = True
) -> ValidationReport:
    """
    Convenience function to validate curriculum.

    Usage:
        report = validate_curriculum(
            curriculum=generated_json,
            document_text=rag_context,
            expected_sections=4,
            expected_lectures=[3, 4, 2, 3]
        )
    """
    validator = PostGenerationValidator(
        document_text=document_text,
        expected_sections=expected_sections,
        expected_lectures_per_section=expected_lectures,
        strict_mode=strict
    )
    return validator.validate(curriculum)


def quick_validate(curriculum: dict, document_text: str) -> Tuple[bool, str]:
    """
    Quick validation - returns (is_valid, summary_message).

    Usage:
        is_valid, message = quick_validate(curriculum, rag_context)
        if not is_valid:
            print(f"Validation failed: {message}")
    """
    report = validate_curriculum(curriculum, document_text, strict=False)

    if report.is_valid:
        return True, f"Valid ({report.overall_score:.0%} score)"
    else:
        issues = [str(i) for i in report.issues if i.severity == ValidationSeverity.ERROR]
        return False, f"Invalid: {'; '.join(issues[:3])}"
