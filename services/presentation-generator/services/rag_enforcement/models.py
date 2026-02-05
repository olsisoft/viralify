"""
RAG Enforcement Models

Data structures for RAG compliance verification and enforcement.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class FactStatus(str, Enum):
    """Status of a fact verification"""
    SUPPORTED = "supported"          # Fact found in sources
    UNSUPPORTED = "unsupported"      # Fact not found but not contradicted
    CONTRADICTED = "contradicted"    # Fact contradicts sources
    HALLUCINATION = "hallucination"  # Fact clearly made up


class ComplianceLevel(str, Enum):
    """Overall compliance level"""
    COMPLIANT = "compliant"          # >= 90% grounded
    PARTIAL = "partial"              # 70-90% grounded
    NON_COMPLIANT = "non_compliant"  # < 70% grounded
    REJECTED = "rejected"            # Too many hallucinations


@dataclass
class Citation:
    """A citation reference in the content"""
    ref_id: str                      # e.g., "1", "2"
    text: str                        # The cited text
    source_chunk: Optional[str] = None  # Matched source chunk
    is_valid: bool = True            # Whether citation exists in sources
    similarity: float = 0.0          # Similarity to source


@dataclass
class SentenceScore:
    """Verification result for a single sentence"""
    sentence: str
    similarity: float                # Best match similarity (0-1)
    matched_source: Optional[str] = None  # Best matching source chunk
    is_grounded: bool = False        # similarity > threshold
    citations: List[Citation] = field(default_factory=list)
    fact_status: FactStatus = FactStatus.UNSUPPORTED


@dataclass
class CitationReport:
    """Report on citation validation"""
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    uncited_sentences: int = 0       # Sentences without citations
    total_sentences: int = 0
    citations: List[Citation] = field(default_factory=list)
    uncited_sentence_list: List[str] = field(default_factory=list)

    @property
    def citation_rate(self) -> float:
        """Percentage of sentences with valid citations"""
        if self.total_sentences == 0:
            return 0.0
        cited = self.total_sentences - self.uncited_sentences
        return cited / self.total_sentences

    @property
    def validity_rate(self) -> float:
        """Percentage of valid citations"""
        if self.total_citations == 0:
            return 1.0  # No citations = no invalid ones
        return self.valid_citations / self.total_citations


@dataclass
class SentenceReport:
    """Report on sentence-level verification"""
    total_sentences: int = 0
    grounded_sentences: int = 0
    ungrounded_sentences: int = 0
    average_similarity: float = 0.0
    sentence_scores: List[SentenceScore] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        """Percentage of grounded sentences"""
        if self.total_sentences == 0:
            return 0.0
        return self.grounded_sentences / self.total_sentences

    def get_ungrounded(self) -> List[SentenceScore]:
        """Get all ungrounded sentences"""
        return [s for s in self.sentence_scores if not s.is_grounded]

    def get_worst_sentences(self, n: int = 5) -> List[SentenceScore]:
        """Get N sentences with lowest similarity"""
        sorted_scores = sorted(self.sentence_scores, key=lambda x: x.similarity)
        return sorted_scores[:n]


@dataclass
class EnforcementResult:
    """Result of RAG enforcement"""
    content: str                     # The generated/validated content
    is_compliant: bool = False       # Meets compliance threshold
    compliance_level: ComplianceLevel = ComplianceLevel.NON_COMPLIANT

    # Detailed reports
    citation_report: Optional[CitationReport] = None
    sentence_report: Optional[SentenceReport] = None

    # Scores
    overall_score: float = 0.0       # Combined compliance score
    citation_score: float = 0.0      # Citation-based score
    grounding_score: float = 0.0     # Sentence grounding score

    # Attempt info
    attempt_number: int = 1
    total_attempts: int = 1

    # Issues found
    hallucinations: List[str] = field(default_factory=list)
    ungrounded_facts: List[str] = field(default_factory=list)

    # Metadata
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            "is_compliant": self.is_compliant,
            "compliance_level": self.compliance_level.value,
            "overall_score": round(self.overall_score, 3),
            "citation_score": round(self.citation_score, 3),
            "grounding_score": round(self.grounding_score, 3),
            "attempt_number": self.attempt_number,
            "total_attempts": self.total_attempts,
            "hallucination_count": len(self.hallucinations),
            "ungrounded_count": len(self.ungrounded_facts),
            "processing_time_ms": round(self.processing_time_ms, 1),
        }


@dataclass
class EnforcementConfig:
    """Configuration for RAG enforcement"""
    # Thresholds
    min_compliance_score: float = 0.90      # Minimum overall score
    min_citation_rate: float = 0.80         # Minimum citation coverage
    min_grounding_score: float = 0.85       # Minimum sentence grounding
    sentence_similarity_threshold: float = 0.60  # Threshold for "grounded"

    # Retry settings
    max_attempts: int = 3

    # Weights for combined score
    citation_weight: float = 0.3
    grounding_weight: float = 0.7

    # Citation settings
    require_citations: bool = True          # Enforce inline citations
    min_words_for_citation: int = 10        # Sentences under this don't need citation

    # Strictness levels for retries
    strictness_prompts: Dict[str, str] = field(default_factory=lambda: {
        "standard": "Use the source documents as your primary reference.",
        "strict": "You MUST use ONLY information from the source documents. Do NOT add any external knowledge.",
        "ultra_strict": "CRITICAL: Every single fact must come from the sources. If information is missing, write '[INSUFFICIENT SOURCE]' instead of making it up."
    })
