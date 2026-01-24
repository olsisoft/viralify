"""
Coherence Check Service

Verifies pedagogical coherence across lectures:
1. Validates that prerequisites are met before each lecture
2. Identifies concepts introduced in each lecture
3. Maps lecture dependencies for optimal learning flow
4. Enriches lectures with coherence metadata
"""
import json
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from openai import AsyncOpenAI
import os

from models.course_models import CourseOutline, Section, Lecture


@dataclass
class ConceptNode:
    """A concept in the knowledge graph."""
    name: str
    introduced_in: Optional[str] = None  # lecture_id where first introduced
    used_in: List[str] = field(default_factory=list)  # lecture_ids where used
    complexity_level: int = 1  # 1-5 complexity scale


@dataclass
class CoherenceIssue:
    """A coherence issue found in the course structure."""
    issue_type: str  # "missing_prerequisite", "concept_gap", "order_issue"
    severity: str  # "error", "warning", "info"
    lecture_id: str
    lecture_title: str
    description: str
    suggestion: str


@dataclass
class CoherenceCheckResult:
    """Result of coherence analysis."""
    is_coherent: bool
    score: float  # 0-100 coherence score
    issues: List[CoherenceIssue] = field(default_factory=list)
    concept_graph: Dict[str, ConceptNode] = field(default_factory=dict)
    lecture_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    summary: str = ""


class CoherenceCheckService:
    """
    Service for checking and ensuring pedagogical coherence.

    This service analyzes the course structure to:
    1. Extract concepts from each lecture
    2. Build a concept dependency graph
    3. Validate lecture ordering based on prerequisites
    4. Enrich lectures with coherence metadata
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.client = AsyncOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            timeout=60.0,
            max_retries=2,
        )

    async def check_coherence(
        self,
        outline: CourseOutline,
        verbose: bool = False,
    ) -> CoherenceCheckResult:
        """
        Check the pedagogical coherence of a course outline.

        Args:
            outline: The course outline to analyze
            verbose: If True, print detailed analysis

        Returns:
            CoherenceCheckResult with issues and enriched data
        """
        print(f"[COHERENCE] Checking coherence for: {outline.title}", flush=True)

        result = CoherenceCheckResult(
            is_coherent=True,
            score=100.0,
            issues=[],
            concept_graph={},
            lecture_dependencies={},
        )

        # Step 1: Extract concepts from each lecture
        all_lectures = self._get_all_lectures(outline)
        if not all_lectures:
            result.summary = "No lectures to analyze"
            return result

        # Step 2: Use AI to analyze concepts and dependencies
        concept_analysis = await self._analyze_concepts(outline, all_lectures)

        if not concept_analysis:
            result.summary = "Could not analyze concepts"
            result.score = 50.0
            return result

        # Step 3: Build concept graph
        result.concept_graph = self._build_concept_graph(concept_analysis, all_lectures)

        # Step 4: Check for coherence issues
        issues = self._identify_issues(concept_analysis, all_lectures, result.concept_graph)
        result.issues = issues

        # Step 5: Calculate score based on issues
        if issues:
            error_count = sum(1 for i in issues if i.severity == "error")
            warning_count = sum(1 for i in issues if i.severity == "warning")
            result.score = max(0, 100 - (error_count * 15) - (warning_count * 5))
            result.is_coherent = error_count == 0

        # Step 6: Build lecture dependencies
        result.lecture_dependencies = self._build_dependencies(concept_analysis, all_lectures)

        # Generate summary
        if result.is_coherent:
            result.summary = f"✅ Course is coherent (score: {result.score:.0f}/100)"
            if issues:
                result.summary += f" with {len(issues)} minor suggestions"
        else:
            result.summary = f"⚠️ Coherence issues found (score: {result.score:.0f}/100): {len([i for i in issues if i.severity == 'error'])} errors"

        if verbose:
            print(f"[COHERENCE] {result.summary}", flush=True)
            for issue in issues:
                print(f"[COHERENCE] [{issue.severity}] {issue.lecture_title}: {issue.description}", flush=True)

        return result

    async def enrich_outline_with_coherence(
        self,
        outline: CourseOutline,
    ) -> CourseOutline:
        """
        Enrich the course outline with coherence metadata.

        This adds prerequisites, introduces, and prepares_for fields
        to each lecture based on concept analysis.

        Args:
            outline: The course outline to enrich

        Returns:
            Enriched CourseOutline
        """
        print(f"[COHERENCE] Enriching outline with coherence data", flush=True)

        # Get concept analysis
        all_lectures = self._get_all_lectures(outline)
        if not all_lectures:
            return outline

        concept_analysis = await self._analyze_concepts(outline, all_lectures)
        if not concept_analysis:
            return outline

        # Enrich each lecture
        for section in outline.sections:
            for lecture in section.lectures:
                lecture_data = concept_analysis.get(lecture.id, {})

                # Add key concepts
                lecture.key_concepts = lecture_data.get("key_concepts", [])

                # Add prerequisites (concepts that should be known)
                lecture.prerequisites = lecture_data.get("prerequisites", [])

                # Add introduces (new concepts in this lecture)
                lecture.introduces = lecture_data.get("introduces", [])

                # Add prepares_for (concepts this prepares for)
                lecture.prepares_for = lecture_data.get("prepares_for", [])

        print(f"[COHERENCE] Enriched {len(all_lectures)} lectures", flush=True)
        return outline

    def _get_all_lectures(self, outline: CourseOutline) -> List[Tuple[int, int, Lecture]]:
        """Get all lectures with their section and lecture indices."""
        lectures = []
        for sec_idx, section in enumerate(outline.sections):
            for lec_idx, lecture in enumerate(section.lectures):
                lectures.append((sec_idx, lec_idx, lecture))
        return lectures

    async def _analyze_concepts(
        self,
        outline: CourseOutline,
        lectures: List[Tuple[int, int, Lecture]],
    ) -> Dict[str, Dict[str, Any]]:
        """Use AI to analyze concepts in each lecture."""
        # Build lecture summary for AI
        lecture_summaries = []
        for sec_idx, lec_idx, lecture in lectures:
            summary = {
                "id": lecture.id,
                "section": sec_idx + 1,
                "lecture": lec_idx + 1,
                "title": lecture.title,
                "description": lecture.description,
                "objectives": lecture.objectives[:3],  # Limit for token efficiency
                "difficulty": lecture.difficulty.value,
            }
            lecture_summaries.append(summary)

        prompt = f"""Analyze the pedagogical structure of this course outline and identify concepts for each lecture.

Course: {outline.title}
Description: {outline.description}

Lectures (in order):
{json.dumps(lecture_summaries, indent=2, ensure_ascii=False)}

For EACH lecture, identify:
1. "key_concepts": The main concepts/topics covered (2-5 items)
2. "prerequisites": Concepts the student should already understand (from previous lectures or external knowledge)
3. "introduces": NEW concepts first introduced in this lecture
4. "prepares_for": Concepts this lecture prepares the student for (to be covered later)

Return a JSON object where keys are lecture IDs and values contain the analysis:
{{
    "lecture_id": {{
        "key_concepts": ["concept1", "concept2"],
        "prerequisites": ["prereq1"],
        "introduces": ["new_concept1"],
        "prepares_for": ["future_concept1"]
    }}
}}

IMPORTANT:
- Be specific with concept names (e.g., "binary search algorithm" not just "algorithms")
- Prerequisites should reference concepts from earlier lectures or assumed background
- Introduces should only list truly NEW concepts
- Consider the difficulty progression when analyzing prerequisites"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert curriculum designer analyzing pedagogical coherence. Return valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=3000,
            )

            content = response.choices[0].message.content.strip()
            return json.loads(content)

        except Exception as e:
            print(f"[COHERENCE] Concept analysis error: {e}", flush=True)
            return {}

    def _build_concept_graph(
        self,
        analysis: Dict[str, Dict[str, Any]],
        lectures: List[Tuple[int, int, Lecture]],
    ) -> Dict[str, ConceptNode]:
        """Build a graph of concepts and where they're introduced/used."""
        graph = {}

        # Create lecture order mapping
        lecture_order = {lecture.id: (sec_idx, lec_idx) for sec_idx, lec_idx, lecture in lectures}

        for lecture_id, data in analysis.items():
            # Track introduced concepts
            for concept in data.get("introduces", []):
                if concept not in graph:
                    graph[concept] = ConceptNode(
                        name=concept,
                        introduced_in=lecture_id,
                        used_in=[lecture_id],
                    )
                else:
                    graph[concept].used_in.append(lecture_id)

            # Track key concepts (usage)
            for concept in data.get("key_concepts", []):
                if concept not in graph:
                    graph[concept] = ConceptNode(
                        name=concept,
                        used_in=[lecture_id],
                    )
                elif lecture_id not in graph[concept].used_in:
                    graph[concept].used_in.append(lecture_id)

        return graph

    def _identify_issues(
        self,
        analysis: Dict[str, Dict[str, Any]],
        lectures: List[Tuple[int, int, Lecture]],
        concept_graph: Dict[str, ConceptNode],
    ) -> List[CoherenceIssue]:
        """Identify coherence issues in the course structure."""
        issues = []

        # Create lecture order mapping
        lecture_order = {lecture.id: (sec_idx, lec_idx) for sec_idx, lec_idx, lecture in lectures}
        lecture_titles = {lecture.id: lecture.title for _, _, lecture in lectures}

        # Check each lecture's prerequisites
        for lecture_id, data in analysis.items():
            order = lecture_order.get(lecture_id, (0, 0))

            for prereq in data.get("prerequisites", []):
                # Check if prerequisite concept exists in the course
                if prereq in concept_graph:
                    node = concept_graph[prereq]
                    if node.introduced_in:
                        intro_order = lecture_order.get(node.introduced_in, (99, 99))
                        # Check if introduced AFTER this lecture (order issue)
                        if intro_order > order:
                            issues.append(CoherenceIssue(
                                issue_type="order_issue",
                                severity="error",
                                lecture_id=lecture_id,
                                lecture_title=lecture_titles.get(lecture_id, ""),
                                description=f"Prerequisite '{prereq}' is introduced in a later lecture",
                                suggestion=f"Consider moving '{prereq}' introduction earlier or reordering lectures"
                            ))
                else:
                    # External prerequisite - just a warning
                    issues.append(CoherenceIssue(
                        issue_type="missing_prerequisite",
                        severity="warning",
                        lecture_id=lecture_id,
                        lecture_title=lecture_titles.get(lecture_id, ""),
                        description=f"Prerequisite '{prereq}' is not covered in the course",
                        suggestion=f"Consider adding a brief introduction to '{prereq}' or verify it's assumed knowledge"
                    ))

        # Check for large concept gaps
        prev_concepts: Set[str] = set()
        for sec_idx, lec_idx, lecture in lectures:
            current_prereqs = set(analysis.get(lecture.id, {}).get("prerequisites", []))
            current_intro = set(analysis.get(lecture.id, {}).get("introduces", []))

            # Check if there are many new prerequisites not in previous concepts
            gap = current_prereqs - prev_concepts
            if len(gap) > 3:
                issues.append(CoherenceIssue(
                    issue_type="concept_gap",
                    severity="info",
                    lecture_id=lecture.id,
                    lecture_title=lecture.title,
                    description=f"Large concept gap: {len(gap)} prerequisites not from previous lectures",
                    suggestion="Consider adding transition content or prerequisite check"
                ))

            # Update previous concepts
            prev_concepts.update(current_intro)
            prev_concepts.update(analysis.get(lecture.id, {}).get("key_concepts", []))

        return issues

    def _build_dependencies(
        self,
        analysis: Dict[str, Dict[str, Any]],
        lectures: List[Tuple[int, int, Lecture]],
    ) -> Dict[str, List[str]]:
        """Build lecture dependency mapping."""
        dependencies = {}

        # Create concept to lecture mapping
        concept_to_lecture = {}
        for lecture_id, data in analysis.items():
            for concept in data.get("introduces", []):
                concept_to_lecture[concept] = lecture_id

        # Build dependencies based on prerequisites
        for lecture_id, data in analysis.items():
            deps = []
            for prereq in data.get("prerequisites", []):
                if prereq in concept_to_lecture:
                    dep_lecture = concept_to_lecture[prereq]
                    if dep_lecture != lecture_id and dep_lecture not in deps:
                        deps.append(dep_lecture)
            dependencies[lecture_id] = deps

        return dependencies


# Singleton instance
_coherence_service = None


def get_coherence_service() -> CoherenceCheckService:
    """Get singleton coherence service instance."""
    global _coherence_service
    if _coherence_service is None:
        _coherence_service = CoherenceCheckService()
    return _coherence_service
