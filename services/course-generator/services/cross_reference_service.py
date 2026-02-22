"""
Cross-Reference Service

Manages cross-referencing between sources to:
1. Identify where the same concept is discussed differently
2. Consolidate information from multiple perspectives
3. Detect complementary and conflicting information
4. Build a unified view for course generation
"""
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import os

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    _USE_SHARED_LLM = False

from models.source_models import Source, PedagogicalRole
from services.knowledge_graph import KnowledgeGraph, Concept, CrossReference


@dataclass
class SourceContribution:
    """What a source contributes to a topic."""
    source_id: str
    source_name: str
    source_type: str
    pedagogical_role: str

    # Contributions
    provides_theory: bool = False
    provides_examples: bool = False
    provides_reference: bool = False
    provides_data: bool = False

    # Key insights from this source
    key_insights: List[str] = field(default_factory=list)

    # Unique content not in other sources
    unique_content: List[str] = field(default_factory=list)


@dataclass
class TopicCrossReference:
    """Cross-reference analysis for a specific topic."""
    topic: str

    # Contributing sources
    source_contributions: List[SourceContribution] = field(default_factory=list)

    # Consolidated view
    consolidated_definition: Optional[str] = None
    consolidated_examples: List[str] = field(default_factory=list)

    # Agreement/disagreement
    points_of_agreement: List[str] = field(default_factory=list)
    points_of_disagreement: List[str] = field(default_factory=list)

    # Completeness
    coverage_score: float = 0.0  # 0-1 how well sources cover the topic
    missing_aspects: List[str] = field(default_factory=list)


@dataclass
class CrossReferenceReport:
    """Complete cross-reference report for a course."""
    course_topic: str
    sources_analyzed: int

    # Topic-level cross-references
    topic_cross_refs: List[TopicCrossReference] = field(default_factory=list)

    # Source-level summaries
    source_summaries: Dict[str, SourceContribution] = field(default_factory=dict)

    # Overall metrics
    average_coverage: float = 0.0
    total_concepts_covered: int = 0
    concepts_with_multiple_sources: int = 0


class CrossReferenceService:
    """
    Service for analyzing cross-references between sources.

    This helps in:
    - Understanding how different sources complement each other
    - Identifying gaps in coverage
    - Building a complete picture from multiple perspectives
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        if _USE_SHARED_LLM:
            self.client = get_llm_client()
        else:
            self.client = AsyncOpenAI(
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                timeout=90.0,
                max_retries=2,
            )

    async def analyze_cross_references(
        self,
        sources: List[Source],
        knowledge_graph: KnowledgeGraph,
        topic: str,
        verbose: bool = False,
    ) -> CrossReferenceReport:
        """
        Analyze cross-references between sources.

        Args:
            sources: List of Source objects
            knowledge_graph: Pre-built knowledge graph
            topic: Course topic
            verbose: Print detailed progress

        Returns:
            CrossReferenceReport with analysis
        """
        print(f"[CROSS_REF] Analyzing cross-references for: {topic}", flush=True)
        print(f"[CROSS_REF] Sources: {len(sources)}, Concepts: {knowledge_graph.total_concepts}", flush=True)

        report = CrossReferenceReport(
            course_topic=topic,
            sources_analyzed=len(sources),
        )

        # Step 1: Analyze each source's contribution
        for source in sources:
            contribution = await self._analyze_source_contribution(source, topic)
            report.source_summaries[source.id] = contribution

        # Step 2: Analyze cross-references for key concepts
        key_concepts = self._get_key_concepts(knowledge_graph)

        for concept in key_concepts[:15]:  # Limit for API efficiency
            if verbose:
                print(f"[CROSS_REF] Analyzing: {concept.name}", flush=True)

            topic_ref = await self._analyze_concept_cross_reference(
                concept, sources, report.source_summaries
            )
            report.topic_cross_refs.append(topic_ref)

        # Step 3: Calculate metrics
        report.total_concepts_covered = knowledge_graph.total_concepts
        report.concepts_with_multiple_sources = len(knowledge_graph.cross_references)

        if report.topic_cross_refs:
            report.average_coverage = sum(
                t.coverage_score for t in report.topic_cross_refs
            ) / len(report.topic_cross_refs)

        print(f"[CROSS_REF] Analysis complete: {len(report.topic_cross_refs)} topics analyzed", flush=True)

        return report

    async def _analyze_source_contribution(
        self,
        source: Source,
        topic: str,
    ) -> SourceContribution:
        """Analyze what a single source contributes."""
        contribution = SourceContribution(
            source_id=source.id,
            source_name=source.name,
            source_type=source.source_type.value,
            pedagogical_role=source.pedagogical_role.value,
        )

        # Set flags based on pedagogical role
        role = source.pedagogical_role
        contribution.provides_theory = role in [PedagogicalRole.THEORY, PedagogicalRole.AUTO]
        contribution.provides_examples = role in [PedagogicalRole.EXAMPLE, PedagogicalRole.AUTO]
        contribution.provides_reference = role == PedagogicalRole.REFERENCE
        contribution.provides_data = role == PedagogicalRole.DATA

        # Extract key insights using AI
        if source.raw_content:
            content = source.raw_content[:4000]

            prompt = f"""Analyze this source's key contributions to the topic "{topic}":

Source: {source.name}
Type: {source.source_type.value}

Content excerpt:
{content}

Identify:
1. 3-5 key insights this source provides
2. Any unique content not typically found elsewhere

Return JSON:
{{
    "key_insights": ["insight1", "insight2"],
    "unique_content": ["unique1", "unique2"]
}}"""

            try:
                response = await self.client.chat.completions.create(
                    model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Analyze source contributions. Return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=500,
                )

                data = json.loads(response.choices[0].message.content)
                contribution.key_insights = data.get("key_insights", [])
                contribution.unique_content = data.get("unique_content", [])

            except Exception as e:
                print(f"[CROSS_REF] Contribution analysis error for {source.name}: {e}", flush=True)

        return contribution

    async def _analyze_concept_cross_reference(
        self,
        concept: Concept,
        sources: List[Source],
        source_summaries: Dict[str, SourceContribution],
    ) -> TopicCrossReference:
        """Analyze cross-references for a specific concept."""
        topic_ref = TopicCrossReference(topic=concept.name)

        # Gather definitions from all sources
        definitions_by_source = {}
        for defn in concept.definitions:
            if defn.source_id not in definitions_by_source:
                definitions_by_source[defn.source_id] = []
            definitions_by_source[defn.source_id].append(defn.definition_text)

        # Build source contributions for this topic
        for source_id, definitions in definitions_by_source.items():
            if source_id in source_summaries:
                summary = source_summaries[source_id]
                contribution = SourceContribution(
                    source_id=source_id,
                    source_name=summary.source_name,
                    source_type=summary.source_type,
                    pedagogical_role=summary.pedagogical_role,
                    provides_theory=summary.provides_theory,
                    provides_examples=summary.provides_examples,
                    provides_reference=summary.provides_reference,
                    provides_data=summary.provides_data,
                    key_insights=definitions,
                )
                topic_ref.source_contributions.append(contribution)

        # Analyze agreement/disagreement if multiple sources
        if len(topic_ref.source_contributions) > 1:
            await self._analyze_agreement(topic_ref)

        # Calculate coverage score
        role_coverage = {
            "theory": any(c.provides_theory for c in topic_ref.source_contributions),
            "examples": any(c.provides_examples for c in topic_ref.source_contributions),
            "reference": any(c.provides_reference for c in topic_ref.source_contributions),
            "data": any(c.provides_data for c in topic_ref.source_contributions),
        }

        covered_count = sum(1 for v in role_coverage.values() if v)
        topic_ref.coverage_score = covered_count / 4.0

        # Identify missing aspects
        if not role_coverage["theory"]:
            topic_ref.missing_aspects.append("theoretical foundation")
        if not role_coverage["examples"]:
            topic_ref.missing_aspects.append("practical examples")
        if not role_coverage["reference"]:
            topic_ref.missing_aspects.append("official documentation")
        if not role_coverage["data"]:
            topic_ref.missing_aspects.append("supporting data/statistics")

        # Use consolidated definition from knowledge graph if available
        if concept.consolidated_definition:
            topic_ref.consolidated_definition = concept.consolidated_definition

        return topic_ref

    async def _analyze_agreement(
        self,
        topic_ref: TopicCrossReference,
    ) -> None:
        """Analyze agreement/disagreement between sources on a topic."""
        if len(topic_ref.source_contributions) < 2:
            return

        # Gather all perspectives
        perspectives = []
        for contrib in topic_ref.source_contributions:
            for insight in contrib.key_insights:
                perspectives.append(f"[{contrib.source_name}]: {insight}")

        if not perspectives:
            return

        prompt = f"""Analyze these different perspectives on "{topic_ref.topic}":

{chr(10).join(perspectives)}

Identify:
1. Points where sources agree
2. Points where sources disagree or conflict
3. Any examples mentioned (consolidate into a list)

Return JSON:
{{
    "agreement": ["point1", "point2"],
    "disagreement": ["conflict1"],
    "examples": ["example1", "example2"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Analyze source agreement/disagreement. Return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500,
            )

            data = json.loads(response.choices[0].message.content)
            topic_ref.points_of_agreement = data.get("agreement", [])
            topic_ref.points_of_disagreement = data.get("disagreement", [])
            topic_ref.consolidated_examples = data.get("examples", [])

        except Exception as e:
            print(f"[CROSS_REF] Agreement analysis error for {topic_ref.topic}: {e}", flush=True)

    def _get_key_concepts(self, graph: KnowledgeGraph) -> List[Concept]:
        """Get the most important concepts from the graph."""
        # Sort by frequency (mentioned in multiple sources) and complexity
        concepts = list(graph.concepts.values())

        # Prioritize concepts with multiple sources
        concepts.sort(key=lambda c: (len(c.definitions), c.frequency), reverse=True)

        return concepts

    def get_cross_reference_summary(
        self,
        report: CrossReferenceReport,
    ) -> Dict[str, Any]:
        """Get a summary of cross-references for API response."""
        return {
            "course_topic": report.course_topic,
            "sources_analyzed": report.sources_analyzed,
            "total_concepts_covered": report.total_concepts_covered,
            "concepts_with_multiple_sources": report.concepts_with_multiple_sources,
            "average_coverage": round(report.average_coverage, 2),
            "source_summaries": [
                {
                    "source_id": s.source_id,
                    "source_name": s.source_name,
                    "pedagogical_role": s.pedagogical_role,
                    "key_insights_count": len(s.key_insights),
                    "unique_content_count": len(s.unique_content),
                }
                for s in report.source_summaries.values()
            ],
            "topic_cross_refs": [
                {
                    "topic": t.topic,
                    "sources_count": len(t.source_contributions),
                    "coverage_score": round(t.coverage_score, 2),
                    "has_agreement": len(t.points_of_agreement) > 0,
                    "has_conflicts": len(t.points_of_disagreement) > 0,
                    "missing_aspects": t.missing_aspects,
                }
                for t in report.topic_cross_refs
            ],
        }


# Singleton instance
_cross_reference_service = None


def get_cross_reference_service() -> CrossReferenceService:
    """Get singleton cross-reference service instance."""
    global _cross_reference_service
    if _cross_reference_service is None:
        _cross_reference_service = CrossReferenceService()
    return _cross_reference_service
