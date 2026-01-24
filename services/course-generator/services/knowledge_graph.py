"""
Knowledge Graph Service

Builds and manages a knowledge graph of concepts extracted from:
1. Source documents (PDFs, videos, URLs, notes)
2. Course lectures (generated content)

The knowledge graph enables:
- Concept extraction and linking
- Cross-reference detection between sources
- Consolidated definitions from multiple perspectives
- Prerequisite mapping
"""
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from openai import AsyncOpenAI
import os

from models.source_models import Source, PedagogicalRole


@dataclass
class ConceptDefinition:
    """A definition of a concept from a specific source."""
    source_id: str
    source_name: str
    source_type: str  # file, youtube, url, note
    pedagogical_role: str  # theory, example, reference, etc.
    definition_text: str
    context: str  # surrounding context
    location: str  # page number, timestamp, section
    confidence: float  # 0-1 confidence in extraction


@dataclass
class Concept:
    """A concept in the knowledge graph."""
    id: str
    name: str
    canonical_name: str  # Normalized name for matching
    aliases: List[str] = field(default_factory=list)

    # Definitions from different sources
    definitions: List[ConceptDefinition] = field(default_factory=list)

    # Consolidated definition (synthesized from all sources)
    consolidated_definition: Optional[str] = None

    # Relationships
    prerequisites: List[str] = field(default_factory=list)  # concept IDs
    related_concepts: List[str] = field(default_factory=list)  # concept IDs
    parent_concepts: List[str] = field(default_factory=list)  # broader concepts
    child_concepts: List[str] = field(default_factory=list)  # narrower concepts

    # Metadata
    complexity_level: int = 3  # 1-5 complexity scale
    domain_tags: List[str] = field(default_factory=list)
    first_seen_in: Optional[str] = None  # source_id where first mentioned
    frequency: int = 0  # number of times mentioned across sources

    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CrossReference:
    """A cross-reference between sources on a concept."""
    concept_id: str
    concept_name: str

    # Sources that discuss this concept
    source_ids: List[str] = field(default_factory=list)

    # How sources relate
    agreement_score: float = 0.0  # 0-1 how much sources agree
    complementary_aspects: List[str] = field(default_factory=list)  # unique insights per source
    conflicts: List[str] = field(default_factory=list)  # conflicting information


@dataclass
class KnowledgeGraph:
    """Complete knowledge graph for a course."""
    course_id: Optional[str] = None

    # Core data
    concepts: Dict[str, Concept] = field(default_factory=dict)  # concept_id -> Concept
    cross_references: List[CrossReference] = field(default_factory=list)

    # Indices for fast lookup
    concept_by_name: Dict[str, str] = field(default_factory=dict)  # name -> concept_id
    concepts_by_source: Dict[str, List[str]] = field(default_factory=dict)  # source_id -> [concept_ids]

    # Statistics
    total_concepts: int = 0
    total_cross_references: int = 0
    sources_analyzed: int = 0

    created_at: datetime = field(default_factory=datetime.utcnow)


class KnowledgeGraphBuilder:
    """
    Service for building and managing knowledge graphs.

    Extracts concepts from sources and builds relationships between them.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.client = AsyncOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            timeout=90.0,
            max_retries=2,
        )

    async def build_knowledge_graph(
        self,
        sources: List[Source],
        topic: str,
        course_id: Optional[str] = None,
        verbose: bool = False,
    ) -> KnowledgeGraph:
        """
        Build a knowledge graph from a list of sources.

        Args:
            sources: List of Source objects with raw_content
            topic: Course topic for context
            course_id: Optional course ID
            verbose: If True, print detailed progress

        Returns:
            KnowledgeGraph with extracted concepts and relationships
        """
        print(f"[KNOWLEDGE_GRAPH] Building graph for topic: {topic}", flush=True)
        print(f"[KNOWLEDGE_GRAPH] Analyzing {len(sources)} sources", flush=True)

        graph = KnowledgeGraph(course_id=course_id)

        # Step 1: Extract concepts from each source
        for source in sources:
            if not source.raw_content:
                continue

            if verbose:
                print(f"[KNOWLEDGE_GRAPH] Extracting from: {source.name}", flush=True)

            concepts = await self._extract_concepts_from_source(source, topic)
            graph.sources_analyzed += 1

            for concept in concepts:
                self._add_concept_to_graph(graph, concept, source)

        # Step 2: Build relationships between concepts
        print(f"[KNOWLEDGE_GRAPH] Building relationships for {len(graph.concepts)} concepts", flush=True)
        await self._build_relationships(graph, topic)

        # Step 3: Identify cross-references
        cross_refs = self._identify_cross_references(graph)
        graph.cross_references = cross_refs
        graph.total_cross_references = len(cross_refs)

        # Step 4: Generate consolidated definitions
        await self._generate_consolidated_definitions(graph, topic)

        graph.total_concepts = len(graph.concepts)

        print(f"[KNOWLEDGE_GRAPH] Built graph: {graph.total_concepts} concepts, {graph.total_cross_references} cross-refs", flush=True)

        return graph

    async def _extract_concepts_from_source(
        self,
        source: Source,
        topic: str,
    ) -> List[Dict[str, Any]]:
        """Extract concepts from a single source."""
        # Truncate content for API limits
        content = source.raw_content[:8000] if source.raw_content else ""

        if not content:
            return []

        prompt = f"""Extract key concepts from this source document related to the topic: "{topic}"

Source: {source.name}
Type: {source.source_type.value}
Role: {source.pedagogical_role.value}

Content:
{content}

For each concept, extract:
1. "name": The concept name (specific, not too broad)
2. "definition": How this source defines/explains the concept
3. "context": A brief quote or context where it appears
4. "complexity": Complexity level 1-5 (1=basic, 5=expert)
5. "related": Other concepts it relates to (list)

Return a JSON array of concepts:
[
    {{
        "name": "concept name",
        "definition": "how the source explains it",
        "context": "relevant quote or context",
        "complexity": 3,
        "related": ["related concept 1", "related concept 2"]
    }}
]

Focus on:
- Technical terms and methodologies
- Key frameworks or patterns
- Important tools or technologies
- Core principles and theories

Limit to the 10-15 most important concepts. Return valid JSON only."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting key concepts from educational content. Return valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()
            # Handle both array and object responses
            data = json.loads(content)
            if isinstance(data, dict):
                data = data.get("concepts", [])
            return data

        except Exception as e:
            print(f"[KNOWLEDGE_GRAPH] Extraction error for {source.name}: {e}", flush=True)
            return []

    def _add_concept_to_graph(
        self,
        graph: KnowledgeGraph,
        concept_data: Dict[str, Any],
        source: Source,
    ) -> None:
        """Add a concept to the graph, merging if it already exists."""
        name = concept_data.get("name", "").strip()
        if not name:
            return

        canonical_name = self._normalize_name(name)

        # Check if concept already exists
        existing_id = graph.concept_by_name.get(canonical_name)

        if existing_id and existing_id in graph.concepts:
            # Merge with existing concept
            concept = graph.concepts[existing_id]
            concept.frequency += 1

            # Add definition from this source
            definition = ConceptDefinition(
                source_id=source.id,
                source_name=source.name,
                source_type=source.source_type.value,
                pedagogical_role=source.pedagogical_role.value,
                definition_text=concept_data.get("definition", ""),
                context=concept_data.get("context", ""),
                location="",  # Could be enhanced with chunk location
                confidence=0.8,
            )
            concept.definitions.append(definition)

            # Add aliases if name differs
            if name != concept.name and name not in concept.aliases:
                concept.aliases.append(name)

        else:
            # Create new concept
            import uuid
            concept_id = f"concept_{uuid.uuid4().hex[:8]}"

            definition = ConceptDefinition(
                source_id=source.id,
                source_name=source.name,
                source_type=source.source_type.value,
                pedagogical_role=source.pedagogical_role.value,
                definition_text=concept_data.get("definition", ""),
                context=concept_data.get("context", ""),
                location="",
                confidence=0.8,
            )

            concept = Concept(
                id=concept_id,
                name=name,
                canonical_name=canonical_name,
                definitions=[definition],
                complexity_level=concept_data.get("complexity", 3),
                related_concepts=concept_data.get("related", []),
                first_seen_in=source.id,
                frequency=1,
            )

            graph.concepts[concept_id] = concept
            graph.concept_by_name[canonical_name] = concept_id

        # Track source -> concepts mapping
        if source.id not in graph.concepts_by_source:
            graph.concepts_by_source[source.id] = []
        if concept.id not in graph.concepts_by_source[source.id]:
            graph.concepts_by_source[source.id].append(concept.id)

    async def _build_relationships(
        self,
        graph: KnowledgeGraph,
        topic: str,
    ) -> None:
        """Build prerequisite and hierarchical relationships between concepts."""
        if len(graph.concepts) < 2:
            return

        concept_names = [c.name for c in graph.concepts.values()][:30]  # Limit for API

        prompt = f"""Analyze the relationships between these concepts related to "{topic}":

Concepts: {json.dumps(concept_names, ensure_ascii=False)}

Identify:
1. Prerequisites: Which concepts must be understood before others
2. Hierarchies: Parent/child relationships (broader/narrower)

Return JSON:
{{
    "prerequisites": [
        {{"concept": "concept_name", "requires": ["prereq1", "prereq2"]}}
    ],
    "hierarchies": [
        {{"parent": "broad concept", "children": ["specific1", "specific2"]}}
    ]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze concept relationships for educational content. Return valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000,
            )

            data = json.loads(response.choices[0].message.content)

            # Apply prerequisites
            for prereq_item in data.get("prerequisites", []):
                concept_name = prereq_item.get("concept", "")
                requires = prereq_item.get("requires", [])

                canonical = self._normalize_name(concept_name)
                if canonical in graph.concept_by_name:
                    concept_id = graph.concept_by_name[canonical]
                    concept = graph.concepts[concept_id]

                    for req in requires:
                        req_canonical = self._normalize_name(req)
                        if req_canonical in graph.concept_by_name:
                            req_id = graph.concept_by_name[req_canonical]
                            if req_id not in concept.prerequisites:
                                concept.prerequisites.append(req_id)

            # Apply hierarchies
            for hier in data.get("hierarchies", []):
                parent_name = hier.get("parent", "")
                children = hier.get("children", [])

                parent_canonical = self._normalize_name(parent_name)
                if parent_canonical in graph.concept_by_name:
                    parent_id = graph.concept_by_name[parent_canonical]
                    parent_concept = graph.concepts[parent_id]

                    for child_name in children:
                        child_canonical = self._normalize_name(child_name)
                        if child_canonical in graph.concept_by_name:
                            child_id = graph.concept_by_name[child_canonical]
                            child_concept = graph.concepts[child_id]

                            if child_id not in parent_concept.child_concepts:
                                parent_concept.child_concepts.append(child_id)
                            if parent_id not in child_concept.parent_concepts:
                                child_concept.parent_concepts.append(parent_id)

        except Exception as e:
            print(f"[KNOWLEDGE_GRAPH] Relationship building error: {e}", flush=True)

    def _identify_cross_references(
        self,
        graph: KnowledgeGraph,
    ) -> List[CrossReference]:
        """Identify concepts that appear in multiple sources."""
        cross_refs = []

        for concept in graph.concepts.values():
            source_ids = list(set(d.source_id for d in concept.definitions))

            if len(source_ids) > 1:
                # This concept has multiple sources
                complementary = []
                for defn in concept.definitions:
                    aspect = f"{defn.source_name} ({defn.pedagogical_role}): {defn.definition_text[:100]}..."
                    complementary.append(aspect)

                cross_ref = CrossReference(
                    concept_id=concept.id,
                    concept_name=concept.name,
                    source_ids=source_ids,
                    agreement_score=0.8,  # Could be calculated more precisely
                    complementary_aspects=complementary,
                    conflicts=[],
                )
                cross_refs.append(cross_ref)

        return cross_refs

    async def _generate_consolidated_definitions(
        self,
        graph: KnowledgeGraph,
        topic: str,
    ) -> None:
        """Generate consolidated definitions for concepts with multiple sources."""
        # Only consolidate concepts with multiple definitions
        concepts_to_consolidate = [
            c for c in graph.concepts.values()
            if len(c.definitions) > 1
        ][:10]  # Limit for API calls

        if not concepts_to_consolidate:
            return

        for concept in concepts_to_consolidate:
            definitions_text = "\n".join([
                f"- [{d.source_name}] ({d.pedagogical_role}): {d.definition_text}"
                for d in concept.definitions
            ])

            prompt = f"""Synthesize a consolidated definition for the concept "{concept.name}" based on these perspectives:

{definitions_text}

Create a single, comprehensive definition that:
1. Incorporates insights from all sources
2. Notes any differences in perspective
3. Provides a clear, educational explanation

Keep the consolidated definition to 2-3 sentences."""

            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert at synthesizing educational content."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    max_tokens=300,
                )

                concept.consolidated_definition = response.choices[0].message.content.strip()

            except Exception as e:
                print(f"[KNOWLEDGE_GRAPH] Consolidation error for {concept.name}: {e}", flush=True)

    def _normalize_name(self, name: str) -> str:
        """Normalize a concept name for matching."""
        return name.lower().strip().replace("-", " ").replace("_", " ")

    def get_concept_summary(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Get a summary of the knowledge graph for API responses."""
        return {
            "total_concepts": graph.total_concepts,
            "total_cross_references": graph.total_cross_references,
            "sources_analyzed": graph.sources_analyzed,
            "concepts": [
                {
                    "id": c.id,
                    "name": c.name,
                    "complexity": c.complexity_level,
                    "sources_count": len(c.definitions),
                    "has_consolidated_definition": c.consolidated_definition is not None,
                    "prerequisites_count": len(c.prerequisites),
                }
                for c in graph.concepts.values()
            ],
            "cross_references": [
                {
                    "concept": cr.concept_name,
                    "sources_count": len(cr.source_ids),
                    "agreement_score": cr.agreement_score,
                }
                for cr in graph.cross_references
            ],
        }


# Singleton instance
_knowledge_graph_builder = None


def get_knowledge_graph_builder() -> KnowledgeGraphBuilder:
    """Get singleton knowledge graph builder instance."""
    global _knowledge_graph_builder
    if _knowledge_graph_builder is None:
        _knowledge_graph_builder = KnowledgeGraphBuilder()
    return _knowledge_graph_builder
