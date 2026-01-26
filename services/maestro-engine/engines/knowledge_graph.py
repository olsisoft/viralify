"""
Knowledge Graph Engine

Layer 2 of the MAESTRO pipeline.
Builds a prerequisite graph between concepts with topological ordering.
"""

from typing import List, Dict, Any, Set, Optional
from collections import defaultdict, deque

from models.data_models import Concept, SkillLevel


class KnowledgeGraphEngine:
    """
    Builds and manages the concept prerequisite graph.

    Features:
    - Topological ordering (Kahn's algorithm)
    - Cycle detection and resolution
    - Learning path generation
    - Prerequisite chain analysis
    """

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # concept_id -> prereqs
        self.reverse_adj: Dict[str, List[str]] = defaultdict(list)  # prereq -> concepts that need it

    def build_graph(self, concepts: List[Concept]) -> None:
        """
        Build the prerequisite graph from concepts.

        Args:
            concepts: List of Concept objects with prerequisites
        """
        self.concepts = {c.id: c for c in concepts}
        self.adjacency = defaultdict(list)
        self.reverse_adj = defaultdict(list)

        for concept in concepts:
            for prereq_id in concept.prerequisites:
                if prereq_id in self.concepts:
                    self.adjacency[concept.id].append(prereq_id)
                    self.reverse_adj[prereq_id].append(concept.id)

        print(f"[KNOWLEDGE_GRAPH] Built graph with {len(self.concepts)} concepts", flush=True)

    def topological_sort(self) -> List[str]:
        """
        Perform topological sort using Kahn's algorithm.

        Returns:
            List of concept IDs in topological order (prerequisites first)
        """
        # Calculate in-degrees
        in_degree = {cid: 0 for cid in self.concepts}
        for concept_id, prereqs in self.adjacency.items():
            in_degree[concept_id] = len(prereqs)

        # Start with concepts that have no prerequisites
        queue = deque([cid for cid, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # Reduce in-degree for concepts that depend on current
            for dependent in self.reverse_adj[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check for cycles
        if len(result) != len(self.concepts):
            print(f"[KNOWLEDGE_GRAPH] Warning: Cycle detected, {len(self.concepts) - len(result)} concepts unreachable", flush=True)
            # Add remaining concepts at the end
            remaining = [cid for cid in self.concepts if cid not in result]
            result.extend(remaining)

        return result

    def get_learning_order(self) -> List[Concept]:
        """
        Get concepts in learning order (prerequisites first).

        Returns:
            List of Concept objects in topological order
        """
        order = self.topological_sort()
        return [self.concepts[cid] for cid in order if cid in self.concepts]

    def get_prerequisite_chain(self, concept_id: str) -> List[str]:
        """
        Get all transitive prerequisites for a concept.

        Args:
            concept_id: The concept to analyze

        Returns:
            List of prerequisite concept IDs in order
        """
        visited = set()
        chain = []

        def dfs(cid: str):
            if cid in visited:
                return
            visited.add(cid)
            for prereq in self.adjacency.get(cid, []):
                dfs(prereq)
            chain.append(cid)

        if concept_id in self.concepts:
            dfs(concept_id)
            # Remove the concept itself from its chain
            if chain and chain[-1] == concept_id:
                chain.pop()

        return chain

    def get_dependent_concepts(self, concept_id: str) -> List[str]:
        """
        Get all concepts that depend on this concept.

        Args:
            concept_id: The concept to analyze

        Returns:
            List of dependent concept IDs
        """
        visited = set()
        dependents = []

        def dfs(cid: str):
            for dependent in self.reverse_adj.get(cid, []):
                if dependent not in visited:
                    visited.add(dependent)
                    dependents.append(dependent)
                    dfs(dependent)

        if concept_id in self.concepts:
            dfs(concept_id)

        return dependents

    def validate_prerequisites(self) -> List[Dict[str, Any]]:
        """
        Validate the prerequisite graph for issues.

        Returns:
            List of validation issues found
        """
        issues = []

        # Check for missing prerequisites
        for concept_id, prereqs in self.adjacency.items():
            for prereq in prereqs:
                if prereq not in self.concepts:
                    issues.append({
                        "type": "missing_prerequisite",
                        "concept_id": concept_id,
                        "missing_prereq": prereq,
                    })

        # Check for cycles
        visited = set()
        rec_stack = set()

        def has_cycle(cid: str) -> bool:
            visited.add(cid)
            rec_stack.add(cid)

            for prereq in self.adjacency.get(cid, []):
                if prereq not in visited:
                    if has_cycle(prereq):
                        return True
                elif prereq in rec_stack:
                    issues.append({
                        "type": "cycle_detected",
                        "concept_id": cid,
                        "cycle_with": prereq,
                    })
                    return True

            rec_stack.remove(cid)
            return False

        for cid in self.concepts:
            if cid not in visited:
                has_cycle(cid)

        # Check for skill level consistency
        for concept_id, prereqs in self.adjacency.items():
            concept = self.concepts[concept_id]
            for prereq_id in prereqs:
                prereq = self.concepts.get(prereq_id)
                if prereq and prereq.skill_level.value > concept.skill_level.value:
                    issues.append({
                        "type": "skill_level_inconsistency",
                        "concept_id": concept_id,
                        "concept_level": concept.skill_level.value,
                        "prereq_id": prereq_id,
                        "prereq_level": prereq.skill_level.value,
                    })

        return issues

    def get_concepts_by_level(self, skill_level: SkillLevel) -> List[Concept]:
        """
        Get all concepts at a specific skill level.

        Args:
            skill_level: The target skill level

        Returns:
            List of concepts at that level
        """
        return [c for c in self.concepts.values() if c.skill_level == skill_level]

    def get_entry_points(self) -> List[Concept]:
        """
        Get concepts with no prerequisites (entry points).

        Returns:
            List of entry point concepts
        """
        return [
            self.concepts[cid]
            for cid in self.concepts
            if not self.adjacency.get(cid)
        ]

    def get_terminal_points(self) -> List[Concept]:
        """
        Get concepts that no other concept depends on (terminal points).

        Returns:
            List of terminal concepts
        """
        return [
            self.concepts[cid]
            for cid in self.concepts
            if not self.reverse_adj.get(cid)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Export graph structure as dictionary"""
        return {
            "concepts": [c.to_dict() for c in self.concepts.values()],
            "edges": [
                {"from": prereq, "to": cid}
                for cid, prereqs in self.adjacency.items()
                for prereq in prereqs
            ],
            "entry_points": [c.id for c in self.get_entry_points()],
            "terminal_points": [c.id for c in self.get_terminal_points()],
        }


def build_knowledge_graph(concepts: List[Concept]) -> KnowledgeGraphEngine:
    """
    Convenience function to build a knowledge graph.

    Example:
        graph = build_knowledge_graph(concepts)
        learning_order = graph.get_learning_order()
    """
    engine = KnowledgeGraphEngine()
    engine.build_graph(concepts)
    return engine
