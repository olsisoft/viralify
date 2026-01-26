"""
Domain Discovery Engine

Layer 1 of the MAESTRO pipeline.
Discovers domain structure, themes, and concepts using LLM.
"""

import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from models.data_models import (
    Concept,
    DifficultyVector,
    ProgressionPath,
    SkillLevel,
    BloomLevel,
    SKILL_LEVEL_RANGES,
    PROGRESSION_RANGES,
)


DOMAIN_ANALYSIS_PROMPT = """Analyze the following subject and provide a comprehensive domain analysis.

SUBJECT: {subject}
PROGRESSION PATH: {progression_path}
LANGUAGE: {language}

Provide a domain analysis including:
1. A brief overview of the subject
2. 4-6 core themes that structure the subject
3. Key learning objectives (what learners will be able to do)
4. Prerequisite knowledge needed

Respond in JSON format:
{{
    "overview": "Brief overview of the subject domain",
    "core_themes": [
        {{
            "name": "Theme name",
            "description": "What this theme covers",
            "importance": "Why this theme is essential"
        }}
    ],
    "learning_objectives": [
        "Objective 1: What learners will achieve",
        "Objective 2: What learners will achieve"
    ],
    "prerequisite_knowledge": [
        "Prior knowledge or skill needed"
    ]
}}"""


CONCEPT_EXTRACTION_PROMPT = """Extract key concepts for this theme at the specified skill level.

SUBJECT: {subject}
THEME: {theme_name}
THEME DESCRIPTION: {theme_description}
TARGET SKILL LEVEL: {skill_level}
LANGUAGE: {language}

Extract 3-5 concepts appropriate for {skill_level} level learners.
For each concept, provide:
- A clear name
- A description
- Related keywords
- Bloom's taxonomy level (remember, understand, apply, analyze, evaluate, create)
- Difficulty scores (0.0-1.0) for:
  - conceptual_complexity: How abstract the concept is
  - prerequisites_depth: How many prerequisites needed
  - information_density: Amount of information to process
  - cognitive_load: Mental effort required
- Estimated duration in minutes (5-20)
- Prerequisites (names of concepts that should be learned first)

Respond in JSON format:
{{
    "concepts": [
        {{
            "name": "Concept name",
            "description": "Clear explanation of the concept",
            "keywords": ["keyword1", "keyword2"],
            "bloom_level": "understand",
            "conceptual_complexity": 0.3,
            "prerequisites_depth": 0.2,
            "information_density": 0.4,
            "cognitive_load": 0.25,
            "estimated_duration_minutes": 10,
            "prerequisites": ["Prerequisite concept name"]
        }}
    ]
}}"""


class DomainDiscoveryEngine:
    """
    Discovers domain structure and extracts concepts.

    Pipeline:
    1. Analyze domain to identify themes
    2. Extract concepts for each theme at appropriate levels
    3. Refine prerequisites between concepts
    """

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None, model: str = "gpt-4o-mini"):
        self.client = openai_client or AsyncOpenAI()
        self.model = model

    async def analyze_domain(
        self,
        subject: str,
        progression_path: ProgressionPath,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Analyze the subject domain to identify structure.

        Returns:
            Dict with overview, themes, objectives, prerequisites
        """
        print(f"[DOMAIN_DISCOVERY] Analyzing domain: {subject}", flush=True)

        prompt = DOMAIN_ANALYSIS_PROMPT.format(
            subject=subject,
            progression_path=progression_path.value,
            language=language,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert curriculum designer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
            )

            result = json.loads(response.choices[0].message.content)
            print(f"[DOMAIN_DISCOVERY] Found {len(result.get('core_themes', []))} themes", flush=True)
            return result

        except Exception as e:
            print(f"[DOMAIN_DISCOVERY] Error analyzing domain: {e}", flush=True)
            return {
                "overview": f"Course on {subject}",
                "core_themes": [{"name": subject, "description": f"Introduction to {subject}", "importance": "Core topic"}],
                "learning_objectives": [f"Understand {subject}"],
                "prerequisite_knowledge": [],
            }

    async def extract_concepts(
        self,
        subject: str,
        themes: List[Dict[str, Any]],
        progression_path: ProgressionPath,
        language: str = "en",
    ) -> List[Concept]:
        """
        Extract concepts from themes for the progression path.

        Args:
            subject: Course subject
            themes: List of theme dicts from domain analysis
            progression_path: Target progression
            language: Content language

        Returns:
            List of Concept objects
        """
        # Determine skill levels to cover
        start_level, end_level = PROGRESSION_RANGES[progression_path]
        skill_levels = self._get_skill_levels_in_range(start_level, end_level)

        print(f"[DOMAIN_DISCOVERY] Extracting concepts for levels: {[s.value for s in skill_levels]}", flush=True)

        all_concepts = []
        concept_counter = 0

        for theme in themes:
            for skill_level in skill_levels:
                concepts = await self._extract_concepts_for_theme(
                    subject=subject,
                    theme=theme,
                    skill_level=skill_level,
                    language=language,
                )

                # Assign unique IDs
                for concept in concepts:
                    concept.id = f"c_{concept_counter:03d}"
                    concept_counter += 1
                    all_concepts.append(concept)

        print(f"[DOMAIN_DISCOVERY] Extracted {len(all_concepts)} total concepts", flush=True)

        # Refine prerequisites
        all_concepts = self._refine_prerequisites(all_concepts)

        return all_concepts

    async def _extract_concepts_for_theme(
        self,
        subject: str,
        theme: Dict[str, Any],
        skill_level: SkillLevel,
        language: str,
    ) -> List[Concept]:
        """Extract concepts for a single theme at a skill level"""
        prompt = CONCEPT_EXTRACTION_PROMPT.format(
            subject=subject,
            theme_name=theme.get("name", ""),
            theme_description=theme.get("description", ""),
            skill_level=skill_level.value,
            language=language,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert curriculum designer specializing in concept extraction."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
            )

            data = json.loads(response.choices[0].message.content)
            concepts_data = data.get("concepts", [])

            concepts = []
            for c_data in concepts_data:
                difficulty = DifficultyVector(
                    conceptual_complexity=c_data.get("conceptual_complexity", 0.5),
                    prerequisites_depth=c_data.get("prerequisites_depth", 0.5),
                    information_density=c_data.get("information_density", 0.5),
                    cognitive_load=c_data.get("cognitive_load", 0.5),
                )

                concept = Concept(
                    name=c_data.get("name", ""),
                    description=c_data.get("description", ""),
                    keywords=c_data.get("keywords", []),
                    difficulty=difficulty,
                    prerequisites=c_data.get("prerequisites", []),
                    estimated_duration_minutes=c_data.get("estimated_duration_minutes", 10),
                )
                concepts.append(concept)

            return concepts

        except Exception as e:
            print(f"[DOMAIN_DISCOVERY] Error extracting concepts for theme {theme.get('name')}: {e}", flush=True)
            return []

    def _get_skill_levels_in_range(
        self,
        start: SkillLevel,
        end: SkillLevel,
    ) -> List[SkillLevel]:
        """Get all skill levels between start and end (inclusive)"""
        all_levels = list(SkillLevel)
        start_idx = all_levels.index(start)
        end_idx = all_levels.index(end)
        return all_levels[start_idx:end_idx + 1]

    def _refine_prerequisites(self, concepts: List[Concept]) -> List[Concept]:
        """
        Refine prerequisites to use concept IDs instead of names.
        Also ensures no circular dependencies.
        """
        # Build name to ID mapping
        name_to_id = {c.name.lower(): c.id for c in concepts}

        for concept in concepts:
            refined_prereqs = []
            for prereq_name in concept.prerequisites:
                prereq_id = name_to_id.get(prereq_name.lower())
                if prereq_id and prereq_id != concept.id:
                    refined_prereqs.append(prereq_id)
            concept.prerequisites = refined_prereqs

        # Remove circular dependencies using DFS
        def has_cycle(concept_id: str, visited: set, rec_stack: set) -> bool:
            visited.add(concept_id)
            rec_stack.add(concept_id)

            concept = next((c for c in concepts if c.id == concept_id), None)
            if concept:
                for prereq_id in concept.prerequisites:
                    if prereq_id not in visited:
                        if has_cycle(prereq_id, visited, rec_stack):
                            return True
                    elif prereq_id in rec_stack:
                        # Found cycle - remove this prerequisite
                        concept.prerequisites.remove(prereq_id)
                        return False

            rec_stack.remove(concept_id)
            return False

        visited = set()
        for concept in concepts:
            if concept.id not in visited:
                has_cycle(concept.id, visited, set())

        return concepts


async def discover_domain(
    subject: str,
    progression_path: str = "beginner_to_intermediate",
    language: str = "en",
) -> tuple[Dict[str, Any], List[Concept]]:
    """
    Convenience function to discover domain and extract concepts.

    Example:
        domain, concepts = await discover_domain(
            subject="Python Programming",
            progression_path="beginner_to_intermediate"
        )
    """
    engine = DomainDiscoveryEngine()
    path = ProgressionPath(progression_path)

    domain = await engine.analyze_domain(subject, path, language)
    concepts = await engine.extract_concepts(
        subject=subject,
        themes=domain.get("core_themes", []),
        progression_path=path,
        language=language,
    )

    return domain, concepts
