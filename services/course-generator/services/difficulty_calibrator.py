"""
Difficulty Calibrator Service

This service calibrates concepts extracted from RAG with 4D difficulty vectors.
It uses LLM to analyze each concept and assign difficulty scores across
4 dimensions: conceptual complexity, prerequisites depth, information density,
and cognitive load (Bloom's taxonomy).

Integration with existing KnowledgeGraphBuilder:
- Takes concepts from knowledge graph
- Enriches them with difficulty vectors
- Enables smooth difficulty progression planning
"""

import asyncio
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    _USE_SHARED_LLM = False

from models.difficulty_models import (
    DifficultyVector,
    CalibratedConcept,
    DifficultyProgression,
    BloomLevel,
    SkillLevel,
    BLOOM_TO_COGNITIVE_LOAD,
)


# Prompt for difficulty calibration
CALIBRATION_PROMPT = """Analyze the following concept and provide a 4-dimensional difficulty assessment.

CONCEPT:
Name: {concept_name}
Description: {concept_description}
Keywords: {keywords}
Prerequisites: {prerequisites}

CONTEXT:
Course Subject: {course_subject}
Target Audience Level: {target_level}
Language: {language}

Provide difficulty scores for each dimension (0.0 to 1.0):

1. CONCEPTUAL_COMPLEXITY (0.0-1.0)
   - 0.0-0.2: Very concrete, tangible concepts
   - 0.2-0.4: Somewhat abstract but relatable
   - 0.4-0.6: Moderately abstract, requires mental models
   - 0.6-0.8: Highly abstract, theoretical concepts
   - 0.8-1.0: Extremely abstract, cutting-edge theory

2. PREREQUISITES_DEPTH (0.0-1.0)
   - 0.0-0.2: No prerequisites needed
   - 0.2-0.4: Basic foundational knowledge
   - 0.4-0.6: Several prerequisite concepts required
   - 0.6-0.8: Deep prerequisite chain
   - 0.8-1.0: Extensive prior expertise required

3. INFORMATION_DENSITY (0.0-1.0)
   - 0.0-0.2: Simple, few facts to remember
   - 0.2-0.4: Moderate amount of information
   - 0.4-0.6: Substantial content to process
   - 0.6-0.8: Dense, complex information
   - 0.8-1.0: Extremely dense, overwhelming detail

4. COGNITIVE_LOAD (0.0-1.0) - Based on Bloom's Taxonomy:
   - 0.0-0.15: Remember (recall facts)
   - 0.15-0.35: Understand (explain concepts)
   - 0.35-0.50: Apply (use in new situations)
   - 0.50-0.70: Analyze (draw connections)
   - 0.70-0.85: Evaluate (justify decisions)
   - 0.85-1.0: Create (produce original work)

Also provide:
- BLOOM_LEVEL: The primary Bloom's taxonomy level (remember, understand, apply, analyze, evaluate, create)
- ESTIMATED_DURATION_MINUTES: How long to teach this concept (5-30 minutes)

Respond in JSON format:
{{
    "conceptual_complexity": <float>,
    "prerequisites_depth": <float>,
    "information_density": <float>,
    "cognitive_load": <float>,
    "bloom_level": "<string>",
    "estimated_duration_minutes": <int>,
    "reasoning": "<brief explanation>"
}}"""


BATCH_CALIBRATION_PROMPT = """Analyze the following concepts and provide 4-dimensional difficulty assessments for each.

CONCEPTS:
{concepts_json}

CONTEXT:
Course Subject: {course_subject}
Target Audience Level: {target_level}
Language: {language}

For EACH concept, provide difficulty scores (0.0-1.0) for:
1. conceptual_complexity - How abstract is the concept
2. prerequisites_depth - How many prerequisites needed
3. information_density - Amount of information to process
4. cognitive_load - Mental effort (Bloom's level: 0.1=remember, 0.25=understand, 0.45=apply, 0.6=analyze, 0.8=evaluate, 0.95=create)

Also provide bloom_level and estimated_duration_minutes for each.

Respond in JSON format:
{{
    "calibrations": [
        {{
            "concept_id": "<id>",
            "conceptual_complexity": <float>,
            "prerequisites_depth": <float>,
            "information_density": <float>,
            "cognitive_load": <float>,
            "bloom_level": "<string>",
            "estimated_duration_minutes": <int>
        }},
        ...
    ]
}}"""


class DifficultyCalibratorService:
    """
    Service for calibrating concept difficulty using 4D vectors.

    Integrates with the existing knowledge graph system to:
    1. Take concepts extracted from RAG documents
    2. Analyze each concept for difficulty dimensions
    3. Assign calibrated difficulty vectors
    4. Enable smooth progression planning
    """

    def __init__(
        self,
        openai_client=None,
        model: Optional[str] = None,
        batch_size: int = 10,
        cache_enabled: bool = True,
    ):
        if _USE_SHARED_LLM:
            self.client = openai_client or get_llm_client()
            self.model = model or get_model_name("fast")
        else:
            from openai import AsyncOpenAI as _AsyncOpenAI
            self.client = openai_client or _AsyncOpenAI()
            self.model = model or "gpt-4o-mini"
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, DifficultyVector] = {}

    def _cache_key(self, concept_name: str, description: str) -> str:
        """Generate cache key for a concept"""
        content = f"{concept_name}:{description}"
        return hashlib.md5(content.encode()).hexdigest()

    async def calibrate_concept(
        self,
        concept_name: str,
        concept_description: str,
        keywords: List[str] = None,
        prerequisites: List[str] = None,
        course_subject: str = "",
        target_level: str = "intermediate",
        language: str = "en",
    ) -> Tuple[DifficultyVector, Dict[str, Any]]:
        """
        Calibrate a single concept with 4D difficulty vector.

        Returns:
            Tuple of (DifficultyVector, metadata dict with bloom_level, duration, reasoning)
        """
        # Check cache
        cache_key = self._cache_key(concept_name, concept_description)
        if self.cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            return cached, {"from_cache": True}

        prompt = CALIBRATION_PROMPT.format(
            concept_name=concept_name,
            concept_description=concept_description,
            keywords=", ".join(keywords or []),
            prerequisites=", ".join(prerequisites or ["None"]),
            course_subject=course_subject,
            target_level=target_level,
            language=language,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational content analyst specializing in difficulty calibration and Bloom's taxonomy."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            result = json.loads(response.choices[0].message.content)

            difficulty = DifficultyVector(
                conceptual_complexity=float(result.get("conceptual_complexity", 0.5)),
                prerequisites_depth=float(result.get("prerequisites_depth", 0.5)),
                information_density=float(result.get("information_density", 0.5)),
                cognitive_load=float(result.get("cognitive_load", 0.5)),
            )

            metadata = {
                "bloom_level": result.get("bloom_level", "understand"),
                "estimated_duration_minutes": result.get("estimated_duration_minutes", 10),
                "reasoning": result.get("reasoning", ""),
                "from_cache": False,
            }

            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = difficulty

            return difficulty, metadata

        except Exception as e:
            print(f"[DIFFICULTY_CALIBRATOR] Error calibrating '{concept_name}': {e}", flush=True)
            # Return default difficulty on error
            return DifficultyVector(), {"error": str(e), "from_cache": False}

    async def calibrate_concepts_batch(
        self,
        concepts: List[Dict[str, Any]],
        course_subject: str = "",
        target_level: str = "intermediate",
        language: str = "en",
    ) -> List[CalibratedConcept]:
        """
        Calibrate multiple concepts in batches for efficiency.

        Args:
            concepts: List of dicts with 'id', 'name', 'description', 'keywords', 'prerequisites'
            course_subject: Overall course subject for context
            target_level: Target audience level
            language: Content language

        Returns:
            List of CalibratedConcept objects
        """
        calibrated = []

        # Process in batches
        for i in range(0, len(concepts), self.batch_size):
            batch = concepts[i:i + self.batch_size]
            batch_results = await self._calibrate_batch(
                batch, course_subject, target_level, language
            )
            calibrated.extend(batch_results)

        print(f"[DIFFICULTY_CALIBRATOR] Calibrated {len(calibrated)} concepts", flush=True)
        return calibrated

    async def _calibrate_batch(
        self,
        concepts: List[Dict[str, Any]],
        course_subject: str,
        target_level: str,
        language: str,
    ) -> List[CalibratedConcept]:
        """Calibrate a batch of concepts in a single LLM call"""
        # Prepare concepts JSON
        concepts_for_prompt = [
            {
                "concept_id": c.get("id", f"concept_{i}"),
                "name": c.get("name", ""),
                "description": c.get("description", "")[:500],  # Truncate for token efficiency
                "keywords": c.get("keywords", [])[:5],
                "prerequisites": c.get("prerequisites", []),
            }
            for i, c in enumerate(concepts)
        ]

        prompt = BATCH_CALIBRATION_PROMPT.format(
            concepts_json=json.dumps(concepts_for_prompt, indent=2),
            course_subject=course_subject,
            target_level=target_level,
            language=language,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational content analyst."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            result = json.loads(response.choices[0].message.content)
            calibrations = result.get("calibrations", [])

            # Map calibrations back to concepts
            calibration_map = {c["concept_id"]: c for c in calibrations}

            calibrated_concepts = []
            for i, concept in enumerate(concepts):
                concept_id = concept.get("id", f"concept_{i}")
                cal = calibration_map.get(concept_id, {})

                difficulty = DifficultyVector(
                    conceptual_complexity=float(cal.get("conceptual_complexity", 0.5)),
                    prerequisites_depth=float(cal.get("prerequisites_depth", 0.5)),
                    information_density=float(cal.get("information_density", 0.5)),
                    cognitive_load=float(cal.get("cognitive_load", 0.5)),
                )

                calibrated_concepts.append(CalibratedConcept(
                    concept_id=concept_id,
                    name=concept.get("name", ""),
                    description=concept.get("description", ""),
                    difficulty=difficulty,
                    prerequisites=concept.get("prerequisites", []),
                    estimated_duration_minutes=cal.get("estimated_duration_minutes", 10),
                    keywords=concept.get("keywords", []),
                    source_ids=concept.get("source_ids", []),
                ))

            return calibrated_concepts

        except Exception as e:
            print(f"[DIFFICULTY_CALIBRATOR] Batch calibration error: {e}", flush=True)
            # Return concepts with default difficulty on error
            return [
                CalibratedConcept(
                    concept_id=c.get("id", f"concept_{i}"),
                    name=c.get("name", ""),
                    description=c.get("description", ""),
                    difficulty=DifficultyVector(),
                    prerequisites=c.get("prerequisites", []),
                    keywords=c.get("keywords", []),
                    source_ids=c.get("source_ids", []),
                )
                for i, c in enumerate(concepts)
            ]

    def calibrate_from_knowledge_graph(
        self,
        knowledge_graph: Any,  # KnowledgeGraph from knowledge_graph.py
    ) -> List[Dict[str, Any]]:
        """
        Extract concepts from existing knowledge graph for calibration.

        This bridges the existing KnowledgeGraphBuilder with the difficulty calibrator.
        """
        concepts = []
        for concept in knowledge_graph.concepts:
            concepts.append({
                "id": concept.concept_id,
                "name": concept.name,
                "description": concept.definitions[0].definition_text if concept.definitions else "",
                "keywords": concept.aliases,
                "prerequisites": concept.prerequisites,
                "source_ids": [d.source_id for d in concept.definitions],
            })
        return concepts

    def create_progression(
        self,
        calibrated_concepts: List[CalibratedConcept],
        max_difficulty_jump: float = 0.15,
    ) -> DifficultyProgression:
        """Create a difficulty progression from calibrated concepts"""
        return DifficultyProgression(
            concepts=calibrated_concepts,
            max_difficulty_jump=max_difficulty_jump,
        )


# Singleton instance
_calibrator_instance: Optional[DifficultyCalibratorService] = None


def get_difficulty_calibrator() -> DifficultyCalibratorService:
    """Get the singleton difficulty calibrator instance"""
    global _calibrator_instance
    if _calibrator_instance is None:
        _calibrator_instance = DifficultyCalibratorService()
    return _calibrator_instance


async def calibrate_concepts(
    concepts: List[Dict[str, Any]],
    course_subject: str = "",
    target_level: str = "intermediate",
    language: str = "en",
) -> List[CalibratedConcept]:
    """
    Convenience function to calibrate concepts with difficulty vectors.

    Example:
        concepts = [
            {"id": "c1", "name": "Variables", "description": "..."},
            {"id": "c2", "name": "Functions", "description": "...", "prerequisites": ["c1"]},
        ]
        calibrated = await calibrate_concepts(concepts, course_subject="Python")

        for c in calibrated:
            print(f"{c.name}: {c.difficulty.composite_score:.2f} ({c.skill_level})")
    """
    calibrator = get_difficulty_calibrator()
    return await calibrator.calibrate_concepts_batch(
        concepts, course_subject, target_level, language
    )
