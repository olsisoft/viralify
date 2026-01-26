"""
Difficulty Calibrator Engine

Layer 3 of the MAESTRO pipeline.
Calibrates concept difficulty using 4D vectors and LLM analysis.
"""

import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from models.data_models import (
    Concept,
    DifficultyVector,
    SkillLevel,
    BloomLevel,
    SKILL_LEVEL_RANGES,
)


CALIBRATION_PROMPT = """Analyze the difficulty of these concepts for learning purposes.

CONCEPTS:
{concepts_json}

For each concept, provide a 4-dimensional difficulty analysis:
1. conceptual_complexity (0.0-1.0): How abstract and complex is the concept?
2. prerequisites_depth (0.0-1.0): How many prerequisites and how deep is the chain?
3. information_density (0.0-1.0): How much information must be processed?
4. cognitive_load (0.0-1.0): How much mental effort is required?

Also determine:
- bloom_level: The Bloom's taxonomy level (remember, understand, apply, analyze, evaluate, create)
- estimated_duration: Time in minutes to teach this concept (5-30)

Respond in JSON format:
{{
    "calibrations": [
        {{
            "concept_id": "id_here",
            "conceptual_complexity": 0.5,
            "prerequisites_depth": 0.3,
            "information_density": 0.4,
            "cognitive_load": 0.35,
            "bloom_level": "understand",
            "estimated_duration_minutes": 10,
            "reasoning": "Brief explanation of the calibration"
        }}
    ]
}}"""


class DifficultyCalibratorEngine:
    """
    Calibrates concept difficulty using 4D vectors.

    The 4D difficulty vector provides nuanced difficulty assessment:
    - conceptual_complexity: Abstraction level
    - prerequisites_depth: Dependency depth
    - information_density: Content volume
    - cognitive_load: Mental effort required
    """

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None, model: str = "gpt-4o-mini"):
        self.client = openai_client or AsyncOpenAI()
        self.model = model

    async def calibrate_concepts(
        self,
        concepts: List[Concept],
        batch_size: int = 10,
    ) -> List[Concept]:
        """
        Calibrate difficulty for a list of concepts.

        Args:
            concepts: Concepts to calibrate
            batch_size: Number of concepts per LLM call

        Returns:
            Concepts with calibrated difficulty vectors
        """
        print(f"[DIFFICULTY_CALIBRATOR] Calibrating {len(concepts)} concepts", flush=True)

        calibrated = []

        # Process in batches
        for i in range(0, len(concepts), batch_size):
            batch = concepts[i:i + batch_size]
            calibrated_batch = await self._calibrate_batch(batch)
            calibrated.extend(calibrated_batch)

        print(f"[DIFFICULTY_CALIBRATOR] Calibrated {len(calibrated)} concepts", flush=True)
        return calibrated

    async def _calibrate_batch(self, concepts: List[Concept]) -> List[Concept]:
        """Calibrate a batch of concepts"""
        # Prepare concepts for prompt
        concepts_data = [
            {
                "id": c.id,
                "name": c.name,
                "description": c.description,
                "keywords": c.keywords,
                "prerequisites": c.prerequisites,
            }
            for c in concepts
        ]

        prompt = CALIBRATION_PROMPT.format(
            concepts_json=json.dumps(concepts_data, indent=2)
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in educational assessment and difficulty calibration."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            data = json.loads(response.choices[0].message.content)
            calibrations = {c["concept_id"]: c for c in data.get("calibrations", [])}

            # Apply calibrations to concepts
            result = []
            for concept in concepts:
                cal = calibrations.get(concept.id, {})
                if cal:
                    concept.difficulty = DifficultyVector(
                        conceptual_complexity=cal.get("conceptual_complexity", 0.5),
                        prerequisites_depth=cal.get("prerequisites_depth", 0.5),
                        information_density=cal.get("information_density", 0.5),
                        cognitive_load=cal.get("cognitive_load", 0.5),
                    )
                    concept.estimated_duration_minutes = cal.get("estimated_duration_minutes", 10)
                result.append(concept)

            return result

        except Exception as e:
            print(f"[DIFFICULTY_CALIBRATOR] Error calibrating batch: {e}", flush=True)
            return concepts  # Return uncalibrated concepts on error

    def calculate_progression_score(self, concepts: List[Concept]) -> Dict[str, Any]:
        """
        Calculate progression metrics for a sequence of concepts.

        Returns metrics about difficulty progression smoothness.
        """
        if not concepts:
            return {"smooth": True, "issues": []}

        difficulties = [c.difficulty.composite_score for c in concepts]
        issues = []

        # Check for smooth progression
        max_jump = 0.15
        for i in range(1, len(difficulties)):
            jump = difficulties[i] - difficulties[i - 1]
            if jump > max_jump:
                issues.append({
                    "type": "excessive_jump",
                    "from_concept": concepts[i - 1].id,
                    "to_concept": concepts[i].id,
                    "jump": round(jump, 3),
                })

        # Calculate smoothness score
        jumps = [abs(difficulties[i] - difficulties[i - 1]) for i in range(1, len(difficulties))]
        avg_jump = sum(jumps) / len(jumps) if jumps else 0
        smoothness = 1.0 - min(1.0, avg_jump / max_jump)

        return {
            "smooth": len(issues) == 0,
            "smoothness_score": round(smoothness, 3),
            "average_jump": round(avg_jump, 3),
            "max_jump_observed": round(max(jumps) if jumps else 0, 3),
            "issues": issues,
            "difficulty_range": [
                round(min(difficulties), 3),
                round(max(difficulties), 3),
            ],
        }

    @staticmethod
    def get_skill_level(composite_score: float) -> SkillLevel:
        """Convert composite difficulty score to skill level"""
        for level, (min_score, max_score) in SKILL_LEVEL_RANGES.items():
            if min_score <= composite_score < max_score:
                return level
        return SkillLevel.EXPERT

    @staticmethod
    def get_bloom_level(cognitive_load: float) -> BloomLevel:
        """Derive Bloom's level from cognitive load"""
        if cognitive_load < 0.15:
            return BloomLevel.REMEMBER
        elif cognitive_load < 0.35:
            return BloomLevel.UNDERSTAND
        elif cognitive_load < 0.50:
            return BloomLevel.APPLY
        elif cognitive_load < 0.70:
            return BloomLevel.ANALYZE
        elif cognitive_load < 0.85:
            return BloomLevel.EVALUATE
        return BloomLevel.CREATE


async def calibrate_concepts(concepts: List[Concept]) -> List[Concept]:
    """
    Convenience function to calibrate concept difficulty.

    Example:
        calibrated = await calibrate_concepts(concepts)
        for c in calibrated:
            print(f"{c.name}: {c.difficulty.composite_score:.2f}")
    """
    engine = DifficultyCalibratorEngine()
    return await engine.calibrate_concepts(concepts)
