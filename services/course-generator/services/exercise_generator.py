"""
Exercise Generator Service

Generates practical exercises with solutions based on concept content.
Aligned with Bloom's Taxonomy for appropriate cognitive challenge.

Exercise types:
- Code exercises (for tech courses)
- Case studies (for business courses)
- Problem solving (for analytical courses)
- Creative exercises (for design courses)
- Reflection exercises (for soft skills)
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    _USE_SHARED_LLM = False

from models.difficulty_models import BloomLevel, SkillLevel


class ExerciseType(str, Enum):
    """Types of practical exercises"""
    CODE_IMPLEMENTATION = "code_implementation"
    CODE_DEBUG = "code_debug"
    CODE_REVIEW = "code_review"
    CASE_STUDY = "case_study"
    PROBLEM_SOLVING = "problem_solving"
    DESIGN_CHALLENGE = "design_challenge"
    ANALYSIS = "analysis"
    REFLECTION = "reflection"
    HANDS_ON_LAB = "hands_on_lab"
    PROJECT_MINI = "project_mini"


# Bloom level to exercise type mapping
BLOOM_EXERCISE_MAPPING: Dict[BloomLevel, List[ExerciseType]] = {
    BloomLevel.REMEMBER: [ExerciseType.REFLECTION],
    BloomLevel.UNDERSTAND: [ExerciseType.REFLECTION, ExerciseType.ANALYSIS],
    BloomLevel.APPLY: [ExerciseType.CODE_IMPLEMENTATION, ExerciseType.HANDS_ON_LAB, ExerciseType.PROBLEM_SOLVING],
    BloomLevel.ANALYZE: [ExerciseType.CODE_REVIEW, ExerciseType.CASE_STUDY, ExerciseType.ANALYSIS],
    BloomLevel.EVALUATE: [ExerciseType.CODE_REVIEW, ExerciseType.CASE_STUDY, ExerciseType.DESIGN_CHALLENGE],
    BloomLevel.CREATE: [ExerciseType.PROJECT_MINI, ExerciseType.DESIGN_CHALLENGE, ExerciseType.CODE_IMPLEMENTATION],
}


@dataclass
class ExerciseSolution:
    """Solution for a practical exercise"""
    solution_text: str
    code: Optional[str] = None
    explanation: str = ""
    hints: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)


@dataclass
class PracticalExercise:
    """A practical exercise with instructions and solution"""
    id: str
    title: str
    description: str
    exercise_type: ExerciseType
    bloom_level: BloomLevel
    skill_level: SkillLevel

    # Instructions
    instructions: List[str]
    requirements: List[str]
    starter_code: Optional[str] = None
    resources: List[str] = field(default_factory=list)

    # Solution
    solution: Optional[ExerciseSolution] = None

    # Metadata
    estimated_time_minutes: int = 15
    points: int = 10
    difficulty_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "exercise_type": self.exercise_type.value,
            "bloom_level": self.bloom_level.value,
            "skill_level": self.skill_level.value,
            "instructions": self.instructions,
            "requirements": self.requirements,
            "starter_code": self.starter_code,
            "resources": self.resources,
            "solution": {
                "solution_text": self.solution.solution_text if self.solution else "",
                "code": self.solution.code if self.solution else None,
                "explanation": self.solution.explanation if self.solution else "",
                "hints": self.solution.hints if self.solution else [],
                "common_mistakes": self.solution.common_mistakes if self.solution else [],
            } if self.solution else None,
            "estimated_time_minutes": self.estimated_time_minutes,
            "points": self.points,
            "difficulty_score": self.difficulty_score,
        }


class ExerciseGenerator:
    """
    Generates practical exercises aligned with Bloom's Taxonomy.

    Supports multiple exercise types adapted to:
    - Concept difficulty level
    - Bloom's cognitive level
    - Content domain (tech, business, etc.)
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        if _USE_SHARED_LLM:
            self.client = get_llm_client()
        else:
            self.client = AsyncOpenAI(
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                timeout=120.0,
                max_retries=2,
            )

    async def generate_exercise(
        self,
        concept_name: str,
        concept_description: str,
        bloom_level: BloomLevel,
        skill_level: SkillLevel,
        exercise_type: Optional[ExerciseType] = None,
        language: str = "fr",
        domain: str = "tech",
    ) -> PracticalExercise:
        """
        Generate a practical exercise for a concept.

        Args:
            concept_name: Name of the concept
            concept_description: Description of the concept
            bloom_level: Target Bloom's taxonomy level
            skill_level: Target skill level
            exercise_type: Optional specific exercise type
            language: Output language
            domain: Content domain (tech, business, etc.)

        Returns:
            PracticalExercise with instructions and solution
        """
        print(f"[EXERCISE] Generating exercise for '{concept_name}' at {bloom_level.value} level", flush=True)

        # Determine exercise type if not specified
        if exercise_type is None:
            possible_types = BLOOM_EXERCISE_MAPPING.get(bloom_level, [ExerciseType.PROBLEM_SOLVING])
            exercise_type = possible_types[0]  # Take first recommended type

        prompt = self._build_prompt(
            concept_name=concept_name,
            concept_description=concept_description,
            bloom_level=bloom_level,
            skill_level=skill_level,
            exercise_type=exercise_type,
            language=language,
            domain=domain,
        )

        try:
            response = await self.client.chat.completions.create(
                model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert en création d'exercices pédagogiques pratiques alignés sur la taxonomie de Bloom."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=2500,
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            # Build solution
            solution = ExerciseSolution(
                solution_text=data.get("solution", {}).get("solution_text", ""),
                code=data.get("solution", {}).get("code"),
                explanation=data.get("solution", {}).get("explanation", ""),
                hints=data.get("solution", {}).get("hints", []),
                common_mistakes=data.get("solution", {}).get("common_mistakes", []),
            )

            # Calculate difficulty score
            skill_scores = {
                SkillLevel.BEGINNER: 0.2,
                SkillLevel.INTERMEDIATE: 0.4,
                SkillLevel.ADVANCED: 0.6,
                SkillLevel.VERY_ADVANCED: 0.8,
                SkillLevel.EXPERT: 0.95,
            }
            difficulty_score = skill_scores.get(skill_level, 0.5)

            exercise = PracticalExercise(
                id=str(uuid.uuid4())[:8],
                title=data.get("title", f"Exercice: {concept_name}"),
                description=data.get("description", ""),
                exercise_type=exercise_type,
                bloom_level=bloom_level,
                skill_level=skill_level,
                instructions=data.get("instructions", []),
                requirements=data.get("requirements", []),
                starter_code=data.get("starter_code"),
                resources=data.get("resources", []),
                solution=solution,
                estimated_time_minutes=data.get("estimated_time_minutes", 15),
                points=data.get("points", 10),
                difficulty_score=difficulty_score,
            )

            print(f"[EXERCISE] Generated '{exercise.title}' ({exercise_type.value})", flush=True)
            return exercise

        except Exception as e:
            print(f"[EXERCISE] Error generating exercise: {e}", flush=True)
            # Return minimal exercise on error
            return PracticalExercise(
                id=str(uuid.uuid4())[:8],
                title=f"Exercice: {concept_name}",
                description=f"Exercice pratique sur {concept_name}",
                exercise_type=exercise_type,
                bloom_level=bloom_level,
                skill_level=skill_level,
                instructions=["Exercice non généré - erreur"],
                requirements=[],
            )

    def _build_prompt(
        self,
        concept_name: str,
        concept_description: str,
        bloom_level: BloomLevel,
        skill_level: SkillLevel,
        exercise_type: ExerciseType,
        language: str,
        domain: str,
    ) -> str:
        """Build the prompt for exercise generation"""

        # Exercise type specific instructions
        type_instructions = {
            ExerciseType.CODE_IMPLEMENTATION: """
L'exercice doit demander d'implémenter du code:
- Fournir un starter_code avec des TODO
- La solution doit inclure le code complet
- Inclure des tests ou critères de validation""",
            ExerciseType.CODE_DEBUG: """
L'exercice doit présenter du code avec des bugs:
- Fournir du code bugué dans starter_code
- Les instructions expliquent les symptômes
- La solution identifie et corrige les bugs""",
            ExerciseType.CODE_REVIEW: """
L'exercice doit évaluer la qualité du code:
- Fournir du code à reviewer
- Demander d'identifier les améliorations
- La solution liste les points à améliorer""",
            ExerciseType.CASE_STUDY: """
L'exercice doit analyser un cas réel:
- Présenter une situation concrète
- Poser des questions d'analyse
- La solution fournit l'analyse détaillée""",
            ExerciseType.PROBLEM_SOLVING: """
L'exercice doit résoudre un problème:
- Présenter un problème clair
- Fournir les données nécessaires
- La solution détaille le raisonnement""",
            ExerciseType.DESIGN_CHALLENGE: """
L'exercice doit concevoir une solution:
- Définir les contraintes et objectifs
- Laisser de la créativité
- La solution présente une approche type""",
            ExerciseType.ANALYSIS: """
L'exercice doit analyser des données/situations:
- Fournir les éléments à analyser
- Guider l'analyse avec des questions
- La solution montre l'analyse complète""",
            ExerciseType.REFLECTION: """
L'exercice doit encourager la réflexion:
- Poser des questions ouvertes
- Relier à l'expérience personnelle
- La solution donne des pistes de réflexion""",
            ExerciseType.HANDS_ON_LAB: """
L'exercice doit être pratique et guidé:
- Étapes claires et progressives
- Environnement ou outils spécifiés
- La solution montre les résultats attendus""",
            ExerciseType.PROJECT_MINI: """
L'exercice doit être un mini-projet:
- Objectif clair et réalisable
- Plusieurs composants à créer
- La solution est un exemple complet""",
        }

        return f"""Génère un exercice pratique pour ce concept:

CONCEPT:
Nom: {concept_name}
Description: {concept_description}

NIVEAU:
- Bloom: {bloom_level.value} (niveau cognitif)
- Compétence: {skill_level.value}
- Domaine: {domain}

TYPE D'EXERCICE: {exercise_type.value}
{type_instructions.get(exercise_type, "")}

LANGUE: {language}

Génère l'exercice en JSON:
{{
    "title": "Titre de l'exercice",
    "description": "Description courte de l'exercice",
    "instructions": [
        "Étape 1: ...",
        "Étape 2: ...",
        "Étape 3: ..."
    ],
    "requirements": [
        "Prérequis ou ressources nécessaires"
    ],
    "starter_code": "// Code de départ si applicable\\n...",
    "resources": ["Lien ou ressource utile"],
    "solution": {{
        "solution_text": "Explication de la solution",
        "code": "// Code solution si applicable\\n...",
        "explanation": "Explication détaillée du pourquoi",
        "hints": ["Indice 1", "Indice 2"],
        "common_mistakes": ["Erreur courante 1", "Erreur courante 2"]
    }},
    "estimated_time_minutes": 15,
    "points": 10
}}

RÈGLES:
1. L'exercice doit être adapté au niveau {skill_level.value}
2. Les instructions doivent être claires et progressives
3. La solution doit être complète et pédagogique
4. Les indices aident sans donner la réponse
5. Les erreurs courantes préviennent les pièges

Réponds UNIQUEMENT avec le JSON."""

    async def generate_exercises_for_concept(
        self,
        concept_name: str,
        concept_description: str,
        bloom_level: BloomLevel,
        skill_level: SkillLevel,
        num_exercises: int = 2,
        language: str = "fr",
        domain: str = "tech",
    ) -> List[PracticalExercise]:
        """
        Generate multiple exercises for a concept with variety.

        Args:
            concept_name: Name of the concept
            concept_description: Description
            bloom_level: Target Bloom level
            skill_level: Target skill level
            num_exercises: Number of exercises to generate
            language: Output language
            domain: Content domain

        Returns:
            List of varied PracticalExercise objects
        """
        # Get recommended exercise types for this Bloom level
        recommended_types = BLOOM_EXERCISE_MAPPING.get(
            bloom_level,
            [ExerciseType.PROBLEM_SOLVING]
        )

        exercises = []
        for i in range(min(num_exercises, len(recommended_types))):
            exercise_type = recommended_types[i % len(recommended_types)]

            exercise = await self.generate_exercise(
                concept_name=concept_name,
                concept_description=concept_description,
                bloom_level=bloom_level,
                skill_level=skill_level,
                exercise_type=exercise_type,
                language=language,
                domain=domain,
            )
            exercises.append(exercise)

        return exercises


# Singleton instance
_exercise_generator: Optional[ExerciseGenerator] = None


def get_exercise_generator() -> ExerciseGenerator:
    """Get singleton exercise generator instance"""
    global _exercise_generator
    if _exercise_generator is None:
        _exercise_generator = ExerciseGenerator()
    return _exercise_generator


async def generate_exercise(
    concept_name: str,
    concept_description: str,
    bloom_level: str = "apply",
    skill_level: str = "intermediate",
    exercise_type: Optional[str] = None,
    language: str = "fr",
) -> PracticalExercise:
    """
    Convenience function to generate a practical exercise.

    Example:
        exercise = await generate_exercise(
            concept_name="Python Functions",
            concept_description="Functions allow code reuse...",
            bloom_level="apply",
            skill_level="intermediate",
        )
        print(f"Exercise: {exercise.title}")
        print(f"Instructions: {exercise.instructions}")
        print(f"Solution: {exercise.solution.solution_text}")
    """
    generator = get_exercise_generator()

    bloom = BloomLevel(bloom_level)
    skill = SkillLevel(skill_level)
    ex_type = ExerciseType(exercise_type) if exercise_type else None

    return await generator.generate_exercise(
        concept_name=concept_name,
        concept_description=concept_description,
        bloom_level=bloom,
        skill_level=skill,
        exercise_type=ex_type,
        language=language,
    )
