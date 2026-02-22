"""
Quiz Generator Service

Generates Udemy-style quizzes based on lecture content using GPT.
Supports multiple question types: MCQ, True/False, Multi-select, Fill-in-blank, Matching.

Enhanced with Bloom's Taxonomy alignment (MAESTRO integration):
- Questions are calibrated to cognitive levels
- Difficulty progression within quizzes
- Question types mapped to Bloom levels
"""
import json
import os
import uuid
from typing import List, Optional, Dict, Any

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    _USE_SHARED_LLM = False

from models.course_models import Lecture, Section, CourseOutline, ProfileCategory
from models.lesson_elements import (
    Quiz,
    QuizQuestion,
    QuizQuestionType,
    QuizConfig,
    QuizFrequency,
)
from models.difficulty_models import BloomLevel, SkillLevel, DifficultyVector


# Bloom's Taxonomy mapping to question types and verbs
BLOOM_QUESTION_MAPPING: Dict[BloomLevel, Dict[str, Any]] = {
    BloomLevel.REMEMBER: {
        "preferred_types": [QuizQuestionType.MULTIPLE_CHOICE, QuizQuestionType.TRUE_FALSE],
        "verbs": ["définir", "lister", "identifier", "nommer", "rappeler", "reconnaître"],
        "description": "Recall facts and basic concepts",
        "cognitive_demand": "low",
    },
    BloomLevel.UNDERSTAND: {
        "preferred_types": [QuizQuestionType.MULTIPLE_CHOICE, QuizQuestionType.TRUE_FALSE, QuizQuestionType.FILL_BLANK],
        "verbs": ["expliquer", "décrire", "résumer", "interpréter", "paraphraser", "classifier"],
        "description": "Explain ideas or concepts",
        "cognitive_demand": "low-medium",
    },
    BloomLevel.APPLY: {
        "preferred_types": [QuizQuestionType.MULTIPLE_CHOICE, QuizQuestionType.FILL_BLANK, QuizQuestionType.MULTI_SELECT],
        "verbs": ["utiliser", "implémenter", "exécuter", "résoudre", "démontrer", "appliquer"],
        "description": "Use information in new situations",
        "cognitive_demand": "medium",
    },
    BloomLevel.ANALYZE: {
        "preferred_types": [QuizQuestionType.MULTI_SELECT, QuizQuestionType.MATCHING, QuizQuestionType.MULTIPLE_CHOICE],
        "verbs": ["comparer", "contraster", "différencier", "organiser", "déconstruire", "attribuer"],
        "description": "Draw connections among ideas",
        "cognitive_demand": "medium-high",
    },
    BloomLevel.EVALUATE: {
        "preferred_types": [QuizQuestionType.MULTI_SELECT, QuizQuestionType.MULTIPLE_CHOICE],
        "verbs": ["juger", "critiquer", "justifier", "argumenter", "évaluer", "défendre"],
        "description": "Justify a decision or course of action",
        "cognitive_demand": "high",
    },
    BloomLevel.CREATE: {
        "preferred_types": [QuizQuestionType.FILL_BLANK, QuizQuestionType.MATCHING],
        "verbs": ["concevoir", "construire", "développer", "formuler", "créer", "produire"],
        "description": "Produce new or original work",
        "cognitive_demand": "high",
    },
}

# Skill level to primary Bloom levels mapping
SKILL_TO_BLOOM_LEVELS: Dict[SkillLevel, List[BloomLevel]] = {
    SkillLevel.BEGINNER: [BloomLevel.REMEMBER, BloomLevel.UNDERSTAND],
    SkillLevel.INTERMEDIATE: [BloomLevel.UNDERSTAND, BloomLevel.APPLY],
    SkillLevel.ADVANCED: [BloomLevel.APPLY, BloomLevel.ANALYZE],
    SkillLevel.VERY_ADVANCED: [BloomLevel.ANALYZE, BloomLevel.EVALUATE],
    SkillLevel.EXPERT: [BloomLevel.EVALUATE, BloomLevel.CREATE],
}


class QuizGenerator:
    """
    Generates quizzes based on lecture/section content.

    Enhanced with Bloom's Taxonomy alignment:
    - Questions calibrated to cognitive levels
    - Difficulty matches content skill level
    - Question types optimized per Bloom level
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        if _USE_SHARED_LLM:
            self.client = get_llm_client()
        else:
            self.client = AsyncOpenAI(
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                timeout=120.0,
                max_retries=2
            )

    def _get_bloom_context(self, skill_level: SkillLevel) -> Dict[str, Any]:
        """Get Bloom's taxonomy context for a skill level"""
        bloom_levels = SKILL_TO_BLOOM_LEVELS.get(skill_level, [BloomLevel.UNDERSTAND, BloomLevel.APPLY])

        all_verbs = []
        all_types = []
        descriptions = []

        for bloom in bloom_levels:
            mapping = BLOOM_QUESTION_MAPPING[bloom]
            all_verbs.extend(mapping["verbs"])
            all_types.extend(mapping["preferred_types"])
            descriptions.append(f"{bloom.value}: {mapping['description']}")

        # Remove duplicates while preserving order
        unique_types = list(dict.fromkeys(all_types))
        unique_verbs = list(dict.fromkeys(all_verbs))

        return {
            "bloom_levels": [b.value for b in bloom_levels],
            "verbs": unique_verbs[:10],  # Limit to 10 verbs
            "preferred_types": unique_types[:4],  # Limit to 4 types
            "descriptions": descriptions,
        }

    def _skill_level_from_string(self, difficulty: str) -> SkillLevel:
        """Convert difficulty string to SkillLevel enum"""
        mapping = {
            "beginner": SkillLevel.BEGINNER,
            "intermediate": SkillLevel.INTERMEDIATE,
            "advanced": SkillLevel.ADVANCED,
            "very_advanced": SkillLevel.VERY_ADVANCED,
            "expert": SkillLevel.EXPERT,
        }
        return mapping.get(difficulty.lower(), SkillLevel.INTERMEDIATE)

    async def generate_lecture_quiz(
        self,
        lecture: Lecture,
        section: Section,
        course_title: str,
        config: QuizConfig,
        category: Optional[ProfileCategory] = None,
    ) -> Quiz:
        """Generate a quiz for a single lecture"""
        print(f"[QUIZ] Generating quiz for lecture: {lecture.title}", flush=True)

        return await self._generate_quiz(
            title=f"Quiz: {lecture.title}",
            content_summary=self._build_lecture_summary(lecture),
            objectives=lecture.objectives,
            difficulty=lecture.difficulty.value,
            config=config,
            category=category,
        )

    async def generate_section_quiz(
        self,
        section: Section,
        course_title: str,
        config: QuizConfig,
        category: Optional[ProfileCategory] = None,
    ) -> Quiz:
        """Generate a quiz for an entire section"""
        print(f"[QUIZ] Generating quiz for section: {section.title}", flush=True)

        # Combine all lecture content
        all_objectives = []
        lectures_summary = []
        for lecture in section.lectures:
            all_objectives.extend(lecture.objectives)
            lectures_summary.append(f"- {lecture.title}: {lecture.description}")

        return await self._generate_quiz(
            title=f"Quiz de section: {section.title}",
            content_summary=f"Section: {section.title}\n{section.description}\n\nLectures:\n" + "\n".join(lectures_summary),
            objectives=all_objectives,
            difficulty="intermediate",  # Section quizzes are balanced
            config=config,
            category=category,
        )

    async def generate_course_final_quiz(
        self,
        outline: CourseOutline,
        config: QuizConfig,
    ) -> Quiz:
        """Generate a comprehensive final quiz for the entire course"""
        print(f"[QUIZ] Generating final quiz for course: {outline.title}", flush=True)

        # Build comprehensive summary
        all_objectives = []
        sections_summary = []
        for section in outline.sections:
            sections_summary.append(f"\n## {section.title}")
            for lecture in section.lectures:
                sections_summary.append(f"- {lecture.title}")
                all_objectives.extend(lecture.objectives)

        # Increase questions for final quiz
        final_config = QuizConfig(
            enabled=True,
            frequency=config.frequency,
            questions_per_quiz=min(config.questions_per_quiz * 2, 20),  # Double but max 20
            question_types=config.question_types,
            passing_score=config.passing_score,
            show_explanations=config.show_explanations,
        )

        return await self._generate_quiz(
            title=f"Examen final: {outline.title}",
            content_summary=f"Cours: {outline.title}\n{outline.description}\n\nContenu:\n" + "\n".join(sections_summary),
            objectives=all_objectives[:20],  # Limit objectives
            difficulty=outline.difficulty_end.value,
            config=final_config,
            category=outline.category,
        )

    async def _generate_quiz(
        self,
        title: str,
        content_summary: str,
        objectives: List[str],
        difficulty: str,
        config: QuizConfig,
        category: Optional[ProfileCategory] = None,
        bloom_level: Optional[BloomLevel] = None,
    ) -> Quiz:
        """Generate a quiz using GPT with Bloom's Taxonomy alignment"""
        question_types_str = ", ".join([qt.value for qt in config.question_types])

        # Get Bloom's taxonomy context
        skill_level = self._skill_level_from_string(difficulty)
        bloom_context = self._get_bloom_context(skill_level)

        bloom_guidance = f"""
TAXONOMIE DE BLOOM (IMPORTANT):
- Niveaux cognitifs cibles: {', '.join(bloom_context['bloom_levels'])}
- Verbes d'action à utiliser: {', '.join(bloom_context['verbs'][:6])}
- Types de questions recommandés: {', '.join([t.value for t in bloom_context['preferred_types']])}

Les questions doivent:
1. Utiliser les verbes d'action appropriés au niveau cognitif
2. Avoir une complexité croissante dans le quiz
3. Tester la compréhension réelle, pas la mémorisation simple
"""

        prompt = f"""Tu es un expert en création de quiz éducatifs style Udemy, spécialisé dans l'alignement avec la taxonomie de Bloom.

Génère un quiz basé sur ce contenu:

TITRE: {title}
CONTENU: {content_summary}

OBJECTIFS D'APPRENTISSAGE:
{chr(10).join(f"- {obj}" for obj in objectives[:10])}

DIFFICULTÉ: {difficulty}
{f"CATÉGORIE: {category.value}" if category else ""}
{bloom_guidance}

CONFIGURATION:
- Nombre de questions: {config.questions_per_quiz}
- Types de questions autorisés: {question_types_str}
- Score de passage: {config.passing_score}%

TYPES DE QUESTIONS:
- multiple_choice: QCM avec une seule bonne réponse (4 options)
- multi_select: QCM avec plusieurs bonnes réponses (4-5 options)
- true_false: Vrai ou Faux
- fill_blank: Compléter le texte (une phrase avec un blanc)
- matching: Associer des éléments (3-4 paires)

Génère le quiz en JSON:
{{
    "title": "{title}",
    "description": "Description du quiz",
    "questions": [
        {{
            "type": "multiple_choice",
            "question": "La question ?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answers": [0],
            "explanation": "Explication de la bonne réponse",
            "points": 1
        }},
        {{
            "type": "true_false",
            "question": "Affirmation vraie ou fausse",
            "options": ["Vrai", "Faux"],
            "correct_answers": [0],
            "explanation": "Explication",
            "points": 1
        }},
        {{
            "type": "multi_select",
            "question": "Sélectionnez toutes les bonnes réponses",
            "options": ["A", "B", "C", "D"],
            "correct_answers": [0, 2],
            "explanation": "Explication",
            "points": 2
        }},
        {{
            "type": "fill_blank",
            "question": "Le langage ___ est utilisé pour le web",
            "options": ["JavaScript", "Python", "Java"],
            "correct_answers": [0],
            "explanation": "JavaScript est le langage du web",
            "points": 1
        }},
        {{
            "type": "matching",
            "question": "Associez les termes à leur définition",
            "matching_pairs": {{"Terme1": "Définition1", "Terme2": "Définition2"}},
            "options": [],
            "correct_answers": [],
            "explanation": "Explication des associations",
            "points": 2
        }}
    ]
}}

RÈGLES:
1. Questions claires et précises utilisant les verbes de Bloom appropriés
2. Options plausibles (pas de réponses évidentes)
3. Explications pédagogiques mentionnant le niveau cognitif testé
4. Variété dans les types de questions selon les recommandations Bloom
5. Difficulté progressive du niveau cognitif le plus bas au plus haut
6. Basé UNIQUEMENT sur le contenu fourni
7. Chaque question doit indiquer son niveau Bloom dans le champ "bloom_level"

Réponds UNIQUEMENT avec le JSON, sans markdown."""

        try:
            response = await self.client.chat.completions.create(
                model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000,
            )

            content = response.choices[0].message.content.strip()

            # Clean markdown if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            # Parse questions
            questions = []
            total_points = 0

            for q_data in data.get("questions", []):
                try:
                    q_type = QuizQuestionType(q_data.get("type", "multiple_choice"))
                    points = q_data.get("points", 1)
                    total_points += points

                    question = QuizQuestion(
                        id=str(uuid.uuid4())[:8],
                        type=q_type,
                        question=q_data.get("question", ""),
                        options=q_data.get("options", []),
                        correct_answers=q_data.get("correct_answers", []),
                        explanation=q_data.get("explanation", ""),
                        points=points,
                        matching_pairs=q_data.get("matching_pairs"),
                    )
                    questions.append(question)
                except Exception as e:
                    print(f"[QUIZ] Error parsing question: {e}", flush=True)
                    continue

            quiz = Quiz(
                id=str(uuid.uuid4())[:8],
                title=data.get("title", title),
                description=data.get("description", ""),
                questions=questions,
                total_points=total_points,
                passing_score=config.passing_score,
            )

            print(f"[QUIZ] Generated {len(questions)} questions, {total_points} total points", flush=True)
            return quiz

        except Exception as e:
            print(f"[QUIZ] Error generating quiz: {e}", flush=True)
            # Return empty quiz on error
            return Quiz(
                id=str(uuid.uuid4())[:8],
                title=title,
                description="Quiz non généré - erreur",
                questions=[],
                total_points=0,
                passing_score=config.passing_score,
            )

    def _build_lecture_summary(self, lecture: Lecture) -> str:
        """Build a summary of lecture content for quiz generation"""
        summary_parts = [
            f"Titre: {lecture.title}",
            f"Description: {lecture.description}",
            f"Difficulté: {lecture.difficulty.value}",
        ]

        if lecture.objectives:
            summary_parts.append("Objectifs:")
            for obj in lecture.objectives:
                summary_parts.append(f"  - {obj}")

        return "\n".join(summary_parts)

    def should_generate_quiz(
        self,
        config: QuizConfig,
        lecture_index: int,
        section_index: int,
        total_lectures_in_section: int,
        is_last_section: bool,
    ) -> tuple[bool, str]:
        """
        Determine if a quiz should be generated at this point.

        Returns:
            (should_generate, quiz_type) where quiz_type is 'lecture', 'section', or 'final'
        """
        if not config.enabled:
            return False, ""

        if config.frequency == QuizFrequency.PER_LECTURE:
            return True, "lecture"

        elif config.frequency == QuizFrequency.PER_SECTION:
            # Generate quiz at end of section
            is_last_lecture = lecture_index == total_lectures_in_section - 1
            if is_last_lecture:
                return True, "section"
            return False, ""

        elif config.frequency == QuizFrequency.END_OF_COURSE:
            # Only generate final quiz
            is_last_lecture = lecture_index == total_lectures_in_section - 1
            if is_last_lecture and is_last_section:
                return True, "final"
            return False, ""

        elif config.frequency == QuizFrequency.CUSTOM:
            # Every N lectures
            n = config.custom_frequency or 3
            # Calculate absolute lecture number
            # This is simplified - in practice we'd track total lecture count
            if (lecture_index + 1) % n == 0:
                return True, "lecture"
            return False, ""

        return False, ""

    async def generate_bloom_aligned_quiz(
        self,
        concept_name: str,
        concept_description: str,
        bloom_level: BloomLevel,
        skill_level: SkillLevel,
        config: QuizConfig,
        language: str = "fr",
    ) -> Quiz:
        """
        Generate a quiz specifically aligned to a Bloom's taxonomy level.

        This method is designed for MAESTRO integration where each concept
        has a specific Bloom level and skill level.

        Args:
            concept_name: Name of the concept being tested
            concept_description: Description of the concept
            bloom_level: Target Bloom's taxonomy level
            skill_level: Target skill level
            config: Quiz configuration

        Returns:
            Quiz with questions aligned to the specified Bloom level
        """
        print(f"[QUIZ] Generating Bloom-aligned quiz for '{concept_name}' at {bloom_level.value} level", flush=True)

        bloom_mapping = BLOOM_QUESTION_MAPPING[bloom_level]
        verbs = bloom_mapping["verbs"]
        preferred_types = bloom_mapping["preferred_types"]
        cognitive_demand = bloom_mapping["cognitive_demand"]

        # Filter config question types to prefer Bloom-aligned types
        aligned_types = [qt for qt in config.question_types if qt in preferred_types]
        if not aligned_types:
            aligned_types = list(preferred_types[:2])  # Fallback to Bloom defaults

        question_types_str = ", ".join([qt.value for qt in aligned_types])

        prompt = f"""Tu es un expert en création de quiz éducatifs alignés sur la taxonomie de Bloom.

CONCEPT À ÉVALUER:
Nom: {concept_name}
Description: {concept_description}

NIVEAU BLOOM CIBLE: {bloom_level.value.upper()}
- Description: {bloom_mapping['description']}
- Demande cognitive: {cognitive_demand}
- Verbes à utiliser: {', '.join(verbs)}

NIVEAU DE COMPÉTENCE: {skill_level.value}

CONFIGURATION:
- Nombre de questions: {config.questions_per_quiz}
- Types de questions: {question_types_str}
- Langue: {language}

INSTRUCTIONS BLOOM:
Pour le niveau {bloom_level.value.upper()}, les questions doivent:
{"- Tester le rappel de faits et définitions" if bloom_level == BloomLevel.REMEMBER else ""}
{"- Vérifier la compréhension et l'explication des concepts" if bloom_level == BloomLevel.UNDERSTAND else ""}
{"- Évaluer la capacité à appliquer dans de nouvelles situations" if bloom_level == BloomLevel.APPLY else ""}
{"- Tester la capacité à analyser et comparer" if bloom_level == BloomLevel.ANALYZE else ""}
{"- Évaluer le jugement critique et l'argumentation" if bloom_level == BloomLevel.EVALUATE else ""}
{"- Tester la capacité à créer et innover" if bloom_level == BloomLevel.CREATE else ""}

Génère le quiz en JSON:
{{
    "title": "Quiz: {concept_name}",
    "description": "Évaluation niveau {bloom_level.value}",
    "bloom_level": "{bloom_level.value}",
    "questions": [
        {{
            "type": "<question_type>",
            "question": "<question utilisant un verbe Bloom approprié>",
            "options": ["..."],
            "correct_answers": [<index>],
            "explanation": "<explication pédagogique>",
            "points": <1-3>,
            "bloom_level": "{bloom_level.value}"
        }}
    ]
}}

Réponds UNIQUEMENT avec le JSON, sans markdown."""

        try:
            response = await self.client.chat.completions.create(
                model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2500,
            )

            content = response.choices[0].message.content.strip()

            # Clean markdown if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            # Parse questions
            questions = []
            total_points = 0

            for q_data in data.get("questions", []):
                try:
                    q_type = QuizQuestionType(q_data.get("type", "multiple_choice"))
                    points = q_data.get("points", 1)
                    total_points += points

                    question = QuizQuestion(
                        id=str(uuid.uuid4())[:8],
                        type=q_type,
                        question=q_data.get("question", ""),
                        options=q_data.get("options", []),
                        correct_answers=q_data.get("correct_answers", []),
                        explanation=q_data.get("explanation", ""),
                        points=points,
                        matching_pairs=q_data.get("matching_pairs"),
                    )
                    questions.append(question)
                except Exception as e:
                    print(f"[QUIZ] Error parsing Bloom question: {e}", flush=True)
                    continue

            quiz = Quiz(
                id=str(uuid.uuid4())[:8],
                title=data.get("title", f"Quiz: {concept_name}"),
                description=data.get("description", f"Bloom level: {bloom_level.value}"),
                questions=questions,
                total_points=total_points,
                passing_score=config.passing_score,
            )

            print(f"[QUIZ] Generated {len(questions)} Bloom-aligned questions at {bloom_level.value} level", flush=True)
            return quiz

        except Exception as e:
            print(f"[QUIZ] Error generating Bloom-aligned quiz: {e}", flush=True)
            return Quiz(
                id=str(uuid.uuid4())[:8],
                title=f"Quiz: {concept_name}",
                description="Quiz non généré - erreur",
                questions=[],
                total_points=0,
                passing_score=config.passing_score,
            )


async def generate_quizzes_for_course(
    outline: CourseOutline,
    config: QuizConfig,
) -> dict:
    """
    Generate all quizzes for a course based on configuration.

    Returns:
        {
            "lecture_quizzes": {lecture_id: Quiz},
            "section_quizzes": {section_id: Quiz},
            "final_quiz": Quiz or None
        }
    """
    generator = QuizGenerator()
    result = {
        "lecture_quizzes": {},
        "section_quizzes": {},
        "final_quiz": None,
    }

    total_sections = len(outline.sections)

    for section_idx, section in enumerate(outline.sections):
        is_last_section = section_idx == total_sections - 1
        total_lectures = len(section.lectures)

        for lecture_idx, lecture in enumerate(section.lectures):
            should_gen, quiz_type = generator.should_generate_quiz(
                config=config,
                lecture_index=lecture_idx,
                section_index=section_idx,
                total_lectures_in_section=total_lectures,
                is_last_section=is_last_section,
            )

            if should_gen:
                if quiz_type == "lecture":
                    quiz = await generator.generate_lecture_quiz(
                        lecture=lecture,
                        section=section,
                        course_title=outline.title,
                        config=config,
                        category=outline.category,
                    )
                    result["lecture_quizzes"][lecture.id] = quiz

                elif quiz_type == "section":
                    quiz = await generator.generate_section_quiz(
                        section=section,
                        course_title=outline.title,
                        config=config,
                        category=outline.category,
                    )
                    result["section_quizzes"][section.id] = quiz

                elif quiz_type == "final":
                    quiz = await generator.generate_course_final_quiz(
                        outline=outline,
                        config=config,
                    )
                    result["final_quiz"] = quiz

    return result
