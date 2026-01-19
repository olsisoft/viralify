"""
Quiz Generator Service

Generates Udemy-style quizzes based on lecture content using GPT.
Supports multiple question types: MCQ, True/False, Multi-select, Fill-in-blank, Matching.
"""
import json
import os
import uuid
from typing import List, Optional

from openai import AsyncOpenAI

from models.course_models import Lecture, Section, CourseOutline, ProfileCategory
from models.lesson_elements import (
    Quiz,
    QuizQuestion,
    QuizQuestionType,
    QuizConfig,
    QuizFrequency,
)


class QuizGenerator:
    """Generates quizzes based on lecture/section content"""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.client = AsyncOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            timeout=120.0,
            max_retries=2
        )

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
    ) -> Quiz:
        """Generate a quiz using GPT"""
        question_types_str = ", ".join([qt.value for qt in config.question_types])

        prompt = f"""Tu es un expert en création de quiz éducatifs style Udemy.

Génère un quiz basé sur ce contenu:

TITRE: {title}
CONTENU: {content_summary}

OBJECTIFS D'APPRENTISSAGE:
{chr(10).join(f"- {obj}" for obj in objectives[:10])}

DIFFICULTÉ: {difficulty}
{f"CATÉGORIE: {category.value}" if category else ""}

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
1. Questions claires et précises
2. Options plausibles (pas de réponses évidentes)
3. Explications pédagogiques
4. Variété dans les types de questions
5. Difficulté progressive
6. Basé UNIQUEMENT sur le contenu fourni

Réponds UNIQUEMENT avec le JSON, sans markdown."""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
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
