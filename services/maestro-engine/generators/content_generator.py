"""
Content Generator

Layer 5 of the MAESTRO pipeline.
Generates complete lesson content with scripts, slides, quizzes, and exercises.
"""

import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from models.data_models import (
    Concept,
    Lesson,
    ScriptSegment,
    ScriptSegmentType,
    SlideContent,
    QuizQuestion,
    PracticalExercise,
    BloomLevel,
    SkillLevel,
    BLOOM_TO_COGNITIVE_LOAD,
)


LESSON_GENERATION_PROMPT = """Generate a complete lesson for this concept.

CONCEPT:
Name: {concept_name}
Description: {concept_description}
Keywords: {keywords}
Skill Level: {skill_level}
Bloom Level: {bloom_level}
Duration Target: {duration_minutes} minutes

LANGUAGE: {language}

Generate a lesson with:
1. Script segments (intro, explanation, example, summary)
2. Key takeaways (3-5 points)
3. Slides with content
4. Quiz questions aligned to Bloom's level
5. Practical exercise

Respond in JSON:
{{
    "title": "Lesson title",
    "description": "Brief description",
    "script_segments": [
        {{
            "type": "intro",
            "content": "Hook and context text",
            "duration_seconds": 30,
            "key_points": ["Point 1"]
        }},
        {{
            "type": "explanation",
            "content": "Main explanation text",
            "duration_seconds": 120,
            "key_points": ["Point 1", "Point 2"]
        }},
        {{
            "type": "example",
            "content": "Concrete example text",
            "duration_seconds": 90,
            "key_points": ["Example point"]
        }},
        {{
            "type": "summary",
            "content": "Recap text",
            "duration_seconds": 30,
            "key_points": ["Takeaway"]
        }}
    ],
    "key_takeaways": [
        "Key point 1",
        "Key point 2",
        "Key point 3"
    ],
    "slides": [
        {{
            "title": "Slide title",
            "bullet_points": ["Point 1", "Point 2"],
            "visual_suggestion": "Diagram or illustration suggestion",
            "speaker_notes": "Notes for presenter"
        }}
    ],
    "quiz_questions": [
        {{
            "type": "multiple_choice",
            "question": "Question text?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answers": [0],
            "explanation": "Why this is correct",
            "bloom_level": "understand",
            "points": 1
        }}
    ],
    "exercise": {{
        "title": "Exercise title",
        "description": "What the learner will do",
        "instructions": ["Step 1", "Step 2", "Step 3"],
        "starter_code": "// Optional starter code",
        "solution": "// Solution code",
        "hints": ["Hint 1", "Hint 2"],
        "estimated_time_minutes": 15
    }}
}}"""


class ContentGenerator:
    """
    Generates complete lesson content.

    Features:
    - Segmented scripts (intro, explanation, example, summary)
    - Bloom-aligned quiz questions
    - Practical exercises with solutions
    - Slide content with visual suggestions
    """

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None, model: str = "gpt-4o-mini"):
        self.client = openai_client or AsyncOpenAI()
        self.model = model

    async def generate_lesson(
        self,
        concept: Concept,
        language: str = "en",
    ) -> Lesson:
        """
        Generate a complete lesson for a concept.

        Args:
            concept: The concept to create a lesson for
            language: Output language

        Returns:
            Complete Lesson object
        """
        print(f"[CONTENT_GENERATOR] Generating lesson for '{concept.name}'", flush=True)

        prompt = LESSON_GENERATION_PROMPT.format(
            concept_name=concept.name,
            concept_description=concept.description,
            keywords=", ".join(concept.keywords),
            skill_level=concept.skill_level.value,
            bloom_level=concept.bloom_level.value,
            duration_minutes=concept.estimated_duration_minutes,
            language=language,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert instructional designer creating engaging educational content."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )

            data = json.loads(response.choices[0].message.content)
            lesson = self._parse_lesson(data, concept)

            print(f"[CONTENT_GENERATOR] Generated lesson '{lesson.title}' with {len(lesson.script_segments)} segments", flush=True)
            return lesson

        except Exception as e:
            print(f"[CONTENT_GENERATOR] Error generating lesson: {e}", flush=True)
            return self._create_fallback_lesson(concept)

    def _parse_lesson(self, data: Dict[str, Any], concept: Concept) -> Lesson:
        """Parse lesson data from LLM response"""
        # Parse script segments
        script_segments = []
        for seg_data in data.get("script_segments", []):
            segment = ScriptSegment(
                type=ScriptSegmentType(seg_data.get("type", "explanation")),
                content=seg_data.get("content", ""),
                duration_seconds=seg_data.get("duration_seconds", 60),
                key_points=seg_data.get("key_points", []),
            )
            script_segments.append(segment)

        # Combine segments into full script
        full_script = " ".join(seg.content for seg in script_segments)

        # Parse slides
        slides = []
        for slide_data in data.get("slides", []):
            slide = SlideContent(
                title=slide_data.get("title", ""),
                bullet_points=slide_data.get("bullet_points", []),
                visual_suggestion=slide_data.get("visual_suggestion", ""),
                speaker_notes=slide_data.get("speaker_notes", ""),
            )
            slides.append(slide)

        # Parse quiz questions
        quiz_questions = []
        for q_data in data.get("quiz_questions", []):
            question = QuizQuestion(
                type=q_data.get("type", "multiple_choice"),
                question=q_data.get("question", ""),
                options=q_data.get("options", []),
                correct_answers=q_data.get("correct_answers", [0]),
                explanation=q_data.get("explanation", ""),
                bloom_level=BloomLevel(q_data.get("bloom_level", "understand")),
                points=q_data.get("points", 1),
            )
            quiz_questions.append(question)

        # Parse exercise
        ex_data = data.get("exercise", {})
        exercises = []
        if ex_data:
            exercise = PracticalExercise(
                title=ex_data.get("title", f"Exercise: {concept.name}"),
                description=ex_data.get("description", ""),
                instructions=ex_data.get("instructions", []),
                starter_code=ex_data.get("starter_code"),
                solution=ex_data.get("solution", ""),
                hints=ex_data.get("hints", []),
                estimated_time_minutes=ex_data.get("estimated_time_minutes", 15),
            )
            exercises.append(exercise)

        # Calculate total duration from segments
        total_duration = sum(seg.duration_seconds for seg in script_segments) // 60
        if total_duration == 0:
            total_duration = concept.estimated_duration_minutes

        lesson = Lesson(
            concept_id=concept.id,
            title=data.get("title", f"Lesson: {concept.name}"),
            description=data.get("description", concept.description),
            script=full_script,
            script_segments=script_segments,
            key_takeaways=data.get("key_takeaways", []),
            slides=slides,
            quiz_questions=quiz_questions,
            exercises=exercises,
            skill_level=concept.skill_level,
            bloom_level=concept.bloom_level,
            estimated_duration_minutes=total_duration,
        )

        return lesson

    def _create_fallback_lesson(self, concept: Concept) -> Lesson:
        """Create a minimal lesson on error"""
        return Lesson(
            concept_id=concept.id,
            title=f"Lesson: {concept.name}",
            description=concept.description,
            script=f"In this lesson, we will learn about {concept.name}. {concept.description}",
            script_segments=[
                ScriptSegment(
                    type=ScriptSegmentType.EXPLANATION,
                    content=concept.description,
                    duration_seconds=concept.estimated_duration_minutes * 60,
                    key_points=[],
                )
            ],
            key_takeaways=[f"Understand {concept.name}"],
            skill_level=concept.skill_level,
            bloom_level=concept.bloom_level,
            estimated_duration_minutes=concept.estimated_duration_minutes,
        )

    async def generate_lessons_for_concepts(
        self,
        concepts: List[Concept],
        language: str = "en",
    ) -> List[Lesson]:
        """
        Generate lessons for multiple concepts.

        Args:
            concepts: List of concepts
            language: Output language

        Returns:
            List of Lesson objects
        """
        lessons = []
        for i, concept in enumerate(concepts):
            lesson = await self.generate_lesson(concept, language)
            lesson.sequence_order = i
            lessons.append(lesson)

        return lessons


async def generate_lesson(
    concept: Concept,
    language: str = "en",
) -> Lesson:
    """
    Convenience function to generate a lesson.

    Example:
        lesson = await generate_lesson(concept, language="en")
        print(f"Lesson: {lesson.title}")
        for seg in lesson.script_segments:
            print(f"  [{seg.type.value}] {seg.duration_seconds}s")
    """
    generator = ContentGenerator()
    return await generator.generate_lesson(concept, language)
