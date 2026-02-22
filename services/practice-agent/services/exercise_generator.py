"""
Dynamic Exercise Generator Service

Generates practice exercises from course content using LLM and RAG context.
This replaces static exercise files with AI-generated, course-tailored exercises.
"""

import json
import os
import uuid
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from models.practice_models import (
    Exercise,
    ExerciseType,
    ExerciseCategory,
    DifficultyLevel,
    ValidationCheck,
    ExpectedOutput,
)

try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    _USE_SHARED_LLM = False


class ExerciseGenerationConfig(BaseModel):
    """Configuration for exercise generation"""
    exercises_per_lecture: int = Field(default=2, ge=1, le=5)
    exercises_per_section: int = Field(default=3, ge=1, le=10)
    include_coding: bool = Field(default=True)
    include_debugging: bool = Field(default=True)
    include_quiz: bool = Field(default=True)
    include_architecture: bool = Field(default=False)
    difficulty_progression: bool = Field(default=True)


class GeneratedExerciseSet(BaseModel):
    """A set of generated exercises for a course"""
    course_id: str
    course_title: str
    category: str
    exercises: List[Exercise]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    source_context: Optional[str] = None


# Map course categories to exercise categories
CATEGORY_MAPPING = {
    "tech": [ExerciseCategory.PYTHON, ExerciseCategory.DOCKER, ExerciseCategory.LINUX, ExerciseCategory.GIT],
    "business": [ExerciseCategory.DATABASES, ExerciseCategory.NETWORKING],
    "education": [ExerciseCategory.PYTHON, ExerciseCategory.DATABASES],
    "creative": [ExerciseCategory.PYTHON],
    "health": [ExerciseCategory.DATABASES],
    "lifestyle": [ExerciseCategory.PYTHON],
}

# Sandbox configurations by detected technology
SANDBOX_CONFIGS = {
    "python": {
        "sandbox_type": "docker",
        "sandbox_config": {
            "image": "python:3.11-slim",
            "command": "python",
            "memory_limit": "256m",
            "cpu_limit": 0.5,
        }
    },
    "docker": {
        "sandbox_type": "docker",
        "sandbox_config": {
            "image": "docker:dind",
            "privileged": True,
            "memory_limit": "512m",
        }
    },
    "kubernetes": {
        "sandbox_type": "kubernetes",
        "sandbox_config": {
            "namespace": "practice",
            "resource_quota": "small",
        }
    },
    "terraform": {
        "sandbox_type": "docker",
        "sandbox_config": {
            "image": "hashicorp/terraform:latest",
            "memory_limit": "256m",
        }
    },
    "linux": {
        "sandbox_type": "docker",
        "sandbox_config": {
            "image": "ubuntu:22.04",
            "shell": "/bin/bash",
            "memory_limit": "256m",
        }
    },
    "git": {
        "sandbox_type": "docker",
        "sandbox_config": {
            "image": "alpine/git",
            "memory_limit": "128m",
        }
    },
    "database": {
        "sandbox_type": "docker",
        "sandbox_config": {
            "image": "postgres:15-alpine",
            "memory_limit": "256m",
        }
    },
    "javascript": {
        "sandbox_type": "docker",
        "sandbox_config": {
            "image": "node:20-alpine",
            "command": "node",
            "memory_limit": "256m",
        }
    },
    "default": {
        "sandbox_type": "docker",
        "sandbox_config": {
            "image": "python:3.11-slim",
            "memory_limit": "256m",
        }
    }
}


class ExerciseGeneratorService:
    """
    Generates practice exercises dynamically from course content.

    Features:
    - Fetches course data from course-generator API
    - Retrieves RAG context for source documents
    - Uses GPT-4 to generate tailored exercises
    - Creates validation checks and sandbox configs
    - Caches generated exercises per course
    """

    def __init__(self):
        self.llm = ChatOpenAI(model=get_model_name("quality") if _USE_SHARED_LLM else "gpt-4o", temperature=0.7)
        self.llm_mini = ChatOpenAI(model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini", temperature=0.3)
        self.course_generator_url = os.getenv("COURSE_GENERATOR_URL", "http://course-generator:8007")
        self._http_client: Optional[httpx.AsyncClient] = None

        # In-memory cache for generated exercises
        self._exercise_cache: Dict[str, GeneratedExerciseSet] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def close(self):
        """Close HTTP client"""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def get_course_content(self, course_id: str) -> Optional[Dict[str, Any]]:
        """Fetch course content from course-generator service"""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.course_generator_url}/api/v1/courses/jobs/{course_id}"
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"[EXERCISE_GEN] Failed to fetch course {course_id}: {response.status_code}", flush=True)
                return None
        except Exception as e:
            print(f"[EXERCISE_GEN] Error fetching course: {e}", flush=True)
            return None

    async def get_rag_context(self, course_id: str, user_id: str = "system") -> Optional[str]:
        """Fetch RAG context for course source documents"""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.course_generator_url}/api/v1/courses/{course_id}/sources/context",
                params={"user_id": user_id, "max_chunks": 20}
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("context", "")
            return None
        except Exception as e:
            print(f"[EXERCISE_GEN] Error fetching RAG context: {e}", flush=True)
            return None

    def _detect_technologies(self, course_content: Dict[str, Any]) -> List[str]:
        """Detect technologies mentioned in course content"""
        technologies = set()

        # Keywords to detect
        tech_keywords = {
            "python": ["python", "pip", "pytest", "django", "flask", "pandas", "numpy"],
            "docker": ["docker", "dockerfile", "container", "docker-compose", "image"],
            "kubernetes": ["kubernetes", "k8s", "kubectl", "pod", "deployment", "helm"],
            "terraform": ["terraform", "tf", "infrastructure as code", "iac", "hcl"],
            "linux": ["linux", "bash", "shell", "ubuntu", "centos", "unix", "terminal"],
            "git": ["git", "github", "gitlab", "commit", "branch", "merge", "repository"],
            "database": ["database", "sql", "postgres", "mysql", "mongodb", "redis"],
            "javascript": ["javascript", "node", "npm", "react", "vue", "typescript"],
            "cicd": ["ci/cd", "jenkins", "github actions", "gitlab ci", "pipeline"],
        }

        # Get text from outline
        outline = course_content.get("outline", {})
        title = outline.get("title", "").lower()
        description = outline.get("description", "").lower()

        # Get text from sections and lectures
        sections_text = ""
        for section in outline.get("sections", []):
            sections_text += section.get("title", "").lower() + " "
            sections_text += section.get("description", "").lower() + " "
            for lecture in section.get("lectures", []):
                sections_text += lecture.get("title", "").lower() + " "
                sections_text += lecture.get("description", "").lower() + " "

        all_text = f"{title} {description} {sections_text}"

        # Detect technologies
        for tech, keywords in tech_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    technologies.add(tech)
                    break

        return list(technologies) if technologies else ["python"]  # Default to Python

    def _get_exercise_category(self, tech: str) -> ExerciseCategory:
        """Map technology to exercise category"""
        mapping = {
            "python": ExerciseCategory.PYTHON,
            "docker": ExerciseCategory.DOCKER,
            "kubernetes": ExerciseCategory.KUBERNETES,
            "terraform": ExerciseCategory.TERRAFORM,
            "linux": ExerciseCategory.LINUX,
            "git": ExerciseCategory.GIT,
            "database": ExerciseCategory.DATABASES,
            "javascript": ExerciseCategory.PYTHON,  # Fallback
            "cicd": ExerciseCategory.CI_CD,
        }
        return mapping.get(tech, ExerciseCategory.PYTHON)

    def _get_sandbox_config(self, tech: str) -> Dict[str, Any]:
        """Get sandbox configuration for a technology"""
        return SANDBOX_CONFIGS.get(tech, SANDBOX_CONFIGS["default"])

    async def generate_exercises_for_course(
        self,
        course_id: str,
        config: Optional[ExerciseGenerationConfig] = None,
        force_regenerate: bool = False,
    ) -> GeneratedExerciseSet:
        """
        Generate a complete set of exercises for a course.

        Args:
            course_id: The course job ID
            config: Exercise generation configuration
            force_regenerate: If True, regenerate even if cached

        Returns:
            GeneratedExerciseSet with all generated exercises
        """
        # Check cache first
        if not force_regenerate and course_id in self._exercise_cache:
            print(f"[EXERCISE_GEN] Returning cached exercises for {course_id}", flush=True)
            return self._exercise_cache[course_id]

        config = config or ExerciseGenerationConfig()

        # Fetch course content
        course_data = await self.get_course_content(course_id)
        if not course_data:
            raise ValueError(f"Course {course_id} not found or not completed")

        if course_data.get("status") != "completed":
            raise ValueError(f"Course {course_id} is not completed yet (status: {course_data.get('status')})")

        outline = course_data.get("outline", {})
        course_title = outline.get("title", "Unknown Course")
        course_category = outline.get("category", "tech")

        # Fetch RAG context if available
        rag_context = await self.get_rag_context(course_id)

        # Detect technologies
        technologies = self._detect_technologies(course_data)
        print(f"[EXERCISE_GEN] Detected technologies: {technologies}", flush=True)

        # Generate exercises for each section
        all_exercises = []
        sections = outline.get("sections", [])

        for section_idx, section in enumerate(sections):
            section_title = section.get("title", "")
            section_desc = section.get("description", "")
            lectures = section.get("lectures", [])

            # Determine difficulty based on section position
            if config.difficulty_progression:
                if section_idx < len(sections) // 3:
                    base_difficulty = DifficultyLevel.BEGINNER
                elif section_idx < 2 * len(sections) // 3:
                    base_difficulty = DifficultyLevel.INTERMEDIATE
                else:
                    base_difficulty = DifficultyLevel.ADVANCED
            else:
                base_difficulty = DifficultyLevel.INTERMEDIATE

            # Generate exercises for section
            section_exercises = await self._generate_section_exercises(
                course_title=course_title,
                section_title=section_title,
                section_description=section_desc,
                lectures=lectures,
                technologies=technologies,
                difficulty=base_difficulty,
                config=config,
                rag_context=rag_context,
                course_id=course_id,
                section_id=section.get("id", str(section_idx)),
            )

            all_exercises.extend(section_exercises)

        # Create exercise set
        exercise_set = GeneratedExerciseSet(
            course_id=course_id,
            course_title=course_title,
            category=course_category,
            exercises=all_exercises,
            source_context=rag_context[:500] if rag_context else None,
        )

        # Cache the result
        self._exercise_cache[course_id] = exercise_set

        print(f"[EXERCISE_GEN] Generated {len(all_exercises)} exercises for course {course_id}", flush=True)

        return exercise_set

    async def _generate_section_exercises(
        self,
        course_title: str,
        section_title: str,
        section_description: str,
        lectures: List[Dict],
        technologies: List[str],
        difficulty: DifficultyLevel,
        config: ExerciseGenerationConfig,
        rag_context: Optional[str],
        course_id: str,
        section_id: str,
    ) -> List[Exercise]:
        """Generate exercises for a single section"""
        exercises = []

        # Build lecture content summary
        lectures_summary = "\n".join([
            f"- {l.get('title', '')}: {l.get('description', '')}"
            for l in lectures
        ])

        # Determine exercise types to generate
        exercise_types = []
        if config.include_coding:
            exercise_types.append("coding")
        if config.include_debugging:
            exercise_types.append("debugging")
        if config.include_quiz:
            exercise_types.append("multiple_choice")
        if config.include_architecture:
            exercise_types.append("architecture")

        if not exercise_types:
            exercise_types = ["coding"]

        # Generate exercises for this section
        num_exercises = min(config.exercises_per_section, len(exercise_types) * 2)

        for i in range(num_exercises):
            ex_type = exercise_types[i % len(exercise_types)]
            tech = technologies[i % len(technologies)]

            # Vary difficulty slightly within section
            if i > 0 and config.difficulty_progression:
                difficulty_levels = [DifficultyLevel.BEGINNER, DifficultyLevel.INTERMEDIATE,
                                    DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]
                current_idx = difficulty_levels.index(difficulty)
                # Slight progression within section
                new_idx = min(current_idx + (i // 2), len(difficulty_levels) - 1)
                ex_difficulty = difficulty_levels[new_idx]
            else:
                ex_difficulty = difficulty

            # Generate the exercise
            exercise = await self._generate_single_exercise(
                course_title=course_title,
                section_title=section_title,
                section_description=section_description,
                lectures_summary=lectures_summary,
                technology=tech,
                exercise_type=ex_type,
                difficulty=ex_difficulty,
                rag_context=rag_context,
                course_id=course_id,
                section_id=section_id,
                exercise_index=i,
            )

            if exercise:
                exercises.append(exercise)

        return exercises

    async def _generate_single_exercise(
        self,
        course_title: str,
        section_title: str,
        section_description: str,
        lectures_summary: str,
        technology: str,
        exercise_type: str,
        difficulty: DifficultyLevel,
        rag_context: Optional[str],
        course_id: str,
        section_id: str,
        exercise_index: int,
    ) -> Optional[Exercise]:
        """Generate a single exercise using LLM"""

        # Build context
        rag_section = ""
        if rag_context:
            rag_section = f"""
Source Documentation Context:
{rag_context[:2000]}
"""

        # Prepare the prompt based on exercise type
        type_instructions = self._get_type_specific_instructions(exercise_type, technology)

        prompt = f"""Generate a practical {exercise_type} exercise for the following course content.

Course: {course_title}
Section: {section_title}
Section Description: {section_description}

Lectures in this section:
{lectures_summary}

{rag_section}

Technology Focus: {technology}
Difficulty Level: {difficulty.value}
Exercise Type: {exercise_type}

{type_instructions}

Generate a comprehensive exercise in JSON format with the following structure:
{{
    "title": "Clear, descriptive title (max 80 chars)",
    "description": "Brief description of what the learner will accomplish (1-2 sentences)",
    "instructions": "Detailed step-by-step instructions in markdown format. Include:\\n- Context and background\\n- Clear objectives\\n- Step-by-step tasks\\n- Expected outcomes",
    "starter_code": "Initial code or configuration for the learner to start with (if applicable)",
    "starter_files": {{"filename.ext": "content"}},
    "hints": ["Progressive hint 1 (vague)", "Hint 2 (more specific)", "Hint 3 (almost gives away the answer)"],
    "solution": "Complete working solution code",
    "solution_explanation": "Detailed explanation of why this solution works and key concepts",
    "validation_checks": [
        {{
            "name": "check_name",
            "description": "What this check validates",
            "check_type": "output|file|code_contains|command",
            "patterns": ["regex patterns to match"],
            "contains": ["strings that must be present"],
            "points": 20,
            "required": true
        }}
    ],
    "estimated_minutes": 15,
    "points": 100,
    "tags": ["relevant", "tags", "for", "searching"]
}}

IMPORTANT:
- The exercise MUST be directly related to the course content
- Make it practical and hands-on
- Ensure validation_checks can actually verify the solution
- Include at least 3 validation checks
- Make hints progressively more helpful
- Solution must pass all validation checks
- Use {technology} specific patterns and best practices

Respond ONLY with valid JSON, no additional text.
"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=f"You are an expert {technology} instructor creating practice exercises. Generate exercises that are practical, educational, and directly test the skills taught in the course."),
                HumanMessage(content=prompt)
            ])

            # Parse JSON response
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            exercise_data = json.loads(content.strip())

            # Create exercise with proper IDs and metadata
            exercise_id = f"{course_id}-{section_id}-{exercise_index}-{uuid.uuid4().hex[:6]}"
            category = self._get_exercise_category(technology)
            sandbox_config = self._get_sandbox_config(technology)

            # Build validation checks
            validation_checks = []
            for check_data in exercise_data.get("validation_checks", []):
                validation_checks.append(ValidationCheck(
                    name=check_data.get("name", "check"),
                    description=check_data.get("description"),
                    check_type=check_data.get("check_type", "output"),
                    patterns=check_data.get("patterns"),
                    contains=check_data.get("contains"),
                    expected_value=check_data.get("expected_value"),
                    points=check_data.get("points", 20),
                    required=check_data.get("required", False),
                ))

            # Map exercise type string to enum
            type_mapping = {
                "coding": ExerciseType.CODING,
                "debugging": ExerciseType.DEBUGGING,
                "multiple_choice": ExerciseType.MULTIPLE_CHOICE,
                "architecture": ExerciseType.ARCHITECTURE,
                "configuration": ExerciseType.CONFIGURATION,
                "troubleshooting": ExerciseType.TROUBLESHOOTING,
            }
            ex_type = type_mapping.get(exercise_type, ExerciseType.CODING)

            return Exercise(
                id=exercise_id,
                title=exercise_data.get("title", "Untitled Exercise"),
                description=exercise_data.get("description", ""),
                instructions=exercise_data.get("instructions", ""),
                difficulty=difficulty,
                type=ex_type,
                category=category,
                tags=exercise_data.get("tags", []) + [technology, section_title.lower()],
                starter_code=exercise_data.get("starter_code"),
                starter_files=exercise_data.get("starter_files", {}),
                validation_checks=validation_checks,
                hints=exercise_data.get("hints", []),
                solution=exercise_data.get("solution"),
                solution_explanation=exercise_data.get("solution_explanation"),
                course_id=course_id,
                lecture_ids=[],  # Could be populated with actual lecture IDs
                sandbox_type=sandbox_config["sandbox_type"],
                sandbox_config=sandbox_config["sandbox_config"],
                timeout_seconds=300,
                estimated_minutes=exercise_data.get("estimated_minutes", 15),
                points=exercise_data.get("points", 100),
            )

        except json.JSONDecodeError as e:
            print(f"[EXERCISE_GEN] JSON parse error: {e}", flush=True)
            return None
        except Exception as e:
            print(f"[EXERCISE_GEN] Error generating exercise: {e}", flush=True)
            return None

    def _get_type_specific_instructions(self, exercise_type: str, technology: str) -> str:
        """Get type-specific generation instructions"""
        instructions = {
            "coding": f"""
For this CODING exercise:
- Provide clear starter code that sets up the problem
- The learner should write functional {technology} code
- Include input/output examples
- Validation should check code output and key patterns
""",
            "debugging": f"""
For this DEBUGGING exercise:
- Provide broken code with subtle bugs
- Include error messages the learner might see
- The solution should fix all bugs
- Validation should verify the fixed behavior
""",
            "multiple_choice": """
For this MULTIPLE CHOICE exercise:
- Generate a quiz-style question in the instructions
- Include 4 options labeled A, B, C, D
- The solution should be the correct answer with explanation
- Validation should check the selected answer
- starter_code should contain the question and options formatted nicely
""",
            "architecture": f"""
For this ARCHITECTURE exercise:
- Present a system design problem
- Ask the learner to create a design document or diagram description
- Include constraints and requirements
- Validation should check for key components mentioned
""",
            "configuration": f"""
For this CONFIGURATION exercise:
- Provide a scenario requiring {technology} configuration
- Include partial config as starter
- The learner should complete the configuration
- Validation should check for required settings
""",
            "troubleshooting": f"""
For this TROUBLESHOOTING exercise:
- Present a broken system or failing scenario
- Provide logs, error messages, and symptoms
- The learner should identify and fix the issue
- Validation should verify the fix
""",
        }
        return instructions.get(exercise_type, instructions["coding"])

    async def get_exercises_for_course(
        self,
        course_id: str,
        difficulty: Optional[DifficultyLevel] = None,
        exercise_type: Optional[ExerciseType] = None,
    ) -> List[Exercise]:
        """
        Get exercises for a course, generating them if not cached.

        Args:
            course_id: The course job ID
            difficulty: Optional filter by difficulty
            exercise_type: Optional filter by type

        Returns:
            List of exercises for the course
        """
        # Generate if not cached
        if course_id not in self._exercise_cache:
            await self.generate_exercises_for_course(course_id)

        exercise_set = self._exercise_cache.get(course_id)
        if not exercise_set:
            return []

        exercises = exercise_set.exercises

        # Apply filters
        if difficulty:
            exercises = [e for e in exercises if e.difficulty == difficulty]
        if exercise_type:
            exercises = [e for e in exercises if e.type == exercise_type]

        return exercises

    def clear_cache(self, course_id: Optional[str] = None):
        """Clear cached exercises"""
        if course_id:
            self._exercise_cache.pop(course_id, None)
        else:
            self._exercise_cache.clear()

    async def regenerate_exercise(
        self,
        course_id: str,
        exercise_id: str,
        feedback: Optional[str] = None,
    ) -> Optional[Exercise]:
        """
        Regenerate a specific exercise, optionally incorporating feedback.

        Args:
            course_id: The course job ID
            exercise_id: The exercise to regenerate
            feedback: Optional feedback about what to improve

        Returns:
            New exercise or None if failed
        """
        if course_id not in self._exercise_cache:
            return None

        exercise_set = self._exercise_cache[course_id]
        old_exercise = next((e for e in exercise_set.exercises if e.id == exercise_id), None)

        if not old_exercise:
            return None

        # Build regeneration prompt
        prompt = f"""Improve this exercise based on feedback.

Original Exercise:
Title: {old_exercise.title}
Type: {old_exercise.type.value}
Difficulty: {old_exercise.difficulty.value}
Instructions: {old_exercise.instructions[:500]}

Feedback: {feedback or "Make the exercise more engaging and practical."}

Generate an improved version in the same JSON format, keeping the same type and difficulty level but addressing the feedback.
"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are improving a practice exercise based on feedback."),
                HumanMessage(content=prompt)
            ])

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            exercise_data = json.loads(content.strip())

            # Create new exercise with same category and sandbox
            new_exercise = Exercise(
                id=f"{exercise_id}-v2",
                title=exercise_data.get("title", old_exercise.title),
                description=exercise_data.get("description", old_exercise.description),
                instructions=exercise_data.get("instructions", old_exercise.instructions),
                difficulty=old_exercise.difficulty,
                type=old_exercise.type,
                category=old_exercise.category,
                tags=old_exercise.tags,
                starter_code=exercise_data.get("starter_code", old_exercise.starter_code),
                hints=exercise_data.get("hints", old_exercise.hints),
                solution=exercise_data.get("solution", old_exercise.solution),
                solution_explanation=exercise_data.get("solution_explanation", old_exercise.solution_explanation),
                course_id=course_id,
                sandbox_type=old_exercise.sandbox_type,
                sandbox_config=old_exercise.sandbox_config,
                estimated_minutes=old_exercise.estimated_minutes,
                points=old_exercise.points,
            )

            # Replace in cache
            exercise_set.exercises = [
                new_exercise if e.id == exercise_id else e
                for e in exercise_set.exercises
            ]

            return new_exercise

        except Exception as e:
            print(f"[EXERCISE_GEN] Error regenerating exercise: {e}", flush=True)
            return None


# Singleton instance
_exercise_generator: Optional[ExerciseGeneratorService] = None


def get_exercise_generator() -> ExerciseGeneratorService:
    """Get or create the exercise generator singleton"""
    global _exercise_generator
    if _exercise_generator is None:
        _exercise_generator = ExerciseGeneratorService()
    return _exercise_generator
