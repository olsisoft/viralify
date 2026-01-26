"""
MAESTRO Data Models

Core data structures for the MAESTRO course generation system.
Defines concepts, modules, lessons, and difficulty calibration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import uuid


class SkillLevel(str, Enum):
    """Skill levels mapped from difficulty scores"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    VERY_ADVANCED = "very_advanced"
    EXPERT = "expert"


class BloomLevel(str, Enum):
    """Bloom's Taxonomy cognitive levels"""
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class ProgressionPath(str, Enum):
    """Pre-defined learning progression paths"""
    BEGINNER_TO_INTERMEDIATE = "beginner_to_intermediate"
    INTERMEDIATE_TO_ADVANCED = "intermediate_to_advanced"
    ADVANCED_TO_EXPERT = "advanced_to_expert"
    FULL_RANGE = "full_range"


class ScriptSegmentType(str, Enum):
    """Types of script segments"""
    INTRO = "intro"
    EXPLANATION = "explanation"
    EXAMPLE = "example"
    SUMMARY = "summary"
    TRANSITION = "transition"


# Bloom level to cognitive load score mapping
BLOOM_TO_COGNITIVE_LOAD: Dict[BloomLevel, float] = {
    BloomLevel.REMEMBER: 0.1,
    BloomLevel.UNDERSTAND: 0.25,
    BloomLevel.APPLY: 0.45,
    BloomLevel.ANALYZE: 0.6,
    BloomLevel.EVALUATE: 0.8,
    BloomLevel.CREATE: 0.95,
}

# Skill level to difficulty score ranges
SKILL_LEVEL_RANGES: Dict[SkillLevel, Tuple[float, float]] = {
    SkillLevel.BEGINNER: (0.0, 0.20),
    SkillLevel.INTERMEDIATE: (0.20, 0.40),
    SkillLevel.ADVANCED: (0.40, 0.60),
    SkillLevel.VERY_ADVANCED: (0.60, 0.80),
    SkillLevel.EXPERT: (0.80, 1.0),
}

# Progression path to skill level ranges
PROGRESSION_RANGES: Dict[ProgressionPath, Tuple[SkillLevel, SkillLevel]] = {
    ProgressionPath.BEGINNER_TO_INTERMEDIATE: (SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE),
    ProgressionPath.INTERMEDIATE_TO_ADVANCED: (SkillLevel.INTERMEDIATE, SkillLevel.ADVANCED),
    ProgressionPath.ADVANCED_TO_EXPERT: (SkillLevel.ADVANCED, SkillLevel.EXPERT),
    ProgressionPath.FULL_RANGE: (SkillLevel.BEGINNER, SkillLevel.EXPERT),
}


@dataclass
class DifficultyVector:
    """4-dimensional difficulty vector for concept calibration"""
    conceptual_complexity: float = 0.5
    prerequisites_depth: float = 0.5
    information_density: float = 0.5
    cognitive_load: float = 0.5

    WEIGHTS = {
        "conceptual_complexity": 0.25,
        "prerequisites_depth": 0.20,
        "information_density": 0.25,
        "cognitive_load": 0.30,
    }

    def __post_init__(self):
        """Clamp values to [0.0, 1.0]"""
        self.conceptual_complexity = max(0.0, min(1.0, self.conceptual_complexity))
        self.prerequisites_depth = max(0.0, min(1.0, self.prerequisites_depth))
        self.information_density = max(0.0, min(1.0, self.information_density))
        self.cognitive_load = max(0.0, min(1.0, self.cognitive_load))

    @property
    def composite_score(self) -> float:
        """Calculate weighted composite difficulty score"""
        return (
            self.WEIGHTS["conceptual_complexity"] * self.conceptual_complexity +
            self.WEIGHTS["prerequisites_depth"] * self.prerequisites_depth +
            self.WEIGHTS["information_density"] * self.information_density +
            self.WEIGHTS["cognitive_load"] * self.cognitive_load
        )

    @property
    def skill_level(self) -> SkillLevel:
        """Convert composite score to skill level"""
        score = self.composite_score
        for level, (min_score, max_score) in SKILL_LEVEL_RANGES.items():
            if min_score <= score < max_score:
                return level
        return SkillLevel.EXPERT

    @property
    def bloom_level(self) -> BloomLevel:
        """Derive Bloom level from cognitive load"""
        load = self.cognitive_load
        if load < 0.15:
            return BloomLevel.REMEMBER
        elif load < 0.35:
            return BloomLevel.UNDERSTAND
        elif load < 0.50:
            return BloomLevel.APPLY
        elif load < 0.70:
            return BloomLevel.ANALYZE
        elif load < 0.85:
            return BloomLevel.EVALUATE
        return BloomLevel.CREATE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conceptual_complexity": round(self.conceptual_complexity, 3),
            "prerequisites_depth": round(self.prerequisites_depth, 3),
            "information_density": round(self.information_density, 3),
            "cognitive_load": round(self.cognitive_load, 3),
            "composite_score": round(self.composite_score, 3),
            "skill_level": self.skill_level.value,
            "bloom_level": self.bloom_level.value,
        }


@dataclass
class Concept:
    """A concept in the course curriculum"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    difficulty: DifficultyVector = field(default_factory=DifficultyVector)
    prerequisites: List[str] = field(default_factory=list)  # concept IDs
    estimated_duration_minutes: int = 10

    @property
    def skill_level(self) -> SkillLevel:
        return self.difficulty.skill_level

    @property
    def bloom_level(self) -> BloomLevel:
        return self.difficulty.bloom_level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "difficulty": self.difficulty.to_dict(),
            "prerequisites": self.prerequisites,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "skill_level": self.skill_level.value,
            "bloom_level": self.bloom_level.value,
        }


@dataclass
class ScriptSegment:
    """A segment of the lesson script"""
    type: ScriptSegmentType
    content: str
    duration_seconds: int
    key_points: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "content": self.content,
            "duration_seconds": self.duration_seconds,
            "key_points": self.key_points,
        }


@dataclass
class SlideContent:
    """Content for a single slide"""
    title: str
    bullet_points: List[str] = field(default_factory=list)
    visual_suggestion: str = ""
    speaker_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "bullet_points": self.bullet_points,
            "visual_suggestion": self.visual_suggestion,
            "speaker_notes": self.speaker_notes,
        }


@dataclass
class QuizQuestion:
    """A quiz question"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = "multiple_choice"  # multiple_choice, true_false, fill_blank
    question: str = ""
    options: List[str] = field(default_factory=list)
    correct_answers: List[int] = field(default_factory=list)
    explanation: str = ""
    bloom_level: BloomLevel = BloomLevel.UNDERSTAND
    points: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "question": self.question,
            "options": self.options,
            "correct_answers": self.correct_answers,
            "explanation": self.explanation,
            "bloom_level": self.bloom_level.value,
            "points": self.points,
        }


@dataclass
class PracticalExercise:
    """A practical exercise"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    instructions: List[str] = field(default_factory=list)
    starter_code: Optional[str] = None
    solution: str = ""
    hints: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 15

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "instructions": self.instructions,
            "starter_code": self.starter_code,
            "solution": self.solution,
            "hints": self.hints,
            "estimated_time_minutes": self.estimated_time_minutes,
        }


@dataclass
class Lesson:
    """A complete lesson generated for a concept"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    concept_id: str = ""
    title: str = ""
    description: str = ""

    # Script
    script: str = ""
    script_segments: List[ScriptSegment] = field(default_factory=list)
    key_takeaways: List[str] = field(default_factory=list)

    # Visual content
    slides: List[SlideContent] = field(default_factory=list)

    # Assessment
    quiz_questions: List[QuizQuestion] = field(default_factory=list)
    exercises: List[PracticalExercise] = field(default_factory=list)

    # Metadata
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE
    bloom_level: BloomLevel = BloomLevel.UNDERSTAND
    estimated_duration_minutes: int = 10
    sequence_order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "concept_id": self.concept_id,
            "title": self.title,
            "description": self.description,
            "script": self.script,
            "script_segments": [s.to_dict() for s in self.script_segments],
            "key_takeaways": self.key_takeaways,
            "slides": [s.to_dict() for s in self.slides],
            "quiz_questions": [q.to_dict() for q in self.quiz_questions],
            "exercises": [e.to_dict() for e in self.exercises],
            "skill_level": self.skill_level.value,
            "bloom_level": self.bloom_level.value,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "sequence_order": self.sequence_order,
        }


@dataclass
class Module:
    """A module grouping related concepts"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    learning_objectives: List[str] = field(default_factory=list)
    concept_ids: List[str] = field(default_factory=list)
    lessons: List[Lesson] = field(default_factory=list)
    skill_level_range: Tuple[SkillLevel, SkillLevel] = (SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE)

    @property
    def total_duration_minutes(self) -> int:
        return sum(l.estimated_duration_minutes for l in self.lessons)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "learning_objectives": self.learning_objectives,
            "concept_ids": self.concept_ids,
            "lessons": [l.to_dict() for l in self.lessons],
            "skill_level_range": [self.skill_level_range[0].value, self.skill_level_range[1].value],
            "total_duration_minutes": self.total_duration_minutes,
        }


@dataclass
class CoursePackage:
    """Complete generated course package"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    title: str = ""
    description: str = ""
    subject: str = ""
    language: str = "en"

    # Structure
    modules: List[Module] = field(default_factory=list)
    concepts: List[Concept] = field(default_factory=list)

    # Configuration
    progression_path: ProgressionPath = ProgressionPath.BEGINNER_TO_INTERMEDIATE
    total_duration_minutes: int = 0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    generation_time_seconds: float = 0.0
    tokens_used: int = 0

    @property
    def total_lessons(self) -> int:
        return sum(len(m.lessons) for m in self.modules)

    @property
    def total_concepts(self) -> int:
        return len(self.concepts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "subject": self.subject,
            "language": self.language,
            "modules": [m.to_dict() for m in self.modules],
            "concepts": [c.to_dict() for c in self.concepts],
            "progression_path": self.progression_path.value,
            "total_duration_minutes": self.total_duration_minutes,
            "total_lessons": self.total_lessons,
            "total_concepts": self.total_concepts,
            "created_at": self.created_at.isoformat(),
            "generation_time_seconds": round(self.generation_time_seconds, 2),
            "tokens_used": self.tokens_used,
        }


@dataclass
class CourseRequest:
    """Request to generate a course"""
    subject: str
    progression_path: ProgressionPath = ProgressionPath.BEGINNER_TO_INTERMEDIATE
    total_duration_hours: float = 5.0
    num_modules: int = 5
    language: str = "en"
    include_quizzes: bool = True
    include_exercises: bool = True
    questions_per_lesson: int = 3
    exercises_per_lesson: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "progression_path": self.progression_path.value,
            "total_duration_hours": self.total_duration_hours,
            "num_modules": self.num_modules,
            "language": self.language,
            "include_quizzes": self.include_quizzes,
            "include_exercises": self.include_exercises,
            "questions_per_lesson": self.questions_per_lesson,
            "exercises_per_lesson": self.exercises_per_lesson,
        }
