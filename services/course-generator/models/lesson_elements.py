"""
Category-based Lesson Elements Configuration

Defines lesson elements specific to each profile category,
common elements for all courses, and quiz configuration.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .course_models import ProfileCategory


class LessonElementType(str, Enum):
    """Types of lesson elements available"""

    # Common elements (all categories)
    CONCEPT_INTRO = "concept_intro"
    VOICEOVER = "voiceover"
    CURRICULUM_SLIDE = "curriculum_slide"
    CONCLUSION = "conclusion"
    QUIZ = "quiz"

    # Tech elements
    CODE_DEMO = "code_demo"
    TERMINAL_OUTPUT = "terminal_output"
    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    DEBUG_TIPS = "debug_tips"
    CODE_EXECUTION = "code_execution"

    # Business elements
    CASE_STUDY = "case_study"
    FRAMEWORK_TEMPLATE = "framework_template"
    ROI_METRICS = "roi_metrics"
    ACTION_CHECKLIST = "action_checklist"
    MARKET_ANALYSIS = "market_analysis"

    # Health elements
    EXERCISE_DEMO = "exercise_demo"
    SAFETY_WARNING = "safety_warning"
    BODY_DIAGRAM = "body_diagram"
    PROGRESSION_PLAN = "progression_plan"
    REST_GUIDANCE = "rest_guidance"

    # Creative elements
    BEFORE_AFTER = "before_after"
    TECHNIQUE_DEMO = "technique_demo"
    TOOL_TUTORIAL = "tool_tutorial"
    CREATIVE_EXERCISE = "creative_exercise"
    CRITIQUE_SECTION = "critique_section"

    # Education elements
    MEMORY_AID = "memory_aid"
    PRACTICE_PROBLEM = "practice_problem"
    MULTIPLE_EXPLANATIONS = "multiple_explanations"
    SUMMARY_CARD = "summary_card"

    # Lifestyle elements
    DAILY_ROUTINE = "daily_routine"
    REFLECTION_EXERCISE = "reflection_exercise"
    GOAL_SETTING = "goal_setting"
    HABIT_TRACKER = "habit_tracker"
    MILESTONE = "milestone"


class LessonElement(BaseModel):
    """Definition of a lesson element"""

    id: LessonElementType
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Description of what this element does")
    icon: str = Field(default="", description="Icon identifier or emoji")
    is_common: bool = Field(default=False, description="Whether this is a common element for all categories")
    is_required: bool = Field(default=False, description="Whether this element is required")
    categories: List[ProfileCategory] = Field(default_factory=list, description="Categories this element applies to")
    presentation_type: str = Field(
        default="slide", description="How this element is presented: slide, animation, overlay, interactive"
    )


# Common elements for ALL courses
COMMON_ELEMENTS: List[LessonElement] = [
    LessonElement(
        id=LessonElementType.CONCEPT_INTRO,
        name="Introduction du concept",
        description="Slide d'introduction expliquant le concept principal de la leçon",
        icon="💡",
        is_common=True,
        is_required=True,
        presentation_type="slide",
    ),
    LessonElement(
        id=LessonElementType.VOICEOVER,
        name="Narration vocale",
        description="Explication audio accompagnant les visuels",
        icon="🎙️",
        is_common=True,
        is_required=True,
        presentation_type="audio",
    ),
    LessonElement(
        id=LessonElementType.CURRICULUM_SLIDE,
        name="Slide de curriculum",
        description="Position de la leçon dans le cours",
        icon="📍",
        is_common=True,
        is_required=True,
        presentation_type="slide",
    ),
    LessonElement(
        id=LessonElementType.CONCLUSION,
        name="Conclusion",
        description="Récapitulatif des points clés de la leçon",
        icon="✅",
        is_common=True,
        is_required=True,
        presentation_type="slide",
    ),
    LessonElement(
        id=LessonElementType.QUIZ,
        name="Quiz d'évaluation",
        description="Questions pour évaluer la compréhension",
        icon="❓",
        is_common=True,
        is_required=True,  # Quiz is now required
        presentation_type="interactive",
    ),
]


# Category-specific elements
CATEGORY_ELEMENTS: Dict[ProfileCategory, List[LessonElement]] = {
    ProfileCategory.TECH: [
        LessonElement(
            id=LessonElementType.CODE_DEMO,
            name="Démo de code",
            description="Animation de frappe de code avec explication",
            icon="💻",
            categories=[ProfileCategory.TECH],
            presentation_type="animation",
        ),
        LessonElement(
            id=LessonElementType.TERMINAL_OUTPUT,
            name="Sortie terminal",
            description="Affichage du résultat d'exécution dans un terminal",
            icon="⬛",
            categories=[ProfileCategory.TECH],
            presentation_type="animation",
        ),
        LessonElement(
            id=LessonElementType.ARCHITECTURE_DIAGRAM,
            name="Diagramme d'architecture",
            description="Schéma technique illustrant l'architecture ou le flux",
            icon="📐",
            categories=[ProfileCategory.TECH],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.DEBUG_TIPS,
            name="Conseils de débogage",
            description="Erreurs courantes et comment les résoudre",
            icon="🐛",
            categories=[ProfileCategory.TECH],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.CODE_EXECUTION,
            name="Exécution de code",
            description="Exécution en direct du code avec sortie",
            icon="▶️",
            categories=[ProfileCategory.TECH],
            presentation_type="animation",
        ),
    ],
    ProfileCategory.BUSINESS: [
        LessonElement(
            id=LessonElementType.CASE_STUDY,
            name="Étude de cas",
            description="Analyse d'un cas réel illustrant le concept",
            icon="📊",
            categories=[ProfileCategory.BUSINESS],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.FRAMEWORK_TEMPLATE,
            name="Framework/Template",
            description="Modèle ou framework actionnable à utiliser",
            icon="📋",
            categories=[ProfileCategory.BUSINESS],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.ROI_METRICS,
            name="Métriques ROI",
            description="Indicateurs de performance et retour sur investissement",
            icon="📈",
            categories=[ProfileCategory.BUSINESS],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.ACTION_CHECKLIST,
            name="Checklist d'actions",
            description="Liste d'actions concrètes à mettre en œuvre",
            icon="✔️",
            categories=[ProfileCategory.BUSINESS],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.MARKET_ANALYSIS,
            name="Analyse de marché",
            description="Données et tendances du marché",
            icon="🌍",
            categories=[ProfileCategory.BUSINESS],
            presentation_type="slide",
        ),
    ],
    ProfileCategory.HEALTH: [
        LessonElement(
            id=LessonElementType.EXERCISE_DEMO,
            name="Démonstration d'exercice",
            description="Vidéo ou animation montrant l'exercice",
            icon="🏋️",
            categories=[ProfileCategory.HEALTH],
            presentation_type="animation",
        ),
        LessonElement(
            id=LessonElementType.SAFETY_WARNING,
            name="Avertissement sécurité",
            description="Précautions et contre-indications",
            icon="⚠️",
            categories=[ProfileCategory.HEALTH],
            presentation_type="overlay",
        ),
        LessonElement(
            id=LessonElementType.BODY_DIAGRAM,
            name="Diagramme anatomique",
            description="Illustration des muscles ou parties du corps concernés",
            icon="🫀",
            categories=[ProfileCategory.HEALTH],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.PROGRESSION_PLAN,
            name="Plan de progression",
            description="Étapes pour progresser dans l'exercice",
            icon="📶",
            categories=[ProfileCategory.HEALTH],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.REST_GUIDANCE,
            name="Conseils de récupération",
            description="Temps de repos et récupération recommandés",
            icon="😴",
            categories=[ProfileCategory.HEALTH],
            presentation_type="slide",
        ),
    ],
    ProfileCategory.CREATIVE: [
        LessonElement(
            id=LessonElementType.BEFORE_AFTER,
            name="Avant/Après",
            description="Comparaison visuelle du résultat",
            icon="🔄",
            categories=[ProfileCategory.CREATIVE],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.TECHNIQUE_DEMO,
            name="Démonstration technique",
            description="Étapes de réalisation de la technique",
            icon="🎨",
            categories=[ProfileCategory.CREATIVE],
            presentation_type="animation",
        ),
        LessonElement(
            id=LessonElementType.TOOL_TUTORIAL,
            name="Tutoriel outil",
            description="Guide d'utilisation de l'outil ou logiciel",
            icon="🛠️",
            categories=[ProfileCategory.CREATIVE],
            presentation_type="animation",
        ),
        LessonElement(
            id=LessonElementType.CREATIVE_EXERCISE,
            name="Exercice créatif",
            description="Exercice pratique à réaliser",
            icon="✏️",
            categories=[ProfileCategory.CREATIVE],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.CRITIQUE_SECTION,
            name="Section critique",
            description="Analyse et amélioration d'un travail",
            icon="🔍",
            categories=[ProfileCategory.CREATIVE],
            presentation_type="slide",
        ),
    ],
    ProfileCategory.EDUCATION: [
        LessonElement(
            id=LessonElementType.MEMORY_AID,
            name="Aide-mémoire",
            description="Mnémotechniques et astuces de mémorisation",
            icon="🧠",
            categories=[ProfileCategory.EDUCATION],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.PRACTICE_PROBLEM,
            name="Exercice pratique",
            description="Problème à résoudre pour s'entraîner",
            icon="📝",
            categories=[ProfileCategory.EDUCATION],
            presentation_type="interactive",
        ),
        LessonElement(
            id=LessonElementType.MULTIPLE_EXPLANATIONS,
            name="Explications multiples",
            description="Plusieurs approches pour expliquer le concept",
            icon="🔀",
            categories=[ProfileCategory.EDUCATION],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.SUMMARY_CARD,
            name="Fiche résumé",
            description="Carte de synthèse à conserver",
            icon="📇",
            categories=[ProfileCategory.EDUCATION],
            presentation_type="slide",
        ),
    ],
    ProfileCategory.LIFESTYLE: [
        LessonElement(
            id=LessonElementType.DAILY_ROUTINE,
            name="Routine quotidienne",
            description="Actions à intégrer dans le quotidien",
            icon="☀️",
            categories=[ProfileCategory.LIFESTYLE],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.REFLECTION_EXERCISE,
            name="Exercice de réflexion",
            description="Questions pour réfléchir et s'auto-évaluer",
            icon="💭",
            categories=[ProfileCategory.LIFESTYLE],
            presentation_type="interactive",
        ),
        LessonElement(
            id=LessonElementType.GOAL_SETTING,
            name="Définition d'objectifs",
            description="Framework pour définir ses objectifs",
            icon="🎯",
            categories=[ProfileCategory.LIFESTYLE],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.HABIT_TRACKER,
            name="Suivi d'habitudes",
            description="Système pour suivre ses progrès",
            icon="📅",
            categories=[ProfileCategory.LIFESTYLE],
            presentation_type="slide",
        ),
        LessonElement(
            id=LessonElementType.MILESTONE,
            name="Jalon/Célébration",
            description="Points d'étape et célébration des progrès",
            icon="🏆",
            categories=[ProfileCategory.LIFESTYLE],
            presentation_type="slide",
        ),
    ],
}


class QuizFrequency(str, Enum):
    """How often quizzes should appear in the course"""

    PER_LECTURE = "per_lecture"  # Quiz at the end of each lecture
    PER_SECTION = "per_section"  # Quiz at the end of each section
    END_OF_COURSE = "end_of_course"  # Single quiz at the end
    CUSTOM = "custom"  # Every N lectures


class QuizQuestionType(str, Enum):
    """Types of quiz questions (Udemy style)"""

    MULTIPLE_CHOICE = "multiple_choice"  # Single correct answer
    MULTI_SELECT = "multi_select"  # Multiple correct answers
    TRUE_FALSE = "true_false"  # True or False
    FILL_BLANK = "fill_blank"  # Fill in the blank
    MATCHING = "matching"  # Match items


class QuizQuestion(BaseModel):
    """A single quiz question"""

    id: str = Field(default="", description="Question ID")
    type: QuizQuestionType = Field(default=QuizQuestionType.MULTIPLE_CHOICE)
    question: str = Field(..., description="The question text")
    options: List[str] = Field(default_factory=list, description="Answer options")
    correct_answers: List[int] = Field(default_factory=list, description="Indices of correct answers")
    explanation: str = Field(default="", description="Explanation of the correct answer")
    points: int = Field(default=1, description="Points for this question")
    # For matching questions
    matching_pairs: Optional[Dict[str, str]] = Field(None, description="Key-value pairs to match")


class QuizConfig(BaseModel):
    """Configuration for quiz generation"""

    enabled: bool = Field(default=True, description="Whether quizzes are enabled")
    frequency: QuizFrequency = Field(default=QuizFrequency.PER_SECTION)
    custom_frequency: Optional[int] = Field(None, description="Quiz every N lectures (if frequency=custom)")
    questions_per_quiz: int = Field(default=5, ge=1, le=20, description="Number of questions per quiz")
    question_types: List[QuizQuestionType] = Field(
        default=[QuizQuestionType.MULTIPLE_CHOICE, QuizQuestionType.TRUE_FALSE],
        description="Types of questions to include",
    )
    passing_score: int = Field(default=70, ge=0, le=100, description="Minimum score to pass (%)")
    show_explanations: bool = Field(default=True, description="Show explanations after answers")
    allow_retry: bool = Field(default=True, description="Allow retrying the quiz")


class Quiz(BaseModel):
    """A complete quiz"""

    id: str = Field(default="", description="Quiz ID")
    title: str = Field(default="", description="Quiz title")
    description: str = Field(default="", description="Quiz description")
    questions: List[QuizQuestion] = Field(default_factory=list)
    time_limit_minutes: Optional[int] = Field(None, description="Time limit in minutes (optional)")
    total_points: int = Field(default=0, description="Total possible points")
    passing_score: int = Field(default=70, description="Minimum passing percentage")


class AdaptiveLessonElementConfig(BaseModel):
    """Adaptive lesson element configuration based on category"""

    # Common elements (always included, some required)
    common_elements: Dict[LessonElementType, bool] = Field(
        default_factory=lambda: {
            LessonElementType.CONCEPT_INTRO: True,
            LessonElementType.VOICEOVER: True,
            LessonElementType.CURRICULUM_SLIDE: True,
            LessonElementType.CONCLUSION: True,
            LessonElementType.QUIZ: True,
        }
    )

    # Category-specific elements selected by user
    category_elements: Dict[LessonElementType, bool] = Field(default_factory=dict)

    # AI-suggested elements (from Option C)
    ai_suggested_elements: List[LessonElementType] = Field(default_factory=list)

    # Quiz configuration
    quiz_config: QuizConfig = Field(default_factory=QuizConfig)


def get_elements_for_category(category: ProfileCategory) -> List[LessonElement]:
    """Get all available elements for a category (common + category-specific)"""
    return COMMON_ELEMENTS + CATEGORY_ELEMENTS.get(category, [])


def get_default_elements_for_category(category: ProfileCategory) -> Dict[LessonElementType, bool]:
    """Get default enabled elements for a category"""
    elements = {}

    # Common elements - all enabled by default
    for el in COMMON_ELEMENTS:
        elements[el.id] = True

    # Category elements - first 3 enabled by default
    category_els = CATEGORY_ELEMENTS.get(category, [])
    for i, el in enumerate(category_els):
        elements[el.id] = i < 3  # Enable first 3 by default

    return elements


def get_element_by_id(element_id: LessonElementType) -> Optional[LessonElement]:
    """Get element definition by ID"""
    # Check common elements
    for el in COMMON_ELEMENTS:
        if el.id == element_id:
            return el

    # Check category elements
    for category_elements in CATEGORY_ELEMENTS.values():
        for el in category_elements:
            if el.id == element_id:
                return el

    return None
