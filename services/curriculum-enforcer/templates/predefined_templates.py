"""
Predefined Curriculum Templates
Ready-to-use templates for different learning contexts.
"""

from typing import Dict, Optional, List
from datetime import datetime

from ..models.curriculum_models import (
    ContextType,
    LessonPhase,
    PhaseConfig,
    LessonTemplate,
    CurriculumTemplate,
)


# =============================================================================
# EDUCATION TEMPLATE
# Traditional learning: Hook → Concept → Theory → Code → Demo → Recap
# =============================================================================

EDUCATION_LESSON_DEFAULT = LessonTemplate(
    name="Standard Education Lesson",
    description="Engaging educational flow with hook, concept explanation, practical demo, and recap",
    total_duration_target_seconds=600,  # 10 minutes
    phases=[
        PhaseConfig(
            phase=LessonPhase.HOOK,
            required=True,
            order=0,
            min_duration_seconds=15,
            max_duration_seconds=45,
            slide_count=1,
            prompt_template="Start with an engaging question, surprising fact, or relatable problem. Make the learner feel the pain point.",
            tone="energetic",
            required_elements=["question_or_statement"]
        ),
        PhaseConfig(
            phase=LessonPhase.CONCEPT,
            required=True,
            order=1,
            min_duration_seconds=45,
            max_duration_seconds=90,
            slide_count=2,
            prompt_template="Explain the main concept using a simple analogy. Avoid jargon. Make it click.",
            tone="conversational",
            required_elements=["analogy", "simple_explanation"]
        ),
        PhaseConfig(
            phase=LessonPhase.THEORY,
            required=True,
            order=2,
            min_duration_seconds=60,
            max_duration_seconds=120,
            slide_count=2,
            prompt_template="Now explain the formal/technical details. Use proper terminology but remain clear.",
            tone="educational",
            required_elements=["definition", "key_terms"]
        ),
        PhaseConfig(
            phase=LessonPhase.VISUALIZATION,
            required=False,
            order=3,
            min_duration_seconds=30,
            max_duration_seconds=60,
            slide_count=1,
            prompt_template="Show a diagram, flowchart, or visual representation of the concept.",
            required_elements=["diagram"]
        ),
        PhaseConfig(
            phase=LessonPhase.CODE_DEMO,
            required=True,
            order=4,
            min_duration_seconds=90,
            max_duration_seconds=180,
            slide_count=3,
            prompt_template="Demonstrate with real code. Type it out, explain each line, show the output.",
            tone="practical",
            required_elements=["code_block", "explanation", "output"]
        ),
        PhaseConfig(
            phase=LessonPhase.RECAP,
            required=True,
            order=5,
            min_duration_seconds=30,
            max_duration_seconds=60,
            slide_count=1,
            prompt_template="Summarize the 3-5 key takeaways. What should the learner remember?",
            tone="clear",
            required_elements=["bullet_points", "key_takeaways"]
        ),
    ],
    allow_phase_reordering=False,
    allow_optional_phases=True
)

EDUCATION_LESSON_INTRO = LessonTemplate(
    name="Section Introduction Lesson",
    description="Lighter lesson to introduce a new section",
    total_duration_target_seconds=300,  # 5 minutes
    phases=[
        PhaseConfig(phase=LessonPhase.HOOK, required=True, order=0, min_duration_seconds=15, max_duration_seconds=30, slide_count=1),
        PhaseConfig(phase=LessonPhase.OBJECTIVES, required=True, order=1, min_duration_seconds=30, max_duration_seconds=60, slide_count=1),
        PhaseConfig(phase=LessonPhase.CONTEXT, required=True, order=2, min_duration_seconds=45, max_duration_seconds=90, slide_count=2),
        PhaseConfig(phase=LessonPhase.TEASER, required=True, order=3, min_duration_seconds=30, max_duration_seconds=60, slide_count=1),
    ]
)

EDUCATION_TEMPLATE = CurriculumTemplate(
    id="education_standard",
    name="Standard Education",
    context_type=ContextType.EDUCATION,
    description="Traditional learning flow optimized for knowledge retention: Hook → Concept → Theory → Demo → Recap",
    default_lesson_template=EDUCATION_LESSON_DEFAULT,
    lesson_templates={
        "intro": EDUCATION_LESSON_INTRO,
        "deep_dive": EDUCATION_LESSON_DEFAULT,
    },
    include_course_intro=True,
    include_course_conclusion=True,
    quiz_frequency="per_section"
)


# =============================================================================
# ENTERPRISE TEMPLATE
# Corporate training: Problem → Solution → ROI → Implementation → Action Items
# =============================================================================

ENTERPRISE_LESSON_DEFAULT = LessonTemplate(
    name="Enterprise Training Lesson",
    description="Business-focused flow emphasizing problem-solving and ROI",
    total_duration_target_seconds=480,  # 8 minutes
    phases=[
        PhaseConfig(
            phase=LessonPhase.CONTEXT,
            required=True,
            order=0,
            min_duration_seconds=30,
            max_duration_seconds=60,
            slide_count=1,
            prompt_template="Frame the business problem or challenge. Use data or statistics if available.",
            tone="professional",
            required_elements=["problem_statement", "impact"]
        ),
        PhaseConfig(
            phase=LessonPhase.CONCEPT,
            required=True,
            order=1,
            min_duration_seconds=60,
            max_duration_seconds=90,
            slide_count=2,
            prompt_template="Present the solution or approach. Focus on outcomes, not features.",
            tone="confident",
            required_elements=["solution_overview"]
        ),
        PhaseConfig(
            phase=LessonPhase.USE_CASE,
            required=True,
            order=2,
            min_duration_seconds=60,
            max_duration_seconds=120,
            slide_count=2,
            prompt_template="Show a real-world application. Preferably from a recognizable company.",
            required_elements=["example", "results"]
        ),
        PhaseConfig(
            phase=LessonPhase.ROI,
            required=True,
            order=3,
            min_duration_seconds=30,
            max_duration_seconds=60,
            slide_count=1,
            prompt_template="Quantify the business value. Time saved, cost reduced, revenue increased.",
            required_elements=["metrics", "comparison"]
        ),
        PhaseConfig(
            phase=LessonPhase.CODE_DEMO,
            required=False,
            order=4,
            min_duration_seconds=60,
            max_duration_seconds=120,
            slide_count=2,
            prompt_template="If applicable, show a technical implementation.",
            tone="practical"
        ),
        PhaseConfig(
            phase=LessonPhase.ACTION_ITEMS,
            required=True,
            order=5,
            min_duration_seconds=30,
            max_duration_seconds=60,
            slide_count=1,
            prompt_template="Clear next steps the learner can take immediately.",
            required_elements=["checklist", "quick_wins"]
        ),
    ],
    allow_phase_reordering=False,
    allow_optional_phases=True
)

ENTERPRISE_TEMPLATE = CurriculumTemplate(
    id="enterprise_standard",
    name="Enterprise Training",
    context_type=ContextType.ENTERPRISE,
    description="Business-focused flow: Problem → Solution → ROI → Implementation → Action Items",
    default_lesson_template=ENTERPRISE_LESSON_DEFAULT,
    lesson_templates={},
    include_course_intro=True,
    include_course_conclusion=True,
    quiz_frequency="end_only",
    brand_guidelines={
        "tone": "professional",
        "avoid": ["slang", "humor", "casual_language"],
        "include": ["data", "metrics", "case_studies"]
    }
)


# =============================================================================
# BOOTCAMP TEMPLATE
# Intensive learning: Quick Concept → Practice → Practice → Practice → Test
# =============================================================================

BOOTCAMP_LESSON_DEFAULT = LessonTemplate(
    name="Bootcamp Lesson",
    description="Intensive hands-on learning with minimal theory",
    total_duration_target_seconds=900,  # 15 minutes
    phases=[
        PhaseConfig(
            phase=LessonPhase.CONCEPT,
            required=True,
            order=0,
            min_duration_seconds=30,
            max_duration_seconds=60,
            slide_count=1,
            prompt_template="Quick concept explanation. Get to the code fast.",
            tone="direct"
        ),
        PhaseConfig(
            phase=LessonPhase.CODE_DEMO,
            required=True,
            order=1,
            min_duration_seconds=120,
            max_duration_seconds=180,
            slide_count=3,
            prompt_template="Show the implementation. Type it live.",
            required_elements=["code_block", "output"]
        ),
        PhaseConfig(
            phase=LessonPhase.EXERCISE,
            required=True,
            order=2,
            min_duration_seconds=120,
            max_duration_seconds=240,
            slide_count=2,
            prompt_template="First practice exercise. Start simple.",
            required_elements=["exercise_prompt", "starter_code"]
        ),
        PhaseConfig(
            phase=LessonPhase.EXERCISE,
            required=True,
            order=3,
            min_duration_seconds=120,
            max_duration_seconds=240,
            slide_count=2,
            prompt_template="Second exercise. Increase difficulty.",
            required_elements=["exercise_prompt"]
        ),
        PhaseConfig(
            phase=LessonPhase.CHALLENGE,
            required=True,
            order=4,
            min_duration_seconds=120,
            max_duration_seconds=300,
            slide_count=2,
            prompt_template="Challenge exercise. Apply everything learned.",
            required_elements=["challenge_prompt"]
        ),
        PhaseConfig(
            phase=LessonPhase.REVIEW,
            required=True,
            order=5,
            min_duration_seconds=30,
            max_duration_seconds=60,
            slide_count=1,
            prompt_template="Quick self-assessment. What did you learn?",
            required_elements=["reflection_questions"]
        ),
    ],
    allow_phase_reordering=False,
    strict_duration_enforcement=False
)

BOOTCAMP_TEMPLATE = CurriculumTemplate(
    id="bootcamp_intensive",
    name="Bootcamp Intensive",
    context_type=ContextType.BOOTCAMP,
    description="Intensive hands-on learning: Quick Concept → Practice → Practice → Challenge",
    default_lesson_template=BOOTCAMP_LESSON_DEFAULT,
    lesson_templates={},
    include_course_intro=True,
    include_course_conclusion=True,
    quiz_frequency="per_lesson"
)


# =============================================================================
# TUTORIAL TEMPLATE
# Quick how-to: Goal → Steps → Result
# =============================================================================

TUTORIAL_LESSON_DEFAULT = LessonTemplate(
    name="Quick Tutorial",
    description="Focused how-to with clear goal and steps",
    total_duration_target_seconds=300,  # 5 minutes
    phases=[
        PhaseConfig(
            phase=LessonPhase.OBJECTIVES,
            required=True,
            order=0,
            min_duration_seconds=15,
            max_duration_seconds=30,
            slide_count=1,
            prompt_template="What will you be able to do by the end?",
            tone="clear"
        ),
        PhaseConfig(
            phase=LessonPhase.PREREQUISITES,
            required=False,
            order=1,
            min_duration_seconds=15,
            max_duration_seconds=30,
            slide_count=1,
            prompt_template="What do you need before starting?"
        ),
        PhaseConfig(
            phase=LessonPhase.CODE_DEMO,
            required=True,
            order=2,
            min_duration_seconds=120,
            max_duration_seconds=180,
            slide_count=4,
            prompt_template="Step-by-step implementation. Number each step.",
            required_elements=["numbered_steps", "code_block"]
        ),
        PhaseConfig(
            phase=LessonPhase.RECAP,
            required=True,
            order=3,
            min_duration_seconds=15,
            max_duration_seconds=30,
            slide_count=1,
            prompt_template="Verify the result. Confirm it worked."
        ),
        PhaseConfig(
            phase=LessonPhase.NEXT_STEPS,
            required=False,
            order=4,
            min_duration_seconds=15,
            max_duration_seconds=30,
            slide_count=1,
            prompt_template="What can they do next? Related tutorials?"
        ),
    ]
)

TUTORIAL_TEMPLATE = CurriculumTemplate(
    id="tutorial_quick",
    name="Quick Tutorial",
    context_type=ContextType.TUTORIAL,
    description="Focused how-to: Goal → Steps → Result",
    default_lesson_template=TUTORIAL_LESSON_DEFAULT,
    lesson_templates={},
    include_course_intro=False,
    include_course_conclusion=False,
    quiz_frequency="end_only"
)


# =============================================================================
# WORKSHOP TEMPLATE
# Hands-on collaborative: Intro → Exercise → Debrief → Exercise → Conclusion
# =============================================================================

WORKSHOP_LESSON_DEFAULT = LessonTemplate(
    name="Workshop Session",
    description="Hands-on collaborative learning with exercises and discussion",
    total_duration_target_seconds=1200,  # 20 minutes
    phases=[
        PhaseConfig(
            phase=LessonPhase.CONTEXT,
            required=True,
            order=0,
            min_duration_seconds=60,
            max_duration_seconds=120,
            slide_count=2,
            prompt_template="Set the stage. What will we build today?"
        ),
        PhaseConfig(
            phase=LessonPhase.EXAMPLE,
            required=True,
            order=1,
            min_duration_seconds=120,
            max_duration_seconds=180,
            slide_count=3,
            prompt_template="Walk through a complete example together."
        ),
        PhaseConfig(
            phase=LessonPhase.EXERCISE,
            required=True,
            order=2,
            min_duration_seconds=180,
            max_duration_seconds=300,
            slide_count=2,
            prompt_template="First exercise. Work independently or in pairs."
        ),
        PhaseConfig(
            phase=LessonPhase.REVIEW,
            required=True,
            order=3,
            min_duration_seconds=60,
            max_duration_seconds=120,
            slide_count=1,
            prompt_template="Debrief. Share solutions, discuss challenges."
        ),
        PhaseConfig(
            phase=LessonPhase.EXERCISE,
            required=True,
            order=4,
            min_duration_seconds=180,
            max_duration_seconds=300,
            slide_count=2,
            prompt_template="Second exercise. Build on the first one."
        ),
        PhaseConfig(
            phase=LessonPhase.RECAP,
            required=True,
            order=5,
            min_duration_seconds=60,
            max_duration_seconds=120,
            slide_count=1,
            prompt_template="Wrap up. Key learnings and takeaways."
        ),
    ]
)

WORKSHOP_TEMPLATE = CurriculumTemplate(
    id="workshop_collaborative",
    name="Workshop Collaborative",
    context_type=ContextType.WORKSHOP,
    description="Hands-on collaborative: Intro → Exercise → Debrief → Exercise → Conclusion",
    default_lesson_template=WORKSHOP_LESSON_DEFAULT,
    lesson_templates={},
    include_course_intro=True,
    include_course_conclusion=True,
    quiz_frequency="end_only"
)


# =============================================================================
# CERTIFICATION TEMPLATE
# Exam prep: Theory → Examples → Practice → Quiz
# =============================================================================

CERTIFICATION_LESSON_DEFAULT = LessonTemplate(
    name="Certification Prep Lesson",
    description="Comprehensive coverage for exam preparation",
    total_duration_target_seconds=720,  # 12 minutes
    phases=[
        PhaseConfig(
            phase=LessonPhase.OBJECTIVES,
            required=True,
            order=0,
            min_duration_seconds=30,
            max_duration_seconds=45,
            slide_count=1,
            prompt_template="Exam objectives covered in this lesson.",
            required_elements=["objective_list"]
        ),
        PhaseConfig(
            phase=LessonPhase.THEORY,
            required=True,
            order=1,
            min_duration_seconds=120,
            max_duration_seconds=180,
            slide_count=3,
            prompt_template="Comprehensive theory explanation. Cover all exam points.",
            required_elements=["definition", "key_concepts", "important_notes"]
        ),
        PhaseConfig(
            phase=LessonPhase.EXAMPLE,
            required=True,
            order=2,
            min_duration_seconds=90,
            max_duration_seconds=150,
            slide_count=2,
            prompt_template="Worked examples similar to exam questions."
        ),
        PhaseConfig(
            phase=LessonPhase.EXERCISE,
            required=True,
            order=3,
            min_duration_seconds=120,
            max_duration_seconds=180,
            slide_count=2,
            prompt_template="Practice questions in exam format."
        ),
        PhaseConfig(
            phase=LessonPhase.QUIZ,
            required=True,
            order=4,
            min_duration_seconds=90,
            max_duration_seconds=150,
            slide_count=3,
            prompt_template="Self-assessment quiz matching exam style.",
            required_elements=["multiple_choice", "explanations"]
        ),
        PhaseConfig(
            phase=LessonPhase.RECAP,
            required=True,
            order=5,
            min_duration_seconds=30,
            max_duration_seconds=60,
            slide_count=1,
            prompt_template="Key points to remember for the exam."
        ),
    ]
)

CERTIFICATION_TEMPLATE = CurriculumTemplate(
    id="certification_prep",
    name="Certification Prep",
    context_type=ContextType.CERTIFICATION,
    description="Exam preparation: Theory → Examples → Practice → Quiz",
    default_lesson_template=CERTIFICATION_LESSON_DEFAULT,
    lesson_templates={},
    include_course_intro=True,
    include_course_conclusion=True,
    quiz_frequency="per_lesson"
)


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

class TemplateRegistry:
    """Registry for managing curriculum templates."""

    _templates: Dict[str, CurriculumTemplate] = {
        "education_standard": EDUCATION_TEMPLATE,
        "enterprise_standard": ENTERPRISE_TEMPLATE,
        "bootcamp_intensive": BOOTCAMP_TEMPLATE,
        "tutorial_quick": TUTORIAL_TEMPLATE,
        "workshop_collaborative": WORKSHOP_TEMPLATE,
        "certification_prep": CERTIFICATION_TEMPLATE,
    }

    _context_defaults: Dict[ContextType, str] = {
        ContextType.EDUCATION: "education_standard",
        ContextType.ENTERPRISE: "enterprise_standard",
        ContextType.BOOTCAMP: "bootcamp_intensive",
        ContextType.TUTORIAL: "tutorial_quick",
        ContextType.WORKSHOP: "workshop_collaborative",
        ContextType.CERTIFICATION: "certification_prep",
    }

    @classmethod
    def get(cls, template_id: str) -> Optional[CurriculumTemplate]:
        """Get a template by ID."""
        return cls._templates.get(template_id)

    @classmethod
    def get_by_context(cls, context: ContextType) -> Optional[CurriculumTemplate]:
        """Get the default template for a context type."""
        template_id = cls._context_defaults.get(context)
        if template_id:
            return cls._templates.get(template_id)
        return None

    @classmethod
    def register(cls, template: CurriculumTemplate) -> None:
        """Register a custom template."""
        cls._templates[template.id] = template

    @classmethod
    def list_all(cls) -> List[CurriculumTemplate]:
        """List all available templates."""
        return list(cls._templates.values())

    @classmethod
    def list_ids(cls) -> List[str]:
        """List all template IDs."""
        return list(cls._templates.keys())


# Convenience functions
def get_template_by_context(context: ContextType) -> Optional[CurriculumTemplate]:
    """Get the default template for a context type."""
    return TemplateRegistry.get_by_context(context)


def get_all_templates() -> List[CurriculumTemplate]:
    """Get all available templates."""
    return TemplateRegistry.list_all()
