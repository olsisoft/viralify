"""
Curriculum Enforcer Module
Enforces pedagogical structure on lesson content.

Usage:
    from curriculum_enforcer import CurriculumEnforcerService, ContextType, LessonContent

    enforcer = CurriculumEnforcerService()

    # Validate lesson content
    validation = await enforcer.validate_only(
        content=lesson_content,
        context_type=ContextType.EDUCATION
    )

    # Enforce structure (with auto-fix)
    result = await enforcer.enforce(
        EnforcementRequest(
            content=lesson_content,
            context_type=ContextType.ENTERPRISE,
            auto_fix=True
        )
    )
"""

from .models import (
    ContextType,
    LessonPhase,
    PhaseConfig,
    LessonTemplate,
    CurriculumTemplate,
    LessonContent,
    ValidationResult,
    PhaseViolation,
    EnforcementRequest,
    EnforcementResult,
)

from .services import (
    LessonStructureValidator,
    CurriculumEnforcerService,
)

from .templates import (
    EDUCATION_TEMPLATE,
    ENTERPRISE_TEMPLATE,
    BOOTCAMP_TEMPLATE,
    TUTORIAL_TEMPLATE,
    WORKSHOP_TEMPLATE,
    CERTIFICATION_TEMPLATE,
    get_template_by_context,
    get_all_templates,
    TemplateRegistry,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "ContextType",
    "LessonPhase",
    "PhaseConfig",
    "LessonTemplate",
    "CurriculumTemplate",
    "LessonContent",
    "ValidationResult",
    "PhaseViolation",
    "EnforcementRequest",
    "EnforcementResult",
    # Services
    "LessonStructureValidator",
    "CurriculumEnforcerService",
    # Templates
    "EDUCATION_TEMPLATE",
    "ENTERPRISE_TEMPLATE",
    "BOOTCAMP_TEMPLATE",
    "TUTORIAL_TEMPLATE",
    "WORKSHOP_TEMPLATE",
    "CERTIFICATION_TEMPLATE",
    "get_template_by_context",
    "get_all_templates",
    "TemplateRegistry",
]
