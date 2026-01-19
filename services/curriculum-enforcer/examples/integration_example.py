"""
Integration Example: Curriculum Enforcer with Course Generator

This example shows how to integrate the Curriculum Enforcer module
with the existing Course Generator service.
"""

import asyncio
from typing import Dict, Any, List

# Import from Curriculum Enforcer module
import sys
sys.path.insert(0, '..')

from models.curriculum_models import (
    ContextType,
    LessonPhase,
    LessonContent,
    EnforcementRequest,
)
from services.curriculum_enforcer_service import CurriculumEnforcerService
from templates.predefined_templates import (
    EDUCATION_TEMPLATE,
    ENTERPRISE_TEMPLATE,
    get_template_by_context,
)


async def enforce_lesson_structure(
    lesson_data: Dict[str, Any],
    context_type: ContextType = ContextType.EDUCATION
) -> Dict[str, Any]:
    """
    Enforce curriculum structure on a lesson.

    This function can be called from the Course Generator
    before sending to Presentation Generator.
    """
    enforcer = CurriculumEnforcerService()

    # Convert lesson data to LessonContent
    content = LessonContent(
        lesson_id=lesson_data.get("id", "unknown"),
        title=lesson_data.get("title", "Untitled"),
        slides=lesson_data.get("slides", []),
        lesson_type=lesson_data.get("type"),
        section_position=lesson_data.get("section_position", 0),
        total_lessons_in_section=lesson_data.get("total_lessons", 1)
    )

    # Create enforcement request
    request = EnforcementRequest(
        content=content,
        context_type=context_type,
        auto_fix=True,
        preserve_content=True
    )

    # Enforce structure
    result = await enforcer.enforce(request)

    if result.success:
        print(f"[CurriculumEnforcer] Lesson '{content.title}' validated successfully")
        print(f"  Score: {result.original_validation.score:.2%}")
    else:
        print(f"[CurriculumEnforcer] Lesson '{content.title}' restructured")
        print(f"  Original score: {result.original_validation.score:.2%}")
        print(f"  Final score: {result.final_validation.score:.2%}" if result.final_validation else "")
        print(f"  Changes: {result.changes_made}")

    # Return restructured lesson
    if result.restructured_content:
        return {
            "id": result.restructured_content.lesson_id,
            "title": result.restructured_content.title,
            "slides": result.restructured_content.slides,
            "validated": True,
            "validation_score": result.final_validation.score if result.final_validation else result.original_validation.score
        }

    return lesson_data


async def validate_course_structure(
    course_outline: Dict[str, Any],
    context_type: ContextType = ContextType.EDUCATION
) -> Dict[str, Any]:
    """
    Validate entire course structure before generation.
    """
    enforcer = CurriculumEnforcerService()

    print(f"\n=== Validating Course: {course_outline.get('title', 'Untitled')} ===")
    print(f"Context: {context_type.value}")
    print(f"Template: {get_template_by_context(context_type).name}")

    validated_sections = []

    for section in course_outline.get("sections", []):
        validated_lessons = []

        for lesson in section.get("lectures", []):
            validated = await enforce_lesson_structure(lesson, context_type)
            validated_lessons.append(validated)

        validated_sections.append({
            **section,
            "lectures": validated_lessons
        })

    return {
        **course_outline,
        "sections": validated_sections,
        "curriculum_enforced": True,
        "context_type": context_type.value
    }


def show_available_templates():
    """Display available curriculum templates."""
    enforcer = CurriculumEnforcerService()

    print("\n=== Available Curriculum Templates ===\n")

    for template in enforcer.list_templates():
        print(f"ID: {template['id']}")
        print(f"  Name: {template['name']}")
        print(f"  Context: {template['context_type']}")
        print(f"  Required Phases: {' -> '.join(template['phases'])}")
        print()


def show_template_phases(context_type: ContextType):
    """Display phases for a specific context type."""
    enforcer = CurriculumEnforcerService()

    print(f"\n=== Phases for {context_type.value} ===\n")

    for phase in enforcer.get_template_phases(context_type):
        required = "REQUIRED" if phase['required'] else "optional"
        print(f"{phase['order']}. {phase['phase']} [{required}]")
        print(f"   Duration: {phase['duration_range']}")
        if phase['prompt']:
            print(f"   Prompt: {phase['prompt'][:60]}...")
        print()


# Example usage
async def main():
    # Show available templates
    show_available_templates()

    # Show education template phases
    show_template_phases(ContextType.EDUCATION)

    # Sample course outline
    sample_course = {
        "title": "Python Decorators Masterclass",
        "sections": [
            {
                "title": "Introduction to Decorators",
                "lectures": [
                    {
                        "id": "lesson-1",
                        "title": "What are Decorators?",
                        "slides": [
                            # Missing HOOK - should be auto-generated
                            {
                                "type": "concept",
                                "title": "Understanding Decorators",
                                "content": "A decorator is a function that wraps another function",
                                "voiceover": "Decorators are a powerful Python feature...",
                                "duration": 60
                            },
                            {
                                "type": "code",
                                "title": "Basic Decorator Syntax",
                                "content": "@my_decorator\ndef my_function():\n    pass",
                                "voiceover": "Here's how you use a decorator...",
                                "duration": 90
                            },
                            # Missing RECAP - should be auto-generated
                        ]
                    }
                ]
            }
        ]
    }

    # Validate with EDUCATION context
    print("\n" + "="*60)
    print("VALIDATING WITH EDUCATION CONTEXT")
    print("="*60)
    validated_edu = await validate_course_structure(
        sample_course,
        ContextType.EDUCATION
    )

    # Validate with ENTERPRISE context
    print("\n" + "="*60)
    print("VALIDATING WITH ENTERPRISE CONTEXT")
    print("="*60)
    validated_ent = await validate_course_structure(
        sample_course,
        ContextType.ENTERPRISE
    )

    print("\n=== Validation Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
