"""
Pedagogical Agent Nodes

Implementation of all nodes in the LangGraph workflow.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List

from openai import AsyncOpenAI

from agents.pedagogical_state import (
    PedagogicalAgentState,
    ContentPreferences,
    QuizPlacement,
    ValidationResult,
)
from agents.pedagogical_prompts import (
    CONTEXT_ANALYSIS_PROMPT,
    PROFILE_ADAPTATION_PROMPT,
    ELEMENT_SUGGESTION_PROMPT,
    QUIZ_PLANNING_PROMPT,
    LANGUAGE_VALIDATION_PROMPT,
    STRUCTURE_VALIDATION_PROMPT,
    FINALIZATION_PROMPT,
)
from models.course_models import ProfileCategory
from models.lesson_elements import CATEGORY_ELEMENTS, get_elements_for_category


# Language name mapping
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French (Français)",
    "es": "Spanish (Español)",
    "de": "German (Deutsch)",
    "pt": "Portuguese (Português)",
    "it": "Italian (Italiano)",
    "nl": "Dutch (Nederlands)",
    "pl": "Polish (Polski)",
    "ru": "Russian (Русский)",
    "zh": "Chinese (中文)",
}


async def get_openai_client() -> AsyncOpenAI:
    """Get OpenAI client instance"""
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=60.0,
        max_retries=2
    )


async def analyze_context(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Analyze the course topic and context.

    Determines learner persona, topic complexity, and content requirements.
    """
    print(f"[AGENT] Analyzing context for: {state['topic']}", flush=True)
    state["current_node"] = "analyze_context"

    client = await get_openai_client()

    description_section = f"DESCRIPTION: {state.get('description')}" if state.get('description') else ""

    prompt = CONTEXT_ANALYSIS_PROMPT.format(
        topic=state["topic"],
        description_section=description_section,
        category=state["profile_category"].value if state.get("profile_category") else "education",
        target_audience=state.get("target_audience", "general learners"),
        difficulty_start=state.get("difficulty_start", "beginner"),
        difficulty_end=state.get("difficulty_end", "intermediate"),
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500
        )

        result = json.loads(response.choices[0].message.content)

        return {
            "detected_persona": result.get("detected_persona", "student"),
            "topic_complexity": result.get("topic_complexity", "intermediate"),
            "requires_code": result.get("requires_code", False),
            "requires_diagrams": result.get("requires_diagrams", True),
            "requires_hands_on": result.get("requires_hands_on", False),
            "domain_keywords": result.get("domain_keywords", []),
        }

    except Exception as e:
        print(f"[AGENT] Context analysis error: {e}", flush=True)
        return {
            "detected_persona": "student",
            "topic_complexity": "intermediate",
            "requires_code": False,
            "requires_diagrams": True,
            "requires_hands_on": False,
            "domain_keywords": [],
            "errors": state.get("errors", []) + [f"Context analysis failed: {str(e)}"],
        }


async def fetch_rag_images(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Extract diagrams and images from RAG documents.

    If RAG context is available, identifies any diagrams, schemas, or
    images that can be reused in the course.
    """
    print(f"[AGENT] Fetching RAG images", flush=True)
    state["current_node"] = "fetch_rag_images"

    # Currently a placeholder - would integrate with document parser
    # to extract image references from uploaded documents

    if not state.get("rag_context") and not state.get("document_ids"):
        return {
            "rag_images": [],
            "rag_diagrams_available": False,
        }

    # TODO: Implement actual RAG image extraction
    # This would scan uploaded documents for embedded images/diagrams
    # and create references for use in course generation

    return {
        "rag_images": [],
        "rag_diagrams_available": False,
    }


async def adapt_for_profile(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Adapt content mix based on profile analysis.

    Sets content weights (code vs diagrams vs theory) based on
    the detected learner persona and topic requirements.
    """
    print(f"[AGENT] Adapting content for profile: {state.get('detected_persona')}", flush=True)
    state["current_node"] = "adapt_for_profile"

    client = await get_openai_client()

    # Get available elements for the category
    category = state.get("profile_category", ProfileCategory.EDUCATION)
    available_elements = get_elements_for_category(category)
    elements_list = "\n".join([f"- {el.id.value}: {el.name} - {el.description}" for el in available_elements])

    prompt = PROFILE_ADAPTATION_PROMPT.format(
        detected_persona=state.get("detected_persona", "student"),
        topic_complexity=state.get("topic_complexity", "intermediate"),
        category=category.value,
        requires_code=state.get("requires_code", False),
        requires_diagrams=state.get("requires_diagrams", True),
        requires_hands_on=state.get("requires_hands_on", False),
        available_elements=elements_list,
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=600
        )

        result = json.loads(response.choices[0].message.content)

        prefs = result.get("content_preferences", {})
        content_preferences: ContentPreferences = {
            "code_weight": prefs.get("code_weight", 0.5),
            "diagram_weight": prefs.get("diagram_weight", 0.5),
            "demo_weight": prefs.get("demo_weight", 0.5),
            "theory_weight": prefs.get("theory_weight", 0.5),
            "case_study_weight": prefs.get("case_study_weight", 0.3),
        }

        return {
            "content_preferences": content_preferences,
            "recommended_elements": result.get("recommended_elements", []),
        }

    except Exception as e:
        print(f"[AGENT] Profile adaptation error: {e}", flush=True)
        # Return defaults based on category
        return {
            "content_preferences": {
                "code_weight": 0.5,
                "diagram_weight": 0.5,
                "demo_weight": 0.5,
                "theory_weight": 0.5,
                "case_study_weight": 0.3,
            },
            "recommended_elements": [],
            "errors": state.get("errors", []) + [f"Profile adaptation failed: {str(e)}"],
        }


async def suggest_elements(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Suggest lesson elements for each lecture.

    Maps specific elements to each lecture based on content and preferences.
    """
    print(f"[AGENT] Suggesting elements for lectures", flush=True)
    state["current_node"] = "suggest_elements"

    outline = state.get("outline")
    if not outline:
        return {
            "element_mapping": {},
            "errors": state.get("errors", []) + ["No outline available for element suggestion"],
        }

    client = await get_openai_client()
    category = state.get("profile_category", ProfileCategory.EDUCATION)
    available_elements = get_elements_for_category(category)
    elements_list = "\n".join([f"- {el.id.value}: {el.name}" for el in available_elements])

    # Build outline structure string
    outline_lines = []
    for section in outline.sections:
        outline_lines.append(f"Section: {section.title}")
        for lecture in section.lectures:
            outline_lines.append(f"  - Lecture {lecture.id}: {lecture.title}")
            if lecture.objectives:
                outline_lines.append(f"    Objectives: {', '.join(lecture.objectives[:2])}")

    outline_structure = "\n".join(outline_lines)

    prefs = state.get("content_preferences", {})
    prompt = ELEMENT_SUGGESTION_PROMPT.format(
        topic=state["topic"],
        category=category.value,
        code_weight=prefs.get("code_weight", 0.5),
        diagram_weight=prefs.get("diagram_weight", 0.5),
        demo_weight=prefs.get("demo_weight", 0.5),
        available_elements=elements_list,
        outline_structure=outline_structure,
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1500
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "element_mapping": result.get("element_mapping", {}),
        }

    except Exception as e:
        print(f"[AGENT] Element suggestion error: {e}", flush=True)
        return {
            "element_mapping": {},
            "errors": state.get("errors", []) + [f"Element suggestion failed: {str(e)}"],
        }


async def plan_quizzes(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Plan quiz placement and configuration.

    Determines where quizzes should be placed and what they should cover.
    """
    print(f"[AGENT] Planning quiz placement", flush=True)
    state["current_node"] = "plan_quizzes"

    if not state.get("quiz_enabled", True):
        return {
            "quiz_placement": [],
            "quiz_total_count": 0,
        }

    outline = state.get("outline")
    if not outline:
        return {
            "quiz_placement": [],
            "quiz_total_count": 0,
        }

    client = await get_openai_client()

    # Build outline structure
    outline_lines = []
    section_objectives = []
    for section in outline.sections:
        outline_lines.append(f"Section: {section.title}")
        sec_objs = []
        for lecture in section.lectures:
            outline_lines.append(f"  - {lecture.id}: {lecture.title}")
            sec_objs.extend(lecture.objectives[:2] if lecture.objectives else [])
        section_objectives.append(f"{section.title}: {', '.join(sec_objs)}")

    prompt = QUIZ_PLANNING_PROMPT.format(
        quiz_enabled=state.get("quiz_enabled", True),
        quiz_frequency=state.get("quiz_frequency", "per_section"),
        questions_per_quiz=5,
        outline_structure="\n".join(outline_lines),
        section_objectives="\n".join(section_objectives),
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000
        )

        result = json.loads(response.choices[0].message.content)
        placements = result.get("quiz_placement", [])

        quiz_placement: List[QuizPlacement] = [
            {
                "lecture_id": p.get("lecture_id", ""),
                "quiz_type": p.get("quiz_type", "section_review"),
                "difficulty": p.get("difficulty", "medium"),
                "question_count": p.get("question_count", 5),
                "topics_covered": p.get("topics_covered", []),
            }
            for p in placements
        ]

        return {
            "quiz_placement": quiz_placement,
            "quiz_total_count": result.get("total_quiz_count", len(quiz_placement)),
        }

    except Exception as e:
        print(f"[AGENT] Quiz planning error: {e}", flush=True)
        return {
            "quiz_placement": [],
            "quiz_total_count": 0,
            "errors": state.get("errors", []) + [f"Quiz planning failed: {str(e)}"],
        }


async def validate_language(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Validate language compliance of the outline.

    Ensures all content is in the target language.
    """
    print(f"[AGENT] Validating language: {state.get('target_language')}", flush=True)
    state["current_node"] = "validate_language"

    outline = state.get("outline")
    if not outline:
        return {"language_validated": False}

    target_lang = state.get("target_language", "en")
    language_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    # For English, skip validation (default)
    if target_lang == "en":
        return {"language_validated": True}

    client = await get_openai_client()

    # Extract content to validate
    content_items = [
        f"Title: {outline.title}",
        f"Description: {outline.description}",
    ]
    for section in outline.sections:
        content_items.append(f"Section: {section.title}")
        for lecture in section.lectures:
            content_items.append(f"  Lecture: {lecture.title}")
            if lecture.description:
                content_items.append(f"    Description: {lecture.description}")

    prompt = LANGUAGE_VALIDATION_PROMPT.format(
        target_language=target_lang,
        language_name=language_name,
        outline_content="\n".join(content_items[:50]),  # Limit for token size
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "language_validated": result.get("is_valid", True),
        }

    except Exception as e:
        print(f"[AGENT] Language validation error: {e}", flush=True)
        return {
            "language_validated": True,  # Allow to proceed
            "errors": state.get("errors", []) + [f"Language validation failed: {str(e)}"],
        }


async def validate_structure(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Validate pedagogical structure quality.

    Evaluates the course structure for pedagogical soundness.
    """
    print(f"[AGENT] Validating structure", flush=True)
    state["current_node"] = "validate_structure"

    outline = state.get("outline")
    if not outline:
        return {
            "structure_validated": False,
            "validation_result": {
                "is_valid": False,
                "warnings": ["No outline to validate"],
                "suggestions": [],
                "pedagogical_score": 0,
            },
        }

    client = await get_openai_client()

    # Build structure summary
    structure_lines = [
        f"Course: {outline.title}",
        f"Total sections: {outline.section_count}",
        f"Total lectures: {outline.total_lectures}",
    ]
    for section in outline.sections:
        structure_lines.append(f"\nSection {section.order + 1}: {section.title}")
        for lecture in section.lectures:
            structure_lines.append(
                f"  {lecture.order + 1}. {lecture.title} ({lecture.difficulty.value})"
            )

    prompt = STRUCTURE_VALIDATION_PROMPT.format(
        outline_structure="\n".join(structure_lines),
        difficulty_start=state.get("difficulty_start", "beginner"),
        difficulty_end=state.get("difficulty_end", "intermediate"),
        total_duration=state.get("total_duration_minutes", 60),
        target_audience=state.get("target_audience", "general learners"),
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=800
        )

        result = json.loads(response.choices[0].message.content)

        validation_result: ValidationResult = {
            "is_valid": result.get("is_valid", True),
            "warnings": result.get("warnings", []),
            "suggestions": result.get("suggestions", []),
            "pedagogical_score": result.get("pedagogical_score", 75),
        }

        return {
            "structure_validated": validation_result["is_valid"],
            "validation_result": validation_result,
        }

    except Exception as e:
        print(f"[AGENT] Structure validation error: {e}", flush=True)
        return {
            "structure_validated": True,  # Allow to proceed
            "validation_result": {
                "is_valid": True,
                "warnings": [],
                "suggestions": [],
                "pedagogical_score": 70,
            },
            "errors": state.get("errors", []) + [f"Structure validation failed: {str(e)}"],
        }


async def finalize_plan(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Finalize the pedagogical plan.

    Applies all enhancements to the outline and generates metadata.
    """
    print(f"[AGENT] Finalizing plan", flush=True)
    state["current_node"] = "finalize_plan"

    outline = state.get("outline")
    if not outline:
        return {
            "final_outline": None,
            "generation_metadata": {"error": "No outline to finalize"},
        }

    # Apply element mapping to lectures
    element_mapping = state.get("element_mapping", {})
    content_preferences = state.get("content_preferences", {})

    for section in outline.sections:
        for lecture in section.lectures:
            # Assign elements if available in mapping
            if lecture.id in element_mapping:
                lecture.lesson_elements = element_mapping[lecture.id]
            elif state.get("recommended_elements"):
                lecture.lesson_elements = state["recommended_elements"][:5]

            # Assign content weights
            lecture.element_weights = {
                "code": content_preferences.get("code_weight", 0.5),
                "diagram": content_preferences.get("diagram_weight", 0.5),
                "demo": content_preferences.get("demo_weight", 0.5),
                "theory": content_preferences.get("theory_weight", 0.5),
            }

    # Generate metadata
    validation_result = state.get("validation_result", {})
    metadata = {
        "total_lectures": outline.total_lectures,
        "total_quizzes": state.get("quiz_total_count", 0),
        "estimated_duration_minutes": state.get("total_duration_minutes", 60),
        "pedagogical_score": validation_result.get("pedagogical_score", 75),
        "detected_persona": state.get("detected_persona", "student"),
        "topic_complexity": state.get("topic_complexity", "intermediate"),
        "content_mix": {
            "code_weight": content_preferences.get("code_weight", 0.5),
            "diagram_weight": content_preferences.get("diagram_weight", 0.5),
            "theory_weight": content_preferences.get("theory_weight", 0.5),
        },
        "language_validated": state.get("language_validated", True),
        "structure_validated": state.get("structure_validated", True),
        "generation_timestamp": datetime.utcnow().isoformat(),
        "agent_version": "1.0.0",
    }

    return {
        "final_outline": outline,
        "generation_metadata": metadata,
    }
