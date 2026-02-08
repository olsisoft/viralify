"""
Pedagogical Agent Nodes

Implementation of all nodes in the LangGraph workflow.

Supports multiple providers via shared.llm_provider:
- OpenAI, DeepSeek, Groq, Mistral, Together AI, xAI Grok
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import (
        get_llm_client,
        get_model_name,
    )
    USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    USE_SHARED_LLM = False
    print("[AGENT] Warning: shared.llm_provider not found, using direct OpenAI", flush=True)

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
    OUTLINE_REFINEMENT_PROMPT,
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


def get_client_and_model():
    """Get LLM client and model name (uses shared provider if available)"""
    if USE_SHARED_LLM:
        client = get_llm_client()
        model = get_model_name("fast")  # Use fast model for agent nodes
        return client, model
    else:
        # Fallback to direct OpenAI
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60.0,
            max_retries=2
        )
        return client, "gpt-4o-mini"


async def analyze_context(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Analyze the course topic and context.

    Determines learner persona, topic complexity, and content requirements.
    """
    print(f"[AGENT] Analyzing context for: {state['topic']}", flush=True)
    state["current_node"] = "analyze_context"

    client, model = get_client_and_model()

    description_section = f"DESCRIPTION: {state.get('description')}" if state.get('description') else ""

    prompt = CONTEXT_ANALYSIS_PROMPT.format(
        topic=state["topic"],
        description_section=description_section,
        category=(state["profile_category"].value if hasattr(state.get("profile_category"), 'value') else state.get("profile_category")) or "education",
        target_audience=state.get("target_audience", "general learners"),
        difficulty_start=state.get("difficulty_start", "beginner"),
        difficulty_end=state.get("difficulty_end", "intermediate"),
    )

    try:
        response = await client.chat.completions.create(
            model=model,
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
    Node: Extract real images from RAG documents.

    Fetches actual images extracted from PDFs and other documents
    using the retrieval service. Returns file paths to real images
    that can be used in slides.

    The function:
    1. Gets the topic and outline from state
    2. Calls retrieval_service.get_images_for_topic() for each lecture
    3. Returns real image paths with relevance scores
    """
    print(f"[AGENT] Fetching RAG images from documents", flush=True)
    state["current_node"] = "fetch_rag_images"

    document_ids = state.get("document_ids", [])
    user_id = state.get("user_id", "")
    topic = state.get("topic", "")
    outline = state.get("outline", {})

    if not document_ids:
        print(f"[AGENT] No document IDs - skipping image extraction", flush=True)
        return {
            "rag_images": [],
            "rag_diagrams_available": False,
        }

    if not user_id:
        print(f"[AGENT] No user ID - skipping image extraction", flush=True)
        return {
            "rag_images": [],
            "rag_diagrams_available": False,
        }

    try:
        # Import retrieval service
        from services.retrieval_service import get_retrieval_service
        retrieval_service = get_retrieval_service()

        rag_images = []
        image_types_for_diagrams = ["diagram", "chart", "architecture", "flowchart", "schema"]

        # Extract lectures from outline
        lectures = []
        sections = outline.get("sections", [])
        for section in sections:
            for lecture in section.get("lectures", []):
                lectures.append(lecture)

        # If no outline yet, use the main topic
        if not lectures:
            lectures = [{"id": "main", "title": topic}]

        print(f"[AGENT] Searching images for {len(lectures)} lectures in {len(document_ids)} documents", flush=True)

        # Fetch images for each lecture topic
        for lecture in lectures:
            lecture_id = lecture.get("id", "unknown")
            lecture_title = lecture.get("title", topic)

            # Search for images relevant to this lecture
            images = await retrieval_service.get_images_for_topic(
                topic=lecture_title,
                document_ids=document_ids,
                user_id=user_id,
                image_types=image_types_for_diagrams,
                max_images=3,
                min_relevance=0.3,
            )

            for img in images:
                rag_images.append({
                    "lecture_id": lecture_id,
                    "lecture_title": lecture_title,
                    "image_id": img.get("image_id"),
                    "document_id": img.get("document_id"),
                    "file_path": img.get("file_path"),
                    "file_name": img.get("file_name"),
                    "detected_type": img.get("detected_type", "diagram"),
                    "context_text": img.get("context_text", ""),
                    "caption": img.get("caption", ""),
                    "description": img.get("description", ""),
                    "page_number": img.get("page_number"),
                    "document_name": img.get("document_name", ""),
                    "relevance_score": img.get("relevance_score", 0.0),
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                })

        # Determine if we have usable diagrams
        has_diagrams = any(
            img.get("detected_type") in image_types_for_diagrams and
            img.get("relevance_score", 0) >= 0.5
            for img in rag_images
        )

        # Create summary
        if rag_images:
            best_score = max(img.get("relevance_score", 0) for img in rag_images)
            types_found = set(img.get("detected_type") for img in rag_images)
            summary = f"Found {len(rag_images)} images (types: {', '.join(types_found)}, best score: {best_score:.2f})"
        else:
            summary = "No relevant images found in documents"

        print(f"[AGENT] {summary}", flush=True)

        return {
            "rag_images": rag_images,
            "rag_diagrams_available": has_diagrams,
            "rag_visual_summary": summary,
            "rag_has_architecture": any(img.get("detected_type") == "architecture" for img in rag_images),
            "rag_has_process_flows": any(img.get("detected_type") == "flowchart" for img in rag_images),
        }

    except Exception as e:
        print(f"[AGENT] RAG image extraction error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            "rag_images": [],
            "rag_diagrams_available": False,
            "errors": state.get("errors", []) + [f"RAG image extraction failed: {str(e)}"],
        }


async def adapt_for_profile(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Adapt content mix based on profile analysis.

    Sets content weights (code vs diagrams vs theory) based on
    the detected learner persona and topic requirements.
    """
    print(f"[AGENT] Adapting content for profile: {state.get('detected_persona')}", flush=True)
    state["current_node"] = "adapt_for_profile"

    client, model = get_client_and_model()

    # Get available elements for the category
    category_raw = state.get("profile_category", "education")
    # Convert string to enum if needed
    if isinstance(category_raw, str):
        try:
            category = ProfileCategory(category_raw.lower())
        except ValueError:
            category = ProfileCategory.EDUCATION
    else:
        category = category_raw if category_raw else ProfileCategory.EDUCATION
    available_elements = get_elements_for_category(category)
    elements_list = "\n".join([f"- {el.id.value}: {el.name} - {el.description}" for el in available_elements])

    prompt = PROFILE_ADAPTATION_PROMPT.format(
        detected_persona=state.get("detected_persona", "student"),
        topic_complexity=state.get("topic_complexity", "intermediate"),
        category=category.value if hasattr(category, 'value') else category,
        requires_code=state.get("requires_code", False),
        requires_diagrams=state.get("requires_diagrams", True),
        requires_hands_on=state.get("requires_hands_on", False),
        available_elements=elements_list,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
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

    client, model = get_client_and_model()
    category_raw = state.get("profile_category", "education")
    # Convert string to enum if needed
    if isinstance(category_raw, str):
        try:
            category = ProfileCategory(category_raw.lower())
        except ValueError:
            category = ProfileCategory.EDUCATION
    else:
        category = category_raw if category_raw else ProfileCategory.EDUCATION
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
        category=category.value if hasattr(category, 'value') else category,
        code_weight=prefs.get("code_weight", 0.5),
        diagram_weight=prefs.get("diagram_weight", 0.5),
        demo_weight=prefs.get("demo_weight", 0.5),
        available_elements=elements_list,
        outline_structure=outline_structure,
    )

    try:
        response = await client.chat.completions.create(
            model=model,
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

    client, model = get_client_and_model()

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
            model=model,
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

    client, model = get_client_and_model()

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
            model=model,
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

    client, model = get_client_and_model()

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
            model=model,
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


# =============================================================================
# FEEDBACK LOOP: Refinement Node and Conditional
# =============================================================================

# Minimum acceptable pedagogical score
MIN_PEDAGOGICAL_SCORE = 60


def should_refine(state: PedagogicalAgentState) -> str:
    """
    Conditional: Determine if outline needs refinement.

    Returns:
        "refine" - if validation failed and attempts remain
        "finalize" - if validation passed or max attempts reached
    """
    validation_result = state.get("validation_result", {})
    is_valid = validation_result.get("is_valid", True)
    pedagogical_score = validation_result.get("pedagogical_score", 75)

    current_attempts = state.get("refinement_attempts", 0)
    max_attempts = state.get("max_refinement_attempts", 2)

    # Check if refinement is needed
    needs_refinement = (
        not is_valid or
        pedagogical_score < MIN_PEDAGOGICAL_SCORE
    )

    # Check if we can still refine
    can_refine = current_attempts < max_attempts

    if needs_refinement and can_refine:
        print(f"[AGENT] Refinement needed: score={pedagogical_score}, attempt={current_attempts + 1}/{max_attempts}", flush=True)
        return "refine"
    elif needs_refinement and not can_refine:
        print(f"[AGENT] Max refinement attempts reached ({max_attempts}), proceeding with score={pedagogical_score}", flush=True)
        return "finalize"
    else:
        print(f"[AGENT] Validation passed: score={pedagogical_score}, proceeding to finalize", flush=True)
        return "finalize"


async def refine_outline(state: PedagogicalAgentState) -> Dict[str, Any]:
    """
    Node: Refine outline based on validation feedback.

    Uses the validation warnings and suggestions to improve the outline.
    """
    current_attempts = state.get("refinement_attempts", 0)
    print(f"[AGENT] Refining outline (attempt {current_attempts + 1})", flush=True)
    state["current_node"] = "refine_outline"

    outline = state.get("outline")
    validation_result = state.get("validation_result", {})

    if not outline:
        return {
            "refinement_attempts": current_attempts + 1,
            "errors": state.get("errors", []) + ["No outline to refine"],
        }

    client, model = get_client_and_model()

    # Build structure summary for the prompt
    structure_lines = [
        f"Course: {outline.title}",
        f"Total sections: {outline.section_count}",
    ]
    for section in outline.sections:
        structure_lines.append(f"\nSection {section.order + 1}: {section.title}")
        structure_lines.append(f"  Description: {section.description or 'N/A'}")
        for lecture in section.lectures:
            structure_lines.append(
                f"  {lecture.order + 1}. {lecture.title} ({lecture.difficulty.value}, {lecture.estimated_duration_minutes}min)"
            )
            if lecture.description:
                structure_lines.append(f"      → {lecture.description[:100]}...")

    warnings = validation_result.get("warnings", [])
    suggestions = validation_result.get("suggestions", [])

    prompt = OUTLINE_REFINEMENT_PROMPT.format(
        outline_structure="\n".join(structure_lines),
        pedagogical_score=validation_result.get("pedagogical_score", 50),
        min_score=MIN_PEDAGOGICAL_SCORE,
        warnings="\n  - ".join(warnings) if warnings else "None",
        suggestions="\n  - ".join(suggestions) if suggestions else "None",
        topic=state.get("topic", "Unknown"),
        target_audience=state.get("target_audience", "general learners"),
        difficulty_start=state.get("difficulty_start", "beginner"),
        difficulty_end=state.get("difficulty_end", "intermediate"),
        language=LANGUAGE_NAMES.get(state.get("target_language", "en"), "English"),
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=2000
        )

        result = json.loads(response.choices[0].message.content)
        refined_sections = result.get("refined_sections", [])
        refinements_made = result.get("refinements_made", [])

        if refined_sections:
            # Apply refinements to the outline
            from models.course_models import CourseSection, CourseLecture, DifficultyLevel

            new_sections = []
            for sect_data in refined_sections:
                lectures = []
                for lect_data in sect_data.get("lectures", []):
                    # Parse difficulty
                    diff_str = lect_data.get("difficulty", "intermediate").lower()
                    try:
                        difficulty = DifficultyLevel(diff_str)
                    except ValueError:
                        difficulty = DifficultyLevel.INTERMEDIATE

                    lecture = CourseLecture(
                        id=f"lec_{sect_data['order']}_{lect_data['order']}",
                        title=lect_data.get("title", "Untitled"),
                        description=lect_data.get("description", ""),
                        order=lect_data.get("order", 0),
                        difficulty=difficulty,
                        estimated_duration_minutes=lect_data.get("duration_minutes", 10),
                        key_concepts=lect_data.get("key_concepts", []),
                    )
                    lectures.append(lecture)

                section = CourseSection(
                    id=f"sec_{sect_data['order']}",
                    title=sect_data.get("title", "Untitled Section"),
                    description=sect_data.get("description", ""),
                    order=sect_data.get("order", 0),
                    lectures=lectures,
                )
                new_sections.append(section)

            # Update outline with refined sections
            outline.sections = new_sections
            outline.section_count = len(new_sections)
            outline.total_lectures = sum(len(s.lectures) for s in new_sections)

            print(f"[AGENT] Applied {len(refinements_made)} refinements", flush=True)
            for refinement in refinements_made[:3]:  # Log first 3
                print(f"[AGENT]   → {refinement}", flush=True)

        # Track refinement history
        refinement_history = state.get("refinement_history", [])
        refinement_history.append({
            "attempt": current_attempts + 1,
            "previous_score": validation_result.get("pedagogical_score", 0),
            "refinements_made": refinements_made,
            "expected_improvement": result.get("expected_score_improvement", 0),
        })

        return {
            "outline": outline,
            "refinement_attempts": current_attempts + 1,
            "refinement_history": refinement_history,
            # Reset validation to trigger re-validation
            "structure_validated": False,
            "validation_result": {},
        }

    except Exception as e:
        print(f"[AGENT] Refinement error: {e}", flush=True)
        return {
            "refinement_attempts": current_attempts + 1,
            "errors": state.get("errors", []) + [f"Refinement failed: {str(e)}"],
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
