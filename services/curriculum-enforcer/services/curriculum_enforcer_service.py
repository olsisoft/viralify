"""
Curriculum Enforcer Service
Main orchestrator for validating and restructuring lesson content.
"""

import os
import json
import uuid
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from openai import AsyncOpenAI

from ..models.curriculum_models import (
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
from ..templates.predefined_templates import TemplateRegistry
from .lesson_validator import LessonStructureValidator


class CurriculumEnforcerService:
    """
    Main service for enforcing curriculum structure.

    Features:
    - Validate lesson content against templates
    - Automatically restructure content to match template
    - Generate missing phases
    - Suggest improvements
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the curriculum enforcer."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        self.validator = LessonStructureValidator(openai_api_key=self.api_key)

    async def enforce(
        self,
        request: EnforcementRequest
    ) -> EnforcementResult:
        """
        Enforce curriculum structure on lesson content.

        Args:
            request: EnforcementRequest with content and options

        Returns:
            EnforcementResult with validated and optionally restructured content
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())

        try:
            # Get the template
            if request.template_id:
                template = TemplateRegistry.get(request.template_id)
            else:
                template = TemplateRegistry.get_by_context(request.context_type)

            if not template:
                return EnforcementResult(
                    request_id=request_id,
                    success=False,
                    original_validation=ValidationResult(
                        is_valid=False,
                        score=0.0,
                        violations=[],
                        suggestions=[]
                    ),
                    template_used="none",
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    error=f"Template not found: {request.template_id or request.context_type.value}"
                )

            lesson_template = template.default_lesson_template

            # Validate original content
            original_validation = await self.validator.validate(
                content=request.content,
                template=lesson_template
            )

            # If valid or auto_fix disabled, return validation result
            if original_validation.is_valid or not request.auto_fix:
                return EnforcementResult(
                    request_id=request_id,
                    success=original_validation.is_valid,
                    original_validation=original_validation,
                    template_used=template.id,
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )

            # Auto-fix: restructure content
            restructured = await self._restructure_content(
                content=request.content,
                template=lesson_template,
                validation=original_validation,
                preserve_content=request.preserve_content
            )

            # Validate restructured content
            final_validation = await self.validator.validate(
                content=restructured["content"],
                template=lesson_template
            )

            processing_time = int((time.time() - start_time) * 1000)

            return EnforcementResult(
                request_id=request_id,
                success=final_validation.is_valid,
                original_validation=original_validation,
                final_validation=final_validation,
                restructured_content=restructured["content"],
                changes_made=restructured["changes"],
                template_used=template.id,
                processing_time_ms=processing_time
            )

        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return EnforcementResult(
                request_id=request_id,
                success=False,
                original_validation=ValidationResult(
                    is_valid=False,
                    score=0.0,
                    violations=[],
                    suggestions=[]
                ),
                template_used=request.template_id or "unknown",
                processing_time_ms=processing_time,
                error=str(e)
            )

    async def _restructure_content(
        self,
        content: LessonContent,
        template: LessonTemplate,
        validation: ValidationResult,
        preserve_content: bool = True
    ) -> Dict[str, Any]:
        """
        Restructure content to match template.
        """
        changes = []
        new_slides = list(content.slides)

        # Handle missing phases
        for missing_phase in validation.missing_required_phases:
            # Find the phase config
            phase_config = next(
                (p for p in template.phases if p.phase == missing_phase),
                None
            )

            if phase_config:
                # Generate content for missing phase
                if self.client:
                    generated_slide = await self._generate_phase_content(
                        phase=missing_phase,
                        config=phase_config,
                        lesson_context=content.title,
                        existing_content=content.slides
                    )
                else:
                    generated_slide = self._create_placeholder_slide(
                        phase=missing_phase,
                        config=phase_config
                    )

                # Insert at correct position
                insert_position = self._find_insert_position(
                    phase_config=phase_config,
                    detected_phases=validation.detected_phases,
                    total_slides=len(new_slides)
                )

                new_slides.insert(insert_position, generated_slide)
                changes.append(f"Added '{missing_phase.value}' phase at position {insert_position}")

        # Reorder if needed (only if not preserving content strictly)
        if not preserve_content and not template.allow_phase_reordering:
            new_slides, reorder_changes = self._reorder_slides(
                slides=new_slides,
                template=template,
                detected_phases=validation.detected_phases
            )
            changes.extend(reorder_changes)

        # Create new content object
        restructured_content = LessonContent(
            lesson_id=content.lesson_id,
            title=content.title,
            slides=new_slides,
            lesson_type=content.lesson_type,
            section_position=content.section_position,
            total_lessons_in_section=content.total_lessons_in_section
        )

        return {
            "content": restructured_content,
            "changes": changes
        }

    async def _generate_phase_content(
        self,
        phase: LessonPhase,
        config: PhaseConfig,
        lesson_context: str,
        existing_content: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate content for a missing phase using GPT-4.
        """
        # Summarize existing content for context
        content_summary = "\n".join([
            f"- {s.get('title', 'Slide')}: {str(s.get('content', ''))[:100]}..."
            for s in existing_content[:5]
        ])

        system_prompt = f"""You are a curriculum content generator.
Generate content for a missing lesson phase.

Phase: {phase.value}
Description: {config.prompt_template or f'Generate {phase.value} content'}
Tone: {config.tone or 'educational'}
Duration target: {config.min_duration_seconds}-{config.max_duration_seconds} seconds

Return JSON:
{{
    "type": "{phase.value}",
    "title": "Slide title",
    "content": "Main content text",
    "voiceover": "What the narrator should say",
    "duration": {config.min_duration_seconds + 15},
    "elements": ["list", "of", "visual", "elements"]
}}

Make it engaging and educational. Match the lesson topic."""

        user_content = f"""Lesson: {lesson_context}

Existing content:
{content_summary}

Generate the {phase.value} phase content."""

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=500
        )

        return json.loads(response.choices[0].message.content)

    def _create_placeholder_slide(
        self,
        phase: LessonPhase,
        config: PhaseConfig
    ) -> Dict[str, Any]:
        """
        Create a placeholder slide for a missing phase.
        """
        placeholders = {
            LessonPhase.HOOK: {
                "title": "Have you ever wondered...",
                "content": "[Add an engaging hook here]",
                "voiceover": "Welcome to this lesson. Let me start with a question..."
            },
            LessonPhase.CONCEPT: {
                "title": "Understanding the Concept",
                "content": "[Explain the main concept with a simple analogy]",
                "voiceover": "Let me explain this in simple terms..."
            },
            LessonPhase.THEORY: {
                "title": "Technical Details",
                "content": "[Add formal definitions and technical explanation]",
                "voiceover": "Now let's look at the technical details..."
            },
            LessonPhase.CODE_DEMO: {
                "title": "Let's Code",
                "content": "[Add code example here]",
                "voiceover": "Let me show you how to implement this..."
            },
            LessonPhase.RECAP: {
                "title": "Key Takeaways",
                "content": "[Summarize 3-5 main points]",
                "voiceover": "Let's recap what we learned..."
            },
            LessonPhase.EXERCISE: {
                "title": "Your Turn",
                "content": "[Add practice exercise]",
                "voiceover": "Now it's your turn to practice..."
            },
            LessonPhase.QUIZ: {
                "title": "Check Your Understanding",
                "content": "[Add quiz questions]",
                "voiceover": "Let's test what you've learned..."
            },
        }

        default = {
            "title": f"{phase.value.replace('_', ' ').title()} Section",
            "content": f"[Add {phase.value} content]",
            "voiceover": f"This is the {phase.value} section..."
        }

        base = placeholders.get(phase, default)

        return {
            "type": phase.value,
            "title": base["title"],
            "content": base["content"],
            "voiceover": base["voiceover"],
            "duration": config.min_duration_seconds + 15,
            "generated": True
        }

    def _find_insert_position(
        self,
        phase_config: PhaseConfig,
        detected_phases: List[Dict],
        total_slides: int
    ) -> int:
        """
        Find the best position to insert a new phase.
        """
        target_order = phase_config.order

        # Find the first detected phase with a higher order
        for dp in detected_phases:
            # We need to get the order for this detected phase
            # For now, use slide_index as a proxy
            if dp["slide_index"] > target_order:
                return dp["slide_index"]

        # Default: at the end or at the target order
        return min(target_order, total_slides)

    def _reorder_slides(
        self,
        slides: List[Dict],
        template: LessonTemplate,
        detected_phases: List[Dict]
    ) -> tuple[List[Dict], List[str]]:
        """
        Reorder slides to match template order.
        """
        changes = []

        # Create order map
        order_map = {p.phase.value: p.order for p in template.phases}

        # Map slides to their detected phases
        slide_phases = {}
        for dp in detected_phases:
            slide_phases[dp["slide_index"]] = dp["phase"].value

        # Sort slides by their phase order
        def get_order(idx_slide):
            idx, slide = idx_slide
            phase = slide_phases.get(idx) or slide.get("type", "")
            return order_map.get(phase, 999)

        indexed_slides = list(enumerate(slides))
        sorted_slides = sorted(indexed_slides, key=get_order)

        # Check if order changed
        original_order = [i for i, _ in indexed_slides]
        new_order = [i for i, _ in sorted_slides]

        if original_order != new_order:
            changes.append(f"Reordered slides from {original_order} to {new_order}")

        return [s for _, s in sorted_slides], changes

    async def validate_only(
        self,
        content: LessonContent,
        context_type: ContextType = ContextType.EDUCATION,
        template_id: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate content without restructuring.
        Convenience method for quick validation.
        """
        if template_id:
            template = TemplateRegistry.get(template_id)
        else:
            template = TemplateRegistry.get_by_context(context_type)

        if not template:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                violations=[PhaseViolation(
                    phase=LessonPhase.HOOK,
                    violation_type="missing",
                    message="Template not found",
                    severity="error"
                )],
                suggestions=["Provide a valid template"]
            )

        return await self.validator.validate(
            content=content,
            template=template.default_lesson_template
        )

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available curriculum templates.
        """
        return [
            {
                "id": t.id,
                "name": t.name,
                "context_type": t.context_type.value,
                "description": t.description,
                "phases": [p.phase.value for p in t.default_lesson_template.phases if p.required]
            }
            for t in TemplateRegistry.list_all()
        ]

    def get_template_phases(
        self,
        context_type: ContextType
    ) -> List[Dict[str, Any]]:
        """
        Get the phases for a specific context type.
        """
        template = TemplateRegistry.get_by_context(context_type)
        if not template:
            return []

        return [
            {
                "phase": p.phase.value,
                "required": p.required,
                "order": p.order,
                "duration_range": f"{p.min_duration_seconds}-{p.max_duration_seconds}s",
                "prompt": p.prompt_template
            }
            for p in template.default_lesson_template.phases
        ]

    def register_custom_template(
        self,
        template: CurriculumTemplate
    ) -> None:
        """
        Register a custom curriculum template.
        """
        TemplateRegistry.register(template)


# Factory function
def create_curriculum_enforcer(
    openai_api_key: Optional[str] = None
) -> CurriculumEnforcerService:
    """Create a configured CurriculumEnforcerService instance."""
    return CurriculumEnforcerService(openai_api_key=openai_api_key)
