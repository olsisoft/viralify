"""
Lesson Structure Validator
Validates lesson content against curriculum templates.
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple
from openai import AsyncOpenAI

from ..models.curriculum_models import (
    LessonPhase,
    PhaseConfig,
    LessonTemplate,
    LessonContent,
    ValidationResult,
    PhaseViolation,
)


class LessonStructureValidator:
    """
    Validates lesson content against a template.
    Uses GPT-4 for intelligent phase detection.
    """

    # Keywords to detect phases
    PHASE_KEYWORDS = {
        LessonPhase.HOOK: [
            "ever wondered", "imagine", "what if", "have you ever",
            "problem", "struggle", "frustrating", "challenge"
        ],
        LessonPhase.CONCEPT: [
            "basically", "simply put", "in other words", "think of it like",
            "analogy", "similar to", "like a", "just like"
        ],
        LessonPhase.THEORY: [
            "formally", "definition", "according to", "technically",
            "the term", "officially", "in computer science"
        ],
        LessonPhase.CODE_DEMO: [
            "let's code", "let's write", "here's how", "implementation",
            "def ", "function", "class ", "import", "```"
        ],
        LessonPhase.EXERCISE: [
            "try this", "your turn", "exercise", "practice",
            "implement", "create your own"
        ],
        LessonPhase.QUIZ: [
            "quiz", "test yourself", "check your understanding",
            "which of the following", "true or false"
        ],
        LessonPhase.RECAP: [
            "in summary", "to summarize", "key takeaways", "remember",
            "we learned", "we covered", "main points"
        ],
        LessonPhase.USE_CASE: [
            "real-world", "use case", "in production", "companies use",
            "example from", "at google", "at amazon"
        ],
        LessonPhase.ROI: [
            "saves", "reduces", "increases", "roi", "value",
            "cost", "revenue", "efficiency"
        ],
        LessonPhase.ACTION_ITEMS: [
            "next steps", "action items", "todo", "implement this",
            "start by", "your homework"
        ],
    }

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the validator."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    async def validate(
        self,
        content: LessonContent,
        template: LessonTemplate,
        use_ai: bool = True
    ) -> ValidationResult:
        """
        Validate lesson content against a template.

        Args:
            content: The lesson content to validate
            template: The template to validate against
            use_ai: Whether to use GPT-4 for intelligent detection

        Returns:
            ValidationResult with violations and suggestions
        """
        # Detect phases in content
        if use_ai and self.client:
            try:
                detected_phases = await self._ai_detect_phases(content)
            except Exception as e:
                print(f"[LessonValidator] AI detection failed: {e}, using heuristics", flush=True)
                detected_phases = self._heuristic_detect_phases(content)
        else:
            detected_phases = self._heuristic_detect_phases(content)

        # Validate against template
        violations = []
        suggestions = []

        # Get required phases from template
        required_phases = [p for p in template.phases if p.required]
        optional_phases = [p for p in template.phases if not p.required]

        # Check for missing required phases
        detected_phase_types = {dp["phase"] for dp in detected_phases}
        missing_required = []

        for phase_config in required_phases:
            if phase_config.phase not in detected_phase_types:
                missing_required.append(phase_config.phase)
                violations.append(PhaseViolation(
                    phase=phase_config.phase,
                    violation_type="missing",
                    message=f"Required phase '{phase_config.phase.value}' is missing",
                    severity="error"
                ))

        # Check phase order
        if not template.allow_phase_reordering:
            order_violations = self._check_phase_order(detected_phases, template.phases)
            violations.extend(order_violations)

        # Check duration constraints
        duration_violations = self._check_durations(detected_phases, template.phases)
        violations.extend(duration_violations)

        # Check for extra phases not in template
        template_phase_types = {p.phase for p in template.phases}
        extra_phases = [
            dp["phase"] for dp in detected_phases
            if dp["phase"] not in template_phase_types
        ]

        # Calculate score
        total_required = len(required_phases)
        missing_count = len(missing_required)
        error_count = len([v for v in violations if v.severity == "error"])
        warning_count = len([v for v in violations if v.severity == "warning"])

        if total_required == 0:
            score = 1.0
        else:
            # Base score from required phases
            score = (total_required - missing_count) / total_required
            # Penalty for errors and warnings
            score -= error_count * 0.1
            score -= warning_count * 0.05
            score = max(0.0, min(1.0, score))

        # Generate suggestions
        if missing_required:
            suggestions.append(
                f"Add the following phases: {', '.join(p.value for p in missing_required)}"
            )

        if extra_phases:
            suggestions.append(
                f"Consider removing or reorganizing: {', '.join(p.value for p in extra_phases)}"
            )

        return ValidationResult(
            is_valid=len(missing_required) == 0 and error_count == 0,
            score=score,
            violations=violations,
            suggestions=suggestions,
            detected_phases=detected_phases,
            missing_required_phases=missing_required,
            extra_phases=extra_phases
        )

    def _heuristic_detect_phases(
        self,
        content: LessonContent
    ) -> List[Dict[str, Any]]:
        """
        Detect phases using keyword matching.
        """
        detected = []

        for i, slide in enumerate(content.slides):
            slide_content = json.dumps(slide).lower()
            slide_type = slide.get("type", "").lower()

            # Check each phase's keywords
            best_match = None
            best_score = 0

            for phase, keywords in self.PHASE_KEYWORDS.items():
                score = sum(1 for kw in keywords if kw in slide_content)
                if score > best_score:
                    best_score = score
                    best_match = phase

            # Also check slide type
            if not best_match or best_score < 2:
                type_map = {
                    "hook": LessonPhase.HOOK,
                    "concept": LessonPhase.CONCEPT,
                    "theory": LessonPhase.THEORY,
                    "code": LessonPhase.CODE_DEMO,
                    "demo": LessonPhase.CODE_DEMO,
                    "exercise": LessonPhase.EXERCISE,
                    "quiz": LessonPhase.QUIZ,
                    "recap": LessonPhase.RECAP,
                    "summary": LessonPhase.RECAP,
                    "conclusion": LessonPhase.RECAP,
                }
                best_match = type_map.get(slide_type, best_match)

            if best_match:
                detected.append({
                    "phase": best_match,
                    "slide_index": i,
                    "confidence": min(best_score / 5, 1.0) if best_score > 0 else 0.5,
                    "duration_seconds": slide.get("duration", 60)
                })

        return detected

    async def _ai_detect_phases(
        self,
        content: LessonContent
    ) -> List[Dict[str, Any]]:
        """
        Detect phases using GPT-4 for intelligent analysis.
        """
        # Prepare slides for analysis
        slides_text = []
        for i, slide in enumerate(content.slides):
            slide_summary = f"Slide {i+1}: "
            if slide.get("title"):
                slide_summary += f"Title: {slide['title']}. "
            if slide.get("content"):
                slide_summary += f"Content: {str(slide['content'])[:200]}..."
            if slide.get("voiceover"):
                slide_summary += f" Voiceover: {str(slide['voiceover'])[:200]}..."
            slides_text.append(slide_summary)

        system_prompt = """You are a curriculum structure analyzer.
Analyze lesson slides and detect which pedagogical phase each slide represents.

Phases to detect:
- hook: Engaging opener, problem statement, surprising fact
- concept: Simple explanation, analogy, intuitive understanding
- theory: Formal definition, technical details, terminology
- code_demo: Live coding, implementation, code examples
- example: Worked example, walkthrough
- exercise: Practice activity, hands-on task
- quiz: Assessment, knowledge check
- recap: Summary, key takeaways
- use_case: Real-world application, case study
- roi: Business value, metrics
- action_items: Next steps, homework

Return a JSON array of detected phases:
[
    {"slide_index": 0, "phase": "hook", "confidence": 0.9},
    {"slide_index": 1, "phase": "concept", "confidence": 0.85},
    ...
]"""

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join(slides_text)}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500
        )

        result = json.loads(response.choices[0].message.content)
        phases = result.get("phases", result) if isinstance(result, dict) else result

        # Convert to proper format
        detected = []
        for item in phases:
            phase_str = item.get("phase", "")
            try:
                phase = LessonPhase(phase_str)
                detected.append({
                    "phase": phase,
                    "slide_index": item.get("slide_index", 0),
                    "confidence": item.get("confidence", 0.5),
                    "duration_seconds": content.slides[item.get("slide_index", 0)].get("duration", 60)
                })
            except ValueError:
                continue

        return detected

    def _check_phase_order(
        self,
        detected: List[Dict[str, Any]],
        template_phases: List[PhaseConfig]
    ) -> List[PhaseViolation]:
        """
        Check if phases appear in the correct order.
        """
        violations = []

        # Build expected order map
        order_map = {p.phase: p.order for p in template_phases}

        # Check detected phases are in order
        last_order = -1
        for dp in detected:
            phase = dp["phase"]
            if phase in order_map:
                expected_order = order_map[phase]
                if expected_order < last_order:
                    violations.append(PhaseViolation(
                        phase=phase,
                        violation_type="wrong_order",
                        message=f"Phase '{phase.value}' appears after a phase that should come later",
                        severity="warning",
                        slide_index=dp["slide_index"]
                    ))
                last_order = max(last_order, expected_order)

        return violations

    def _check_durations(
        self,
        detected: List[Dict[str, Any]],
        template_phases: List[PhaseConfig]
    ) -> List[PhaseViolation]:
        """
        Check if phase durations are within bounds.
        """
        violations = []

        # Build duration map
        duration_map = {
            p.phase: (p.min_duration_seconds, p.max_duration_seconds)
            for p in template_phases
        }

        for dp in detected:
            phase = dp["phase"]
            duration = dp.get("duration_seconds", 0)

            if phase in duration_map:
                min_dur, max_dur = duration_map[phase]

                if duration < min_dur:
                    violations.append(PhaseViolation(
                        phase=phase,
                        violation_type="too_short",
                        message=f"Phase '{phase.value}' is too short ({duration}s < {min_dur}s)",
                        severity="warning",
                        slide_index=dp["slide_index"]
                    ))
                elif duration > max_dur:
                    violations.append(PhaseViolation(
                        phase=phase,
                        violation_type="too_long",
                        message=f"Phase '{phase.value}' is too long ({duration}s > {max_dur}s)",
                        severity="info",
                        slide_index=dp["slide_index"]
                    ))

        return violations

    def validate_sync(
        self,
        content: LessonContent,
        template: LessonTemplate
    ) -> ValidationResult:
        """
        Synchronous validation using only heuristics.
        """
        detected_phases = self._heuristic_detect_phases(content)

        # Simplified validation
        required_phases = [p for p in template.phases if p.required]
        detected_phase_types = {dp["phase"] for dp in detected_phases}
        missing_required = [
            p.phase for p in required_phases
            if p.phase not in detected_phase_types
        ]

        violations = [
            PhaseViolation(
                phase=phase,
                violation_type="missing",
                message=f"Required phase '{phase.value}' is missing",
                severity="error"
            )
            for phase in missing_required
        ]

        total_required = len(required_phases)
        missing_count = len(missing_required)
        score = (total_required - missing_count) / total_required if total_required > 0 else 1.0

        return ValidationResult(
            is_valid=missing_count == 0,
            score=score,
            violations=violations,
            suggestions=[f"Add: {', '.join(p.value for p in missing_required)}"] if missing_required else [],
            detected_phases=detected_phases,
            missing_required_phases=missing_required,
            extra_phases=[]
        )
