"""
Input Validator Agent

Validates that all frontend choices and required configurations are present
before starting the course generation pipeline. This ensures no missing data
causes issues downstream.
"""
from typing import List, Dict, Any

from agents.base import (
    BaseAgent,
    AgentType,
    AgentStatus,
    CourseGenerationState,
    ValidationError,
)


# Required fields for course generation
REQUIRED_FIELDS = [
    "topic",
    "structure",
]

# Optional fields with defaults
OPTIONAL_FIELDS_DEFAULTS = {
    "content_language": "en",
    "programming_language": None,  # Only needed for coding courses
    "lesson_elements": {
        "concept_intro": True,
        "diagram_schema": True,
        "code_typing": False,
        "code_execution": False,
        "voiceover_explanation": True,
        "curriculum_slide": True,
    },
}

# Fields required within structure config
STRUCTURE_REQUIRED = [
    "total_duration_minutes",
    "number_of_sections",
    "lectures_per_section",
]

# Fields required within lesson_elements
LESSON_ELEMENTS_KEYS = [
    "concept_intro",
    "diagram_schema",
    "code_typing",
    "code_execution",
    "voiceover_explanation",
    "curriculum_slide",
]

# Valid values for various fields
VALID_LANGUAGES = ["en", "fr", "es", "de", "pt", "it", "nl", "pl", "ru", "zh"]
VALID_PROGRAMMING_LANGUAGES = [
    "python", "javascript", "typescript", "java", "csharp", "cpp",
    "go", "rust", "ruby", "php", "swift", "kotlin"
]
VALID_DIFFICULTIES = ["beginner", "intermediate", "advanced", "expert"]
VALID_STYLES = ["modern", "minimal", "corporate", "creative", "dark", "light"]
VALID_TYPING_SPEEDS = ["slow", "natural", "moderate", "fast"]


class InputValidatorAgent(BaseAgent):
    """
    Validates all input parameters from the frontend before generation starts.

    This agent checks:
    1. All required fields are present
    2. Field values are valid (within allowed ranges/options)
    3. Configurations are consistent (e.g., code_execution requires code_typing)
    4. Duration and structure are realistic
    """

    def __init__(self):
        super().__init__(AgentType.INPUT_VALIDATOR)

    async def process(self, state: CourseGenerationState) -> CourseGenerationState:
        """
        Validate all inputs and return state with validation results.

        Args:
            state: Current course generation state

        Returns:
            State updated with validation results
        """
        self.log(f"Validating input for job: {state.get('job_id', 'unknown')}")

        errors: List[ValidationError] = []
        warnings: List[str] = []
        missing_fields: List[str] = []

        # 0. Apply defaults for optional fields
        for field, default_value in OPTIONAL_FIELDS_DEFAULTS.items():
            if not state.get(field):
                state[field] = default_value
                if default_value is not None:
                    warnings.append(f"Using default value for '{field}'")

        # 1. Check required top-level fields
        for field in REQUIRED_FIELDS:
            if not state.get(field):
                errors.append({
                    "field": field,
                    "message": f"Required field '{field}' is missing or empty",
                    "severity": "error"
                })
                missing_fields.append(field)

        # 2. Validate structure config
        structure = state.get("structure", {})
        if structure:
            structure_errors = self._validate_structure(structure)
            errors.extend(structure_errors)
        else:
            missing_fields.append("structure")

        # 3. Validate lesson elements config
        lesson_elements = state.get("lesson_elements", {})
        if lesson_elements:
            element_errors, element_warnings = self._validate_lesson_elements(
                lesson_elements, state.get("programming_language")
            )
            errors.extend(element_errors)
            warnings.extend(element_warnings)
        else:
            missing_fields.append("lesson_elements")

        # 4. Validate language settings
        content_lang = state.get("content_language", "")
        if content_lang and content_lang not in VALID_LANGUAGES:
            errors.append({
                "field": "content_language",
                "message": f"Invalid content language '{content_lang}'. Valid: {VALID_LANGUAGES}",
                "severity": "error"
            })

        prog_lang = state.get("programming_language", "")
        if prog_lang and prog_lang not in VALID_PROGRAMMING_LANGUAGES:
            warnings.append(
                f"Programming language '{prog_lang}' is not in standard list. "
                "Code execution may not be supported."
            )

        # 5. Validate difficulty settings
        diff_start = state.get("difficulty_start", "").lower()
        diff_end = state.get("difficulty_end", "").lower()

        if diff_start and diff_start not in VALID_DIFFICULTIES:
            errors.append({
                "field": "difficulty_start",
                "message": f"Invalid difficulty '{diff_start}'. Valid: {VALID_DIFFICULTIES}",
                "severity": "error"
            })

        if diff_end and diff_end not in VALID_DIFFICULTIES:
            errors.append({
                "field": "difficulty_end",
                "message": f"Invalid difficulty '{diff_end}'. Valid: {VALID_DIFFICULTIES}",
                "severity": "error"
            })

        # Check difficulty progression makes sense
        if diff_start and diff_end:
            diff_order = VALID_DIFFICULTIES
            if diff_order.index(diff_start) > diff_order.index(diff_end):
                warnings.append(
                    f"Difficulty goes from '{diff_start}' to '{diff_end}' which is backwards. "
                    "Consider swapping start and end."
                )

        # 6. Validate style and speed
        style = state.get("style", "")
        if style and style not in VALID_STYLES:
            warnings.append(f"Style '{style}' is not standard. Using default.")

        typing_speed = state.get("typing_speed", "")
        if typing_speed and typing_speed not in VALID_TYPING_SPEEDS:
            warnings.append(f"Typing speed '{typing_speed}' is not standard. Using 'natural'.")

        # 7. Validate quiz config if present
        quiz_config = state.get("quiz_config")
        if quiz_config:
            quiz_errors = self._validate_quiz_config(quiz_config)
            errors.extend(quiz_errors)

        # 8. Cross-field validations
        cross_warnings = self._validate_cross_field_consistency(state)
        warnings.extend(cross_warnings)

        # Determine overall validation status
        has_errors = len([e for e in errors if e["severity"] == "error"]) > 0

        # Update state with validation results
        state["input_validated"] = not has_errors
        state["input_validation_errors"] = errors
        state["missing_required_fields"] = missing_fields
        state["warnings"] = state.get("warnings", []) + warnings

        # Add to agent history
        self.add_to_history(
            state,
            AgentStatus.COMPLETED if not has_errors else AgentStatus.FAILED,
        )

        if has_errors:
            error_messages = [e["message"] for e in errors if e["severity"] == "error"]
            self.log(f"Validation FAILED with {len(error_messages)} errors")
            for msg in error_messages[:5]:  # Log first 5 errors
                self.log(f"  - {msg}")
        else:
            self.log(f"Validation PASSED with {len(warnings)} warnings")

        return state

    def _validate_structure(self, structure: Dict[str, Any]) -> List[ValidationError]:
        """Validate the course structure configuration"""
        errors = []

        # Check required structure fields
        for field in STRUCTURE_REQUIRED:
            if field not in structure or structure.get(field) is None:
                errors.append({
                    "field": f"structure.{field}",
                    "message": f"Structure field '{field}' is required",
                    "severity": "error"
                })

        # Validate duration
        duration = structure.get("total_duration_minutes", 0)
        if duration < 5:
            errors.append({
                "field": "structure.total_duration_minutes",
                "message": "Course duration must be at least 5 minutes",
                "severity": "error"
            })
        elif duration > 600:  # 10 hours
            errors.append({
                "field": "structure.total_duration_minutes",
                "message": "Course duration cannot exceed 600 minutes (10 hours)",
                "severity": "error"
            })

        # Validate section count
        sections = structure.get("number_of_sections", 0)
        if sections < 1:
            errors.append({
                "field": "structure.number_of_sections",
                "message": "Must have at least 1 section",
                "severity": "error"
            })
        elif sections > 20:
            errors.append({
                "field": "structure.number_of_sections",
                "message": "Cannot have more than 20 sections",
                "severity": "error"
            })

        # Validate lectures per section
        lectures = structure.get("lectures_per_section", 0)
        if lectures < 1:
            errors.append({
                "field": "structure.lectures_per_section",
                "message": "Must have at least 1 lecture per section",
                "severity": "error"
            })
        elif lectures > 10:
            errors.append({
                "field": "structure.lectures_per_section",
                "message": "Cannot have more than 10 lectures per section",
                "severity": "error"
            })

        # Check if duration is realistic for structure
        if sections > 0 and lectures > 0 and duration > 0:
            total_lectures = sections * lectures
            avg_lecture_minutes = duration / total_lectures
            if avg_lecture_minutes < 1:
                errors.append({
                    "field": "structure",
                    "message": f"With {total_lectures} lectures in {duration} minutes, "
                              f"each lecture would be {avg_lecture_minutes:.1f} min (too short)",
                    "severity": "error"
                })

        return errors

    def _validate_lesson_elements(
        self,
        elements: Dict[str, Any],
        programming_language: str
    ) -> tuple[List[ValidationError], List[str]]:
        """Validate lesson elements configuration"""
        errors = []
        warnings = []

        # Check that at least some elements are enabled
        enabled_count = sum(1 for k in LESSON_ELEMENTS_KEYS if elements.get(k, False))
        if enabled_count == 0:
            errors.append({
                "field": "lesson_elements",
                "message": "At least one lesson element must be enabled",
                "severity": "error"
            })

        # Code execution requires code typing
        if elements.get("code_execution") and not elements.get("code_typing"):
            warnings.append(
                "code_execution is enabled but code_typing is disabled. "
                "Enabling code_typing automatically."
            )
            elements["code_typing"] = True

        # Warn if no voiceover
        if not elements.get("voiceover_explanation"):
            warnings.append(
                "voiceover_explanation is disabled. "
                "Course will be generated without narration."
            )

        return errors, warnings

    def _validate_quiz_config(self, quiz_config: Dict[str, Any]) -> List[ValidationError]:
        """Validate quiz configuration"""
        errors = []

        if quiz_config.get("enabled"):
            frequency = quiz_config.get("frequency", "")
            valid_frequencies = ["per_lecture", "per_section", "end_only", "custom"]

            if frequency and frequency not in valid_frequencies:
                errors.append({
                    "field": "quiz_config.frequency",
                    "message": f"Invalid quiz frequency '{frequency}'",
                    "severity": "error"
                })

            if frequency == "custom":
                interval = quiz_config.get("custom_interval")
                if not interval or interval < 1:
                    errors.append({
                        "field": "quiz_config.custom_interval",
                        "message": "Custom quiz frequency requires a positive interval",
                        "severity": "error"
                    })

        return errors

    def _validate_cross_field_consistency(
        self,
        state: CourseGenerationState
    ) -> List[str]:
        """Check consistency across different configuration fields"""
        warnings = []

        # Avatar requires voice
        if state.get("include_avatar") and not state.get("voice_id"):
            warnings.append(
                "Avatar is enabled but no voice_id specified. "
                "Using default voice for avatar."
            )

        # Code-heavy course should have code elements
        topic = (state.get("topic") or "").lower()
        code_keywords = ["programming", "coding", "development", "api", "framework"]
        is_code_topic = any(kw in topic for kw in code_keywords)

        elements = state.get("lesson_elements", {})
        if is_code_topic and not elements.get("code_typing"):
            warnings.append(
                f"Topic '{state.get('topic')}' appears to be code-related "
                "but code_typing is disabled. Consider enabling it."
            )

        # Long courses should have curriculum slides
        structure = state.get("structure", {})
        duration = structure.get("total_duration_minutes", 0)
        if duration > 30 and not elements.get("curriculum_slide"):
            warnings.append(
                "Course is longer than 30 minutes but curriculum_slide is disabled. "
                "Consider enabling it for better navigation."
            )

        return warnings


def create_input_validator() -> InputValidatorAgent:
    """Factory function to create an InputValidatorAgent"""
    return InputValidatorAgent()
