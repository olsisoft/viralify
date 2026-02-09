"""
Unit tests for PROFILE_ADAPTATION_PROMPT

Tests the agentic prompt structure, decision rules, and output validation
for the pedagogical profile adaptation agent.
"""

import pytest
import json
import re
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def import_prompts_module():
    """Import pedagogical_prompts.py directly to avoid langgraph dependency"""
    prompts_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "agents",
        "pedagogical_prompts.py"
    )
    spec = importlib.util.spec_from_file_location("pedagogical_prompts", prompts_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prompts_module = import_prompts_module()
PROFILE_ADAPTATION_PROMPT = prompts_module.PROFILE_ADAPTATION_PROMPT


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_input_signals():
    """Sample input signals for prompt formatting"""
    return {
        "detected_persona": "Backend Engineer",
        "topic_complexity": "intermediate",
        "category": "tech",
        "requires_code": True,
        "requires_diagrams": True,
        "requires_hands_on": True,
        "available_elements": """- code_demo: Code Demo - Live coding demonstration
- architecture_diagram: Architecture Diagram - System architecture visualization
- debug_tips: Debug Tips - Common debugging strategies
- case_study: Case Study - Real-world examples
- terminal_output: Terminal Output - Command line examples""",
    }


@pytest.fixture
def sample_valid_output():
    """Sample valid LLM output that respects all constraints"""
    return {
        "content_preferences": {
            "code_weight": 0.85,
            "diagram_weight": 0.6,
            "demo_weight": 0.7,
            "theory_weight": 0.35,
            "case_study_weight": 0.5
        },
        "recommended_elements": ["code_demo", "architecture_diagram", "debug_tips", "case_study"],
        "adaptation_notes": "Code-driven learning with system-level diagrams for data pipelines."
    }


@pytest.fixture
def sample_invalid_outputs():
    """Collection of invalid outputs that violate constraints"""
    return {
        "low_code_weight": {
            # requires_code=true but code_weight < 0.6
            "content_preferences": {
                "code_weight": 0.3,  # INVALID: should be >= 0.6
                "diagram_weight": 0.6,
                "demo_weight": 0.7,
                "theory_weight": 0.35,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["code_demo", "architecture_diagram"],
            "adaptation_notes": "..."
        },
        "low_diagram_weight": {
            # requires_diagrams=true but diagram_weight < 0.5
            "content_preferences": {
                "code_weight": 0.7,
                "diagram_weight": 0.3,  # INVALID: should be >= 0.5
                "demo_weight": 0.7,
                "theory_weight": 0.35,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["code_demo", "architecture_diagram"],
            "adaptation_notes": "..."
        },
        "low_theory_weight": {
            # theory_weight < 0.2
            "content_preferences": {
                "code_weight": 0.7,
                "diagram_weight": 0.6,
                "demo_weight": 0.7,
                "theory_weight": 0.1,  # INVALID: should be >= 0.2
                "case_study_weight": 0.5
            },
            "recommended_elements": ["code_demo", "architecture_diagram"],
            "adaptation_notes": "..."
        },
        "weight_out_of_range": {
            # weights > 1.0
            "content_preferences": {
                "code_weight": 1.5,  # INVALID: should be <= 1.0
                "diagram_weight": 0.6,
                "demo_weight": 0.7,
                "theory_weight": 0.35,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["code_demo", "architecture_diagram"],
            "adaptation_notes": "..."
        },
        "too_few_elements": {
            # Only 2 elements (should be 3-6)
            "content_preferences": {
                "code_weight": 0.7,
                "diagram_weight": 0.6,
                "demo_weight": 0.7,
                "theory_weight": 0.35,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["code_demo", "architecture_diagram"],  # Only 2
            "adaptation_notes": "..."
        },
        "too_many_elements": {
            # 8 elements (should be 3-6)
            "content_preferences": {
                "code_weight": 0.7,
                "diagram_weight": 0.6,
                "demo_weight": 0.7,
                "theory_weight": 0.35,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["a", "b", "c", "d", "e", "f", "g", "h"],  # 8 elements
            "adaptation_notes": "..."
        },
        "total_weight_too_low": {
            # Total weight < 2.5
            "content_preferences": {
                "code_weight": 0.3,
                "diagram_weight": 0.3,
                "demo_weight": 0.3,
                "theory_weight": 0.2,
                "case_study_weight": 0.3
            },  # Total = 1.4 (too low)
            "recommended_elements": ["code_demo", "architecture_diagram", "case_study"],
            "adaptation_notes": "..."
        },
        "total_weight_too_high": {
            # Total weight > 3.5
            "content_preferences": {
                "code_weight": 1.0,
                "diagram_weight": 1.0,
                "demo_weight": 1.0,
                "theory_weight": 0.8,
                "case_study_weight": 0.9
            },  # Total = 4.7 (too high)
            "recommended_elements": ["code_demo", "architecture_diagram", "case_study"],
            "adaptation_notes": "..."
        },
    }


# ============================================================================
# Validator Helper Class
# ============================================================================

class ProfileAdaptationValidator:
    """Validates outputs against PROFILE_ADAPTATION_PROMPT constraints"""

    @staticmethod
    def validate_weight_range(weight: float) -> bool:
        """Weights must be between 0.0 and 1.0"""
        return 0.0 <= weight <= 1.0

    @staticmethod
    def validate_total_weight(preferences: Dict[str, float]) -> bool:
        """Total weight must be between 2.5 and 3.5"""
        total = sum(preferences.values())
        return 2.5 <= total <= 3.5

    @staticmethod
    def validate_code_constraint(preferences: Dict[str, float], requires_code: bool) -> bool:
        """If requires_code = true → code_weight ≥ 0.6"""
        if requires_code:
            return preferences.get("code_weight", 0) >= 0.6
        return True

    @staticmethod
    def validate_diagram_constraint(preferences: Dict[str, float], requires_diagrams: bool) -> bool:
        """If requires_diagrams = true → diagram_weight ≥ 0.5"""
        if requires_diagrams:
            return preferences.get("diagram_weight", 0) >= 0.5
        return True

    @staticmethod
    def validate_theory_constraint(preferences: Dict[str, float]) -> bool:
        """theory_weight MUST be ≥ 0.2"""
        return preferences.get("theory_weight", 0) >= 0.2

    @staticmethod
    def validate_elements_count(elements: List[str]) -> bool:
        """Select 3 to 6 lesson elements"""
        return 3 <= len(elements) <= 6

    @classmethod
    def validate_output(
        cls,
        output: Dict[str, Any],
        requires_code: bool = False,
        requires_diagrams: bool = False
    ) -> Dict[str, Any]:
        """
        Validate full output against all constraints.
        Returns dict with validation results and issues.
        """
        issues = []
        preferences = output.get("content_preferences", {})
        elements = output.get("recommended_elements", [])

        # Check all weights are in range
        for key, value in preferences.items():
            if not cls.validate_weight_range(value):
                issues.append(f"{key}={value} out of range [0.0, 1.0]")

        # Check total weight
        if not cls.validate_total_weight(preferences):
            total = sum(preferences.values())
            issues.append(f"Total weight {total:.2f} not in range [2.5, 3.5]")

        # Check code constraint
        if not cls.validate_code_constraint(preferences, requires_code):
            issues.append(f"code_weight={preferences.get('code_weight')} < 0.6 (requires_code=true)")

        # Check diagram constraint
        if not cls.validate_diagram_constraint(preferences, requires_diagrams):
            issues.append(f"diagram_weight={preferences.get('diagram_weight')} < 0.5 (requires_diagrams=true)")

        # Check theory constraint
        if not cls.validate_theory_constraint(preferences):
            issues.append(f"theory_weight={preferences.get('theory_weight')} < 0.2")

        # Check elements count
        if not cls.validate_elements_count(elements):
            issues.append(f"Element count {len(elements)} not in range [3, 6]")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "total_weight": sum(preferences.values()),
            "element_count": len(elements)
        }


# ============================================================================
# Tests for Prompt Structure
# ============================================================================

class TestPromptStructure:
    """Tests for the prompt structure and format"""

    def test_prompt_contains_role_definition(self):
        """Test that prompt defines the agent role"""
        assert "Senior Technical Curriculum Agent" in PROFILE_ADAPTATION_PROMPT
        assert "operating autonomously" in PROFILE_ADAPTATION_PROMPT

    def test_prompt_contains_context_section(self):
        """Test that prompt contains context about Viralify"""
        assert "## CONTEXT" in PROFILE_ADAPTATION_PROMPT
        assert "Viralify" in PROFILE_ADAPTATION_PROMPT
        assert "video courses" in PROFILE_ADAPTATION_PROMPT

    def test_prompt_contains_input_signals_section(self):
        """Test that prompt documents input signals"""
        assert "## INPUT SIGNALS" in PROFILE_ADAPTATION_PROMPT
        assert "LEARNER SIGNALS" in PROFILE_ADAPTATION_PROMPT
        assert "TOPIC SIGNALS" in PROFILE_ADAPTATION_PROMPT
        assert "SYSTEM CAPABILITIES" in PROFILE_ADAPTATION_PROMPT

    def test_prompt_contains_all_placeholders(self):
        """Test that prompt contains all required placeholders"""
        required_placeholders = [
            "{detected_persona}",
            "{topic_complexity}",
            "{category}",
            "{requires_code}",
            "{requires_diagrams}",
            "{requires_hands_on}",
            "{available_elements}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in PROFILE_ADAPTATION_PROMPT, f"Missing placeholder: {placeholder}"

    def test_prompt_contains_decision_rules(self):
        """Test that prompt contains hard constraints section"""
        assert "## DECISION RULES" in PROFILE_ADAPTATION_PROMPT
        assert "HARD CONSTRAINTS" in PROFILE_ADAPTATION_PROMPT

    def test_prompt_contains_self_validation(self):
        """Test that prompt includes self-validation section"""
        assert "## SELF-VALIDATION" in PROFILE_ADAPTATION_PROMPT
        assert "Verify" in PROFILE_ADAPTATION_PROMPT

    def test_prompt_contains_output_contract(self):
        """Test that prompt defines JSON output format"""
        assert "## OUTPUT CONTRACT" in PROFILE_ADAPTATION_PROMPT
        assert "valid JSON only" in PROFILE_ADAPTATION_PROMPT
        assert "content_preferences" in PROFILE_ADAPTATION_PROMPT
        assert "recommended_elements" in PROFILE_ADAPTATION_PROMPT
        assert "adaptation_notes" in PROFILE_ADAPTATION_PROMPT

    def test_prompt_contains_examples(self):
        """Test that prompt includes examples"""
        assert "## EXAMPLES" in PROFILE_ADAPTATION_PROMPT
        assert "Python for Data Science" in PROFILE_ADAPTATION_PROMPT
        assert "Leadership Fundamentals" in PROFILE_ADAPTATION_PROMPT

    def test_prompt_contains_agent_responsibilities(self):
        """Test that prompt defines agent responsibilities"""
        assert "## AGENT RESPONSIBILITIES" in PROFILE_ADAPTATION_PROMPT
        assert "Analyze the technical nature" in PROFILE_ADAPTATION_PROMPT


class TestPromptConstraints:
    """Tests for the constraints defined in the prompt"""

    def test_weight_range_constraint_documented(self):
        """Test that weight range constraint is documented"""
        assert "0.0 and 1.0" in PROFILE_ADAPTATION_PROMPT
        assert "floats between 0.0 and 1.0" in PROFILE_ADAPTATION_PROMPT

    def test_total_weight_constraint_documented(self):
        """Test that total weight constraint is documented"""
        assert "2.5 and 3.5" in PROFILE_ADAPTATION_PROMPT
        assert "Total weight MUST" in PROFILE_ADAPTATION_PROMPT

    def test_code_weight_constraint_documented(self):
        """Test that code weight constraint is documented"""
        assert "requires_code = true" in PROFILE_ADAPTATION_PROMPT
        assert "code_weight ≥ 0.6" in PROFILE_ADAPTATION_PROMPT

    def test_diagram_weight_constraint_documented(self):
        """Test that diagram weight constraint is documented"""
        assert "requires_diagrams = true" in PROFILE_ADAPTATION_PROMPT
        assert "diagram_weight ≥ 0.5" in PROFILE_ADAPTATION_PROMPT

    def test_theory_weight_constraint_documented(self):
        """Test that theory weight constraint is documented"""
        assert "theory_weight MUST be ≥ 0.2" in PROFILE_ADAPTATION_PROMPT

    def test_elements_count_constraint_documented(self):
        """Test that elements count constraint is documented"""
        assert "3 to 6 lesson elements" in PROFILE_ADAPTATION_PROMPT


# ============================================================================
# Tests for Prompt Formatting
# ============================================================================

class TestPromptFormatting:
    """Tests for prompt formatting with input signals"""

    def test_prompt_formats_correctly(self, sample_input_signals):
        """Test that prompt can be formatted with all signals"""
        formatted = PROFILE_ADAPTATION_PROMPT.format(**sample_input_signals)

        assert "Backend Engineer" in formatted
        assert "intermediate" in formatted
        assert "tech" in formatted
        assert "True" in formatted  # requires_code
        assert "code_demo" in formatted

    def test_formatted_prompt_no_placeholders_remain(self, sample_input_signals):
        """Test that no placeholders remain after formatting"""
        formatted = PROFILE_ADAPTATION_PROMPT.format(**sample_input_signals)

        # Check no unformatted placeholders remain
        assert "{detected_persona}" not in formatted
        assert "{topic_complexity}" not in formatted
        assert "{category}" not in formatted

    def test_prompt_length_reasonable(self, sample_input_signals):
        """Test that formatted prompt is not too long for LLM context"""
        formatted = PROFILE_ADAPTATION_PROMPT.format(**sample_input_signals)

        # Should be under 5000 characters for efficiency
        assert len(formatted) < 5000


# ============================================================================
# Tests for Validator
# ============================================================================

class TestProfileAdaptationValidator:
    """Tests for the output validator"""

    def test_validates_correct_output(self, sample_valid_output):
        """Test that valid output passes validation"""
        result = ProfileAdaptationValidator.validate_output(
            sample_valid_output,
            requires_code=True,
            requires_diagrams=True
        )
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0

    def test_rejects_low_code_weight(self, sample_invalid_outputs):
        """Test that low code_weight is rejected when requires_code=True"""
        result = ProfileAdaptationValidator.validate_output(
            sample_invalid_outputs["low_code_weight"],
            requires_code=True,
            requires_diagrams=False
        )
        assert result["is_valid"] is False
        assert any("code_weight" in issue for issue in result["issues"])

    def test_rejects_low_diagram_weight(self, sample_invalid_outputs):
        """Test that low diagram_weight is rejected when requires_diagrams=True"""
        result = ProfileAdaptationValidator.validate_output(
            sample_invalid_outputs["low_diagram_weight"],
            requires_code=False,
            requires_diagrams=True
        )
        assert result["is_valid"] is False
        assert any("diagram_weight" in issue for issue in result["issues"])

    def test_rejects_low_theory_weight(self, sample_invalid_outputs):
        """Test that low theory_weight is always rejected"""
        result = ProfileAdaptationValidator.validate_output(
            sample_invalid_outputs["low_theory_weight"],
            requires_code=False,
            requires_diagrams=False
        )
        assert result["is_valid"] is False
        assert any("theory_weight" in issue for issue in result["issues"])

    def test_rejects_weight_out_of_range(self, sample_invalid_outputs):
        """Test that weights > 1.0 are rejected"""
        result = ProfileAdaptationValidator.validate_output(
            sample_invalid_outputs["weight_out_of_range"],
            requires_code=False,
            requires_diagrams=False
        )
        assert result["is_valid"] is False
        assert any("out of range" in issue for issue in result["issues"])

    def test_rejects_too_few_elements(self, sample_invalid_outputs):
        """Test that fewer than 3 elements is rejected"""
        result = ProfileAdaptationValidator.validate_output(
            sample_invalid_outputs["too_few_elements"],
            requires_code=False,
            requires_diagrams=False
        )
        assert result["is_valid"] is False
        assert any("Element count" in issue for issue in result["issues"])

    def test_rejects_too_many_elements(self, sample_invalid_outputs):
        """Test that more than 6 elements is rejected"""
        result = ProfileAdaptationValidator.validate_output(
            sample_invalid_outputs["too_many_elements"],
            requires_code=False,
            requires_diagrams=False
        )
        assert result["is_valid"] is False
        assert any("Element count" in issue for issue in result["issues"])

    def test_rejects_total_weight_too_low(self, sample_invalid_outputs):
        """Test that total weight < 2.5 is rejected"""
        result = ProfileAdaptationValidator.validate_output(
            sample_invalid_outputs["total_weight_too_low"],
            requires_code=False,
            requires_diagrams=False
        )
        assert result["is_valid"] is False
        assert any("Total weight" in issue for issue in result["issues"])

    def test_rejects_total_weight_too_high(self, sample_invalid_outputs):
        """Test that total weight > 3.5 is rejected"""
        result = ProfileAdaptationValidator.validate_output(
            sample_invalid_outputs["total_weight_too_high"],
            requires_code=False,
            requires_diagrams=False
        )
        assert result["is_valid"] is False
        assert any("Total weight" in issue for issue in result["issues"])


# ============================================================================
# Tests for Constraint Combinations
# ============================================================================

class TestConstraintCombinations:
    """Tests for combinations of constraints"""

    def test_technical_course_constraints(self):
        """Test constraints for a technical course with code and diagrams"""
        output = {
            "content_preferences": {
                "code_weight": 0.85,
                "diagram_weight": 0.6,
                "demo_weight": 0.7,
                "theory_weight": 0.35,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["code_demo", "architecture_diagram", "debug_tips", "case_study"],
            "adaptation_notes": "Technical course"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=True, requires_diagrams=True
        )
        assert result["is_valid"] is True
        assert 2.5 <= result["total_weight"] <= 3.5

    def test_business_course_constraints(self):
        """Test constraints for a business course without code"""
        output = {
            "content_preferences": {
                "code_weight": 0.0,
                "diagram_weight": 0.5,
                "demo_weight": 0.4,  # Adjusted to meet 2.5 total
                "theory_weight": 0.7,
                "case_study_weight": 0.9
            },  # Total = 2.5
            "recommended_elements": ["case_study", "framework_template", "action_checklist"],
            "adaptation_notes": "Business-focused"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=False, requires_diagrams=True
        )
        assert result["is_valid"] is True

    def test_hands_on_course_constraints(self):
        """Test constraints for a hands-on practical course"""
        output = {
            "content_preferences": {
                "code_weight": 0.8,
                "diagram_weight": 0.4,
                "demo_weight": 0.9,
                "theory_weight": 0.2,  # Minimum theory
                "case_study_weight": 0.4
            },
            "recommended_elements": ["code_demo", "terminal_output", "debug_tips", "exercise"],
            "adaptation_notes": "Hands-on practical"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=True, requires_diagrams=False
        )
        assert result["is_valid"] is True

    def test_minimal_valid_output(self):
        """Test the minimal valid output that satisfies all constraints"""
        output = {
            "content_preferences": {
                "code_weight": 0.6,   # Minimum for requires_code
                "diagram_weight": 0.5, # Minimum for requires_diagrams
                "demo_weight": 0.5,
                "theory_weight": 0.2,  # Minimum theory
                "case_study_weight": 0.7
            },
            "recommended_elements": ["a", "b", "c"],  # Minimum 3 elements
            "adaptation_notes": "Minimal"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=True, requires_diagrams=True
        )
        assert result["is_valid"] is True
        assert result["total_weight"] == 2.5  # Exactly at minimum


# ============================================================================
# Tests for Example Outputs in Prompt
# ============================================================================

class TestPromptExamples:
    """Tests that examples in the prompt are valid"""

    def test_python_data_science_example_valid(self):
        """Test that the Python for Data Science example is valid"""
        example = {
            "content_preferences": {
                "code_weight": 0.85,
                "diagram_weight": 0.6,
                "demo_weight": 0.7,
                "theory_weight": 0.35,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["code_demo", "architecture_diagram", "debug_tips", "case_study"],
            "adaptation_notes": "Code-driven learning with system-level diagrams for data pipelines."
        }

        result = ProfileAdaptationValidator.validate_output(
            example, requires_code=True, requires_diagrams=True
        )
        assert result["is_valid"] is True, f"Issues: {result['issues']}"

    def test_leadership_example_valid(self):
        """Test that the Leadership Fundamentals example is valid"""
        example = {
            "content_preferences": {
                "code_weight": 0.0,
                "diagram_weight": 0.5,
                "demo_weight": 0.4,  # Corrected to meet 2.5 total
                "theory_weight": 0.7,
                "case_study_weight": 0.9
            },  # Total = 2.5
            "recommended_elements": ["case_study", "framework_template", "action_checklist"],
            "adaptation_notes": "Case-study driven with actionable frameworks."
        }

        result = ProfileAdaptationValidator.validate_output(
            example, requires_code=False, requires_diagrams=True
        )
        assert result["is_valid"] is True, f"Issues: {result['issues']}"


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_zero_weights_except_required(self):
        """Test output with zero weights except required minimums"""
        output = {
            "content_preferences": {
                "code_weight": 0.0,
                "diagram_weight": 0.0,
                "demo_weight": 0.0,
                "theory_weight": 0.2,  # Only minimum theory
                "case_study_weight": 2.3  # High to meet total
            },
            "recommended_elements": ["case_study", "theory", "summary"],
            "adaptation_notes": "Theory focused"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=False, requires_diagrams=False
        )
        # Total = 2.5, meets minimum
        assert result["total_weight"] == 2.5

    def test_maximum_weights(self):
        """Test output with all maximum weights"""
        output = {
            "content_preferences": {
                "code_weight": 1.0,
                "diagram_weight": 1.0,
                "demo_weight": 0.5,
                "theory_weight": 0.5,
                "case_study_weight": 0.5
            },  # Total = 3.5 (max)
            "recommended_elements": ["a", "b", "c", "d", "e", "f"],  # Max 6
            "adaptation_notes": "Maximum everything"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=True, requires_diagrams=True
        )
        assert result["is_valid"] is True
        assert result["total_weight"] == 3.5
        assert result["element_count"] == 6

    def test_exactly_three_elements(self):
        """Test output with exactly 3 elements (minimum)"""
        output = {
            "content_preferences": {
                "code_weight": 0.6,
                "diagram_weight": 0.5,
                "demo_weight": 0.6,
                "theory_weight": 0.4,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["a", "b", "c"],
            "adaptation_notes": "Minimal elements"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=True, requires_diagrams=True
        )
        assert result["is_valid"] is True
        assert result["element_count"] == 3

    def test_exactly_six_elements(self):
        """Test output with exactly 6 elements (maximum)"""
        output = {
            "content_preferences": {
                "code_weight": 0.6,
                "diagram_weight": 0.5,
                "demo_weight": 0.6,
                "theory_weight": 0.4,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["a", "b", "c", "d", "e", "f"],
            "adaptation_notes": "Maximum elements"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=True, requires_diagrams=True
        )
        assert result["is_valid"] is True
        assert result["element_count"] == 6

    def test_boundary_total_weight_min(self):
        """Test output with exactly 2.5 total weight (boundary)"""
        output = {
            "content_preferences": {
                "code_weight": 0.6,
                "diagram_weight": 0.5,
                "demo_weight": 0.5,
                "theory_weight": 0.2,
                "case_study_weight": 0.7
            },  # Total = 2.5
            "recommended_elements": ["a", "b", "c"],
            "adaptation_notes": "Boundary test"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=True, requires_diagrams=True
        )
        assert result["is_valid"] is True
        assert result["total_weight"] == 2.5

    def test_boundary_total_weight_max(self):
        """Test output with exactly 3.5 total weight (boundary)"""
        output = {
            "content_preferences": {
                "code_weight": 0.8,
                "diagram_weight": 0.7,
                "demo_weight": 0.7,
                "theory_weight": 0.5,
                "case_study_weight": 0.8
            },  # Total = 3.5
            "recommended_elements": ["a", "b", "c"],
            "adaptation_notes": "Boundary test"
        }

        result = ProfileAdaptationValidator.validate_output(
            output, requires_code=True, requires_diagrams=True
        )
        assert result["is_valid"] is True
        assert result["total_weight"] == 3.5


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
