"""
Unit tests for ELEMENT_SUGGESTION_PROMPT

Tests the agentic prompt structure, decision rules, and output validation
for the Technical Lesson Composition Agent.
"""

import pytest
import json
import re
from typing import Dict, Any, List
from dataclasses import dataclass

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
ELEMENT_SUGGESTION_PROMPT = prompts_module.ELEMENT_SUGGESTION_PROMPT


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_input_signals():
    """Sample input signals for prompt formatting"""
    return {
        "topic": "Building Microservices with Go",
        "category": "tech",
        "code_weight": 0.85,
        "diagram_weight": 0.7,
        "demo_weight": 0.6,
        "available_elements": """- code_demo: Code Demo - Live coding demonstration
- terminal_output: Terminal Output - Command line examples
- architecture_diagram: Architecture Diagram - System architecture visualization
- debug_tips: Debug Tips - Common debugging strategies
- case_study: Case Study - Real-world examples
- concept_intro: Concept Introduction - Core concept explanation
- voiceover: Voiceover - Narrative explanation
- conclusion: Conclusion - Summary and key takeaways""",
        "outline_structure": """Section 1: Introduction
- lec_001: Introduction to Microservices
- lec_002: Go Language Basics

Section 2: Core Concepts
- lec_003: Service Design Patterns
- lec_004: API Design with gRPC
- lec_005: Database Integration"""
    }


@pytest.fixture
def sample_valid_output():
    """Sample valid LLM output that respects all constraints"""
    return {
        "element_mapping": {
            "lec_001": ["concept_intro", "architecture_diagram", "voiceover"],
            "lec_002": ["concept_intro", "code_demo", "terminal_output", "voiceover"],
            "lec_003": ["concept_intro", "architecture_diagram", "case_study", "voiceover"],
            "lec_004": ["code_demo", "terminal_output", "architecture_diagram", "debug_tips"],
            "lec_005": ["code_demo", "terminal_output", "architecture_diagram", "case_study", "debug_tips"]
        },
        "reasoning": "Progressive complexity: early lectures focus on concepts, later lectures emphasize hands-on implementation."
    }


@pytest.fixture
def available_elements_list():
    """List of available elements for validation"""
    return [
        "code_demo", "terminal_output", "architecture_diagram", "debug_tips",
        "case_study", "concept_intro", "voiceover", "conclusion",
        "framework_template", "action_checklist"
    ]


@pytest.fixture
def sample_invalid_outputs():
    """Collection of invalid outputs that violate constraints"""
    return {
        "too_few_elements": {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover"],  # Only 2, should be 3-5
                "lec_002": ["code_demo", "terminal_output", "voiceover"],
            },
            "reasoning": "..."
        },
        "too_many_elements": {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "code_demo", "terminal_output",
                           "architecture_diagram", "debug_tips"],  # 6 elements, should be 3-5
            },
            "reasoning": "..."
        },
        "invalid_element": {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "nonexistent_element"],  # Invalid element
            },
            "reasoning": "..."
        },
        "missing_reasoning": {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "code_demo"],
            }
            # Missing "reasoning" field
        },
        "empty_mapping": {
            "element_mapping": {},
            "reasoning": "No lectures provided"
        },
    }


# ============================================================================
# Element Suggestion Validator
# ============================================================================

class ElementSuggestionValidator:
    """Validates outputs against ELEMENT_SUGGESTION_PROMPT constraints"""

    def __init__(self, available_elements: List[str]):
        self.available_elements = available_elements

    def validate_element_count(self, elements: List[str]) -> bool:
        """Each lecture must have 3-5 elements"""
        return 3 <= len(elements) <= 5

    def validate_elements_exist(self, elements: List[str]) -> List[str]:
        """All elements must exist in available_elements list"""
        invalid = [el for el in elements if el not in self.available_elements]
        return invalid

    def validate_code_weight_constraint(
        self,
        element_mapping: Dict[str, List[str]],
        code_weight: float
    ) -> bool:
        """If code_weight >= 0.7, at least 50% of lectures must include code_demo"""
        if code_weight < 0.7:
            return True

        total_lectures = len(element_mapping)
        if total_lectures == 0:
            return True

        lectures_with_code = sum(
            1 for elements in element_mapping.values()
            if "code_demo" in elements
        )
        return (lectures_with_code / total_lectures) >= 0.5

    def validate_diagram_weight_constraint(
        self,
        element_mapping: Dict[str, List[str]],
        diagram_weight: float
    ) -> bool:
        """If diagram_weight >= 0.6, at least 40% of lectures must include a diagram element"""
        if diagram_weight < 0.6:
            return True

        total_lectures = len(element_mapping)
        if total_lectures == 0:
            return True

        diagram_elements = ["architecture_diagram", "body_diagram", "data_pipeline_diagram"]
        lectures_with_diagram = sum(
            1 for elements in element_mapping.values()
            if any(el in elements for el in diagram_elements)
        )
        return (lectures_with_diagram / total_lectures) >= 0.4

    def validate_element_distribution(
        self,
        element_mapping: Dict[str, List[str]],
        code_weight: float = 0.5,
        diagram_weight: float = 0.5
    ) -> List[str]:
        """No single element should appear in more than 70% of lectures (except allowed)"""
        # Common elements that can appear in all lectures
        allowed_elements = ["voiceover", "concept_intro", "conclusion"]

        # If code_weight >= 0.7, code_demo and terminal_output are expected in many lectures
        if code_weight >= 0.7:
            allowed_elements.extend(["code_demo", "terminal_output"])

        # If diagram_weight >= 0.6, diagram elements are expected in many lectures
        if diagram_weight >= 0.6:
            allowed_elements.extend(["architecture_diagram", "body_diagram", "data_pipeline_diagram"])

        total_lectures = len(element_mapping)
        if total_lectures == 0:
            return []

        element_counts: Dict[str, int] = {}
        for elements in element_mapping.values():
            for el in elements:
                element_counts[el] = element_counts.get(el, 0) + 1

        overused = []
        for el, count in element_counts.items():
            if el not in allowed_elements and (count / total_lectures) > 0.7:
                overused.append(f"{el} appears in {count}/{total_lectures} lectures (>70%)")

        return overused

    def validate_output(
        self,
        output: Dict[str, Any],
        code_weight: float = 0.5,
        diagram_weight: float = 0.5
    ) -> Dict[str, Any]:
        """Validate full output against all constraints"""
        issues = []

        element_mapping = output.get("element_mapping", {})
        reasoning = output.get("reasoning")

        # Check reasoning exists
        if not reasoning:
            issues.append("Missing 'reasoning' field")

        # Check each lecture
        for lecture_id, elements in element_mapping.items():
            # Check element count
            if not self.validate_element_count(elements):
                issues.append(f"{lecture_id}: has {len(elements)} elements (must be 3-5)")

            # Check elements exist
            invalid_elements = self.validate_elements_exist(elements)
            if invalid_elements:
                issues.append(f"{lecture_id}: invalid elements {invalid_elements}")

        # Check code weight constraint
        if not self.validate_code_weight_constraint(element_mapping, code_weight):
            issues.append(f"code_weight={code_weight} requires 50%+ lectures with code_demo")

        # Check diagram weight constraint
        if not self.validate_diagram_weight_constraint(element_mapping, diagram_weight):
            issues.append(f"diagram_weight={diagram_weight} requires 40%+ lectures with diagram")

        # Check element distribution (pass weights to allow expected high-frequency elements)
        overused = self.validate_element_distribution(element_mapping, code_weight, diagram_weight)
        issues.extend(overused)

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "lecture_count": len(element_mapping),
            "has_reasoning": bool(reasoning),
        }


# ============================================================================
# Tests for Prompt Structure
# ============================================================================

class TestPromptStructure:
    """Tests for the prompt structure and format"""

    def test_prompt_contains_role_definition(self):
        """Test that prompt defines the agent role"""
        assert "Technical Lesson Composition Agent" in ELEMENT_SUGGESTION_PROMPT
        assert "operating autonomously" in ELEMENT_SUGGESTION_PROMPT

    def test_prompt_contains_context_section(self):
        """Test that prompt contains context about Viralify"""
        assert "## CONTEXT" in ELEMENT_SUGGESTION_PROMPT
        assert "Viralify" in ELEMENT_SUGGESTION_PROMPT
        assert "video courses" in ELEMENT_SUGGESTION_PROMPT

    def test_prompt_contains_inputs_section(self):
        """Test that prompt documents input signals"""
        assert "## INPUTS" in ELEMENT_SUGGESTION_PROMPT
        assert "COURSE SIGNALS" in ELEMENT_SUGGESTION_PROMPT
        assert "GLOBAL CONTENT PREFERENCES" in ELEMENT_SUGGESTION_PROMPT
        assert "SYSTEM CAPABILITIES" in ELEMENT_SUGGESTION_PROMPT
        assert "COURSE STRUCTURE" in ELEMENT_SUGGESTION_PROMPT

    def test_prompt_contains_all_placeholders(self):
        """Test that prompt contains all required placeholders"""
        required_placeholders = [
            "{topic}",
            "{category}",
            "{code_weight}",
            "{diagram_weight}",
            "{demo_weight}",
            "{available_elements}",
            "{outline_structure}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in ELEMENT_SUGGESTION_PROMPT, f"Missing placeholder: {placeholder}"

    def test_prompt_contains_agent_responsibilities(self):
        """Test that prompt defines agent responsibilities"""
        assert "## AGENT RESPONSIBILITIES" in ELEMENT_SUGGESTION_PROMPT
        assert "Analyze the lecture's technical intent" in ELEMENT_SUGGESTION_PROMPT
        assert "Select the 3 to 5 most relevant" in ELEMENT_SUGGESTION_PROMPT

    def test_prompt_contains_decision_constraints(self):
        """Test that prompt contains hard constraints section"""
        assert "## DECISION CONSTRAINTS" in ELEMENT_SUGGESTION_PROMPT
        assert "HARD RULES" in ELEMENT_SUGGESTION_PROMPT

    def test_prompt_contains_self_validation(self):
        """Test that prompt includes self-validation section"""
        assert "## SELF-VALIDATION" in ELEMENT_SUGGESTION_PROMPT
        assert "Verify that" in ELEMENT_SUGGESTION_PROMPT

    def test_prompt_contains_examples(self):
        """Test that prompt includes examples"""
        assert "## EXAMPLES" in ELEMENT_SUGGESTION_PROMPT
        assert "Kubernetes" in ELEMENT_SUGGESTION_PROMPT

    def test_prompt_contains_output_contract(self):
        """Test that prompt defines JSON output format"""
        assert "## OUTPUT CONTRACT" in ELEMENT_SUGGESTION_PROMPT
        assert "valid JSON only" in ELEMENT_SUGGESTION_PROMPT
        assert "element_mapping" in ELEMENT_SUGGESTION_PROMPT
        assert "reasoning" in ELEMENT_SUGGESTION_PROMPT

    def test_prompt_mentions_upstream_agent(self):
        """Test that prompt mentions the upstream agent"""
        assert "Curriculum Optimization Agent" in ELEMENT_SUGGESTION_PROMPT
        assert "downstream" in ELEMENT_SUGGESTION_PROMPT


# ============================================================================
# Tests for Prompt Constraints
# ============================================================================

class TestPromptConstraints:
    """Tests for the constraints defined in the prompt"""

    def test_element_count_constraint_documented(self):
        """Test that element count constraint is documented"""
        assert "3 and 5 elements" in ELEMENT_SUGGESTION_PROMPT or "3 to 5" in ELEMENT_SUGGESTION_PROMPT

    def test_elements_must_exist_constraint_documented(self):
        """Test that elements must exist constraint is documented"""
        assert "MUST exist in the available_elements list" in ELEMENT_SUGGESTION_PROMPT

    def test_code_weight_threshold_documented(self):
        """Test that code weight threshold is documented"""
        assert "code_weight >= 0.7" in ELEMENT_SUGGESTION_PROMPT
        assert "50%" in ELEMENT_SUGGESTION_PROMPT
        assert "code_demo" in ELEMENT_SUGGESTION_PROMPT

    def test_diagram_weight_threshold_documented(self):
        """Test that diagram weight threshold is documented"""
        assert "diagram_weight >= 0.6" in ELEMENT_SUGGESTION_PROMPT
        assert "40%" in ELEMENT_SUGGESTION_PROMPT

    def test_distribution_constraint_documented(self):
        """Test that element distribution constraint is documented"""
        assert "70%" in ELEMENT_SUGGESTION_PROMPT

    def test_progressive_complexity_documented(self):
        """Test that progressive complexity is mentioned"""
        assert "Early lectures" in ELEMENT_SUGGESTION_PROMPT
        assert "later lectures" in ELEMENT_SUGGESTION_PROMPT
        assert "theory" in ELEMENT_SUGGESTION_PROMPT
        assert "practice" in ELEMENT_SUGGESTION_PROMPT


# ============================================================================
# Tests for Prompt Formatting
# ============================================================================

class TestPromptFormatting:
    """Tests for prompt formatting with input signals"""

    def test_prompt_formats_correctly(self, sample_input_signals):
        """Test that prompt can be formatted with all signals"""
        formatted = ELEMENT_SUGGESTION_PROMPT.format(**sample_input_signals)

        assert "Building Microservices with Go" in formatted
        assert "tech" in formatted
        assert "0.85" in formatted
        assert "code_demo" in formatted
        assert "lec_001" in formatted

    def test_formatted_prompt_no_placeholders_remain(self, sample_input_signals):
        """Test that no placeholders remain after formatting"""
        formatted = ELEMENT_SUGGESTION_PROMPT.format(**sample_input_signals)

        assert "{topic}" not in formatted
        assert "{category}" not in formatted
        assert "{code_weight}" not in formatted
        assert "{available_elements}" not in formatted
        assert "{outline_structure}" not in formatted

    def test_prompt_length_reasonable(self, sample_input_signals):
        """Test that formatted prompt is not too long for LLM context"""
        formatted = ELEMENT_SUGGESTION_PROMPT.format(**sample_input_signals)

        # Should be under 6000 characters for efficiency
        assert len(formatted) < 6000


# ============================================================================
# Tests for Validator
# ============================================================================

class TestElementSuggestionValidator:
    """Tests for the output validator"""

    @pytest.fixture
    def validator(self, available_elements_list):
        return ElementSuggestionValidator(available_elements_list)

    def test_validates_correct_output(self, validator, sample_valid_output):
        """Test that valid output passes validation"""
        result = validator.validate_output(
            sample_valid_output,
            code_weight=0.85,
            diagram_weight=0.7
        )
        assert result["is_valid"] is True, f"Issues: {result['issues']}"

    def test_rejects_too_few_elements(self, validator, sample_invalid_outputs):
        """Test that fewer than 3 elements is rejected"""
        result = validator.validate_output(sample_invalid_outputs["too_few_elements"])
        assert result["is_valid"] is False
        assert any("2 elements" in issue for issue in result["issues"])

    def test_rejects_too_many_elements(self, validator, sample_invalid_outputs):
        """Test that more than 5 elements is rejected"""
        result = validator.validate_output(sample_invalid_outputs["too_many_elements"])
        assert result["is_valid"] is False
        assert any("6 elements" in issue for issue in result["issues"])

    def test_rejects_invalid_elements(self, validator, sample_invalid_outputs):
        """Test that non-existent elements are rejected"""
        result = validator.validate_output(sample_invalid_outputs["invalid_element"])
        assert result["is_valid"] is False
        assert any("invalid elements" in issue for issue in result["issues"])

    def test_rejects_missing_reasoning(self, validator, sample_invalid_outputs):
        """Test that missing reasoning is rejected"""
        result = validator.validate_output(sample_invalid_outputs["missing_reasoning"])
        assert result["is_valid"] is False
        assert any("reasoning" in issue for issue in result["issues"])

    def test_code_weight_constraint_enforced(self, validator):
        """Test that code_weight >= 0.7 requires 50%+ code_demo"""
        # Only 1 out of 4 lectures has code_demo (25% < 50%)
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "architecture_diagram"],
                "lec_002": ["concept_intro", "voiceover", "architecture_diagram"],
                "lec_003": ["concept_intro", "voiceover", "architecture_diagram"],
                "lec_004": ["code_demo", "voiceover", "terminal_output"],
            },
            "reasoning": "Test"
        }
        result = validator.validate_output(output, code_weight=0.8)
        assert result["is_valid"] is False
        assert any("code_weight" in issue for issue in result["issues"])

    def test_diagram_weight_constraint_enforced(self, validator):
        """Test that diagram_weight >= 0.6 requires 40%+ diagrams"""
        # Only 1 out of 5 lectures has diagram (20% < 40%)
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "code_demo"],
                "lec_002": ["concept_intro", "voiceover", "code_demo"],
                "lec_003": ["concept_intro", "voiceover", "code_demo"],
                "lec_004": ["concept_intro", "voiceover", "code_demo"],
                "lec_005": ["architecture_diagram", "voiceover", "code_demo"],
            },
            "reasoning": "Test"
        }
        result = validator.validate_output(output, diagram_weight=0.7)
        assert result["is_valid"] is False
        assert any("diagram_weight" in issue for issue in result["issues"])

    def test_code_weight_constraint_passes_when_met(self, validator):
        """Test that code_weight constraint passes when met"""
        # 3 out of 4 lectures have code_demo (75% >= 50%)
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "architecture_diagram"],
                "lec_002": ["code_demo", "voiceover", "terminal_output"],
                "lec_003": ["code_demo", "voiceover", "architecture_diagram"],
                "lec_004": ["code_demo", "voiceover", "terminal_output"],
            },
            "reasoning": "Test"
        }
        result = validator.validate_output(output, code_weight=0.8, diagram_weight=0.5)
        # Should pass code_weight constraint
        assert not any("code_weight" in issue for issue in result["issues"])


# ============================================================================
# Tests for Element Distribution
# ============================================================================

class TestElementDistribution:
    """Tests for element distribution validation"""

    @pytest.fixture
    def validator(self, available_elements_list):
        return ElementSuggestionValidator(available_elements_list)

    def test_detects_overused_element(self, validator):
        """Test that elements used in >70% of lectures are flagged"""
        # debug_tips in 4 out of 4 lectures (100% > 70%)
        output = {
            "element_mapping": {
                "lec_001": ["debug_tips", "voiceover", "code_demo"],
                "lec_002": ["debug_tips", "voiceover", "code_demo"],
                "lec_003": ["debug_tips", "voiceover", "architecture_diagram"],
                "lec_004": ["debug_tips", "voiceover", "terminal_output"],
            },
            "reasoning": "Test"
        }
        result = validator.validate_output(output)
        assert any("debug_tips" in issue and ">70%" in issue for issue in result["issues"])

    def test_allows_common_elements_in_all_lectures(self, validator):
        """Test that common elements (voiceover, concept_intro) can be in all lectures"""
        # voiceover and concept_intro in all lectures is allowed
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "code_demo"],
                "lec_002": ["concept_intro", "voiceover", "architecture_diagram"],
                "lec_003": ["concept_intro", "voiceover", "terminal_output"],
                "lec_004": ["concept_intro", "voiceover", "case_study"],
            },
            "reasoning": "Test"
        }
        result = validator.validate_output(output, code_weight=0.5, diagram_weight=0.5)
        # voiceover and concept_intro should not be flagged
        assert not any("voiceover" in issue for issue in result["issues"])
        assert not any("concept_intro" in issue for issue in result["issues"])


# ============================================================================
# Tests for Example in Prompt
# ============================================================================

class TestPromptExample:
    """Tests that the example in the prompt is valid"""

    @pytest.fixture
    def validator(self):
        elements = [
            "concept_intro", "architecture_diagram", "voiceover",
            "code_demo", "terminal_output", "debug_tips", "case_study"
        ]
        return ElementSuggestionValidator(elements)

    def test_kubernetes_example_valid(self, validator):
        """Test that the Kubernetes example in the prompt is valid"""
        example = {
            "element_mapping": {
                "lec_001_intro": ["concept_intro", "architecture_diagram", "voiceover"],
                "lec_002_pods": ["concept_intro", "code_demo", "terminal_output", "architecture_diagram"],
                "lec_003_deployments": ["code_demo", "terminal_output", "debug_tips", "architecture_diagram"],
                "lec_004_services": ["concept_intro", "code_demo", "architecture_diagram", "case_study"]
            },
            "reasoning": "Progressive complexity: intro focuses on concepts, subsequent lectures emphasize hands-on code with architecture context."
        }

        result = validator.validate_output(
            example,
            code_weight=0.8,
            diagram_weight=0.7
        )
        assert result["is_valid"] is True, f"Issues: {result['issues']}"

    def test_kubernetes_example_element_counts(self, validator):
        """Test that each lecture in example has 3-5 elements"""
        example_elements = {
            "lec_001_intro": ["concept_intro", "architecture_diagram", "voiceover"],  # 3
            "lec_002_pods": ["concept_intro", "code_demo", "terminal_output", "architecture_diagram"],  # 4
            "lec_003_deployments": ["code_demo", "terminal_output", "debug_tips", "architecture_diagram"],  # 4
            "lec_004_services": ["concept_intro", "code_demo", "architecture_diagram", "case_study"]  # 4
        }

        for lecture_id, elements in example_elements.items():
            assert validator.validate_element_count(elements), \
                f"{lecture_id} has {len(elements)} elements (should be 3-5)"


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @pytest.fixture
    def validator(self, available_elements_list):
        return ElementSuggestionValidator(available_elements_list)

    def test_exactly_three_elements(self, validator):
        """Test output with exactly 3 elements per lecture (minimum)"""
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "code_demo"],
                "lec_002": ["concept_intro", "voiceover", "architecture_diagram"],
            },
            "reasoning": "Minimal elements"
        }
        result = validator.validate_output(output)
        # Should not have element count issues
        assert not any("elements" in issue and "must be" in issue for issue in result["issues"])

    def test_exactly_five_elements(self, validator):
        """Test output with exactly 5 elements per lecture (maximum)"""
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "code_demo", "terminal_output", "debug_tips"],
            },
            "reasoning": "Maximum elements"
        }
        result = validator.validate_output(output)
        # Should not have element count issues
        assert not any("elements" in issue and "must be" in issue for issue in result["issues"])

    def test_single_lecture(self, validator):
        """Test output with single lecture"""
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "code_demo"],
            },
            "reasoning": "Single lecture course"
        }
        result = validator.validate_output(output, code_weight=0.8)
        # Single lecture with code_demo meets 50% requirement (1/1 = 100%)
        assert not any("code_weight" in issue for issue in result["issues"])

    def test_empty_element_mapping(self, validator):
        """Test that empty element mapping is handled"""
        output = {
            "element_mapping": {},
            "reasoning": "No lectures"
        }
        result = validator.validate_output(output)
        assert result["lecture_count"] == 0

    def test_code_weight_below_threshold(self, validator):
        """Test that code_weight below 0.7 doesn't require code_demo"""
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "architecture_diagram"],
                "lec_002": ["concept_intro", "voiceover", "architecture_diagram"],
            },
            "reasoning": "No code needed"
        }
        # code_weight = 0.5 < 0.7, so no code_demo requirement
        result = validator.validate_output(output, code_weight=0.5)
        assert not any("code_weight" in issue for issue in result["issues"])

    def test_diagram_weight_below_threshold(self, validator):
        """Test that diagram_weight below 0.6 doesn't require diagrams"""
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "code_demo"],
                "lec_002": ["concept_intro", "voiceover", "code_demo"],
            },
            "reasoning": "No diagrams needed"
        }
        # diagram_weight = 0.5 < 0.6, so no diagram requirement
        result = validator.validate_output(output, diagram_weight=0.5)
        assert not any("diagram_weight" in issue for issue in result["issues"])


# ============================================================================
# Tests for Constraint Combinations
# ============================================================================

class TestConstraintCombinations:
    """Tests for combinations of constraints"""

    @pytest.fixture
    def validator(self, available_elements_list):
        return ElementSuggestionValidator(available_elements_list)

    def test_high_code_and_diagram_weights(self, validator):
        """Test with both high code and diagram weights"""
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "code_demo", "architecture_diagram"],
                "lec_002": ["code_demo", "terminal_output", "architecture_diagram"],
                "lec_003": ["code_demo", "architecture_diagram", "debug_tips"],
                "lec_004": ["concept_intro", "code_demo", "architecture_diagram"],
            },
            "reasoning": "High code and diagram emphasis"
        }
        result = validator.validate_output(
            output,
            code_weight=0.9,
            diagram_weight=0.8
        )
        # 4/4 have code_demo (100% >= 50%) ✓
        # 4/4 have diagram (100% >= 40%) ✓
        assert result["is_valid"] is True, f"Issues: {result['issues']}"

    def test_implementation_focused_course(self, validator):
        """Test a practical implementation-focused course"""
        output = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "architecture_diagram"],
                "lec_002": ["code_demo", "terminal_output", "voiceover"],
                "lec_003": ["code_demo", "terminal_output", "debug_tips"],
                "lec_004": ["code_demo", "terminal_output", "case_study"],
                "lec_005": ["code_demo", "terminal_output", "debug_tips", "case_study"],
            },
            "reasoning": "Progressive from concepts to heavy implementation"
        }
        result = validator.validate_output(
            output,
            code_weight=0.85,
            diagram_weight=0.4  # Below threshold
        )
        # 4/5 have code_demo (80% >= 50%) ✓
        # diagram_weight < 0.6, no requirement ✓
        assert result["is_valid"] is True, f"Issues: {result['issues']}"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
