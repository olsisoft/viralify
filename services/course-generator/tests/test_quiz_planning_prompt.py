"""
Unit tests for QUIZ_PLANNING_PROMPT

Tests the prompt structure, constraints, and validation rules
without making actual LLM calls.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Any


# ============================================================================
# Direct import to avoid dependency chain
# ============================================================================

def import_module_from_file(module_name: str, file_path: str):
    """Import a module directly from file path to avoid dependency issues."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get the prompts module
prompts_path = Path(__file__).parent.parent / "agents" / "pedagogical_prompts.py"
prompts_module = import_module_from_file("pedagogical_prompts", str(prompts_path))
QUIZ_PLANNING_PROMPT = prompts_module.QUIZ_PLANNING_PROMPT


# ============================================================================
# QuizPlanningValidator
# ============================================================================

class QuizPlanningValidator:
    """Validates LLM output against QUIZ_PLANNING_PROMPT constraints."""

    VALID_QUIZ_TYPES = ["lecture_check", "section_review", "final_assessment"]
    VALID_DIFFICULTIES = ["easy", "medium", "hard"]
    VALID_QUESTION_TYPES = [
        "multiple_choice", "true_false", "fill_blank",
        "code_review", "code_completion", "debug_exercise",
        "diagram_interpretation", "matching", "ordering",
        "scenario_based"
    ]

    # Question count ranges by quiz type
    QUESTION_COUNT_RANGES = {
        "lecture_check": (3, 5),
        "section_review": (5, 8),
        "final_assessment": (8, 15),
    }

    def __init__(self, total_lectures: int = 5):
        self.total_lectures = total_lectures

    def validate_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the full output structure."""
        issues = []

        # Check required fields
        if "quiz_placement" not in output:
            issues.append("Missing 'quiz_placement' field")
        if "total_quiz_count" not in output:
            issues.append("Missing 'total_quiz_count' field")
        if "coverage_analysis" not in output:
            issues.append("Missing 'coverage_analysis' field")

        if issues:
            return {"is_valid": False, "issues": issues}

        # Validate quiz_placement array
        placement_issues = self.validate_quiz_placement(output["quiz_placement"])
        issues.extend(placement_issues)

        # Validate total_quiz_count matches array length
        if output["total_quiz_count"] != len(output["quiz_placement"]):
            issues.append(
                f"total_quiz_count ({output['total_quiz_count']}) doesn't match "
                f"quiz_placement length ({len(output['quiz_placement'])})"
            )

        # Validate coverage_analysis is not empty
        if not output.get("coverage_analysis") or len(output["coverage_analysis"]) < 10:
            issues.append("coverage_analysis is too short or empty")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }

    def validate_quiz_placement(self, placements: List[Dict]) -> List[str]:
        """Validate the quiz_placement array."""
        issues = []

        for i, quiz in enumerate(placements):
            prefix = f"Quiz {i + 1}"

            # Check required fields
            if "lecture_id" not in quiz:
                issues.append(f"{prefix}: missing 'lecture_id'")
            if "quiz_type" not in quiz:
                issues.append(f"{prefix}: missing 'quiz_type'")
            if "difficulty" not in quiz:
                issues.append(f"{prefix}: missing 'difficulty'")
            if "question_count" not in quiz:
                issues.append(f"{prefix}: missing 'question_count'")
            if "topics_covered" not in quiz:
                issues.append(f"{prefix}: missing 'topics_covered'")

            # Validate quiz_type
            quiz_type = quiz.get("quiz_type", "")
            if quiz_type not in self.VALID_QUIZ_TYPES:
                issues.append(f"{prefix}: invalid quiz_type '{quiz_type}'")

            # Validate difficulty
            difficulty = quiz.get("difficulty", "")
            if difficulty not in self.VALID_DIFFICULTIES:
                issues.append(f"{prefix}: invalid difficulty '{difficulty}'")

            # Validate question_count
            question_count = quiz.get("question_count", 0)
            if quiz_type in self.QUESTION_COUNT_RANGES:
                min_count, max_count = self.QUESTION_COUNT_RANGES[quiz_type]
                if not (min_count <= question_count <= max_count):
                    issues.append(
                        f"{prefix}: question_count {question_count} out of range "
                        f"for {quiz_type} (expected {min_count}-{max_count})"
                    )

            # Validate question_count never exceeds 15
            if question_count > 15:
                issues.append(f"{prefix}: question_count {question_count} exceeds maximum 15")

            # Validate topics_covered is not empty
            topics = quiz.get("topics_covered", [])
            if not topics:
                issues.append(f"{prefix}: topics_covered is empty")
            elif len(topics) > 5:
                issues.append(f"{prefix}: topics_covered has {len(topics)} topics (max 5)")

            # Validate question_types if present
            question_types = quiz.get("question_types", [])
            for qtype in question_types:
                if qtype not in self.VALID_QUESTION_TYPES:
                    issues.append(f"{prefix}: invalid question_type '{qtype}'")

        return issues

    def validate_frequency_compliance(
        self,
        placements: List[Dict],
        frequency: str,
        section_count: int,
        lecture_count: int
    ) -> List[str]:
        """Validate that quiz placement respects frequency setting."""
        issues = []

        quiz_types = [p.get("quiz_type") for p in placements]

        if frequency == "per_lecture":
            # Should have lecture_check after each lecture
            lecture_checks = quiz_types.count("lecture_check")
            if lecture_checks < lecture_count * 0.8:  # Allow some flexibility
                issues.append(
                    f"per_lecture frequency: expected ~{lecture_count} lecture_checks, "
                    f"got {lecture_checks}"
                )

        elif frequency == "per_section":
            # Should have section_review at end of each section
            section_reviews = quiz_types.count("section_review")
            # May also have a final_assessment
            if section_reviews < section_count - 1:  # -1 because last might be final
                issues.append(
                    f"per_section frequency: expected ~{section_count} section_reviews, "
                    f"got {section_reviews}"
                )

        elif frequency == "end_only":
            # Should have exactly one final_assessment
            if len(placements) != 1:
                issues.append(f"end_only frequency: expected 1 quiz, got {len(placements)}")
            if placements and placements[0].get("quiz_type") != "final_assessment":
                issues.append("end_only frequency: quiz should be final_assessment")

        return issues

    def validate_difficulty_progression(
        self,
        placements: List[Dict],
        lecture_positions: Dict[str, float]  # lecture_id -> position (0-1)
    ) -> List[str]:
        """Validate that difficulty progresses appropriately."""
        issues = []

        for quiz in placements:
            lecture_id = quiz.get("lecture_id", "")
            difficulty = quiz.get("difficulty", "")
            position = lecture_positions.get(lecture_id, 0.5)

            # First 30%: should be easy or medium
            if position < 0.3 and difficulty == "hard":
                issues.append(
                    f"Quiz at {lecture_id} (position {position:.0%}): "
                    f"'hard' difficulty too early in course"
                )

            # final_assessment should always be hard
            if quiz.get("quiz_type") == "final_assessment" and difficulty != "hard":
                issues.append(
                    f"final_assessment should have 'hard' difficulty, got '{difficulty}'"
                )

        return issues


# ============================================================================
# Tests for Prompt Structure
# ============================================================================

class TestPromptStructure:
    """Tests for the overall prompt structure."""

    def test_prompt_has_role_definition(self):
        """Test that prompt defines the agent role."""
        assert "Quiz Assessment Planning Agent" in QUIZ_PLANNING_PROMPT
        assert "autonomously" in QUIZ_PLANNING_PROMPT

    def test_prompt_has_context_section(self):
        """Test that prompt has context section."""
        assert "## CONTEXT" in QUIZ_PLANNING_PROMPT
        assert "Viralify" in QUIZ_PLANNING_PROMPT

    def test_prompt_has_inputs_section(self):
        """Test that prompt has inputs section."""
        assert "## INPUTS" in QUIZ_PLANNING_PROMPT
        assert "{quiz_enabled}" in QUIZ_PLANNING_PROMPT
        assert "{quiz_frequency}" in QUIZ_PLANNING_PROMPT
        assert "{questions_per_quiz}" in QUIZ_PLANNING_PROMPT

    def test_prompt_has_responsibilities_section(self):
        """Test that prompt has agent responsibilities."""
        assert "## AGENT RESPONSIBILITIES" in QUIZ_PLANNING_PROMPT

    def test_prompt_has_decision_rules_section(self):
        """Test that prompt has decision rules."""
        assert "## DECISION RULES" in QUIZ_PLANNING_PROMPT
        assert "HARD CONSTRAINTS" in QUIZ_PLANNING_PROMPT

    def test_prompt_has_self_validation_section(self):
        """Test that prompt has self-validation section."""
        assert "## SELF-VALIDATION" in QUIZ_PLANNING_PROMPT

    def test_prompt_has_examples_section(self):
        """Test that prompt has examples."""
        assert "## EXAMPLES" in QUIZ_PLANNING_PROMPT

    def test_prompt_has_output_contract(self):
        """Test that prompt has output contract."""
        assert "## OUTPUT CONTRACT" in QUIZ_PLANNING_PROMPT
        assert "valid JSON only" in QUIZ_PLANNING_PROMPT


# ============================================================================
# Tests for Placeholders
# ============================================================================

class TestPromptPlaceholders:
    """Tests for prompt placeholders."""

    def test_quiz_enabled_placeholder(self):
        """Test quiz_enabled placeholder exists."""
        assert "{quiz_enabled}" in QUIZ_PLANNING_PROMPT

    def test_quiz_frequency_placeholder(self):
        """Test quiz_frequency placeholder exists."""
        assert "{quiz_frequency}" in QUIZ_PLANNING_PROMPT

    def test_questions_per_quiz_placeholder(self):
        """Test questions_per_quiz placeholder exists."""
        assert "{questions_per_quiz}" in QUIZ_PLANNING_PROMPT

    def test_outline_structure_placeholder(self):
        """Test outline_structure placeholder exists."""
        assert "{outline_structure}" in QUIZ_PLANNING_PROMPT

    def test_section_objectives_placeholder(self):
        """Test section_objectives placeholder exists."""
        assert "{section_objectives}" in QUIZ_PLANNING_PROMPT

    def test_all_placeholders_can_be_formatted(self):
        """Test that all placeholders can be formatted."""
        formatted = QUIZ_PLANNING_PROMPT.format(
            quiz_enabled=True,
            quiz_frequency="per_section",
            questions_per_quiz=5,
            outline_structure="Section: Intro\n  - lec_001: Welcome",
            section_objectives="Intro: Learn basics",
        )
        assert "{" not in formatted or "{{" in QUIZ_PLANNING_PROMPT


# ============================================================================
# Tests for Frequency Rules
# ============================================================================

class TestFrequencyRules:
    """Tests for frequency rule documentation in prompt."""

    def test_per_lecture_rule(self):
        """Test per_lecture frequency is documented."""
        assert "per_lecture" in QUIZ_PLANNING_PROMPT
        assert "lecture_check" in QUIZ_PLANNING_PROMPT

    def test_per_section_rule(self):
        """Test per_section frequency is documented."""
        assert "per_section" in QUIZ_PLANNING_PROMPT
        assert "section_review" in QUIZ_PLANNING_PROMPT

    def test_end_only_rule(self):
        """Test end_only frequency is documented."""
        assert "end_only" in QUIZ_PLANNING_PROMPT
        assert "final_assessment" in QUIZ_PLANNING_PROMPT

    def test_custom_rule(self):
        """Test custom frequency is documented."""
        assert "custom" in QUIZ_PLANNING_PROMPT


# ============================================================================
# Tests for Question Count Rules
# ============================================================================

class TestQuestionCountRules:
    """Tests for question count rules in prompt."""

    def test_lecture_check_count_range(self):
        """Test lecture_check question count range is documented."""
        assert "3-5" in QUIZ_PLANNING_PROMPT or "3 to 5" in QUIZ_PLANNING_PROMPT

    def test_section_review_count_range(self):
        """Test section_review question count range is documented."""
        assert "5-8" in QUIZ_PLANNING_PROMPT or "5 to 8" in QUIZ_PLANNING_PROMPT

    def test_final_assessment_count_range(self):
        """Test final_assessment question count range is documented."""
        assert "8-15" in QUIZ_PLANNING_PROMPT or "8 to 15" in QUIZ_PLANNING_PROMPT

    def test_max_questions_rule(self):
        """Test maximum 15 questions rule is documented."""
        assert "15" in QUIZ_PLANNING_PROMPT
        assert "NEVER exceed" in QUIZ_PLANNING_PROMPT or "never exceed" in QUIZ_PLANNING_PROMPT.lower()


# ============================================================================
# Tests for Difficulty Progression Rules
# ============================================================================

class TestDifficultyProgressionRules:
    """Tests for difficulty progression rules in prompt."""

    def test_first_30_percent_rule(self):
        """Test first 30% difficulty rule is documented."""
        assert "30%" in QUIZ_PLANNING_PROMPT
        assert "easy" in QUIZ_PLANNING_PROMPT

    def test_middle_40_percent_rule(self):
        """Test middle 40% difficulty rule is documented."""
        assert "40%" in QUIZ_PLANNING_PROMPT
        assert "medium" in QUIZ_PLANNING_PROMPT

    def test_last_30_percent_rule(self):
        """Test last 30% difficulty rule is documented."""
        # Already checked 30% above, just verify hard is mentioned
        assert "hard" in QUIZ_PLANNING_PROMPT

    def test_final_assessment_always_hard(self):
        """Test final_assessment always hard rule."""
        assert "final_assessment" in QUIZ_PLANNING_PROMPT
        assert "always" in QUIZ_PLANNING_PROMPT.lower()


# ============================================================================
# Tests for Question Type Matching Rules
# ============================================================================

class TestQuestionTypeMatchingRules:
    """Tests for question type matching rules in prompt."""

    def test_concepts_question_types(self):
        """Test concept content question types are documented."""
        assert "multiple_choice" in QUIZ_PLANNING_PROMPT
        assert "true_false" in QUIZ_PLANNING_PROMPT

    def test_code_question_types(self):
        """Test code content question types are documented."""
        assert "code_review" in QUIZ_PLANNING_PROMPT
        assert "code_completion" in QUIZ_PLANNING_PROMPT

    def test_architecture_question_types(self):
        """Test architecture content question types are documented."""
        assert "diagram_interpretation" in QUIZ_PLANNING_PROMPT
        assert "matching" in QUIZ_PLANNING_PROMPT

    def test_procedure_question_types(self):
        """Test procedure content question types are documented."""
        assert "ordering" in QUIZ_PLANNING_PROMPT

    def test_scenario_based_type(self):
        """Test scenario_based question type is documented."""
        assert "scenario_based" in QUIZ_PLANNING_PROMPT


# ============================================================================
# Tests for Coverage Rules
# ============================================================================

class TestCoverageRules:
    """Tests for coverage rules in prompt."""

    def test_all_objectives_covered(self):
        """Test all objectives coverage rule."""
        assert "learning objective" in QUIZ_PLANNING_PROMPT.lower()
        assert "covered" in QUIZ_PLANNING_PROMPT.lower()

    def test_max_topics_per_quiz(self):
        """Test max 5 topics per quiz rule."""
        assert "5" in QUIZ_PLANNING_PROMPT
        assert "topics" in QUIZ_PLANNING_PROMPT.lower()

    def test_final_assessment_covers_all(self):
        """Test final_assessment covers all sections rule."""
        assert "ALL sections" in QUIZ_PLANNING_PROMPT or "all sections" in QUIZ_PLANNING_PROMPT.lower()


# ============================================================================
# Tests for Self-Validation Checklist
# ============================================================================

class TestSelfValidationChecklist:
    """Tests for self-validation checklist items."""

    def test_frequency_check(self):
        """Test frequency compliance check is listed."""
        assert "frequency" in QUIZ_PLANNING_PROMPT.lower()

    def test_question_count_check(self):
        """Test question count range check is listed."""
        assert "3-15" in QUIZ_PLANNING_PROMPT or "question count" in QUIZ_PLANNING_PROMPT.lower()

    def test_difficulty_progression_check(self):
        """Test difficulty progression check is listed."""
        assert "difficulty" in QUIZ_PLANNING_PROMPT.lower()
        assert "progression" in QUIZ_PLANNING_PROMPT.lower()

    def test_objectives_coverage_check(self):
        """Test objectives coverage check is listed."""
        assert "objectives" in QUIZ_PLANNING_PROMPT.lower()
        assert "covered" in QUIZ_PLANNING_PROMPT.lower()

    def test_question_types_check(self):
        """Test question types check is listed."""
        assert "question types" in QUIZ_PLANNING_PROMPT.lower()

    def test_total_count_matches_check(self):
        """Test total_quiz_count matches check is listed."""
        assert "total_quiz_count" in QUIZ_PLANNING_PROMPT


# ============================================================================
# Tests for Examples
# ============================================================================

class TestPromptExamples:
    """Tests for example validity in prompt."""

    def test_kubernetes_example_exists(self):
        """Test Kubernetes example exists."""
        assert "Kubernetes" in QUIZ_PLANNING_PROMPT

    def test_leadership_example_exists(self):
        """Test Leadership example exists."""
        assert "Leadership" in QUIZ_PLANNING_PROMPT

    def test_kubernetes_example_has_valid_structure(self):
        """Test Kubernetes example has valid JSON structure."""
        # Extract the Kubernetes example (rough extraction)
        assert '"quiz_placement"' in QUIZ_PLANNING_PROMPT
        assert '"lecture_id"' in QUIZ_PLANNING_PROMPT
        assert '"quiz_type"' in QUIZ_PLANNING_PROMPT
        assert '"difficulty"' in QUIZ_PLANNING_PROMPT
        assert '"question_count"' in QUIZ_PLANNING_PROMPT
        assert '"topics_covered"' in QUIZ_PLANNING_PROMPT
        assert '"question_types"' in QUIZ_PLANNING_PROMPT

    def test_examples_show_progression(self):
        """Test that examples show difficulty progression."""
        # Kubernetes example should show easy -> medium -> hard
        assert '"easy"' in QUIZ_PLANNING_PROMPT
        assert '"medium"' in QUIZ_PLANNING_PROMPT
        assert '"hard"' in QUIZ_PLANNING_PROMPT


# ============================================================================
# Tests for Output Contract
# ============================================================================

class TestOutputContract:
    """Tests for output contract specification."""

    def test_json_only_requirement(self):
        """Test JSON only requirement is stated."""
        assert "valid JSON only" in QUIZ_PLANNING_PROMPT

    def test_no_markdown_requirement(self):
        """Test no markdown requirement is stated."""
        assert "no markdown" in QUIZ_PLANNING_PROMPT.lower()

    def test_output_schema_defined(self):
        """Test output schema is defined."""
        assert '"quiz_placement"' in QUIZ_PLANNING_PROMPT
        assert '"total_quiz_count"' in QUIZ_PLANNING_PROMPT
        assert '"coverage_analysis"' in QUIZ_PLANNING_PROMPT


# ============================================================================
# Tests for Validator
# ============================================================================

class TestQuizPlanningValidator:
    """Tests for the QuizPlanningValidator class."""

    @pytest.fixture
    def validator(self):
        return QuizPlanningValidator(total_lectures=10)

    def test_valid_output_passes(self, validator):
        """Test that valid output passes validation."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_003",
                    "quiz_type": "section_review",
                    "difficulty": "easy",
                    "question_count": 5,
                    "topics_covered": ["topic1", "topic2"],
                    "question_types": ["multiple_choice", "true_false"]
                },
                {
                    "lecture_id": "lec_006",
                    "quiz_type": "section_review",
                    "difficulty": "medium",
                    "question_count": 6,
                    "topics_covered": ["topic3", "topic4"],
                    "question_types": ["code_review", "scenario_based"]
                },
                {
                    "lecture_id": "lec_010",
                    "quiz_type": "final_assessment",
                    "difficulty": "hard",
                    "question_count": 10,
                    "topics_covered": ["all topics"],
                    "question_types": ["multiple_choice", "code_review", "scenario_based"]
                }
            ],
            "total_quiz_count": 3,
            "coverage_analysis": "Full coverage of all learning objectives across 3 quizzes."
        }

        result = validator.validate_output(output)
        assert result["is_valid"] is True, f"Issues: {result['issues']}"

    def test_missing_quiz_placement_fails(self, validator):
        """Test that missing quiz_placement fails."""
        output = {
            "total_quiz_count": 0,
            "coverage_analysis": "No quizzes"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("quiz_placement" in issue for issue in result["issues"])

    def test_missing_total_quiz_count_fails(self, validator):
        """Test that missing total_quiz_count fails."""
        output = {
            "quiz_placement": [],
            "coverage_analysis": "No quizzes"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("total_quiz_count" in issue for issue in result["issues"])

    def test_count_mismatch_fails(self, validator):
        """Test that count mismatch fails."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "lecture_check",
                    "difficulty": "easy",
                    "question_count": 3,
                    "topics_covered": ["topic1"]
                }
            ],
            "total_quiz_count": 5,  # Wrong!
            "coverage_analysis": "Analysis"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("doesn't match" in issue for issue in result["issues"])

    def test_invalid_quiz_type_fails(self, validator):
        """Test that invalid quiz_type fails."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "invalid_type",
                    "difficulty": "easy",
                    "question_count": 3,
                    "topics_covered": ["topic1"]
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Analysis"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("invalid quiz_type" in issue for issue in result["issues"])

    def test_invalid_difficulty_fails(self, validator):
        """Test that invalid difficulty fails."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "lecture_check",
                    "difficulty": "super_hard",
                    "question_count": 3,
                    "topics_covered": ["topic1"]
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Analysis"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("invalid difficulty" in issue for issue in result["issues"])

    def test_question_count_too_low_fails(self, validator):
        """Test that question_count below range fails."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "lecture_check",
                    "difficulty": "easy",
                    "question_count": 1,  # Below 3
                    "topics_covered": ["topic1"]
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Analysis"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("out of range" in issue for issue in result["issues"])

    def test_question_count_too_high_fails(self, validator):
        """Test that question_count above 15 fails."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "final_assessment",
                    "difficulty": "hard",
                    "question_count": 20,  # Above 15
                    "topics_covered": ["topic1"]
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Analysis"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("exceeds maximum" in issue for issue in result["issues"])

    def test_too_many_topics_fails(self, validator):
        """Test that more than 5 topics fails."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "lecture_check",
                    "difficulty": "easy",
                    "question_count": 3,
                    "topics_covered": ["t1", "t2", "t3", "t4", "t5", "t6"]  # 6 topics
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Analysis"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("max 5" in issue for issue in result["issues"])

    def test_empty_topics_fails(self, validator):
        """Test that empty topics_covered fails."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "lecture_check",
                    "difficulty": "easy",
                    "question_count": 3,
                    "topics_covered": []
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Analysis"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("empty" in issue for issue in result["issues"])

    def test_invalid_question_type_fails(self, validator):
        """Test that invalid question_type fails."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "lecture_check",
                    "difficulty": "easy",
                    "question_count": 3,
                    "topics_covered": ["topic1"],
                    "question_types": ["invalid_type"]
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Analysis"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("invalid question_type" in issue for issue in result["issues"])


# ============================================================================
# Tests for Frequency Compliance Validation
# ============================================================================

class TestFrequencyComplianceValidation:
    """Tests for frequency compliance validation."""

    @pytest.fixture
    def validator(self):
        return QuizPlanningValidator(total_lectures=10)

    def test_per_lecture_compliance(self, validator):
        """Test per_lecture frequency compliance."""
        placements = [
            {"quiz_type": "lecture_check"} for _ in range(8)
        ]
        issues = validator.validate_frequency_compliance(
            placements, "per_lecture", section_count=2, lecture_count=10
        )
        assert len(issues) == 0

    def test_per_lecture_non_compliance(self, validator):
        """Test per_lecture frequency non-compliance."""
        placements = [
            {"quiz_type": "lecture_check"} for _ in range(3)  # Too few
        ]
        issues = validator.validate_frequency_compliance(
            placements, "per_lecture", section_count=2, lecture_count=10
        )
        assert len(issues) > 0

    def test_per_section_compliance(self, validator):
        """Test per_section frequency compliance."""
        placements = [
            {"quiz_type": "section_review"},
            {"quiz_type": "section_review"},
            {"quiz_type": "final_assessment"}
        ]
        issues = validator.validate_frequency_compliance(
            placements, "per_section", section_count=3, lecture_count=10
        )
        assert len(issues) == 0

    def test_end_only_compliance(self, validator):
        """Test end_only frequency compliance."""
        placements = [
            {"quiz_type": "final_assessment"}
        ]
        issues = validator.validate_frequency_compliance(
            placements, "end_only", section_count=3, lecture_count=10
        )
        assert len(issues) == 0

    def test_end_only_non_compliance_multiple_quizzes(self, validator):
        """Test end_only non-compliance with multiple quizzes."""
        placements = [
            {"quiz_type": "section_review"},
            {"quiz_type": "final_assessment"}
        ]
        issues = validator.validate_frequency_compliance(
            placements, "end_only", section_count=3, lecture_count=10
        )
        assert len(issues) > 0

    def test_end_only_non_compliance_wrong_type(self, validator):
        """Test end_only non-compliance with wrong quiz type."""
        placements = [
            {"quiz_type": "section_review"}  # Should be final_assessment
        ]
        issues = validator.validate_frequency_compliance(
            placements, "end_only", section_count=3, lecture_count=10
        )
        assert len(issues) > 0


# ============================================================================
# Tests for Difficulty Progression Validation
# ============================================================================

class TestDifficultyProgressionValidation:
    """Tests for difficulty progression validation."""

    @pytest.fixture
    def validator(self):
        return QuizPlanningValidator(total_lectures=10)

    def test_valid_progression(self, validator):
        """Test valid difficulty progression."""
        placements = [
            {"lecture_id": "lec_002", "quiz_type": "section_review", "difficulty": "easy"},
            {"lecture_id": "lec_005", "quiz_type": "section_review", "difficulty": "medium"},
            {"lecture_id": "lec_010", "quiz_type": "final_assessment", "difficulty": "hard"}
        ]
        positions = {"lec_002": 0.2, "lec_005": 0.5, "lec_010": 1.0}

        issues = validator.validate_difficulty_progression(placements, positions)
        assert len(issues) == 0

    def test_hard_too_early_fails(self, validator):
        """Test that hard difficulty too early fails."""
        placements = [
            {"lecture_id": "lec_001", "quiz_type": "lecture_check", "difficulty": "hard"}
        ]
        positions = {"lec_001": 0.1}  # First 10%

        issues = validator.validate_difficulty_progression(placements, positions)
        assert len(issues) > 0
        assert any("too early" in issue for issue in issues)

    def test_final_assessment_not_hard_fails(self, validator):
        """Test that final_assessment with non-hard difficulty fails."""
        placements = [
            {"lecture_id": "lec_010", "quiz_type": "final_assessment", "difficulty": "medium"}
        ]
        positions = {"lec_010": 1.0}

        issues = validator.validate_difficulty_progression(placements, positions)
        assert len(issues) > 0
        assert any("final_assessment should have 'hard'" in issue for issue in issues)


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def validator(self):
        return QuizPlanningValidator(total_lectures=10)

    def test_empty_quiz_placement_valid(self, validator):
        """Test that empty quiz_placement is valid when quiz_enabled=false."""
        output = {
            "quiz_placement": [],
            "total_quiz_count": 0,
            "coverage_analysis": "No quizzes as quiz_enabled is false."
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is True

    def test_single_lecture_course(self):
        """Test validation for single lecture course."""
        validator = QuizPlanningValidator(total_lectures=1)
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "final_assessment",
                    "difficulty": "hard",
                    "question_count": 8,
                    "topics_covered": ["topic1", "topic2"]
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Single lecture course with comprehensive final quiz."
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is True

    def test_boundary_question_counts(self, validator):
        """Test boundary question counts."""
        # Exactly at boundaries
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "lecture_check",
                    "difficulty": "easy",
                    "question_count": 3,  # Minimum
                    "topics_covered": ["topic1"]
                },
                {
                    "lecture_id": "lec_005",
                    "quiz_type": "section_review",
                    "difficulty": "medium",
                    "question_count": 8,  # Maximum for section_review
                    "topics_covered": ["topic2"]
                },
                {
                    "lecture_id": "lec_010",
                    "quiz_type": "final_assessment",
                    "difficulty": "hard",
                    "question_count": 15,  # Maximum overall
                    "topics_covered": ["topic3"]
                }
            ],
            "total_quiz_count": 3,
            "coverage_analysis": "Boundary test"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is True

    def test_exactly_5_topics_valid(self, validator):
        """Test that exactly 5 topics is valid."""
        output = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "lecture_check",
                    "difficulty": "easy",
                    "question_count": 5,
                    "topics_covered": ["t1", "t2", "t3", "t4", "t5"]  # Exactly 5
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Full coverage of all 5 topics in single quiz."
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is True
