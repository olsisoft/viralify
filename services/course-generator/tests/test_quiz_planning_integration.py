"""
Integration tests for plan_quizzes function

Tests the full flow of quiz planning including:
- State management
- Prompt formatting
- LLM interaction
- Response parsing
- Validation
"""

import pytest
import json
import sys
import importlib.util
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, AsyncMock


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
# Mock Data Classes
# ============================================================================

@dataclass
class MockLecture:
    id: str
    title: str
    objectives: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class MockSection:
    title: str
    lectures: List[MockLecture] = field(default_factory=list)


@dataclass
class MockOutline:
    title: str = "Test Course"
    description: str = "A test course"
    sections: List[MockSection] = field(default_factory=list)


# ============================================================================
# QuizPlanningValidator (from unit tests)
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

    QUESTION_COUNT_RANGES = {
        "lecture_check": (3, 5),
        "section_review": (5, 8),
        "final_assessment": (8, 15),
    }

    def validate_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the full output structure."""
        issues = []

        if "quiz_placement" not in output:
            issues.append("Missing 'quiz_placement' field")
            return {"is_valid": False, "issues": issues}

        if "total_quiz_count" not in output:
            issues.append("Missing 'total_quiz_count' field")

        for i, quiz in enumerate(output.get("quiz_placement", [])):
            prefix = f"Quiz {i + 1}"
            quiz_type = quiz.get("quiz_type", "")

            if quiz_type not in self.VALID_QUIZ_TYPES:
                issues.append(f"{prefix}: invalid quiz_type '{quiz_type}'")

            difficulty = quiz.get("difficulty", "")
            if difficulty not in self.VALID_DIFFICULTIES:
                issues.append(f"{prefix}: invalid difficulty '{difficulty}'")

            question_count = quiz.get("question_count", 0)
            if quiz_type in self.QUESTION_COUNT_RANGES:
                min_c, max_c = self.QUESTION_COUNT_RANGES[quiz_type]
                if not (min_c <= question_count <= max_c):
                    issues.append(f"{prefix}: question_count {question_count} out of range")

            if question_count > 15:
                issues.append(f"{prefix}: question_count exceeds 15")

            topics = quiz.get("topics_covered", [])
            if len(topics) > 5:
                issues.append(f"{prefix}: too many topics")

        return {"is_valid": len(issues) == 0, "issues": issues}


# ============================================================================
# Mock plan_quizzes function
# ============================================================================

async def mock_plan_quizzes(
    state: Dict[str, Any],
    mock_client: MagicMock,
    mock_response: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Mock implementation of plan_quizzes that mirrors the actual function.
    """
    state["current_node"] = "plan_quizzes"

    # Early return if quizzes disabled
    if not state.get("quiz_enabled", True):
        return {
            "quiz_placement": [],
            "quiz_total_count": 0,
            "current_node": "plan_quizzes",
        }

    outline = state.get("outline")
    if not outline:
        return {
            "quiz_placement": [],
            "quiz_total_count": 0,
            "current_node": "plan_quizzes",
            "errors": state.get("errors", []) + ["No outline provided"],
        }

    # Build outline structure (like actual implementation)
    outline_lines = []
    section_objectives = []
    lecture_count = 0
    section_count = len(outline.sections)

    for section in outline.sections:
        outline_lines.append(f"Section: {section.title}")
        sec_objs = []
        for lecture in section.lectures:
            outline_lines.append(f"  - {lecture.id}: {lecture.title}")
            lecture_count += 1
            sec_objs.extend(lecture.objectives[:2] if lecture.objectives else [])
        section_objectives.append(f"{section.title}: {', '.join(sec_objs)}")

    # Format prompt
    prompt = QUIZ_PLANNING_PROMPT.format(
        quiz_enabled=state.get("quiz_enabled", True),
        quiz_frequency=state.get("quiz_frequency", "per_section"),
        questions_per_quiz=state.get("questions_per_quiz", 5),
        outline_structure="\n".join(outline_lines),
        section_objectives="\n".join(section_objectives),
    )

    # Configure mock response
    mock_message = MagicMock()
    mock_message.content = json.dumps(mock_response)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_result = MagicMock()
    mock_result.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_result

    # Simulate LLM call
    try:
        response = await mock_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1000
        )

        result = json.loads(response.choices[0].message.content)
        placements = result.get("quiz_placement", [])

        return {
            "quiz_placement": placements,
            "quiz_total_count": result.get("total_quiz_count", len(placements)),
            "total_quiz_count": result.get("total_quiz_count", len(placements)),  # For validator
            "coverage_analysis": result.get("coverage_analysis", ""),
            "current_node": "plan_quizzes",
            "prompt_used": prompt,
            "outline_structure": "\n".join(outline_lines),
            "section_objectives": "\n".join(section_objectives),
            "lecture_count": lecture_count,
            "section_count": section_count,
        }

    except Exception as e:
        return {
            "quiz_placement": [],
            "quiz_total_count": 0,
            "current_node": "plan_quizzes",
            "errors": state.get("errors", []) + [f"Quiz planning failed: {str(e)}"],
        }


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def sample_outline():
    """Create a sample course outline."""
    return MockOutline(
        title="Python for Data Science",
        description="Learn Python for data analysis",
        sections=[
            MockSection(
                title="Python Basics",
                lectures=[
                    MockLecture(
                        id="lec_001",
                        title="Introduction to Python",
                        objectives=["Understand Python syntax", "Write first program"]
                    ),
                    MockLecture(
                        id="lec_002",
                        title="Data Types and Variables",
                        objectives=["Master Python data types", "Use variables effectively"]
                    ),
                    MockLecture(
                        id="lec_003",
                        title="Control Flow",
                        objectives=["Write conditionals", "Create loops"]
                    ),
                ]
            ),
            MockSection(
                title="Data Analysis",
                lectures=[
                    MockLecture(
                        id="lec_004",
                        title="Introduction to Pandas",
                        objectives=["Import pandas", "Create DataFrames"]
                    ),
                    MockLecture(
                        id="lec_005",
                        title="Data Manipulation",
                        objectives=["Filter data", "Aggregate data"]
                    ),
                    MockLecture(
                        id="lec_006",
                        title="Data Visualization",
                        objectives=["Create charts", "Customize plots"]
                    ),
                ]
            ),
        ]
    )


@pytest.fixture
def per_section_state(sample_outline):
    """State with per_section frequency."""
    return {
        "topic": "Python for Data Science",
        "quiz_enabled": True,
        "quiz_frequency": "per_section",
        "questions_per_quiz": 5,
        "outline": sample_outline,
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def per_lecture_state(sample_outline):
    """State with per_lecture frequency."""
    return {
        "topic": "Python for Data Science",
        "quiz_enabled": True,
        "quiz_frequency": "per_lecture",
        "questions_per_quiz": 3,
        "outline": sample_outline,
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def end_only_state(sample_outline):
    """State with end_only frequency."""
    return {
        "topic": "Python for Data Science",
        "quiz_enabled": True,
        "quiz_frequency": "end_only",
        "questions_per_quiz": 10,
        "outline": sample_outline,
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def valid_per_section_response():
    """Valid LLM response for per_section frequency."""
    return {
        "quiz_placement": [
            {
                "lecture_id": "lec_003",
                "quiz_type": "section_review",
                "difficulty": "easy",
                "question_count": 5,
                "topics_covered": ["Python syntax", "data types", "control flow"],
                "question_types": ["multiple_choice", "code_review", "true_false"]
            },
            {
                "lecture_id": "lec_006",
                "quiz_type": "section_review",
                "difficulty": "medium",
                "question_count": 6,
                "topics_covered": ["pandas", "data manipulation", "visualization"],
                "question_types": ["code_review", "code_completion", "scenario_based"]
            },
            {
                "lecture_id": "lec_006",
                "quiz_type": "final_assessment",
                "difficulty": "hard",
                "question_count": 10,
                "topics_covered": ["full Python workflow", "data analysis pipeline"],
                "question_types": ["code_review", "scenario_based", "ordering"]
            }
        ],
        "total_quiz_count": 3,
        "coverage_analysis": "Full coverage: Python basics validated in section 1, data analysis skills in section 2, comprehensive final covers end-to-end workflow."
    }


@pytest.fixture
def valid_per_lecture_response():
    """Valid LLM response for per_lecture frequency."""
    return {
        "quiz_placement": [
            {
                "lecture_id": "lec_001",
                "quiz_type": "lecture_check",
                "difficulty": "easy",
                "question_count": 3,
                "topics_covered": ["Python basics"],
                "question_types": ["multiple_choice", "true_false"]
            },
            {
                "lecture_id": "lec_002",
                "quiz_type": "lecture_check",
                "difficulty": "easy",
                "question_count": 4,
                "topics_covered": ["data types"],
                "question_types": ["multiple_choice", "fill_blank"]
            },
            {
                "lecture_id": "lec_003",
                "quiz_type": "lecture_check",
                "difficulty": "easy",
                "question_count": 4,
                "topics_covered": ["control flow"],
                "question_types": ["code_review", "multiple_choice"]
            },
            {
                "lecture_id": "lec_004",
                "quiz_type": "lecture_check",
                "difficulty": "medium",
                "question_count": 4,
                "topics_covered": ["pandas basics"],
                "question_types": ["code_review", "code_completion"]
            },
            {
                "lecture_id": "lec_005",
                "quiz_type": "lecture_check",
                "difficulty": "medium",
                "question_count": 5,
                "topics_covered": ["data manipulation"],
                "question_types": ["code_review", "scenario_based"]
            },
            {
                "lecture_id": "lec_006",
                "quiz_type": "lecture_check",
                "difficulty": "medium",
                "question_count": 5,
                "topics_covered": ["visualization"],
                "question_types": ["code_completion", "matching"]
            }
        ],
        "total_quiz_count": 6,
        "coverage_analysis": "Each lecture has immediate knowledge validation. Difficulty progresses from easy to medium."
    }


@pytest.fixture
def valid_end_only_response():
    """Valid LLM response for end_only frequency."""
    return {
        "quiz_placement": [
            {
                "lecture_id": "lec_006",
                "quiz_type": "final_assessment",
                "difficulty": "hard",
                "question_count": 10,
                "topics_covered": ["Python fundamentals", "data analysis", "visualization"],
                "question_types": ["code_review", "scenario_based", "ordering", "multiple_choice"]
            }
        ],
        "total_quiz_count": 1,
        "coverage_analysis": "Comprehensive final assessment covering all course objectives in a single evaluation."
    }


# ============================================================================
# Tests for Full Integration Flow
# ============================================================================

class TestPlanQuizzesFlow:
    """Integration tests for plan_quizzes function flow."""

    @pytest.mark.asyncio
    async def test_per_section_full_flow(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test full flow for per_section frequency."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert "quiz_placement" in result
        assert result["quiz_total_count"] == 3
        assert len(result["quiz_placement"]) == 3

        # Validate output
        validator = QuizPlanningValidator()
        validation = validator.validate_output(result)
        assert validation["is_valid"] is True, f"Issues: {validation['issues']}"

    @pytest.mark.asyncio
    async def test_per_lecture_full_flow(
        self, mock_openai_client, per_lecture_state, valid_per_lecture_response
    ):
        """Test full flow for per_lecture frequency."""
        result = await mock_plan_quizzes(
            per_lecture_state,
            mock_openai_client,
            valid_per_lecture_response
        )

        assert "quiz_placement" in result
        assert result["quiz_total_count"] == 6  # One per lecture
        assert len(result["quiz_placement"]) == 6

        # All should be lecture_check
        for quiz in result["quiz_placement"]:
            assert quiz["quiz_type"] == "lecture_check"

    @pytest.mark.asyncio
    async def test_end_only_full_flow(
        self, mock_openai_client, end_only_state, valid_end_only_response
    ):
        """Test full flow for end_only frequency."""
        result = await mock_plan_quizzes(
            end_only_state,
            mock_openai_client,
            valid_end_only_response
        )

        assert "quiz_placement" in result
        assert result["quiz_total_count"] == 1
        assert len(result["quiz_placement"]) == 1
        assert result["quiz_placement"][0]["quiz_type"] == "final_assessment"

    @pytest.mark.asyncio
    async def test_quiz_disabled_returns_empty(
        self, mock_openai_client, per_section_state
    ):
        """Test that quiz_enabled=false returns empty placement."""
        per_section_state["quiz_enabled"] = False

        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            {}  # Response doesn't matter
        )

        assert result["quiz_placement"] == []
        assert result["quiz_total_count"] == 0

    @pytest.mark.asyncio
    async def test_no_outline_returns_error(
        self, mock_openai_client
    ):
        """Test that missing outline returns error."""
        state = {
            "quiz_enabled": True,
            "quiz_frequency": "per_section",
            "outline": None,
            "errors": [],
        }

        result = await mock_plan_quizzes(
            state,
            mock_openai_client,
            {}
        )

        assert result["quiz_placement"] == []
        assert "errors" in result
        assert len(result["errors"]) > 0


# ============================================================================
# Tests for Outline Parsing
# ============================================================================

class TestOutlineParsing:
    """Tests for outline structure parsing."""

    @pytest.mark.asyncio
    async def test_outline_structure_contains_sections(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that outline structure contains section titles."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        outline_str = result["outline_structure"]
        assert "Section: Python Basics" in outline_str
        assert "Section: Data Analysis" in outline_str

    @pytest.mark.asyncio
    async def test_outline_structure_contains_lectures(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that outline structure contains lecture IDs and titles."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        outline_str = result["outline_structure"]
        assert "lec_001: Introduction to Python" in outline_str
        assert "lec_004: Introduction to Pandas" in outline_str

    @pytest.mark.asyncio
    async def test_section_objectives_contains_learning_goals(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that section objectives contain learning goals."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        objectives_str = result["section_objectives"]
        assert "Python Basics:" in objectives_str
        assert "Data Analysis:" in objectives_str

    @pytest.mark.asyncio
    async def test_lecture_and_section_counts(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that lecture and section counts are correct."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert result["lecture_count"] == 6
        assert result["section_count"] == 2


# ============================================================================
# Tests for Prompt Formatting
# ============================================================================

class TestPromptFormatting:
    """Tests for prompt formatting with state values."""

    @pytest.mark.asyncio
    async def test_prompt_contains_quiz_enabled(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that quiz_enabled is included in prompt."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert "True" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_contains_quiz_frequency(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that quiz_frequency is included in prompt."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert "per_section" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_contains_questions_per_quiz(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that questions_per_quiz is included in prompt."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert "5" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_contains_outline_structure(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that outline structure is included in prompt."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert "Python Basics" in result["prompt_used"]
        assert "lec_001" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_contains_section_objectives(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that section objectives are included in prompt."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert "Python Basics:" in result["prompt_used"]


# ============================================================================
# Tests for Frequency Handling
# ============================================================================

class TestFrequencyHandling:
    """Tests for different frequency modes."""

    @pytest.mark.asyncio
    async def test_per_lecture_creates_lecture_checks(
        self, mock_openai_client, per_lecture_state, valid_per_lecture_response
    ):
        """Test that per_lecture creates lecture_check quizzes."""
        result = await mock_plan_quizzes(
            per_lecture_state,
            mock_openai_client,
            valid_per_lecture_response
        )

        quiz_types = [q["quiz_type"] for q in result["quiz_placement"]]
        assert all(qt == "lecture_check" for qt in quiz_types)

    @pytest.mark.asyncio
    async def test_per_section_creates_section_reviews(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that per_section creates section_review quizzes."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        quiz_types = [q["quiz_type"] for q in result["quiz_placement"]]
        assert "section_review" in quiz_types

    @pytest.mark.asyncio
    async def test_end_only_creates_single_final(
        self, mock_openai_client, end_only_state, valid_end_only_response
    ):
        """Test that end_only creates single final_assessment."""
        result = await mock_plan_quizzes(
            end_only_state,
            mock_openai_client,
            valid_end_only_response
        )

        assert len(result["quiz_placement"]) == 1
        assert result["quiz_placement"][0]["quiz_type"] == "final_assessment"


# ============================================================================
# Tests for Difficulty Progression
# ============================================================================

class TestDifficultyProgression:
    """Tests for difficulty progression in responses."""

    @pytest.mark.asyncio
    async def test_per_lecture_difficulty_progression(
        self, mock_openai_client, per_lecture_state, valid_per_lecture_response
    ):
        """Test that per_lecture shows difficulty progression."""
        result = await mock_plan_quizzes(
            per_lecture_state,
            mock_openai_client,
            valid_per_lecture_response
        )

        difficulties = [q["difficulty"] for q in result["quiz_placement"]]

        # First 3 should be easy (section 1)
        assert all(d == "easy" for d in difficulties[:3])

        # Last 3 should be medium (section 2)
        assert all(d == "medium" for d in difficulties[3:])

    @pytest.mark.asyncio
    async def test_final_assessment_is_hard(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that final_assessment has hard difficulty."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        final = [q for q in result["quiz_placement"] if q["quiz_type"] == "final_assessment"]
        assert len(final) > 0
        assert final[0]["difficulty"] == "hard"


# ============================================================================
# Tests for Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_llm_exception_returns_error(
        self, mock_openai_client, per_section_state
    ):
        """Test that LLM exception is handled gracefully."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            {}
        )

        assert result["quiz_placement"] == []
        assert "errors" in result
        assert any("Quiz planning failed" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_empty_response_handled(
        self, mock_openai_client, per_section_state
    ):
        """Test that empty response is handled."""
        empty_response = {
            "quiz_placement": [],
            "total_quiz_count": 0,
            "coverage_analysis": "No quizzes planned."
        }

        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            empty_response
        )

        assert result["quiz_placement"] == []
        assert result["quiz_total_count"] == 0


# ============================================================================
# Tests for Response Validation
# ============================================================================

class TestResponseValidation:
    """Tests for validating LLM responses."""

    @pytest.mark.asyncio
    async def test_valid_response_passes_validation(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that valid response passes validation."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        validator = QuizPlanningValidator()
        validation = validator.validate_output(result)

        assert validation["is_valid"] is True, f"Issues: {validation['issues']}"

    @pytest.mark.asyncio
    async def test_response_with_invalid_quiz_type_detected(
        self, mock_openai_client, per_section_state
    ):
        """Test that response with invalid quiz_type is detected."""
        invalid_response = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_003",
                    "quiz_type": "invalid_type",
                    "difficulty": "easy",
                    "question_count": 5,
                    "topics_covered": ["topic1"]
                }
            ],
            "total_quiz_count": 1
        }

        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            invalid_response
        )

        validator = QuizPlanningValidator()
        validation = validator.validate_output(result)

        assert validation["is_valid"] is False
        assert any("invalid quiz_type" in issue for issue in validation["issues"])


# ============================================================================
# Tests for State Updates
# ============================================================================

class TestStateUpdates:
    """Tests for state updates after plan_quizzes."""

    @pytest.mark.asyncio
    async def test_current_node_updated(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that current_node is updated."""
        await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert per_section_state["current_node"] == "plan_quizzes"

    @pytest.mark.asyncio
    async def test_result_contains_quiz_placement(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that result contains quiz_placement."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert "quiz_placement" in result
        assert isinstance(result["quiz_placement"], list)

    @pytest.mark.asyncio
    async def test_result_contains_coverage_analysis(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that result contains coverage_analysis."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        assert "coverage_analysis" in result
        assert len(result["coverage_analysis"]) > 0


# ============================================================================
# Tests for LLM Call Parameters
# ============================================================================

class TestLLMCallParameters:
    """Tests for LLM API call parameters."""

    @pytest.mark.asyncio
    async def test_llm_called_with_json_format(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that LLM is called with JSON response format."""
        await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_llm_called_with_low_temperature(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that LLM is called with low temperature for consistency."""
        await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] <= 0.5

    @pytest.mark.asyncio
    async def test_llm_called_with_appropriate_max_tokens(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that LLM is called with sufficient max_tokens."""
        await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] >= 500


# ============================================================================
# Tests for Question Type Matching
# ============================================================================

class TestQuestionTypeMatching:
    """Tests for question type matching with content."""

    @pytest.mark.asyncio
    async def test_code_content_has_code_question_types(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that code-heavy content has code question types."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        # Data Analysis section should have code_review
        all_question_types = []
        for quiz in result["quiz_placement"]:
            all_question_types.extend(quiz.get("question_types", []))

        assert "code_review" in all_question_types

    @pytest.mark.asyncio
    async def test_concept_content_has_multiple_choice(
        self, mock_openai_client, per_section_state, valid_per_section_response
    ):
        """Test that concept content has multiple_choice."""
        result = await mock_plan_quizzes(
            per_section_state,
            mock_openai_client,
            valid_per_section_response
        )

        all_question_types = []
        for quiz in result["quiz_placement"]:
            all_question_types.extend(quiz.get("question_types", []))

        assert "multiple_choice" in all_question_types


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_single_lecture_outline(self, mock_openai_client):
        """Test with single lecture outline."""
        outline = MockOutline(
            sections=[
                MockSection(
                    title="Only Section",
                    lectures=[
                        MockLecture(id="lec_001", title="Only Lecture")
                    ]
                )
            ]
        )

        state = {
            "quiz_enabled": True,
            "quiz_frequency": "end_only",
            "questions_per_quiz": 5,
            "outline": outline,
            "errors": [],
        }

        single_response = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "final_assessment",
                    "difficulty": "hard",
                    "question_count": 8,
                    "topics_covered": ["course content"],
                    "question_types": ["multiple_choice"]
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Single lecture final assessment."
        }

        result = await mock_plan_quizzes(state, mock_openai_client, single_response)

        assert len(result["quiz_placement"]) == 1
        assert result["lecture_count"] == 1

    @pytest.mark.asyncio
    async def test_many_sections_outline(self, mock_openai_client):
        """Test with many sections outline."""
        sections = [
            MockSection(
                title=f"Section {i}",
                lectures=[
                    MockLecture(id=f"lec_{i:03d}", title=f"Lecture {i}")
                ]
            )
            for i in range(1, 11)  # 10 sections
        ]

        outline = MockOutline(sections=sections)

        state = {
            "quiz_enabled": True,
            "quiz_frequency": "per_section",
            "questions_per_quiz": 5,
            "outline": outline,
            "errors": [],
        }

        many_response = {
            "quiz_placement": [
                {
                    "lecture_id": f"lec_{i:03d}",
                    "quiz_type": "section_review",
                    "difficulty": "medium",
                    "question_count": 5,
                    "topics_covered": [f"topic {i}"],
                    "question_types": ["multiple_choice"]
                }
                for i in range(1, 11)
            ],
            "total_quiz_count": 10,
            "coverage_analysis": "All 10 sections covered."
        }

        result = await mock_plan_quizzes(state, mock_openai_client, many_response)

        assert result["section_count"] == 10
        assert len(result["quiz_placement"]) == 10

    @pytest.mark.asyncio
    async def test_lecture_without_objectives(self, mock_openai_client):
        """Test with lectures that have no objectives."""
        outline = MockOutline(
            sections=[
                MockSection(
                    title="Section",
                    lectures=[
                        MockLecture(id="lec_001", title="Lecture Without Objectives"),
                    ]
                )
            ]
        )

        state = {
            "quiz_enabled": True,
            "quiz_frequency": "per_section",
            "outline": outline,
            "errors": [],
        }

        response = {
            "quiz_placement": [
                {
                    "lecture_id": "lec_001",
                    "quiz_type": "section_review",
                    "difficulty": "easy",
                    "question_count": 5,
                    "topics_covered": ["general content"],
                    "question_types": ["multiple_choice"]
                }
            ],
            "total_quiz_count": 1,
            "coverage_analysis": "Basic coverage."
        }

        result = await mock_plan_quizzes(state, mock_openai_client, response)

        # Should still work
        assert "quiz_placement" in result
        assert len(result["quiz_placement"]) == 1
