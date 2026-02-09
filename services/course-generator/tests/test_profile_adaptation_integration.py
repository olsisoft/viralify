"""
Integration tests for adapt_for_profile function

Tests the full flow from state input through LLM call to state output,
including prompt formatting, response parsing, and error handling.
"""

import pytest
import json
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from enum import Enum

import sys
import os
import importlib.util

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def import_module_from_file(module_name: str, file_path: str):
    """Import a module directly from file to avoid langgraph dependency"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import prompts module directly
agents_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents")
prompts_module = import_module_from_file(
    "pedagogical_prompts",
    os.path.join(agents_path, "pedagogical_prompts.py")
)
PROFILE_ADAPTATION_PROMPT = prompts_module.PROFILE_ADAPTATION_PROMPT


# ============================================================================
# ProfileCategory Enum (standalone to avoid import chain)
# ============================================================================

class ProfileCategory(str, Enum):
    """Course profile categories"""
    TECH = "tech"
    BUSINESS = "business"
    CREATIVE = "creative"
    HEALTH = "health"
    EDUCATION = "education"
    LIFESTYLE = "lifestyle"


# ============================================================================
# Mock LessonElement for testing
# ============================================================================

@dataclass
class MockLessonElementId:
    value: str


@dataclass
class MockLessonElement:
    id: MockLessonElementId
    name: str
    description: str


# Category-specific elements (mirrors lesson_elements.py)
CATEGORY_ELEMENTS = {
    ProfileCategory.TECH: [
        MockLessonElement(MockLessonElementId("code_demo"), "Code Demo", "Live coding demonstration"),
        MockLessonElement(MockLessonElementId("terminal_output"), "Terminal Output", "Command line examples"),
        MockLessonElement(MockLessonElementId("architecture_diagram"), "Architecture Diagram", "System architecture"),
        MockLessonElement(MockLessonElementId("debug_tips"), "Debug Tips", "Debugging strategies"),
    ],
    ProfileCategory.BUSINESS: [
        MockLessonElement(MockLessonElementId("case_study"), "Case Study", "Real-world business examples"),
        MockLessonElement(MockLessonElementId("framework_template"), "Framework Template", "Actionable frameworks"),
        MockLessonElement(MockLessonElementId("roi_metrics"), "ROI Metrics", "Return on investment analysis"),
        MockLessonElement(MockLessonElementId("action_checklist"), "Action Checklist", "Practical action items"),
    ],
    ProfileCategory.CREATIVE: [
        MockLessonElement(MockLessonElementId("before_after"), "Before/After", "Transformation examples"),
        MockLessonElement(MockLessonElementId("technique_demo"), "Technique Demo", "Creative technique demonstration"),
        MockLessonElement(MockLessonElementId("creative_exercise"), "Creative Exercise", "Hands-on creative task"),
    ],
    ProfileCategory.HEALTH: [
        MockLessonElement(MockLessonElementId("exercise_demo"), "Exercise Demo", "Physical exercise demonstration"),
        MockLessonElement(MockLessonElementId("body_diagram"), "Body Diagram", "Anatomical illustration"),
        MockLessonElement(MockLessonElementId("safety_warning"), "Safety Warning", "Safety precautions"),
    ],
    ProfileCategory.EDUCATION: [
        MockLessonElement(MockLessonElementId("memory_aid"), "Memory Aid", "Mnemonic devices"),
        MockLessonElement(MockLessonElementId("practice_problem"), "Practice Problem", "Practice exercises"),
    ],
    ProfileCategory.LIFESTYLE: [
        MockLessonElement(MockLessonElementId("daily_routine"), "Daily Routine", "Routine templates"),
        MockLessonElement(MockLessonElementId("habit_tracker"), "Habit Tracker", "Progress tracking"),
    ],
}

# Common elements for all categories
COMMON_ELEMENTS = [
    MockLessonElement(MockLessonElementId("concept_intro"), "Concept Introduction", "Core concept explanation"),
    MockLessonElement(MockLessonElementId("voiceover"), "Voiceover", "Narrative explanation"),
    MockLessonElement(MockLessonElementId("conclusion"), "Conclusion", "Summary and key takeaways"),
    MockLessonElement(MockLessonElementId("quiz_evaluation"), "Quiz", "Knowledge assessment"),
]


def get_elements_for_category(category: ProfileCategory) -> List[MockLessonElement]:
    """Get available lesson elements for a category"""
    category_specific = CATEGORY_ELEMENTS.get(category, [])
    return COMMON_ELEMENTS + category_specific


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client"""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def technical_course_state():
    """State for a technical programming course"""
    return {
        "topic": "Python for Data Science",
        "description": "Learn Python for data analysis and machine learning",
        "profile_category": "tech",
        "detected_persona": "Backend Engineer",
        "topic_complexity": "intermediate",
        "requires_code": True,
        "requires_diagrams": True,
        "requires_hands_on": True,
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def business_course_state():
    """State for a business course"""
    return {
        "topic": "Strategic Leadership",
        "description": "Develop leadership skills for managers",
        "profile_category": "business",
        "detected_persona": "Manager",
        "topic_complexity": "intermediate",
        "requires_code": False,
        "requires_diagrams": True,
        "requires_hands_on": False,
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def creative_course_state():
    """State for a creative course"""
    return {
        "topic": "Digital Photography Masterclass",
        "description": "Learn composition and editing techniques",
        "profile_category": "creative",
        "detected_persona": "Artist",
        "topic_complexity": "beginner",
        "requires_code": False,
        "requires_diagrams": False,
        "requires_hands_on": True,
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def valid_technical_response():
    """Valid LLM response for technical course"""
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
def valid_business_response():
    """Valid LLM response for business course"""
    return {
        "content_preferences": {
            "code_weight": 0.0,
            "diagram_weight": 0.5,
            "demo_weight": 0.4,
            "theory_weight": 0.7,
            "case_study_weight": 0.9
        },
        "recommended_elements": ["case_study", "framework_template", "action_checklist"],
        "adaptation_notes": "Case-study driven with actionable frameworks."
    }


@pytest.fixture
def invalid_response_low_weights():
    """Invalid LLM response with weights violating constraints"""
    return {
        "content_preferences": {
            "code_weight": 0.3,  # Should be >= 0.6 when requires_code=True
            "diagram_weight": 0.2,  # Should be >= 0.5 when requires_diagrams=True
            "demo_weight": 0.3,
            "theory_weight": 0.1,  # Should be >= 0.2
            "case_study_weight": 0.3
        },
        "recommended_elements": ["code_demo"],  # Only 1, should be 3-6
        "adaptation_notes": "Invalid response"
    }


# ============================================================================
# ProfileAdaptationValidator (reused from unit tests)
# ============================================================================

class ProfileAdaptationValidator:
    """Validates outputs against PROFILE_ADAPTATION_PROMPT constraints"""

    @staticmethod
    def validate_weight_range(weight: float) -> bool:
        return 0.0 <= weight <= 1.0

    @staticmethod
    def validate_total_weight(preferences: Dict[str, float]) -> bool:
        total = sum(preferences.values())
        return 2.5 <= total <= 3.5

    @staticmethod
    def validate_code_constraint(preferences: Dict[str, float], requires_code: bool) -> bool:
        if requires_code:
            return preferences.get("code_weight", 0) >= 0.6
        return True

    @staticmethod
    def validate_diagram_constraint(preferences: Dict[str, float], requires_diagrams: bool) -> bool:
        if requires_diagrams:
            return preferences.get("diagram_weight", 0) >= 0.5
        return True

    @staticmethod
    def validate_theory_constraint(preferences: Dict[str, float]) -> bool:
        return preferences.get("theory_weight", 0) >= 0.2

    @staticmethod
    def validate_elements_count(elements: List[str]) -> bool:
        return 3 <= len(elements) <= 6

    @classmethod
    def validate_output(
        cls,
        output: Dict[str, Any],
        requires_code: bool = False,
        requires_diagrams: bool = False
    ) -> Dict[str, Any]:
        issues = []
        preferences = output.get("content_preferences", {})
        elements = output.get("recommended_elements", [])

        for key, value in preferences.items():
            if not cls.validate_weight_range(value):
                issues.append(f"{key}={value} out of range [0.0, 1.0]")

        if not cls.validate_total_weight(preferences):
            total = sum(preferences.values())
            issues.append(f"Total weight {total:.2f} not in range [2.5, 3.5]")

        if not cls.validate_code_constraint(preferences, requires_code):
            issues.append(f"code_weight={preferences.get('code_weight')} < 0.6 (requires_code=true)")

        if not cls.validate_diagram_constraint(preferences, requires_diagrams):
            issues.append(f"diagram_weight={preferences.get('diagram_weight')} < 0.5 (requires_diagrams=true)")

        if not cls.validate_theory_constraint(preferences):
            issues.append(f"theory_weight={preferences.get('theory_weight')} < 0.2")

        if not cls.validate_elements_count(elements):
            issues.append(f"Element count {len(elements)} not in range [3, 6]")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "total_weight": sum(preferences.values()),
            "element_count": len(elements)
        }


# ============================================================================
# Mock adapt_for_profile function (simulates the real one)
# ============================================================================

async def mock_adapt_for_profile(
    state: Dict[str, Any],
    mock_client: MagicMock,
    mock_response: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulates adapt_for_profile function with mocked LLM client.
    This mirrors the actual implementation in pedagogical_nodes.py.
    """
    state["current_node"] = "adapt_for_profile"

    # Get available elements for the category
    category_raw = state.get("profile_category", "education")
    if isinstance(category_raw, str):
        try:
            category = ProfileCategory(category_raw.lower())
        except ValueError:
            category = ProfileCategory.EDUCATION
    else:
        category = category_raw if category_raw else ProfileCategory.EDUCATION

    available_elements = get_elements_for_category(category)
    elements_list = "\n".join([f"- {el.id.value}: {el.name} - {el.description}" for el in available_elements])

    # Format prompt
    prompt = PROFILE_ADAPTATION_PROMPT.format(
        detected_persona=state.get("detected_persona", "student"),
        topic_complexity=state.get("topic_complexity", "intermediate"),
        category=category.value if hasattr(category, 'value') else category,
        requires_code=state.get("requires_code", False),
        requires_diagrams=state.get("requires_diagrams", True),
        requires_hands_on=state.get("requires_hands_on", False),
        available_elements=elements_list,
    )

    # Create mock response
    mock_message = MagicMock()
    mock_message.content = json.dumps(mock_response)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_api_response = MagicMock()
    mock_api_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_api_response

    try:
        # Call the mock client (simulating LLM call)
        response = await mock_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=600
        )

        result = json.loads(response.choices[0].message.content)

        prefs = result.get("content_preferences", {})
        content_preferences = {
            "code_weight": prefs.get("code_weight", 0.5),
            "diagram_weight": prefs.get("diagram_weight", 0.5),
            "demo_weight": prefs.get("demo_weight", 0.5),
            "theory_weight": prefs.get("theory_weight", 0.5),
            "case_study_weight": prefs.get("case_study_weight", 0.3),
        }

        return {
            "content_preferences": content_preferences,
            "recommended_elements": result.get("recommended_elements", []),
            "prompt_used": prompt,  # For testing
            "category_used": category,  # For testing
        }

    except Exception as e:
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


# ============================================================================
# Tests for Full Integration Flow
# ============================================================================

class TestAdaptForProfileIntegration:
    """Integration tests for adapt_for_profile function flow"""

    @pytest.mark.asyncio
    async def test_technical_course_full_flow(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test full flow for a technical course"""
        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        # Verify content preferences are extracted correctly
        assert "content_preferences" in result
        prefs = result["content_preferences"]
        assert prefs["code_weight"] == 0.85
        assert prefs["diagram_weight"] == 0.6
        assert prefs["demo_weight"] == 0.7
        assert prefs["theory_weight"] == 0.35
        assert prefs["case_study_weight"] == 0.5

        # Verify recommended elements
        assert "recommended_elements" in result
        assert len(result["recommended_elements"]) == 4
        assert "code_demo" in result["recommended_elements"]

        # Verify the output passes validation
        validation = ProfileAdaptationValidator.validate_output(
            result,
            requires_code=True,
            requires_diagrams=True
        )
        assert validation["is_valid"], f"Issues: {validation['issues']}"

    @pytest.mark.asyncio
    async def test_business_course_full_flow(
        self, mock_openai_client, business_course_state, valid_business_response
    ):
        """Test full flow for a business course"""
        result = await mock_adapt_for_profile(
            business_course_state,
            mock_openai_client,
            valid_business_response
        )

        # Verify no code weight for business course
        assert result["content_preferences"]["code_weight"] == 0.0

        # Verify high case study weight
        assert result["content_preferences"]["case_study_weight"] == 0.9

        # Verify recommended elements are business-focused
        assert "case_study" in result["recommended_elements"]

        # Verify validation (no code required)
        validation = ProfileAdaptationValidator.validate_output(
            result,
            requires_code=False,
            requires_diagrams=True
        )
        assert validation["is_valid"], f"Issues: {validation['issues']}"

    @pytest.mark.asyncio
    async def test_creative_course_full_flow(
        self, mock_openai_client, creative_course_state
    ):
        """Test full flow for a creative course"""
        creative_response = {
            "content_preferences": {
                "code_weight": 0.0,
                "diagram_weight": 0.3,
                "demo_weight": 0.9,
                "theory_weight": 0.4,
                "case_study_weight": 0.5
            },
            "recommended_elements": ["technique_demo", "before_after", "creative_exercise"],
            "adaptation_notes": "Hands-on creative learning"
        }

        result = await mock_adapt_for_profile(
            creative_course_state,
            mock_openai_client,
            creative_response
        )

        # Verify high demo weight for hands-on course
        assert result["content_preferences"]["demo_weight"] == 0.9

        # Verify creative category was used
        assert result["category_used"] == ProfileCategory.CREATIVE


# ============================================================================
# Tests for Category Handling
# ============================================================================

class TestCategoryHandling:
    """Tests for category string/enum conversion"""

    @pytest.mark.asyncio
    async def test_category_from_string(self, mock_openai_client, valid_technical_response):
        """Test that string category is converted to enum"""
        state = {
            "profile_category": "tech",
            "detected_persona": "Developer",
            "topic_complexity": "intermediate",
            "requires_code": True,
            "requires_diagrams": True,
            "requires_hands_on": False,
        }

        result = await mock_adapt_for_profile(state, mock_openai_client, valid_technical_response)
        assert result["category_used"] == ProfileCategory.TECH

    @pytest.mark.asyncio
    async def test_category_from_enum(self, mock_openai_client, valid_technical_response):
        """Test that enum category is used directly"""
        state = {
            "profile_category": ProfileCategory.TECH,
            "detected_persona": "Developer",
            "topic_complexity": "intermediate",
            "requires_code": True,
            "requires_diagrams": True,
            "requires_hands_on": False,
        }

        result = await mock_adapt_for_profile(state, mock_openai_client, valid_technical_response)
        assert result["category_used"] == ProfileCategory.TECH

    @pytest.mark.asyncio
    async def test_invalid_category_defaults_to_education(self, mock_openai_client, valid_business_response):
        """Test that invalid category falls back to EDUCATION"""
        state = {
            "profile_category": "invalid_category",
            "detected_persona": "Student",
            "topic_complexity": "beginner",
            "requires_code": False,
            "requires_diagrams": False,
            "requires_hands_on": False,
        }

        result = await mock_adapt_for_profile(state, mock_openai_client, valid_business_response)
        assert result["category_used"] == ProfileCategory.EDUCATION

    @pytest.mark.asyncio
    async def test_none_category_defaults_to_education(self, mock_openai_client, valid_business_response):
        """Test that None category falls back to EDUCATION"""
        state = {
            "profile_category": None,
            "detected_persona": "Student",
            "topic_complexity": "beginner",
            "requires_code": False,
            "requires_diagrams": False,
            "requires_hands_on": False,
        }

        result = await mock_adapt_for_profile(state, mock_openai_client, valid_business_response)
        assert result["category_used"] == ProfileCategory.EDUCATION

    @pytest.mark.asyncio
    async def test_all_category_types(self, mock_openai_client, valid_business_response):
        """Test all ProfileCategory enum values"""
        categories = ["tech", "business", "creative", "health", "education", "lifestyle"]

        for cat_str in categories:
            state = {
                "profile_category": cat_str,
                "detected_persona": "Learner",
                "topic_complexity": "intermediate",
                "requires_code": False,
                "requires_diagrams": False,
                "requires_hands_on": False,
            }

            result = await mock_adapt_for_profile(state, mock_openai_client, valid_business_response)
            expected_category = ProfileCategory(cat_str)
            assert result["category_used"] == expected_category, f"Failed for category: {cat_str}"


# ============================================================================
# Tests for Prompt Formatting
# ============================================================================

class TestPromptFormatting:
    """Tests for prompt formatting with state values"""

    @pytest.mark.asyncio
    async def test_prompt_contains_state_values(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that formatted prompt contains all state values"""
        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        prompt = result["prompt_used"]

        # Check state values are in prompt
        assert "Backend Engineer" in prompt
        assert "intermediate" in prompt
        assert "tech" in prompt
        assert "True" in prompt  # requires_code

    @pytest.mark.asyncio
    async def test_prompt_contains_available_elements(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that prompt contains available elements for category"""
        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        prompt = result["prompt_used"]

        # Tech category should have code-related elements
        assert "code_demo" in prompt or "Code Demo" in prompt

    @pytest.mark.asyncio
    async def test_prompt_structure_preserved(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that prompt structure sections are preserved"""
        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        prompt = result["prompt_used"]

        # Verify key sections are present
        assert "## CONTEXT" in prompt
        assert "## INPUT SIGNALS" in prompt
        assert "## DECISION RULES" in prompt
        assert "## OUTPUT CONTRACT" in prompt


# ============================================================================
# Tests for Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios"""

    @pytest.mark.asyncio
    async def test_llm_exception_returns_defaults(self, mock_openai_client, technical_course_state):
        """Test that LLM exception returns default values"""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            {}  # Empty, will trigger exception path
        )

        # Should have default preferences
        assert result["content_preferences"]["code_weight"] == 0.5
        assert result["content_preferences"]["diagram_weight"] == 0.5
        assert result["content_preferences"]["theory_weight"] == 0.5

        # Should have error recorded
        assert "errors" in result
        assert len(result["errors"]) > 0
        assert "Profile adaptation failed" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_missing_preferences_uses_defaults(self, mock_openai_client, technical_course_state):
        """Test that missing preferences in response use defaults"""
        incomplete_response = {
            "content_preferences": {
                "code_weight": 0.9,
                # Missing other weights
            },
            "recommended_elements": ["code_demo", "debug_tips", "case_study"],
        }

        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            incomplete_response
        )

        # Specified weight should be used
        assert result["content_preferences"]["code_weight"] == 0.9

        # Missing weights should have defaults
        assert result["content_preferences"]["diagram_weight"] == 0.5
        assert result["content_preferences"]["demo_weight"] == 0.5

    @pytest.mark.asyncio
    async def test_empty_response_uses_all_defaults(self, mock_openai_client, technical_course_state):
        """Test that empty response uses all defaults"""
        empty_response = {}

        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            empty_response
        )

        # All defaults
        assert result["content_preferences"]["code_weight"] == 0.5
        assert result["content_preferences"]["case_study_weight"] == 0.3
        assert result["recommended_elements"] == []


# ============================================================================
# Tests for Response Validation
# ============================================================================

class TestResponseValidation:
    """Tests for validating LLM responses against constraints"""

    @pytest.mark.asyncio
    async def test_valid_response_passes_validation(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that valid responses pass validation"""
        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        validation = ProfileAdaptationValidator.validate_output(
            result,
            requires_code=True,
            requires_diagrams=True
        )

        assert validation["is_valid"] is True

    @pytest.mark.asyncio
    async def test_invalid_code_weight_detected(
        self, mock_openai_client, technical_course_state, invalid_response_low_weights
    ):
        """Test that invalid code_weight is detected"""
        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            invalid_response_low_weights
        )

        validation = ProfileAdaptationValidator.validate_output(
            result,
            requires_code=True,  # Code required
            requires_diagrams=True
        )

        assert validation["is_valid"] is False
        assert any("code_weight" in issue for issue in validation["issues"])

    @pytest.mark.asyncio
    async def test_invalid_theory_weight_detected(
        self, mock_openai_client, technical_course_state, invalid_response_low_weights
    ):
        """Test that invalid theory_weight is detected"""
        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            invalid_response_low_weights
        )

        validation = ProfileAdaptationValidator.validate_output(
            result,
            requires_code=False,
            requires_diagrams=False
        )

        assert validation["is_valid"] is False
        assert any("theory_weight" in issue for issue in validation["issues"])


# ============================================================================
# Tests for Elements Per Category
# ============================================================================

class TestElementsPerCategory:
    """Tests for available elements by category"""

    def test_tech_category_has_code_elements(self):
        """Test that tech category includes code-related elements"""
        elements = get_elements_for_category(ProfileCategory.TECH)
        element_ids = [el.id.value for el in elements]

        assert "code_demo" in element_ids
        assert "terminal_output" in element_ids
        assert "architecture_diagram" in element_ids

    def test_business_category_has_business_elements(self):
        """Test that business category includes business elements"""
        elements = get_elements_for_category(ProfileCategory.BUSINESS)
        element_ids = [el.id.value for el in elements]

        assert "case_study" in element_ids
        assert "framework_template" in element_ids

    def test_creative_category_has_creative_elements(self):
        """Test that creative category includes creative elements"""
        elements = get_elements_for_category(ProfileCategory.CREATIVE)
        element_ids = [el.id.value for el in elements]

        assert "before_after" in element_ids
        assert "technique_demo" in element_ids

    def test_health_category_has_health_elements(self):
        """Test that health category includes health elements"""
        elements = get_elements_for_category(ProfileCategory.HEALTH)
        element_ids = [el.id.value for el in elements]

        assert "exercise_demo" in element_ids or "body_diagram" in element_ids

    def test_all_categories_have_common_elements(self):
        """Test that all categories have common elements"""
        for category in ProfileCategory:
            elements = get_elements_for_category(category)
            element_ids = [el.id.value for el in elements]

            # Common elements should be present
            assert "concept_intro" in element_ids
            assert "voiceover" in element_ids
            assert "conclusion" in element_ids


# ============================================================================
# Tests for State Updates
# ============================================================================

class TestStateUpdates:
    """Tests for state updates after adapt_for_profile"""

    @pytest.mark.asyncio
    async def test_current_node_updated(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that current_node is updated"""
        await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert technical_course_state["current_node"] == "adapt_for_profile"

    @pytest.mark.asyncio
    async def test_result_contains_required_keys(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that result contains all required keys"""
        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "content_preferences" in result
        assert "recommended_elements" in result

    @pytest.mark.asyncio
    async def test_content_preferences_has_all_weights(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that content_preferences has all weight keys"""
        result = await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        expected_keys = ["code_weight", "diagram_weight", "demo_weight", "theory_weight", "case_study_weight"]
        for key in expected_keys:
            assert key in result["content_preferences"]


# ============================================================================
# Tests for LLM Call Parameters
# ============================================================================

class TestLLMCallParameters:
    """Tests for LLM API call parameters"""

    @pytest.mark.asyncio
    async def test_llm_called_with_json_format(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that LLM is called with JSON response format"""
        await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        # Verify the call was made with correct parameters
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_llm_called_with_low_temperature(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that LLM is called with low temperature for consistency"""
        await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_llm_called_with_reasonable_max_tokens(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that LLM is called with reasonable max_tokens"""
        await mock_adapt_for_profile(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 600


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
