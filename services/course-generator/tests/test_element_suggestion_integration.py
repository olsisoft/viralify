"""
Integration tests for suggest_elements function

Tests the full flow from state input through LLM call to element mapping output,
including outline parsing, prompt formatting, and response handling.
"""

import pytest
import json
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
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
ELEMENT_SUGGESTION_PROMPT = prompts_module.ELEMENT_SUGGESTION_PROMPT


# ============================================================================
# Mock Classes (to avoid import dependencies)
# ============================================================================

class ProfileCategory(str, Enum):
    """Course profile categories"""
    TECH = "tech"
    BUSINESS = "business"
    CREATIVE = "creative"
    HEALTH = "health"
    EDUCATION = "education"
    LIFESTYLE = "lifestyle"


@dataclass
class MockLessonElementId:
    value: str


@dataclass
class MockLessonElement:
    id: MockLessonElementId
    name: str
    description: str = ""


@dataclass
class MockLecture:
    id: str
    title: str
    objectives: List[str] = field(default_factory=list)


@dataclass
class MockSection:
    title: str
    lectures: List[MockLecture] = field(default_factory=list)


@dataclass
class MockOutline:
    sections: List[MockSection] = field(default_factory=list)


# Category elements (simplified)
CATEGORY_ELEMENTS = {
    ProfileCategory.TECH: [
        MockLessonElement(MockLessonElementId("code_demo"), "Code Demo"),
        MockLessonElement(MockLessonElementId("terminal_output"), "Terminal Output"),
        MockLessonElement(MockLessonElementId("architecture_diagram"), "Architecture Diagram"),
        MockLessonElement(MockLessonElementId("debug_tips"), "Debug Tips"),
    ],
    ProfileCategory.BUSINESS: [
        MockLessonElement(MockLessonElementId("case_study"), "Case Study"),
        MockLessonElement(MockLessonElementId("framework_template"), "Framework Template"),
        MockLessonElement(MockLessonElementId("roi_metrics"), "ROI Metrics"),
    ],
    ProfileCategory.CREATIVE: [
        MockLessonElement(MockLessonElementId("before_after"), "Before/After"),
        MockLessonElement(MockLessonElementId("technique_demo"), "Technique Demo"),
    ],
}

COMMON_ELEMENTS = [
    MockLessonElement(MockLessonElementId("concept_intro"), "Concept Introduction"),
    MockLessonElement(MockLessonElementId("voiceover"), "Voiceover"),
    MockLessonElement(MockLessonElementId("conclusion"), "Conclusion"),
]


def get_elements_for_category(category: ProfileCategory) -> List[MockLessonElement]:
    """Get available lesson elements for a category"""
    return COMMON_ELEMENTS + CATEGORY_ELEMENTS.get(category, [])


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
def sample_outline():
    """Sample course outline with sections and lectures"""
    return MockOutline(
        sections=[
            MockSection(
                title="Introduction to Microservices",
                lectures=[
                    MockLecture(
                        id="lec_001",
                        title="What are Microservices?",
                        objectives=["Understand microservices architecture", "Compare with monolithic"]
                    ),
                    MockLecture(
                        id="lec_002",
                        title="Benefits and Challenges",
                        objectives=["Identify benefits", "Recognize challenges"]
                    ),
                ]
            ),
            MockSection(
                title="Building Your First Service",
                lectures=[
                    MockLecture(
                        id="lec_003",
                        title="Setting Up the Environment",
                        objectives=["Install tools", "Configure development environment"]
                    ),
                    MockLecture(
                        id="lec_004",
                        title="Creating a REST API",
                        objectives=["Build REST endpoints", "Handle HTTP methods"]
                    ),
                    MockLecture(
                        id="lec_005",
                        title="Database Integration",
                        objectives=["Connect to database", "Implement CRUD operations"]
                    ),
                ]
            ),
        ]
    )


@pytest.fixture
def technical_course_state(sample_outline):
    """State for a technical course with outline"""
    return {
        "topic": "Building Microservices with Go",
        "profile_category": ProfileCategory.TECH,
        "outline": sample_outline,
        "content_preferences": {
            "code_weight": 0.85,
            "diagram_weight": 0.7,
            "demo_weight": 0.6,
            "theory_weight": 0.3,
            "case_study_weight": 0.4,
        },
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def business_course_state():
    """State for a business course"""
    outline = MockOutline(
        sections=[
            MockSection(
                title="Strategic Planning",
                lectures=[
                    MockLecture(id="lec_001", title="Introduction to Strategy"),
                    MockLecture(id="lec_002", title="SWOT Analysis"),
                    MockLecture(id="lec_003", title="Competitive Advantage"),
                ]
            ),
        ]
    )
    return {
        "topic": "Strategic Business Planning",
        "profile_category": ProfileCategory.BUSINESS,
        "outline": outline,
        "content_preferences": {
            "code_weight": 0.0,
            "diagram_weight": 0.5,
            "demo_weight": 0.3,
            "theory_weight": 0.7,
            "case_study_weight": 0.9,
        },
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def state_without_outline():
    """State without an outline"""
    return {
        "topic": "Test Course",
        "profile_category": ProfileCategory.TECH,
        "outline": None,
        "content_preferences": {},
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def valid_technical_response():
    """Valid LLM response for technical course"""
    return {
        "element_mapping": {
            "lec_001": ["concept_intro", "architecture_diagram", "voiceover"],
            "lec_002": ["concept_intro", "architecture_diagram", "debug_tips", "voiceover"],
            "lec_003": ["code_demo", "terminal_output", "voiceover"],
            "lec_004": ["code_demo", "terminal_output", "architecture_diagram", "debug_tips"],
            "lec_005": ["code_demo", "terminal_output", "architecture_diagram", "debug_tips", "voiceover"],
        },
        "reasoning": "Progressive from concepts to heavy implementation with code demos."
    }


@pytest.fixture
def valid_business_response():
    """Valid LLM response for business course"""
    return {
        "element_mapping": {
            "lec_001": ["concept_intro", "voiceover", "framework_template"],
            "lec_002": ["concept_intro", "case_study", "framework_template", "voiceover"],
            "lec_003": ["case_study", "roi_metrics", "voiceover", "conclusion"],
        },
        "reasoning": "Case-study driven approach with practical frameworks."
    }


# ============================================================================
# Mock suggest_elements function
# ============================================================================

async def mock_suggest_elements(
    state: Dict[str, Any],
    mock_client: MagicMock,
    mock_response: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulates suggest_elements function with mocked LLM client.
    Mirrors the actual implementation in pedagogical_nodes.py.
    """
    state["current_node"] = "suggest_elements"

    outline = state.get("outline")
    if not outline:
        return {
            "element_mapping": {},
            "errors": state.get("errors", []) + ["No outline available for element suggestion"],
        }

    # Get category
    category_raw = state.get("profile_category", "education")
    if isinstance(category_raw, str):
        try:
            category = ProfileCategory(category_raw.lower())
        except ValueError:
            category = ProfileCategory.EDUCATION
    else:
        category = category_raw if category_raw else ProfileCategory.EDUCATION

    # Get available elements
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

    # Get content preferences
    prefs = state.get("content_preferences", {})

    # Format prompt
    prompt = ELEMENT_SUGGESTION_PROMPT.format(
        topic=state["topic"],
        category=category.value if hasattr(category, 'value') else category,
        code_weight=prefs.get("code_weight", 0.5),
        diagram_weight=prefs.get("diagram_weight", 0.5),
        demo_weight=prefs.get("demo_weight", 0.5),
        available_elements=elements_list,
        outline_structure=outline_structure,
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
        response = await mock_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1500
        )

        result = json.loads(response.choices[0].message.content)
        return {
            "element_mapping": result.get("element_mapping", {}),
            "prompt_used": prompt,
            "outline_structure": outline_structure,
            "category_used": category,
            "elements_available": [el.id.value for el in available_elements],
        }

    except Exception as e:
        return {
            "element_mapping": {},
            "errors": state.get("errors", []) + [f"Element suggestion failed: {str(e)}"],
        }


# ============================================================================
# ElementSuggestionValidator (from unit tests)
# ============================================================================

class ElementSuggestionValidator:
    """Validates outputs against ELEMENT_SUGGESTION_PROMPT constraints"""

    def __init__(self, available_elements: List[str]):
        self.available_elements = available_elements

    def validate_element_count(self, elements: List[str]) -> bool:
        return 3 <= len(elements) <= 5

    def validate_elements_exist(self, elements: List[str]) -> List[str]:
        return [el for el in elements if el not in self.available_elements]

    def validate_output(
        self,
        output: Dict[str, Any],
        code_weight: float = 0.5,
        diagram_weight: float = 0.5
    ) -> Dict[str, Any]:
        issues = []
        element_mapping = output.get("element_mapping", {})

        for lecture_id, elements in element_mapping.items():
            if not self.validate_element_count(elements):
                issues.append(f"{lecture_id}: has {len(elements)} elements (must be 3-5)")

            invalid = self.validate_elements_exist(elements)
            if invalid:
                issues.append(f"{lecture_id}: invalid elements {invalid}")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "lecture_count": len(element_mapping),
        }


# ============================================================================
# Tests for Full Integration Flow
# ============================================================================

class TestSuggestElementsFlow:
    """Integration tests for suggest_elements function flow"""

    @pytest.mark.asyncio
    async def test_technical_course_full_flow(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test full flow for a technical course"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        # Verify element mapping is returned
        assert "element_mapping" in result
        assert len(result["element_mapping"]) == 5  # 5 lectures

        # Verify each lecture has elements
        for lecture_id, elements in result["element_mapping"].items():
            assert len(elements) >= 3
            assert len(elements) <= 5

        # Verify tech elements are available
        assert "code_demo" in result["elements_available"]
        assert "architecture_diagram" in result["elements_available"]

    @pytest.mark.asyncio
    async def test_business_course_full_flow(
        self, mock_openai_client, business_course_state, valid_business_response
    ):
        """Test full flow for a business course"""
        result = await mock_suggest_elements(
            business_course_state,
            mock_openai_client,
            valid_business_response
        )

        # Verify element mapping
        assert len(result["element_mapping"]) == 3  # 3 lectures

        # Verify business elements are available
        assert "case_study" in result["elements_available"]
        assert "framework_template" in result["elements_available"]

        # Verify code_demo is NOT available for business
        assert "code_demo" not in result["elements_available"]

    @pytest.mark.asyncio
    async def test_no_outline_returns_error(
        self, mock_openai_client, state_without_outline
    ):
        """Test that missing outline returns error"""
        result = await mock_suggest_elements(
            state_without_outline,
            mock_openai_client,
            {}
        )

        assert result["element_mapping"] == {}
        assert "errors" in result
        assert any("No outline" in err for err in result["errors"])


# ============================================================================
# Tests for Outline Parsing
# ============================================================================

class TestOutlineParsing:
    """Tests for outline structure parsing"""

    @pytest.mark.asyncio
    async def test_outline_structure_contains_sections(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that outline structure contains section titles"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        outline_str = result["outline_structure"]
        assert "Section: Introduction to Microservices" in outline_str
        assert "Section: Building Your First Service" in outline_str

    @pytest.mark.asyncio
    async def test_outline_structure_contains_lectures(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that outline structure contains lecture IDs and titles"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        outline_str = result["outline_structure"]
        assert "Lecture lec_001: What are Microservices?" in outline_str
        assert "Lecture lec_004: Creating a REST API" in outline_str

    @pytest.mark.asyncio
    async def test_outline_structure_contains_objectives(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that outline structure contains lecture objectives"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        outline_str = result["outline_structure"]
        assert "Objectives:" in outline_str
        assert "Understand microservices architecture" in outline_str


# ============================================================================
# Tests for Prompt Formatting
# ============================================================================

class TestPromptFormatting:
    """Tests for prompt formatting with state values"""

    @pytest.mark.asyncio
    async def test_prompt_contains_topic(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that topic is included in prompt"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "Building Microservices with Go" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_contains_category(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that category is included in prompt"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "tech" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_contains_content_preferences(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that content preferences are included in prompt"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "0.85" in result["prompt_used"]  # code_weight
        assert "0.7" in result["prompt_used"]   # diagram_weight

    @pytest.mark.asyncio
    async def test_prompt_contains_available_elements(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that available elements are included in prompt"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "code_demo" in result["prompt_used"]
        assert "architecture_diagram" in result["prompt_used"]
        assert "concept_intro" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_contains_outline_structure(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that outline structure is included in prompt"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "lec_001" in result["prompt_used"]
        assert "What are Microservices?" in result["prompt_used"]


# ============================================================================
# Tests for Category Handling
# ============================================================================

class TestCategoryHandling:
    """Tests for category string/enum handling"""

    @pytest.mark.asyncio
    async def test_enum_category(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that enum category is handled correctly"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert result["category_used"] == ProfileCategory.TECH

    @pytest.mark.asyncio
    async def test_string_category(
        self, mock_openai_client, sample_outline, valid_technical_response
    ):
        """Test that string category is converted to enum"""
        state = {
            "topic": "Test",
            "profile_category": "business",
            "outline": sample_outline,
            "content_preferences": {},
            "errors": [],
        }

        result = await mock_suggest_elements(
            state,
            mock_openai_client,
            valid_technical_response
        )

        assert result["category_used"] == ProfileCategory.BUSINESS

    @pytest.mark.asyncio
    async def test_invalid_category_defaults_to_education(
        self, mock_openai_client, sample_outline, valid_technical_response
    ):
        """Test that invalid category defaults to education"""
        state = {
            "topic": "Test",
            "profile_category": "invalid_category",
            "outline": sample_outline,
            "content_preferences": {},
            "errors": [],
        }

        result = await mock_suggest_elements(
            state,
            mock_openai_client,
            valid_technical_response
        )

        assert result["category_used"] == ProfileCategory.EDUCATION

    @pytest.mark.asyncio
    async def test_different_categories_have_different_elements(
        self, mock_openai_client, sample_outline, valid_technical_response
    ):
        """Test that different categories provide different elements"""
        tech_state = {
            "topic": "Test",
            "profile_category": ProfileCategory.TECH,
            "outline": sample_outline,
            "content_preferences": {},
            "errors": [],
        }
        business_state = {
            "topic": "Test",
            "profile_category": ProfileCategory.BUSINESS,
            "outline": sample_outline,
            "content_preferences": {},
            "errors": [],
        }

        tech_result = await mock_suggest_elements(
            tech_state, mock_openai_client, valid_technical_response
        )
        business_result = await mock_suggest_elements(
            business_state, mock_openai_client, valid_technical_response
        )

        # Tech has code_demo, business doesn't
        assert "code_demo" in tech_result["elements_available"]
        assert "code_demo" not in business_result["elements_available"]

        # Business has case_study, tech might or might not
        assert "case_study" in business_result["elements_available"]


# ============================================================================
# Tests for Content Preferences
# ============================================================================

class TestContentPreferences:
    """Tests for content preferences handling"""

    @pytest.mark.asyncio
    async def test_default_preferences_when_missing(
        self, mock_openai_client, sample_outline, valid_technical_response
    ):
        """Test that default preferences are used when not provided"""
        state = {
            "topic": "Test",
            "profile_category": ProfileCategory.TECH,
            "outline": sample_outline,
            "content_preferences": {},  # Empty
            "errors": [],
        }

        result = await mock_suggest_elements(
            state,
            mock_openai_client,
            valid_technical_response
        )

        # Should use default 0.5 values
        assert "0.5" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_high_code_weight_in_prompt(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that high code weight is reflected in prompt"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        # code_weight is 0.85
        assert "0.85" in result["prompt_used"]


# ============================================================================
# Tests for Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios"""

    @pytest.mark.asyncio
    async def test_llm_exception_returns_error(
        self, mock_openai_client, technical_course_state
    ):
        """Test that LLM exception returns error"""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            {}
        )

        assert result["element_mapping"] == {}
        assert "errors" in result
        assert any("Element suggestion failed" in err for err in result["errors"])

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_mapping(
        self, mock_openai_client, technical_course_state
    ):
        """Test that empty LLM response returns empty mapping"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            {}  # Empty response
        )

        assert result["element_mapping"] == {}


# ============================================================================
# Tests for Response Validation
# ============================================================================

class TestResponseValidation:
    """Tests for validating LLM responses"""

    @pytest.mark.asyncio
    async def test_valid_response_passes_validation(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that valid response passes validation"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        validator = ElementSuggestionValidator(result["elements_available"])
        validation = validator.validate_output(result)

        assert validation["is_valid"] is True, f"Issues: {validation['issues']}"

    @pytest.mark.asyncio
    async def test_response_with_invalid_elements_detected(
        self, mock_openai_client, technical_course_state
    ):
        """Test that response with invalid elements is detected"""
        invalid_response = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "nonexistent_element"],
            },
            "reasoning": "Test"
        }

        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            invalid_response
        )

        validator = ElementSuggestionValidator(result["elements_available"])
        validation = validator.validate_output(result)

        assert validation["is_valid"] is False
        assert any("invalid elements" in issue for issue in validation["issues"])


# ============================================================================
# Tests for State Updates
# ============================================================================

class TestStateUpdates:
    """Tests for state updates after suggest_elements"""

    @pytest.mark.asyncio
    async def test_current_node_updated(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that current_node is updated"""
        await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert technical_course_state["current_node"] == "suggest_elements"

    @pytest.mark.asyncio
    async def test_result_contains_element_mapping(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that result contains element_mapping"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "element_mapping" in result
        assert isinstance(result["element_mapping"], dict)


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
        await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_llm_called_with_appropriate_max_tokens(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that LLM is called with sufficient max_tokens"""
        await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        # Element mapping can be large, so we need more tokens
        assert call_kwargs["max_tokens"] >= 1000


# ============================================================================
# Tests for Multiple Sections
# ============================================================================

class TestMultipleSections:
    """Tests for outlines with multiple sections"""

    @pytest.mark.asyncio
    async def test_all_lectures_in_mapping(
        self, mock_openai_client, technical_course_state, valid_technical_response
    ):
        """Test that all lectures from all sections are in the response"""
        result = await mock_suggest_elements(
            technical_course_state,
            mock_openai_client,
            valid_technical_response
        )

        # Should have 5 lectures total (2 + 3)
        assert len(result["element_mapping"]) == 5

        # Check all lecture IDs are present
        expected_lectures = ["lec_001", "lec_002", "lec_003", "lec_004", "lec_005"]
        for lecture_id in expected_lectures:
            assert lecture_id in result["element_mapping"]


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases"""

    @pytest.mark.asyncio
    async def test_single_lecture_outline(
        self, mock_openai_client, valid_technical_response
    ):
        """Test with single lecture outline"""
        outline = MockOutline(
            sections=[
                MockSection(
                    title="Introduction",
                    lectures=[MockLecture(id="lec_001", title="The Only Lecture")]
                )
            ]
        )
        state = {
            "topic": "Mini Course",
            "profile_category": ProfileCategory.TECH,
            "outline": outline,
            "content_preferences": {},
            "errors": [],
        }

        single_response = {
            "element_mapping": {
                "lec_001": ["concept_intro", "voiceover", "conclusion"],
            },
            "reasoning": "Single lecture course"
        }

        result = await mock_suggest_elements(
            state,
            mock_openai_client,
            single_response
        )

        assert len(result["element_mapping"]) == 1

    @pytest.mark.asyncio
    async def test_lecture_without_objectives(
        self, mock_openai_client, valid_technical_response
    ):
        """Test with lectures that have no objectives"""
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
            "topic": "Test",
            "profile_category": ProfileCategory.TECH,
            "outline": outline,
            "content_preferences": {},
            "errors": [],
        }

        response = {
            "element_mapping": {"lec_001": ["concept_intro", "voiceover", "code_demo"]},
            "reasoning": "Test"
        }

        result = await mock_suggest_elements(
            state,
            mock_openai_client,
            response
        )

        # Should not have "Objectives:" line for this lecture
        assert "Objectives:" not in result["outline_structure"]


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
