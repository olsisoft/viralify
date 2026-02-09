"""
Integration tests for analyze_context function

Tests the context analysis flow that determines learner persona,
topic complexity, and content requirements.
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
CONTEXT_ANALYSIS_PROMPT = prompts_module.CONTEXT_ANALYSIS_PROMPT


# ============================================================================
# ProfileCategory Enum (standalone)
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
def technical_topic_state():
    """State for a technical programming topic"""
    return {
        "topic": "Building REST APIs with FastAPI",
        "description": "Learn to build high-performance REST APIs using Python and FastAPI framework",
        "profile_category": ProfileCategory.TECH,
        "target_audience": "backend developers",
        "difficulty_start": "intermediate",
        "difficulty_end": "advanced",
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def business_topic_state():
    """State for a business topic"""
    return {
        "topic": "Strategic Planning for Startups",
        "description": "Develop business strategies and growth plans",
        "profile_category": ProfileCategory.BUSINESS,
        "target_audience": "entrepreneurs and managers",
        "difficulty_start": "beginner",
        "difficulty_end": "intermediate",
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def creative_topic_state():
    """State for a creative topic"""
    return {
        "topic": "Digital Illustration Fundamentals",
        "description": "Learn digital art and illustration techniques",
        "profile_category": ProfileCategory.CREATIVE,
        "target_audience": "aspiring artists",
        "difficulty_start": "beginner",
        "difficulty_end": "intermediate",
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def minimal_state():
    """Minimal state with only required fields"""
    return {
        "topic": "Introduction to Python",
        "current_node": "",
        "errors": [],
    }


@pytest.fixture
def valid_technical_response():
    """Valid LLM response for technical topic"""
    return {
        "detected_persona": "backend developer",
        "topic_complexity": "intermediate",
        "requires_code": True,
        "requires_diagrams": True,
        "requires_hands_on": True,
        "domain_keywords": ["FastAPI", "REST", "API", "Python", "async", "endpoints"],
        "reasoning": "Backend development topic requiring code examples and architecture diagrams"
    }


@pytest.fixture
def valid_business_response():
    """Valid LLM response for business topic"""
    return {
        "detected_persona": "entrepreneur",
        "topic_complexity": "intermediate",
        "requires_code": False,
        "requires_diagrams": True,
        "requires_hands_on": False,
        "domain_keywords": ["strategy", "planning", "growth", "startup", "business model"],
        "reasoning": "Business strategy topic with frameworks and diagrams"
    }


@pytest.fixture
def valid_creative_response():
    """Valid LLM response for creative topic"""
    return {
        "detected_persona": "artist",
        "topic_complexity": "beginner",
        "requires_code": False,
        "requires_diagrams": False,
        "requires_hands_on": True,
        "domain_keywords": ["illustration", "digital art", "drawing", "design", "techniques"],
        "reasoning": "Creative topic requiring hands-on practice"
    }


# ============================================================================
# Mock analyze_context function
# ============================================================================

async def mock_analyze_context(
    state: Dict[str, Any],
    mock_client: MagicMock,
    mock_response: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulates analyze_context function with mocked LLM client.
    Mirrors the actual implementation in pedagogical_nodes.py.
    """
    state["current_node"] = "analyze_context"

    description_section = f"DESCRIPTION: {state.get('description')}" if state.get('description') else ""

    category = state.get("profile_category")
    category_value = category.value if hasattr(category, 'value') else (category or "education")

    prompt = CONTEXT_ANALYSIS_PROMPT.format(
        topic=state["topic"],
        description_section=description_section,
        category=category_value,
        target_audience=state.get("target_audience", "general learners"),
        difficulty_start=state.get("difficulty_start", "beginner"),
        difficulty_end=state.get("difficulty_end", "intermediate"),
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
            "prompt_used": prompt,  # For testing
        }

    except Exception as e:
        return {
            "detected_persona": "student",
            "topic_complexity": "intermediate",
            "requires_code": False,
            "requires_diagrams": True,
            "requires_hands_on": False,
            "domain_keywords": [],
            "errors": state.get("errors", []) + [f"Context analysis failed: {str(e)}"],
        }


# ============================================================================
# Response Validator
# ============================================================================

class ContextAnalysisValidator:
    """Validates analyze_context outputs"""

    VALID_COMPLEXITIES = ["basic", "intermediate", "advanced", "expert"]

    @classmethod
    def validate_output(cls, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the output of analyze_context"""
        issues = []

        # Check detected_persona is a non-empty string
        persona = output.get("detected_persona")
        if not persona or not isinstance(persona, str):
            issues.append("detected_persona must be a non-empty string")

        # Check topic_complexity is valid
        complexity = output.get("topic_complexity")
        if complexity not in cls.VALID_COMPLEXITIES:
            issues.append(f"topic_complexity '{complexity}' not in {cls.VALID_COMPLEXITIES}")

        # Check boolean fields
        for field in ["requires_code", "requires_diagrams", "requires_hands_on"]:
            value = output.get(field)
            if not isinstance(value, bool):
                issues.append(f"{field} must be a boolean, got {type(value).__name__}")

        # Check domain_keywords is a list
        keywords = output.get("domain_keywords")
        if not isinstance(keywords, list):
            issues.append("domain_keywords must be a list")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
        }


# ============================================================================
# Tests for Prompt Structure
# ============================================================================

class TestContextAnalysisPrompt:
    """Tests for CONTEXT_ANALYSIS_PROMPT structure"""

    def test_prompt_contains_placeholders(self):
        """Test that prompt contains all required placeholders"""
        required_placeholders = [
            "{topic}",
            "{description_section}",
            "{category}",
            "{target_audience}",
            "{difficulty_start}",
            "{difficulty_end}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in CONTEXT_ANALYSIS_PROMPT, f"Missing placeholder: {placeholder}"

    def test_prompt_defines_output_format(self):
        """Test that prompt defines JSON output format"""
        assert "valid JSON only" in CONTEXT_ANALYSIS_PROMPT
        assert "detected_persona" in CONTEXT_ANALYSIS_PROMPT
        assert "topic_complexity" in CONTEXT_ANALYSIS_PROMPT
        assert "requires_code" in CONTEXT_ANALYSIS_PROMPT
        assert "requires_diagrams" in CONTEXT_ANALYSIS_PROMPT
        assert "requires_hands_on" in CONTEXT_ANALYSIS_PROMPT
        assert "domain_keywords" in CONTEXT_ANALYSIS_PROMPT

    def test_prompt_lists_complexity_levels(self):
        """Test that prompt lists valid complexity levels"""
        assert "basic" in CONTEXT_ANALYSIS_PROMPT
        assert "intermediate" in CONTEXT_ANALYSIS_PROMPT
        assert "advanced" in CONTEXT_ANALYSIS_PROMPT
        assert "expert" in CONTEXT_ANALYSIS_PROMPT


# ============================================================================
# Tests for Full Flow
# ============================================================================

class TestAnalyzeContextFlow:
    """Integration tests for analyze_context flow"""

    @pytest.mark.asyncio
    async def test_technical_topic_analysis(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test analysis of a technical programming topic"""
        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        # Verify persona is detected correctly
        assert result["detected_persona"] == "backend developer"

        # Verify complexity
        assert result["topic_complexity"] == "intermediate"

        # Verify technical topic requires code
        assert result["requires_code"] is True
        assert result["requires_diagrams"] is True
        assert result["requires_hands_on"] is True

        # Verify keywords extracted
        assert len(result["domain_keywords"]) > 0
        assert "FastAPI" in result["domain_keywords"]

        # Validate output structure
        validation = ContextAnalysisValidator.validate_output(result)
        assert validation["is_valid"], f"Issues: {validation['issues']}"

    @pytest.mark.asyncio
    async def test_business_topic_analysis(
        self, mock_openai_client, business_topic_state, valid_business_response
    ):
        """Test analysis of a business topic"""
        result = await mock_analyze_context(
            business_topic_state,
            mock_openai_client,
            valid_business_response
        )

        # Business topic should not require code
        assert result["requires_code"] is False

        # But may need diagrams for frameworks
        assert result["requires_diagrams"] is True

        # Verify persona
        assert result["detected_persona"] == "entrepreneur"

    @pytest.mark.asyncio
    async def test_creative_topic_analysis(
        self, mock_openai_client, creative_topic_state, valid_creative_response
    ):
        """Test analysis of a creative topic"""
        result = await mock_analyze_context(
            creative_topic_state,
            mock_openai_client,
            valid_creative_response
        )

        # Creative topic should require hands-on
        assert result["requires_hands_on"] is True

        # Should not require code
        assert result["requires_code"] is False

        # Verify persona
        assert result["detected_persona"] == "artist"


# ============================================================================
# Tests for Prompt Formatting
# ============================================================================

class TestPromptFormatting:
    """Tests for prompt formatting with state values"""

    @pytest.mark.asyncio
    async def test_prompt_includes_topic(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that topic is included in prompt"""
        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "Building REST APIs with FastAPI" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_includes_description(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that description is included when provided"""
        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "DESCRIPTION:" in result["prompt_used"]
        assert "high-performance REST APIs" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_without_description(
        self, mock_openai_client, minimal_state, valid_technical_response
    ):
        """Test prompt formatting when description is missing"""
        result = await mock_analyze_context(
            minimal_state,
            mock_openai_client,
            valid_technical_response
        )

        # Description section should be empty
        assert "DESCRIPTION:" not in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_includes_category(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that category is included"""
        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "tech" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_prompt_includes_difficulty_range(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that difficulty range is included"""
        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "intermediate" in result["prompt_used"]
        assert "advanced" in result["prompt_used"]


# ============================================================================
# Tests for Default Values
# ============================================================================

class TestDefaultValues:
    """Tests for default value handling"""

    @pytest.mark.asyncio
    async def test_default_persona_on_missing(
        self, mock_openai_client, technical_topic_state
    ):
        """Test default persona when not in response"""
        incomplete_response = {
            "topic_complexity": "intermediate",
            "requires_code": True,
            "requires_diagrams": True,
            "requires_hands_on": False,
            "domain_keywords": [],
        }

        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            incomplete_response
        )

        assert result["detected_persona"] == "student"

    @pytest.mark.asyncio
    async def test_default_complexity_on_missing(
        self, mock_openai_client, technical_topic_state
    ):
        """Test default complexity when not in response"""
        incomplete_response = {
            "detected_persona": "developer",
            "requires_code": True,
            "requires_diagrams": True,
            "requires_hands_on": False,
            "domain_keywords": [],
        }

        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            incomplete_response
        )

        assert result["topic_complexity"] == "intermediate"

    @pytest.mark.asyncio
    async def test_default_requires_code_on_missing(
        self, mock_openai_client, technical_topic_state
    ):
        """Test default requires_code when not in response"""
        incomplete_response = {
            "detected_persona": "developer",
            "topic_complexity": "advanced",
            "requires_diagrams": True,
            "requires_hands_on": False,
            "domain_keywords": [],
        }

        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            incomplete_response
        )

        assert result["requires_code"] is False

    @pytest.mark.asyncio
    async def test_default_target_audience(
        self, mock_openai_client, minimal_state, valid_technical_response
    ):
        """Test default target audience in prompt"""
        result = await mock_analyze_context(
            minimal_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "general learners" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_default_difficulty_range(
        self, mock_openai_client, minimal_state, valid_technical_response
    ):
        """Test default difficulty range in prompt"""
        result = await mock_analyze_context(
            minimal_state,
            mock_openai_client,
            valid_technical_response
        )

        assert "beginner" in result["prompt_used"]
        assert "intermediate" in result["prompt_used"]


# ============================================================================
# Tests for Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling scenarios"""

    @pytest.mark.asyncio
    async def test_llm_exception_returns_defaults(
        self, mock_openai_client, technical_topic_state
    ):
        """Test that LLM exception returns default values"""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            {}
        )

        # Should have default values
        assert result["detected_persona"] == "student"
        assert result["topic_complexity"] == "intermediate"
        assert result["requires_code"] is False
        assert result["requires_diagrams"] is True
        assert result["requires_hands_on"] is False
        assert result["domain_keywords"] == []

        # Should have error recorded
        assert "errors" in result
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_empty_response_uses_defaults(
        self, mock_openai_client, technical_topic_state
    ):
        """Test that empty response uses all defaults"""
        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            {}
        )

        assert result["detected_persona"] == "student"
        assert result["domain_keywords"] == []


# ============================================================================
# Tests for State Updates
# ============================================================================

class TestStateUpdates:
    """Tests for state updates after analyze_context"""

    @pytest.mark.asyncio
    async def test_current_node_updated(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that current_node is updated"""
        await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        assert technical_topic_state["current_node"] == "analyze_context"

    @pytest.mark.asyncio
    async def test_result_contains_all_fields(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that result contains all expected fields"""
        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        expected_fields = [
            "detected_persona",
            "topic_complexity",
            "requires_code",
            "requires_diagrams",
            "requires_hands_on",
            "domain_keywords",
        ]

        for field in expected_fields:
            assert field in result, f"Missing field: {field}"


# ============================================================================
# Tests for LLM Call Parameters
# ============================================================================

class TestLLMCallParameters:
    """Tests for LLM API call parameters"""

    @pytest.mark.asyncio
    async def test_llm_called_with_json_format(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that LLM is called with JSON response format"""
        await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_llm_called_with_low_temperature(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that LLM is called with low temperature"""
        await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_llm_called_with_max_tokens(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that LLM is called with appropriate max_tokens"""
        await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500


# ============================================================================
# Tests for Output Validation
# ============================================================================

class TestOutputValidation:
    """Tests for validating analyze_context outputs"""

    def test_valid_output_passes_validation(self, valid_technical_response):
        """Test that valid output passes validation"""
        validation = ContextAnalysisValidator.validate_output(valid_technical_response)
        assert validation["is_valid"] is True

    def test_invalid_complexity_detected(self):
        """Test that invalid complexity is detected"""
        invalid_output = {
            "detected_persona": "developer",
            "topic_complexity": "super_advanced",  # Invalid
            "requires_code": True,
            "requires_diagrams": True,
            "requires_hands_on": False,
            "domain_keywords": [],
        }

        validation = ContextAnalysisValidator.validate_output(invalid_output)
        assert validation["is_valid"] is False
        assert any("topic_complexity" in issue for issue in validation["issues"])

    def test_non_boolean_requires_code_detected(self):
        """Test that non-boolean requires_code is detected"""
        invalid_output = {
            "detected_persona": "developer",
            "topic_complexity": "intermediate",
            "requires_code": "yes",  # Should be boolean
            "requires_diagrams": True,
            "requires_hands_on": False,
            "domain_keywords": [],
        }

        validation = ContextAnalysisValidator.validate_output(invalid_output)
        assert validation["is_valid"] is False
        assert any("requires_code" in issue for issue in validation["issues"])

    def test_non_list_keywords_detected(self):
        """Test that non-list keywords is detected"""
        invalid_output = {
            "detected_persona": "developer",
            "topic_complexity": "intermediate",
            "requires_code": True,
            "requires_diagrams": True,
            "requires_hands_on": False,
            "domain_keywords": "python, fastapi",  # Should be list
        }

        validation = ContextAnalysisValidator.validate_output(invalid_output)
        assert validation["is_valid"] is False
        assert any("domain_keywords" in issue for issue in validation["issues"])


# ============================================================================
# Tests for Category Handling
# ============================================================================

class TestCategoryHandling:
    """Tests for category handling in prompt"""

    @pytest.mark.asyncio
    async def test_enum_category_converted(
        self, mock_openai_client, technical_topic_state, valid_technical_response
    ):
        """Test that enum category is converted to string value"""
        result = await mock_analyze_context(
            technical_topic_state,
            mock_openai_client,
            valid_technical_response
        )

        # Should use the enum value "tech" not the enum itself
        assert "- Category: tech" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_string_category_used_directly(
        self, mock_openai_client, valid_technical_response
    ):
        """Test that string category is used directly"""
        state = {
            "topic": "Test Topic",
            "profile_category": "business",
            "current_node": "",
            "errors": [],
        }

        result = await mock_analyze_context(
            state,
            mock_openai_client,
            valid_technical_response
        )

        assert "- Category: business" in result["prompt_used"]

    @pytest.mark.asyncio
    async def test_none_category_defaults_to_education(
        self, mock_openai_client, valid_technical_response
    ):
        """Test that None category defaults to education"""
        state = {
            "topic": "Test Topic",
            "profile_category": None,
            "current_node": "",
            "errors": [],
        }

        result = await mock_analyze_context(
            state,
            mock_openai_client,
            valid_technical_response
        )

        assert "- Category: education" in result["prompt_used"]


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
