"""
Tests for RobustJSONParser

Tests all parsing strategies and edge cases.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


# Direct import to avoid dependency chain
def import_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


parser_path = Path(__file__).parent.parent / "services" / "json_parser.py"
parser_module = import_module_from_file("json_parser", str(parser_path))

RobustJSONParser = parser_module.RobustJSONParser
JSONParseError = parser_module.JSONParseError
LanguageValidationResponse = parser_module.LanguageValidationResponse
parse_json_response = parser_module.parse_json_response


# =============================================================================
# Test Direct Parse (Strategy 1)
# =============================================================================

class TestDirectParse:
    """Test direct JSON parsing."""

    @pytest.fixture
    def parser(self):
        return RobustJSONParser()

    def test_simple_object(self, parser):
        result = parser.parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_nested_object(self, parser):
        result = parser.parse('{"a": {"b": {"c": 1}}}')
        assert result == {"a": {"b": {"c": 1}}}

    def test_array(self, parser):
        result = parser.parse('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_with_whitespace(self, parser):
        result = parser.parse('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_boolean_values(self, parser):
        result = parser.parse('{"valid": true, "invalid": false}')
        assert result == {"valid": True, "invalid": False}

    def test_null_value(self, parser):
        result = parser.parse('{"value": null}')
        assert result == {"value": None}


# =============================================================================
# Test Markdown Extraction (Strategy 2)
# =============================================================================

class TestMarkdownExtraction:
    """Test JSON extraction from markdown code blocks."""

    @pytest.fixture
    def parser(self):
        return RobustJSONParser()

    def test_json_code_block(self, parser):
        content = '```json\n{"is_valid": true}\n```'
        result = parser.parse(content)
        assert result == {"is_valid": True}

    def test_generic_code_block(self, parser):
        content = '```\n{"is_valid": false}\n```'
        result = parser.parse(content)
        assert result == {"is_valid": False}

    def test_code_block_with_text_before(self, parser):
        content = 'Here is the JSON:\n```json\n{"result": "ok"}\n```'
        result = parser.parse(content)
        assert result == {"result": "ok"}

    def test_code_block_with_text_after(self, parser):
        content = '```json\n{"data": 123}\n```\nThat was the result.'
        result = parser.parse(content)
        assert result == {"data": 123}

    def test_inline_code(self, parser):
        content = 'The result is `{"status": "done"}`'
        result = parser.parse(content)
        assert result == {"status": "done"}


# =============================================================================
# Test Regex Extraction (Strategy 3)
# =============================================================================

class TestRegexExtraction:
    """Test JSON extraction with regex."""

    @pytest.fixture
    def parser(self):
        return RobustJSONParser()

    def test_json_with_prefix_text(self, parser):
        content = 'Sure! Here is the analysis:\n{"is_valid": true, "score": 95}'
        result = parser.parse(content)
        assert result == {"is_valid": True, "score": 95}

    def test_json_with_suffix_text(self, parser):
        content = '{"result": "success"}\n\nHope this helps!'
        result = parser.parse(content)
        assert result == {"result": "success"}

    def test_json_surrounded_by_text(self, parser):
        content = 'Analysis complete:\n{"data": [1,2,3]}\nEnd of report.'
        result = parser.parse(content)
        assert result == {"data": [1, 2, 3]}

    def test_array_extraction(self, parser):
        content = 'The items are: [{"id": 1}, {"id": 2}]'
        result = parser.parse(content)
        assert result == [{"id": 1}, {"id": 2}]


# =============================================================================
# Test JSON Repair (Strategy 4)
# =============================================================================

class TestJSONRepair:
    """Test JSON syntax error repair."""

    @pytest.fixture
    def parser(self):
        return RobustJSONParser()

    def test_trailing_comma_object(self, parser):
        content = '{"key": "value",}'
        result = parser.parse(content)
        assert result == {"key": "value"}

    def test_trailing_comma_array(self, parser):
        content = '[1, 2, 3,]'
        result = parser.parse(content)
        assert result == [1, 2, 3]

    def test_single_quotes(self, parser):
        content = "{'key': 'value'}"
        result = parser.parse(content)
        assert result == {"key": "value"}

    def test_python_none(self, parser):
        content = '{"value": None}'
        result = parser.parse(content)
        assert result == {"value": None}

    def test_python_true_false(self, parser):
        content = '{"a": True, "b": False}'
        result = parser.parse(content)
        assert result == {"a": True, "b": False}

    def test_unquoted_keys(self, parser):
        content = '{key: "value", another: 123}'
        result = parser.parse(content)
        assert result == {"key": "value", "another": 123}


# =============================================================================
# Test Combined Scenarios
# =============================================================================

class TestCombinedScenarios:
    """Test combinations of issues."""

    @pytest.fixture
    def parser(self):
        return RobustJSONParser()

    def test_markdown_with_trailing_comma(self, parser):
        content = '```json\n{"is_valid": true,}\n```'
        result = parser.parse(content)
        assert result == {"is_valid": True}

    def test_text_prefix_with_python_bools(self, parser):
        content = 'Result: {"success": True, "failed": False}'
        result = parser.parse(content)
        assert result == {"success": True, "failed": False}

    def test_real_world_llm_response(self, parser):
        content = '''I've analyzed the content. Here's my assessment:

```json
{
    "is_valid": true,
    "issues": [],
    "overall_language_quality": "excellent",
    "summary": "All content properly localized."
}
```

Let me know if you need anything else!'''
        result = parser.parse(content)
        assert result["is_valid"] is True
        assert result["issues"] == []
        assert result["overall_language_quality"] == "excellent"


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error cases."""

    @pytest.fixture
    def parser(self):
        return RobustJSONParser()

    def test_completely_invalid_raises_error(self, parser):
        content = 'This is not JSON at all, just plain text.'
        with pytest.raises(JSONParseError) as exc_info:
            parser.parse(content)
        assert "Failed to parse JSON" in str(exc_info.value)
        assert len(exc_info.value.attempts) > 0

    def test_empty_string_raises_error(self, parser):
        with pytest.raises(JSONParseError):
            parser.parse('')

    def test_partial_json_raises_error(self, parser):
        content = '{"key": "value"'  # Missing closing brace
        with pytest.raises(JSONParseError):
            parser.parse(content)


# =============================================================================
# Test Pydantic Validation
# =============================================================================

class TestPydanticValidation:
    """Test parsing with Pydantic model validation."""

    @pytest.fixture
    def parser(self):
        return RobustJSONParser()

    def test_valid_language_response(self, parser):
        content = '{"is_valid": true, "issues": [], "overall_language_quality": "excellent"}'
        result = parser.parse_and_validate(content, LanguageValidationResponse)
        assert isinstance(result, LanguageValidationResponse)
        assert result.is_valid is True
        assert result.issues == []

    def test_minimal_language_response(self, parser):
        content = '{"is_valid": false}'
        result = parser.parse_and_validate(content, LanguageValidationResponse)
        assert result.is_valid is False
        # Defaults should be applied
        assert result.issues == []
        assert result.overall_language_quality == "good"

    def test_invalid_model_raises_error(self, parser):
        content = '{"not_the_right_field": true}'
        with pytest.raises(Exception):  # ValidationError
            parser.parse_and_validate(content, LanguageValidationResponse)


# =============================================================================
# Test LLM Fallback (Async)
# =============================================================================

class TestLLMFallback:
    """Test LLM-based repair fallback."""

    @pytest.mark.asyncio
    async def test_llm_repair_called_on_failure(self):
        # Create mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"is_valid": true}'))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        parser = RobustJSONParser(mock_client)

        # Content that will fail local parsing
        content = 'This is completely invalid and needs LLM repair'

        result = await parser.parse_with_llm_fallback(content)
        assert result == {"is_valid": True}
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_llm_call_when_local_succeeds(self):
        mock_client = AsyncMock()
        parser = RobustJSONParser(mock_client)

        # Content that will parse locally
        content = '{"is_valid": true}'

        result = await parser.parse_with_llm_fallback(content)
        assert result == {"is_valid": True}
        mock_client.chat.completions.create.assert_not_called()


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_parse_json_response(self):
        result = parse_json_response('{"status": "ok"}')
        assert result == {"status": "ok"}

    def test_parse_json_response_with_markdown(self):
        result = parse_json_response('```json\n{"value": 42}\n```')
        assert result == {"value": 42}


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    @pytest.fixture
    def parser(self):
        return RobustJSONParser()

    def test_unicode_content(self, parser):
        content = '{"message": "Bonjour le monde! 你好世界 🌍"}'
        result = parser.parse(content)
        assert result["message"] == "Bonjour le monde! 你好世界 🌍"

    def test_nested_quotes(self, parser):
        content = '{"text": "He said \\"hello\\""}'
        result = parser.parse(content)
        assert result["text"] == 'He said "hello"'

    def test_large_numbers(self, parser):
        content = '{"big": 9999999999999999999}'
        result = parser.parse(content)
        assert result["big"] == 9999999999999999999

    def test_scientific_notation(self, parser):
        content = '{"value": 1.5e10}'
        result = parser.parse(content)
        assert result["value"] == 1.5e10

    def test_empty_object(self, parser):
        result = parser.parse('{}')
        assert result == {}

    def test_empty_array(self, parser):
        result = parser.parse('[]')
        assert result == []

    def test_deeply_nested(self, parser):
        content = '{"a": {"b": {"c": {"d": {"e": 5}}}}}'
        result = parser.parse(content)
        assert result["a"]["b"]["c"]["d"]["e"] == 5
