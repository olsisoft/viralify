"""
Robust JSON Parser for LLM Responses

Provides multiple fallback strategies to parse JSON from LLM responses
that may contain formatting issues, markdown wrappers, or syntax errors.

Strategies (in order):
1. Direct JSON parse
2. Extract from markdown code blocks
3. Regex-based JSON extraction
4. JSON repair (fix common syntax errors)
5. LLM-based repair (last resort)
"""

import json
import re
import os
from typing import Any, Dict, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError

# For LLM-based repair
from openai import AsyncOpenAI


T = TypeVar('T', bound=BaseModel)


class JSONParseError(Exception):
    """Raised when all parsing strategies fail."""
    def __init__(self, message: str, original_content: str, attempts: list):
        super().__init__(message)
        self.original_content = original_content
        self.attempts = attempts


class RobustJSONParser:
    """
    Robust JSON parser with multiple fallback strategies.

    Usage:
        parser = RobustJSONParser()

        # Simple parse
        result = parser.parse(llm_response)

        # With Pydantic validation
        result = parser.parse_and_validate(llm_response, MyModel)

        # Async with LLM repair fallback
        result = await parser.parse_with_llm_fallback(llm_response, MyModel)
    """

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        self.client = openai_client
        self._repair_model = os.getenv("JSON_REPAIR_MODEL", "gpt-4o-mini")

    def parse(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON with multiple fallback strategies.

        Args:
            content: Raw LLM response that should contain JSON

        Returns:
            Parsed JSON as dictionary

        Raises:
            JSONParseError: If all strategies fail
        """
        attempts = []

        # Strategy 1: Direct parse
        result = self._try_direct_parse(content)
        if result is not None:
            return result
        attempts.append("direct_parse: failed")

        # Strategy 2: Extract from markdown
        result = self._try_markdown_extraction(content)
        if result is not None:
            return result
        attempts.append("markdown_extraction: failed")

        # Strategy 3: Regex extraction
        result = self._try_regex_extraction(content)
        if result is not None:
            return result
        attempts.append("regex_extraction: failed")

        # Strategy 4: JSON repair
        result = self._try_json_repair(content)
        if result is not None:
            return result
        attempts.append("json_repair: failed")

        raise JSONParseError(
            f"Failed to parse JSON after {len(attempts)} attempts",
            original_content=content,
            attempts=attempts
        )

    def parse_and_validate(
        self,
        content: str,
        model: Type[T]
    ) -> T:
        """
        Parse JSON and validate against a Pydantic model.

        Args:
            content: Raw LLM response
            model: Pydantic model class for validation

        Returns:
            Validated Pydantic model instance
        """
        parsed = self.parse(content)
        return model.model_validate(parsed)

    async def parse_with_llm_fallback(
        self,
        content: str,
        model: Optional[Type[T]] = None,
        max_retries: int = 2
    ) -> Union[Dict[str, Any], T]:
        """
        Parse with LLM-based repair as final fallback.

        If all local strategies fail, uses an LLM to fix the JSON.

        Args:
            content: Raw LLM response
            model: Optional Pydantic model for validation
            max_retries: Max LLM repair attempts

        Returns:
            Parsed (and optionally validated) result
        """
        # Try local strategies first
        try:
            parsed = self.parse(content)
            if model:
                return model.model_validate(parsed)
            return parsed
        except JSONParseError as e:
            if not self.client:
                raise

            # LLM-based repair
            for attempt in range(max_retries):
                repaired = await self._llm_repair(content, model, attempt)
                if repaired:
                    try:
                        parsed = json.loads(repaired)
                        if model:
                            return model.model_validate(parsed)
                        return parsed
                    except (json.JSONDecodeError, ValidationError):
                        continue

            raise

    # =========================================================================
    # Strategy Implementations
    # =========================================================================

    def _try_direct_parse(self, content: str) -> Optional[Dict[str, Any]]:
        """Strategy 1: Direct JSON parse."""
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            return None

    def _try_markdown_extraction(self, content: str) -> Optional[Dict[str, Any]]:
        """Strategy 2: Extract JSON from markdown code blocks."""
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
            r'`([\s\S]*?)`',                 # ` ... ` (inline)
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue

        return None

    def _try_regex_extraction(self, content: str) -> Optional[Dict[str, Any]]:
        """Strategy 3: Find JSON object/array with regex."""
        # Find content between first { and last }
        obj_match = re.search(r'\{[\s\S]*\}', content)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass

        # Find content between first [ and last ]
        arr_match = re.search(r'\[[\s\S]*\]', content)
        if arr_match:
            try:
                return json.loads(arr_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def _try_json_repair(self, content: str) -> Optional[Dict[str, Any]]:
        """Strategy 4: Repair common JSON syntax errors."""
        # Extract potential JSON first
        json_str = content.strip()

        obj_match = re.search(r'\{[\s\S]*\}', content)
        if obj_match:
            json_str = obj_match.group()

        repairs = [
            # Fix trailing commas: {"a": 1,} -> {"a": 1}
            (r',\s*}', '}'),
            (r',\s*]', ']'),

            # Fix unquoted keys: {key: "value"} -> {"key": "value"}
            (r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3'),

            # Fix single quotes: {'key': 'value'} -> {"key": "value"}
            (r"'([^']*)'", r'"\1"'),

            # Fix Python None/True/False
            (r'\bNone\b', 'null'),
            (r'\bTrue\b', 'true'),
            (r'\bFalse\b', 'false'),

            # Fix missing quotes on string values (limited)
            # This is risky, only for simple cases
        ]

        repaired = json_str
        for pattern, replacement in repairs:
            repaired = re.sub(pattern, replacement, repaired)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None

    async def _llm_repair(
        self,
        content: str,
        model: Optional[Type[BaseModel]],
        attempt: int
    ) -> Optional[str]:
        """Strategy 5: Use LLM to repair malformed JSON."""
        if not self.client:
            return None

        schema_hint = ""
        if model:
            schema_hint = f"\n\nExpected schema:\n{model.model_json_schema()}"

        prompt = f"""The following text should be valid JSON but has syntax errors.
Fix the JSON and return ONLY the corrected JSON, nothing else.

Original content:
```
{content[:2000]}
```
{schema_hint}

Return ONLY valid JSON, no explanation."""

        try:
            response = await self.client.chat.completions.create(
                model=self._repair_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None


# =============================================================================
# Convenience Functions
# =============================================================================

_default_parser: Optional[RobustJSONParser] = None


def get_parser(client: Optional[AsyncOpenAI] = None) -> RobustJSONParser:
    """Get or create the default parser instance."""
    global _default_parser
    if _default_parser is None:
        _default_parser = RobustJSONParser(client)
    return _default_parser


def parse_json_response(content: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response with automatic repair.

    Convenience function for simple use cases.
    """
    return get_parser().parse(content)


def parse_and_validate(content: str, model: Type[T]) -> T:
    """
    Parse and validate JSON against a Pydantic model.

    Convenience function for validated parsing.
    """
    return get_parser().parse_and_validate(content, model)


async def parse_with_repair(
    content: str,
    model: Optional[Type[T]] = None,
    client: Optional[AsyncOpenAI] = None
) -> Union[Dict[str, Any], T]:
    """
    Parse with LLM repair fallback.

    Convenience function for robust parsing with LLM fallback.
    """
    parser = RobustJSONParser(client)
    return await parser.parse_with_llm_fallback(content, model)


# =============================================================================
# Pydantic Models for Common Responses
# =============================================================================

class LanguageValidationResponse(BaseModel):
    """Expected response from validate_language."""
    is_valid: bool
    issues: list = []
    overall_language_quality: str = "good"
    summary: Optional[str] = None


class QuizPlanningResponse(BaseModel):
    """Expected response from plan_quizzes."""
    quiz_placement: list
    total_quiz_count: int
    coverage_analysis: str = ""


class StructureValidationResponse(BaseModel):
    """Expected response from validate_structure."""
    is_valid: bool
    score: int = 70
    issues: list = []
    suggestions: list = []


class ContextAnalysisResponse(BaseModel):
    """Expected response from analyze_context."""
    detected_persona: str = "student"
    topic_complexity: str = "intermediate"
    requires_code: bool = False
    requires_diagrams: bool = True
    requires_hands_on: bool = False
    domain_keywords: list = []


class ProfileAdaptationResponse(BaseModel):
    """Expected response from adapt_for_profile."""
    content_preferences: dict = {}
    recommended_elements: list = []


class ElementSuggestionResponse(BaseModel):
    """Expected response from suggest_elements."""
    element_mapping: dict = {}


class OutlineRefinementResponse(BaseModel):
    """Expected response from refine_outline."""
    refined_sections: list = []
    refinements_made: list = []
    expected_score_improvement: int = 0
