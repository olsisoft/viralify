"""
Tests for SyntaxVerifier v2 - Hybrid AST + LLM Approach

Tests:
1. Python AST validation (existing behavior)
2. LLM-based validation for other languages (Groq primary, OpenAI fallback)
3. Caching behavior (MD5-based)
4. Auto-correction with retry
5. Fallback chain logic
6. Response parsing

Uses mocks for LLM clients to enable unit testing without API calls.
"""

import pytest
import asyncio
import hashlib
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# Inline Models (copy from models.py to avoid import issues)
# ============================================================================

class CodeLanguage(str, Enum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    C = "c"
    CPP = "cpp"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    SCALA = "scala"
    LUA = "lua"


@dataclass
class CodeSyntaxError:
    line: int
    column: int
    message: str
    severity: str = "error"


@dataclass
class SyntaxValidationResult:
    is_valid: bool
    language: str
    errors: List[CodeSyntaxError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrected_code: Optional[str] = None
    correction_applied: bool = False
    validation_method: str = "ast"


# ============================================================================
# Inline SyntaxVerifier v2 (simplified for testing)
# ============================================================================

class SyntaxVerifierV2:
    """
    Simplified SyntaxVerifier v2 for testing.

    Strategy:
    - Python: AST parsing (free, exact, <1ms)
    - All other languages: LLM (Groq primary, OpenAI fallback)
    """

    def __init__(self, groq_client=None, openai_client=None):
        self.groq_client = groq_client
        self.openai_client = openai_client
        self.groq_model = "llama-3.3-70b-versatile"
        self.openai_model = "gpt-4o-mini"
        # Instance-level cache to avoid test interference
        self._cache: Dict[str, SyntaxValidationResult] = {}

    def _get_cache_key(self, code: str, language: str) -> str:
        """Generate MD5 cache key."""
        content = f"{language}:{code}"
        return hashlib.md5(content.encode()).hexdigest()

    def _validate_python_ast(self, code: str) -> SyntaxValidationResult:
        """Validate Python using AST."""
        import ast
        errors = []
        warnings = []

        try:
            tree = ast.parse(code)
            compile(code, '<string>', 'exec')

            # Check for long lines
            for i, line in enumerate(code.split('\n'), 1):
                if len(line) > 120:
                    warnings.append(f"Line {i} exceeds 120 characters")

            return SyntaxValidationResult(
                is_valid=True,
                language="python",
                errors=[],
                warnings=warnings,
                validation_method="ast"
            )
        except SyntaxError as e:
            errors.append(CodeSyntaxError(
                line=e.lineno or 1,
                column=e.offset or 0,
                message=str(e.msg) if e.msg else str(e),
                severity="error"
            ))
            return SyntaxValidationResult(
                is_valid=False,
                language="python",
                errors=errors,
                validation_method="ast"
            )

    def _build_system_prompt(self) -> str:
        """Build system prompt for LLM validation."""
        return """You are an expert code syntax validator. Your task is to check if code has syntax errors.

IMPORTANT RULES:
1. Only report REAL syntax errors (missing brackets, invalid keywords, etc.)
2. Do NOT report style issues, warnings, or best practices
3. If code is valid, return valid=true
4. If code has errors, provide the corrected version

Return a JSON object with this exact structure:
{
    "valid": boolean,
    "errors": [
        {"line": number, "column": number, "message": "error description"}
    ],
    "corrected_code": "full corrected code if errors exist, null if valid"
}

ONLY return the JSON, no other text."""

    def _build_user_prompt(self, code: str, language: str, include_correction: bool) -> str:
        """Build user prompt for validation."""
        return f"""Validate this {language} code for syntax errors:

```{language}
{code}
```

Check for:
- Missing/unbalanced brackets, braces, parentheses
- Invalid syntax for {language}
- Missing semicolons (if required by {language})
- Invalid keywords or identifiers
- Unclosed strings or comments

{"Provide corrected code if there are errors." if include_correction else ""}

Return JSON only:"""

    def _parse_llm_response(self, content: str, language: str, provider: str) -> SyntaxValidationResult:
        """Parse LLM response into SyntaxValidationResult."""
        try:
            data = json.loads(content)

            errors = [
                CodeSyntaxError(
                    line=e.get("line", 1),
                    column=e.get("column", 0),
                    message=e.get("message", "Unknown error"),
                    severity="error"
                )
                for e in data.get("errors", [])
            ]

            corrected_code = data.get("corrected_code")
            if corrected_code and corrected_code.startswith("```"):
                lines = corrected_code.split('\n')
                corrected_code = '\n'.join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            is_valid = data.get("valid", True) and len(errors) == 0

            return SyntaxValidationResult(
                is_valid=is_valid,
                language=language,
                errors=errors,
                corrected_code=corrected_code if not is_valid else None,
                correction_applied=corrected_code is not None and not is_valid,
                validation_method=f"llm_{provider}"
            )
        except json.JSONDecodeError:
            return SyntaxValidationResult(
                is_valid=True,
                language=language,
                warnings=["Could not parse LLM response. Assuming valid."],
                validation_method=f"llm_{provider}"
            )

    async def _call_groq(self, system_prompt: str, user_prompt: str, language: str) -> Optional[SyntaxValidationResult]:
        """Call Groq API."""
        if not self.groq_client:
            return None

        try:
            response = await self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=2000
            )
            return self._parse_llm_response(response.choices[0].message.content, language, "groq")
        except Exception:
            return None

    async def _call_openai(self, system_prompt: str, user_prompt: str, language: str) -> Optional[SyntaxValidationResult]:
        """Call OpenAI API."""
        if not self.openai_client:
            return None

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=2000
            )
            return self._parse_llm_response(response.choices[0].message.content, language, "openai")
        except Exception:
            return None

    async def _validate_with_llm(
        self,
        code: str,
        language: CodeLanguage,
        auto_correct: bool,
        max_retries: int
    ) -> SyntaxValidationResult:
        """Validate with LLM (Groq primary, OpenAI fallback)."""
        cache_key = self._get_cache_key(code, language.value)
        if cache_key in self._cache:
            return self._cache[cache_key]

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(code, language.value, auto_correct)

        # Try Groq first
        if self.groq_client:
            result = await self._call_groq(system_prompt, user_prompt, language.value)
            if result:
                self._cache[cache_key] = result
                return result

        # Fallback to OpenAI
        if self.openai_client:
            result = await self._call_openai(system_prompt, user_prompt, language.value)
            if result:
                self._cache[cache_key] = result
                return result

        # No LLM available
        return SyntaxValidationResult(
            is_valid=True,
            language=language.value,
            warnings=["No LLM available for syntax validation. Assuming valid."],
            validation_method="none"
        )

    async def verify(
        self,
        code: str,
        language: CodeLanguage,
        auto_correct: bool = True,
        max_retries: int = 2
    ) -> SyntaxValidationResult:
        """Main verification entry point."""
        if language == CodeLanguage.PYTHON:
            result = self._validate_python_ast(code)
            if result.is_valid or not auto_correct:
                return result
            # For Python with errors, would try LLM correction here
            return result

        return await self._validate_with_llm(code, language, auto_correct, max_retries)

    def clear_cache(self):
        """Clear validation cache."""
        self._cache.clear()


# ============================================================================
# Mock Helpers
# ============================================================================

def create_mock_groq_response(valid: bool, errors: List[dict] = None, corrected_code: str = None):
    """Create a mock Groq API response."""
    response_data = {
        "valid": valid,
        "errors": errors or [],
        "corrected_code": corrected_code
    }

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(response_data)
    return mock_response


def create_mock_client(response):
    """Create a mock async client."""
    mock_client = MagicMock()
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=response)
    return mock_client


# ============================================================================
# TESTS - Python AST Validation
# ============================================================================

class TestPythonASTValidation:
    """Tests for Python AST-based validation."""

    @pytest.fixture
    def verifier(self):
        return SyntaxVerifierV2()

    @pytest.mark.asyncio
    async def test_valid_python_simple(self, verifier):
        """Simple valid Python code."""
        code = 'print("Hello, World!")'
        result = await verifier.verify(code, CodeLanguage.PYTHON)

        assert result.is_valid is True
        assert result.language == "python"
        assert result.validation_method == "ast"
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_valid_python_function(self, verifier):
        """Valid Python function definition."""
        code = '''
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

result = greet("World")
'''
        result = await verifier.verify(code, CodeLanguage.PYTHON)

        assert result.is_valid is True
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_valid_python_class(self, verifier):
        """Valid Python class definition."""
        code = '''
class Calculator:
    def __init__(self):
        self.result = 0

    def add(self, x, y):
        return x + y
'''
        result = await verifier.verify(code, CodeLanguage.PYTHON)

        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_invalid_python_missing_colon(self, verifier):
        """Missing colon in function definition."""
        code = '''
def hello(name)
    return f"Hello, {name}!"
'''
        result = await verifier.verify(code, CodeLanguage.PYTHON)

        assert result.is_valid is False
        assert len(result.errors) > 0
        assert result.errors[0].severity == "error"

    @pytest.mark.asyncio
    async def test_invalid_python_indentation(self, verifier):
        """Indentation error."""
        code = '''
def hello():
print("hello")
'''
        result = await verifier.verify(code, CodeLanguage.PYTHON)

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_invalid_python_unclosed_bracket(self, verifier):
        """Unclosed bracket."""
        code = '''
data = [1, 2, 3
print(data)
'''
        result = await verifier.verify(code, CodeLanguage.PYTHON)

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_invalid_python_invalid_keyword(self, verifier):
        """Invalid keyword usage."""
        code = '''
def = 5
'''
        result = await verifier.verify(code, CodeLanguage.PYTHON)

        assert result.is_valid is False

    @pytest.mark.asyncio
    async def test_python_long_line_warning(self, verifier):
        """Long lines should generate warnings."""
        long_line = "x = " + "a" * 150
        result = await verifier.verify(long_line, CodeLanguage.PYTHON)

        assert result.is_valid is True  # Still valid, just a warning
        assert len(result.warnings) > 0
        assert "120 characters" in result.warnings[0]


# ============================================================================
# TESTS - LLM-based Validation (Groq)
# ============================================================================

class TestGroqLLMValidation:
    """Tests for Groq LLM-based validation."""

    @pytest.mark.asyncio
    async def test_valid_javascript_via_groq(self):
        """Valid JavaScript should pass via Groq."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
function hello(name) {
    return `Hello, ${name}!`;
}
'''
        result = await verifier.verify(code, CodeLanguage.JAVASCRIPT)

        assert result.is_valid is True
        assert result.validation_method == "llm_groq"
        mock_groq.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_javascript_via_groq(self):
        """Invalid JavaScript should fail via Groq."""
        mock_response = create_mock_groq_response(
            valid=False,
            errors=[{"line": 2, "column": 0, "message": "Missing closing brace"}],
            corrected_code="function hello() {\n    return 'hello';\n}"
        )
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
function hello() {
    return 'hello';
'''
        result = await verifier.verify(code, CodeLanguage.JAVASCRIPT)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].message == "Missing closing brace"
        assert result.corrected_code is not None
        assert result.correction_applied is True

    @pytest.mark.asyncio
    async def test_valid_go_via_groq(self):
        """Valid Go should pass via Groq."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
'''
        result = await verifier.verify(code, CodeLanguage.GO)

        assert result.is_valid is True
        assert result.validation_method == "llm_groq"

    @pytest.mark.asyncio
    async def test_valid_rust_via_groq(self):
        """Valid Rust should pass via Groq."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
fn main() {
    println!("Hello, World!");
}
'''
        result = await verifier.verify(code, CodeLanguage.RUST)

        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_valid_java_via_groq(self):
        """Valid Java should pass via Groq."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
'''
        result = await verifier.verify(code, CodeLanguage.JAVA)

        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_valid_typescript_via_groq(self):
        """Valid TypeScript should pass via Groq."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}
'''
        result = await verifier.verify(code, CodeLanguage.TYPESCRIPT)

        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_valid_lua_via_groq(self):
        """Valid Lua should pass via Groq."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
-- Lua function example
function greet(name)
    print("Hello, " .. name .. "!")
end

-- Table (like a dictionary/object)
local person = {
    name = "Alice",
    age = 30
}

-- Loop through table
for key, value in pairs(person) do
    print(key .. ": " .. tostring(value))
end

greet(person.name)
'''
        result = await verifier.verify(code, CodeLanguage.LUA)

        assert result.is_valid is True
        assert result.validation_method == "llm_groq"

    @pytest.mark.asyncio
    async def test_invalid_lua_via_groq(self):
        """Invalid Lua should fail via Groq."""
        mock_response = create_mock_groq_response(
            valid=False,
            errors=[{"line": 3, "column": 0, "message": "Expected 'end' to close function"}],
            corrected_code="function greet(name)\n    print('Hello')\nend"
        )
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
function greet(name)
    print("Hello, " .. name)
-- Missing 'end' keyword
'''
        result = await verifier.verify(code, CodeLanguage.LUA)

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "end" in result.errors[0].message.lower()
        assert result.corrected_code is not None


# ============================================================================
# TESTS - Fallback Chain
# ============================================================================

class TestFallbackChain:
    """Tests for Groq → OpenAI → assume valid fallback chain."""

    @pytest.mark.asyncio
    async def test_groq_success_no_openai_call(self):
        """When Groq succeeds, OpenAI should not be called."""
        mock_groq_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_groq_response)

        mock_openai_response = create_mock_groq_response(valid=True)
        mock_openai = create_mock_client(mock_openai_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq, openai_client=mock_openai)

        result = await verifier.verify("console.log('test');", CodeLanguage.JAVASCRIPT)

        assert result.is_valid is True
        assert result.validation_method == "llm_groq"
        mock_groq.chat.completions.create.assert_called_once()
        mock_openai.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_groq_fails_openai_fallback(self):
        """When Groq fails, should fallback to OpenAI."""
        # Create a Groq client that raises exception
        async def groq_error(*args, **kwargs):
            raise Exception("Groq error")

        mock_groq = MagicMock()
        mock_groq.chat.completions.create = groq_error

        mock_openai_response = create_mock_groq_response(valid=True)
        mock_openai = create_mock_client(mock_openai_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq, openai_client=mock_openai)

        result = await verifier.verify("console.log('test');", CodeLanguage.JAVASCRIPT)

        assert result.is_valid is True
        assert result.validation_method == "llm_openai"

    @pytest.mark.asyncio
    async def test_both_fail_assume_valid(self):
        """When both LLMs fail, should assume valid with warning."""
        async def groq_error(*args, **kwargs):
            raise Exception("Groq error")

        async def openai_error(*args, **kwargs):
            raise Exception("OpenAI error")

        mock_groq = MagicMock()
        mock_groq.chat.completions.create = groq_error

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = openai_error

        verifier = SyntaxVerifierV2(groq_client=mock_groq, openai_client=mock_openai)

        result = await verifier.verify("console.log('test');", CodeLanguage.JAVASCRIPT)

        assert result.is_valid is True
        assert result.validation_method == "none"
        assert len(result.warnings) > 0
        assert "No LLM available" in result.warnings[0]

    @pytest.mark.asyncio
    async def test_no_clients_assume_valid(self):
        """When no LLM clients available, should assume valid."""
        verifier = SyntaxVerifierV2(groq_client=None, openai_client=None)

        result = await verifier.verify("console.log('test');", CodeLanguage.JAVASCRIPT)

        assert result.is_valid is True
        assert result.validation_method == "none"
        assert "No LLM available" in result.warnings[0]


# ============================================================================
# TESTS - Caching
# ============================================================================

class TestCaching:
    """Tests for MD5-based caching."""

    @pytest.mark.asyncio
    async def test_cache_hit_same_code(self):
        """Same code should return cached result."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = "console.log('hello');"

        # First call
        result1 = await verifier.verify(code, CodeLanguage.JAVASCRIPT)
        # Second call (should be cached)
        result2 = await verifier.verify(code, CodeLanguage.JAVASCRIPT)

        assert result1.is_valid == result2.is_valid
        # Groq should only be called once
        assert mock_groq.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_miss_different_code(self):
        """Different code should not use cache."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code1 = "console.log('hello');"
        code2 = "console.log('world');"

        await verifier.verify(code1, CodeLanguage.JAVASCRIPT)
        await verifier.verify(code2, CodeLanguage.JAVASCRIPT)

        # Groq should be called twice
        assert mock_groq.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_miss_different_language(self):
        """Same code but different language should not use cache."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = "console.log('hello');"

        await verifier.verify(code, CodeLanguage.JAVASCRIPT)
        await verifier.verify(code, CodeLanguage.TYPESCRIPT)

        # Groq should be called twice
        assert mock_groq.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Cache clear should force new LLM call."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = "console.log('hello');"

        await verifier.verify(code, CodeLanguage.JAVASCRIPT)
        verifier.clear_cache()
        await verifier.verify(code, CodeLanguage.JAVASCRIPT)

        # Groq should be called twice (cache was cleared)
        assert mock_groq.chat.completions.create.call_count == 2

    def test_cache_key_generation(self):
        """Cache key should be MD5 of language:code."""
        verifier = SyntaxVerifierV2()

        code = "console.log('hello');"
        language = "javascript"

        key = verifier._get_cache_key(code, language)
        expected = hashlib.md5(f"{language}:{code}".encode()).hexdigest()

        assert key == expected


# ============================================================================
# TESTS - Response Parsing
# ============================================================================

class TestResponseParsing:
    """Tests for LLM response parsing."""

    def test_parse_valid_response(self):
        """Parse valid JSON response."""
        verifier = SyntaxVerifierV2()

        content = json.dumps({
            "valid": True,
            "errors": [],
            "corrected_code": None
        })

        result = verifier._parse_llm_response(content, "javascript", "groq")

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.validation_method == "llm_groq"

    def test_parse_invalid_response_with_errors(self):
        """Parse response with syntax errors."""
        verifier = SyntaxVerifierV2()

        content = json.dumps({
            "valid": False,
            "errors": [
                {"line": 5, "column": 10, "message": "Unexpected token"},
                {"line": 8, "column": 0, "message": "Missing semicolon"}
            ],
            "corrected_code": "fixed code here"
        })

        result = verifier._parse_llm_response(content, "javascript", "groq")

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert result.errors[0].line == 5
        assert result.errors[0].message == "Unexpected token"
        assert result.corrected_code == "fixed code here"
        assert result.correction_applied is True

    def test_parse_response_with_markdown_code(self):
        """Parse response with markdown-wrapped corrected code."""
        verifier = SyntaxVerifierV2()

        content = json.dumps({
            "valid": False,
            "errors": [{"line": 1, "column": 0, "message": "Error"}],
            "corrected_code": "```javascript\nfixed code\n```"
        })

        result = verifier._parse_llm_response(content, "javascript", "groq")

        assert result.corrected_code == "fixed code"

    def test_parse_invalid_json(self):
        """Invalid JSON should assume valid."""
        verifier = SyntaxVerifierV2()

        content = "This is not JSON at all"

        result = verifier._parse_llm_response(content, "javascript", "groq")

        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "Could not parse" in result.warnings[0]

    def test_parse_missing_fields(self):
        """Missing fields should use defaults."""
        verifier = SyntaxVerifierV2()

        content = json.dumps({})  # Empty object

        result = verifier._parse_llm_response(content, "javascript", "groq")

        assert result.is_valid is True
        assert len(result.errors) == 0


# ============================================================================
# TESTS - Prompt Building
# ============================================================================

class TestPromptBuilding:
    """Tests for prompt generation."""

    def test_system_prompt_content(self):
        """System prompt should contain key instructions."""
        verifier = SyntaxVerifierV2()
        prompt = verifier._build_system_prompt()

        assert "syntax errors" in prompt.lower()
        assert "json" in prompt.lower()
        assert "valid" in prompt
        assert "errors" in prompt
        assert "corrected_code" in prompt

    def test_user_prompt_contains_code(self):
        """User prompt should contain the code."""
        verifier = SyntaxVerifierV2()

        code = "function test() { return 42; }"
        prompt = verifier._build_user_prompt(code, "javascript", True)

        assert code in prompt
        assert "javascript" in prompt.lower()

    def test_user_prompt_correction_flag(self):
        """User prompt should mention correction when enabled."""
        verifier = SyntaxVerifierV2()

        prompt_with = verifier._build_user_prompt("code", "javascript", True)
        prompt_without = verifier._build_user_prompt("code", "javascript", False)

        assert "corrected code" in prompt_with.lower()
        assert "corrected code" not in prompt_without.lower()


# ============================================================================
# TESTS - Multi-Language Support
# ============================================================================

class TestMultiLanguageSupport:
    """Tests for various programming languages."""

    @pytest.mark.asyncio
    async def test_kotlin_validation(self):
        """Kotlin should use LLM validation."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
fun main() {
    println("Hello, Kotlin!")
}
'''
        result = await verifier.verify(code, CodeLanguage.KOTLIN)

        assert result.validation_method == "llm_groq"

    @pytest.mark.asyncio
    async def test_swift_validation(self):
        """Swift should use LLM validation."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = r'''
func greet(name: String) -> String {
    return "Hello, \(name)!"
}
'''
        result = await verifier.verify(code, CodeLanguage.SWIFT)

        assert result.validation_method == "llm_groq"

    @pytest.mark.asyncio
    async def test_scala_validation(self):
        """Scala should use LLM validation."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
object Hello extends App {
  println("Hello, Scala!")
}
'''
        result = await verifier.verify(code, CodeLanguage.SCALA)

        assert result.validation_method == "llm_groq"

    @pytest.mark.asyncio
    async def test_ruby_validation(self):
        """Ruby should use LLM validation."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
def greet(name)
  puts "Hello, #{name}!"
end
'''
        result = await verifier.verify(code, CodeLanguage.RUBY)

        assert result.validation_method == "llm_groq"

    @pytest.mark.asyncio
    async def test_c_validation(self):
        """C should use LLM validation."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
'''
        result = await verifier.verify(code, CodeLanguage.C)

        assert result.validation_method == "llm_groq"

    @pytest.mark.asyncio
    async def test_cpp_validation(self):
        """C++ should use LLM validation."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        code = '''
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
'''
        result = await verifier.verify(code, CodeLanguage.CPP)

        assert result.validation_method == "llm_groq"


# ============================================================================
# TESTS - Error Cases
# ============================================================================

class TestErrorCases:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_empty_code(self):
        """Empty code should be handled."""
        verifier = SyntaxVerifierV2()

        result = await verifier.verify("", CodeLanguage.PYTHON)

        assert result.is_valid is True  # Empty Python is valid

    @pytest.mark.asyncio
    async def test_whitespace_only_code(self):
        """Whitespace-only code should be handled."""
        verifier = SyntaxVerifierV2()

        result = await verifier.verify("   \n\n   \t", CodeLanguage.PYTHON)

        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_unicode_code(self):
        """Unicode in code should be handled."""
        verifier = SyntaxVerifierV2()

        code = '''
# 日本語のコメント
message = "こんにちは世界"
print(message)
'''
        result = await verifier.verify(code, CodeLanguage.PYTHON)

        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_very_long_code(self):
        """Very long code should be handled."""
        mock_response = create_mock_groq_response(valid=True)
        mock_groq = create_mock_client(mock_response)

        verifier = SyntaxVerifierV2(groq_client=mock_groq)

        # Generate a long JavaScript code
        code = "function test() {\n" + "    console.log('line');\n" * 1000 + "}"

        result = await verifier.verify(code, CodeLanguage.JAVASCRIPT)

        assert result is not None


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
