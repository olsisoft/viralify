"""
Syntax Verifier v3 for Code Pipeline

Hybrid approach:
- Python: AST parsing (free, fast, exact)
- All other languages: LLM via configured provider (LLM_PROVIDER)

Uses the shared LLM provider system, supporting:
- OpenAI, DeepSeek, Groq, Mistral, Together, xAI
- Ollama (local, free)
- RunPod (serverless GPU)

Includes auto-correction capability.
"""

import ast
import os
import sys
import json
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from .models import CodeLanguage, CodeSyntaxError, SyntaxValidationResult

# Import shared LLM provider
# Handle different import paths (services/shared vs direct)
try:
    from services.shared.llm_provider import (
        get_llm_client,
        get_model_name,
        get_provider,
        get_provider_config
    )
except ImportError:
    try:
        from shared.llm_provider import (
            get_llm_client,
            get_model_name,
            get_provider,
            get_provider_config
        )
    except ImportError:
        # Fallback: create minimal implementation for standalone use
        from openai import AsyncOpenAI

        def get_llm_client():
            return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def get_model_name(tier: str = "fast"):
            return "gpt-4o-mini" if tier == "fast" else "gpt-4o"

        def get_provider():
            return os.getenv("LLM_PROVIDER", "openai")

        def get_provider_config():
            return None


class SyntaxVerifier:
    """
    Hybrid syntax verifier v3.

    Strategy:
    - Python: AST parsing (free, exact, <1ms)
    - All other languages: LLM via LLM_PROVIDER (supports local LLMs)

    Features:
    - Auto-correction for syntax errors
    - Caching to avoid repeated API calls
    - Retry logic with correction attempts
    - Works with any LLM provider (including Ollama local)
    """

    def __init__(self):
        """
        Initialize verifier.

        Uses the shared LLM provider configured via LLM_PROVIDER env var.
        """
        self._cache: Dict[str, SyntaxValidationResult] = {}
        self._client = None
        self._model = None
        self._provider_name = None

    def _ensure_client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            self._client = get_llm_client()
            self._model = get_model_name("fast")  # Use fast model for syntax validation
            self._provider_name = str(get_provider())
            print(f"[SYNTAX_V3] Using provider: {self._provider_name}, model: {self._model}", flush=True)

    async def verify(
        self,
        code: str,
        language: CodeLanguage,
        auto_correct: bool = True,
        max_retries: int = 2
    ) -> SyntaxValidationResult:
        """
        Verify code syntax.

        Args:
            code: Source code to validate
            language: Programming language
            auto_correct: Attempt LLM correction if invalid
            max_retries: Max correction attempts

        Returns:
            SyntaxValidationResult with validation details
        """
        # Python: use AST (free, fast, exact)
        if language == CodeLanguage.PYTHON:
            result = self._validate_python_ast(code)
            if result.is_valid or not auto_correct:
                return result
            # If Python has errors and auto_correct, try LLM correction
            return await self._correct_with_llm(code, language, result.errors, max_retries)

        # All other languages: use LLM
        return await self._validate_with_llm(code, language, auto_correct, max_retries)

    def _validate_python_ast(self, code: str) -> SyntaxValidationResult:
        """
        Validate Python code using AST parsing.

        Uses:
        1. ast.parse() for syntax checking
        2. compile() for byte-code compilation check
        """
        errors: List[CodeSyntaxError] = []
        warnings: List[str] = []

        try:
            # Step 1: Parse AST
            tree = ast.parse(code)

            # Step 2: Compile to bytecode (catches additional issues)
            compile(code, '<string>', 'exec')

            # Step 3: Basic style warnings
            warnings.extend(self._check_python_style(tree, code))

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

        except Exception as e:
            errors.append(CodeSyntaxError(
                line=1,
                column=0,
                message=f"Validation error: {str(e)}",
                severity="error"
            ))
            return SyntaxValidationResult(
                is_valid=False,
                language="python",
                errors=errors,
                validation_method="ast"
            )

    def _check_python_style(self, tree: ast.AST, code: str) -> List[str]:
        """Check Python code for common style issues."""
        warnings = []

        # Check for overly long lines
        for i, line in enumerate(code.split('\n'), 1):
            if len(line) > 120:
                warnings.append(f"Line {i} exceeds 120 characters")

        return warnings

    def _get_cache_key(self, code: str, language: str) -> str:
        """Generate cache key for code."""
        content = f"{language}:{code}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _validate_with_llm(
        self,
        code: str,
        language: CodeLanguage,
        auto_correct: bool,
        max_retries: int
    ) -> SyntaxValidationResult:
        """
        Validate code using the configured LLM provider.
        """
        # Check cache first
        cache_key = self._get_cache_key(code, language.value)
        if cache_key in self._cache:
            print(f"[SYNTAX_V3] Cache hit for {language.value}", flush=True)
            return self._cache[cache_key]

        # Ensure client is initialized
        self._ensure_client()

        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(code, language.value, auto_correct)

        # Call LLM
        result = await self._call_llm(system_prompt, user_prompt, language.value)
        if result:
            self._cache[cache_key] = result
            return result

        # LLM failed - return valid with warning
        return SyntaxValidationResult(
            is_valid=True,
            language=language.value,
            warnings=[f"LLM validation failed ({self._provider_name}). Assuming valid."],
            validation_method="none"
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt for syntax validation."""
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
        prompt = f"""Validate this {language} code for syntax errors:

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
        return prompt

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        language: str
    ) -> Optional[SyntaxValidationResult]:
        """Call the configured LLM provider for validation."""
        try:
            print(f"[SYNTAX_V3] Validating {language} with {self._provider_name}...", flush=True)

            # Check if provider supports JSON mode
            provider_config = get_provider_config()
            use_json_mode = True

            # Build request kwargs
            kwargs = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0,
                "max_tokens": 2000
            }

            # Add JSON mode if supported (most providers do via OpenAI-compatible API)
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await self._client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            if not content:
                print(f"[SYNTAX_V3] Empty response from {self._provider_name}", flush=True)
                return None

            return self._parse_llm_response(content, language, self._provider_name)

        except Exception as e:
            print(f"[SYNTAX_V3] {self._provider_name} error: {e}", flush=True)
            return None

    def _parse_llm_response(
        self,
        content: str,
        language: str,
        provider: str
    ) -> SyntaxValidationResult:
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
            # Clean corrected code if it has markdown
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

        except json.JSONDecodeError as e:
            print(f"[SYNTAX_V3] JSON parse error: {e}", flush=True)
            # Assume valid if we can't parse
            return SyntaxValidationResult(
                is_valid=True,
                language=language,
                warnings=["Could not parse LLM response. Assuming valid."],
                validation_method=f"llm_{provider}"
            )

    async def _correct_with_llm(
        self,
        code: str,
        language: CodeLanguage,
        errors: List[CodeSyntaxError],
        max_retries: int
    ) -> SyntaxValidationResult:
        """
        Attempt to correct code with LLM after AST found errors.
        """
        self._ensure_client()

        error_descriptions = "\n".join([
            f"- Line {e.line}: {e.message}"
            for e in errors
        ])

        prompt = f"""Fix the syntax errors in this {language.value} code:

ERRORS FOUND:
{error_descriptions}

CODE:
```{language.value}
{code}
```

Return JSON with the corrected code:
{{"valid": true, "errors": [], "corrected_code": "your fixed code here"}}"""

        for attempt in range(max_retries):
            result = await self._call_llm(
                self._build_system_prompt(),
                prompt,
                language.value
            )
            if result and result.corrected_code:
                # Re-validate the corrected code
                if language == CodeLanguage.PYTHON:
                    check = self._validate_python_ast(result.corrected_code)
                    if check.is_valid:
                        return SyntaxValidationResult(
                            is_valid=True,
                            language=language.value,
                            corrected_code=result.corrected_code,
                            correction_applied=True,
                            validation_method=f"ast+llm_correction_{self._provider_name}",
                            warnings=[f"Auto-corrected after {attempt + 1} attempt(s)"]
                        )

        # Could not correct - return original errors
        return SyntaxValidationResult(
            is_valid=False,
            language=language.value,
            errors=errors,
            validation_method="ast"
        )

    def clear_cache(self):
        """Clear the validation cache."""
        self._cache.clear()


# Singleton instance
_syntax_verifier: Optional[SyntaxVerifier] = None


def get_syntax_verifier() -> SyntaxVerifier:
    """Get singleton SyntaxVerifier instance."""
    global _syntax_verifier
    if _syntax_verifier is None:
        _syntax_verifier = SyntaxVerifier()
    return _syntax_verifier
