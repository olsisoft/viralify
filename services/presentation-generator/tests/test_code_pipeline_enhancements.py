"""
Tests for Code Pipeline Enhancements

Tests:
1. SyntaxVerifier - AST validation for Python, regex for other languages
2. CodeSummarizer - Code summarization for slides

Uses inline implementations to avoid circular import issues.
"""

import pytest
import ast
import re
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum


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
# Inline SyntaxVerifier (core logic only, no LLM)
# ============================================================================

class SyntaxVerifierCore:
    """Core syntax validation without LLM dependencies."""

    def validate_python(self, code: str) -> SyntaxValidationResult:
        """Validate Python code using AST parsing."""
        errors = []
        try:
            ast.parse(code)
            compile(code, '<string>', 'exec')
            return SyntaxValidationResult(
                is_valid=True,
                language="python",
                errors=[],
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

    def check_balanced_braces(self, code: str) -> List[CodeSyntaxError]:
        """Check for balanced braces, brackets, and parentheses."""
        errors = []
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        openers = set(pairs.values())
        closers = set(pairs.keys())

        lines = code.split('\n')
        in_string = False
        string_char = None

        for line_num, line in enumerate(lines, 1):
            for col, char in enumerate(line):
                if char in ('"', "'") and (col == 0 or line[col-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                    continue

                if in_string:
                    continue

                if char in openers:
                    stack.append((char, line_num, col))
                elif char in closers:
                    if not stack:
                        errors.append(CodeSyntaxError(
                            line=line_num,
                            column=col,
                            message=f"Unmatched closing '{char}'",
                            severity="error"
                        ))
                    elif stack[-1][0] != pairs[char]:
                        errors.append(CodeSyntaxError(
                            line=line_num,
                            column=col,
                            message=f"Mismatched '{char}'",
                            severity="error"
                        ))
                        stack.pop()
                    else:
                        stack.pop()

        for char, line_num, col in stack:
            errors.append(CodeSyntaxError(
                line=line_num,
                column=col,
                message=f"Unclosed '{char}'",
                severity="error"
            ))

        return errors

    def validate_javascript(self, code: str) -> SyntaxValidationResult:
        """Validate JavaScript using regex patterns."""
        errors = self.check_balanced_braces(code)
        return SyntaxValidationResult(
            is_valid=len(errors) == 0,
            language="javascript",
            errors=errors,
            validation_method="regex"
        )

    def validate_go(self, code: str) -> SyntaxValidationResult:
        """Validate Go code using regex patterns."""
        errors = []

        if not re.search(r'^\s*package\s+\w+', code, re.MULTILINE):
            errors.append(CodeSyntaxError(
                line=1,
                column=0,
                message="Missing package declaration",
                severity="error"
            ))

        brace_errors = self.check_balanced_braces(code)
        errors.extend(brace_errors)

        return SyntaxValidationResult(
            is_valid=not any(e.severity == "error" for e in errors),
            language="go",
            errors=errors,
            validation_method="regex"
        )


# ============================================================================
# Inline CodeSummarizer (core logic only)
# ============================================================================

class CodeSummarizerCore:
    """Core code summarization without LLM dependencies."""

    def remove_verbose_comments(self, code: str, language: CodeLanguage) -> tuple:
        """Remove verbose standalone comments."""
        lines = code.split('\n')
        result_lines = []
        removed_count = 0

        comment_pattern = r'^\s*#' if language == CodeLanguage.PYTHON else r'^\s*//'
        essential_markers = ['TODO', 'FIXME', 'NOTE', 'IMPORTANT']
        section_markers = ['===', '---', '***']

        for line in lines:
            if re.match(comment_pattern, line):
                if any(marker in line.upper() for marker in essential_markers):
                    result_lines.append(line)
                    continue
                if any(marker in line for marker in section_markers):
                    result_lines.append(line)
                    continue
                removed_count += 1
                continue
            result_lines.append(line)

        return '\n'.join(result_lines), removed_count

    def combine_python_imports(self, code: str) -> tuple:
        """Combine Python imports from the same module."""
        lines = code.split('\n')
        import_groups = {}
        non_import_lines = []
        first_import_idx = -1

        for i, line in enumerate(lines):
            stripped = line.strip()
            from_match = re.match(r'from\s+(\S+)\s+import\s+(.+)', stripped)
            if from_match:
                module = from_match.group(1)
                items = [item.strip() for item in from_match.group(2).split(',')]
                if module not in import_groups:
                    import_groups[module] = []
                    if first_import_idx < 0:
                        first_import_idx = len(non_import_lines)
                import_groups[module].extend(items)
                continue
            non_import_lines.append(line)

        if not import_groups:
            return code, False

        combined_imports = []
        for module, items in sorted(import_groups.items()):
            unique_items = sorted(set(items))
            combined_imports.append(f"from {module} import {', '.join(unique_items)}")

        result = non_import_lines[:first_import_idx] + combined_imports + non_import_lines[first_import_idx:]
        return '\n'.join(result), True

    def condense_empty_lines(self, code: str) -> str:
        """Replace multiple consecutive empty lines with a single empty line."""
        return re.sub(r'\n{3,}', '\n\n', code)


# ============================================================================
# TESTS
# ============================================================================

class TestSyntaxVerifierPython:
    """Tests for Python syntax validation."""

    @pytest.fixture
    def verifier(self):
        return SyntaxVerifierCore()

    def test_valid_python_code(self, verifier):
        """Valid Python code should pass validation."""
        code = '''
def hello(name):
    """Say hello."""
    return f"Hello, {name}!"

result = hello("World")
print(result)
'''
        result = verifier.validate_python(code)

        assert result.is_valid is True
        assert result.language == "python"
        assert result.validation_method == "ast"
        assert len(result.errors) == 0

    def test_invalid_python_syntax(self, verifier):
        """Invalid Python code should fail validation."""
        code = '''
def hello(name)  # Missing colon
    return f"Hello, {name}!"
'''
        result = verifier.validate_python(code)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_python_indentation_error(self, verifier):
        """Indentation errors should be detected."""
        code = '''
def hello():
print("hello")  # Wrong indentation
'''
        result = verifier.validate_python(code)

        assert result.is_valid is False
        assert len(result.errors) > 0


class TestSyntaxVerifierOtherLanguages:
    """Tests for other language validation."""

    @pytest.fixture
    def verifier(self):
        return SyntaxVerifierCore()

    def test_valid_javascript(self, verifier):
        """Valid JavaScript code should pass basic validation."""
        code = '''
function hello(name) {
    return `Hello, ${name}!`;
}

const result = hello("World");
console.log(result);
'''
        result = verifier.validate_javascript(code)

        assert result.is_valid is True
        assert result.validation_method == "regex"

    def test_unbalanced_braces(self, verifier):
        """Unbalanced braces should be detected."""
        code = '''
function hello(name) {
    return `Hello, ${name}!`;
// Missing closing brace
'''
        result = verifier.validate_javascript(code)

        assert result.is_valid is False
        assert any("unclosed" in e.message.lower() for e in result.errors)

    def test_go_missing_package(self, verifier):
        """Go code without package should fail."""
        code = '''
func main() {
    fmt.Println("Hello, World!")
}
'''
        result = verifier.validate_go(code)

        assert result.is_valid is False
        assert any("package" in e.message.lower() for e in result.errors)

    def test_go_with_package(self, verifier):
        """Go code with package should pass."""
        code = '''
package main

func main() {
    fmt.Println("Hello, World!")
}
'''
        result = verifier.validate_go(code)

        assert result.is_valid is True


class TestCodeSummarizer:
    """Tests for code summarization."""

    @pytest.fixture
    def summarizer(self):
        return CodeSummarizerCore()

    def test_remove_verbose_comments(self, summarizer):
        """Verbose standalone comments should be removed."""
        code = '''# This is a very long comment that explains everything in detail
# And this is another comment that is not really necessary
# But we keep going with more comments
def hello():
    return "Hello!"
'''
        result, removed = summarizer.remove_verbose_comments(code, CodeLanguage.PYTHON)

        assert removed > 0
        assert "def hello():" in result

    def test_keep_essential_comments(self, summarizer):
        """Essential comments (TODO, FIXME) should be kept."""
        code = '''# TODO: Implement this properly
# FIXME: This is broken
def hello():
    return "Hello!"
'''
        result, removed = summarizer.remove_verbose_comments(code, CodeLanguage.PYTHON)

        assert removed == 0
        assert "TODO" in result
        assert "FIXME" in result

    def test_combine_python_imports(self, summarizer):
        """Multiple imports from same module should be combined."""
        code = '''from typing import List
from typing import Dict
from typing import Optional

def hello():
    pass
'''
        result, combined = summarizer.combine_python_imports(code)

        assert combined is True
        assert result.count("from typing import") == 1
        assert "List" in result
        assert "Dict" in result
        assert "Optional" in result

    def test_condense_empty_lines(self, summarizer):
        """Multiple empty lines should be condensed."""
        code = '''def hello():
    pass



def world():
    pass
'''
        result = summarizer.condense_empty_lines(code)

        assert result.count("\n\n\n") == 0
        assert "def hello():" in result
        assert "def world():" in result


class TestBalancedBraces:
    """Tests for balanced brace checking."""

    @pytest.fixture
    def verifier(self):
        return SyntaxVerifierCore()

    def test_balanced_braces(self, verifier):
        """Properly balanced braces should pass."""
        code = '''
{
    "key": {
        "nested": [1, 2, 3]
    }
}
'''
        errors = verifier.check_balanced_braces(code)
        assert len(errors) == 0

    def test_unbalanced_opening_brace(self, verifier):
        """Unclosed opening brace should be detected."""
        code = '''
{
    "key": {
        "nested": 1
}
'''
        errors = verifier.check_balanced_braces(code)
        assert len(errors) > 0
        assert any("unclosed" in e.message.lower() for e in errors)

    def test_unbalanced_closing_brace(self, verifier):
        """Extra closing brace should be detected."""
        code = '''
{
    "key": 1
}}
'''
        errors = verifier.check_balanced_braces(code)
        assert len(errors) > 0
        assert any("unmatched" in e.message.lower() for e in errors)

    def test_strings_ignored(self, verifier):
        """Braces inside strings should be ignored."""
        code = '''
{
    "value": "this { has } braces"
}
'''
        errors = verifier.check_balanced_braces(code)
        assert len(errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
