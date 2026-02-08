"""
Code Summarizer for Presentation Slides

Reduces code length for visual display while preserving key concepts.
Keeps full code for execution, provides summarized version for slides.
"""

import re
import os
from typing import Optional, List, Tuple
from openai import AsyncOpenAI

from .models import CodeLanguage, SummarizedCode


class CodeSummarizer:
    """
    Summarizes code for presentation display.

    Strategies:
    1. Remove verbose comments (keep essential ones)
    2. Combine imports on one line
    3. Remove long docstrings
    4. Replace secondary implementations with ellipsis
    5. Keep main structure visible
    """

    # Default max lines for slide display
    DEFAULT_MAX_LINES = 25

    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """
        Initialize summarizer.

        Args:
            client: OpenAI client for LLM-based summarization (optional)
        """
        self.client = client or AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("CODE_SUMMARIZER_MODEL", "gpt-4o-mini")

    async def summarize(
        self,
        code: str,
        language: CodeLanguage,
        max_lines: int = DEFAULT_MAX_LINES,
        preserve_key_lines: Optional[List[int]] = None
    ) -> SummarizedCode:
        """
        Summarize code for slide display.

        Args:
            code: Full source code
            language: Programming language
            max_lines: Maximum lines for display (default: 25)
            preserve_key_lines: Line numbers to always keep

        Returns:
            SummarizedCode with display and full versions
        """
        lines = code.split('\n')
        original_count = len(lines)

        # If already short enough, return as-is
        if original_count <= max_lines:
            return SummarizedCode(
                display_code=code,
                full_code=code,
                lines_removed=0,
                summary_strategy="none",
                key_sections_preserved=["all"]
            )

        # Apply summarization strategies in order
        strategies_applied = []
        display_code = code

        # Strategy 1: Remove verbose comments
        display_code, removed_comments = self._remove_verbose_comments(display_code, language)
        if removed_comments > 0:
            strategies_applied.append(f"comments:{removed_comments}")

        # Strategy 2: Combine imports
        display_code, combined_imports = self._combine_imports(display_code, language)
        if combined_imports:
            strategies_applied.append("imports")

        # Strategy 3: Remove long docstrings
        display_code, removed_docs = self._remove_docstrings(display_code, language)
        if removed_docs > 0:
            strategies_applied.append(f"docstrings:{removed_docs}")

        # Strategy 4: Condense empty lines
        display_code = self._condense_empty_lines(display_code)

        # Check if we're within limit
        current_lines = len(display_code.split('\n'))
        if current_lines <= max_lines:
            return SummarizedCode(
                display_code=display_code.strip(),
                full_code=code,
                lines_removed=original_count - current_lines,
                summary_strategy=",".join(strategies_applied),
                key_sections_preserved=self._identify_key_sections(display_code, language)
            )

        # Strategy 5: Replace secondary implementations with ellipsis
        display_code = await self._replace_with_ellipsis(
            display_code,
            language,
            max_lines,
            preserve_key_lines
        )
        strategies_applied.append("ellipsis")

        current_lines = len(display_code.split('\n'))

        return SummarizedCode(
            display_code=display_code.strip(),
            full_code=code,
            lines_removed=original_count - current_lines,
            summary_strategy=",".join(strategies_applied),
            key_sections_preserved=self._identify_key_sections(display_code, language)
        )

    def _remove_verbose_comments(
        self,
        code: str,
        language: CodeLanguage
    ) -> Tuple[str, int]:
        """
        Remove verbose comments while keeping essential ones.

        Essential comments:
        - TODO, FIXME, NOTE, IMPORTANT
        - Comments on the same line as code
        - Section headers (# === or // ---)
        """
        lines = code.split('\n')
        result_lines = []
        removed_count = 0

        # Comment patterns by language
        if language in (CodeLanguage.PYTHON, CodeLanguage.RUBY, CodeLanguage.BASH):
            comment_pattern = r'^\s*#'
            inline_pattern = r'.*\S.*#'
        elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT,
                          CodeLanguage.JAVA, CodeLanguage.GO, CodeLanguage.RUST,
                          CodeLanguage.CSHARP, CodeLanguage.KOTLIN, CodeLanguage.SWIFT,
                          CodeLanguage.C, CodeLanguage.CPP, CodeLanguage.SCALA):
            comment_pattern = r'^\s*//'
            inline_pattern = r'.*\S.*//'
        else:
            comment_pattern = r'^\s*#'  # Default
            inline_pattern = r'.*\S.*#'

        essential_markers = ['TODO', 'FIXME', 'NOTE', 'IMPORTANT', 'WARNING', 'HACK']
        section_markers = ['===', '---', '***', '###']

        for line in lines:
            # Check if line is a standalone comment
            if re.match(comment_pattern, line):
                # Keep essential comments
                if any(marker in line.upper() for marker in essential_markers):
                    result_lines.append(line)
                    continue

                # Keep section headers
                if any(marker in line for marker in section_markers):
                    result_lines.append(line)
                    continue

                # Remove verbose standalone comments
                removed_count += 1
                continue

            result_lines.append(line)

        return '\n'.join(result_lines), removed_count

    def _combine_imports(
        self,
        code: str,
        language: CodeLanguage
    ) -> Tuple[str, bool]:
        """
        Combine multiple import statements where possible.
        """
        if language == CodeLanguage.PYTHON:
            return self._combine_python_imports(code)
        elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT):
            return self._combine_js_imports(code)
        else:
            return code, False

    def _combine_python_imports(self, code: str) -> Tuple[str, bool]:
        """Combine Python imports from the same module."""
        lines = code.split('\n')
        import_groups = {}  # module -> [items]
        non_import_lines = []
        first_import_idx = -1

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Match "from X import Y, Z"
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

        # Rebuild with combined imports
        combined_imports = []
        for module, items in sorted(import_groups.items()):
            unique_items = sorted(set(items))
            combined_imports.append(f"from {module} import {', '.join(unique_items)}")

        # Insert combined imports at first import position
        result = non_import_lines[:first_import_idx] + combined_imports + non_import_lines[first_import_idx:]
        return '\n'.join(result), True

    def _combine_js_imports(self, code: str) -> Tuple[str, bool]:
        """Combine JavaScript/TypeScript imports from the same module."""
        lines = code.split('\n')
        import_groups = {}  # module -> [items]
        non_import_lines = []
        first_import_idx = -1

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Match "import { X, Y } from 'module'"
            match = re.match(r"import\s*\{\s*([^}]+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]", stripped)
            if match:
                items = [item.strip() for item in match.group(1).split(',')]
                module = match.group(2)
                if module not in import_groups:
                    import_groups[module] = []
                    if first_import_idx < 0:
                        first_import_idx = len(non_import_lines)
                import_groups[module].extend(items)
                continue

            non_import_lines.append(line)

        if not import_groups:
            return code, False

        # Rebuild with combined imports
        combined_imports = []
        for module, items in sorted(import_groups.items()):
            unique_items = sorted(set(items))
            combined_imports.append(f"import {{ {', '.join(unique_items)} }} from '{module}'")

        result = non_import_lines[:first_import_idx] + combined_imports + non_import_lines[first_import_idx:]
        return '\n'.join(result), True

    def _remove_docstrings(
        self,
        code: str,
        language: CodeLanguage
    ) -> Tuple[str, int]:
        """
        Remove or shorten long docstrings.
        """
        if language != CodeLanguage.PYTHON:
            return code, 0

        # Pattern for multi-line docstrings
        docstring_pattern = r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'

        removed_count = 0

        def replace_docstring(match):
            nonlocal removed_count
            docstring = match.group(0)
            lines_in_doc = docstring.count('\n')

            # Keep short docstrings (1-2 lines)
            if lines_in_doc <= 2:
                return docstring

            # Shorten long docstrings to first line + ellipsis
            first_line = docstring.split('\n')[0]
            quote_style = '"""' if docstring.startswith('"""') else "'''"
            removed_count += 1
            return f'{quote_style}{first_line.strip().strip(quote_style)}...{quote_style}'

        result = re.sub(docstring_pattern, replace_docstring, code)
        return result, removed_count

    def _condense_empty_lines(self, code: str) -> str:
        """Replace multiple consecutive empty lines with a single empty line."""
        return re.sub(r'\n{3,}', '\n\n', code)

    async def _replace_with_ellipsis(
        self,
        code: str,
        language: CodeLanguage,
        max_lines: int,
        preserve_key_lines: Optional[List[int]] = None
    ) -> str:
        """
        Replace secondary implementations with ellipsis.

        Uses LLM to identify which parts are essential vs. which can be replaced.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a code summarizer for presentation slides.

Your task: Reduce code to {max_lines} lines while keeping the ESSENTIAL parts visible.

Rules:
1. Keep function/class signatures and the FIRST few lines of implementation
2. Replace repetitive or secondary logic with "# ... (implementation details)"
3. Keep imports (already optimized)
4. NEVER remove the main concept being demonstrated
5. Use language-appropriate ellipsis comments

Return ONLY the summarized code, no explanations."""
                    },
                    {
                        "role": "user",
                        "content": f"""Summarize this {language.value} code to max {max_lines} lines:

```{language.value}
{code}
```

Return only the summarized code:"""
                    }
                ],
                temperature=0.0
            )

            summarized = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            if summarized.startswith("```"):
                lines = summarized.split('\n')
                summarized = '\n'.join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            return summarized

        except Exception as e:
            print(f"[CODE_SUMMARIZER] LLM summarization failed: {e}", flush=True)
            # Fallback: truncate with ellipsis
            return self._truncate_with_ellipsis(code, language, max_lines)

    def _truncate_with_ellipsis(
        self,
        code: str,
        language: CodeLanguage,
        max_lines: int
    ) -> str:
        """
        Simple fallback: keep first N-1 lines + ellipsis.
        """
        lines = code.split('\n')
        if len(lines) <= max_lines:
            return code

        # Determine ellipsis comment style
        if language in (CodeLanguage.PYTHON, CodeLanguage.RUBY, CodeLanguage.BASH):
            ellipsis = "# ... (continued)"
        elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT,
                          CodeLanguage.JAVA, CodeLanguage.GO, CodeLanguage.RUST,
                          CodeLanguage.CSHARP, CodeLanguage.KOTLIN, CodeLanguage.SWIFT,
                          CodeLanguage.C, CodeLanguage.CPP, CodeLanguage.SCALA):
            ellipsis = "// ... (continued)"
        else:
            ellipsis = "# ... (continued)"

        return '\n'.join(lines[:max_lines - 1]) + '\n' + ellipsis

    def _identify_key_sections(
        self,
        code: str,
        language: CodeLanguage
    ) -> List[str]:
        """
        Identify key sections preserved in summarized code.
        """
        sections = []

        # Check for common patterns
        if language == CodeLanguage.PYTHON:
            if re.search(r'\bdef\s+\w+', code):
                sections.append("functions")
            if re.search(r'\bclass\s+\w+', code):
                sections.append("classes")
            if re.search(r'^import\s+|^from\s+', code, re.MULTILINE):
                sections.append("imports")

        elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT):
            if re.search(r'\bfunction\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>', code):
                sections.append("functions")
            if re.search(r'\bclass\s+\w+', code):
                sections.append("classes")
            if re.search(r'^import\s+', code, re.MULTILINE):
                sections.append("imports")

        elif language == CodeLanguage.JAVA:
            if re.search(r'\bpublic\s+\w+\s+\w+\s*\(', code):
                sections.append("methods")
            if re.search(r'\bclass\s+\w+', code):
                sections.append("classes")

        elif language == CodeLanguage.GO:
            if re.search(r'\bfunc\s+', code):
                sections.append("functions")
            if re.search(r'\btype\s+\w+\s+struct', code):
                sections.append("structs")

        return sections if sections else ["main"]


# Singleton instance
_code_summarizer: Optional[CodeSummarizer] = None


def get_code_summarizer() -> CodeSummarizer:
    """Get singleton CodeSummarizer instance."""
    global _code_summarizer
    if _code_summarizer is None:
        _code_summarizer = CodeSummarizer()
    return _code_summarizer
