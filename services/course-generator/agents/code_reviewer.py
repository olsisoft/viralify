"""
Code Reviewer Agent

Validates code quality before it's included in the course.
Acts as a gatekeeper to ensure only production-quality code passes through.
"""
import json
import re
from typing import Dict, Any, List, Optional

from agents.base import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentResult,
    CourseGenerationState,
    CodeBlockState,
)


# Patterns that indicate lazy/placeholder code
LAZY_CODE_PATTERNS = [
    (r'\bTODO\b', "Contains TODO comment"),
    (r'\bFIXME\b', "Contains FIXME comment"),
    (r'\.\.\.', "Contains ellipsis placeholder"),
    (r'pass\s*#', "Contains empty pass statement"),
    (r'#\s*implement', "Contains 'implement' comment"),
    (r'#\s*your code', "Contains 'your code' placeholder"),
    (r'raise NotImplementedError', "Contains NotImplementedError"),
    (r'print\s*\(\s*["\']hello\s*world["\']', "Contains trivial hello world"),
]

# Minimum complexity indicators by level
COMPLEXITY_REQUIREMENTS = {
    "beginner": {
        "min_lines": 5,
        "min_functions": 0,
        "min_classes": 0,
        "require_comments": True,
        "require_error_handling": False,
    },
    "intermediate": {
        "min_lines": 15,
        "min_functions": 1,
        "min_classes": 0,
        "require_comments": True,
        "require_error_handling": True,
    },
    "advanced": {
        "min_lines": 30,
        "min_functions": 2,
        "min_classes": 1,
        "require_comments": True,
        "require_error_handling": True,
    },
    "expert": {
        "min_lines": 50,
        "min_functions": 3,
        "min_classes": 1,
        "require_comments": True,
        "require_error_handling": True,
    },
}

# Review prompt for LLM-based quality check
CODE_REVIEW_PROMPT = """### ROLE
Tu es un Technical Reviewer intransigeant. Ton rôle est de valider si le code
produit est digne d'un cours expert ou s'il est "trop léger".

### CODE À REVIEWER
```{language}
{code}
```

### NIVEAU ATTENDU DE L'APPRENANT
{persona_level}

### CONCEPT À DÉMONTRER
{concept}

### CRITÈRES DE REJET (Si l'un d'eux est vrai, REJETTE)
1. **"Hello World" Syndrome:** Le code est trop simpliste pour le niveau du Persona.
2. **Manque de Robustesse:** Aucune gestion d'erreurs, pas de validation d'inputs.
3. **Hardcoding:** Valeurs en dur au lieu de paramètres configurables.
4. **Paresse:** Commentaires "TODO" ou "Implémentez ici".
5. **Non-Exécutable:** Le code ne peut pas tourner sans modifications.
6. **Hors-Sujet:** Le code ne démontre pas le concept demandé.

### PROCESSUS D'ANALYSE
1. Vérifie la cohérence entre le niveau Persona et la complexité du code.
2. Analyse si le code démontre correctement le concept.
3. Évalue la "Propreté" (Clean Code): noms de variables, modularité.
4. Vérifie que le code est exécutable tel quel.

### FORMAT DE RÉPONSE (JSON STRICT)
{{
  "status": "APPROVED" ou "REJECTED",
  "complexity_assessment": {{
    "expected_for_persona": 1-10,
    "actual": 1-10,
    "gap_description": "description de l'écart"
  }},
  "rejection_reasons": ["raison 1", "raison 2"],
  "quality_score": 1-10,
  "suggestions": ["amélioration 1", "amélioration 2"],
  "retry_prompt": "Si REJECTED, prompt pour régénérer un meilleur code"
}}
"""


class CodeReviewerAgent(BaseAgent):
    """
    Reviews code quality before inclusion in the course.

    This agent performs:
    1. Static analysis (pattern matching for lazy code)
    2. Complexity checks based on persona level
    3. LLM-based quality review
    4. Provides feedback for code refinement if rejected

    Uses the fast (cheaper) model tier since review is an evaluation
    task that doesn't require the same generation capabilities as
    code creation.
    """

    # Review/evaluation uses the fast (cheaper) model
    MODEL_TIER = "fast"

    def __init__(self):
        super().__init__(AgentType.CODE_REVIEWER)
        self.max_retries = 3

    async def process(self, state: CourseGenerationState) -> CourseGenerationState:
        """
        Review the current code block.

        Args:
            state: Current course generation state

        Returns:
            State updated with review results
        """
        code_block = state.get("current_code_block")

        if not code_block:
            self.log("No code block to review")
            return state

        code = code_block.get("refined_code") or code_block.get("raw_code", "")
        language = code_block.get("language", "python")
        persona_level = code_block.get("persona_level", state.get("persona_level", "intermediate"))
        concept = code_block.get("concept", "")

        self.log(f"Reviewing code for concept: {concept[:50]}... (Level: {persona_level})")

        # Step 1: Static analysis
        static_issues = self._static_analysis(code, language)

        if static_issues:
            self.log(f"Static analysis found {len(static_issues)} issues")
            code_block["review_status"] = "rejected"
            code_block["rejection_reasons"] = static_issues
            code_block["retry_count"] = code_block.get("retry_count", 0) + 1

            if code_block["retry_count"] < self.max_retries:
                code_block["retry_needed"] = True
                code_block["retry_prompt"] = self._build_retry_prompt(static_issues)

            state["current_code_block"] = code_block
            state["code_blocks_rejected"] = state.get("code_blocks_rejected", 0) + 1

            self.add_to_history(state, AgentStatus.COMPLETED)
            return state

        # Step 2: Complexity check
        complexity_ok, complexity_issues = self._check_complexity(
            code, language, persona_level
        )

        if not complexity_ok:
            self.log(f"Complexity check failed: {complexity_issues}")
            code_block["review_status"] = "rejected"
            code_block["rejection_reasons"] = complexity_issues
            code_block["retry_count"] = code_block.get("retry_count", 0) + 1

            if code_block["retry_count"] < self.max_retries:
                code_block["retry_needed"] = True
                code_block["retry_prompt"] = self._build_retry_prompt(complexity_issues)

            state["current_code_block"] = code_block
            state["code_blocks_rejected"] = state.get("code_blocks_rejected", 0) + 1

            self.add_to_history(state, AgentStatus.COMPLETED)
            return state

        # Step 3: LLM-based quality review
        review_result = await self._llm_review(
            code, language, persona_level, concept
        )

        if review_result.data.get("status") == "APPROVED":
            code_block["review_status"] = "approved"
            code_block["quality_score"] = review_result.data.get("quality_score", 7)
            state["code_blocks_approved"] = state.get("code_blocks_approved", 0) + 1
            self.log(f"Code APPROVED with quality score: {code_block['quality_score']}/10")
        else:
            code_block["review_status"] = "rejected"
            code_block["rejection_reasons"] = review_result.data.get("rejection_reasons", [])
            code_block["retry_count"] = code_block.get("retry_count", 0) + 1

            if code_block["retry_count"] < self.max_retries:
                code_block["retry_needed"] = True
                code_block["retry_prompt"] = review_result.data.get("retry_prompt", "")

            state["code_blocks_rejected"] = state.get("code_blocks_rejected", 0) + 1
            self.log(f"Code REJECTED: {code_block['rejection_reasons']}")

        state["current_code_block"] = code_block
        self.add_to_history(state, AgentStatus.COMPLETED, review_result)

        return state

    def _static_analysis(self, code: str, language: str) -> List[str]:
        """
        Perform static analysis to detect lazy code patterns.

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            List of issues found
        """
        issues = []

        for pattern, message in LAZY_CODE_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(message)

        # Language-specific checks
        if language == "python":
            # Check for bare except
            if re.search(r'except\s*:', code):
                issues.append("Contains bare except clause (should specify exception type)")

            # Check for single-letter variables (except i, j, k in loops)
            single_vars = re.findall(r'\b([a-hln-z])\s*=', code)
            if single_vars:
                issues.append(f"Contains non-descriptive variable names: {single_vars[:3]}")

        elif language in ["javascript", "typescript"]:
            # Check for var instead of let/const
            if re.search(r'\bvar\s+\w+', code):
                issues.append("Uses 'var' instead of 'let' or 'const'")

            # Check for == instead of ===
            if re.search(r'[^=!]==[^=]', code):
                issues.append("Uses loose equality (==) instead of strict equality (===)")

        return issues

    def _check_complexity(
        self,
        code: str,
        language: str,
        persona_level: str
    ) -> tuple[bool, List[str]]:
        """
        Check if code complexity matches the persona level.

        Args:
            code: Code to check
            language: Programming language
            persona_level: Expected complexity level

        Returns:
            Tuple of (is_ok, issues_list)
        """
        issues = []
        level = persona_level.lower()
        requirements = COMPLEXITY_REQUIREMENTS.get(level, COMPLEXITY_REQUIREMENTS["intermediate"])

        # Count lines (excluding empty and comment-only lines)
        lines = [l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        line_count = len(lines)

        if line_count < requirements["min_lines"]:
            issues.append(
                f"Code too short for {level} level: {line_count} lines "
                f"(minimum {requirements['min_lines']})"
            )

        # Count functions
        if language == "python":
            func_count = len(re.findall(r'\bdef\s+\w+', code))
            class_count = len(re.findall(r'\bclass\s+\w+', code))
        elif language in ["javascript", "typescript"]:
            func_count = len(re.findall(r'\bfunction\s+\w+|\w+\s*=\s*(?:async\s*)?\(', code))
            class_count = len(re.findall(r'\bclass\s+\w+', code))
        else:
            func_count = 0
            class_count = 0

        if func_count < requirements["min_functions"]:
            issues.append(
                f"Not enough functions for {level} level: {func_count} "
                f"(minimum {requirements['min_functions']})"
            )

        if class_count < requirements["min_classes"]:
            issues.append(
                f"Not enough classes for {level} level: {class_count} "
                f"(minimum {requirements['min_classes']})"
            )

        # Check for error handling
        if requirements["require_error_handling"]:
            has_error_handling = (
                re.search(r'\btry\s*:', code) or  # Python
                re.search(r'\btry\s*\{', code) or  # JS/Java
                re.search(r'\.catch\s*\(', code)   # Promise catch
            )
            if not has_error_handling:
                issues.append(f"Missing error handling for {level} level code")

        # Check for comments/docstrings
        if requirements["require_comments"]:
            has_comments = (
                re.search(r'"""[\s\S]*?"""', code) or  # Docstring
                re.search(r"'''[\s\S]*?'''", code) or  # Docstring
                re.search(r'#\s*\w+', code) or         # Python comment
                re.search(r'//\s*\w+', code) or        # JS comment
                re.search(r'/\*[\s\S]*?\*/', code)     # Block comment
            )
            if not has_comments:
                issues.append("Missing comments/documentation")

        return len(issues) == 0, issues

    async def _llm_review(
        self,
        code: str,
        language: str,
        persona_level: str,
        concept: str
    ) -> AgentResult:
        """
        Perform LLM-based quality review.

        Args:
            code: Code to review
            language: Programming language
            persona_level: Expected complexity level
            concept: Concept being demonstrated

        Returns:
            AgentResult with review decision
        """
        prompt = CODE_REVIEW_PROMPT.format(
            language=language,
            code=code,
            persona_level=persona_level,
            concept=concept
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature for consistent reviews
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            return AgentResult(
                success=True,
                data=result,
                metadata={"review_type": "llm"}
            )

        except Exception as e:
            self.log(f"LLM review error: {e}")
            # On error, be lenient and approve
            return AgentResult(
                success=True,
                data={
                    "status": "APPROVED",
                    "quality_score": 6,
                    "suggestions": ["LLM review unavailable - manual review recommended"]
                },
                warnings=[f"LLM review failed: {str(e)}"]
            )

    def _build_retry_prompt(self, issues: List[str]) -> str:
        """Build a prompt for code regeneration based on issues."""
        issues_text = "\n".join(f"- {issue}" for issue in issues)

        return f"""Le code a été REJETÉ pour les raisons suivantes:

{issues_text}

INSTRUCTIONS POUR LA RÉGÉNÉRATION:
1. Corrige TOUS les problèmes listés ci-dessus.
2. Assure-toi que le code est complet et exécutable.
3. Adapte la complexité au niveau de l'apprenant.
4. Inclus une gestion d'erreurs appropriée.
5. Utilise des noms de variables descriptifs.
6. Ajoute des commentaires expliquant la logique.

Régénère un code de qualité production qui passe ces vérifications."""

    async def review_code(
        self,
        code: str,
        language: str,
        persona_level: str,
        concept: str
    ) -> Dict[str, Any]:
        """
        Convenience method to review code without full state.

        Args:
            code: Code to review
            language: Programming language
            persona_level: Expected complexity level
            concept: Concept being demonstrated

        Returns:
            Review result dictionary
        """
        # Static analysis
        static_issues = self._static_analysis(code, language)
        if static_issues:
            return {
                "status": "REJECTED",
                "rejection_reasons": static_issues,
                "retry_prompt": self._build_retry_prompt(static_issues)
            }

        # Complexity check
        complexity_ok, complexity_issues = self._check_complexity(
            code, language, persona_level
        )
        if not complexity_ok:
            return {
                "status": "REJECTED",
                "rejection_reasons": complexity_issues,
                "retry_prompt": self._build_retry_prompt(complexity_issues)
            }

        # LLM review
        result = await self._llm_review(code, language, persona_level, concept)
        return result.data


def create_code_reviewer() -> CodeReviewerAgent:
    """Factory function to create a CodeReviewerAgent"""
    return CodeReviewerAgent()
