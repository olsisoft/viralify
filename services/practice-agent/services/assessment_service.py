"""
Assessment Service

Evaluates learner submissions and generates assessments.
"""

from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from models.practice_models import Exercise, ExerciseAttempt
from models.sandbox_models import ExecutionResult, SandboxResult
from models.assessment_models import (
    AssessmentResult,
    CodeAnalysis,
    CodeQualityMetric,
    PedagogicalFeedback,
    FeedbackType,
    UnderstandingLevel,
)
from agents.feedback_generator import FeedbackGenerator

try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    _USE_SHARED_LLM = False


class AssessmentService:
    """
    Evaluates learner submissions against exercise requirements.

    Features:
    - Code correctness validation
    - Quality assessment
    - Pedagogical feedback generation
    - Understanding level estimation
    """

    def __init__(self):
        self.llm = ChatOpenAI(model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini", temperature=0.3)
        self.feedback_generator = FeedbackGenerator()

    async def evaluate(
        self,
        exercise: Dict[str, Any],
        submitted_code: str,
        execution_result: Optional[Dict[str, Any]],
    ) -> AssessmentResult:
        """
        Evaluate a submission against exercise requirements.
        """
        # Create attempt record
        attempt = ExerciseAttempt(
            exercise_id=exercise.get("id", "unknown"),
            submitted_code=submitted_code,
            execution_output=execution_result.get("execution", {}).get("stdout", "") if execution_result else "",
            execution_errors=execution_result.get("execution", {}).get("stderr", "") if execution_result else "",
        )

        # Run validation checks
        checks_passed = []
        checks_failed = []
        partial_credit = {}
        total_score = 0
        max_score = 0

        validation_checks = exercise.get("validation_checks", [])
        expected_outputs = exercise.get("expected_outputs", [])

        # Check expected outputs
        for expected in expected_outputs:
            check_name = expected.get("name", expected.get("type", "check"))
            points = expected.get("points", 10)
            max_score += points

            passed = self._check_output(
                expected,
                execution_result.get("execution", {}) if execution_result else {},
            )

            if passed:
                checks_passed.append(check_name)
                total_score += points
                partial_credit[check_name] = points
            else:
                checks_failed.append(check_name)
                partial_credit[check_name] = 0

        # Run validation checks
        for check in validation_checks:
            check_name = check.get("name", "validation")
            points = check.get("points", 10)
            max_score += points

            passed = await self._run_validation_check(
                check,
                submitted_code,
                execution_result,
            )

            if passed:
                checks_passed.append(check_name)
                total_score += points
            else:
                checks_failed.append(check_name)

        # Determine pass/fail
        # Pass if >70% score and no required checks failed
        required_checks = [c.get("name") for c in validation_checks if c.get("required", True)]
        required_failed = [c for c in required_checks if c in checks_failed]
        passed = (total_score / max_score >= 0.7 if max_score > 0 else False) and not required_failed

        # Analyze code quality
        code_analysis = await self._analyze_code(
            submitted_code,
            exercise,
            execution_result,
        )

        # Assess understanding level
        understanding = self._assess_understanding(
            passed,
            len(checks_passed),
            len(checks_failed),
            code_analysis,
        )

        # Generate feedback
        assessment_result = AssessmentResult(
            exercise_id=exercise.get("id", "unknown"),
            attempt_id=attempt.id,
            passed=passed,
            score=total_score,
            max_score=max_score,
            percentage=round(total_score / max_score * 100, 1) if max_score > 0 else 0,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            partial_credit=partial_credit,
            code_analysis=code_analysis,
            understanding_level=understanding,
        )

        # Generate summary feedback
        assessment_result.summary_feedback = await self._generate_summary_feedback(
            exercise,
            attempt,
            assessment_result,
        )

        return assessment_result

    def _check_output(
        self,
        expected: Dict[str, Any],
        execution: Dict[str, Any],
    ) -> bool:
        """Check if execution output matches expected"""
        check_type = expected.get("type", "stdout")
        stdout = execution.get("stdout", "")
        stderr = execution.get("stderr", "")
        exit_code = execution.get("exit_code", -1)

        if check_type == "stdout" or check_type == "output":
            # Check stdout
            if expected.get("exact_match"):
                return stdout.strip() == expected["exact_match"].strip()
            if expected.get("contains"):
                return all(s in stdout for s in expected["contains"])
            if expected.get("not_contains"):
                return all(s not in stdout for s in expected["not_contains"])
            if expected.get("pattern"):
                import re
                return bool(re.search(expected["pattern"], stdout))

        elif check_type == "exit_code":
            expected_code = expected.get("expected_value", 0)
            return exit_code == expected_code

        elif check_type == "no_error":
            return exit_code == 0 and not stderr

        elif check_type == "stderr":
            if expected.get("contains"):
                return all(s in stderr for s in expected["contains"])
            if expected.get("empty"):
                return not stderr.strip()

        return False

    async def _run_validation_check(
        self,
        check: Dict[str, Any],
        code: str,
        execution_result: Optional[Dict[str, Any]],
    ) -> bool:
        """Run a custom validation check"""
        check_type = check.get("check_type", "code_contains")

        if check_type == "code_contains":
            patterns = check.get("patterns", [])
            return all(p in code for p in patterns)

        elif check_type == "code_not_contains":
            patterns = check.get("patterns", [])
            return all(p not in code for p in patterns)

        elif check_type == "code_structure":
            # Use LLM to check code structure
            prompt = f"""
Vérifie si ce code respecte la structure demandée.

Code:
```
{code}
```

Structure requise: {check.get("description", "")}

Réponds uniquement par "true" ou "false".
"""
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip().lower() == "true"

        elif check_type == "custom":
            # Custom validation script would be executed in sandbox
            # For now, always pass
            return True

        return False

    async def _analyze_code(
        self,
        code: str,
        exercise: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]],
    ) -> CodeAnalysis:
        """Analyze code quality"""
        analysis = CodeAnalysis(
            compiles=execution_result.get("execution", {}).get("exit_code", 1) == 0 if execution_result else False,
            runs_without_error=not execution_result.get("execution", {}).get("stderr") if execution_result else False,
        )

        # Use LLM to analyze code quality
        prompt = f"""
Analyse ce code pour l'exercice "{exercise.get('title', 'Exercice')}".

Code:
```
{code}
```

Évalue sur ces critères (score 0-100 chacun):
1. Lisibilité: clarté, nommage, structure
2. Bonnes pratiques: patterns, conventions
3. Efficacité: performance, ressources
4. Sécurité: vulnérabilités potentielles

Réponds en JSON:
{{
    "readability": {{"score": 80, "feedback": "..."}},
    "best_practices": {{"score": 70, "feedback": "..."}},
    "efficiency": {{"score": 85, "feedback": "..."}},
    "security": {{"score": 90, "feedback": "..."}},
    "overall_score": 81,
    "main_issues": ["issue 1", "issue 2"],
    "follows_best_practices": true
}}
"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="Tu es un expert en revue de code."),
                HumanMessage(content=prompt)
            ])

            import json
            data = json.loads(response.content)

            # Build metrics
            for metric_name in ["readability", "best_practices", "efficiency", "security"]:
                if metric_name in data:
                    analysis.metrics.append(CodeQualityMetric(
                        name=metric_name,
                        score=data[metric_name].get("score", 0),
                        feedback=data[metric_name].get("feedback", ""),
                    ))

            analysis.overall_quality_score = data.get("overall_score", 0)
            analysis.follows_best_practices = data.get("follows_best_practices", False)
            analysis.logic_errors = data.get("main_issues", [])

        except Exception as e:
            print(f"[ASSESSMENT] Code analysis error: {e}")

        return analysis

    def _assess_understanding(
        self,
        passed: bool,
        checks_passed: int,
        checks_failed: int,
        code_analysis: CodeAnalysis,
    ) -> UnderstandingLevel:
        """Assess the learner's understanding level"""
        if not passed:
            if checks_passed == 0:
                return UnderstandingLevel.NONE
            elif checks_passed < checks_failed:
                return UnderstandingLevel.SURFACE
            else:
                return UnderstandingLevel.DEVELOPING

        # Passed - assess quality
        quality_score = code_analysis.overall_quality_score

        if quality_score >= 90 and code_analysis.follows_best_practices:
            return UnderstandingLevel.EXPERT
        elif quality_score >= 75:
            return UnderstandingLevel.PROFICIENT
        elif quality_score >= 60:
            return UnderstandingLevel.FUNCTIONAL
        else:
            return UnderstandingLevel.DEVELOPING

    async def _generate_summary_feedback(
        self,
        exercise: Dict[str, Any],
        attempt: ExerciseAttempt,
        assessment: AssessmentResult,
    ) -> str:
        """Generate a summary feedback message"""
        if assessment.passed:
            prompt = f"""
L'apprenant a réussi l'exercice "{exercise.get('title', '')}" avec {assessment.score}/{assessment.max_score} points.

Génère un message de félicitations court (2-3 phrases) qui:
1. Félicite spécifiquement
2. Mentionne un point fort
3. Suggère optionnellement une amélioration

Sois enthousiaste mais concis.
"""
        else:
            prompt = f"""
L'apprenant n'a pas encore réussi l'exercice "{exercise.get('title', '')}".
Score: {assessment.score}/{assessment.max_score}
Vérifications échouées: {', '.join(assessment.checks_failed)}

Génère un message d'encouragement court (2-3 phrases) qui:
1. Reste positif
2. Indique la direction à prendre (sans donner la solution)
3. Encourage à réessayer

Sois encourageant et constructif.
"""

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content
