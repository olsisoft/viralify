"""
Unit tests for LANGUAGE_VALIDATION_PROMPT

Tests the prompt structure, constraints, and validation rules
without making actual LLM calls.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Any


# ============================================================================
# Direct import to avoid dependency chain
# ============================================================================

def import_module_from_file(module_name: str, file_path: str):
    """Import a module directly from file path to avoid dependency issues."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Get the prompts module
prompts_path = Path(__file__).parent.parent / "agents" / "pedagogical_prompts.py"
prompts_module = import_module_from_file("pedagogical_prompts", str(prompts_path))
LANGUAGE_VALIDATION_PROMPT = prompts_module.LANGUAGE_VALIDATION_PROMPT


# ============================================================================
# LanguageValidationValidator
# ============================================================================

class LanguageValidationValidator:
    """Validates LLM output against LANGUAGE_VALIDATION_PROMPT constraints."""

    VALID_SEVERITIES = ["critical", "major", "minor", "suggestion"]
    VALID_QUALITY_LEVELS = ["excellent", "good", "needs_improvement", "poor"]

    # Technical terms that should NOT be flagged as issues
    ALLOWED_TECH_TERMS = [
        # Programming languages
        "python", "javascript", "go", "rust", "java", "typescript",
        # Frameworks/tools
        "react", "django", "kubernetes", "docker", "fastapi", "flask",
        # Protocols
        "http", "rest", "graphql", "grpc", "websocket",
        # Acronyms
        "api", "sdk", "cli", "ide", "sql", "nosql", "json", "xml",
        # File extensions
        ".py", ".js", ".tsx", ".ts", ".go", ".rs",
    ]

    def __init__(self):
        pass

    def validate_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the full output structure."""
        issues = []

        # Check required fields
        if "is_valid" not in output:
            issues.append("Missing 'is_valid' field")
        if "issues" not in output:
            issues.append("Missing 'issues' field")
        if "overall_language_quality" not in output:
            issues.append("Missing 'overall_language_quality' field")

        if issues:
            return {"is_valid": False, "issues": issues}

        # Validate is_valid is boolean
        if not isinstance(output.get("is_valid"), bool):
            issues.append("'is_valid' must be a boolean")

        # Validate issues array
        issue_validation = self.validate_issues(output.get("issues", []))
        issues.extend(issue_validation)

        # Validate overall_language_quality
        quality = output.get("overall_language_quality", "")
        if quality not in self.VALID_QUALITY_LEVELS:
            issues.append(f"Invalid quality level: '{quality}'")

        # Validate consistency between is_valid and issues
        consistency_issues = self.validate_consistency(output)
        issues.extend(consistency_issues)

        return {
            "is_valid": len(issues) == 0,
            "issues": issues
        }

    def validate_issues(self, issues_list: List[Dict]) -> List[str]:
        """Validate the issues array."""
        validation_issues = []

        for i, issue in enumerate(issues_list):
            prefix = f"Issue {i + 1}"

            # Check required fields
            if "location" not in issue:
                validation_issues.append(f"{prefix}: missing 'location'")
            if "issue" not in issue:
                validation_issues.append(f"{prefix}: missing 'issue' description")
            if "suggested_fix" not in issue:
                validation_issues.append(f"{prefix}: missing 'suggested_fix'")

            # Validate severity if present
            severity = issue.get("severity", "")
            if severity and severity not in self.VALID_SEVERITIES:
                validation_issues.append(f"{prefix}: invalid severity '{severity}'")

            # Check that location is specific
            location = issue.get("location", "")
            if location and len(location) < 3:
                validation_issues.append(f"{prefix}: location too vague")

            # Check that suggested_fix is not empty
            fix = issue.get("suggested_fix", "")
            if not fix or len(fix) < 3:
                validation_issues.append(f"{prefix}: suggested_fix is empty or too short")

        return validation_issues

    def validate_consistency(self, output: Dict[str, Any]) -> List[str]:
        """Validate consistency between is_valid, issues, and quality."""
        issues = []

        is_valid = output.get("is_valid", True)
        issues_list = output.get("issues", [])
        quality = output.get("overall_language_quality", "")

        # If is_valid is False, there should be issues
        if not is_valid and len(issues_list) == 0:
            issues.append("is_valid=false but no issues reported")

        # If is_valid is True, there should be no critical/major issues
        if is_valid:
            critical_major = [i for i in issues_list
                            if i.get("severity") in ["critical", "major"]]
            if critical_major:
                issues.append(
                    f"is_valid=true but {len(critical_major)} critical/major issues exist"
                )

        # Quality should match issue severity
        severities = [i.get("severity", "minor") for i in issues_list]
        if "critical" in severities and quality in ["excellent", "good"]:
            issues.append("Quality is 'excellent'/'good' but critical issues exist")
        if quality == "excellent" and len(issues_list) > 0:
            issues.append("Quality is 'excellent' but issues were reported")
        if quality == "poor" and len(issues_list) == 0:
            issues.append("Quality is 'poor' but no issues reported")

        return issues

    def check_tech_term_flagging(
        self,
        issues_list: List[Dict],
        content: str
    ) -> List[str]:
        """Check if technical terms were incorrectly flagged."""
        problems = []

        for issue in issues_list:
            issue_text = issue.get("issue", "").lower()
            location = issue.get("location", "").lower()

            # Check if the issue is about a tech term that should be allowed
            for term in self.ALLOWED_TECH_TERMS:
                if term in issue_text or term in location:
                    # If the issue is specifically about this tech term being untranslated
                    if "untranslated" in issue_text or "english" in issue_text:
                        if term.lower() in self.ALLOWED_TECH_TERMS:
                            problems.append(
                                f"Tech term '{term}' incorrectly flagged as issue"
                            )

        return problems


# ============================================================================
# Tests for Prompt Structure
# ============================================================================

class TestPromptStructure:
    """Tests for the overall prompt structure."""

    def test_prompt_has_role_definition(self):
        """Test that prompt defines the agent role."""
        assert "Language Compliance Validation Agent" in LANGUAGE_VALIDATION_PROMPT
        assert "autonomously" in LANGUAGE_VALIDATION_PROMPT

    def test_prompt_has_context_section(self):
        """Test that prompt has context section."""
        assert "## CONTEXT" in LANGUAGE_VALIDATION_PROMPT
        assert "Viralify" in LANGUAGE_VALIDATION_PROMPT

    def test_prompt_has_inputs_section(self):
        """Test that prompt has inputs section."""
        assert "## INPUTS" in LANGUAGE_VALIDATION_PROMPT
        assert "{target_language}" in LANGUAGE_VALIDATION_PROMPT
        assert "{language_name}" in LANGUAGE_VALIDATION_PROMPT
        assert "{outline_content}" in LANGUAGE_VALIDATION_PROMPT

    def test_prompt_has_responsibilities_section(self):
        """Test that prompt has agent responsibilities."""
        assert "## AGENT RESPONSIBILITIES" in LANGUAGE_VALIDATION_PROMPT

    def test_prompt_has_decision_rules_section(self):
        """Test that prompt has decision rules."""
        assert "## DECISION RULES" in LANGUAGE_VALIDATION_PROMPT
        assert "HARD CONSTRAINTS" in LANGUAGE_VALIDATION_PROMPT

    def test_prompt_has_self_validation_section(self):
        """Test that prompt has self-validation section."""
        assert "## SELF-VALIDATION" in LANGUAGE_VALIDATION_PROMPT

    def test_prompt_has_examples_section(self):
        """Test that prompt has examples."""
        assert "## EXAMPLES" in LANGUAGE_VALIDATION_PROMPT

    def test_prompt_has_output_contract(self):
        """Test that prompt has output contract."""
        assert "## OUTPUT CONTRACT" in LANGUAGE_VALIDATION_PROMPT
        assert "valid JSON only" in LANGUAGE_VALIDATION_PROMPT


# ============================================================================
# Tests for Placeholders
# ============================================================================

class TestPromptPlaceholders:
    """Tests for prompt placeholders."""

    def test_target_language_placeholder(self):
        """Test target_language placeholder exists."""
        assert "{target_language}" in LANGUAGE_VALIDATION_PROMPT

    def test_language_name_placeholder(self):
        """Test language_name placeholder exists."""
        assert "{language_name}" in LANGUAGE_VALIDATION_PROMPT

    def test_outline_content_placeholder(self):
        """Test outline_content placeholder exists."""
        assert "{outline_content}" in LANGUAGE_VALIDATION_PROMPT

    def test_all_placeholders_can_be_formatted(self):
        """Test that all placeholders can be formatted."""
        formatted = LANGUAGE_VALIDATION_PROMPT.format(
            target_language="fr",
            language_name="French",
            outline_content="Title: Test Course\nSection: Introduction",
        )
        assert "{target_language}" not in formatted
        assert "{language_name}" not in formatted
        assert "{outline_content}" not in formatted


# ============================================================================
# Tests for Language Detection Rules
# ============================================================================

class TestLanguageDetectionRules:
    """Tests for language detection rules in prompt."""

    def test_95_percent_threshold(self):
        """Test 95% threshold is documented."""
        assert "95%" in LANGUAGE_VALIDATION_PROMPT

    def test_technical_terms_allowed(self):
        """Test technical terms exception is documented."""
        assert "Technical terms" in LANGUAGE_VALIDATION_PROMPT or \
               "technical terms" in LANGUAGE_VALIDATION_PROMPT
        assert "English" in LANGUAGE_VALIDATION_PROMPT

    def test_industry_standard_mentioned(self):
        """Test industry-standard exception is mentioned."""
        assert "industry-standard" in LANGUAGE_VALIDATION_PROMPT


# ============================================================================
# Tests for Issue Severity Levels
# ============================================================================

class TestIssueSeverityLevels:
    """Tests for issue severity levels in prompt."""

    def test_critical_severity(self):
        """Test critical severity is documented."""
        assert "critical" in LANGUAGE_VALIDATION_PROMPT
        assert "Wrong language entirely" in LANGUAGE_VALIDATION_PROMPT or \
               "wrong language" in LANGUAGE_VALIDATION_PROMPT.lower()

    def test_major_severity(self):
        """Test major severity is documented."""
        assert "major" in LANGUAGE_VALIDATION_PROMPT

    def test_minor_severity(self):
        """Test minor severity is documented."""
        assert "minor" in LANGUAGE_VALIDATION_PROMPT

    def test_suggestion_severity(self):
        """Test suggestion severity is documented."""
        assert "suggestion" in LANGUAGE_VALIDATION_PROMPT


# ============================================================================
# Tests for Quality Scoring Rules
# ============================================================================

class TestQualityScoringRules:
    """Tests for quality scoring rules in prompt."""

    def test_excellent_quality(self):
        """Test excellent quality is documented."""
        assert "excellent" in LANGUAGE_VALIDATION_PROMPT

    def test_good_quality(self):
        """Test good quality is documented."""
        assert "good" in LANGUAGE_VALIDATION_PROMPT

    def test_needs_improvement_quality(self):
        """Test needs_improvement quality is documented."""
        assert "needs_improvement" in LANGUAGE_VALIDATION_PROMPT

    def test_poor_quality(self):
        """Test poor quality is documented."""
        assert "poor" in LANGUAGE_VALIDATION_PROMPT


# ============================================================================
# Tests for Allowed Exceptions
# ============================================================================

class TestAllowedExceptions:
    """Tests for allowed exceptions in prompt."""

    def test_programming_language_names(self):
        """Test programming language names are allowed."""
        assert "Python" in LANGUAGE_VALIDATION_PROMPT
        assert "JavaScript" in LANGUAGE_VALIDATION_PROMPT
        assert "Go" in LANGUAGE_VALIDATION_PROMPT

    def test_framework_names(self):
        """Test framework names are allowed."""
        assert "React" in LANGUAGE_VALIDATION_PROMPT or \
               "Django" in LANGUAGE_VALIDATION_PROMPT or \
               "Kubernetes" in LANGUAGE_VALIDATION_PROMPT

    def test_protocol_names(self):
        """Test protocol names are allowed."""
        assert "HTTP" in LANGUAGE_VALIDATION_PROMPT
        assert "REST" in LANGUAGE_VALIDATION_PROMPT
        assert "GraphQL" in LANGUAGE_VALIDATION_PROMPT

    def test_file_extensions(self):
        """Test file extensions are allowed."""
        assert ".py" in LANGUAGE_VALIDATION_PROMPT
        assert ".js" in LANGUAGE_VALIDATION_PROMPT

    def test_tech_acronyms(self):
        """Test tech acronyms are allowed."""
        assert "API" in LANGUAGE_VALIDATION_PROMPT
        assert "SDK" in LANGUAGE_VALIDATION_PROMPT
        assert "CLI" in LANGUAGE_VALIDATION_PROMPT


# ============================================================================
# Tests for Self-Validation Checklist
# ============================================================================

class TestSelfValidationChecklist:
    """Tests for self-validation checklist items."""

    def test_content_elements_check(self):
        """Test content elements check is listed."""
        assert "content elements" in LANGUAGE_VALIDATION_PROMPT.lower()

    def test_specific_location_check(self):
        """Test specific location check is listed."""
        assert "location" in LANGUAGE_VALIDATION_PROMPT.lower()

    def test_suggested_fix_check(self):
        """Test suggested fix check is listed."""
        assert "suggested fix" in LANGUAGE_VALIDATION_PROMPT.lower()

    def test_severity_check(self):
        """Test severity check is listed."""
        assert "severity" in LANGUAGE_VALIDATION_PROMPT.lower()

    def test_tech_terms_check(self):
        """Test tech terms check is listed."""
        assert "technical terms" in LANGUAGE_VALIDATION_PROMPT.lower()


# ============================================================================
# Tests for Examples
# ============================================================================

class TestPromptExamples:
    """Tests for example validity in prompt."""

    def test_french_example_exists(self):
        """Test French example exists."""
        assert "French" in LANGUAGE_VALIDATION_PROMPT

    def test_spanish_example_exists(self):
        """Test Spanish example exists."""
        assert "Spanish" in LANGUAGE_VALIDATION_PROMPT

    def test_example_shows_invalid_case(self):
        """Test example shows invalid case."""
        assert '"is_valid": false' in LANGUAGE_VALIDATION_PROMPT

    def test_example_shows_valid_case(self):
        """Test example shows valid case."""
        assert '"is_valid": true' in LANGUAGE_VALIDATION_PROMPT

    def test_example_has_issues_array(self):
        """Test example has issues array."""
        assert '"issues"' in LANGUAGE_VALIDATION_PROMPT

    def test_example_has_severity(self):
        """Test example has severity."""
        assert '"severity"' in LANGUAGE_VALIDATION_PROMPT


# ============================================================================
# Tests for Output Contract
# ============================================================================

class TestOutputContract:
    """Tests for output contract specification."""

    def test_json_only_requirement(self):
        """Test JSON only requirement is stated."""
        assert "valid JSON only" in LANGUAGE_VALIDATION_PROMPT

    def test_no_markdown_requirement(self):
        """Test no markdown requirement is stated."""
        assert "no markdown" in LANGUAGE_VALIDATION_PROMPT.lower()

    def test_output_schema_defined(self):
        """Test output schema is defined."""
        assert '"is_valid"' in LANGUAGE_VALIDATION_PROMPT
        assert '"issues"' in LANGUAGE_VALIDATION_PROMPT
        assert '"overall_language_quality"' in LANGUAGE_VALIDATION_PROMPT


# ============================================================================
# Tests for Validator
# ============================================================================

class TestLanguageValidationValidator:
    """Tests for the LanguageValidationValidator class."""

    @pytest.fixture
    def validator(self):
        return LanguageValidationValidator()

    def test_valid_output_passes(self, validator):
        """Test that valid output passes validation."""
        output = {
            "is_valid": True,
            "issues": [],
            "overall_language_quality": "excellent",
            "summary": "All content properly localized."
        }

        result = validator.validate_output(output)
        assert result["is_valid"] is True, f"Issues: {result['issues']}"

    def test_valid_output_with_minor_issues(self, validator):
        """Test valid output with minor issues."""
        output = {
            "is_valid": True,
            "issues": [
                {
                    "location": "Section 1: Introduction",
                    "issue": "Word 'Introduction' could be translated",
                    "suggested_fix": "Section 1: Introducción",
                    "severity": "suggestion"
                }
            ],
            "overall_language_quality": "good"
        }

        result = validator.validate_output(output)
        assert result["is_valid"] is True, f"Issues: {result['issues']}"

    def test_missing_is_valid_fails(self, validator):
        """Test that missing is_valid fails."""
        output = {
            "issues": [],
            "overall_language_quality": "excellent"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("is_valid" in issue for issue in result["issues"])

    def test_missing_issues_fails(self, validator):
        """Test that missing issues fails."""
        output = {
            "is_valid": True,
            "overall_language_quality": "excellent"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("issues" in issue for issue in result["issues"])

    def test_missing_quality_fails(self, validator):
        """Test that missing overall_language_quality fails."""
        output = {
            "is_valid": True,
            "issues": []
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("overall_language_quality" in issue for issue in result["issues"])

    def test_invalid_quality_level_fails(self, validator):
        """Test that invalid quality level fails."""
        output = {
            "is_valid": True,
            "issues": [],
            "overall_language_quality": "perfect"  # Not a valid level
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("Invalid quality level" in issue for issue in result["issues"])

    def test_invalid_severity_fails(self, validator):
        """Test that invalid severity fails."""
        output = {
            "is_valid": False,
            "issues": [
                {
                    "location": "Title",
                    "issue": "Wrong language",
                    "suggested_fix": "Corrected title",
                    "severity": "severe"  # Not a valid severity
                }
            ],
            "overall_language_quality": "poor"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("invalid severity" in issue for issue in result["issues"])

    def test_issue_missing_location_fails(self, validator):
        """Test that issue missing location fails."""
        output = {
            "is_valid": False,
            "issues": [
                {
                    "issue": "Wrong language",
                    "suggested_fix": "Corrected text",
                    "severity": "critical"
                }
            ],
            "overall_language_quality": "poor"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("missing 'location'" in issue for issue in result["issues"])

    def test_issue_missing_suggested_fix_fails(self, validator):
        """Test that issue missing suggested_fix fails."""
        output = {
            "is_valid": False,
            "issues": [
                {
                    "location": "Title",
                    "issue": "Wrong language",
                    "severity": "critical"
                }
            ],
            "overall_language_quality": "poor"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("missing 'suggested_fix'" in issue for issue in result["issues"])


# ============================================================================
# Tests for Consistency Validation
# ============================================================================

class TestConsistencyValidation:
    """Tests for consistency validation between fields."""

    @pytest.fixture
    def validator(self):
        return LanguageValidationValidator()

    def test_is_valid_false_with_no_issues_fails(self, validator):
        """Test that is_valid=false with no issues fails."""
        output = {
            "is_valid": False,
            "issues": [],
            "overall_language_quality": "needs_improvement"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("no issues reported" in issue for issue in result["issues"])

    def test_is_valid_true_with_critical_issue_fails(self, validator):
        """Test that is_valid=true with critical issues fails."""
        output = {
            "is_valid": True,
            "issues": [
                {
                    "location": "Title",
                    "issue": "Entire title in wrong language",
                    "suggested_fix": "Titre correct",
                    "severity": "critical"
                }
            ],
            "overall_language_quality": "good"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("critical/major issues exist" in issue for issue in result["issues"])

    def test_excellent_with_issues_fails(self, validator):
        """Test that excellent quality with issues fails."""
        output = {
            "is_valid": True,
            "issues": [
                {
                    "location": "Title",
                    "issue": "Minor formatting",
                    "suggested_fix": "Fixed",
                    "severity": "minor"
                }
            ],
            "overall_language_quality": "excellent"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("'excellent' but issues" in issue for issue in result["issues"])

    def test_poor_with_no_issues_fails(self, validator):
        """Test that poor quality with no issues fails."""
        output = {
            "is_valid": False,
            "issues": [],
            "overall_language_quality": "poor"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False

    def test_critical_issue_with_excellent_quality_fails(self, validator):
        """Test that critical issue with excellent/good quality fails."""
        output = {
            "is_valid": False,
            "issues": [
                {
                    "location": "Title",
                    "issue": "Wrong language",
                    "suggested_fix": "Corrected",
                    "severity": "critical"
                }
            ],
            "overall_language_quality": "good"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is False
        assert any("critical issues exist" in issue for issue in result["issues"])


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def validator(self):
        return LanguageValidationValidator()

    def test_empty_issues_with_valid_true(self, validator):
        """Test empty issues with is_valid=true is valid."""
        output = {
            "is_valid": True,
            "issues": [],
            "overall_language_quality": "excellent"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is True

    def test_multiple_issues_with_valid_false(self, validator):
        """Test multiple issues with is_valid=false is valid."""
        output = {
            "is_valid": False,
            "issues": [
                {
                    "location": "Section 1",
                    "issue": "Wrong language",
                    "suggested_fix": "Section 1 corrigée",
                    "severity": "critical"
                },
                {
                    "location": "Section 2",
                    "issue": "Mixed content",
                    "suggested_fix": "Contenu corrigé",
                    "severity": "major"
                }
            ],
            "overall_language_quality": "poor"
        }
        result = validator.validate_output(output)
        assert result["is_valid"] is True, f"Issues: {result['issues']}"

    def test_all_severity_levels_valid(self, validator):
        """Test all severity levels are accepted."""
        for severity in ["critical", "major", "minor", "suggestion"]:
            output = {
                "is_valid": severity in ["minor", "suggestion"],
                "issues": [
                    {
                        "location": "Test location",
                        "issue": "Test issue",
                        "suggested_fix": "Test fix here",
                        "severity": severity
                    }
                ],
                "overall_language_quality": "needs_improvement" if severity in ["critical", "major"] else "good"
            }
            result = validator.validate_output(output)
            # Should not fail on severity validation
            severity_issues = [i for i in result["issues"] if "invalid severity" in i]
            assert len(severity_issues) == 0, f"Severity '{severity}' incorrectly flagged"

    def test_all_quality_levels_valid(self, validator):
        """Test all quality levels are accepted."""
        for quality in ["excellent", "good", "needs_improvement", "poor"]:
            has_issues = quality != "excellent"
            output = {
                "is_valid": quality in ["excellent", "good"],
                "issues": [
                    {
                        "location": "Test",
                        "issue": "Test",
                        "suggested_fix": "Fixed test",
                        "severity": "minor"
                    }
                ] if has_issues else [],
                "overall_language_quality": quality
            }
            result = validator.validate_output(output)
            quality_issues = [i for i in result["issues"] if "Invalid quality level" in i]
            assert len(quality_issues) == 0, f"Quality '{quality}' incorrectly flagged"
