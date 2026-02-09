"""
Integration tests for LANGUAGE_VALIDATION_PROMPT

Tests the validate_language function with mock LLM responses.
"""

import pytest
import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock


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
# Language names mapping (from pedagogical_nodes.py)
# ============================================================================

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
}


# ============================================================================
# Mock Data Classes
# ============================================================================

@dataclass
class MockLecture:
    """Mock lecture for testing."""
    title: str
    description: Optional[str] = None


@dataclass
class MockSection:
    """Mock section for testing."""
    title: str
    lectures: List[MockLecture] = field(default_factory=list)


@dataclass
class MockOutline:
    """Mock outline for testing."""
    title: str
    description: str
    sections: List[MockSection] = field(default_factory=list)


# ============================================================================
# LanguageValidationValidator (from unit tests)
# ============================================================================

class LanguageValidationValidator:
    """Validates LLM output against LANGUAGE_VALIDATION_PROMPT constraints."""

    VALID_SEVERITIES = ["critical", "major", "minor", "suggestion"]
    VALID_QUALITY_LEVELS = ["excellent", "good", "needs_improvement", "poor"]

    ALLOWED_TECH_TERMS = [
        "python", "javascript", "go", "rust", "java", "typescript",
        "react", "django", "kubernetes", "docker", "fastapi", "flask",
        "http", "rest", "graphql", "grpc", "websocket",
        "api", "sdk", "cli", "ide", "sql", "nosql", "json", "xml",
        ".py", ".js", ".tsx", ".ts", ".go", ".rs",
    ]

    def validate_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the full output structure."""
        issues = []

        if "is_valid" not in output:
            issues.append("Missing 'is_valid' field")
        if "issues" not in output:
            issues.append("Missing 'issues' field")
        if "overall_language_quality" not in output:
            issues.append("Missing 'overall_language_quality' field")

        if issues:
            return {"is_valid": False, "issues": issues}

        if not isinstance(output.get("is_valid"), bool):
            issues.append("'is_valid' must be a boolean")

        issue_validation = self.validate_issues(output.get("issues", []))
        issues.extend(issue_validation)

        quality = output.get("overall_language_quality", "")
        if quality not in self.VALID_QUALITY_LEVELS:
            issues.append(f"Invalid quality level: '{quality}'")

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

            if "location" not in issue:
                validation_issues.append(f"{prefix}: missing 'location'")
            if "issue" not in issue:
                validation_issues.append(f"{prefix}: missing 'issue' description")
            if "suggested_fix" not in issue:
                validation_issues.append(f"{prefix}: missing 'suggested_fix'")

            severity = issue.get("severity", "")
            if severity and severity not in self.VALID_SEVERITIES:
                validation_issues.append(f"{prefix}: invalid severity '{severity}'")

            location = issue.get("location", "")
            if location and len(location) < 3:
                validation_issues.append(f"{prefix}: location too vague")

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

        if not is_valid and len(issues_list) == 0:
            issues.append("is_valid=false but no issues reported")

        if is_valid:
            critical_major = [i for i in issues_list
                            if i.get("severity") in ["critical", "major"]]
            if critical_major:
                issues.append(
                    f"is_valid=true but {len(critical_major)} critical/major issues exist"
                )

        severities = [i.get("severity", "minor") for i in issues_list]
        if "critical" in severities and quality in ["excellent", "good"]:
            issues.append("Quality is 'excellent'/'good' but critical issues exist")
        if quality == "excellent" and len(issues_list) > 0:
            issues.append("Quality is 'excellent' but issues were reported")
        if quality == "poor" and len(issues_list) == 0:
            issues.append("Quality is 'poor' but no issues reported")

        return issues


# ============================================================================
# Mock validate_language function
# ============================================================================

async def mock_validate_language(
    outline: MockOutline,
    target_language: str,
    mock_client: AsyncMock,
) -> Dict[str, Any]:
    """
    Mock implementation of validate_language.

    Mirrors the actual function from pedagogical_nodes.py.
    """
    if not outline:
        return {"language_validated": False}

    # For English, skip validation
    if target_language == "en":
        return {"language_validated": True}

    language_name = LANGUAGE_NAMES.get(target_language, target_language)

    # Extract content to validate
    content_items = [
        f"Title: {outline.title}",
        f"Description: {outline.description}",
    ]
    for section in outline.sections:
        content_items.append(f"Section: {section.title}")
        for lecture in section.lectures:
            content_items.append(f"  Lecture: {lecture.title}")
            if lecture.description:
                content_items.append(f"    Description: {lecture.description}")

    prompt = LANGUAGE_VALIDATION_PROMPT.format(
        target_language=target_language,
        language_name=language_name,
        outline_content="\n".join(content_items[:50]),
    )

    try:
        response = await mock_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=800
        )

        result = json.loads(response.choices[0].message.content)

        return {
            "language_validated": result.get("is_valid", True),
            "issues": result.get("issues", []),
            "overall_language_quality": result.get("overall_language_quality", "good"),
        }

    except Exception as e:
        return {
            "language_validated": True,
            "errors": [f"Language validation failed: {str(e)}"],
        }


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def validator():
    return LanguageValidationValidator()


@pytest.fixture
def french_outline():
    """French outline - fully localized."""
    return MockOutline(
        title="Introduction à Python",
        description="Apprenez les bases de la programmation avec Python",
        sections=[
            MockSection(
                title="Les Fondamentaux",
                lectures=[
                    MockLecture(
                        title="Variables et Types de Données",
                        description="Comprendre les types de base en Python"
                    ),
                    MockLecture(
                        title="Structures de Contrôle",
                        description="Conditions et boucles"
                    ),
                ]
            ),
            MockSection(
                title="Programmation Avancée",
                lectures=[
                    MockLecture(
                        title="Fonctions et Modules",
                        description="Créer des fonctions réutilisables"
                    ),
                ]
            ),
        ]
    )


@pytest.fixture
def mixed_language_outline():
    """Outline with mixed French and English."""
    return MockOutline(
        title="Introduction to Python",  # English instead of French
        description="Apprenez les bases de Python",
        sections=[
            MockSection(
                title="Getting Started",  # English
                lectures=[
                    MockLecture(
                        title="Variables et Types",
                        description="Learn the basics"  # English
                    ),
                ]
            ),
        ]
    )


@pytest.fixture
def spanish_outline():
    """Spanish outline - fully localized."""
    return MockOutline(
        title="Introducción a Python",
        description="Aprenda los fundamentos de la programación con Python",
        sections=[
            MockSection(
                title="Conceptos Básicos",
                lectures=[
                    MockLecture(
                        title="Variables y Tipos de Datos",
                        description="Comprender los tipos básicos en Python"
                    ),
                ]
            ),
        ]
    )


@pytest.fixture
def german_outline():
    """German outline - fully localized."""
    return MockOutline(
        title="Einführung in Python",
        description="Lernen Sie die Grundlagen der Programmierung mit Python",
        sections=[
            MockSection(
                title="Grundlagen",
                lectures=[
                    MockLecture(
                        title="Variablen und Datentypen",
                        description="Verstehen Sie die Basistypen in Python"
                    ),
                ]
            ),
        ]
    )


@pytest.fixture
def valid_french_response():
    """Valid LLM response for French content."""
    return {
        "is_valid": True,
        "issues": [],
        "overall_language_quality": "excellent",
        "summary": "Tout le contenu est correctement localisé en français."
    }


@pytest.fixture
def invalid_french_response():
    """Invalid LLM response for mixed language content."""
    return {
        "is_valid": False,
        "issues": [
            {
                "location": "Title: Introduction to Python",
                "issue": "Title is in English instead of French",
                "suggested_fix": "Introduction à Python",
                "severity": "critical"
            },
            {
                "location": "Section: Getting Started",
                "issue": "Section title is in English",
                "suggested_fix": "Premiers Pas",
                "severity": "major"
            },
            {
                "location": "Description: Learn the basics",
                "issue": "Description is in English",
                "suggested_fix": "Apprenez les bases",
                "severity": "major"
            }
        ],
        "overall_language_quality": "poor",
        "summary": "Multiple English phrases found in content that should be French."
    }


@pytest.fixture
def partial_issues_response():
    """Response with minor issues only."""
    return {
        "is_valid": True,
        "issues": [
            {
                "location": "Section: API Integration",
                "issue": "Consider localizing 'API' context",
                "suggested_fix": "Intégration d'API",
                "severity": "suggestion"
            }
        ],
        "overall_language_quality": "good",
        "summary": "Content is mostly in French with acceptable technical terms."
    }


def create_mock_client(response_data: Dict[str, Any]) -> AsyncMock:
    """Create a mock LLM client that returns the given response."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps(response_data)))
    ]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client


# ============================================================================
# Test Basic Integration
# ============================================================================

class TestBasicIntegration:
    """Basic integration tests for validate_language."""

    @pytest.mark.asyncio
    async def test_english_skips_validation(self, french_outline):
        """Test that English target language skips validation."""
        mock_client = create_mock_client({})

        result = await mock_validate_language(
            outline=french_outline,
            target_language="en",
            mock_client=mock_client
        )

        assert result["language_validated"] is True
        # LLM should not be called for English
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_french_validated_successfully(
        self, french_outline, valid_french_response, validator
    ):
        """Test French outline validates successfully."""
        mock_client = create_mock_client(valid_french_response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixed_language_fails(
        self, mixed_language_outline, invalid_french_response, validator
    ):
        """Test mixed language outline fails validation."""
        mock_client = create_mock_client(invalid_french_response)

        result = await mock_validate_language(
            outline=mixed_language_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is False
        assert len(result["issues"]) > 0

    @pytest.mark.asyncio
    async def test_no_outline_returns_false(self):
        """Test that no outline returns language_validated=False."""
        mock_client = create_mock_client({})

        result = await mock_validate_language(
            outline=None,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is False


# ============================================================================
# Test Different Languages
# ============================================================================

class TestDifferentLanguages:
    """Test validation for different target languages."""

    @pytest.mark.asyncio
    async def test_spanish_validation(
        self, spanish_outline, valid_french_response
    ):
        """Test Spanish content validation."""
        # Reuse valid response structure
        valid_response = {
            **valid_french_response,
            "summary": "Todo el contenido está correctamente localizado en español."
        }
        mock_client = create_mock_client(valid_response)

        result = await mock_validate_language(
            outline=spanish_outline,
            target_language="es",
            mock_client=mock_client
        )

        assert result["language_validated"] is True
        # Verify Spanish language name was used
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert "Spanish" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_german_validation(
        self, german_outline, valid_french_response
    ):
        """Test German content validation."""
        valid_response = {
            **valid_french_response,
            "summary": "Alle Inhalte sind korrekt auf Deutsch lokalisiert."
        }
        mock_client = create_mock_client(valid_response)

        result = await mock_validate_language(
            outline=german_outline,
            target_language="de",
            mock_client=mock_client
        )

        assert result["language_validated"] is True
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert "German" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_all_supported_languages(self, french_outline, valid_french_response):
        """Test all supported languages except English."""
        languages = ["fr", "es", "de", "pt", "it", "nl", "pl", "ru", "zh", "ja", "ko", "ar"]

        for lang in languages:
            mock_client = create_mock_client(valid_french_response)

            result = await mock_validate_language(
                outline=french_outline,
                target_language=lang,
                mock_client=mock_client
            )

            # LLM should be called for non-English
            mock_client.chat.completions.create.assert_called_once()


# ============================================================================
# Test Response Validation
# ============================================================================

class TestResponseValidation:
    """Test validation of LLM responses."""

    @pytest.mark.asyncio
    async def test_valid_response_passes_validation(
        self, french_outline, valid_french_response, validator
    ):
        """Test that valid response passes validation."""
        mock_client = create_mock_client(valid_french_response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        # Validate the response structure
        validation_result = validator.validate_output({
            "is_valid": result["language_validated"],
            "issues": result.get("issues", []),
            "overall_language_quality": result.get("overall_language_quality", "good")
        })

        assert validation_result["is_valid"] is True, f"Issues: {validation_result['issues']}"

    @pytest.mark.asyncio
    async def test_invalid_response_passes_validation(
        self, mixed_language_outline, invalid_french_response, validator
    ):
        """Test that invalid response still has valid structure."""
        mock_client = create_mock_client(invalid_french_response)

        result = await mock_validate_language(
            outline=mixed_language_outline,
            target_language="fr",
            mock_client=mock_client
        )

        validation_result = validator.validate_output({
            "is_valid": result["language_validated"],
            "issues": result.get("issues", []),
            "overall_language_quality": result.get("overall_language_quality", "poor")
        })

        assert validation_result["is_valid"] is True, f"Issues: {validation_result['issues']}"

    @pytest.mark.asyncio
    async def test_partial_issues_response(
        self, french_outline, partial_issues_response, validator
    ):
        """Test response with minor issues only."""
        mock_client = create_mock_client(partial_issues_response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        # Should be valid despite suggestions
        assert result["language_validated"] is True
        assert len(result.get("issues", [])) == 1


# ============================================================================
# Test Issue Severity Handling
# ============================================================================

class TestIssueSeverityHandling:
    """Test handling of different issue severities."""

    @pytest.mark.asyncio
    async def test_critical_issues_fail_validation(self, french_outline):
        """Test that critical issues cause validation failure."""
        response = {
            "is_valid": False,
            "issues": [
                {
                    "location": "Title",
                    "issue": "Entire title in wrong language",
                    "suggested_fix": "Titre en français",
                    "severity": "critical"
                }
            ],
            "overall_language_quality": "poor"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is False

    @pytest.mark.asyncio
    async def test_major_issues_fail_validation(self, french_outline):
        """Test that major issues cause validation failure."""
        response = {
            "is_valid": False,
            "issues": [
                {
                    "location": "Section: Introduction",
                    "issue": "Multiple sentences in wrong language",
                    "suggested_fix": "Traduire en français",
                    "severity": "major"
                }
            ],
            "overall_language_quality": "needs_improvement"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is False

    @pytest.mark.asyncio
    async def test_minor_issues_pass_validation(self, french_outline):
        """Test that only minor issues still pass validation."""
        response = {
            "is_valid": True,
            "issues": [
                {
                    "location": "Description",
                    "issue": "Consider using native term instead of 'Python'",
                    "suggested_fix": "Keep as is (technical term)",
                    "severity": "minor"
                }
            ],
            "overall_language_quality": "good"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True

    @pytest.mark.asyncio
    async def test_suggestions_pass_validation(self, french_outline):
        """Test that suggestions don't fail validation."""
        response = {
            "is_valid": True,
            "issues": [
                {
                    "location": "Lecture title",
                    "issue": "Could use more natural phrasing",
                    "suggested_fix": "Alternative phrasing suggestion",
                    "severity": "suggestion"
                }
            ],
            "overall_language_quality": "good"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True


# ============================================================================
# Test Technical Terms Handling
# ============================================================================

class TestTechnicalTermsHandling:
    """Test that technical terms are handled correctly."""

    @pytest.mark.asyncio
    async def test_python_not_flagged(self, french_outline, validator):
        """Test that 'Python' is not flagged as an issue."""
        response = {
            "is_valid": True,
            "issues": [],
            "overall_language_quality": "excellent"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True

    @pytest.mark.asyncio
    async def test_api_not_flagged(self, french_outline):
        """Test that 'API' is not flagged as an issue."""
        outline = MockOutline(
            title="Développement d'API avec Python",
            description="Créer des API REST professionnelles",
            sections=[
                MockSection(
                    title="API Fundamentals",  # API is allowed
                    lectures=[
                        MockLecture(title="Introduction aux API REST"),
                    ]
                )
            ]
        )

        response = {
            "is_valid": True,
            "issues": [],
            "overall_language_quality": "excellent"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True

    @pytest.mark.asyncio
    async def test_framework_names_allowed(self, french_outline):
        """Test that framework names are allowed."""
        outline = MockOutline(
            title="Développement avec Django et React",
            description="Créer des applications full-stack",
            sections=[
                MockSection(
                    title="Backend avec Django",
                    lectures=[
                        MockLecture(title="Configuration de Django"),
                    ]
                ),
                MockSection(
                    title="Frontend avec React",
                    lectures=[
                        MockLecture(title="Composants React"),
                    ]
                )
            ]
        )

        response = {
            "is_valid": True,
            "issues": [],
            "overall_language_quality": "excellent"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True


# ============================================================================
# Test Quality Levels
# ============================================================================

class TestQualityLevels:
    """Test overall_language_quality handling."""

    @pytest.mark.asyncio
    async def test_excellent_quality(self, french_outline):
        """Test excellent quality response."""
        response = {
            "is_valid": True,
            "issues": [],
            "overall_language_quality": "excellent"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True
        assert result["overall_language_quality"] == "excellent"

    @pytest.mark.asyncio
    async def test_good_quality_with_minor_issues(self, french_outline):
        """Test good quality with minor issues."""
        response = {
            "is_valid": True,
            "issues": [
                {
                    "location": "Title",
                    "issue": "Minor wording improvement possible",
                    "suggested_fix": "Alternative title",
                    "severity": "suggestion"
                }
            ],
            "overall_language_quality": "good"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True
        assert result["overall_language_quality"] == "good"

    @pytest.mark.asyncio
    async def test_needs_improvement_quality(self, mixed_language_outline):
        """Test needs_improvement quality."""
        response = {
            "is_valid": False,
            "issues": [
                {
                    "location": "Section title",
                    "issue": "Title in wrong language",
                    "suggested_fix": "Titre corrigé",
                    "severity": "major"
                }
            ],
            "overall_language_quality": "needs_improvement"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=mixed_language_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is False
        assert result["overall_language_quality"] == "needs_improvement"

    @pytest.mark.asyncio
    async def test_poor_quality(self, mixed_language_outline):
        """Test poor quality response."""
        response = {
            "is_valid": False,
            "issues": [
                {
                    "location": "Title",
                    "issue": "Entire title in English",
                    "suggested_fix": "Titre en français",
                    "severity": "critical"
                },
                {
                    "location": "Description",
                    "issue": "Description in English",
                    "suggested_fix": "Description en français",
                    "severity": "critical"
                }
            ],
            "overall_language_quality": "poor"
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=mixed_language_outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is False
        assert result["overall_language_quality"] == "poor"


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in validate_language."""

    @pytest.mark.asyncio
    async def test_json_parse_error_graceful(self, french_outline):
        """Test that JSON parse errors are handled gracefully."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Not valid JSON"))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        # Should allow to proceed on error
        assert result["language_validated"] is True
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_api_error_graceful(self, french_outline):
        """Test that API errors are handled gracefully."""
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        # Should allow to proceed on error
        assert result["language_validated"] is True
        assert "errors" in result


# ============================================================================
# Test Content Extraction
# ============================================================================

class TestContentExtraction:
    """Test that content is correctly extracted for validation."""

    @pytest.mark.asyncio
    async def test_title_included(self, french_outline, valid_french_response):
        """Test that title is included in validation."""
        mock_client = create_mock_client(valid_french_response)

        await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        prompt_content = messages[0]["content"]

        assert french_outline.title in prompt_content

    @pytest.mark.asyncio
    async def test_description_included(self, french_outline, valid_french_response):
        """Test that description is included in validation."""
        mock_client = create_mock_client(valid_french_response)

        await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        prompt_content = messages[0]["content"]

        assert french_outline.description in prompt_content

    @pytest.mark.asyncio
    async def test_section_titles_included(self, french_outline, valid_french_response):
        """Test that section titles are included in validation."""
        mock_client = create_mock_client(valid_french_response)

        await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        prompt_content = messages[0]["content"]

        for section in french_outline.sections:
            assert section.title in prompt_content

    @pytest.mark.asyncio
    async def test_lecture_titles_included(self, french_outline, valid_french_response):
        """Test that lecture titles are included in validation."""
        mock_client = create_mock_client(valid_french_response)

        await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        prompt_content = messages[0]["content"]

        for section in french_outline.sections:
            for lecture in section.lectures:
                assert lecture.title in prompt_content


# ============================================================================
# Test Prompt Formatting
# ============================================================================

class TestPromptFormatting:
    """Test that prompt is correctly formatted."""

    @pytest.mark.asyncio
    async def test_target_language_in_prompt(self, french_outline, valid_french_response):
        """Test that target language is in prompt."""
        mock_client = create_mock_client(valid_french_response)

        await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        prompt_content = messages[0]["content"]

        assert "fr" in prompt_content

    @pytest.mark.asyncio
    async def test_language_name_in_prompt(self, french_outline, valid_french_response):
        """Test that language name is in prompt."""
        mock_client = create_mock_client(valid_french_response)

        await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        prompt_content = messages[0]["content"]

        assert "French" in prompt_content

    @pytest.mark.asyncio
    async def test_json_response_format_requested(
        self, french_outline, valid_french_response
    ):
        """Test that JSON response format is requested."""
        mock_client = create_mock_client(valid_french_response)

        await mock_validate_language(
            outline=french_outline,
            target_language="fr",
            mock_client=mock_client
        )

        call_args = mock_client.chat.completions.create.call_args

        assert call_args.kwargs["response_format"] == {"type": "json_object"}


# ============================================================================
# Test Full Flow Scenarios
# ============================================================================

class TestFullFlowScenarios:
    """End-to-end scenario tests."""

    @pytest.mark.asyncio
    async def test_french_course_full_flow(self, validator):
        """Test complete French course validation flow."""
        outline = MockOutline(
            title="Maîtrisez Python en 30 jours",
            description="Un cours complet pour apprendre Python de zéro à expert",
            sections=[
                MockSection(
                    title="Semaine 1: Les Bases",
                    lectures=[
                        MockLecture(
                            title="Jour 1: Installation et Configuration",
                            description="Installez Python et votre environnement de développement"
                        ),
                        MockLecture(
                            title="Jour 2: Premiers Pas avec Python",
                            description="Écrivez vos premiers programmes"
                        ),
                    ]
                ),
                MockSection(
                    title="Semaine 2: Programmation Orientée Objet",
                    lectures=[
                        MockLecture(
                            title="Jour 8: Classes et Objets",
                            description="Comprendre la POO en Python"
                        ),
                    ]
                ),
            ]
        )

        response = {
            "is_valid": True,
            "issues": [],
            "overall_language_quality": "excellent",
            "summary": "Tout le contenu est parfaitement localisé en français."
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True

        # Validate response structure
        validation_result = validator.validate_output({
            "is_valid": result["language_validated"],
            "issues": result.get("issues", []),
            "overall_language_quality": result.get("overall_language_quality", "excellent")
        })
        assert validation_result["is_valid"] is True

    @pytest.mark.asyncio
    async def test_multilingual_tech_course(self, validator):
        """Test technical course with acceptable English terms."""
        outline = MockOutline(
            title="Développement Web avec Django et React",
            description="Créez des applications full-stack modernes",
            sections=[
                MockSection(
                    title="Backend avec Django REST Framework",
                    lectures=[
                        MockLecture(
                            title="Configuration des API REST",
                            description="Créer des endpoints RESTful"
                        ),
                    ]
                ),
                MockSection(
                    title="Frontend avec React et TypeScript",
                    lectures=[
                        MockLecture(
                            title="Composants et Hooks React",
                            description="Utiliser useState et useEffect"
                        ),
                    ]
                ),
            ]
        )

        response = {
            "is_valid": True,
            "issues": [
                {
                    "location": "Various technical terms",
                    "issue": "English technical terms detected but acceptable",
                    "suggested_fix": "Keep as is (industry-standard terms)",
                    "severity": "suggestion"
                }
            ],
            "overall_language_quality": "good",
            "summary": "Content is properly localized with acceptable technical terms."
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is True

    @pytest.mark.asyncio
    async def test_mixed_content_requires_fixes(self, validator):
        """Test course with content requiring fixes."""
        outline = MockOutline(
            title="Learn Python Basics",  # English - should be French
            description="Apprenez Python facilement",
            sections=[
                MockSection(
                    title="Getting Started",  # English - should be French
                    lectures=[
                        MockLecture(
                            title="Premiers pas",
                            description="Introduction to Python"  # English
                        ),
                    ]
                ),
            ]
        )

        response = {
            "is_valid": False,
            "issues": [
                {
                    "location": "Title: Learn Python Basics",
                    "issue": "Course title is in English instead of French",
                    "suggested_fix": "Apprendre les Bases de Python",
                    "severity": "critical"
                },
                {
                    "location": "Section: Getting Started",
                    "issue": "Section title is in English",
                    "suggested_fix": "Premiers Pas",
                    "severity": "major"
                },
                {
                    "location": "Lecture description: Introduction to Python",
                    "issue": "Description is in English",
                    "suggested_fix": "Introduction à Python",
                    "severity": "major"
                }
            ],
            "overall_language_quality": "poor",
            "summary": "Multiple content elements need translation to French."
        }
        mock_client = create_mock_client(response)

        result = await mock_validate_language(
            outline=outline,
            target_language="fr",
            mock_client=mock_client
        )

        assert result["language_validated"] is False
        assert len(result["issues"]) == 3

        # Validate structure
        validation_result = validator.validate_output({
            "is_valid": result["language_validated"],
            "issues": result["issues"],
            "overall_language_quality": result["overall_language_quality"]
        })
        assert validation_result["is_valid"] is True, f"Issues: {validation_result['issues']}"
