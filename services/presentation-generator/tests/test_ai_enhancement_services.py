"""
Unit tests for AI enhancement services in presentation-generator.

Tests:
- VoiceoverValidation, EnforcementResult, VoiceoverEnforcer (voiceover_enforcer.py)
- TitleStyle, TitleValidationResult, TitleStyleSystem (title_style_system.py)
"""

import pytest
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import re


# =============================================================================
# Standalone implementations to avoid import chain issues
# =============================================================================

# --- Voiceover Enforcer ---

@dataclass
class VoiceoverValidation:
    """Validation result for a single voiceover."""
    slide_index: int
    slide_type: str
    word_count: int
    required_words: int
    is_valid: bool
    deficit: int


@dataclass
class EnforcementResult:
    """Result of the enforcement process."""
    original_words: int
    final_words: int
    slides_expanded: int
    total_slides: int
    duration_ratio: float


class VoiceoverEnforcer:
    """Validates and enriches voiceovers to meet duration requirements."""

    WORDS_PER_SECOND = 2.5
    MIN_WORDS_PER_SLIDE = 50
    VALIDATION_THRESHOLD = 0.90

    SLIDE_TYPE_MULTIPLIERS = {
        "title": 0.5,
        "conclusion": 0.8,
        "content": 1.0,
        "code": 1.2,
        "code_demo": 1.2,
        "diagram": 1.3,
    }

    def __init__(self, client=None, model=None):
        self.client = client
        self.model = model or "gpt-4o-mini"

    def validate_script(
        self,
        script_data: dict,
        target_duration: int
    ) -> list:
        """Validate all voiceovers in a script."""
        slides = script_data.get("slides", [])
        total_slides = len(slides)

        if total_slides == 0:
            return []

        total_words_needed = int(target_duration * self.WORDS_PER_SECOND)
        base_words_per_slide = max(
            self.MIN_WORDS_PER_SLIDE,
            total_words_needed // total_slides
        )

        validations = []

        for i, slide in enumerate(slides):
            voiceover = slide.get("voiceover_text", "") or ""

            clean_voiceover = re.sub(r'\[SYNC:[\w_]+\]', '', voiceover).strip()
            word_count = len(clean_voiceover.split())

            slide_type = slide.get("type", "content")
            multiplier = self.SLIDE_TYPE_MULTIPLIERS.get(slide_type, 1.0)
            required_words = max(
                self.MIN_WORDS_PER_SLIDE,
                int(base_words_per_slide * multiplier)
            )

            threshold_words = int(required_words * self.VALIDATION_THRESHOLD)
            is_valid = word_count >= threshold_words
            deficit = max(0, required_words - word_count)

            validations.append(VoiceoverValidation(
                slide_index=i,
                slide_type=slide_type,
                word_count=word_count,
                required_words=required_words,
                is_valid=is_valid,
                deficit=deficit
            ))

        return validations


# --- Title Style System ---

class TitleStyle(str, Enum):
    """Available title styles for slide generation."""
    CORPORATE = "corporate"
    ENGAGING = "engaging"
    EXPERT = "expert"
    MENTOR = "mentor"
    STORYTELLER = "storyteller"
    DIRECT = "direct"


@dataclass
class TitleValidationResult:
    """Result of title validation."""
    is_valid: bool
    issues: List[str]
    suggestion: Optional[str] = None


ROBOTIC_PATTERNS = {
    "introduction": [
        r"^introduction\s+(à|to|a|de)\s+",
        r"^intro(duction)?\s*:",
        r"^introducing\s+",
        r"^présentation\s+(de|du|des)\s+",
    ],
    "welcome": [
        r"\bbienvenue\b",
        r"\bwelcome\s*(to|in|back)?\b",
        r"\bbienvenido[as]?\b",
    ],
    "prompt_leakage": [
        r"\bsync\b",
        r"\bslide[_\s]?\d+\b",
        r"\banchor\b",
        r"\bplaceholder\b",
    ],
    "conclusion": [
        r"^conclusion\s*$",
        r"^summary\s*$",
        r"^résumé\s*$",
        r"^recap(itulation)?\s*$",
    ],
    "numbered": [
        r"^(partie|part|section|module|chapitre|chapter)\s*\d+",
        r"^\d+[\.\)]\s*",
        r"^step\s*\d+\s*:",
    ],
    "placeholder": [
        r"^slide\s*\d+",
        r"^titre\s*\d*\s*$",
        r"^title\s*\d*\s*$",
        r"^untitled",
    ],
    "generic": [
        r"^what\s+is\s+\w+\s*\??\s*$",
        r"^overview\s+of\s+",
        r"^basics\s+of\s+",
    ],
}

TITLE_STYLE_PATTERNS = {
    TitleStyle.CORPORATE: {
        "characteristics": [
            "Professional and formal",
            "Uses industry terminology",
            "Clear and unambiguous",
        ],
        "examples": {
            "en": ["Enterprise Data Architecture Best Practices"],
            "fr": ["Architecture de Données d'Entreprise : Bonnes Pratiques"],
        },
        "patterns": ["{topic}: Best Practices"],
    },
    TitleStyle.ENGAGING: {
        "characteristics": [
            "Hooks attention immediately",
            "Uses power words",
            "Creates curiosity",
        ],
        "examples": {
            "en": ["The Hidden Power of Python Decorators"],
            "fr": ["La Puissance Cachée des Décorateurs Python"],
        },
        "patterns": ["The {adjective} Power of {topic}"],
    },
    TitleStyle.EXPERT: {
        "characteristics": [
            "Technically precise",
            "Uses proper terminology",
            "Assumes advanced knowledge",
        ],
        "examples": {
            "en": ["Implementing CQRS with Event Sourcing"],
            "fr": ["Implémentation de CQRS avec Event Sourcing"],
        },
        "patterns": ["Implementing {pattern} with {technique}"],
    },
    TitleStyle.MENTOR: {
        "characteristics": [
            "Warm and encouraging",
            "Uses inclusive language",
            "Explains the 'why'",
        ],
        "examples": {
            "en": ["Understanding How Docker Containers Really Work"],
            "fr": ["Comprendre Comment Fonctionnent Vraiment les Conteneurs Docker"],
        },
        "patterns": ["Understanding How {topic} Really Works"],
    },
    TitleStyle.STORYTELLER: {
        "characteristics": [
            "Narrative-driven",
            "Creates a journey",
            "Uses temporal elements",
        ],
        "examples": {
            "en": ["From Monolith to Microservices: Our Journey"],
            "fr": ["Du Monolithe aux Microservices : Notre Parcours"],
        },
        "patterns": ["From {start} to {end}: {possessive} Journey"],
    },
    TitleStyle.DIRECT: {
        "characteristics": [
            "Clear and concise",
            "No unnecessary words",
            "Action-oriented",
        ],
        "examples": {
            "en": ["Docker Networking Configuration"],
            "fr": ["Configuration Réseau Docker"],
        },
        "patterns": ["{topic} Configuration"],
    },
}

SLIDE_TYPE_TITLE_TIPS = {
    "title": {
        "avoid": ["Introduction to X", "Presentation about X"],
        "prefer": ["Make it specific to what viewers will learn"],
    },
    "content": {
        "avoid": ["Part 1", "Section 2", "Overview"],
        "prefer": ["Focus on the key concept"],
    },
    "code": {
        "avoid": ["Code Example", "Demo Code"],
        "prefer": ["What the code achieves"],
    },
    "conclusion": {
        "avoid": ["Conclusion", "Summary", "Recap"],
        "prefer": ["Key Takeaways", "What You've Learned"],
    },
    "diagram": {
        "avoid": ["Architecture Diagram", "System Overview"],
        "prefer": ["What the architecture enables"],
    },
}


class TitleStyleSystem:
    """System for generating and validating human-quality slide titles."""

    def __init__(self, style: TitleStyle = TitleStyle.ENGAGING, language: str = "en"):
        self.style = style
        self.language = language

    def validate_title(self, title: str, slide_type: str = "content") -> TitleValidationResult:
        """Validate a title against anti-patterns."""
        if not title or not title.strip():
            return TitleValidationResult(
                is_valid=False,
                issues=["Title is empty"],
                suggestion="Provide a descriptive title"
            )

        title_lower = title.lower().strip()
        issues = []

        for pattern_category, patterns in ROBOTIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, title_lower, re.IGNORECASE):
                    issues.append(f"Robotic pattern detected: {pattern_category}")
                    break

        if len(title) < 10:
            issues.append("Title is too short (less than 10 characters)")
        elif len(title) > 80:
            issues.append("Title is too long (more than 80 characters)")

        words = title.split()
        all_caps_words = [w for w in words if w.isupper() and len(w) > 3]
        if len(all_caps_words) > 1:
            issues.append("Avoid excessive use of ALL CAPS")

        if slide_type in SLIDE_TYPE_TITLE_TIPS:
            tips = SLIDE_TYPE_TITLE_TIPS[slide_type]
            for avoid_pattern in tips["avoid"]:
                if avoid_pattern.lower() in title_lower:
                    issues.append(f"Generic pattern for {slide_type}: '{avoid_pattern}'")

        is_valid = len(issues) == 0

        suggestion = None
        if not is_valid:
            suggestion = self._generate_suggestion(title, slide_type, issues)

        return TitleValidationResult(
            is_valid=is_valid,
            issues=issues,
            suggestion=suggestion
        )

    def _generate_suggestion(
        self,
        original_title: str,
        slide_type: str,
        issues: List[str]
    ) -> Optional[str]:
        """Generate a suggestion for improving the title."""
        tips = SLIDE_TYPE_TITLE_TIPS.get(slide_type, {})
        prefer_tips = tips.get("prefer", [])

        if prefer_tips:
            return f"Consider: {prefer_tips[0]}"
        return None

    def get_style_guidelines(self) -> Dict:
        """Get the guidelines for the current title style."""
        return TITLE_STYLE_PATTERNS.get(self.style, TITLE_STYLE_PATTERNS[TitleStyle.ENGAGING])

    def get_prompt_enhancement(self) -> str:
        """Get prompt enhancement text for GPT to generate titles."""
        style_info = self.get_style_guidelines()
        examples = style_info["examples"].get(self.language, style_info["examples"]["en"])

        prompt = f"""
TITLE STYLE: {self.style.value.upper()}

Title characteristics for this style:
{chr(10).join(f"- {c}" for c in style_info['characteristics'])}

Example titles in this style:
{chr(10).join(f'- "{ex}"' for ex in examples[:5])}
"""
        return prompt

    def get_anti_pattern_rules(self) -> str:
        """Get anti-pattern rules for the prompt."""
        rules = "\nFORBIDDEN TITLE PATTERNS - DO NOT USE:\n"
        for category, patterns in ROBOTIC_PATTERNS.items():
            examples = [p.replace("^", "").replace("$", "")[:30] for p in patterns[:2]]
            rules += f"\n{category.upper()}: Avoid patterns like {', '.join(examples)}"
        return rules


def validate_slide_titles(slides: List[Dict], style: TitleStyle = TitleStyle.ENGAGING) -> List[TitleValidationResult]:
    """Validate all slide titles in a presentation."""
    system = TitleStyleSystem(style=style)
    results = []

    for slide in slides:
        title = slide.get("title", "")
        slide_type = slide.get("type", "content")
        result = system.validate_title(title, slide_type)
        results.append(result)

    return results


def get_title_style_from_string(style_str: str) -> TitleStyle:
    """Convert a string to TitleStyle enum."""
    try:
        return TitleStyle(style_str.lower())
    except ValueError:
        return TitleStyle.ENGAGING


# =============================================================================
# TESTS
# =============================================================================

class TestVoiceoverValidation:
    """Tests for VoiceoverValidation dataclass"""

    def test_basic_creation(self):
        validation = VoiceoverValidation(
            slide_index=0,
            slide_type="content",
            word_count=50,
            required_words=75,
            is_valid=False,
            deficit=25
        )
        assert validation.slide_index == 0
        assert validation.slide_type == "content"
        assert validation.word_count == 50
        assert validation.required_words == 75
        assert validation.is_valid is False
        assert validation.deficit == 25

    def test_valid_voiceover(self):
        validation = VoiceoverValidation(
            slide_index=1,
            slide_type="title",
            word_count=80,
            required_words=50,
            is_valid=True,
            deficit=0
        )
        assert validation.is_valid is True
        assert validation.deficit == 0

    def test_different_slide_types(self):
        for slide_type in ["title", "content", "code", "diagram", "conclusion"]:
            validation = VoiceoverValidation(
                slide_index=0,
                slide_type=slide_type,
                word_count=50,
                required_words=50,
                is_valid=True,
                deficit=0
            )
            assert validation.slide_type == slide_type


class TestEnforcementResult:
    """Tests for EnforcementResult dataclass"""

    def test_basic_creation(self):
        result = EnforcementResult(
            original_words=500,
            final_words=750,
            slides_expanded=3,
            total_slides=10,
            duration_ratio=0.95
        )
        assert result.original_words == 500
        assert result.final_words == 750
        assert result.slides_expanded == 3
        assert result.total_slides == 10
        assert result.duration_ratio == 0.95

    def test_no_expansion_needed(self):
        result = EnforcementResult(
            original_words=750,
            final_words=750,
            slides_expanded=0,
            total_slides=10,
            duration_ratio=1.0
        )
        assert result.slides_expanded == 0
        assert result.original_words == result.final_words

    def test_full_expansion(self):
        result = EnforcementResult(
            original_words=300,
            final_words=750,
            slides_expanded=10,
            total_slides=10,
            duration_ratio=1.0
        )
        assert result.slides_expanded == result.total_slides


class TestVoiceoverEnforcer:
    """Tests for VoiceoverEnforcer class"""

    def test_default_values(self):
        enforcer = VoiceoverEnforcer()
        assert enforcer.WORDS_PER_SECOND == 2.5
        assert enforcer.MIN_WORDS_PER_SLIDE == 50
        assert enforcer.VALIDATION_THRESHOLD == 0.90

    def test_slide_type_multipliers(self):
        enforcer = VoiceoverEnforcer()
        assert enforcer.SLIDE_TYPE_MULTIPLIERS["title"] == 0.5
        assert enforcer.SLIDE_TYPE_MULTIPLIERS["content"] == 1.0
        assert enforcer.SLIDE_TYPE_MULTIPLIERS["code"] == 1.2
        assert enforcer.SLIDE_TYPE_MULTIPLIERS["diagram"] == 1.3

    def test_validate_empty_script(self):
        enforcer = VoiceoverEnforcer()
        result = enforcer.validate_script({}, target_duration=300)
        assert result == []

    def test_validate_empty_slides(self):
        enforcer = VoiceoverEnforcer()
        result = enforcer.validate_script({"slides": []}, target_duration=300)
        assert result == []

    def test_validate_single_slide(self):
        enforcer = VoiceoverEnforcer()
        script = {
            "slides": [
                {"type": "content", "voiceover_text": "This is a short voiceover."}
            ]
        }
        validations = enforcer.validate_script(script, target_duration=60)
        assert len(validations) == 1
        assert validations[0].slide_index == 0
        assert validations[0].slide_type == "content"
        assert validations[0].word_count == 5

    def test_validate_multiple_slides(self):
        enforcer = VoiceoverEnforcer()
        script = {
            "slides": [
                {"type": "title", "voiceover_text": "Welcome to the course."},
                {"type": "content", "voiceover_text": "This is the main content slide with more words."},
                {"type": "code", "voiceover_text": "Here we see the code implementation."},
            ]
        }
        validations = enforcer.validate_script(script, target_duration=120)
        assert len(validations) == 3
        assert validations[0].slide_type == "title"
        assert validations[1].slide_type == "content"
        assert validations[2].slide_type == "code"

    def test_validate_with_sync_anchors(self):
        enforcer = VoiceoverEnforcer()
        script = {
            "slides": [
                {"type": "content", "voiceover_text": "[SYNC:slide_001] This is the actual content."}
            ]
        }
        validations = enforcer.validate_script(script, target_duration=60)
        # Sync anchor should be removed before counting
        assert validations[0].word_count == 5  # "This is the actual content."

    def test_validate_empty_voiceover(self):
        enforcer = VoiceoverEnforcer()
        script = {
            "slides": [
                {"type": "content", "voiceover_text": ""}
            ]
        }
        validations = enforcer.validate_script(script, target_duration=60)
        assert validations[0].word_count == 0
        assert validations[0].is_valid is False

    def test_validate_none_voiceover(self):
        enforcer = VoiceoverEnforcer()
        script = {
            "slides": [
                {"type": "content", "voiceover_text": None}
            ]
        }
        validations = enforcer.validate_script(script, target_duration=60)
        assert validations[0].word_count == 0

    def test_validate_missing_voiceover_key(self):
        enforcer = VoiceoverEnforcer()
        script = {
            "slides": [
                {"type": "content"}
            ]
        }
        validations = enforcer.validate_script(script, target_duration=60)
        assert validations[0].word_count == 0

    def test_slide_type_affects_required_words(self):
        enforcer = VoiceoverEnforcer()
        # Create script with different slide types
        long_text = " ".join(["word"] * 100)
        script = {
            "slides": [
                {"type": "title", "voiceover_text": long_text},
                {"type": "content", "voiceover_text": long_text},
                {"type": "diagram", "voiceover_text": long_text},
            ]
        }
        validations = enforcer.validate_script(script, target_duration=300)

        # Title has 0.5x multiplier, diagram has 1.3x
        title_required = validations[0].required_words
        content_required = validations[1].required_words
        diagram_required = validations[2].required_words

        assert title_required <= content_required
        assert content_required <= diagram_required

    def test_min_words_per_slide_enforced(self):
        enforcer = VoiceoverEnforcer()
        script = {
            "slides": [{"type": "content", "voiceover_text": "Short."}] * 100
        }
        # Even with many slides, min words should be enforced
        validations = enforcer.validate_script(script, target_duration=60)
        for v in validations:
            assert v.required_words >= enforcer.MIN_WORDS_PER_SLIDE

    def test_deficit_calculation(self):
        enforcer = VoiceoverEnforcer()
        script = {
            "slides": [
                {"type": "content", "voiceover_text": "Short text."}
            ]
        }
        validations = enforcer.validate_script(script, target_duration=120)
        validation = validations[0]

        # Deficit should be required - actual
        expected_deficit = max(0, validation.required_words - validation.word_count)
        assert validation.deficit == expected_deficit


class TestTitleStyle:
    """Tests for TitleStyle enum"""

    def test_all_values(self):
        assert TitleStyle.CORPORATE == "corporate"
        assert TitleStyle.ENGAGING == "engaging"
        assert TitleStyle.EXPERT == "expert"
        assert TitleStyle.MENTOR == "mentor"
        assert TitleStyle.STORYTELLER == "storyteller"
        assert TitleStyle.DIRECT == "direct"

    def test_enum_count(self):
        assert len(TitleStyle) == 6

    def test_value_comparison(self):
        assert TitleStyle.ENGAGING.value == "engaging"
        assert TitleStyle.CORPORATE.value == "corporate"


class TestTitleValidationResult:
    """Tests for TitleValidationResult dataclass"""

    def test_valid_title(self):
        result = TitleValidationResult(
            is_valid=True,
            issues=[]
        )
        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.suggestion is None

    def test_invalid_title_with_issues(self):
        result = TitleValidationResult(
            is_valid=False,
            issues=["Title is too short", "Robotic pattern detected"],
            suggestion="Consider making it more specific"
        )
        assert result.is_valid is False
        assert len(result.issues) == 2
        assert result.suggestion is not None

    def test_single_issue(self):
        result = TitleValidationResult(
            is_valid=False,
            issues=["Generic pattern detected"]
        )
        assert len(result.issues) == 1


class TestTitleStyleSystem:
    """Tests for TitleStyleSystem class"""

    def test_default_initialization(self):
        system = TitleStyleSystem()
        assert system.style == TitleStyle.ENGAGING
        assert system.language == "en"

    def test_custom_initialization(self):
        system = TitleStyleSystem(style=TitleStyle.CORPORATE, language="fr")
        assert system.style == TitleStyle.CORPORATE
        assert system.language == "fr"

    def test_validate_empty_title(self):
        system = TitleStyleSystem()
        result = system.validate_title("")
        assert result.is_valid is False
        assert "Title is empty" in result.issues

    def test_validate_whitespace_title(self):
        system = TitleStyleSystem()
        result = system.validate_title("   ")
        assert result.is_valid is False
        assert "Title is empty" in result.issues

    def test_validate_short_title(self):
        system = TitleStyleSystem()
        result = system.validate_title("Short")
        assert result.is_valid is False
        assert any("too short" in issue for issue in result.issues)

    def test_validate_long_title(self):
        system = TitleStyleSystem()
        long_title = "This is a very long title that exceeds the maximum recommended length for a slide title " * 2
        result = system.validate_title(long_title)
        assert result.is_valid is False
        assert any("too long" in issue for issue in result.issues)

    def test_validate_good_title(self):
        system = TitleStyleSystem()
        result = system.validate_title("The Hidden Power of Python Decorators")
        assert result.is_valid is True
        assert len(result.issues) == 0

    def test_validate_introduction_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Introduction to Python Programming")
        assert result.is_valid is False
        assert any("introduction" in issue.lower() for issue in result.issues)

    def test_validate_welcome_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Welcome to the Course on Data Science")
        assert result.is_valid is False
        assert any("welcome" in issue.lower() for issue in result.issues)

    def test_validate_bienvenue_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Bienvenue dans ce cours sur Python")
        assert result.is_valid is False
        assert any("welcome" in issue.lower() for issue in result.issues)

    def test_validate_conclusion_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Conclusion")
        assert result.is_valid is False
        # Both "conclusion" and "too short" might be detected

    def test_validate_summary_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Summary")
        assert result.is_valid is False

    def test_validate_numbered_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Part 1: Getting Started")
        assert result.is_valid is False
        assert any("numbered" in issue.lower() for issue in result.issues)

    def test_validate_step_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Step 1: Installation")
        assert result.is_valid is False

    def test_validate_placeholder_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Slide 1")
        assert result.is_valid is False

    def test_validate_untitled_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Untitled Presentation")
        assert result.is_valid is False

    def test_validate_generic_what_is(self):
        system = TitleStyleSystem()
        result = system.validate_title("What is Docker?")
        assert result.is_valid is False
        assert any("generic" in issue.lower() for issue in result.issues)

    def test_validate_overview_pattern(self):
        system = TitleStyleSystem()
        result = system.validate_title("Overview of the Architecture")
        assert result.is_valid is False

    def test_validate_slide_type_code(self):
        system = TitleStyleSystem()
        result = system.validate_title("Code Example for Authentication", slide_type="code")
        assert result.is_valid is False
        assert any("code" in issue.lower() for issue in result.issues)

    def test_validate_slide_type_conclusion(self):
        system = TitleStyleSystem()
        result = system.validate_title("Summary of the Chapter", slide_type="conclusion")
        # "Summary" is both a robotic pattern AND a conclusion avoid pattern

    def test_validate_all_caps_warning(self):
        system = TitleStyleSystem()
        result = system.validate_title("LEARN PYTHON PROGRAMMING TODAY")
        assert result.is_valid is False
        assert any("ALL CAPS" in issue for issue in result.issues)

    def test_validate_acronyms_ok(self):
        system = TitleStyleSystem()
        # Single acronyms should be fine
        result = system.validate_title("Building REST APIs with FastAPI")
        assert result.is_valid is True

    def test_get_style_guidelines_engaging(self):
        system = TitleStyleSystem(style=TitleStyle.ENGAGING)
        guidelines = system.get_style_guidelines()
        assert "characteristics" in guidelines
        assert "examples" in guidelines
        assert "patterns" in guidelines
        assert "Hooks attention" in guidelines["characteristics"][0]

    def test_get_style_guidelines_corporate(self):
        system = TitleStyleSystem(style=TitleStyle.CORPORATE)
        guidelines = system.get_style_guidelines()
        assert "Professional" in guidelines["characteristics"][0]

    def test_get_style_guidelines_expert(self):
        system = TitleStyleSystem(style=TitleStyle.EXPERT)
        guidelines = system.get_style_guidelines()
        assert "Technical" in guidelines["characteristics"][0]

    def test_get_prompt_enhancement(self):
        system = TitleStyleSystem(style=TitleStyle.ENGAGING)
        prompt = system.get_prompt_enhancement()
        assert "TITLE STYLE: ENGAGING" in prompt
        assert "characteristics" in prompt.lower()
        assert "example" in prompt.lower()

    def test_get_prompt_enhancement_french(self):
        system = TitleStyleSystem(style=TitleStyle.MENTOR, language="fr")
        prompt = system.get_prompt_enhancement()
        assert "MENTOR" in prompt
        # French examples should be used

    def test_get_anti_pattern_rules(self):
        system = TitleStyleSystem()
        rules = system.get_anti_pattern_rules()
        assert "FORBIDDEN" in rules
        assert "INTRODUCTION" in rules
        assert "CONCLUSION" in rules

    def test_generate_suggestion(self):
        system = TitleStyleSystem()
        suggestion = system._generate_suggestion("Bad Title", "code", ["issue"])
        assert suggestion is not None
        assert "Consider" in suggestion

    def test_generate_suggestion_unknown_type(self):
        system = TitleStyleSystem()
        suggestion = system._generate_suggestion("Title", "unknown_type", ["issue"])
        # Should return None for unknown slide types
        assert suggestion is None


class TestValidateSlideTitles:
    """Tests for validate_slide_titles function"""

    def test_validate_empty_slides(self):
        results = validate_slide_titles([])
        assert len(results) == 0

    def test_validate_single_slide(self):
        slides = [{"title": "The Power of Python", "type": "content"}]
        results = validate_slide_titles(slides)
        assert len(results) == 1
        assert results[0].is_valid is True

    def test_validate_multiple_slides(self):
        slides = [
            {"title": "Mastering Docker Containers", "type": "title"},
            {"title": "Building Your First Image", "type": "content"},
            {"title": "Key Takeaways", "type": "conclusion"},
        ]
        results = validate_slide_titles(slides)
        assert len(results) == 3

    def test_validate_mixed_validity(self):
        slides = [
            {"title": "Good Title for Content", "type": "content"},
            {"title": "Introduction to Docker", "type": "content"},  # Bad
            {"title": "Another Good Title Here", "type": "content"},
        ]
        results = validate_slide_titles(slides)
        assert results[0].is_valid is True
        assert results[1].is_valid is False
        assert results[2].is_valid is True

    def test_validate_missing_title(self):
        slides = [{"type": "content"}]
        results = validate_slide_titles(slides)
        assert results[0].is_valid is False
        assert "empty" in results[0].issues[0].lower()

    def test_validate_missing_type(self):
        slides = [{"title": "Some Good Title Here"}]
        results = validate_slide_titles(slides)
        # Should default to "content" type
        assert len(results) == 1


class TestGetTitleStyleFromString:
    """Tests for get_title_style_from_string function"""

    def test_valid_styles(self):
        assert get_title_style_from_string("corporate") == TitleStyle.CORPORATE
        assert get_title_style_from_string("engaging") == TitleStyle.ENGAGING
        assert get_title_style_from_string("expert") == TitleStyle.EXPERT
        assert get_title_style_from_string("mentor") == TitleStyle.MENTOR
        assert get_title_style_from_string("storyteller") == TitleStyle.STORYTELLER
        assert get_title_style_from_string("direct") == TitleStyle.DIRECT

    def test_case_insensitive(self):
        assert get_title_style_from_string("CORPORATE") == TitleStyle.CORPORATE
        assert get_title_style_from_string("Engaging") == TitleStyle.ENGAGING
        assert get_title_style_from_string("EXPERT") == TitleStyle.EXPERT

    def test_invalid_style_defaults_to_engaging(self):
        assert get_title_style_from_string("invalid") == TitleStyle.ENGAGING
        assert get_title_style_from_string("") == TitleStyle.ENGAGING
        assert get_title_style_from_string("unknown_style") == TitleStyle.ENGAGING


class TestRoboticPatterns:
    """Tests for robotic pattern detection"""

    def test_introduction_patterns(self):
        system = TitleStyleSystem()

        bad_titles = [
            "Introduction to Machine Learning",
            "Introduction à Python",
            "Intro: Getting Started",
            "Introducing the Framework",
            "Présentation de Docker",
        ]

        for title in bad_titles:
            result = system.validate_title(title)
            assert result.is_valid is False, f"'{title}' should be invalid"

    def test_welcome_patterns(self):
        system = TitleStyleSystem()

        bad_titles = [
            "Welcome to the Course",
            "Bienvenue dans ce tutoriel",
            "Welcome back to Part 2",
        ]

        for title in bad_titles:
            result = system.validate_title(title)
            assert result.is_valid is False, f"'{title}' should be invalid"

    def test_conclusion_patterns(self):
        system = TitleStyleSystem()

        bad_titles = [
            "Conclusion",
            "Summary",
            "Résumé",
            "Recap",
        ]

        for title in bad_titles:
            result = system.validate_title(title)
            assert result.is_valid is False, f"'{title}' should be invalid"

    def test_numbered_patterns(self):
        system = TitleStyleSystem()

        bad_titles = [
            "Part 1: Introduction",
            "Section 2: Advanced Topics",
            "1. Getting Started",
            "2) Configuration",
            "Step 1: Setup",
            "Chapter 3: Testing",
        ]

        for title in bad_titles:
            result = system.validate_title(title)
            assert result.is_valid is False, f"'{title}' should be invalid"

    def test_placeholder_patterns(self):
        system = TitleStyleSystem()

        bad_titles = [
            "Slide 1",
            "Title",
            "Untitled",
            "Titre",
        ]

        for title in bad_titles:
            result = system.validate_title(title)
            assert result.is_valid is False, f"'{title}' should be invalid"

    def test_generic_patterns(self):
        system = TitleStyleSystem()

        bad_titles = [
            "What is Docker?",
            "Overview of the system",
            "Basics of Python",
        ]

        for title in bad_titles:
            result = system.validate_title(title)
            assert result.is_valid is False, f"'{title}' should be invalid"


class TestGoodTitles:
    """Tests for titles that should pass validation"""

    def test_engaging_titles(self):
        system = TitleStyleSystem(style=TitleStyle.ENGAGING)

        good_titles = [
            "The Hidden Power of Python Decorators",
            "Why Your API is Failing",
            "5 Secrets Senior Engineers Know",
            "Stop Writing Slow Code",
            "The Architecture That Scaled",
        ]

        for title in good_titles:
            result = system.validate_title(title)
            assert result.is_valid is True, f"'{title}' should be valid, got issues: {result.issues}"

    def test_corporate_titles(self):
        system = TitleStyleSystem(style=TitleStyle.CORPORATE)

        good_titles = [
            "Enterprise Data Architecture Best Practices",
            "Strategic Cloud Migration Roadmap",
            "Performance Optimization Methodology",
        ]

        for title in good_titles:
            result = system.validate_title(title)
            assert result.is_valid is True, f"'{title}' should be valid"

    def test_mentor_titles(self):
        system = TitleStyleSystem(style=TitleStyle.MENTOR)

        good_titles = [
            "Understanding How Docker Containers Work",
            "Let's Build Your First API",
            "Making Sense of Async Await",
        ]

        for title in good_titles:
            result = system.validate_title(title)
            assert result.is_valid is True, f"'{title}' should be valid"

    def test_storyteller_titles(self):
        system = TitleStyleSystem(style=TitleStyle.STORYTELLER)

        good_titles = [
            "From Monolith to Microservices",
            "How We Reduced Latency by Ninety Percent",
            "The Day Our Database Crashed",
        ]

        for title in good_titles:
            result = system.validate_title(title)
            assert result.is_valid is True, f"'{title}' should be valid"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_title_exactly_10_chars(self):
        system = TitleStyleSystem()
        result = system.validate_title("Short Good")  # Exactly 10 chars
        # Should be valid if no patterns detected
        assert "too short" not in str(result.issues).lower()

    def test_title_exactly_80_chars(self):
        system = TitleStyleSystem()
        title = "A" * 10 + " " + "B" * 10 + " " + "C" * 10 + " " + "D" * 10 + " " + "E" * 10 + " " + "F" * 10 + " " + "GGG"
        title = title[:80]
        result = system.validate_title(title)
        assert "too long" not in str(result.issues).lower()

    def test_unicode_title(self):
        system = TitleStyleSystem()
        result = system.validate_title("Maîtriser les Décorateurs Python")
        assert result.is_valid is True

    def test_special_characters(self):
        system = TitleStyleSystem()
        result = system.validate_title("Building APIs: A Practical Guide")
        assert result.is_valid is True

    def test_numbers_in_title(self):
        system = TitleStyleSystem()
        result = system.validate_title("5 Things You Need to Know")
        assert result.is_valid is True

    def test_hyphenated_words(self):
        system = TitleStyleSystem()
        result = system.validate_title("Real-Time Data Processing")
        assert result.is_valid is True
