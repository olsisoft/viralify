"""
Title Style System

Ensures slide titles sound human-written rather than robotic.
Supports multiple title styles for different use cases:
- corporate: Professional, formal for enterprise training
- engaging: Dynamic, hooks attention for content creators
- expert: Technical precision for advanced audiences
- mentor: Warm, guiding for educational content
- storyteller: Narrative-driven for tutorials
- direct: Clear, no-frills for documentation-style content
"""
import re
from enum import Enum
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


class TitleStyle(str, Enum):
    """Available title styles for slide generation."""
    CORPORATE = "corporate"      # Professional, formal
    ENGAGING = "engaging"        # Dynamic, attention-grabbing
    EXPERT = "expert"            # Technical precision
    MENTOR = "mentor"            # Warm, educational
    STORYTELLER = "storyteller"  # Narrative-driven
    DIRECT = "direct"            # Clear, concise


@dataclass
class TitleValidationResult:
    """Result of title validation."""
    is_valid: bool
    issues: List[str]
    suggestion: Optional[str] = None


# Anti-patterns that indicate robotic/generic titles
ROBOTIC_PATTERNS = {
    # Generic introductions (in multiple languages)
    "introduction": [
        r"^introduction\s+(à|to|a|de)\s+",
        r"^intro(duction)?\s*:",
        r"^introducing\s+",
        r"^présentation\s+(de|du|des)\s+",
        r"^vorstellung\s+",
        r"^introducción\s+(a|de)\s+",
    ],
    # Generic conclusions
    "conclusion": [
        r"^conclusion\s*$",
        r"^conclusi[oó]n\s*$",
        r"^summary\s*$",
        r"^résumé\s*$",
        r"^recap(itulation)?\s*$",
        r"^to\s+conclude\s*$",
        r"^en\s+conclusion\s*$",
        r"^zusammenfassung\s*$",
    ],
    # Section numbering
    "numbered": [
        r"^(partie|part|section|module|chapitre|chapter)\s*\d+",
        r"^\d+[\.\)]\s*",
        r"^step\s*\d+\s*:",
        r"^étape\s*\d+\s*:",
    ],
    # Generic placeholders
    "placeholder": [
        r"^slide\s*\d+",
        r"^titre\s*\d*\s*$",
        r"^title\s*\d*\s*$",
        r"^untitled",
        r"^sans\s+titre",
    ],
    # Overly generic
    "generic": [
        r"^what\s+is\s+\w+\s*\??\s*$",
        r"^qu['\u2019]est[- ]ce\s+que?\s+",
        r"^overview\s+of\s+",
        r"^aperçu\s+(de|du|des)\s+",
        r"^basics\s+of\s+",
        r"^les\s+bases\s+(de|du|des)\s+",
        r"^fundamentals\s+of\s+",
        r"^les\s+fondamentaux\s+(de|du|des)\s+",
    ],
}

# Style-specific title patterns and examples
TITLE_STYLE_PATTERNS = {
    TitleStyle.CORPORATE: {
        "characteristics": [
            "Professional and formal",
            "Uses industry terminology",
            "Clear and unambiguous",
            "Suitable for boardroom presentations",
        ],
        "examples": {
            "en": [
                "Enterprise Data Architecture Best Practices",
                "Compliance Framework Implementation Guide",
                "Strategic Cloud Migration Roadmap",
                "Performance Optimization Methodology",
                "Risk Assessment and Mitigation Strategies",
            ],
            "fr": [
                "Architecture de Données d'Entreprise : Bonnes Pratiques",
                "Guide d'Implémentation du Cadre de Conformité",
                "Feuille de Route pour la Migration Cloud",
                "Méthodologie d'Optimisation des Performances",
                "Stratégies d'Évaluation et d'Atténuation des Risques",
            ],
        },
        "patterns": [
            "{topic}: Best Practices",
            "{topic} Implementation Guide",
            "Strategic {topic} Approach",
            "{topic} Framework Overview",
            "{topic}: Key Considerations",
        ],
    },
    TitleStyle.ENGAGING: {
        "characteristics": [
            "Hooks attention immediately",
            "Uses power words",
            "Creates curiosity",
            "Suitable for content creators and marketing",
        ],
        "examples": {
            "en": [
                "The Hidden Power of Python Decorators",
                "Why Your API is Failing (And How to Fix It)",
                "5 Kubernetes Secrets Senior Engineers Know",
                "Stop Writing Slow Code: A Performance Deep Dive",
                "The Architecture That Scaled to 1 Million Users",
            ],
            "fr": [
                "La Puissance Cachée des Décorateurs Python",
                "Pourquoi Votre API Échoue (Et Comment la Corriger)",
                "5 Secrets Kubernetes que les Seniors Connaissent",
                "Arrêtez d'Écrire du Code Lent : Plongée en Performance",
                "L'Architecture qui a Scalé à 1 Million d'Utilisateurs",
            ],
        },
        "patterns": [
            "The {adjective} Power of {topic}",
            "Why {problem} (And How to Fix It)",
            "{number} {topic} Secrets Experts Know",
            "Stop {bad_practice}: A {topic} Deep Dive",
            "The {topic} That {impressive_result}",
        ],
    },
    TitleStyle.EXPERT: {
        "characteristics": [
            "Technically precise",
            "Uses proper terminology",
            "Assumes advanced knowledge",
            "Suitable for senior developers and architects",
        ],
        "examples": {
            "en": [
                "Implementing CQRS with Event Sourcing",
                "Memory-Efficient Data Structures in Rust",
                "Consensus Algorithms: Raft vs Paxos",
                "Zero-Copy Deserialization Patterns",
                "Lock-Free Concurrent Data Structures",
            ],
            "fr": [
                "Implémentation de CQRS avec Event Sourcing",
                "Structures de Données Optimisées en Mémoire avec Rust",
                "Algorithmes de Consensus : Raft vs Paxos",
                "Patterns de Désérialisation Zero-Copy",
                "Structures de Données Concurrentes Sans Verrou",
            ],
        },
        "patterns": [
            "Implementing {pattern} with {technique}",
            "{adjective} {topic} in {language}",
            "{topic_a} vs {topic_b}: A Technical Analysis",
            "{technique} {topic} Patterns",
            "Advanced {topic} Techniques",
        ],
    },
    TitleStyle.MENTOR: {
        "characteristics": [
            "Warm and encouraging",
            "Uses inclusive language",
            "Explains the 'why'",
            "Suitable for educational platforms",
        ],
        "examples": {
            "en": [
                "Understanding How Docker Containers Really Work",
                "Let's Build Your First REST API Together",
                "Making Sense of Async/Await in Python",
                "Your Path to Mastering Kubernetes",
                "Demystifying Machine Learning Pipelines",
            ],
            "fr": [
                "Comprendre Comment Fonctionnent Vraiment les Conteneurs Docker",
                "Construisons Ensemble Votre Première API REST",
                "Donner du Sens à Async/Await en Python",
                "Votre Parcours vers la Maîtrise de Kubernetes",
                "Démystifier les Pipelines de Machine Learning",
            ],
        },
        "patterns": [
            "Understanding How {topic} Really Works",
            "Let's Build {project} Together",
            "Making Sense of {concept}",
            "Your Path to Mastering {topic}",
            "Demystifying {topic}",
        ],
    },
    TitleStyle.STORYTELLER: {
        "characteristics": [
            "Narrative-driven",
            "Creates a journey",
            "Uses temporal elements",
            "Suitable for case studies and tutorials",
        ],
        "examples": {
            "en": [
                "From Monolith to Microservices: Our Journey",
                "How We Reduced Latency by 90%",
                "The Day Our Database Crashed (And What We Learned)",
                "Building a Real-Time Analytics Platform from Scratch",
                "When Caching Goes Wrong: A Debugging Story",
            ],
            "fr": [
                "Du Monolithe aux Microservices : Notre Parcours",
                "Comment Nous Avons Réduit la Latence de 90%",
                "Le Jour Où Notre Base de Données a Crashé",
                "Construire une Plateforme d'Analytics Temps Réel",
                "Quand le Cache Pose Problème : Une Histoire de Debug",
            ],
        },
        "patterns": [
            "From {start} to {end}: {possessive} Journey",
            "How We {achieved_result}",
            "The Day {incident} (And What We Learned)",
            "Building {project} from Scratch",
            "When {topic} Goes Wrong: A {type} Story",
        ],
    },
    TitleStyle.DIRECT: {
        "characteristics": [
            "Clear and concise",
            "No unnecessary words",
            "Action-oriented",
            "Suitable for documentation and reference material",
        ],
        "examples": {
            "en": [
                "Docker Networking Configuration",
                "API Error Handling",
                "Database Query Optimization",
                "Setting Up CI/CD Pipelines",
                "Kubernetes Pod Security Policies",
            ],
            "fr": [
                "Configuration Réseau Docker",
                "Gestion des Erreurs API",
                "Optimisation des Requêtes Base de Données",
                "Mise en Place des Pipelines CI/CD",
                "Politiques de Sécurité des Pods Kubernetes",
            ],
        },
        "patterns": [
            "{topic} Configuration",
            "{topic} {action}",
            "{verb}ing {topic}",
            "Setting Up {topic}",
            "{topic} {aspect}",
        ],
    },
}

# Slide type to recommended title formats (to avoid robotic patterns)
SLIDE_TYPE_TITLE_TIPS = {
    "title": {
        "avoid": ["Introduction to X", "Presentation about X"],
        "prefer": ["Make it specific to what viewers will learn", "Use the main hook/benefit"],
    },
    "content": {
        "avoid": ["Part 1", "Section 2", "Overview"],
        "prefer": ["Focus on the key concept", "What will the learner understand?"],
    },
    "code": {
        "avoid": ["Code Example", "Demo Code"],
        "prefer": ["What the code achieves", "The technique being demonstrated"],
    },
    "conclusion": {
        "avoid": ["Conclusion", "Summary", "Recap"],
        "prefer": ["Key Takeaways", "What You've Learned", "Next Steps"],
    },
    "diagram": {
        "avoid": ["Architecture Diagram", "System Overview"],
        "prefer": ["What the architecture enables", "The flow being illustrated"],
    },
}


class TitleStyleSystem:
    """
    System for generating and validating human-quality slide titles.
    """

    def __init__(self, style: TitleStyle = TitleStyle.ENGAGING, language: str = "en"):
        """
        Initialize the title style system.

        Args:
            style: The title style to use
            language: Content language (en, fr, es, de, etc.)
        """
        self.style = style
        self.language = language

    def validate_title(self, title: str, slide_type: str = "content") -> TitleValidationResult:
        """
        Validate a title against anti-patterns.

        Args:
            title: The title to validate
            slide_type: The type of slide (title, content, code, conclusion, diagram)

        Returns:
            TitleValidationResult with validation details
        """
        if not title or not title.strip():
            return TitleValidationResult(
                is_valid=False,
                issues=["Title is empty"],
                suggestion="Provide a descriptive title"
            )

        title_lower = title.lower().strip()
        issues = []

        # Check against robotic patterns
        for pattern_category, patterns in ROBOTIC_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, title_lower, re.IGNORECASE):
                    issues.append(f"Robotic pattern detected: {pattern_category}")
                    break

        # Check length
        if len(title) < 10:
            issues.append("Title is too short (less than 10 characters)")
        elif len(title) > 80:
            issues.append("Title is too long (more than 80 characters)")

        # Check for all caps (except acronyms)
        words = title.split()
        all_caps_words = [w for w in words if w.isupper() and len(w) > 3]
        if len(all_caps_words) > 1:
            issues.append("Avoid excessive use of ALL CAPS")

        # Check for specific slide type issues
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
        """
        Get the guidelines for the current title style.

        Returns:
            Dictionary with characteristics, examples, and patterns
        """
        return TITLE_STYLE_PATTERNS.get(self.style, TITLE_STYLE_PATTERNS[TitleStyle.ENGAGING])

    def get_prompt_enhancement(self) -> str:
        """
        Get prompt enhancement text for GPT to generate titles in the selected style.

        Returns:
            String to append to the system prompt
        """
        style_info = self.get_style_guidelines()
        examples = style_info["examples"].get(self.language, style_info["examples"]["en"])

        prompt = f"""
TITLE STYLE: {self.style.value.upper()}

Title characteristics for this style:
{chr(10).join(f"- {c}" for c in style_info['characteristics'])}

Example titles in this style:
{chr(10).join(f'- "{ex}"' for ex in examples[:5])}

CRITICAL TITLE RULES:
1. NEVER use generic patterns like "Introduction to X", "Part 1", "Conclusion", "Overview of X"
2. NEVER use numbered sections like "Step 1:", "Section 2:", "Module 3"
3. NEVER use placeholder-style titles like "Title", "Slide 1", "Untitled"
4. Each title must be SPECIFIC to the content being taught
5. Titles should sound HUMAN-WRITTEN, not AI-generated
6. Slide titles should tell the learner WHAT THEY WILL UNDERSTAND, not just the topic

For title slides: Use an engaging hook that captures the main value proposition
For content slides: Focus on the key insight or skill being taught
For code slides: Describe what the code achieves or the technique shown
For conclusion slides: Use "Key Takeaways", "What You've Learned", or "Your Next Steps"
For diagram slides: Describe what the viewer will understand from the visual

BAD TITLE EXAMPLES (NEVER USE):
- "Introduction à Python" (too generic)
- "Part 2: Variables" (numbered section)
- "Conclusion" (robotic)
- "What is Docker?" (generic question format)
- "Code Example" (describes format, not content)

GOOD TITLE EXAMPLES:
- "The Hidden Power of Python Decorators" (engaging, specific)
- "Making Your Functions Remember: Memoization in Action" (descriptive, benefit-focused)
- "Your First Docker Container in 5 Minutes" (specific outcome)
- "From Chaos to Clarity: Structuring Your Codebase" (narrative, benefit-focused)
"""
        return prompt

    def get_anti_pattern_rules(self) -> str:
        """
        Get anti-pattern rules for the prompt.

        Returns:
            String describing patterns to avoid
        """
        rules = """
FORBIDDEN TITLE PATTERNS - DO NOT USE:
"""
        for category, patterns in ROBOTIC_PATTERNS.items():
            examples = [p.replace("^", "").replace("$", "").replace("\\s+", " ").replace("\\s*", " ")[:30] for p in patterns[:2]]
            rules += f"\n{category.upper()}: Avoid patterns like {', '.join(examples)}"

        return rules


def get_title_style_prompt(style: TitleStyle, language: str = "en") -> str:
    """
    Convenience function to get the full title style prompt enhancement.

    Args:
        style: The title style to use
        language: Content language

    Returns:
        Complete prompt enhancement string
    """
    system = TitleStyleSystem(style=style, language=language)
    return system.get_prompt_enhancement()


def validate_slide_titles(slides: List[Dict], style: TitleStyle = TitleStyle.ENGAGING) -> List[TitleValidationResult]:
    """
    Validate all slide titles in a presentation.

    Args:
        slides: List of slide dictionaries with 'title' and 'type' keys
        style: Title style for validation context

    Returns:
        List of validation results for each slide
    """
    system = TitleStyleSystem(style=style)
    results = []

    for slide in slides:
        title = slide.get("title", "")
        slide_type = slide.get("type", "content")
        result = system.validate_title(title, slide_type)
        results.append(result)

    return results


def get_title_style_from_string(style_str: str) -> TitleStyle:
    """
    Convert a string to TitleStyle enum.

    Args:
        style_str: String representation of the style

    Returns:
        TitleStyle enum value (defaults to ENGAGING if not found)
    """
    try:
        return TitleStyle(style_str.lower())
    except ValueError:
        return TitleStyle.ENGAGING
