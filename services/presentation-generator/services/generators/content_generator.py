"""
Content Generator

Generates content for each slide based on its type:
- content: bullet points
- diagram: diagram description
- code: routes to CodePipeline
"""

import os
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from openai import AsyncOpenAI

from .structure_generator import SlideStructure, SlideType


@dataclass
class SlideContent:
    """Contenu généré pour un slide"""
    slide_index: int
    slide_type: SlideType

    # Pour slides content
    bullet_points: List[str] = field(default_factory=list)

    # Pour slides diagram
    diagram_description: str = ""
    diagram_type: str = ""  # architecture, flowchart, sequence, etc.

    # Pour slides code (référence au CodePipeline)
    code_concept: str = ""
    code_language: str = ""

    # Pour slides quiz
    quiz_question: str = ""
    quiz_options: List[str] = field(default_factory=list)
    quiz_answer: int = 0


class ContentGenerator:
    """
    Génère le contenu selon le type de slide.

    Routes:
    - content → génère bullet_points
    - diagram → génère diagram_description
    - code → délègue au CodePipeline (via référence)
    - title/conclusion → contenu minimal
    """

    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """
        Initialize generator.

        Args:
            client: OpenAI client (optional)
        """
        self.client = client or AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("CONTENT_MODEL", "gpt-4o-mini")

    async def generate(
        self,
        slide_structure: SlideStructure,
        voiceover_text: str,
        rag_context: Optional[str],
        language: str
    ) -> SlideContent:
        """
        Génère le contenu pour UN slide.

        Args:
            slide_structure: Structure du slide
            voiceover_text: Voiceover généré pour ce slide
            rag_context: Contexte RAG (optionnel)
            language: Langue du contenu

        Returns:
            SlideContent avec le contenu approprié
        """
        content = SlideContent(
            slide_index=slide_structure.index,
            slide_type=slide_structure.slide_type
        )

        if slide_structure.slide_type == SlideType.CONTENT:
            content.bullet_points = await self._generate_bullet_points(
                structure=slide_structure,
                voiceover=voiceover_text,
                rag_context=rag_context,
                language=language
            )

        elif slide_structure.slide_type == SlideType.DIAGRAM:
            description, diagram_type = await self._generate_diagram_description(
                structure=slide_structure,
                voiceover=voiceover_text,
                rag_context=rag_context,
                language=language
            )
            content.diagram_description = description
            content.diagram_type = diagram_type

        elif slide_structure.slide_type == SlideType.CODE:
            # Pour les slides code, on prépare la référence pour le CodePipeline
            content.code_concept = slide_structure.concept_name or slide_structure.title
            content.code_language = self._detect_language(voiceover_text)

        elif slide_structure.slide_type == SlideType.QUIZ:
            question, options, answer = await self._generate_quiz(
                structure=slide_structure,
                voiceover=voiceover_text,
                language=language
            )
            content.quiz_question = question
            content.quiz_options = options
            content.quiz_answer = answer

        return content

    async def _generate_bullet_points(
        self,
        structure: SlideStructure,
        voiceover: str,
        rag_context: Optional[str],
        language: str
    ) -> List[str]:
        """Génère les bullet points pour un slide de contenu."""

        prompt = f"""SLIDE: {structure.title}

VOICEOVER (ce qui sera dit):
{voiceover}

KEY POINTS À COUVRIR:
{chr(10).join(f'- {p}' for p in structure.key_points)}

"""
        if rag_context:
            prompt += f"""CONTENU SOURCE:
{rag_context[:1500]}

"""
        prompt += f"""Génère 3-5 bullet points COURTS pour ce slide.

RÈGLES:
- Maximum 10 mots par bullet point
- Pas de phrases complètes
- Mots-clés ou concepts importants
- En {language}

Retourne un JSON:
{{"bullet_points": ["Point 1", "Point 2", "Point 3"]}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu génères des bullet points concis pour des slides de présentation."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
                max_tokens=300
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("bullet_points", structure.key_points)

        except Exception as e:
            print(f"[CONTENT_GEN] Bullet points error: {e}", flush=True)
            return structure.key_points

    async def _generate_diagram_description(
        self,
        structure: SlideStructure,
        voiceover: str,
        rag_context: Optional[str],
        language: str
    ) -> tuple:
        """Génère la description du diagramme."""

        prompt = f"""SLIDE: {structure.title}

VOICEOVER (ce qui sera dit):
{voiceover}

KEY POINTS:
{chr(10).join(f'- {p}' for p in structure.key_points)}

"""
        if rag_context:
            prompt += f"""CONTENU SOURCE:
{rag_context[:1500]}

"""
        prompt += """Génère une description DÉTAILLÉE pour le diagramme.

La description sera utilisée par un générateur de diagrammes (Mermaid ou Diagrams).

Retourne un JSON:
{
    "diagram_type": "architecture|flowchart|sequence|class|entity",
    "description": "Description complète du diagramme avec composants et relations"
}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu génères des descriptions de diagrammes techniques pour des outils de génération automatique."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=500
            )

            result = json.loads(response.choices[0].message.content)
            return (
                result.get("description", structure.title),
                result.get("diagram_type", "architecture")
            )

        except Exception as e:
            print(f"[CONTENT_GEN] Diagram description error: {e}", flush=True)
            return (structure.title, "architecture")

    async def _generate_quiz(
        self,
        structure: SlideStructure,
        voiceover: str,
        language: str
    ) -> tuple:
        """Génère une question quiz."""

        prompt = f"""SLIDE: {structure.title}

VOICEOVER (contenu couvert):
{voiceover}

Génère une question de quiz style Udemy pour tester la compréhension.

Retourne un JSON:
{{
    "question": "La question...",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": 0
}}

En {language}."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu génères des questions quiz pédagogiques de type QCM."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.6,
                max_tokens=300
            )

            result = json.loads(response.choices[0].message.content)
            return (
                result.get("question", "Question non disponible"),
                result.get("options", ["A", "B", "C", "D"]),
                result.get("correct_answer", 0)
            )

        except Exception as e:
            print(f"[CONTENT_GEN] Quiz error: {e}", flush=True)
            return ("Question non disponible", ["A", "B", "C", "D"], 0)

    def _detect_language(self, text: str) -> str:
        """Détecte le langage de programmation mentionné dans le texte."""
        text_lower = text.lower()

        language_patterns = {
            "python": ["python", "pip", "django", "flask", "pandas"],
            "java": ["java", "spring", "maven", "gradle", "jvm"],
            "javascript": ["javascript", "js", "node", "npm", "react", "vue"],
            "typescript": ["typescript", "ts"],
            "go": ["golang", " go ", "goroutine"],
            "rust": ["rust", "cargo"],
            "sql": ["sql", "query", "select", "database"]
        }

        for lang, patterns in language_patterns.items():
            if any(p in text_lower for p in patterns):
                return lang

        return "python"  # Default


# Singleton instance
_content_generator: Optional[ContentGenerator] = None


def get_content_generator() -> ContentGenerator:
    """Get singleton ContentGenerator instance."""
    global _content_generator
    if _content_generator is None:
        _content_generator = ContentGenerator()
    return _content_generator
