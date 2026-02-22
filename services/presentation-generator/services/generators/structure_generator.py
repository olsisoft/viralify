"""
Structure Generator

Generates the presentation structure (titles, types, durations) WITHOUT content.
Focused prompt (~300 lines vs 1150+ monolithic prompt).
"""

import os
import json
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
from openai import AsyncOpenAI

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    _USE_SHARED_LLM = False


class SlideType(str, Enum):
    """Types de slides supportés"""
    TITLE = "title"
    CONTENT = "content"
    CODE = "code"
    DIAGRAM = "diagram"
    CONCLUSION = "conclusion"
    QUIZ = "quiz"


@dataclass
class SlideStructure:
    """Structure d'un slide sans contenu"""
    index: int
    title: str
    slide_type: SlideType
    duration: float                     # Durée en secondes
    key_points: List[str]               # Points clés à couvrir
    requires_code: bool = False
    requires_diagram: bool = False
    concept_name: Optional[str] = None  # Pour les slides code/diagram


class StructureGenerator:
    """
    Génère la structure du cours (titres, types, durées) SANS contenu.

    Avantages:
    - Prompt simplifié (~300 lignes)
    - Génération rapide
    - Validation des durées avant génération du contenu
    - Retry facile si structure inadaptée
    """

    # Distribution recommandée des types de slides
    DEFAULT_DISTRIBUTION = {
        SlideType.TITLE: 1,           # 1 slide de titre
        SlideType.CONTENT: 0.4,       # 40% du reste
        SlideType.CODE: 0.25,         # 25% du reste
        SlideType.DIAGRAM: 0.15,      # 15% du reste
        SlideType.CONCLUSION: 1,      # 1 slide de conclusion
        SlideType.QUIZ: 0.1           # 10% du reste (optionnel)
    }

    # Durée moyenne par type (secondes)
    DEFAULT_DURATIONS = {
        SlideType.TITLE: 15,
        SlideType.CONTENT: 45,
        SlideType.CODE: 60,
        SlideType.DIAGRAM: 50,
        SlideType.CONCLUSION: 30,
        SlideType.QUIZ: 40
    }

    def __init__(self, client: Optional[AsyncOpenAI] = None):
        """
        Initialize generator.

        Args:
            client: OpenAI client (optional)
        """
        if client:
            self.client = client
        elif _USE_SHARED_LLM:
            self.client = get_llm_client()
        else:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("STRUCTURE_MODEL") or (get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini")

    async def generate(
        self,
        topic: str,
        duration: int,
        target_audience: str,
        practical_focus: str,
        rag_context: Optional[str] = None,
        content_language: str = "fr",
        include_code: bool = True,
        include_diagrams: bool = True
    ) -> List[SlideStructure]:
        """
        Génère la structure du cours.

        Args:
            topic: Sujet du cours
            duration: Durée totale en minutes
            target_audience: Audience cible
            practical_focus: Focus pratique (low/medium/high)
            rag_context: Contexte RAG (optionnel)
            content_language: Langue du contenu
            include_code: Inclure des slides de code
            include_diagrams: Inclure des diagrammes

        Returns:
            Liste de SlideStructure
        """
        print(f"[STRUCTURE_GEN] Generating structure for '{topic}' ({duration}min)", flush=True)

        # Calculer le nombre de slides recommandé
        total_seconds = duration * 60
        avg_slide_duration = 45
        estimated_slides = max(5, total_seconds // avg_slide_duration)

        # Construire le prompt
        prompt = self._build_prompt(
            topic=topic,
            duration=duration,
            total_seconds=total_seconds,
            estimated_slides=estimated_slides,
            target_audience=target_audience,
            practical_focus=practical_focus,
            rag_context=rag_context,
            content_language=content_language,
            include_code=include_code,
            include_diagrams=include_diagrams
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(content_language)
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=2000
            )

            result = json.loads(response.choices[0].message.content)
            structures = self._parse_response(result)

            # Valider et ajuster les durées
            structures = self._validate_durations(structures, total_seconds)

            print(f"[STRUCTURE_GEN] Generated {len(structures)} slides", flush=True)
            return structures

        except Exception as e:
            print(f"[STRUCTURE_GEN] Error: {e}", flush=True)
            # Fallback: structure par défaut
            return self._generate_fallback_structure(
                topic=topic,
                total_seconds=total_seconds,
                include_code=include_code,
                include_diagrams=include_diagrams
            )

    def _get_system_prompt(self, language: str) -> str:
        """Prompt système pour la génération de structure."""
        if language == "fr":
            return """Tu es un expert en conception de cours pédagogiques.

Tu génères UNIQUEMENT la STRUCTURE d'un cours - pas le contenu.

RÈGLES:
1. Chaque slide a un TITRE unique et accrocheur
2. Chaque slide a des KEY_POINTS (ce qui sera couvert)
3. Les durées sont réalistes (15-90 secondes par slide)
4. La progression est logique (simple → complexe)
5. Les slides de code/diagram sont marqués explicitement

FORMAT DE SORTIE (JSON):
{
    "slides": [
        {
            "index": 0,
            "title": "Titre du slide",
            "type": "content|code|diagram|title|conclusion",
            "duration": 45,
            "key_points": ["Point 1", "Point 2"],
            "requires_code": false,
            "requires_diagram": false,
            "concept_name": null
        }
    ],
    "total_duration": 300
}"""
        else:
            return """You are an expert in educational course design.

You generate ONLY the STRUCTURE of a course - not the content.

RULES:
1. Each slide has a UNIQUE and engaging TITLE
2. Each slide has KEY_POINTS (what will be covered)
3. Durations are realistic (15-90 seconds per slide)
4. Progression is logical (simple → complex)
5. Code/diagram slides are explicitly marked

OUTPUT FORMAT (JSON):
{
    "slides": [
        {
            "index": 0,
            "title": "Slide title",
            "type": "content|code|diagram|title|conclusion",
            "duration": 45,
            "key_points": ["Point 1", "Point 2"],
            "requires_code": false,
            "requires_diagram": false,
            "concept_name": null
        }
    ],
    "total_duration": 300
}"""

    def _build_prompt(
        self,
        topic: str,
        duration: int,
        total_seconds: int,
        estimated_slides: int,
        target_audience: str,
        practical_focus: str,
        rag_context: Optional[str],
        content_language: str,
        include_code: bool,
        include_diagrams: bool
    ) -> str:
        """Construit le prompt pour la génération de structure."""

        # Ajuster la distribution selon le focus pratique
        code_ratio = 0.3 if practical_focus == "high" else 0.2 if practical_focus == "medium" else 0.1
        diagram_ratio = 0.2 if include_diagrams else 0

        prompt = f"""SUJET: {topic}
DURÉE TOTALE: {duration} minutes ({total_seconds} secondes)
SLIDES ESTIMÉS: {estimated_slides}
AUDIENCE: {target_audience}
FOCUS PRATIQUE: {practical_focus}
LANGUE: {content_language}

CONTRAINTES:
- Slide de titre (obligatoire): 15 secondes
- Slide de conclusion (obligatoire): 30 secondes
- Slides de contenu: {int((1 - code_ratio - diagram_ratio) * 100)}% du reste
"""

        if include_code:
            prompt += f"- Slides de code: {int(code_ratio * 100)}% du reste\n"
        if include_diagrams:
            prompt += f"- Slides de diagramme: {int(diagram_ratio * 100)}% du reste\n"

        if rag_context:
            prompt += f"""
CONTEXTE DOCUMENTAIRE (utilise ces informations):
{rag_context[:3000]}
"""

        prompt += """
Génère la STRUCTURE du cours avec:
1. Un slide de TITRE accrocheur
2. Des slides de CONTENU qui introduisent les concepts
3. Des slides de CODE pour les démonstrations pratiques (si applicable)
4. Des slides de DIAGRAMME pour visualiser l'architecture (si applicable)
5. Un slide de CONCLUSION récapitulatif

IMPORTANT:
- Titres HUMAINS et ENGAGEANTS (pas "Introduction à X" ou "Conclusion")
- Chaque slide a 2-4 key_points
- La somme des durées = durée totale
- Progression logique du simple au complexe
"""

        return prompt

    def _parse_response(self, result: dict) -> List[SlideStructure]:
        """Parse la réponse LLM en SlideStructure."""
        structures = []

        for slide in result.get("slides", []):
            try:
                slide_type = SlideType(slide.get("type", "content"))
            except ValueError:
                slide_type = SlideType.CONTENT

            structures.append(SlideStructure(
                index=slide.get("index", len(structures)),
                title=slide.get("title", f"Slide {len(structures) + 1}"),
                slide_type=slide_type,
                duration=slide.get("duration", 45),
                key_points=slide.get("key_points", []),
                requires_code=slide.get("requires_code", slide_type == SlideType.CODE),
                requires_diagram=slide.get("requires_diagram", slide_type == SlideType.DIAGRAM),
                concept_name=slide.get("concept_name")
            ))

        return structures

    def _validate_durations(
        self,
        structures: List[SlideStructure],
        total_seconds: int
    ) -> List[SlideStructure]:
        """Valide et ajuste les durées pour atteindre la durée totale."""
        current_total = sum(s.duration for s in structures)

        if current_total == 0:
            # Répartir équitablement
            avg_duration = total_seconds / len(structures)
            for s in structures:
                s.duration = avg_duration
            return structures

        # Ajuster proportionnellement
        ratio = total_seconds / current_total
        for s in structures:
            s.duration = max(15, min(120, s.duration * ratio))

        return structures

    def _generate_fallback_structure(
        self,
        topic: str,
        total_seconds: int,
        include_code: bool,
        include_diagrams: bool
    ) -> List[SlideStructure]:
        """Génère une structure par défaut en cas d'erreur."""
        structures = []

        # Titre
        structures.append(SlideStructure(
            index=0,
            title=topic,
            slide_type=SlideType.TITLE,
            duration=15,
            key_points=["Introduction au sujet"]
        ))

        # Contenu principal
        remaining = total_seconds - 45  # 15 (titre) + 30 (conclusion)
        content_slides = max(3, remaining // 50)

        for i in range(content_slides):
            slide_type = SlideType.CONTENT
            if include_code and i == content_slides // 2:
                slide_type = SlideType.CODE
            elif include_diagrams and i == content_slides - 1:
                slide_type = SlideType.DIAGRAM

            structures.append(SlideStructure(
                index=i + 1,
                title=f"Concept {i + 1}",
                slide_type=slide_type,
                duration=remaining / content_slides,
                key_points=[f"Point clé {i + 1}"],
                requires_code=(slide_type == SlideType.CODE),
                requires_diagram=(slide_type == SlideType.DIAGRAM)
            ))

        # Conclusion
        structures.append(SlideStructure(
            index=len(structures),
            title="Récapitulatif",
            slide_type=SlideType.CONCLUSION,
            duration=30,
            key_points=["Résumé des points clés"]
        ))

        return structures


# Singleton instance
_structure_generator: Optional[StructureGenerator] = None


def get_structure_generator() -> StructureGenerator:
    """Get singleton StructureGenerator instance."""
    global _structure_generator
    if _structure_generator is None:
        _structure_generator = StructureGenerator()
    return _structure_generator
