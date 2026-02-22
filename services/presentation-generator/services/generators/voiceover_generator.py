"""
Voiceover Generator

Generates voiceover text for each slide based on its structure.
Specialized prompt for pedagogical narration.
"""

import os
import json
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    _USE_SHARED_LLM = False

from .structure_generator import SlideStructure, SlideType


class VoiceoverGenerator:
    """
    Génère le voiceover pour chaque slide.

    Avantages:
    - Prompt spécialisé pour la narration pédagogique
    - Génération parallélisable (batch)
    - Contrôle précis du nombre de mots
    - Contexte des voiceovers précédents pour la cohérence
    """

    # Mots par minute pour le speech
    WORDS_PER_MINUTE = 150

    # Multiplicateurs par type de slide
    TYPE_WORD_MULTIPLIERS = {
        SlideType.TITLE: 0.5,       # Titres courts
        SlideType.CONTENT: 1.0,     # Standard
        SlideType.CODE: 1.2,        # Plus d'explication pour le code
        SlideType.DIAGRAM: 1.3,     # Plus d'explication pour les diagrammes
        SlideType.CONCLUSION: 0.8,  # Résumé concis
        SlideType.QUIZ: 0.7         # Questions courtes
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
        self.model = os.getenv("VOICEOVER_MODEL") or (get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini")
        self._semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

    async def generate(
        self,
        slide_structure: SlideStructure,
        previous_voiceovers: List[str],
        rag_context: Optional[str],
        content_language: str,
        topic: str
    ) -> str:
        """
        Génère le voiceover pour UN slide.

        Args:
            slide_structure: Structure du slide
            previous_voiceovers: Voiceovers des slides précédents (pour contexte)
            rag_context: Contexte RAG (optionnel)
            content_language: Langue du contenu
            topic: Sujet global du cours

        Returns:
            Texte du voiceover
        """
        # Calculer le nombre de mots cible
        words_target = self._calculate_words_target(slide_structure)

        prompt = self._build_prompt(
            structure=slide_structure,
            previous_voiceovers=previous_voiceovers,
            rag_context=rag_context,
            content_language=content_language,
            topic=topic,
            words_target=words_target
        )

        try:
            async with self._semaphore:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt(content_language)
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )

            voiceover = response.choices[0].message.content.strip()

            # Nettoyer les guillemets si présents
            if voiceover.startswith('"') and voiceover.endswith('"'):
                voiceover = voiceover[1:-1]

            return voiceover

        except Exception as e:
            print(f"[VOICEOVER_GEN] Error for slide {slide_structure.index}: {e}", flush=True)
            # Fallback: description basique
            return self._generate_fallback(slide_structure, content_language)

    async def generate_batch(
        self,
        structures: List[SlideStructure],
        rag_context: Optional[str],
        content_language: str,
        topic: str
    ) -> List[str]:
        """
        Génère les voiceovers pour tous les slides en parallèle.

        Args:
            structures: Liste des structures de slides
            rag_context: Contexte RAG (optionnel)
            content_language: Langue du contenu
            topic: Sujet global du cours

        Returns:
            Liste de voiceovers (dans l'ordre des structures)
        """
        print(f"[VOICEOVER_GEN] Generating {len(structures)} voiceovers in batch", flush=True)

        voiceovers = []

        # Générer séquentiellement pour maintenir le contexte
        for i, structure in enumerate(structures):
            voiceover = await self.generate(
                slide_structure=structure,
                previous_voiceovers=voiceovers[-3:] if voiceovers else [],  # 3 derniers
                rag_context=rag_context,
                content_language=content_language,
                topic=topic
            )
            voiceovers.append(voiceover)
            print(f"[VOICEOVER_GEN] Generated voiceover {i+1}/{len(structures)}: "
                  f"{len(voiceover.split())} words", flush=True)

        return voiceovers

    def _calculate_words_target(self, structure: SlideStructure) -> int:
        """Calcule le nombre de mots cible pour le voiceover."""
        # Durée en minutes
        duration_minutes = structure.duration / 60

        # Mots de base
        base_words = int(duration_minutes * self.WORDS_PER_MINUTE)

        # Appliquer le multiplicateur
        multiplier = self.TYPE_WORD_MULTIPLIERS.get(structure.slide_type, 1.0)

        return max(30, int(base_words * multiplier))

    def _get_system_prompt(self, language: str) -> str:
        """Prompt système pour la génération de voiceover."""
        if language == "fr":
            return """Tu es un narrateur pédagogique expert.

Tu génères des VOICEOVERS pour des cours vidéo éducatifs.

STYLE:
- Conversationnel et engageant
- Pas de jargon inutile
- Phrases courtes et claires
- Transitions naturelles

RÈGLES:
1. JAMAIS de marqueurs de section comme "Introduction:", "Conclusion:"
2. JAMAIS de listes à puces
3. Parle directement à l'apprenant ("vous allez découvrir...")
4. Utilise des transitions naturelles ("Maintenant que...", "Passons à...")
5. Adapte le ton au type de contenu

RETOURNE UNIQUEMENT le texte du voiceover, sans guillemets ni formatage."""
        else:
            return """You are an expert pedagogical narrator.

You generate VOICEOVERS for educational video courses.

STYLE:
- Conversational and engaging
- No unnecessary jargon
- Short and clear sentences
- Natural transitions

RULES:
1. NEVER use section markers like "Introduction:", "Conclusion:"
2. NEVER use bullet points
3. Speak directly to the learner ("you will discover...")
4. Use natural transitions ("Now that...", "Let's move on to...")
5. Adapt the tone to the content type

RETURN ONLY the voiceover text, without quotes or formatting."""

    def _build_prompt(
        self,
        structure: SlideStructure,
        previous_voiceovers: List[str],
        rag_context: Optional[str],
        content_language: str,
        topic: str,
        words_target: int
    ) -> str:
        """Construit le prompt pour la génération de voiceover."""

        # Type-specific instructions
        type_instructions = {
            SlideType.TITLE: "Accroche qui donne envie d'apprendre. Présente le sujet de manière engageante.",
            SlideType.CONTENT: "Explication claire et pédagogique. Guide l'apprenant à travers les concepts.",
            SlideType.CODE: "Prépare l'apprenant au code qui sera montré. Explique ce qu'on va implémenter et pourquoi.",
            SlideType.DIAGRAM: "Décris le diagramme de manière vivante. Guide le regard de l'apprenant à travers les composants.",
            SlideType.CONCLUSION: "Récapitule les points clés appris. Donne envie d'aller plus loin.",
            SlideType.QUIZ: "Introduis la question de manière engageante. Encourage la réflexion."
        }

        prompt = f"""SLIDE #{structure.index + 1}: {structure.title}
TYPE: {structure.slide_type.value}
DURÉE: {structure.duration} secondes
MOTS CIBLES: ~{words_target} mots

POINTS À COUVRIR:
{chr(10).join(f'- {point}' for point in structure.key_points)}

INSTRUCTION SPÉCIFIQUE:
{type_instructions.get(structure.slide_type, "Explication pédagogique claire.")}

SUJET GLOBAL: {topic}
"""

        if previous_voiceovers:
            # Résumer les voiceovers précédents pour le contexte
            context = " [...] ".join([v[:100] + "..." for v in previous_voiceovers[-2:]])
            prompt += f"""
CONTEXTE (ce qui a été dit avant):
{context}

Assure une transition naturelle avec ce qui précède.
"""

        if rag_context:
            prompt += f"""
CONTENU SOURCE (utilise ces informations):
{rag_context[:2000]}
"""

        prompt += f"""
Génère le voiceover ({words_target} mots environ) en {content_language}:"""

        return prompt

    def _generate_fallback(self, structure: SlideStructure, language: str) -> str:
        """Génère un voiceover de fallback basique."""
        if language == "fr":
            templates = {
                SlideType.TITLE: f"Bienvenue dans ce cours sur {structure.title}.",
                SlideType.CONTENT: f"Parlons maintenant de {structure.title}. {' '.join(structure.key_points)}",
                SlideType.CODE: f"Voyons comment implémenter {structure.concept_name or structure.title}.",
                SlideType.DIAGRAM: f"Ce diagramme illustre {structure.title}.",
                SlideType.CONCLUSION: "Récapitulons ce que nous avons appris dans ce cours.",
                SlideType.QUIZ: "Testons vos connaissances avec une question."
            }
        else:
            templates = {
                SlideType.TITLE: f"Welcome to this course on {structure.title}.",
                SlideType.CONTENT: f"Let's now discuss {structure.title}. {' '.join(structure.key_points)}",
                SlideType.CODE: f"Let's see how to implement {structure.concept_name or structure.title}.",
                SlideType.DIAGRAM: f"This diagram illustrates {structure.title}.",
                SlideType.CONCLUSION: "Let's recap what we've learned in this course.",
                SlideType.QUIZ: "Let's test your knowledge with a question."
            }

        return templates.get(structure.slide_type, f"Let's explore {structure.title}.")


# Singleton instance
_voiceover_generator: Optional[VoiceoverGenerator] = None


def get_voiceover_generator() -> VoiceoverGenerator:
    """Get singleton VoiceoverGenerator instance."""
    global _voiceover_generator
    if _voiceover_generator is None:
        _voiceover_generator = VoiceoverGenerator()
    return _voiceover_generator
