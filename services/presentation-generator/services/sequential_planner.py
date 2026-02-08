"""
Sequential Planner

Orchestrates the decoupled presentation generation flow:
1. Structure Generator → slide titles, types, durations
2. Voiceover Generator → narration for each slide
3. Content Generator → content based on type (bullets, diagrams, code)
4. CodePipeline → for code slides specifically
5. Assembly → combine into final presentation script
"""

import os
import asyncio
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .generators.structure_generator import (
    StructureGenerator,
    SlideStructure,
    SlideType,
    get_structure_generator
)
from .generators.voiceover_generator import (
    VoiceoverGenerator,
    get_voiceover_generator
)
from .generators.content_generator import (
    ContentGenerator,
    SlideContent,
    get_content_generator
)
from .code_pipeline import get_code_pipeline, CodePipeline


@dataclass
class SequentialSlide:
    """Slide complet après génération séquentielle"""
    index: int
    title: str
    slide_type: str
    duration: float

    # Contenu
    voiceover_text: str
    bullet_points: List[str] = field(default_factory=list)
    diagram_description: str = ""
    diagram_type: str = ""

    # Code (si slide code)
    code: str = ""
    code_language: str = ""
    code_highlighted_lines: List[int] = field(default_factory=list)

    # Console (si exécution)
    console_input: str = ""
    console_output: str = ""


@dataclass
class SequentialPresentationScript:
    """Script de présentation généré séquentiellement"""
    topic: str
    duration: int
    slides: List[SequentialSlide] = field(default_factory=list)
    total_words: int = 0
    generation_time_ms: float = 0

    # Métadonnées
    structure_time_ms: float = 0
    voiceover_time_ms: float = 0
    content_time_ms: float = 0
    code_time_ms: float = 0


class SequentialPlanner:
    """
    Planificateur séquentiel pour la génération de présentations.

    Flow:
    1. STRUCTURE: Génère la structure du cours (titres, types, durées)
    2. VOICEOVER: Génère les voiceovers slide par slide
    3. CONTENT: Génère le contenu selon le type de slide
    4. CODE: Utilise CodePipeline pour les slides de code
    5. ASSEMBLY: Combine tout en un script final

    Avantages vs monolithique:
    - Chaque étape a un prompt spécialisé
    - Retry facile sur une étape spécifique
    - Debug plus facile (quelle étape a échoué?)
    - Meilleure qualité (LLM travaille sur un problème à la fois)
    """

    def __init__(self):
        self.structure_gen = get_structure_generator()
        self.voiceover_gen = get_voiceover_generator()
        self.content_gen = get_content_generator()
        self.code_pipeline = get_code_pipeline()

    async def generate_script(
        self,
        topic: str,
        duration: int,
        target_audience: str = "intermediate",
        practical_focus: str = "medium",
        rag_context: Optional[str] = None,
        content_language: str = "fr",
        include_code: bool = True,
        include_diagrams: bool = True,
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> SequentialPresentationScript:
        """
        Génère le script de présentation de manière séquentielle.

        Args:
            topic: Sujet du cours
            duration: Durée en minutes
            target_audience: Audience cible
            practical_focus: Focus pratique (low/medium/high)
            rag_context: Contexte RAG (optionnel)
            content_language: Langue du contenu
            include_code: Inclure des slides de code
            include_diagrams: Inclure des diagrammes
            on_progress: Callback pour signaler la progression

        Returns:
            SequentialPresentationScript complet
        """
        start_time = datetime.now()

        script = SequentialPresentationScript(
            topic=topic,
            duration=duration
        )

        def report_progress(stage: str, progress: float):
            if on_progress:
                on_progress(stage, progress)
            print(f"[SEQUENTIAL] {stage}: {progress*100:.0f}%", flush=True)

        # =====================================================================
        # ÉTAPE 1: STRUCTURE
        # =====================================================================
        report_progress("structure", 0.0)
        structure_start = datetime.now()

        structures = await self.structure_gen.generate(
            topic=topic,
            duration=duration,
            target_audience=target_audience,
            practical_focus=practical_focus,
            rag_context=rag_context,
            content_language=content_language,
            include_code=include_code,
            include_diagrams=include_diagrams
        )

        script.structure_time_ms = (datetime.now() - structure_start).total_seconds() * 1000
        report_progress("structure", 1.0)
        print(f"[SEQUENTIAL] Structure: {len(structures)} slides", flush=True)

        # =====================================================================
        # ÉTAPE 2: VOICEOVER
        # =====================================================================
        report_progress("voiceover", 0.0)
        voiceover_start = datetime.now()

        voiceovers = await self.voiceover_gen.generate_batch(
            structures=structures,
            rag_context=rag_context,
            content_language=content_language,
            topic=topic
        )

        script.voiceover_time_ms = (datetime.now() - voiceover_start).total_seconds() * 1000
        report_progress("voiceover", 1.0)
        script.total_words = sum(len(v.split()) for v in voiceovers)
        print(f"[SEQUENTIAL] Voiceovers: {script.total_words} total words", flush=True)

        # =====================================================================
        # ÉTAPE 3: CONTENT (par type)
        # =====================================================================
        report_progress("content", 0.0)
        content_start = datetime.now()

        contents: List[SlideContent] = []
        for i, (structure, voiceover) in enumerate(zip(structures, voiceovers)):
            content = await self.content_gen.generate(
                slide_structure=structure,
                voiceover_text=voiceover,
                rag_context=rag_context,
                language=content_language
            )
            contents.append(content)
            report_progress("content", (i + 1) / len(structures))

        script.content_time_ms = (datetime.now() - content_start).total_seconds() * 1000

        # =====================================================================
        # ÉTAPE 4: CODE PIPELINE (pour slides code)
        # =====================================================================
        report_progress("code", 0.0)
        code_start = datetime.now()

        code_slides = [
            (i, s, v, c)
            for i, (s, v, c) in enumerate(zip(structures, voiceovers, contents))
            if s.slide_type == SlideType.CODE
        ]

        code_results = {}
        for idx, (i, structure, voiceover, content) in enumerate(code_slides):
            try:
                result = await self.code_pipeline.process(
                    voiceover_text=voiceover,
                    concept_name=content.code_concept or structure.title,
                    preferred_language=content.code_language,
                    audience_level=target_audience,
                    content_language=content_language,
                    execute_code=True,
                    generate_voiceover=False  # On a déjà le voiceover
                )
                if result.success and result.package:
                    code_results[i] = result.package
            except Exception as e:
                print(f"[SEQUENTIAL] Code pipeline error for slide {i}: {e}", flush=True)

            report_progress("code", (idx + 1) / max(1, len(code_slides)))

        script.code_time_ms = (datetime.now() - code_start).total_seconds() * 1000

        # =====================================================================
        # ÉTAPE 5: ASSEMBLY
        # =====================================================================
        report_progress("assembly", 0.0)

        for i, (structure, voiceover, content) in enumerate(zip(structures, voiceovers, contents)):
            slide = SequentialSlide(
                index=i,
                title=structure.title,
                slide_type=structure.slide_type.value,
                duration=structure.duration,
                voiceover_text=voiceover
            )

            if content.slide_type == SlideType.CONTENT:
                slide.bullet_points = content.bullet_points

            elif content.slide_type == SlideType.DIAGRAM:
                slide.diagram_description = content.diagram_description
                slide.diagram_type = content.diagram_type

            elif content.slide_type == SlideType.CODE:
                if i in code_results:
                    package = code_results[i]
                    slide.code = package.display_code or package.generated_code.code
                    slide.code_language = package.spec.language.value
                    slide.code_highlighted_lines = package.generated_code.highlighted_lines
                    if package.console_execution:
                        slide.console_input = package.console_execution.input_shown
                        slide.console_output = package.console_execution.output_shown

            script.slides.append(slide)
            report_progress("assembly", (i + 1) / len(structures))

        script.generation_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        print(f"[SEQUENTIAL] Complete: {len(script.slides)} slides, "
              f"{script.total_words} words, {script.generation_time_ms:.0f}ms total", flush=True)

        return script

    def convert_to_presentation_script(
        self,
        sequential_script: SequentialPresentationScript,
        target_audience: str = "intermediate",
        target_career: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convertit le script séquentiel au format PresentationScript standard.

        Args:
            sequential_script: Script généré séquentiellement
            target_audience: Audience cible
            target_career: Carrière cible (optionnel)

        Returns:
            Dict compatible avec PresentationScript
        """
        slides = []

        for slide in sequential_script.slides:
            slide_dict = {
                "title": slide.title,
                "type": slide.slide_type,
                "duration": slide.duration,
                "voiceover_text": slide.voiceover_text,
            }

            if slide.bullet_points:
                slide_dict["bullet_points"] = slide.bullet_points

            if slide.diagram_description:
                slide_dict["diagram_description"] = slide.diagram_description
                slide_dict["diagram_type"] = slide.diagram_type

            if slide.code:
                slide_dict["code_blocks"] = [{
                    "code": slide.code,
                    "language": slide.code_language,
                    "highlighted_lines": slide.code_highlighted_lines
                }]
                if slide.console_output:
                    slide_dict["console"] = {
                        "input": slide.console_input,
                        "output": slide.console_output
                    }

            slides.append(slide_dict)

        return {
            "topic": sequential_script.topic,
            "duration": sequential_script.duration,
            "target_audience": target_audience,
            "target_career": target_career,
            "slides": slides,
            "total_voiceover_words": sequential_script.total_words,
            "generation_time_ms": sequential_script.generation_time_ms,
            "generation_method": "sequential"
        }


# Singleton instance
_sequential_planner: Optional[SequentialPlanner] = None


def get_sequential_planner() -> SequentialPlanner:
    """Get singleton SequentialPlanner instance."""
    global _sequential_planner
    if _sequential_planner is None:
        _sequential_planner = SequentialPlanner()
    return _sequential_planner
