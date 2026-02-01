"""
Voiceover Enforcer - Validates and expands short voiceovers

This module ensures voiceovers meet the minimum word count required
for the target video duration. If voiceovers are too short, it expands
them using a lightweight LLM call.

Architecture:
1. validate_script() - Check all voiceovers against word count requirements
2. expand_short_voiceovers() - Expand only the slides that are too short
3. Integration in presentation_planner.py after regeneration attempts

Author: Viralify Team
Version: 1.0
"""

import os
import re
import asyncio
from dataclasses import dataclass
from typing import Optional
from openai import AsyncOpenAI


@dataclass
class VoiceoverValidation:
    """Validation result for a single voiceover."""
    slide_index: int
    slide_type: str
    word_count: int
    required_words: int
    is_valid: bool
    deficit: int  # How many words short


@dataclass
class EnforcementResult:
    """Result of the enforcement process."""
    original_words: int
    final_words: int
    slides_expanded: int
    total_slides: int
    duration_ratio: float  # final_words / target_words


class VoiceoverEnforcer:
    """
    Validates and enriches voiceovers to meet duration requirements.

    Usage:
        enforcer = VoiceoverEnforcer()
        validations = enforcer.validate_script(script_data, target_duration=300)

        if any(not v.is_valid for v in validations):
            script_data = await enforcer.expand_short_voiceovers(
                script_data, validations, content_language="fr"
            )
    """

    WORDS_PER_SECOND = 2.5  # 150 words/minute speaking rate
    MIN_WORDS_PER_SLIDE = 50  # Absolute minimum for any slide (increased from 40)
    VALIDATION_THRESHOLD = 0.90  # 90% of required words = valid (increased from 75%)

    # Word requirements by slide type (multiplier of base words_per_slide)
    SLIDE_TYPE_MULTIPLIERS = {
        "title": 0.5,        # Title slides can be shorter
        "conclusion": 0.8,   # Conclusion should summarize
        "content": 1.0,      # Standard content
        "code": 1.2,         # Code needs more explanation
        "code_demo": 1.2,    # Demo needs walkthrough
        "diagram": 1.3,      # Diagrams need detailed description
    }

    def __init__(
        self,
        client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the enforcer.

        Args:
            client: OpenAI client (creates one if not provided)
            model: Model to use for expansion (default: from shared provider or gpt-4o-mini)
        """
        # Try shared LLM provider for multi-provider support (Groq, etc.)
        try:
            from shared.llm_provider import get_llm_client, get_model_name
            self.client = client or get_llm_client()
            self.model = model or get_model_name("fast")
        except ImportError:
            self.client = client or AsyncOpenAI()
            self.model = model or os.getenv("VOICEOVER_EXPANSION_MODEL", "gpt-4o-mini")

    def validate_script(
        self,
        script_data: dict,
        target_duration: int
    ) -> list[VoiceoverValidation]:
        """
        Validate all voiceovers in a script.

        Args:
            script_data: The script dictionary with slides
            target_duration: Target video duration in seconds

        Returns:
            List of VoiceoverValidation for each slide
        """
        slides = script_data.get("slides", [])
        total_slides = len(slides)

        if total_slides == 0:
            return []

        # Calculate base words per slide
        total_words_needed = int(target_duration * self.WORDS_PER_SECOND)
        base_words_per_slide = max(
            self.MIN_WORDS_PER_SLIDE,
            total_words_needed // total_slides
        )

        validations = []

        for i, slide in enumerate(slides):
            voiceover = slide.get("voiceover_text", "") or ""

            # Remove sync anchors for word count
            clean_voiceover = re.sub(r'\[SYNC:[\w_]+\]', '', voiceover).strip()
            word_count = len(clean_voiceover.split())

            # Adjust required words based on slide type
            slide_type = slide.get("type", "content")
            multiplier = self.SLIDE_TYPE_MULTIPLIERS.get(slide_type, 1.0)
            required_words = max(
                self.MIN_WORDS_PER_SLIDE,
                int(base_words_per_slide * multiplier)
            )

            # Validate with threshold
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

    async def expand_short_voiceovers(
        self,
        script_data: dict,
        validations: list[VoiceoverValidation],
        content_language: str = "fr"
    ) -> tuple[dict, EnforcementResult]:
        """
        Expand voiceovers that are too short.

        Args:
            script_data: The script dictionary
            validations: Validation results from validate_script()
            content_language: Language for expansion (fr, en, etc.)

        Returns:
            Tuple of (updated script_data, EnforcementResult)
        """
        slides = script_data.get("slides", [])
        short_slides = [v for v in validations if not v.is_valid]

        original_words = sum(v.word_count for v in validations)

        if not short_slides:
            print(f"[ENFORCER] All voiceovers valid", flush=True)
            target_words = sum(v.required_words for v in validations)
            return script_data, EnforcementResult(
                original_words=original_words,
                final_words=original_words,
                slides_expanded=0,
                total_slides=len(slides),
                duration_ratio=original_words / target_words if target_words > 0 else 1.0
            )

        print(f"[ENFORCER] {len(short_slides)}/{len(slides)} voiceovers too short, expanding...", flush=True)

        # Expand in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)

        async def expand_one(validation: VoiceoverValidation) -> tuple[int, str]:
            async with semaphore:
                slide = slides[validation.slide_index]
                expanded = await self._expand_voiceover(
                    slide,
                    validation.required_words,
                    content_language
                )
                return validation.slide_index, expanded

        # Run expansions in parallel
        tasks = [expand_one(v) for v in short_slides]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Apply expansions
        expanded_count = 0
        for result in results:
            if isinstance(result, Exception):
                print(f"[ENFORCER] Expansion error: {result}", flush=True)
                continue

            idx, expanded_voiceover = result
            if expanded_voiceover and len(expanded_voiceover.split()) > validations[idx].word_count:
                old_count = validations[idx].word_count
                new_count = len(expanded_voiceover.split())

                # Preserve sync anchor if present
                original = slides[idx].get("voiceover_text", "")
                sync_match = re.match(r'(\[SYNC:[\w_]+\])\s*', original)
                if sync_match and not expanded_voiceover.startswith("[SYNC:"):
                    expanded_voiceover = sync_match.group(1) + " " + expanded_voiceover

                slides[idx]["voiceover_text"] = expanded_voiceover
                expanded_count += 1
                print(f"[ENFORCER] Slide {idx+1} ({slides[idx].get('type', 'content')}): {old_count} -> {new_count} words", flush=True)

        script_data["slides"] = slides

        # Calculate final stats
        final_words = sum(
            len(re.sub(r'\[SYNC:[\w_]+\]', '', s.get("voiceover_text", "") or "").split())
            for s in slides
        )
        target_words = sum(v.required_words for v in validations)

        result = EnforcementResult(
            original_words=original_words,
            final_words=final_words,
            slides_expanded=expanded_count,
            total_slides=len(slides),
            duration_ratio=final_words / target_words if target_words > 0 else 1.0
        )

        print(f"[ENFORCER] Expansion complete: {original_words} -> {final_words} words ({result.duration_ratio:.0%})", flush=True)

        return script_data, result

    async def _expand_voiceover(
        self,
        slide: dict,
        target_words: int,
        language: str
    ) -> str:
        """
        Expand a single voiceover to meet word count.

        Args:
            slide: The slide dictionary
            target_words: Target word count
            language: Content language

        Returns:
            Expanded voiceover text
        """
        current_voiceover = slide.get("voiceover_text", "") or ""

        # Remove sync anchor for processing
        clean_voiceover = re.sub(r'\[SYNC:[\w_]+\]\s*', '', current_voiceover).strip()
        current_words = len(clean_voiceover.split())

        if current_words >= target_words:
            return current_voiceover

        # Build context from slide
        slide_type = slide.get("type", "content")
        title = slide.get("title", "")
        bullet_points = slide.get("bullet_points", [])
        code_blocks = slide.get("code_blocks", [])
        diagram_description = slide.get("diagram_description", "")

        context_parts = []
        if title:
            context_parts.append(f"Slide title: {title}")
        if bullet_points:
            context_parts.append(f"Bullet points: {', '.join(bullet_points[:5])}")
        if code_blocks:
            code = code_blocks[0].get("code", "")[:300]
            context_parts.append(f"Code snippet: {code}")
        if diagram_description:
            context_parts.append(f"Diagram: {diagram_description[:200]}")

        context = "\n".join(context_parts) if context_parts else "No additional context"

        # Language-specific instructions
        lang_instructions = {
            "fr": "Écris en français naturel et professionnel. Utilise des transitions comme 'Premièrement', 'Ensuite', 'Notez que', 'En résumé'.",
            "en": "Write in natural, professional English. Use transitions like 'First', 'Next', 'Note that', 'In summary'.",
            "es": "Escribe en español natural y profesional. Usa transiciones como 'Primero', 'Luego', 'Note que', 'En resumen'.",
        }
        lang_instruction = lang_instructions.get(language, lang_instructions["en"])

        prompt = f"""Expand this training video voiceover to approximately {target_words} words.

CURRENT VOICEOVER ({current_words} words):
"{clean_voiceover}"

SLIDE CONTEXT:
Type: {slide_type}
{context}

REQUIREMENTS:
1. Keep the original meaning and key points
2. Add more detailed explanations for each concept
3. Add pedagogical transitions between ideas
4. {lang_instruction}
5. Maintain a conversational, teacher-like tone
6. Do NOT add information unrelated to the slide content
7. Do NOT add greetings or conclusions unless it's a title/conclusion slide

TARGET: Approximately {target_words} words (currently {current_words}, need +{target_words - current_words} more)

OUTPUT: Return ONLY the expanded voiceover text, nothing else. No quotes, no explanations."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at expanding training video narrations. You add depth and clarity while maintaining the original message. Output only the expanded text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )

            expanded = response.choices[0].message.content.strip()

            # Remove any quotes the model might have added
            if expanded.startswith('"') and expanded.endswith('"'):
                expanded = expanded[1:-1]
            if expanded.startswith("'") and expanded.endswith("'"):
                expanded = expanded[1:-1]

            # Verify expansion actually happened
            expanded_words = len(expanded.split())
            if expanded_words > current_words:
                return expanded
            else:
                print(f"[ENFORCER] Expansion didn't increase words ({current_words} -> {expanded_words}), keeping original", flush=True)
                return current_voiceover

        except Exception as e:
            print(f"[ENFORCER] Expansion failed for slide: {e}", flush=True)
            return current_voiceover


# =============================================================================
# Convenience function for integration
# =============================================================================

async def enforce_voiceover_duration(
    script_data: dict,
    target_duration: int,
    content_language: str = "fr",
    client: Optional[AsyncOpenAI] = None
) -> tuple[dict, EnforcementResult]:
    """
    Convenience function to validate and enforce voiceover duration.

    Args:
        script_data: The script dictionary
        target_duration: Target video duration in seconds
        content_language: Language code (fr, en, es, etc.)
        client: Optional OpenAI client

    Returns:
        Tuple of (updated script_data, EnforcementResult)

    Example:
        script_data, result = await enforce_voiceover_duration(
            script_data,
            target_duration=300,
            content_language="fr"
        )
        print(f"Duration ratio: {result.duration_ratio:.0%}")
    """
    enforcer = VoiceoverEnforcer(client=client)

    # Validate
    validations = enforcer.validate_script(script_data, target_duration)

    # Log validation summary
    total_words = sum(v.word_count for v in validations)
    target_words = int(target_duration * VoiceoverEnforcer.WORDS_PER_SECOND)
    invalid_count = sum(1 for v in validations if not v.is_valid)

    print(f"[ENFORCER] Validation: {total_words}/{target_words} words, {invalid_count}/{len(validations)} slides need expansion", flush=True)

    # Expand if needed
    if invalid_count > 0:
        return await enforcer.expand_short_voiceovers(
            script_data, validations, content_language
        )
    else:
        return script_data, EnforcementResult(
            original_words=total_words,
            final_words=total_words,
            slides_expanded=0,
            total_slides=len(validations),
            duration_ratio=total_words / target_words if target_words > 0 else 1.0
        )
