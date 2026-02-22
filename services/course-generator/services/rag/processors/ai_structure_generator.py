"""
AI-Powered Structure Generator

Uses LLM to generate document structure when explicit headings are not available.
Particularly useful for YouTube transcripts and unstructured documents.
"""

import os
from typing import Optional

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    _USE_SHARED_LLM = False

from ..prompts.structure_prompts import StructureExtractionPromptBuilder
from .structure_extractor import DocumentStructure, HeadingInfo


class AIStructureGenerator:
    """
    Generate document structure using AI when explicit structure is unavailable.

    Uses the StructureExtractionPromptBuilder for well-structured prompts
    that guide the LLM to identify logical sections.

    Usage:
        generator = AIStructureGenerator(openai_client)
        structure = await generator.generate(document, is_youtube=True)
    """

    # Maximum characters to analyze
    MAX_CONTENT_CHARS = 4000

    def __init__(
        self,
        openai_client=None,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """
        Initialize the AI structure generator.

        Args:
            openai_client: AsyncOpenAI client (loads from env if not provided)
            model: Model to use for generation
            temperature: Sampling temperature
        """
        if _USE_SHARED_LLM:
            self.client = openai_client or get_llm_client()
            self.model = model or get_model_name("fast")
        else:
            self.client = openai_client
            self.model = model or "gpt-4o-mini"
        self.temperature = temperature

    async def _ensure_client(self):
        """Ensure OpenAI client is initialized."""
        if self.client is None:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=60.0,
            )

    async def generate(
        self,
        document,  # Document object
        is_youtube: bool = False,
        content_language: str = "en",
    ) -> Optional[DocumentStructure]:
        """
        Generate document structure using AI.

        Args:
            document: Document object with raw_content
            is_youtube: Whether this is a YouTube transcript
            content_language: Language of the content

        Returns:
            DocumentStructure if successful, None otherwise
        """
        raw_content = getattr(document, 'raw_content', '') or ''
        if not raw_content:
            return None

        await self._ensure_client()

        # Sample content for analysis
        content_sample = raw_content[:self.MAX_CONTENT_CHARS]

        # Build prompt using the structured builder
        source_type = "youtube" if is_youtube else "document"
        prompt_builder = StructureExtractionPromptBuilder(
            source_type=source_type,
            has_chapters=False,  # We're generating because no explicit chapters
            content_language=content_language,
        )

        system_prompt = prompt_builder.build()
        user_prompt = prompt_builder.build_user_prompt(content_sample)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=self.temperature,
            )

            ai_output = response.choices[0].message.content.strip()
            if ai_output:
                return self._parse_ai_output(ai_output, document, is_youtube)

        except Exception as e:
            print(f"[AI_STRUCTURE] Generation failed: {e}", flush=True)

        return None

    def _parse_ai_output(
        self,
        ai_output: str,
        document,
        is_youtube: bool,
    ) -> DocumentStructure:
        """
        Parse AI-generated structure output.

        The AI outputs tree notation:
        ┌── Main Section
           ├── Subsection
           └── Subsection

        Args:
            ai_output: Raw AI output
            document: Original document
            is_youtube: Whether this is a YouTube transcript

        Returns:
            DocumentStructure
        """
        metadata = getattr(document, 'extracted_metadata', {}) or {}

        structure = DocumentStructure(
            is_youtube=is_youtube,
            title=metadata.get('title', getattr(document, 'filename', '')),
            summary=getattr(document, 'content_summary', ''),
            page_count=getattr(document, 'page_count', 0) or 0,
            word_count=getattr(document, 'word_count', 0) or len((getattr(document, 'raw_content', '') or '').split()),
            duration_seconds=metadata.get('duration_seconds', 0),
            has_toc=False,  # AI-generated, not explicit TOC
        )

        # Parse tree notation
        headings = []
        for line in ai_output.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Detect level from indentation and prefix
            level = 1
            text = line

            # Remove tree notation prefixes
            if '┌──' in line or '├──' in line or '└──' in line:
                # Count leading spaces for level
                original = line
                stripped = line.lstrip()
                indent = len(original) - len(stripped)
                level = max(1, indent // 3 + 1)

                # Extract text after prefix
                for prefix in ['┌──', '├──', '└──']:
                    if prefix in stripped:
                        text = stripped.split(prefix, 1)[1].strip()
                        break

            if text and len(text) > 2:
                headings.append(HeadingInfo(
                    text=text,
                    level=level,
                ))

        structure.headings = headings[:20]  # Limit headings

        print(
            f"[AI_STRUCTURE] Generated {len(structure.headings)} sections "
            f"for {structure.title}",
            flush=True
        )

        return structure

    def format_ai_structure(
        self,
        structure: DocumentStructure,
    ) -> str:
        """
        Format AI-generated structure for display.

        Args:
            structure: AI-generated structure

        Returns:
            Formatted string
        """
        parts = []

        # Header
        if structure.is_youtube:
            title = structure.title
            parts.append(f"\n🎬 VIDEO YOUTUBE: {title}")
            parts.append("   STRUCTURE DÉTECTÉE PAR IA (basée sur le contenu):")
        else:
            parts.append(f"\n📄 DOCUMENT: {structure.title}")
            parts.append("   STRUCTURE DÉTECTÉE PAR IA:")

        # Headings
        for heading in structure.headings:
            indent = "   " * heading.level
            prefix = "├──" if heading.level > 1 else "┌──"
            parts.append(f"   {indent}{prefix} {heading.text}")

        # Summary
        if structure.summary:
            parts.append(f"\n   SUMMARY: {structure.summary}")

        # Stats
        if structure.is_youtube:
            parts.append(f"   STATS: {structure.word_count} mots dans la transcription")
        else:
            parts.append(f"   STATS: {structure.page_count} pages, {structure.word_count} words")

        return "\n".join(parts)


# Module-level factory
_default_generator = None


def get_ai_structure_generator(
    openai_client=None,
    model: Optional[str] = None,
) -> AIStructureGenerator:
    """
    Get or create an AI structure generator instance.

    Args:
        openai_client: Optional OpenAI client
        model: Model to use

    Returns:
        AIStructureGenerator instance
    """
    global _default_generator
    if _default_generator is None or openai_client is not None:
        _default_generator = AIStructureGenerator(openai_client, model)
    return _default_generator
