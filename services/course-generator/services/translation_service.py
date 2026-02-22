"""
Translation Service

Handles course content translation using OpenAI GPT-4.
Supports 10 languages with context-aware translation.
"""
import asyncio
import os
from typing import Dict, List, Optional

from openai import AsyncOpenAI

# Use shared LLM provider for multi-provider support
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    _USE_SHARED_LLM = False

from models.translation_models import (
    SupportedLanguage,
    LANGUAGE_INFO,
    TranslationStatus,
    TranslatedContent,
    LectureTranslation,
    CourseTranslation,
    TranslationJobResponse,
    DetectLanguageResponse,
    LanguageInfo,
)


class TranslationService:
    """
    Service for translating course content to multiple languages.
    Uses configured LLM provider for high-quality, context-aware translations.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        if _USE_SHARED_LLM:
            self.client = get_llm_client()
            self.model = get_model_name("fast")
        else:
            self.client = AsyncOpenAI(
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                timeout=120.0,
            )
            self.model = "gpt-4o-mini"
        print(f"[TRANSLATION] Service initialized with model {self.model}", flush=True)

    async def translate_text(
        self,
        text: str,
        source_language: SupportedLanguage,
        target_language: SupportedLanguage,
        context: Optional[str] = None,
    ) -> str:
        """
        Translate a single text from source to target language.

        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            context: Optional context for better translation

        Returns:
            Translated text
        """
        if source_language == target_language:
            return text

        if not text or not text.strip():
            return text

        source_name = LANGUAGE_INFO[source_language.value]["name"]
        target_name = LANGUAGE_INFO[target_language.value]["name"]

        system_prompt = f"""You are a professional translator specializing in educational content.
Translate the following text from {source_name} to {target_name}.

Guidelines:
- Maintain the original meaning and tone
- Keep technical terms accurate
- Preserve formatting (markdown, bullet points, etc.)
- Adapt cultural references appropriately
- Keep code snippets, URLs, and proper nouns unchanged
- Ensure natural flow in the target language"""

        if context:
            system_prompt += f"\n\nContext: {context}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=4000,
            )

            translated = response.choices[0].message.content.strip()
            return translated

        except Exception as e:
            print(f"[TRANSLATION] Error: {str(e)}", flush=True)
            raise

    async def translate_texts_batch(
        self,
        texts: List[str],
        source_language: SupportedLanguage,
        target_language: SupportedLanguage,
        context: Optional[str] = None,
    ) -> List[str]:
        """
        Translate multiple texts in batch for efficiency.

        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
            context: Optional context

        Returns:
            List of translated texts
        """
        if source_language == target_language:
            return texts

        # Filter empty texts
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]

        if not non_empty:
            return texts

        # Batch texts with separators
        separator = "\n---SEPARATOR---\n"
        combined = separator.join([t for _, t in non_empty])

        source_name = LANGUAGE_INFO[source_language.value]["name"]
        target_name = LANGUAGE_INFO[target_language.value]["name"]

        system_prompt = f"""You are a professional translator for educational content.
Translate the following texts from {source_name} to {target_name}.

The texts are separated by '---SEPARATOR---'. Keep this separator in your response.
Translate each text independently while maintaining consistency across all of them.

Guidelines:
- Maintain original meaning and tone
- Keep technical terms accurate
- Preserve formatting
- Keep code, URLs, proper nouns unchanged"""

        if context:
            system_prompt += f"\n\nContext: {context}"

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": combined},
                ],
                temperature=0.3,
                max_tokens=8000,
            )

            translated_combined = response.choices[0].message.content.strip()
            translated_parts = translated_combined.split("---SEPARATOR---")

            # Map back to original positions
            result = list(texts)  # Copy original
            for (original_idx, _), translated in zip(non_empty, translated_parts):
                result[original_idx] = translated.strip()

            return result

        except Exception as e:
            print(f"[TRANSLATION] Batch error: {str(e)}", flush=True)
            # Fallback to individual translations
            tasks = [
                self.translate_text(t, source_language, target_language, context)
                for t in texts
            ]
            return await asyncio.gather(*tasks)

    async def detect_language(self, text: str) -> DetectLanguageResponse:
        """
        Detect the language of a text.

        Args:
            text: Text to analyze

        Returns:
            DetectLanguageResponse with detected language
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Detect the language of the given text.
Respond with only the ISO 639-1 language code (e.g., 'en', 'fr', 'es', 'de', 'pt', 'it', 'nl', 'pl', 'ru', 'zh').
If uncertain, provide your best guess.""",
                    },
                    {"role": "user", "content": text[:500]},  # Limit text for detection
                ],
                temperature=0,
                max_tokens=10,
            )

            detected_code = response.choices[0].message.content.strip().lower()

            # Validate and map to supported language
            try:
                language = SupportedLanguage(detected_code)
            except ValueError:
                language = SupportedLanguage.ENGLISH  # Default fallback

            lang_info = LANGUAGE_INFO.get(language.value, LANGUAGE_INFO["en"])

            return DetectLanguageResponse(
                detected_language=language,
                confidence=0.9,  # GPT doesn't provide confidence, using fixed value
                language_name=lang_info["name"],
            )

        except Exception as e:
            print(f"[TRANSLATION] Detection error: {str(e)}", flush=True)
            return DetectLanguageResponse(
                detected_language=SupportedLanguage.ENGLISH,
                confidence=0.5,
                language_name="English",
            )

    async def translate_course(
        self,
        course_data: Dict,
        source_language: SupportedLanguage,
        target_language: SupportedLanguage,
    ) -> CourseTranslation:
        """
        Translate an entire course including all lectures.

        Args:
            course_data: Course data dictionary
            source_language: Source language
            target_language: Target language

        Returns:
            CourseTranslation with all translated content
        """
        print(f"[TRANSLATION] Translating course to {target_language.value}", flush=True)

        translation = CourseTranslation(
            course_id=course_data.get("id", ""),
            original_language=source_language,
            target_language=target_language,
            status=TranslationStatus.IN_PROGRESS,
        )

        try:
            # Translate course-level content
            course_title = course_data.get("title", "")
            course_description = course_data.get("description", "")
            objectives = course_data.get("objectives", [])

            # Batch translate course-level content
            course_texts = [course_title, course_description] + objectives
            translated_texts = await self.translate_texts_batch(
                course_texts,
                source_language,
                target_language,
                context="Educational course content",
            )

            translation.title = TranslatedContent(
                original=course_title,
                translated=translated_texts[0],
                language=target_language,
            )

            if course_description:
                translation.description = TranslatedContent(
                    original=course_description,
                    translated=translated_texts[1],
                    language=target_language,
                )

            translation.objectives = [
                TranslatedContent(
                    original=obj,
                    translated=translated_texts[2 + i],
                    language=target_language,
                )
                for i, obj in enumerate(objectives)
            ]

            # Translate lectures
            lectures = course_data.get("lectures", [])
            translation.total_lectures = len(lectures)

            for i, lecture in enumerate(lectures):
                lecture_translation = await self._translate_lecture(
                    lecture,
                    source_language,
                    target_language,
                )
                translation.lectures.append(lecture_translation)
                translation.translated_lectures = i + 1
                print(f"[TRANSLATION] Lecture {i + 1}/{len(lectures)} completed", flush=True)

            translation.status = TranslationStatus.COMPLETED
            print(f"[TRANSLATION] Course translation completed", flush=True)

        except Exception as e:
            translation.status = TranslationStatus.FAILED
            translation.error_message = str(e)
            print(f"[TRANSLATION] Course translation failed: {str(e)}", flush=True)

        return translation

    async def _translate_lecture(
        self,
        lecture: Dict,
        source_language: SupportedLanguage,
        target_language: SupportedLanguage,
    ) -> LectureTranslation:
        """Translate a single lecture"""
        lecture_id = lecture.get("id", "")
        title = lecture.get("title", "")
        description = lecture.get("description", "")
        script = lecture.get("script", "")
        voiceover = lecture.get("voiceover_text", "")
        key_points = lecture.get("key_points", [])

        # Collect all texts for batch translation
        texts = [title, description, script, voiceover] + key_points
        translated = await self.translate_texts_batch(
            texts,
            source_language,
            target_language,
            context="Educational lecture content",
        )

        return LectureTranslation(
            lecture_id=lecture_id,
            original_language=source_language,
            target_language=target_language,
            title=TranslatedContent(
                original=title,
                translated=translated[0],
                language=target_language,
            ),
            description=TranslatedContent(
                original=description,
                translated=translated[1],
                language=target_language,
            ) if description else None,
            script=TranslatedContent(
                original=script,
                translated=translated[2],
                language=target_language,
            ),
            voiceover_text=TranslatedContent(
                original=voiceover,
                translated=translated[3],
                language=target_language,
            ) if voiceover else None,
            key_points=[
                TranslatedContent(
                    original=kp,
                    translated=translated[4 + i],
                    language=target_language,
                )
                for i, kp in enumerate(key_points)
            ],
        )

    def get_supported_languages(self) -> List[LanguageInfo]:
        """Get list of all supported languages"""
        return [
            LanguageInfo(
                code=code,
                name=info["name"],
                native_name=info["native"],
                flag=info["flag"],
            )
            for code, info in LANGUAGE_INFO.items()
        ]


# Global instance
translation_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    """Get or create translation service instance"""
    global translation_service
    if translation_service is None:
        translation_service = TranslationService()
    return translation_service
