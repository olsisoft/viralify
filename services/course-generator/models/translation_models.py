"""
Translation Models

Pydantic models for multi-language course translation.
"""
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SupportedLanguage(str, Enum):
    """Supported languages for translation (Top 10)"""
    ENGLISH = "en"
    FRENCH = "fr"
    SPANISH = "es"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    POLISH = "pl"
    RUSSIAN = "ru"
    CHINESE = "zh"


# Language metadata
LANGUAGE_INFO: Dict[str, Dict] = {
    "en": {"name": "English", "native": "English", "flag": "🇬🇧"},
    "fr": {"name": "French", "native": "Français", "flag": "🇫🇷"},
    "es": {"name": "Spanish", "native": "Español", "flag": "🇪🇸"},
    "de": {"name": "German", "native": "Deutsch", "flag": "🇩🇪"},
    "pt": {"name": "Portuguese", "native": "Português", "flag": "🇧🇷"},
    "it": {"name": "Italian", "native": "Italiano", "flag": "🇮🇹"},
    "nl": {"name": "Dutch", "native": "Nederlands", "flag": "🇳🇱"},
    "pl": {"name": "Polish", "native": "Polski", "flag": "🇵🇱"},
    "ru": {"name": "Russian", "native": "Русский", "flag": "🇷🇺"},
    "zh": {"name": "Chinese", "native": "中文", "flag": "🇨🇳"},
}


class TranslationStatus(str, Enum):
    """Translation job status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TranslatedContent(BaseModel):
    """Translated content for a single field"""
    original: str
    translated: str
    language: SupportedLanguage


class LectureTranslation(BaseModel):
    """Translated lecture content"""
    lecture_id: str
    original_language: SupportedLanguage
    target_language: SupportedLanguage
    title: TranslatedContent
    description: Optional[TranslatedContent] = None
    script: TranslatedContent
    voiceover_text: Optional[TranslatedContent] = None
    key_points: List[TranslatedContent] = Field(default_factory=list)


class CourseTranslation(BaseModel):
    """Full course translation"""
    course_id: str
    original_language: SupportedLanguage
    target_language: SupportedLanguage
    status: TranslationStatus = TranslationStatus.PENDING

    # Course-level translations
    title: Optional[TranslatedContent] = None
    description: Optional[TranslatedContent] = None
    objectives: List[TranslatedContent] = Field(default_factory=list)

    # Lecture translations
    lectures: List[LectureTranslation] = Field(default_factory=list)

    # Metadata
    total_lectures: int = 0
    translated_lectures: int = 0
    error_message: Optional[str] = None


class TranslationRequest(BaseModel):
    """Request to translate content"""
    course_id: str
    source_language: SupportedLanguage = SupportedLanguage.ENGLISH
    target_languages: List[SupportedLanguage]
    include_voiceover: bool = True  # Also generate translated voiceover text


class TranslationJobResponse(BaseModel):
    """Response for translation job"""
    job_id: str
    course_id: str
    target_languages: List[SupportedLanguage]
    status: TranslationStatus
    progress: float = 0.0  # 0.0 to 1.0
    translations: List[CourseTranslation] = Field(default_factory=list)


class TranslateTextRequest(BaseModel):
    """Request to translate a single text"""
    text: str
    source_language: SupportedLanguage = SupportedLanguage.ENGLISH
    target_language: SupportedLanguage
    context: Optional[str] = None  # Additional context for better translation


class TranslateTextResponse(BaseModel):
    """Response for single text translation"""
    original: str
    translated: str
    source_language: SupportedLanguage
    target_language: SupportedLanguage


class DetectLanguageRequest(BaseModel):
    """Request to detect language of text"""
    text: str


class DetectLanguageResponse(BaseModel):
    """Response for language detection"""
    detected_language: SupportedLanguage
    confidence: float
    language_name: str


class LanguageInfo(BaseModel):
    """Language information"""
    code: str
    name: str
    native_name: str
    flag: str


class SupportedLanguagesResponse(BaseModel):
    """Response listing all supported languages"""
    languages: List[LanguageInfo]
    default_source: str = "en"
