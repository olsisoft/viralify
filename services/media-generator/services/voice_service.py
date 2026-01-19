"""
Voice Service - Manages voice selection and avatar-voice matching
Supports multiple languages and providers (ElevenLabs, OpenAI)
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VoiceInfo:
    """Voice information"""
    id: str
    name: str
    provider: str
    gender: str
    language: str
    style: str
    description: str


class VoiceService:
    """Service for managing voices and avatar-voice matching"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize voice service with configuration"""
        if config_path:
            self.config_path = config_path
        else:
            self.config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config",
                "voices.json"
            )
        self._config_cache = None

    def _load_config(self) -> Dict[str, Any]:
        """Load voice configuration from JSON file"""
        if self._config_cache is not None:
            return self._config_cache

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config_cache = json.load(f)
                return self._config_cache
        except FileNotFoundError:
            logger.warning(f"Voice config not found: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid voice config JSON: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found"""
        return {
            "default_voices": {
                "male": "pNInz6obpgDQGcFmaJgB",
                "female": "21m00Tcm4TlvDq8ikWAM",
                "neutral": "21m00Tcm4TlvDq8ikWAM"
            },
            "avatar_voice_mapping": {},
            "voices": {"elevenlabs": {"en": {"male": [], "female": []}}}
        }

    def get_voice_for_avatar(
        self,
        avatar_id: str,
        language: str = "en",
        provider: str = "elevenlabs"
    ) -> str:
        """
        Get the appropriate voice ID for an avatar based on its gender.

        Args:
            avatar_id: The avatar ID (e.g., "avatar-pro-male-alex")
            language: Language code (e.g., "en", "es", "fr")
            provider: Voice provider ("elevenlabs" or "openai")

        Returns:
            Voice ID appropriate for the avatar's gender
        """
        config = self._load_config()

        # Check if avatar has a specific voice mapping
        avatar_mapping = config.get("avatar_voice_mapping", {}).get(avatar_id, {})

        if avatar_mapping:
            gender = avatar_mapping.get("gender", "neutral")
            preferred_voice = avatar_mapping.get("preferred_voice")

            # If preferred voice is set, use it
            if preferred_voice:
                logger.info(f"Using preferred voice '{preferred_voice}' for avatar '{avatar_id}'")
                return preferred_voice
        else:
            # Infer gender from avatar ID
            gender = self._infer_gender_from_avatar_id(avatar_id)

        # Get voice by gender and language
        return self.get_voice_by_gender(gender, language, provider)

    def _infer_gender_from_avatar_id(self, avatar_id: str) -> str:
        """Infer gender from avatar ID naming convention"""
        avatar_id_lower = avatar_id.lower()
        if "-male-" in avatar_id_lower or "male" in avatar_id_lower:
            return "male"
        elif "-female-" in avatar_id_lower or "female" in avatar_id_lower:
            return "female"
        elif "-neutral-" in avatar_id_lower or "neutral" in avatar_id_lower:
            return "neutral"
        return "neutral"

    def get_voice_by_gender(
        self,
        gender: str,
        language: str = "en",
        provider: str = "elevenlabs",
        style: Optional[str] = None
    ) -> str:
        """
        Get a voice ID by gender and language.

        Args:
            gender: "male", "female", or "neutral"
            language: Language code
            provider: Voice provider
            style: Optional voice style preference

        Returns:
            Voice ID
        """
        config = self._load_config()

        # Try to get voice from provider's language voices
        provider_voices = config.get("voices", {}).get(provider, {})
        language_voices = provider_voices.get(language, provider_voices.get("en", {}))
        gender_voices = language_voices.get(gender, [])

        if gender_voices:
            # If style preference, try to match it
            if style:
                for voice in gender_voices:
                    if voice.get("style") == style:
                        logger.info(f"Selected voice '{voice['name']}' ({voice['id']}) for {gender}/{language}/{style}")
                        return voice["id"]

            # Return first available voice for this gender
            voice = gender_voices[0]
            logger.info(f"Selected voice '{voice['name']}' ({voice['id']}) for {gender}/{language}")
            return voice["id"]

        # Fallback to default voices
        default_voices = config.get("default_voices", {})
        default_voice = default_voices.get(gender, default_voices.get("neutral", "21m00Tcm4TlvDq8ikWAM"))
        logger.info(f"Using default voice '{default_voice}' for {gender}")
        return default_voice

    def get_available_voices(
        self,
        language: str = "en",
        provider: str = "elevenlabs",
        gender: Optional[str] = None
    ) -> List[VoiceInfo]:
        """
        Get list of available voices for a language.

        Args:
            language: Language code
            provider: Voice provider
            gender: Optional filter by gender

        Returns:
            List of VoiceInfo objects
        """
        config = self._load_config()
        voices = []

        provider_voices = config.get("voices", {}).get(provider, {})
        language_voices = provider_voices.get(language, {})

        genders_to_check = [gender] if gender else ["male", "female", "neutral"]

        for g in genders_to_check:
            for voice_data in language_voices.get(g, []):
                voices.append(VoiceInfo(
                    id=voice_data["id"],
                    name=voice_data["name"],
                    provider=provider,
                    gender=g,
                    language=language,
                    style=voice_data.get("style", "default"),
                    description=voice_data.get("description", "")
                ))

        return voices

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        config = self._load_config()
        return config.get("supported_languages", ["en"])

    def get_avatar_gender(self, avatar_id: str) -> str:
        """Get the gender associated with an avatar"""
        config = self._load_config()
        avatar_mapping = config.get("avatar_voice_mapping", {}).get(avatar_id, {})

        if avatar_mapping:
            return avatar_mapping.get("gender", "neutral")

        return self._infer_gender_from_avatar_id(avatar_id)


# Singleton instance
_voice_service = None


def get_voice_service() -> VoiceService:
    """Get the voice service singleton"""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service
