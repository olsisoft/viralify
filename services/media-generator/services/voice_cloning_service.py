"""
Voice Cloning Service

Integrates with ElevenLabs API for voice cloning and generation.
Phase 4: Voice Cloning feature.
"""
import asyncio
import os
import uuid
import httpx
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from models.voice_cloning_models import (
    VoiceProfile,
    VoiceSample,
    VoiceProfileStatus,
    VoiceGenerationSettings,
    VoiceProvider,
)


class VoiceCloningService:
    """
    Service for voice cloning via ElevenLabs API.
    Handles voice creation, training, and speech generation.
    """

    ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1"

    # Output formats
    OUTPUT_FORMATS = {
        "mp3_44100_128": "mp3_44100_128",
        "mp3_44100_192": "mp3_44100_192",
        "pcm_16000": "pcm_16000",
        "pcm_22050": "pcm_22050",
        "pcm_44100": "pcm_44100",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_path: str = "/tmp/viralify/cloned_voices"
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            print("[VOICE_CLONE] WARNING: No ElevenLabs API key configured", flush=True)
        else:
            print(f"[VOICE_CLONE] Service initialized with ElevenLabs", flush=True)

    def is_available(self) -> bool:
        """Check if the service is available"""
        return bool(self.api_key)

    async def create_cloned_voice(
        self,
        profile: VoiceProfile,
        samples: List[VoiceSample],
    ) -> Dict[str, Any]:
        """
        Create a cloned voice using ElevenLabs Instant Voice Cloning.

        Args:
            profile: Voice profile with metadata
            samples: List of validated voice samples

        Returns:
            Dict with voice_id and status
        """
        if not self.is_available():
            raise RuntimeError("ElevenLabs API key not configured")

        if not samples:
            raise ValueError("At least one voice sample is required")

        print(f"[VOICE_CLONE] Creating voice clone for profile: {profile.id}", flush=True)
        print(f"[VOICE_CLONE] Using {len(samples)} samples", flush=True)

        # Prepare files for upload - use list to track handles for cleanup
        files = []
        file_handles = []  # Track handles separately for guaranteed cleanup

        try:
            for sample in samples:
                sample_path = Path(sample.file_path)
                if sample_path.exists():
                    fh = open(sample_path, "rb")
                    file_handles.append(fh)
                    files.append(("files", (sample.filename, fh, "audio/mpeg")))

            if not files:
                raise ValueError("No valid sample files found")

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ELEVENLABS_API_URL}/voices/add",
                    headers={"xi-api-key": self.api_key},
                    data={
                        "name": f"viralify_{profile.user_id}_{profile.id[:8]}",
                        "description": profile.description or f"Cloned voice for {profile.name}",
                        "labels": f'{{"user_id": "{profile.user_id}", "profile_id": "{profile.id}"}}'
                    },
                    files=files,
                )

            if response.status_code == 200:
                data = response.json()
                voice_id = data.get("voice_id")

                print(f"[VOICE_CLONE] Voice created successfully: {voice_id}", flush=True)

                return {
                    "success": True,
                    "voice_id": voice_id,
                    "status": VoiceProfileStatus.READY,
                }
            else:
                error_detail = response.text
                print(f"[VOICE_CLONE] API error: {response.status_code} - {error_detail}", flush=True)

                return {
                    "success": False,
                    "error": f"ElevenLabs API error: {response.status_code}",
                    "detail": error_detail,
                    "status": VoiceProfileStatus.FAILED,
                }

        except Exception as e:
            print(f"[VOICE_CLONE] Error creating voice: {e}", flush=True)
            return {
                "success": False,
                "error": str(e),
                "status": VoiceProfileStatus.FAILED,
            }
        finally:
            # Guaranteed cleanup of all file handles
            for fh in file_handles:
                try:
                    fh.close()
                except Exception:
                    pass

    async def generate_speech(
        self,
        voice_id: str,
        text: str,
        settings: Optional[VoiceGenerationSettings] = None,
    ) -> Dict[str, Any]:
        """
        Generate speech using a cloned voice.

        Args:
            voice_id: ElevenLabs voice ID
            text: Text to synthesize
            settings: Voice generation settings

        Returns:
            Dict with audio_url and metadata
        """
        if not self.is_available():
            raise RuntimeError("ElevenLabs API key not configured")

        settings = settings or VoiceGenerationSettings()

        print(f"[VOICE_CLONE] Generating speech with voice: {voice_id}", flush=True)
        print(f"[VOICE_CLONE] Text length: {len(text)} characters", flush=True)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ELEVENLABS_API_URL}/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "Accept": "audio/mpeg",
                    },
                    json={
                        "text": text,
                        "model_id": "eleven_multilingual_v2",
                        "voice_settings": {
                            "stability": settings.stability,
                            "similarity_boost": settings.similarity_boost,
                            "style": settings.style,
                            "use_speaker_boost": settings.use_speaker_boost,
                        }
                    }
                )

            if response.status_code == 200:
                # Save audio file
                audio_id = str(uuid.uuid4())[:8]
                audio_filename = f"cloned_{voice_id}_{audio_id}.mp3"
                audio_path = self.output_path / audio_filename

                with open(audio_path, "wb") as f:
                    f.write(response.content)

                # Get duration
                duration = await self._get_audio_duration(audio_path)

                print(f"[VOICE_CLONE] Audio generated: {audio_path} ({duration:.2f}s)", flush=True)

                return {
                    "success": True,
                    "audio_path": str(audio_path),
                    "audio_filename": audio_filename,
                    "duration_seconds": duration,
                    "characters_used": len(text),
                }
            else:
                error_detail = response.text
                print(f"[VOICE_CLONE] TTS error: {response.status_code} - {error_detail}", flush=True)

                return {
                    "success": False,
                    "error": f"TTS error: {response.status_code}",
                    "detail": error_detail,
                }

        except Exception as e:
            print(f"[VOICE_CLONE] Error generating speech: {e}", flush=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def delete_voice(self, voice_id: str) -> bool:
        """Delete a cloned voice from ElevenLabs"""
        if not self.is_available():
            return False

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.ELEVENLABS_API_URL}/voices/{voice_id}",
                    headers={"xi-api-key": self.api_key},
                )

            if response.status_code == 200:
                print(f"[VOICE_CLONE] Voice deleted: {voice_id}", flush=True)
                return True
            else:
                print(f"[VOICE_CLONE] Delete error: {response.status_code}", flush=True)
                return False

        except Exception as e:
            print(f"[VOICE_CLONE] Error deleting voice: {e}", flush=True)
            return False

    async def get_voice_info(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a cloned voice"""
        if not self.is_available():
            return None

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.ELEVENLABS_API_URL}/voices/{voice_id}",
                    headers={"xi-api-key": self.api_key},
                )

            if response.status_code == 200:
                return response.json()
            return None

        except Exception as e:
            print(f"[VOICE_CLONE] Error getting voice info: {e}", flush=True)
            return None

    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        if not self.is_available():
            return {"error": "API not configured"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.ELEVENLABS_API_URL}/user/subscription",
                    headers={"xi-api-key": self.api_key},
                )

            if response.status_code == 200:
                data = response.json()
                return {
                    "character_count": data.get("character_count", 0),
                    "character_limit": data.get("character_limit", 0),
                    "voice_limit": data.get("voice_limit", 0),
                    "tier": data.get("tier", "unknown"),
                }
            return {"error": f"API error: {response.status_code}"}

        except Exception as e:
            return {"error": str(e)}

    async def list_voices(self) -> List[Dict[str, Any]]:
        """List all voices in the account"""
        if not self.is_available():
            return []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.ELEVENLABS_API_URL}/voices",
                    headers={"xi-api-key": self.api_key},
                )

            if response.status_code == 200:
                data = response.json()
                return data.get("voices", [])
            return []

        except Exception as e:
            print(f"[VOICE_CLONE] Error listing voices: {e}", flush=True)
            return []

    async def _get_audio_duration(self, file_path: Path) -> float:
        """Get audio duration using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(file_path)
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, _ = await result.communicate()

            if result.returncode == 0:
                return float(stdout.decode().strip())

        except Exception as e:
            print(f"[VOICE_CLONE] Error getting duration: {e}", flush=True)

        return 0.0


# Singleton instance
_voice_cloning_service: Optional[VoiceCloningService] = None


def get_voice_cloning_service() -> VoiceCloningService:
    """Get or create the voice cloning service singleton"""
    global _voice_cloning_service
    if _voice_cloning_service is None:
        _voice_cloning_service = VoiceCloningService()
    return _voice_cloning_service
