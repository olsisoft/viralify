"""
Voice Sample Service

Handles voice sample upload, validation, and processing.
Phase 4: Voice Cloning feature.
"""
import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

from models.voice_cloning_models import (
    VoiceSample,
    SampleStatus,
    VoiceSampleRequirements,
)


class VoiceSampleService:
    """
    Service for collecting and validating voice samples.
    Handles upload, quality assessment, and audio analysis.
    """

    # Supported formats
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.ogg', '.webm', '.flac', '.aac'}

    # Size limits
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    MIN_DURATION = 5.0   # Minimum 5 seconds per sample
    MAX_DURATION = 300.0  # Maximum 5 minutes per sample

    # Quality thresholds
    MIN_QUALITY_SCORE = 0.3
    MAX_NOISE_LEVEL = 0.7

    def __init__(self, storage_path: str = "/tmp/viralify/voice_samples"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        print(f"[VOICE_SAMPLE] Service initialized at {storage_path}", flush=True)

    async def process_sample(
        self,
        file_content: bytes,
        filename: str,
        profile_id: str,
        user_id: str,
    ) -> VoiceSample:
        """
        Process an uploaded voice sample.

        Args:
            file_content: Raw audio bytes
            filename: Original filename
            profile_id: Voice profile ID
            user_id: User ID

        Returns:
            Processed VoiceSample
        """
        print(f"[VOICE_SAMPLE] Processing sample: {filename} ({len(file_content)} bytes)", flush=True)

        # Validate file extension
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {', '.join(self.SUPPORTED_FORMATS)}")

        # Check file size
        if len(file_content) > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum: {self.MAX_FILE_SIZE // 1024 // 1024} MB")

        # Generate unique filename
        sample_id = str(uuid.uuid4())
        safe_filename = f"{profile_id}_{sample_id}{ext}"
        file_path = self.storage_path / safe_filename

        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)

        print(f"[VOICE_SAMPLE] Sample saved: {file_path}", flush=True)

        # Get audio duration
        duration = await self._get_audio_duration(file_path)

        if duration < self.MIN_DURATION:
            # Remove file if too short
            file_path.unlink()
            raise ValueError(f"Sample too short ({duration:.1f}s). Minimum: {self.MIN_DURATION}s")

        if duration > self.MAX_DURATION:
            # Remove file if too long
            file_path.unlink()
            raise ValueError(f"Sample too long ({duration:.1f}s). Maximum: {self.MAX_DURATION}s")

        # Analyze audio quality
        quality_score, noise_level, clarity_score = await self._analyze_quality(file_path)

        # Determine status based on quality
        if noise_level > self.MAX_NOISE_LEVEL:
            status = SampleStatus.REJECTED
            rejection_reason = "Too much background noise. Please record in a quieter environment."
        elif quality_score < self.MIN_QUALITY_SCORE:
            status = SampleStatus.REJECTED
            rejection_reason = "Audio quality too low. Please use a better microphone."
        else:
            status = SampleStatus.VALIDATED
            rejection_reason = None

        # Create sample record
        sample = VoiceSample(
            id=sample_id,
            profile_id=profile_id,
            user_id=user_id,
            filename=filename,
            file_path=str(file_path),
            file_size_bytes=len(file_content),
            duration_seconds=duration,
            format=ext.lstrip('.'),
            quality_score=quality_score,
            noise_level=noise_level,
            clarity_score=clarity_score,
            status=status,
            rejection_reason=rejection_reason,
            processed_at=datetime.utcnow(),
        )

        print(f"[VOICE_SAMPLE] Sample processed: {sample_id} - {status.value}", flush=True)

        return sample

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

            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                duration = float(stdout.decode().strip())
                print(f"[VOICE_SAMPLE] Duration: {duration:.2f}s", flush=True)
                return duration
            else:
                print(f"[VOICE_SAMPLE] ffprobe error: {stderr.decode()}", flush=True)
                return 0.0

        except Exception as e:
            print(f"[VOICE_SAMPLE] Error getting duration: {e}", flush=True)
            return 0.0

    async def _analyze_quality(self, file_path: Path) -> Tuple[float, float, float]:
        """
        Analyze audio quality using FFmpeg.

        Returns:
            Tuple of (quality_score, noise_level, clarity_score)
        """
        try:
            # Use FFmpeg to analyze audio
            # Get volume stats for quality assessment
            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-af', 'volumedetect',
                '-f', 'null',
                '-'
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            _, stderr = await result.communicate()
            output = stderr.decode()

            # Parse volume stats
            mean_volume = -20.0  # Default
            max_volume = -10.0   # Default

            for line in output.split('\n'):
                if 'mean_volume' in line:
                    try:
                        mean_volume = float(line.split(':')[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass  # Invalid format, skip this line
                elif 'max_volume' in line:
                    try:
                        max_volume = float(line.split(':')[1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass  # Invalid format, skip this line

            # Calculate quality metrics
            # Good audio: mean around -20dB to -14dB, max around -6dB to -3dB

            # Quality score based on volume levels
            # Penalize if too quiet or too loud
            if -25 <= mean_volume <= -10:
                quality_score = 0.8
            elif -30 <= mean_volume <= -5:
                quality_score = 0.6
            else:
                quality_score = 0.4

            # Clipping detection (if max is 0 or positive, there's clipping)
            if max_volume >= 0:
                quality_score -= 0.2

            # Dynamic range indicates clarity
            dynamic_range = max_volume - mean_volume
            if 10 <= dynamic_range <= 20:
                clarity_score = 0.8
            elif 5 <= dynamic_range <= 25:
                clarity_score = 0.6
            else:
                clarity_score = 0.4

            # Estimate noise level (simplified - in production use silencedetect)
            # Lower mean volume with normal max suggests noise floor
            noise_estimate = max(0, min(1, (mean_volume + 40) / 30))
            noise_level = 1 - noise_estimate  # Invert so higher = more noise

            print(f"[VOICE_SAMPLE] Quality analysis - score: {quality_score:.2f}, noise: {noise_level:.2f}, clarity: {clarity_score:.2f}", flush=True)

            return (
                max(0, min(1, quality_score)),
                max(0, min(1, noise_level)),
                max(0, min(1, clarity_score))
            )

        except Exception as e:
            print(f"[VOICE_SAMPLE] Quality analysis error: {e}", flush=True)
            # Return default scores
            return (0.5, 0.5, 0.5)

    async def convert_to_standard_format(
        self,
        input_path: str,
        output_format: str = "mp3",
        sample_rate: int = 44100,
    ) -> str:
        """
        Convert audio to standard format for training.

        Args:
            input_path: Input file path
            output_format: Output format (mp3, wav)
            sample_rate: Target sample rate

        Returns:
            Path to converted file
        """
        input_file = Path(input_path)
        output_file = input_file.with_suffix(f'.converted.{output_format}')

        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_file),
            '-ar', str(sample_rate),
            '-ac', '1',  # Mono
            '-b:a', '192k',
            str(output_file)
        ]

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        await result.communicate()

        if result.returncode == 0 and output_file.exists():
            print(f"[VOICE_SAMPLE] Converted to {output_format}: {output_file}", flush=True)
            return str(output_file)

        return input_path

    async def transcribe_sample(self, file_path: str) -> Optional[str]:
        """
        Transcribe a voice sample using Whisper.
        This helps improve voice cloning quality.

        Args:
            file_path: Path to audio file

        Returns:
            Transcribed text or None
        """
        # In production, use OpenAI Whisper API or local Whisper model
        # For now, return None (transcript can be manually provided)
        print(f"[VOICE_SAMPLE] Transcription not implemented (placeholder)", flush=True)
        return None

    async def delete_sample(self, sample: VoiceSample) -> bool:
        """Delete a voice sample file"""
        try:
            file_path = Path(sample.file_path)
            if file_path.exists():
                file_path.unlink()
                print(f"[VOICE_SAMPLE] Deleted: {sample.file_path}", flush=True)
            return True
        except Exception as e:
            print(f"[VOICE_SAMPLE] Delete error: {e}", flush=True)
            return False

    def get_requirements(self) -> VoiceSampleRequirements:
        """Get sample requirements for voice cloning"""
        return VoiceSampleRequirements(
            min_samples=1,
            max_samples=25,
            min_duration_seconds=30,
            max_duration_seconds=180,
            ideal_duration_seconds=60,
            supported_formats=[f.lstrip('.') for f in self.SUPPORTED_FORMATS],
            max_file_size_mb=self.MAX_FILE_SIZE // 1024 // 1024,
            sample_rate_hz=44100,
            tips=[
                "Record in a quiet room with no echo",
                "Use a good quality microphone (USB mic or headset)",
                "Speak clearly at a natural pace",
                "Keep consistent distance from microphone",
                "Read varied content - not just the same phrase",
                "Include different tones: normal, excited, calm",
                "Avoid background music or TV",
                "30-60 seconds total is ideal for good quality"
            ]
        )


# Singleton instance
_voice_sample_service: Optional[VoiceSampleService] = None


def get_voice_sample_service() -> VoiceSampleService:
    """Get or create the voice sample service singleton"""
    global _voice_sample_service
    if _voice_sample_service is None:
        _voice_sample_service = VoiceSampleService()
    return _voice_sample_service
