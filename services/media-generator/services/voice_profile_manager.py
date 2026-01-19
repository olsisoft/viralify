"""
Voice Profile Manager

Orchestrates voice cloning workflow: profiles, samples, training, and generation.
Phase 4: Voice Cloning feature.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from models.voice_cloning_models import (
    VoiceProfile,
    VoiceSample,
    VoiceProfileStatus,
    SampleStatus,
    VoiceGenerationSettings,
    VoiceConsentRecord,
    CreateVoiceProfileRequest,
    VoiceSampleRequirements,
)
from services.voice_sample_service import get_voice_sample_service
from services.voice_cloning_service import get_voice_cloning_service


class VoiceProfileRepository:
    """
    In-memory voice profile repository.
    In production, use PostgreSQL.
    """

    def __init__(self):
        self.profiles: Dict[str, VoiceProfile] = {}
        self.user_profiles: Dict[str, List[str]] = {}  # user_id -> [profile_ids]
        self.consents: Dict[str, VoiceConsentRecord] = {}  # profile_id -> consent

    async def save(self, profile: VoiceProfile) -> None:
        """Save profile"""
        self.profiles[profile.id] = profile

        if profile.user_id not in self.user_profiles:
            self.user_profiles[profile.user_id] = []
        if profile.id not in self.user_profiles[profile.user_id]:
            self.user_profiles[profile.user_id].append(profile.id)

    async def get(self, profile_id: str) -> Optional[VoiceProfile]:
        """Get profile by ID"""
        return self.profiles.get(profile_id)

    async def get_by_user(self, user_id: str) -> List[VoiceProfile]:
        """Get all profiles for a user"""
        profile_ids = self.user_profiles.get(user_id, [])
        return [self.profiles[pid] for pid in profile_ids if pid in self.profiles]

    async def delete(self, profile_id: str) -> Optional[VoiceProfile]:
        """Delete profile"""
        profile = self.profiles.pop(profile_id, None)
        if profile:
            if profile.user_id in self.user_profiles:
                self.user_profiles[profile.user_id] = [
                    p for p in self.user_profiles[profile.user_id] if p != profile_id
                ]
            self.consents.pop(profile_id, None)
        return profile

    async def save_consent(self, consent: VoiceConsentRecord) -> None:
        """Save consent record"""
        self.consents[consent.profile_id] = consent

    async def get_consent(self, profile_id: str) -> Optional[VoiceConsentRecord]:
        """Get consent record"""
        return self.consents.get(profile_id)


class VoiceProfileManager:
    """
    Main service for managing voice profiles and cloning workflow.
    """

    # Training requirements
    MIN_SAMPLES = 1
    MIN_TOTAL_DURATION = 30  # seconds
    MAX_TOTAL_DURATION = 180  # seconds
    IDEAL_DURATION = 60  # seconds

    def __init__(self):
        self.repository = VoiceProfileRepository()
        self.sample_service = get_voice_sample_service()
        self.cloning_service = get_voice_cloning_service()
        print("[VOICE_PROFILE] Manager initialized", flush=True)

    # ========================================
    # Profile Management
    # ========================================

    async def create_profile(
        self,
        request: CreateVoiceProfileRequest,
        user_id: str,
    ) -> VoiceProfile:
        """Create a new voice profile"""
        print(f"[VOICE_PROFILE] Creating profile: {request.name} for user {user_id}", flush=True)

        profile = VoiceProfile(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=request.name,
            description=request.description,
            gender=request.gender,
            age=request.age,
            accent=request.accent,
            language=request.language,
            provider=request.provider,
            status=VoiceProfileStatus.DRAFT,
        )

        await self.repository.save(profile)

        print(f"[VOICE_PROFILE] Profile created: {profile.id}", flush=True)
        return profile

    async def get_profile(self, profile_id: str, user_id: str) -> Optional[VoiceProfile]:
        """Get profile with access control"""
        profile = await self.repository.get(profile_id)
        if profile and profile.user_id == user_id:
            return profile
        return None

    async def list_profiles(self, user_id: str) -> List[VoiceProfile]:
        """List all profiles for a user"""
        return await self.repository.get_by_user(user_id)

    async def delete_profile(self, profile_id: str, user_id: str) -> bool:
        """Delete profile and associated resources"""
        profile = await self.repository.get(profile_id)
        if not profile or profile.user_id != user_id:
            return False

        # Delete voice from provider
        if profile.provider_voice_id:
            await self.cloning_service.delete_voice(profile.provider_voice_id)

        # Delete samples
        for sample in profile.samples:
            await self.sample_service.delete_sample(sample)

        # Delete profile
        await self.repository.delete(profile_id)

        print(f"[VOICE_PROFILE] Profile deleted: {profile_id}", flush=True)
        return True

    async def update_profile(
        self,
        profile_id: str,
        user_id: str,
        updates: Dict,
    ) -> Optional[VoiceProfile]:
        """Update profile settings"""
        profile = await self.get_profile(profile_id, user_id)
        if not profile:
            return None

        # Update allowed fields
        if "name" in updates:
            profile.name = updates["name"]
        if "description" in updates:
            profile.description = updates["description"]
        if "default_stability" in updates:
            profile.default_stability = updates["default_stability"]
        if "default_similarity" in updates:
            profile.default_similarity = updates["default_similarity"]
        if "default_style" in updates:
            profile.default_style = updates["default_style"]

        profile.updated_at = datetime.utcnow()
        await self.repository.save(profile)

        return profile

    # ========================================
    # Sample Management
    # ========================================

    async def add_sample(
        self,
        profile_id: str,
        user_id: str,
        file_content: bytes,
        filename: str,
    ) -> Tuple[Optional[VoiceSample], str]:
        """
        Add a voice sample to a profile.

        Returns:
            Tuple of (sample, message)
        """
        profile = await self.get_profile(profile_id, user_id)
        if not profile:
            return None, "Profile not found"

        if profile.status not in [VoiceProfileStatus.DRAFT, VoiceProfileStatus.FAILED]:
            return None, "Cannot add samples to a trained profile"

        try:
            # Process sample
            sample = await self.sample_service.process_sample(
                file_content, filename, profile_id, user_id
            )

            # Add to profile if validated
            if sample.status == SampleStatus.VALIDATED:
                profile.samples.append(sample)
                profile.total_sample_duration += sample.duration_seconds
                profile.updated_at = datetime.utcnow()
                await self.repository.save(profile)

                return sample, f"Sample added ({sample.duration_seconds:.1f}s)"
            else:
                return sample, sample.rejection_reason or "Sample rejected"

        except ValueError as e:
            return None, str(e)

    async def remove_sample(
        self,
        profile_id: str,
        sample_id: str,
        user_id: str,
    ) -> bool:
        """Remove a sample from a profile"""
        profile = await self.get_profile(profile_id, user_id)
        if not profile:
            return False

        if profile.status not in [VoiceProfileStatus.DRAFT, VoiceProfileStatus.FAILED]:
            return False

        # Find and remove sample
        sample_to_remove = None
        for sample in profile.samples:
            if sample.id == sample_id:
                sample_to_remove = sample
                break

        if not sample_to_remove:
            return False

        # Delete file
        await self.sample_service.delete_sample(sample_to_remove)

        # Remove from profile
        profile.samples = [s for s in profile.samples if s.id != sample_id]
        profile.total_sample_duration -= sample_to_remove.duration_seconds
        profile.updated_at = datetime.utcnow()
        await self.repository.save(profile)

        return True

    # ========================================
    # Training
    # ========================================

    def can_start_training(self, profile: VoiceProfile) -> Tuple[bool, str]:
        """Check if profile has enough samples for training"""
        if profile.status not in [VoiceProfileStatus.DRAFT, VoiceProfileStatus.FAILED]:
            return False, "Profile already trained or training"

        validated_samples = [s for s in profile.samples if s.status == SampleStatus.VALIDATED]

        if len(validated_samples) < self.MIN_SAMPLES:
            return False, f"Need at least {self.MIN_SAMPLES} sample(s)"

        if profile.total_sample_duration < self.MIN_TOTAL_DURATION:
            return False, f"Need at least {self.MIN_TOTAL_DURATION}s of audio (current: {profile.total_sample_duration:.1f}s)"

        return True, "Ready for training"

    async def start_training(
        self,
        profile_id: str,
        user_id: str,
        consent_confirmed: bool,
        ip_address: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Start voice cloning training.

        Args:
            profile_id: Profile to train
            user_id: User ID
            consent_confirmed: User confirmed voice ownership
            ip_address: Request IP for consent record

        Returns:
            Tuple of (success, message)
        """
        profile = await self.get_profile(profile_id, user_id)
        if not profile:
            return False, "Profile not found"

        # Verify training readiness
        can_train, reason = self.can_start_training(profile)
        if not can_train:
            return False, reason

        # Verify consent
        if not consent_confirmed:
            return False, "Voice ownership consent required"

        # Record consent
        consent = VoiceConsentRecord(
            user_id=user_id,
            profile_id=profile_id,
            consent_given=True,
            ip_address=ip_address,
            accepted_terms=True,
        )
        await self.repository.save_consent(consent)

        # Update profile consent
        profile.consent_given = True
        profile.consent_timestamp = datetime.utcnow()
        profile.consent_ip_address = ip_address
        profile.status = VoiceProfileStatus.TRAINING
        profile.training_progress = 10
        await self.repository.save(profile)

        print(f"[VOICE_PROFILE] Starting training for profile: {profile_id}", flush=True)

        # Get validated samples
        validated_samples = [s for s in profile.samples if s.status == SampleStatus.VALIDATED]

        try:
            # Create cloned voice
            result = await self.cloning_service.create_cloned_voice(profile, validated_samples)

            if result.get("success"):
                profile.provider_voice_id = result["voice_id"]
                profile.status = VoiceProfileStatus.READY
                profile.training_progress = 100
                profile.trained_at = datetime.utcnow()
                await self.repository.save(profile)

                print(f"[VOICE_PROFILE] Training complete: {profile_id}", flush=True)
                return True, f"Voice clone created successfully"
            else:
                profile.status = VoiceProfileStatus.FAILED
                profile.error_message = result.get("error", "Unknown error")
                await self.repository.save(profile)

                return False, profile.error_message

        except Exception as e:
            profile.status = VoiceProfileStatus.FAILED
            profile.error_message = str(e)
            await self.repository.save(profile)

            print(f"[VOICE_PROFILE] Training failed: {e}", flush=True)
            return False, str(e)

    # ========================================
    # Generation
    # ========================================

    async def generate_speech(
        self,
        profile_id: str,
        user_id: str,
        text: str,
        settings: Optional[VoiceGenerationSettings] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Generate speech using a cloned voice.

        Args:
            profile_id: Voice profile ID
            user_id: User ID
            text: Text to synthesize
            settings: Generation settings

        Returns:
            Tuple of (audio_path, message)
        """
        profile = await self.get_profile(profile_id, user_id)
        if not profile:
            return None, "Profile not found"

        if profile.status != VoiceProfileStatus.READY:
            return None, f"Profile not ready (status: {profile.status})"

        if not profile.provider_voice_id:
            return None, "No voice ID associated with profile"

        # Use profile defaults if not overridden
        if settings is None:
            settings = VoiceGenerationSettings(
                stability=profile.default_stability,
                similarity_boost=profile.default_similarity,
                style=profile.default_style,
            )

        print(f"[VOICE_PROFILE] Generating speech for profile: {profile_id}", flush=True)

        try:
            result = await self.cloning_service.generate_speech(
                profile.provider_voice_id,
                text,
                settings,
            )

            if result.get("success"):
                # Update usage stats
                profile.total_characters_generated += result.get("characters_used", len(text))
                profile.total_generations += 1
                profile.last_used_at = datetime.utcnow()
                await self.repository.save(profile)

                return result["audio_path"], "Speech generated successfully"
            else:
                return None, result.get("error", "Generation failed")

        except Exception as e:
            print(f"[VOICE_PROFILE] Generation error: {e}", flush=True)
            return None, str(e)

    async def preview_voice(
        self,
        profile_id: str,
        user_id: str,
        text: str = "Hello! This is a preview of my cloned voice. How does it sound?",
    ) -> Tuple[Optional[str], str]:
        """Generate a short preview of the cloned voice"""
        settings = VoiceGenerationSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
        )
        return await self.generate_speech(profile_id, user_id, text, settings)

    # ========================================
    # Utility
    # ========================================

    def get_training_requirements(self, profile: Optional[VoiceProfile] = None) -> Dict:
        """Get training requirements and current status"""
        requirements = self.sample_service.get_requirements()

        status = {
            "min_samples": self.MIN_SAMPLES,
            "min_duration_seconds": self.MIN_TOTAL_DURATION,
            "max_duration_seconds": self.MAX_TOTAL_DURATION,
            "ideal_duration_seconds": self.IDEAL_DURATION,
            "requirements": requirements.model_dump(),
        }

        if profile:
            validated_count = len([s for s in profile.samples if s.status == SampleStatus.VALIDATED])
            status["current_samples"] = validated_count
            status["current_duration"] = profile.total_sample_duration
            status["can_train"], status["training_message"] = self.can_start_training(profile)
            status["progress_percent"] = min(100, (profile.total_sample_duration / self.IDEAL_DURATION) * 100)

        return status


# Singleton instance
_voice_profile_manager: Optional[VoiceProfileManager] = None


def get_voice_profile_manager() -> VoiceProfileManager:
    """Get or create the voice profile manager singleton"""
    global _voice_profile_manager
    if _voice_profile_manager is None:
        _voice_profile_manager = VoiceProfileManager()
    return _voice_profile_manager
