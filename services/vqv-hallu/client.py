"""
VQV-HALLU Client Library

This module provides a client for other services to call the VQV-HALLU
hallucination detection service with graceful degradation.

Usage:
    from vqv_hallu_client import VQVHalluClient

    client = VQVHalluClient()

    # Analyze a voiceover
    result = await client.analyze(
        audio_url="https://storage.example.com/audio.mp3",
        source_text="The text that generated the audio",
        audio_id="slide_001",
    )

    if result.should_regenerate:
        # Regenerate the voiceover
        pass
    else:
        # Audio is acceptable
        pass

Features:
    - Automatic retry on failure
    - Graceful degradation when service is unavailable
    - Circuit breaker pattern
    - Configurable timeouts
"""

import os
import asyncio
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, timedelta
import httpx


@dataclass
class VQVAnalysisResult:
    """Result from VQV-HALLU analysis"""
    audio_id: str
    status: str  # "success", "failed", "skipped", "disabled", "unavailable"

    # Scores (may be None if service unavailable)
    final_score: Optional[float] = None
    acoustic_score: Optional[float] = None
    linguistic_score: Optional[float] = None
    semantic_score: Optional[float] = None

    # Verdict
    is_acceptable: bool = True  # Default to True for graceful degradation
    recommended_action: str = "accept"
    primary_issues: List[str] = None

    # Metadata
    processing_time_ms: Optional[int] = None
    service_available: bool = True
    message: Optional[str] = None

    def __post_init__(self):
        if self.primary_issues is None:
            self.primary_issues = []

    @property
    def should_regenerate(self) -> bool:
        """Returns True if the audio should be regenerated"""
        return self.recommended_action in ("regenerate", "manual_review")

    @property
    def should_accept(self) -> bool:
        """Returns True if the audio is acceptable"""
        return self.is_acceptable or self.recommended_action == "accept"


class CircuitBreaker:
    """Simple circuit breaker to avoid hammering failed service"""

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    def record_success(self):
        """Record a successful call"""
        self.failures = 0
        self.state = "closed"

    def record_failure(self):
        """Record a failed call"""
        self.failures += 1
        self.last_failure_time = datetime.utcnow()

        if self.failures >= self.failure_threshold:
            self.state = "open"

    def can_execute(self) -> bool:
        """Check if we can execute a call"""
        if self.state == "closed":
            return True

        if self.state == "open":
            # Check if we should try again (half-open)
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.reset_timeout:
                    self.state = "half-open"
                    return True
            return False

        # half-open state
        return True


class VQVHalluClient:
    """
    Client for VQV-HALLU hallucination detection service.

    Features:
    - Graceful degradation: If service is unavailable, returns acceptable=True
    - Circuit breaker: Avoids hammering a failed service
    - Configurable retry and timeout
    - Feature flag support
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        enabled: Optional[bool] = None,
        timeout: int = 300,
        max_retries: int = 2,
        min_acceptable_score: float = 70.0,
    ):
        """
        Initialize the VQV-HALLU client.

        Args:
            base_url: URL of the VQV-HALLU service (default: from VQV_HALLU_URL env)
            enabled: Whether to use the service (default: from VQV_HALLU_ENABLED env)
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
            min_acceptable_score: Minimum score to consider acceptable
        """
        self.base_url = base_url or os.getenv("VQV_HALLU_URL", "http://vqv-hallu:8008")

        # Feature flag
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.getenv("VQV_HALLU_ENABLED", "true").lower() == "true"

        self.timeout = timeout
        self.max_retries = max_retries
        self.min_acceptable_score = min_acceptable_score

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.getenv("VQV_CIRCUIT_BREAKER_THRESHOLD", "5")),
            reset_timeout=int(os.getenv("VQV_CIRCUIT_BREAKER_RESET", "60")),
        )

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.skipped_requests = 0

    async def analyze(
        self,
        source_text: str,
        audio_url: Optional[str] = None,
        audio_path: Optional[str] = None,
        audio_id: str = "default",
        content_type: str = "technical_course",
        language: str = "fr",
    ) -> VQVAnalysisResult:
        """
        Analyze a voiceover for hallucinations.

        Args:
            source_text: Original text that generated the audio
            audio_url: URL to the audio file
            audio_path: Local path to the audio file
            audio_id: Unique identifier for tracking
            content_type: Type of content (technical_course, narrative, etc.)
            language: Expected language

        Returns:
            VQVAnalysisResult with analysis results or graceful fallback
        """
        self.total_requests += 1

        # Check if service is enabled
        if not self.enabled:
            self.skipped_requests += 1
            return VQVAnalysisResult(
                audio_id=audio_id,
                status="disabled",
                is_acceptable=True,
                recommended_action="accept",
                service_available=False,
                message="VQV-HALLU validation disabled by configuration",
            )

        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            self.skipped_requests += 1
            return VQVAnalysisResult(
                audio_id=audio_id,
                status="circuit_open",
                is_acceptable=True,
                recommended_action="accept",
                service_available=False,
                message="VQV-HALLU service temporarily unavailable (circuit breaker open)",
            )

        # Try to call the service
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._call_service(
                    source_text=source_text,
                    audio_url=audio_url,
                    audio_path=audio_path,
                    audio_id=audio_id,
                    content_type=content_type,
                    language=language,
                )

                self.circuit_breaker.record_success()
                self.successful_requests += 1

                return result

            except Exception as e:
                print(f"[VQV-CLIENT] Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}", flush=True)

                if attempt < self.max_retries:
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(2 ** attempt)
                    continue

                # All retries failed
                self.circuit_breaker.record_failure()
                self.failed_requests += 1

                # Graceful degradation: accept the audio
                return VQVAnalysisResult(
                    audio_id=audio_id,
                    status="unavailable",
                    is_acceptable=True,
                    recommended_action="accept",
                    primary_issues=[f"VQV-HALLU service error: {str(e)}"],
                    service_available=False,
                    message=f"VQV-HALLU unavailable after {self.max_retries + 1} attempts. Audio accepted by default.",
                )

    async def _call_service(
        self,
        source_text: str,
        audio_url: Optional[str],
        audio_path: Optional[str],
        audio_id: str,
        content_type: str,
        language: str,
    ) -> VQVAnalysisResult:
        """Actually call the VQV-HALLU service"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/analyze",
                json={
                    "audio_url": audio_url,
                    "audio_path": audio_path,
                    "source_text": source_text,
                    "audio_id": audio_id,
                    "content_type": content_type,
                    "language": language,
                }
            )

            response.raise_for_status()
            data = response.json()

            return VQVAnalysisResult(
                audio_id=data.get("audio_id", audio_id),
                status=data.get("status", "unknown"),
                final_score=data.get("final_score"),
                acoustic_score=data.get("acoustic_score"),
                linguistic_score=data.get("linguistic_score"),
                semantic_score=data.get("semantic_score"),
                is_acceptable=data.get("is_acceptable", True),
                recommended_action=data.get("recommended_action", "accept"),
                primary_issues=data.get("primary_issues", []),
                processing_time_ms=data.get("processing_time_ms"),
                service_available=True,
                message=data.get("message"),
            )

    async def check_health(self) -> dict:
        """Check the health of the VQV-HALLU service"""
        if not self.enabled:
            return {"status": "disabled", "service_available": False}

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {
                "status": "unavailable",
                "service_available": False,
                "error": str(e),
            }

    def get_statistics(self) -> dict:
        """Get client statistics"""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "skipped_requests": self.skipped_requests,
            "circuit_breaker_state": self.circuit_breaker.state,
            "enabled": self.enabled,
        }


# Singleton instance for easy import
_default_client: Optional[VQVHalluClient] = None


def get_vqv_client() -> VQVHalluClient:
    """Get the default VQV-HALLU client instance"""
    global _default_client
    if _default_client is None:
        _default_client = VQVHalluClient()
    return _default_client


async def validate_voiceover(
    source_text: str,
    audio_url: Optional[str] = None,
    audio_path: Optional[str] = None,
    audio_id: str = "default",
    content_type: str = "technical_course",
    language: str = "fr",
) -> VQVAnalysisResult:
    """
    Convenience function to validate a voiceover.

    This is the recommended way to call VQV-HALLU from other services.

    Example:
        from vqv_hallu_client import validate_voiceover

        result = await validate_voiceover(
            source_text="Bienvenue dans ce cours",
            audio_url="https://storage.example.com/audio.mp3",
            audio_id="slide_001",
        )

        if result.should_regenerate:
            print(f"Should regenerate: {result.primary_issues}")
    """
    client = get_vqv_client()
    return await client.analyze(
        source_text=source_text,
        audio_url=audio_url,
        audio_path=audio_path,
        audio_id=audio_id,
        content_type=content_type,
        language=language,
    )
