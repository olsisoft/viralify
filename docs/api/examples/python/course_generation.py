"""
Viralify API - Course Generation Example (Python)

This example demonstrates how to generate a complete video course
using the Viralify API.

Requirements:
    pip install requests python-dotenv

Usage:
    export VIRALIFY_API_KEY="your_api_key"
    python course_generation.py
"""

import os
import time
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass


# Configuration
API_KEY = os.getenv("VIRALIFY_API_KEY")
BASE_URL = os.getenv("VIRALIFY_BASE_URL", "https://api.viralify.io")


@dataclass
class CourseConfig:
    """Configuration for course generation."""
    topic: str
    difficulty_start: str = "beginner"
    difficulty_end: str = "intermediate"
    num_sections: int = 4
    lectures_per_section: int = 3
    category: str = "tech"
    language: str = "en"
    quiz_enabled: bool = True
    quiz_frequency: str = "per_section"
    title_style: str = "engaging"


class ViralifyClient:
    """Simple Viralify API client."""

    def __init__(self, api_key: str, base_url: str = "https://api.viralify.io"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an API request."""
        response = requests.request(
            method,
            f"{self.base_url}{endpoint}",
            headers=self.headers,
            timeout=60,
            **kwargs
        )
        response.raise_for_status()
        return response.json()

    def preview_outline(self, config: CourseConfig) -> Dict[str, Any]:
        """Preview course outline before generation."""
        return self._request("POST", "/api/v1/courses/preview-outline", json={
            "topic": config.topic,
            "difficulty_start": config.difficulty_start,
            "difficulty_end": config.difficulty_end,
            "structure": {
                "number_of_sections": config.num_sections,
                "lectures_per_section": config.lectures_per_section
            },
            "context": {
                "category": config.category
            }
        })

    def generate_course(self, config: CourseConfig, document_ids: Optional[list] = None) -> str:
        """Start course generation and return job ID."""
        payload = {
            "topic": config.topic,
            "difficulty_start": config.difficulty_start,
            "difficulty_end": config.difficulty_end,
            "structure": {
                "number_of_sections": config.num_sections,
                "lectures_per_section": config.lectures_per_section
            },
            "context": {
                "category": config.category
            },
            "language": config.language,
            "quiz_config": {
                "enabled": config.quiz_enabled,
                "frequency": config.quiz_frequency,
                "questions_per_quiz": 5,
                "passing_score": 70
            },
            "title_style": config.title_style
        }

        if document_ids:
            payload["document_ids"] = document_ids

        result = self._request("POST", "/api/v1/courses/generate", json=payload)
        return result["job_id"]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get course generation job status."""
        return self._request("GET", f"/api/v1/courses/jobs/{job_id}")

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 15,
        timeout: int = 3600,
        callback=None
    ) -> Dict[str, Any]:
        """Wait for job completion with polling."""
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            status = self.get_job_status(job_id)

            if callback:
                callback(status)

            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")

            time.sleep(poll_interval)

    def upload_document(self, file_path: str, user_id: str, role: str = "auto") -> str:
        """Upload a document for RAG."""
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/api/v1/documents/upload",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": f},
                data={"user_id": user_id, "pedagogical_role": role}
            )
            response.raise_for_status()
            return response.json()["id"]


def progress_callback(status: Dict[str, Any]):
    """Print progress updates."""
    progress = status.get("progress", 0)
    stage = status.get("status", "unknown")
    print(f"[{stage.upper()}] Progress: {progress:.1f}%")


def main():
    """Main example function."""
    if not API_KEY:
        print("Error: VIRALIFY_API_KEY environment variable not set")
        return

    client = ViralifyClient(API_KEY, BASE_URL)

    # Course configuration
    config = CourseConfig(
        topic="Introduction to Docker and Containerization",
        difficulty_start="beginner",
        difficulty_end="intermediate",
        num_sections=4,
        lectures_per_section=3,
        category="tech",
        language="en",
        quiz_enabled=True,
        quiz_frequency="per_section",
        title_style="engaging"
    )

    print(f"=== Generating Course: {config.topic} ===\n")

    # Step 1: Preview outline
    print("Step 1: Previewing outline...")
    outline = client.preview_outline(config)
    print(f"  Title: {outline['title']}")
    print(f"  Sections: {outline['section_count']}")
    print(f"  Total Lectures: {outline['total_lectures']}")
    print(f"  Duration: ~{outline.get('estimated_duration_minutes', 'N/A')} minutes")
    print()

    # Step 2: Start generation
    print("Step 2: Starting course generation...")
    job_id = client.generate_course(config)
    print(f"  Job ID: {job_id}")
    print()

    # Step 3: Wait for completion
    print("Step 3: Waiting for completion...")
    try:
        result = client.wait_for_completion(
            job_id,
            poll_interval=15,
            timeout=3600,
            callback=progress_callback
        )

        print("\n=== Course Generation Complete! ===")
        print(f"Videos: {len(result['output_urls']['videos'])} files")
        print(f"ZIP: {result['output_urls']['zip']}")

        # Print video URLs
        print("\nGenerated Videos:")
        for i, url in enumerate(result['output_urls']['videos'], 1):
            print(f"  {i}. {url}")

    except TimeoutError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
