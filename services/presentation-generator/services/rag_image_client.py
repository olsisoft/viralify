"""
RAG Image Client

Client for handling RAG images extracted from documents.
Supports both local file access (shared volume) and HTTP download.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import httpx

# Default paths for RAG images
RAG_IMAGES_BASE_PATH = os.getenv("RAG_IMAGES_PATH", "/tmp/viralify/documents")
COURSE_GENERATOR_URL = os.getenv("COURSE_GENERATOR_URL", "http://course-generator:8007")

# Minimum relevance score to use a RAG image
RAG_IMAGE_MIN_SCORE = float(os.getenv("RAG_IMAGE_MIN_SCORE", "0.7"))

# Image types suitable for diagram slides
DIAGRAM_IMAGE_TYPES = ["diagram", "chart", "architecture", "flowchart", "schema"]


class RAGImageClient:
    """
    Client for fetching and managing RAG images.

    Images extracted from PDFs are stored in course-generator's file system.
    This client handles:
    1. Finding images by their file paths
    2. Copying images to the local output directory
    3. Filtering images by relevance and type
    """

    def __init__(self, output_dir: str = "/tmp/presentations"):
        """
        Initialize the RAG image client.

        Args:
            output_dir: Directory where presentation files are stored
        """
        self.output_dir = output_dir
        self.images_cache: dict = {}

    def find_matching_image(
        self,
        slide_topic: str,
        rag_images: List[dict],
        min_score: float = RAG_IMAGE_MIN_SCORE,
        image_types: List[str] = None,
    ) -> Optional[dict]:
        """
        Find the best matching RAG image for a slide topic.

        Args:
            slide_topic: The topic/title of the slide
            rag_images: List of RAG image references
            min_score: Minimum relevance score to consider
            image_types: Filter by image types (default: diagram types)

        Returns:
            Best matching image dict or None
        """
        if not rag_images:
            return None

        # Default to diagram-suitable types
        allowed_types = image_types or DIAGRAM_IMAGE_TYPES

        # Filter candidates
        candidates = []
        slide_topic_lower = slide_topic.lower()

        for img in rag_images:
            # Check relevance score
            score = img.get("relevance_score", 0)
            if score < min_score:
                continue

            # Check image type
            img_type = img.get("detected_type", "unknown")
            if img_type not in allowed_types:
                continue

            # Check if topic matches (additional local check)
            context = (img.get("context_text") or "").lower()
            caption = (img.get("caption") or "").lower()
            lecture_title = (img.get("lecture_title") or "").lower()

            # Boost score if slide topic words appear in image context
            topic_words = [w for w in slide_topic_lower.split() if len(w) > 3]
            topic_match_bonus = 0
            for word in topic_words:
                if word in context or word in caption:
                    topic_match_bonus += 0.05

            adjusted_score = min(1.0, score + topic_match_bonus)

            candidates.append({
                **img,
                "adjusted_score": adjusted_score,
            })

        if not candidates:
            return None

        # Sort by adjusted score and return best match
        candidates.sort(key=lambda x: x["adjusted_score"], reverse=True)
        best = candidates[0]

        print(f"[RAG_IMAGE] Found matching image for '{slide_topic[:30]}...' "
              f"(type={best['detected_type']}, score={best['adjusted_score']:.2f})", flush=True)

        return best

    def get_image_path(self, image: dict, job_id: str) -> Optional[str]:
        """
        Get the local path to a RAG image, copying it if necessary.

        Args:
            image: RAG image reference dict
            job_id: Current job ID for organizing output files

        Returns:
            Local path to the image file, or None if not accessible
        """
        source_path = image.get("file_path")
        if not source_path:
            print(f"[RAG_IMAGE] No file_path in image reference", flush=True)
            return None

        # Check if source file exists directly (shared volume)
        if os.path.exists(source_path):
            # Copy to job output directory for consistency
            return self._copy_to_output(source_path, job_id, image)

        # Try constructing path from components
        document_id = image.get("document_id", "")
        file_name = image.get("file_name", os.path.basename(source_path))

        # Try alternative paths
        alternative_paths = [
            source_path,
            os.path.join(RAG_IMAGES_BASE_PATH, file_name),
            os.path.join(RAG_IMAGES_BASE_PATH, document_id, file_name),
        ]

        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                return self._copy_to_output(alt_path, job_id, image)

        print(f"[RAG_IMAGE] Image not found locally: {source_path}", flush=True)
        return None

    def _copy_to_output(self, source_path: str, job_id: str, image: dict) -> Optional[str]:
        """
        Copy an image to the job output directory.

        Args:
            source_path: Source image file path
            job_id: Job ID for output directory
            image: Image reference for naming

        Returns:
            Path to copied image
        """
        try:
            # Create output directory
            output_subdir = os.path.join(self.output_dir, job_id, "rag_images")
            os.makedirs(output_subdir, exist_ok=True)

            # Generate unique filename
            image_id = image.get("image_id", "img")
            ext = os.path.splitext(source_path)[1] or ".png"
            dest_filename = f"rag_{image_id}{ext}"
            dest_path = os.path.join(output_subdir, dest_filename)

            # Copy file
            shutil.copy2(source_path, dest_path)

            print(f"[RAG_IMAGE] Copied image to: {dest_path}", flush=True)
            return dest_path

        except Exception as e:
            print(f"[RAG_IMAGE] Failed to copy image: {e}", flush=True)
            return None

    def filter_images_for_slides(
        self,
        rag_images: List[dict],
        slides: List[dict],
        min_score: float = RAG_IMAGE_MIN_SCORE,
    ) -> dict:
        """
        Create a mapping of slide indices to matching RAG images.

        Args:
            rag_images: All available RAG images
            slides: List of slide definitions (with title/topic)
            min_score: Minimum relevance score

        Returns:
            Dict mapping slide index to matching image
        """
        slide_images = {}

        for i, slide in enumerate(slides):
            slide_type = slide.get("type", "")

            # Only match RAG images for diagram slides
            if slide_type not in ["diagram", "DIAGRAM"]:
                continue

            title = slide.get("title", "")
            matching_image = self.find_matching_image(
                slide_topic=title,
                rag_images=rag_images,
                min_score=min_score,
            )

            if matching_image:
                slide_images[i] = matching_image

        return slide_images


# Singleton instance
_rag_image_client: Optional[RAGImageClient] = None


def get_rag_image_client(output_dir: str = "/tmp/presentations") -> RAGImageClient:
    """Get or create the RAG image client singleton."""
    global _rag_image_client
    if _rag_image_client is None:
        _rag_image_client = RAGImageClient(output_dir=output_dir)
    return _rag_image_client
