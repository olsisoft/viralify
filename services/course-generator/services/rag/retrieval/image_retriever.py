"""
Image Retriever for RAG

Retrieves relevant images from documents for use in diagram slides.
Scores images based on relevance to topics.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class RAGImage:
    """Image extracted from a document."""
    image_id: str
    document_id: str
    filename: str
    file_path: str
    image_type: str  # diagram, chart, photo, screenshot, etc.
    context: str  # Surrounding text
    caption: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    page_number: Optional[int] = None
    relevance_score: float = 0.0


@dataclass
class ImageRetrievalResult:
    """Result of image retrieval for a topic."""
    topic: str
    images: List[RAGImage]
    total_available: int
    threshold_used: float


class ImageRetriever:
    """
    Retrieve relevant images from documents for RAG.

    Scores images based on:
    - Context similarity to topic (40%)
    - Caption match (25%)
    - Description match (20%)
    - Keyword overlap (15%)

    Usage:
        retriever = ImageRetriever()
        result = retriever.get_images_for_topic(
            topic="Apache Kafka architecture",
            images=extracted_images,
            min_score=0.7
        )
    """

    # Scoring weights
    WEIGHT_CONTEXT = 0.40
    WEIGHT_CAPTION = 0.25
    WEIGHT_DESCRIPTION = 0.20
    WEIGHT_KEYWORDS = 0.15

    # Default minimum score for image inclusion
    DEFAULT_MIN_SCORE = 0.7

    # Image types preferred for diagrams
    DIAGRAM_TYPES = {'diagram', 'chart', 'architecture', 'flowchart', 'schema', 'graph'}

    def __init__(self, keyword_extractor=None):
        """
        Initialize the image retriever.

        Args:
            keyword_extractor: KeywordExtractor for text analysis
        """
        self.keyword_extractor = keyword_extractor

    def _get_keyword_extractor(self):
        """Get or create keyword extractor."""
        if self.keyword_extractor is None:
            from ..algorithms.keyword_extractor import get_keyword_extractor
            self.keyword_extractor = get_keyword_extractor()
        return self.keyword_extractor

    def get_images_for_topic(
        self,
        topic: str,
        images: List[RAGImage],
        min_score: float = None,
        max_images: int = 5,
        prefer_diagrams: bool = True,
    ) -> ImageRetrievalResult:
        """
        Get relevant images for a topic.

        Args:
            topic: Topic to match images against
            images: List of RAGImage to search
            min_score: Minimum relevance score (0-1)
            max_images: Maximum images to return
            prefer_diagrams: Boost diagram-type images

        Returns:
            ImageRetrievalResult with scored images
        """
        if min_score is None:
            min_score = self.DEFAULT_MIN_SCORE

        if not images or not topic:
            return ImageRetrievalResult(
                topic=topic,
                images=[],
                total_available=len(images) if images else 0,
                threshold_used=min_score,
            )

        extractor = self._get_keyword_extractor()
        topic_keywords = set(extractor.extract(topic))

        scored_images = []

        for image in images:
            score = self._score_image(image, topic, topic_keywords, extractor)

            # Boost diagram types if preferred
            if prefer_diagrams and image.image_type.lower() in self.DIAGRAM_TYPES:
                score = min(1.0, score + 0.1)

            if score >= min_score:
                image.relevance_score = score
                scored_images.append(image)

        # Sort by relevance score
        scored_images.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit results
        top_images = scored_images[:max_images]

        return ImageRetrievalResult(
            topic=topic,
            images=top_images,
            total_available=len(images),
            threshold_used=min_score,
        )

    def _score_image(
        self,
        image: RAGImage,
        topic: str,
        topic_keywords: set,
        extractor,
    ) -> float:
        """
        Score an image's relevance to a topic.

        Args:
            image: RAGImage to score
            topic: Topic text
            topic_keywords: Pre-extracted topic keywords
            extractor: KeywordExtractor instance

        Returns:
            Relevance score (0-1)
        """
        scores = []

        # Context score (40%)
        if image.context:
            context_score = extractor.compute_coverage(topic, image.context)
            scores.append(context_score * self.WEIGHT_CONTEXT)
        else:
            scores.append(0.0)

        # Caption score (25%)
        if image.caption:
            caption_score = extractor.compute_coverage(topic, image.caption)
            scores.append(caption_score * self.WEIGHT_CAPTION)
        else:
            scores.append(0.0)

        # Description score (20%)
        if image.description:
            desc_score = extractor.compute_coverage(topic, image.description)
            scores.append(desc_score * self.WEIGHT_DESCRIPTION)
        else:
            scores.append(0.0)

        # Keyword overlap score (15%)
        if image.keywords:
            image_keywords = set(kw.lower() for kw in image.keywords)
            topic_kw_lower = set(kw.lower() for kw in topic_keywords)
            if topic_kw_lower:
                overlap = len(image_keywords.intersection(topic_kw_lower))
                kw_score = overlap / len(topic_kw_lower)
                scores.append(kw_score * self.WEIGHT_KEYWORDS)
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        return sum(scores)

    def filter_by_type(
        self,
        images: List[RAGImage],
        allowed_types: List[str] = None,
    ) -> List[RAGImage]:
        """
        Filter images by type.

        Args:
            images: List of images
            allowed_types: Allowed image types (uses DIAGRAM_TYPES if None)

        Returns:
            Filtered list
        """
        if allowed_types is None:
            allowed_types = self.DIAGRAM_TYPES

        allowed_lower = {t.lower() for t in allowed_types}
        return [img for img in images if img.image_type.lower() in allowed_lower]

    def get_best_diagram(
        self,
        topic: str,
        images: List[RAGImage],
        min_score: float = 0.7,
    ) -> Optional[RAGImage]:
        """
        Get the single best diagram for a topic.

        Args:
            topic: Topic to match
            images: Available images
            min_score: Minimum relevance score

        Returns:
            Best matching RAGImage or None
        """
        result = self.get_images_for_topic(
            topic=topic,
            images=images,
            min_score=min_score,
            max_images=1,
            prefer_diagrams=True,
        )

        return result.images[0] if result.images else None


# Module-level instance
_default_retriever = None


def get_image_retriever() -> ImageRetriever:
    """Get the default image retriever instance."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = ImageRetriever()
    return _default_retriever


def get_images_for_topic(
    topic: str,
    images: List[RAGImage],
    min_score: float = 0.7,
) -> List[RAGImage]:
    """
    Convenience function to get relevant images.

    Args:
        topic: Topic to match
        images: Available images
        min_score: Minimum score threshold

    Returns:
        List of relevant RAGImage
    """
    return get_image_retriever().get_images_for_topic(topic, images, min_score).images
