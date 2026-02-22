"""
Diagram Detector Service
Uses GPT-4 to analyze content and detect if a diagram would enhance understanding.
"""

import os
import json
from typing import Optional, Dict, Any
from openai import AsyncOpenAI

# Use shared LLM provider for model name resolution
try:
    from shared.llm_provider import get_model_name as _get_model_name
    _HAS_SHARED_LLM = True
except ImportError:
    _HAS_SHARED_LLM = False
    _get_model_name = lambda tier: "gpt-4o-mini"

from models.visual_models import (
    DiagramType,
    DetectionResult,
)


class DiagramDetector:
    """
    Analyzes slide content to determine if a diagram is needed
    and what type would be most effective.
    """

    # Keywords that strongly suggest diagram need
    DIAGRAM_KEYWORDS = {
        # Architecture & Flow
        "architecture": DiagramType.ARCHITECTURE,
        "flow": DiagramType.FLOWCHART,
        "workflow": DiagramType.FLOWCHART,
        "pipeline": DiagramType.FLOWCHART,
        "process": DiagramType.FLOWCHART,
        "steps": DiagramType.FLOWCHART,

        # Sequence & Time
        "sequence": DiagramType.SEQUENCE,
        "interaction": DiagramType.SEQUENCE,
        "communication": DiagramType.SEQUENCE,
        "request": DiagramType.SEQUENCE,
        "response": DiagramType.SEQUENCE,
        "timeline": DiagramType.TIMELINE,
        "history": DiagramType.TIMELINE,
        "evolution": DiagramType.TIMELINE,

        # Structure
        "class": DiagramType.CLASS_DIAGRAM,
        "inheritance": DiagramType.CLASS_DIAGRAM,
        "interface": DiagramType.CLASS_DIAGRAM,
        "relationship": DiagramType.ER_DIAGRAM,
        "entity": DiagramType.ER_DIAGRAM,
        "database": DiagramType.ER_DIAGRAM,
        "schema": DiagramType.ER_DIAGRAM,
        "state": DiagramType.STATE_DIAGRAM,
        "transition": DiagramType.STATE_DIAGRAM,

        # Data Visualization
        "percentage": DiagramType.PIE_CHART,
        "distribution": DiagramType.PIE_CHART,
        "trend": DiagramType.LINE_CHART,
        "growth": DiagramType.LINE_CHART,
        "comparison": DiagramType.BAR_CHART,
        "metrics": DiagramType.BAR_CHART,
        "statistics": DiagramType.HISTOGRAM,
        "correlation": DiagramType.SCATTER_PLOT,

        # Algorithms & Data Structures
        "algorithm": DiagramType.ALGORITHM,
        "sort": DiagramType.ALGORITHM,
        "search": DiagramType.ALGORITHM,
        "tree": DiagramType.DATA_STRUCTURE,
        "graph": DiagramType.GRAPH_THEORY,
        "linked list": DiagramType.DATA_STRUCTURE,
        "stack": DiagramType.DATA_STRUCTURE,
        "queue": DiagramType.DATA_STRUCTURE,
        "hash": DiagramType.DATA_STRUCTURE,

        # Math
        "formula": DiagramType.MATH_FORMULA,
        "equation": DiagramType.MATH_FORMULA,
        "mathematical": DiagramType.MATH_FORMULA,
        "transform": DiagramType.TRANSFORMATION,
    }

    # Slide types that typically need diagrams
    DIAGRAM_SLIDE_TYPES = {
        "concept",
        "architecture",
        "theory",
        "explanation",
        "visualization",
        "demo",
    }

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the detector with OpenAI client."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None

    async def detect(
        self,
        content: str,
        slide_type: Optional[str] = None,
        lesson_context: Optional[str] = None,
        use_ai: bool = True
    ) -> DetectionResult:
        """
        Analyze content and detect if a diagram is needed.

        Args:
            content: The slide content or description
            slide_type: Type of slide (concept, code, etc.)
            lesson_context: Additional context from the lesson
            use_ai: Whether to use GPT-4 for intelligent detection

        Returns:
            DetectionResult with diagram recommendations
        """
        # Quick heuristic check first
        heuristic_result = self._heuristic_detection(content, slide_type)

        # If high confidence from heuristics, skip AI
        if heuristic_result.confidence >= 0.9:
            return heuristic_result

        # Use AI for nuanced detection
        if use_ai and self.client:
            try:
                ai_result = await self._ai_detection(content, slide_type, lesson_context)
                # Merge with heuristic insights
                return self._merge_results(heuristic_result, ai_result)
            except Exception as e:
                print(f"[DiagramDetector] AI detection failed: {e}, falling back to heuristics", flush=True)
                return heuristic_result

        return heuristic_result

    def _heuristic_detection(
        self,
        content: str,
        slide_type: Optional[str] = None
    ) -> DetectionResult:
        """
        Quick keyword-based detection.
        """
        content_lower = content.lower()
        detected_types: Dict[DiagramType, float] = {}

        # Check keywords
        for keyword, diagram_type in self.DIAGRAM_KEYWORDS.items():
            if keyword in content_lower:
                current_score = detected_types.get(diagram_type, 0)
                detected_types[diagram_type] = min(current_score + 0.2, 1.0)

        # Boost score if slide type suggests diagram
        if slide_type and slide_type.lower() in self.DIAGRAM_SLIDE_TYPES:
            for dtype in detected_types:
                detected_types[dtype] = min(detected_types[dtype] + 0.2, 1.0)

        # Check for explicit diagram mentions
        explicit_mentions = [
            "diagram", "chart", "graph", "visualization",
            "illustrate", "show", "display", "visualize"
        ]
        has_explicit = any(m in content_lower for m in explicit_mentions)

        if not detected_types:
            return DetectionResult(
                needs_diagram=has_explicit,
                confidence=0.5 if has_explicit else 0.2,
                suggested_type=DiagramType.FLOWCHART if has_explicit else None,
                suggested_description=None,
                reasoning="No strong indicators found" if not has_explicit else "Explicit visualization request detected"
            )

        # Get best match
        best_type = max(detected_types, key=detected_types.get)
        confidence = detected_types[best_type]

        # Build additional suggestions
        additional = [
            {"type": t.value, "confidence": s}
            for t, s in detected_types.items()
            if t != best_type and s >= 0.3
        ]

        return DetectionResult(
            needs_diagram=confidence >= 0.4,
            confidence=confidence,
            suggested_type=best_type,
            suggested_description=f"A {best_type.value} diagram showing the {best_type.value} described in the content",
            reasoning=f"Detected keywords suggesting {best_type.value}",
            additional_suggestions=additional
        )

    async def _ai_detection(
        self,
        content: str,
        slide_type: Optional[str] = None,
        lesson_context: Optional[str] = None
    ) -> DetectionResult:
        """
        Use GPT-4 for intelligent diagram detection.
        """
        system_prompt = """You are a visual content analyzer for educational videos.
Your job is to determine if a piece of content would benefit from a diagram or visualization.

Analyze the content and respond with a JSON object:
{
    "needs_diagram": true/false,
    "confidence": 0.0-1.0,
    "suggested_type": "flowchart|sequence|class|state|er|architecture|line_chart|bar_chart|pie|algorithm|data_structure|math|animation|null",
    "description": "Brief description of the diagram to create",
    "reasoning": "Why this diagram type is appropriate",
    "additional_suggestions": [{"type": "...", "confidence": 0.0-1.0, "description": "..."}]
}

Diagram types explained:
- flowchart: Process flows, decision trees, workflows
- sequence: API calls, message passing, interactions between components
- class: OOP class hierarchies, interfaces, relationships
- state: State machines, status transitions
- er: Database schemas, entity relationships
- architecture: System architecture, microservices, infrastructure
- line_chart: Trends over time, performance metrics
- bar_chart: Comparisons between categories
- pie: Distribution, percentages
- algorithm: Sorting, searching, step-by-step algorithms
- data_structure: Trees, graphs, linked lists, arrays
- math: Mathematical formulas, equations
- animation: Complex concepts needing animated explanation

Consider:
1. Would a visual SIGNIFICANTLY improve understanding?
2. Is the concept inherently visual (architecture, flow, structure)?
3. Are there relationships between components that need visualization?
4. Would static image suffice, or is animation needed?"""

        user_content = f"Content: {content}"
        if slide_type:
            user_content += f"\nSlide type: {slide_type}"
        if lesson_context:
            user_content += f"\nLesson context: {lesson_context[:500]}"

        response = await self.client.chat.completions.create(
            model=_get_model_name("fast"),  # Fast and cost-effective for classification
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500
        )

        result_json = json.loads(response.choices[0].message.content)

        # Map string type to enum
        suggested_type = None
        if result_json.get("suggested_type"):
            try:
                type_map = {
                    "flowchart": DiagramType.FLOWCHART,
                    "sequence": DiagramType.SEQUENCE,
                    "class": DiagramType.CLASS_DIAGRAM,
                    "state": DiagramType.STATE_DIAGRAM,
                    "er": DiagramType.ER_DIAGRAM,
                    "architecture": DiagramType.ARCHITECTURE,
                    "line_chart": DiagramType.LINE_CHART,
                    "bar_chart": DiagramType.BAR_CHART,
                    "pie": DiagramType.PIE_CHART,
                    "algorithm": DiagramType.ALGORITHM,
                    "data_structure": DiagramType.DATA_STRUCTURE,
                    "math": DiagramType.MATH_FORMULA,
                    "animation": DiagramType.ANIMATION,
                    "mindmap": DiagramType.MINDMAP,
                    "timeline": DiagramType.TIMELINE,
                }
                suggested_type = type_map.get(result_json["suggested_type"].lower())
            except (KeyError, AttributeError):
                pass

        return DetectionResult(
            needs_diagram=result_json.get("needs_diagram", False),
            confidence=float(result_json.get("confidence", 0.5)),
            suggested_type=suggested_type,
            suggested_description=result_json.get("description"),
            reasoning=result_json.get("reasoning"),
            additional_suggestions=result_json.get("additional_suggestions", [])
        )

    def _merge_results(
        self,
        heuristic: DetectionResult,
        ai: DetectionResult
    ) -> DetectionResult:
        """
        Merge heuristic and AI detection results.
        AI result takes precedence but heuristics can boost confidence.
        """
        # If both agree on needing a diagram, boost confidence
        if heuristic.needs_diagram and ai.needs_diagram:
            merged_confidence = min(ai.confidence + 0.1, 1.0)
        else:
            merged_confidence = ai.confidence

        # Use AI's type if available, otherwise fall back to heuristic
        final_type = ai.suggested_type or heuristic.suggested_type

        # Combine additional suggestions
        all_suggestions = ai.additional_suggestions.copy()
        for h_sug in heuristic.additional_suggestions:
            if not any(s.get("type") == h_sug.get("type") for s in all_suggestions):
                all_suggestions.append(h_sug)

        return DetectionResult(
            needs_diagram=ai.needs_diagram,
            confidence=merged_confidence,
            suggested_type=final_type,
            suggested_description=ai.suggested_description or heuristic.suggested_description,
            reasoning=ai.reasoning or heuristic.reasoning,
            additional_suggestions=all_suggestions
        )

    def detect_sync(
        self,
        content: str,
        slide_type: Optional[str] = None
    ) -> DetectionResult:
        """
        Synchronous version using only heuristics.
        Useful when async is not available.
        """
        return self._heuristic_detection(content, slide_type)
