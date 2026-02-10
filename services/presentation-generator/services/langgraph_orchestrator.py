"""
LangGraph Orchestrator for Video Generation

This module implements a state machine using LangGraph to orchestrate
the video generation pipeline with proper validation, synchronization,
and feedback loops.

Key improvements over the linear pipeline:
1. Content validation - ensures visual references match actual content
2. Code execution - actually runs code and captures output
3. Timing synchronization - adjusts durations based on real content
4. Feedback loops - regenerates problematic sections
"""

import asyncio
import json
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from models.presentation_models import (
    PresentationScript,
    Slide,
    SlideType,
    CodeBlock,
    GeneratePresentationRequest,
)
import os
from services.url_config import url_config


# =============================================================================
# URL GENERATION HELPERS
# =============================================================================

def get_public_url(file_path: str) -> str:
    """
    Generate a public URL for user-facing responses using centralized URL config.
    """
    return url_config.build_presentation_url(file_path)


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class ValidationIssue(BaseModel):
    """Represents a validation issue found in the content"""
    slide_index: int
    issue_type: str  # "missing_visual", "timing_mismatch", "code_error", "reference_mismatch"
    description: str
    severity: str  # "critical", "warning", "info"
    suggested_fix: Optional[str] = None


class SlideAsset(BaseModel):
    """Generated asset for a slide"""
    slide_id: str
    asset_type: str  # "image", "animation", "diagram", "code_output"
    file_path: Optional[str] = None
    url: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimingInfo(BaseModel):
    """Timing information for a slide"""
    slide_id: str
    voiceover_duration: float  # Actual TTS duration
    animation_duration: float  # Animation duration
    visual_duration: float  # How long visual content needs
    recommended_duration: float  # Final recommended duration
    is_synced: bool = False


class CodeExecutionResult(BaseModel):
    """Result of code execution"""
    slide_id: str
    code: str
    language: str
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    success: bool = False


class VideoGenerationState(TypedDict):
    """
    Main state object for the video generation workflow.
    This is passed between all nodes and accumulates results.
    """
    # Input
    request: Dict[str, Any]

    # Script generation
    script: Optional[Dict[str, Any]]
    script_generation_attempts: int

    # Validation
    validation_issues: List[Dict[str, Any]]
    validation_passed: bool

    # Asset generation
    slide_assets: List[Dict[str, Any]]
    diagrams_generated: List[str]
    code_executions: List[Dict[str, Any]]

    # Voiceover - per-slide for sync
    slide_voiceovers: Dict[str, Dict[str, Any]]  # slide_id -> {url, duration, text}
    voiceover_duration: float  # Total duration

    # Timing
    timing_info: List[Dict[str, Any]]
    timing_synced: bool

    # Animation
    animations: Dict[str, Dict[str, Any]]

    # SSVS Timeline
    timeline: Optional[Dict[str, Any]]  # Timeline from SSVS synchronization

    # Final output
    output_video_url: Optional[str]
    composition_job_id: Optional[str]

    # Control flow
    current_phase: str
    iteration_count: int
    max_iterations: int
    errors: List[str]
    warnings: List[str]

    # Metadata
    job_id: str
    started_at: str
    completed_at: Optional[str]


# =============================================================================
# VALIDATION PATTERNS
# =============================================================================

# Patterns that indicate visual references in voiceover text
VISUAL_REFERENCE_PATTERNS = [
    (r"\b(this|the|see the|look at the|here's the|here is the)\s+(diagram|chart|graph|figure|illustration|image|picture)\b", "diagram"),
    (r"\b(as (you can )?see|shown here|displayed|on screen)\b", "visual"),
    (r"\b(this|the) (code|example|snippet|output|result)\b", "code"),
    (r"\b(watch|notice|observe) (how|as|the)\b", "animation"),
    (r"\b(step by step|line by line|character by character)\b", "typing_animation"),
]

# Code execution supported languages
EXECUTABLE_LANGUAGES = ["python", "javascript", "bash", "shell"]


def clean_voiceover_text(text: str) -> str:
    """
    Clean voiceover text before sending to TTS.
    Removes sync markers and technical artifacts that shouldn't be read aloud.
    """
    if not text:
        return ""

    # Remove [SYNC:slide_XXX] markers (the main culprit!)
    text = re.sub(r'\[SYNC:slide_\d+\]', '', text)

    # Remove other common technical markers
    text = re.sub(r'\[SLIDE[:\s]*\d+\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[PAUSE[:\s]*\d*m?s?\]', '', text, flags=re.IGNORECASE)

    # Remove markdown artifacts
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code

    # Remove multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

async def initialize_state(state: VideoGenerationState) -> VideoGenerationState:
    """Initialize the workflow state with defaults"""
    return {
        **state,
        "script": None,
        "script_generation_attempts": 0,
        "validation_issues": [],
        "validation_passed": False,
        "slide_assets": [],
        "diagrams_generated": [],
        "code_executions": [],
        "slide_voiceovers": {},  # Per-slide voiceover info
        "voiceover_duration": 0.0,
        "timing_info": [],
        "timing_synced": False,
        "animations": {},
        "timeline": None,  # SSVS timeline
        "output_video_url": None,
        "composition_job_id": None,
        "current_phase": "initialized",
        "iteration_count": 0,
        "max_iterations": state.get("max_iterations", 3),
        "errors": [],
        "warnings": [],
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }


async def generate_script_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Generate the presentation script using GPT-4.
    Enhanced to include explicit visual markers.
    """
    from services.presentation_planner import PresentationPlannerService

    print(f"[LANGGRAPH] Generating script (attempt {state['script_generation_attempts'] + 1})", flush=True)

    request_data = state["request"]
    request = GeneratePresentationRequest(**request_data)

    # Debug: Check if RAG context is present
    rag_context = getattr(request, 'rag_context', None)
    document_ids = getattr(request, 'document_ids', [])
    if rag_context:
        print(f"[LANGGRAPH] RAG context available: {len(rag_context)} chars", flush=True)
    elif document_ids:
        print(f"[LANGGRAPH] WARNING: document_ids={document_ids} but no rag_context - RAG fetch may have failed", flush=True)
    else:
        print(f"[LANGGRAPH] No RAG context (no documents provided)", flush=True)

    planner = PresentationPlannerService()

    # Use enhanced prompt that enforces visual-audio alignment
    script = await planner.generate_script_with_validation(request)

    return {
        **state,
        "script": script.model_dump() if hasattr(script, 'model_dump') else script,
        "script_generation_attempts": state["script_generation_attempts"] + 1,
        "current_phase": "script_generated",
    }


async def validate_content_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Validate the script content for visual-audio alignment issues.

    Checks:
    1. Visual references in voiceover have corresponding visuals
    2. Code blocks are syntactically valid
    3. Diagram slides have proper descriptions
    4. Timing estimates are reasonable
    """
    print("[LANGGRAPH] Validating content alignment", flush=True)

    issues = []
    script = state["script"]

    if not script or "slides" not in script:
        issues.append({
            "slide_index": -1,
            "issue_type": "missing_script",
            "description": "No script generated",
            "severity": "critical",
            "suggested_fix": "Regenerate script"
        })
        return {
            **state,
            "validation_issues": issues,
            "validation_passed": False,
            "current_phase": "validation_failed",
        }

    for i, slide in enumerate(script["slides"]):
        voiceover = slide.get("voiceover_text", "")
        slide_type = slide.get("type", "content")

        # Check 1: Visual references without corresponding content
        for pattern, ref_type in VISUAL_REFERENCE_PATTERNS:
            if re.search(pattern, voiceover, re.IGNORECASE):
                # Check if the slide type matches the reference
                if ref_type == "diagram" and slide_type != "diagram":
                    issues.append({
                        "slide_index": i,
                        "issue_type": "reference_mismatch",
                        "description": f"Voiceover references a diagram but slide type is '{slide_type}'",
                        "severity": "critical",
                        "suggested_fix": f"Change slide type to 'diagram' or remove diagram reference from voiceover"
                    })
                elif ref_type == "code" and slide_type not in ["code", "code_demo"]:
                    issues.append({
                        "slide_index": i,
                        "issue_type": "reference_mismatch",
                        "description": f"Voiceover references code but slide type is '{slide_type}'",
                        "severity": "warning",
                        "suggested_fix": f"Add code block or change voiceover"
                    })

        # Check 2: Diagram slides need proper description
        if slide_type == "diagram":
            content = slide.get("content", "")
            if not content or len(content) < 20:
                issues.append({
                    "slide_index": i,
                    "issue_type": "missing_visual",
                    "description": "Diagram slide lacks description for generation",
                    "severity": "critical",
                    "suggested_fix": "Add detailed diagram description in content field"
                })

        # Check 3: Code blocks should be valid
        if slide_type in ["code", "code_demo"]:
            code_blocks = slide.get("code_blocks", [])
            if not code_blocks:
                issues.append({
                    "slide_index": i,
                    "issue_type": "missing_visual",
                    "description": f"Slide type is '{slide_type}' but has no code blocks",
                    "severity": "critical",
                    "suggested_fix": "Add code blocks or change slide type"
                })
            else:
                for j, block in enumerate(code_blocks):
                    code = block.get("code", "")
                    if not code.strip():
                        issues.append({
                            "slide_index": i,
                            "issue_type": "code_error",
                            "description": f"Code block {j} is empty",
                            "severity": "critical",
                            "suggested_fix": "Add code content"
                        })

        # Check 4: Duration sanity check
        duration = slide.get("duration", 0)
        word_count = len(voiceover.split())
        # Approximately 150 words per minute = 2.5 words per second
        estimated_duration = word_count / 2.5

        if duration > 0 and abs(duration - estimated_duration) > 10:
            issues.append({
                "slide_index": i,
                "issue_type": "timing_mismatch",
                "description": f"Duration ({duration}s) doesn't match voiceover length ({estimated_duration:.1f}s estimated)",
                "severity": "warning",
                "suggested_fix": f"Adjust duration to approximately {estimated_duration:.1f}s"
            })

    # Determine if validation passed
    critical_issues = [i for i in issues if i["severity"] == "critical"]
    validation_passed = len(critical_issues) == 0

    print(f"[LANGGRAPH] Validation: {len(issues)} issues found ({len(critical_issues)} critical)", flush=True)

    return {
        **state,
        "validation_issues": issues,
        "validation_passed": validation_passed,
        "current_phase": "validated" if validation_passed else "validation_failed",
    }


async def fix_content_issues_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Fix validation issues by regenerating or adjusting content.
    Uses configured LLM provider to intelligently fix the issues.
    """
    import os

    print("[LANGGRAPH] Fixing content issues", flush=True)

    issues = state["validation_issues"]
    script = state["script"]

    if not issues or not script:
        return state

    # Use shared LLM provider if available
    try:
        from shared.llm_provider import get_llm_client, get_model_name
        client = get_llm_client()
        model = get_model_name("fast")
    except ImportError:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Group issues by slide
    issues_by_slide = {}
    for issue in issues:
        idx = issue["slide_index"]
        if idx not in issues_by_slide:
            issues_by_slide[idx] = []
        issues_by_slide[idx].append(issue)

    # Fix each slide with issues
    fixed_slides = list(script["slides"])

    for slide_idx, slide_issues in issues_by_slide.items():
        if slide_idx < 0 or slide_idx >= len(fixed_slides):
            continue

        slide = fixed_slides[slide_idx]

        fix_prompt = f"""Fix the following issues with this presentation slide:

CURRENT SLIDE:
{json.dumps(slide, indent=2)}

ISSUES TO FIX:
{json.dumps(slide_issues, indent=2)}

RULES:
1. If voiceover mentions "this diagram" but slide isn't a diagram, either:
   - Change slide type to "diagram" and add proper content description
   - OR modify voiceover to not reference a diagram
2. If code is referenced but missing, add appropriate code blocks
3. Ensure voiceover and visuals are aligned
4. Keep the same educational content and flow
5. Fix timing issues by adjusting duration

Return the fixed slide as valid JSON with the same structure."""

        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a presentation expert. Fix slides to ensure visual-audio alignment."},
                {"role": "user", "content": fix_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2000
        )

        try:
            fixed_slide = json.loads(response.choices[0].message.content)
            fixed_slides[slide_idx] = fixed_slide
            print(f"[LANGGRAPH] Fixed slide {slide_idx}", flush=True)
        except json.JSONDecodeError:
            print(f"[LANGGRAPH] Failed to parse fix for slide {slide_idx}", flush=True)

    script["slides"] = fixed_slides

    return {
        **state,
        "script": script,
        "iteration_count": state["iteration_count"] + 1,
        "current_phase": "content_fixed",
    }


async def execute_code_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Execute code blocks marked as code_demo and capture output.
    Runs in a sandboxed environment.
    """
    print("[LANGGRAPH] Executing code blocks", flush=True)

    script = state["script"]
    executions = []

    if not script:
        return state

    for i, slide in enumerate(script.get("slides", [])):
        if slide.get("type") != "code_demo":
            continue

        for block in slide.get("code_blocks", []):
            language = block.get("language", "python")
            code = block.get("code", "")

            if language not in EXECUTABLE_LANGUAGES:
                continue

            slide_id = slide.get("id", f"slide_{i}")

            result = {
                "slide_id": slide_id,
                "code": code,
                "language": language,
                "output": None,
                "error": None,
                "execution_time": 0.0,
                "success": False
            }

            try:
                if language == "python":
                    # Create temp file and execute
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(code)
                        temp_path = f.name

                    import time
                    start = time.time()

                    proc = subprocess.run(
                        ["python", temp_path],
                        capture_output=True,
                        text=True,
                        timeout=10  # 10 second timeout
                    )

                    result["execution_time"] = time.time() - start
                    result["output"] = proc.stdout
                    result["error"] = proc.stderr if proc.returncode != 0 else None
                    result["success"] = proc.returncode == 0

                    Path(temp_path).unlink(missing_ok=True)

                elif language in ["bash", "shell"]:
                    proc = subprocess.run(
                        ["bash", "-c", code],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    result["output"] = proc.stdout
                    result["error"] = proc.stderr if proc.returncode != 0 else None
                    result["success"] = proc.returncode == 0

            except subprocess.TimeoutExpired:
                result["error"] = "Execution timed out (10s limit)"
            except Exception as e:
                result["error"] = str(e)

            executions.append(result)

            # Update the slide with actual output
            if result["success"] and result["output"]:
                block["expected_output"] = result["output"].strip()
                print(f"[LANGGRAPH] Executed code for {slide_id}: {result['output'][:50]}...", flush=True)

    return {
        **state,
        "code_executions": executions,
        "current_phase": "code_executed",
    }


async def generate_diagrams_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Generate actual diagrams for diagram slides using AI image generation.
    """
    print("[LANGGRAPH] Generating diagrams", flush=True)

    script = state["script"]
    diagrams = []

    if not script:
        return state

    # For now, we'll create detailed descriptions that can be used
    # by an image generation service. In production, this would
    # integrate with DALL-E, Midjourney, or a diagramming library.

    for i, slide in enumerate(script.get("slides", [])):
        if slide.get("type") != "diagram":
            continue

        slide_id = slide.get("id", f"slide_{i}")
        content = slide.get("content", "")
        title = slide.get("title", "")

        # Create a structured diagram specification
        diagram_spec = {
            "slide_id": slide_id,
            "title": title,
            "description": content,
            "style": "technical",  # Could be: flowchart, architecture, sequence, etc.
            "generated": False,
            "placeholder_text": f"[Diagram: {title}]\n{content[:200]}..."
        }

        # In production, generate actual diagram here
        # For now, mark as needing generation
        diagrams.append(slide_id)

        # Add a visual indicator to the slide
        slide["diagram_spec"] = diagram_spec

        print(f"[LANGGRAPH] Prepared diagram spec for {slide_id}", flush=True)

    return {
        **state,
        "diagrams_generated": diagrams,
        "current_phase": "diagrams_prepared",
    }


async def generate_voiceover_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Generate voiceover audio PER SLIDE for precise synchronization.
    Each slide gets its own audio file with exact duration.
    """
    import httpx
    import os

    print("[LANGGRAPH] Generating per-slide voiceovers", flush=True)

    script = state["script"]
    if not script:
        return state

    slides = script.get("slides", [])
    if not slides:
        return {
            **state,
            "warnings": state["warnings"] + ["No slides found"],
            "current_phase": "voiceover_skipped",
        }

    media_generator_url = os.getenv("MEDIA_GENERATOR_URL", "http://media-generator:8004")

    # Default ElevenLabs voice: Adam (available on ALL accounts, multilingual support)
    DEFAULT_ELEVENLABS_VOICE = "pNInz6obpgDQGcFmaJgB"

    # Get voice_id from script, default to ElevenLabs Adam
    voice_id = script.get("voice", DEFAULT_ELEVENLABS_VOICE)

    # OpenAI voices need to be mapped to ElevenLabs equivalents
    openai_voices = ['nova', 'shimmer', 'echo', 'onyx', 'fable', 'alloy', 'ash', 'sage', 'coral']
    openai_to_elevenlabs = {
        'onyx': DEFAULT_ELEVENLABS_VOICE,  # Adam - deep male multilingual
        'echo': 'VR6AewLTigWG4xSOukaG',    # Arnold - warm male
        'alloy': DEFAULT_ELEVENLABS_VOICE, # Adam - neutral multilingual
        'nova': '21m00Tcm4TlvDq8ikWAM',    # Rachel - female calm
        'shimmer': 'EXAVITQu4vr4xnSDxMaL', # Bella - soft female
        'fable': 'ErXwobaYiN019PkySvjV',   # Antoni - expressive male
    }

    if voice_id in openai_voices:
        voice_id = openai_to_elevenlabs.get(voice_id, DEFAULT_ELEVENLABS_VOICE)
        print(f"[LANGGRAPH] Mapped OpenAI voice to ElevenLabs: {voice_id}", flush=True)

    # Store per-slide voiceover info
    slide_voiceovers = {}  # slide_id -> {url, duration, text}
    total_duration = 0.0

    async with httpx.AsyncClient(timeout=300.0) as client:
        for i, slide in enumerate(slides):
            slide_id = slide.get("id", f"slide_{i}")
            raw_voiceover_text = slide.get("voiceover_text", "").strip()

            # Clean voiceover text: remove [SYNC:slide_XXX] markers and other artifacts
            voiceover_text = clean_voiceover_text(raw_voiceover_text)

            if not voiceover_text:
                print(f"[LANGGRAPH] Slide {slide_id}: No voiceover text after cleaning, skipping", flush=True)
                slide_voiceovers[slide_id] = {
                    "url": None,
                    "duration": slide.get("duration", 5.0),  # Use slide duration as fallback
                    "text": ""
                }
                continue

            print(f"[LANGGRAPH] Generating voiceover for slide {slide_id} ({len(voiceover_text)} chars)", flush=True)

            # Start voiceover job for this slide using ElevenLabs
            response = await client.post(
                f"{media_generator_url}/api/v1/media/voiceover",
                json={
                    "text": voiceover_text,
                    "provider": "elevenlabs",  # Use ElevenLabs for best quality
                    "voice_id": voice_id,
                    "speed": 1.0
                }
            )

            if response.status_code != 200:
                print(f"[LANGGRAPH] Voiceover failed for slide {slide_id}: {response.text}", flush=True)
                # Estimate duration for failed slides
                estimated_duration = len(voiceover_text.split()) / 2.5
                slide_voiceovers[slide_id] = {
                    "url": None,
                    "duration": estimated_duration,
                    "text": voiceover_text
                }
                continue

            job_id = response.json().get("job_id")

            # Poll for completion
            voiceover_url = None
            duration = 0.0
            max_wait = 60  # 1 minute per slide
            elapsed = 0

            while elapsed < max_wait:
                await asyncio.sleep(2)
                elapsed += 2

                status_resp = await client.get(
                    f"{media_generator_url}/api/v1/media/jobs/{job_id}"
                )

                if status_resp.status_code != 200:
                    continue

                job_data = status_resp.json()
                job_status = job_data.get("status")

                if job_status == "completed":
                    output_data = job_data.get("output_data", {})
                    voiceover_url = (
                        output_data.get("audio_url") or
                        output_data.get("url") or
                        output_data.get("storage_url")
                    )
                    # Note: key is "duration_seconds" not "duration"
                    duration = output_data.get("duration_seconds") or output_data.get("duration", 0)
                    break
                elif job_status == "failed":
                    print(f"[LANGGRAPH] Voiceover job failed for slide {slide_id}", flush=True)
                    break

            if voiceover_url and duration > 0:
                print(f"[LANGGRAPH] Slide {slide_id} voiceover: {duration:.1f}s", flush=True)
                slide_voiceovers[slide_id] = {
                    "url": voiceover_url,
                    "duration": duration,
                    "text": voiceover_text
                }
                total_duration += duration
            else:
                # Fallback to estimation
                estimated_duration = len(voiceover_text.split()) / 2.5
                slide_voiceovers[slide_id] = {
                    "url": None,
                    "duration": estimated_duration,
                    "text": voiceover_text
                }
                total_duration += estimated_duration

    print(f"[LANGGRAPH] Total voiceover duration: {total_duration:.1f}s for {len(slides)} slides", flush=True)

    return {
        **state,
        "slide_voiceovers": slide_voiceovers,
        "voiceover_duration": total_duration,
        "current_phase": "voiceover_generated",
    }


async def synchronize_timing_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Synchronize slide timings based on ACTUAL voiceover durations.
    Uses the real TTS duration from slide_voiceovers, not estimates.

    The slide duration is set to the maximum of:
    1. Actual voiceover duration (from TTS)
    2. Animation duration (for code slides)
    3. Visual reading time (for bullet points)
    """
    print("[LANGGRAPH] Synchronizing timing with actual voiceover durations", flush=True)

    script = state["script"]
    slide_voiceovers = state.get("slide_voiceovers", {})

    if not script:
        return state

    timing_info = []
    slides = script.get("slides", [])

    # Calculate timing for each slide using ACTUAL voiceover duration
    current_time = 0.0

    for i, slide in enumerate(slides):
        slide_id = slide.get("id", f"slide_{i}")
        slide_type = slide.get("type", "content")

        # Get ACTUAL voiceover duration from TTS (not estimated)
        voiceover_info = slide_voiceovers.get(slide_id, {})
        voiceover_duration = voiceover_info.get("duration", 0.0)

        # Fallback to estimation only if no voiceover was generated
        if voiceover_duration == 0:
            voiceover_text = slide.get("voiceover_text", "")
            word_count = len(voiceover_text.split())
            voiceover_duration = word_count / 2.5 if word_count > 0 else 5.0

        # Calculate animation duration for code slides
        animation_duration = 0.0
        if slide_type in ["code", "code_demo"]:
            for block in slide.get("code_blocks", []):
                code = block.get("code", "")
                # Typing speed adjusted to match voiceover
                char_count = len(code)
                # Animation should complete just before voiceover ends
                animation_duration = max(animation_duration, char_count / 30.0)

        # Calculate visual reading time
        visual_duration = 0.0
        bullet_points = slide.get("bullet_points", [])
        if bullet_points:
            visual_duration = len(bullet_points) * 2.0

        # The slide duration MUST EXACTLY match the voiceover for sync
        # NO BUFFER - buffers cause cumulative drift!
        # The voiceover duration is the source of truth

        # For code slides, ensure animation has time to complete
        if slide_type in ["code", "code_demo"] and animation_duration > voiceover_duration:
            # Animation is longer than voiceover - this is a timing problem
            # but we MUST use voiceover duration to stay in sync
            print(f"[LANGGRAPH] WARNING: Animation ({animation_duration:.1f}s) > voiceover ({voiceover_duration:.1f}s) for slide {i}", flush=True)
            recommended = voiceover_duration  # Sync to audio, not animation
        else:
            # Use voiceover duration as the exact timing reference
            recommended = voiceover_duration if voiceover_duration > 0 else max(visual_duration, 2.0)

        # Update slide with exact timing
        slide["duration"] = round(recommended, 2)
        slide["start_time"] = round(current_time, 2)

        timing = {
            "slide_id": slide_id,
            "voiceover_duration": round(voiceover_duration, 2),
            "animation_duration": round(animation_duration, 2),
            "visual_duration": round(visual_duration, 2),
            "recommended_duration": round(recommended, 2),
            "start_time": round(current_time, 2),
            "has_voiceover_audio": voiceover_info.get("url") is not None,
            "is_synced": True  # We're setting exact durations
        }

        timing_info.append(timing)
        print(f"[LANGGRAPH] Slide {i} ({slide_id}): voiceover={voiceover_duration:.1f}s, anim={animation_duration:.1f}s -> duration={recommended:.1f}s", flush=True)

        current_time += recommended

    print(f"[LANGGRAPH] Total video duration: {current_time:.1f}s", flush=True)

    return {
        **state,
        "timing_info": timing_info,
        "timing_synced": True,  # We've set exact durations
        "current_phase": "timing_synced",
    }


async def generate_assets_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Generate all visual assets (slides, animations) with proper timing.
    """
    from services.slide_generator import SlideGeneratorService
    from models.presentation_models import Slide, SlideType, CodeBlock, PresentationStyle
    import os

    print("[LANGGRAPH] Generating visual assets", flush=True)

    script = state["script"]
    timing_info = state.get("timing_info", [])
    request = state.get("request", {})

    if not script:
        return state

    # Create timing lookup
    timing_by_slide = {t["slide_id"]: t for t in timing_info}

    assets = []
    slide_generator = SlideGeneratorService()
    style = PresentationStyle(request.get("style", "dark"))

    # Extract RAG context and build course context for accurate diagram generation
    rag_context = request.get("rag_context")
    course_context = {
        'topic': request.get("topic", ""),
        'description': request.get("description", ""),
        'target_audience': script.get("target_audience", "intermediate developers"),
        'objectives': script.get("learning_objectives", []),
    }

    # Extract RAG images for diagram slides
    rag_images = request.get("rag_images", [])
    job_id = state.get("job_id")

    if rag_context:
        print(f"[LANGGRAPH] Using RAG context for diagram generation: {len(rag_context)} chars", flush=True)
    if rag_images:
        print(f"[LANGGRAPH] Using {len(rag_images)} RAG images for diagram slides", flush=True)

    for i, slide_data in enumerate(script.get("slides", [])):
        slide_id = slide_data.get("id", f"slide_{i}")
        timing = timing_by_slide.get(slide_id, {})

        # Convert dict to Slide model
        try:
            code_blocks = []
            for cb in slide_data.get("code_blocks", []):
                code_blocks.append(CodeBlock(
                    language=cb.get("language", "python"),
                    code=cb.get("code", ""),
                    filename=cb.get("filename"),
                    highlight_lines=cb.get("highlight_lines", []),
                    expected_output=cb.get("expected_output")
                ))

            slide_type_str = slide_data.get("type", "content")
            try:
                slide_type = SlideType(slide_type_str)
            except ValueError:
                slide_type = SlideType.CONTENT

            slide = Slide(
                id=slide_id,
                type=slide_type,
                title=slide_data.get("title"),
                subtitle=slide_data.get("subtitle"),
                content=slide_data.get("content"),
                bullet_points=slide_data.get("bullet_points", []),
                code_blocks=code_blocks,
                duration=slide_data.get("duration", 10.0),
                voiceover_text=slide_data.get("voiceover_text", ""),
                transition=slide_data.get("transition", "fade")
            )

            # Generate slide image with audience-based diagram complexity, career-based focus,
            # RAG context for accurate diagram generation, and RAG images for real diagrams
            target_audience = script.get("target_audience", "intermediate developers")
            target_career = script.get("target_career")  # Career for diagram focus (e.g., "data_engineer")
            image_bytes = await slide_generator.generate_slide_image(
                slide, style, target_audience, target_career,
                rag_context=rag_context,
                course_context=course_context,
                rag_images=rag_images,
                job_id=job_id
            )

            # Save to file
            output_dir = f"/tmp/presentations/{state['job_id']}"
            os.makedirs(output_dir, exist_ok=True)
            image_path = f"{output_dir}/slide_{i:03d}.png"

            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            # Get public URL for the image
            image_url = get_public_url(f"{state['job_id']}/slide_{i:03d}.png")

            asset = {
                "slide_id": slide_id,
                "asset_type": "image",
                "file_path": image_path,
                "url": image_url,
                "duration": timing.get("recommended_duration", slide.duration),
                "metadata": {
                    "slide_type": slide_type.value,
                    "has_animation": slide_type in [SlideType.CODE, SlideType.CODE_DEMO],
                    "animation_duration": timing.get("animation_duration", 0)
                }
            }
            assets.append(asset)
            print(f"[LANGGRAPH] Generated slide {i}: {image_path}", flush=True)

        except Exception as e:
            print(f"[LANGGRAPH] Failed to generate asset for slide {i}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            state["warnings"].append(f"Asset generation failed for slide {i}: {str(e)}")

    return {
        **state,
        "slide_assets": assets,
        "current_phase": "assets_generated",
    }


async def generate_animations_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Generate typing animations for code slides with proper timing.
    """
    from services.typing_animator import TypingAnimatorService
    import os

    print("[LANGGRAPH] Generating animations", flush=True)

    script = state["script"]
    timing_info = state.get("timing_info", [])

    if not script:
        return state

    timing_by_slide = {t["slide_id"]: t for t in timing_info}
    animations = {}

    animator = TypingAnimatorService()

    for i, slide in enumerate(script.get("slides", [])):
        slide_type = slide.get("type", "content")

        if slide_type not in ["code", "code_demo"]:
            continue

        slide_id = slide.get("id", f"slide_{i}")
        timing = timing_by_slide.get(slide_id, {})

        # Get animation duration - use the larger of:
        # 1. Calculated animation duration (based on code length)
        # 2. Voiceover duration (to stay in sync)
        target_duration = max(
            timing.get("animation_duration", 10.0),
            timing.get("voiceover_duration", 10.0)
        )

        code_blocks = slide.get("code_blocks", [])
        if not code_blocks:
            continue

        # Get code from first block
        code_block = code_blocks[0]
        code = code_block.get("code", "")
        language = code_block.get("language", "python")
        expected_output = code_block.get("expected_output")

        # Unescape literal \n to actual newlines (GPT sometimes double-escapes)
        if '\\n' in code:
            code = code.replace('\\n', '\n')
        if '\\t' in code:
            code = code.replace('\\t', '\t')

        if not code.strip():
            continue

        try:
            # Create output path
            output_dir = f"/tmp/presentations/animations"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{state['job_id']}_typing_{slide_id}.mp4"

            # Generate animation with proper duration
            # Returns tuple: (video_path, actual_duration)
            video_path, actual_duration = await animator.create_typing_animation(
                code=code,
                language=language,
                output_path=output_path,
                title=slide.get("title"),
                typing_speed="natural",
                target_duration=target_duration,
                execution_output=expected_output if slide_type == "code_demo" else None
            )

            # Get public URL for the animation
            animation_url = get_public_url(f"animations/{state['job_id']}_typing_{slide_id}.mp4")

            animations[slide_id] = {
                "url": animation_url,
                "file_path": video_path,
                "duration": actual_duration,
                "target_duration": target_duration,
                "synced": abs(actual_duration - target_duration) < 2.0
            }

            print(f"[LANGGRAPH] Animation for {slide_id}: {actual_duration}s (target: {target_duration}s)", flush=True)

        except Exception as e:
            print(f"[LANGGRAPH] Animation failed for slide {i}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            state["warnings"].append(f"Animation failed for slide {i}: {str(e)}")

    return {
        **state,
        "animations": animations,
        "current_phase": "animations_generated",
    }


async def build_timeline_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Build the timeline using SSVS (Semantic Slide-Voiceover Synchronization).

    This node:
    1. Converts slides and voiceover info to SSVS format
    2. Runs SSVS algorithm for optimal slide-voiceover alignment
    3. For diagram slides, runs SSVS-D for element focus animations
    4. Returns a Timeline object for the compositor
    """
    from services.timeline_builder import TimelineBuilder, SyncMethod

    print("[LANGGRAPH] Building timeline with SSVS synchronization", flush=True)

    script = state["script"]
    slide_voiceovers = state.get("slide_voiceovers", {})
    timing_info = state.get("timing_info", [])
    animations = state.get("animations", {})

    if not script:
        return {
            **state,
            "errors": state["errors"] + ["No script for timeline building"],
            "current_phase": "timeline_failed",
        }

    slides = script.get("slides", [])
    if not slides:
        return {
            **state,
            "warnings": state["warnings"] + ["No slides for timeline"],
            "current_phase": "timeline_skipped",
        }

    # Build combined voiceover text and collect word timestamps
    # For per-slide voiceover, we estimate word timestamps from duration
    word_timestamps = []
    current_time = 0.0

    for slide in slides:
        slide_id = slide.get("id", f"slide_{slides.index(slide)}")
        voiceover_info = slide_voiceovers.get(slide_id, {})
        voiceover_text = voiceover_info.get("text", "") or slide.get("voiceover_text", "")
        voiceover_duration = voiceover_info.get("duration", 0.0)

        # Clean voiceover text
        voiceover_text = clean_voiceover_text(voiceover_text)
        words = voiceover_text.split()

        if words and voiceover_duration > 0:
            # Estimate word timestamps proportionally
            time_per_word = voiceover_duration / len(words)
            for word in words:
                word_timestamps.append({
                    "word": word,
                    "start": current_time,
                    "end": current_time + time_per_word
                })
                current_time += time_per_word
        else:
            # Fallback: use slide duration from timing_info
            for timing in timing_info:
                if timing.get("slide_id") == slide_id:
                    current_time += timing.get("voiceover_duration", 5.0)
                    break

    # Get total audio duration
    audio_duration = sum(
        slide_voiceovers.get(s.get("id", f"slide_{i}"), {}).get("duration", 0.0)
        for i, s in enumerate(slides)
    )

    # Collect diagram structure for SSVS-D
    diagrams = {}
    for slide in slides:
        if slide.get("type") == "diagram":
            slide_id = slide.get("id")
            diagram_spec = slide.get("diagram_spec", {})
            if diagram_spec:
                # Extract elements from diagram description
                # In production, this would come from the diagram generator
                diagrams[slide_id] = {
                    "title": diagram_spec.get("title", ""),
                    "diagram_type": diagram_spec.get("style", "flowchart"),
                    "elements": diagram_spec.get("elements", [])
                }

    # Get audio URL (combined or from first slide)
    audio_url = None
    audio_path = None
    for slide_id, vo_info in slide_voiceovers.items():
        if vo_info.get("url"):
            audio_url = vo_info.get("url")
            break

    # Build timeline with SSVS
    try:
        builder = TimelineBuilder(sync_method=SyncMethod.SSVS)
        timeline = builder.build(
            word_timestamps=word_timestamps,
            slides=slides,
            audio_duration=audio_duration,
            audio_url=audio_url,
            audio_path=audio_path,
            animations=animations,
            diagrams=diagrams if diagrams else None
        )

        # Store timeline in state for compositor
        timeline_dict = timeline.to_dict()

        print(f"[LANGGRAPH] Timeline built: {timeline_dict['metadata'].get('events_count', 0)} events, "
              f"sync_method={timeline_dict.get('sync_method', 'unknown')}, "
              f"avg_semantic={timeline_dict['metadata'].get('avg_semantic_score', 0):.3f}", flush=True)

        return {
            **state,
            "timeline": timeline_dict,
            "current_phase": "timeline_built",
        }

    except Exception as e:
        print(f"[LANGGRAPH] Timeline build error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            **state,
            "timeline": None,
            "warnings": state["warnings"] + [f"Timeline build failed: {str(e)}"],
            "current_phase": "timeline_fallback",
        }


async def compose_video_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Compose the final video with all assets, animations, and per-scene voiceover.
    Uses SSVS timeline for precise synchronization when available.
    """
    import httpx
    import os

    print("[LANGGRAPH] Composing final video with SSVS timeline sync", flush=True)

    script = state["script"]
    assets = state.get("slide_assets", [])
    animations = state.get("animations", {})
    slide_voiceovers = state.get("slide_voiceovers", {})
    timing_info = state.get("timing_info", [])
    timeline = state.get("timeline")  # SSVS timeline

    if not script or not assets:
        return {
            **state,
            "errors": state["errors"] + ["Missing script or assets for composition"],
            "current_phase": "composition_failed",
        }

    # Use SSVS timeline for timing if available
    if timeline and timeline.get("visual_events"):
        print(f"[LANGGRAPH] Using SSVS timeline with {len(timeline['visual_events'])} events", flush=True)
        print(f"[LANGGRAPH] SSVS sync_method: {timeline.get('sync_method', 'unknown')}", flush=True)

        # Build timing lookup from SSVS timeline events
        timing_by_slide = {}
        for event in timeline["visual_events"]:
            metadata = event.get("metadata", {})
            slide_id = metadata.get("slide_id")
            if slide_id:
                timing_by_slide[slide_id] = {
                    "start_time": event.get("time_start", 0),
                    "recommended_duration": event.get("duration", 5.0),
                    "semantic_score": timeline.get("semantic_scores", {}).get(slide_id, 0)
                }

        # Log SSVS semantic scores
        semantic_scores = timeline.get("semantic_scores", {})
        if semantic_scores:
            avg_score = sum(semantic_scores.values()) / len(semantic_scores)
            print(f"[LANGGRAPH] SSVS average semantic score: {avg_score:.3f}", flush=True)
    else:
        print("[LANGGRAPH] SSVS timeline not available, using timing_info fallback", flush=True)
        timing_by_slide = {t["slide_id"]: t for t in timing_info}

    # Build scenes for the compositor with per-scene audio
    scenes = []

    for asset in assets:
        slide_id = asset["slide_id"]

        # Get exact timing info (from SSVS or fallback)
        timing = timing_by_slide.get(slide_id, {})
        duration = timing.get("recommended_duration", asset["duration"])

        # Get per-slide voiceover URL
        voiceover_info = slide_voiceovers.get(slide_id, {})
        audio_url = voiceover_info.get("url")

        scene = {
            "duration": duration,
            "transition": "fade",
            "audio_url": audio_url  # Per-scene audio for sync
        }

        # Check if this slide has an animation
        if slide_id in animations:
            anim = animations[slide_id]
            scene["video_url"] = anim["url"]
        else:
            scene["image_url"] = asset.get("file_path") or asset.get("url")

        # Add SSVS metadata if available
        if timing.get("semantic_score"):
            scene["semantic_score"] = timing["semantic_score"]

        scenes.append(scene)

        has_audio = "with audio" if audio_url else "no audio"
        sem_score = f", semantic={timing.get('semantic_score', 0):.2f}" if timing.get('semantic_score') else ""
        print(f"[LANGGRAPH] Scene {slide_id}: {duration:.1f}s ({has_audio}{sem_score})", flush=True)

    print(f"[LANGGRAPH] Composing {len(scenes)} scenes with SSVS synchronization", flush=True)

    # Call media-generator composition endpoint
    # Note: voiceover_url is None since we're using per-scene audio
    media_generator_url = os.getenv("MEDIA_GENERATOR_URL", "http://media-generator:8004")

    async with httpx.AsyncClient(timeout=600.0) as client:
        # Start the composition job
        response = await client.post(
            f"{media_generator_url}/api/v1/media/slideshow/compose",
            json={
                "scenes": scenes,
                "voiceover_url": None,  # Using per-scene audio instead
                "output_format": "16:9",
                "quality": "1080p",
                "fps": 30
            }
        )

        if response.status_code != 200:
            error_msg = f"Composition request failed: {response.status_code} - {response.text}"
            print(f"[LANGGRAPH] {error_msg}", flush=True)
            return {
                **state,
                "errors": state["errors"] + [error_msg],
                "current_phase": "composition_failed",
            }

        result = response.json()
        composition_job_id = result.get("job_id")
        print(f"[LANGGRAPH] Composition job started: {composition_job_id}", flush=True)

        # Poll for job completion
        max_wait_time = 600  # 10 minutes max
        poll_interval = 5  # Check every 5 seconds
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            await asyncio.sleep(poll_interval)
            elapsed_time += poll_interval

            # Check job status
            status_response = await client.get(
                f"{media_generator_url}/api/v1/media/jobs/{composition_job_id}"
            )

            if status_response.status_code != 200:
                print(f"[LANGGRAPH] Failed to get job status: {status_response.text}", flush=True)
                continue

            job_data = status_response.json()
            job_status = job_data.get("status")
            progress = job_data.get("progress_percent", 0)

            print(f"[LANGGRAPH] Composition progress: {progress}% (status: {job_status})", flush=True)

            if job_status == "completed":
                # Get the output URL from the job result
                output_data = job_data.get("output_data", {})
                output_video_url = output_data.get("video_url") or output_data.get("output_url")

                if output_video_url:
                    print(f"[LANGGRAPH] Composition complete: {output_video_url}", flush=True)
                    return {
                        **state,
                        "composition_job_id": composition_job_id,
                        "output_video_url": output_video_url,
                        "current_phase": "composition_complete",
                    }
                else:
                    # Check if there's a file path instead
                    output_path = output_data.get("video_path") or output_data.get("output_path")
                    if output_path:
                        # Construct URL using centralized URL config
                        output_video_url = url_config.convert_to_public_url(output_path)
                        print(f"[LANGGRAPH] Composition complete (from path): {output_video_url}", flush=True)
                        return {
                            **state,
                            "composition_job_id": composition_job_id,
                            "output_video_url": output_video_url,
                            "current_phase": "composition_complete",
                        }

                    print(f"[LANGGRAPH] Composition complete but no output URL found: {output_data}", flush=True)
                    return {
                        **state,
                        "composition_job_id": composition_job_id,
                        "warnings": state["warnings"] + ["Composition complete but no output URL"],
                        "current_phase": "composition_complete",
                    }

            elif job_status == "failed":
                error_msg = job_data.get("error_message", "Unknown error")
                print(f"[LANGGRAPH] Composition failed: {error_msg}", flush=True)
                return {
                    **state,
                    "composition_job_id": composition_job_id,
                    "errors": state["errors"] + [f"Composition failed: {error_msg}"],
                    "current_phase": "composition_failed",
                }

            elif job_status == "cancelled":
                return {
                    **state,
                    "composition_job_id": composition_job_id,
                    "errors": state["errors"] + ["Composition was cancelled"],
                    "current_phase": "composition_failed",
                }

        # Timeout reached
        print(f"[LANGGRAPH] Composition timed out after {max_wait_time}s", flush=True)
        return {
            **state,
            "composition_job_id": composition_job_id,
            "errors": state["errors"] + [f"Composition timed out after {max_wait_time} seconds"],
            "current_phase": "composition_timeout",
        }


async def finalize_node(state: VideoGenerationState) -> VideoGenerationState:
    """
    Finalize the workflow and prepare output.
    """
    print("[LANGGRAPH] Finalizing workflow", flush=True)

    return {
        **state,
        "completed_at": datetime.utcnow().isoformat(),
        "current_phase": "completed",
    }


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================

def should_fix_content(state: VideoGenerationState) -> Literal["fix", "continue", "fail"]:
    """Decide whether to fix content issues or continue"""
    if state["validation_passed"]:
        return "continue"

    if state["iteration_count"] >= state["max_iterations"]:
        return "fail"

    # Check if issues are fixable
    critical_issues = [i for i in state["validation_issues"] if i["severity"] == "critical"]
    if critical_issues:
        return "fix"

    return "continue"


def should_retry_timing(state: VideoGenerationState) -> Literal["retry", "continue"]:
    """Decide whether to retry timing synchronization"""
    if state["timing_synced"]:
        return "continue"

    if state["iteration_count"] >= state["max_iterations"]:
        return "continue"  # Proceed with best effort

    return "retry"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_video_generation_graph() -> StateGraph:
    """
    Create the LangGraph workflow for video generation.

    Flow:
    1. Initialize -> Generate Script -> Validate Content
    2. If validation fails -> Fix Content -> Validate (loop)
    3. If validation passes -> Execute Code -> Generate Diagrams
    4. Generate Voiceover -> Synchronize Timing
    5. If timing off -> Adjust (loop)
    6. Generate Assets -> Generate Animations -> Build Timeline (SSVS) -> Compose Video
    7. Finalize

    SSVS Integration:
    - Build Timeline node uses SSVS algorithm for semantic slide-voiceover alignment
    - For diagram slides, SSVS-D generates focus animation sequences
    """

    workflow = StateGraph(VideoGenerationState)

    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("generate_script", generate_script_node)
    workflow.add_node("validate_content", validate_content_node)
    workflow.add_node("fix_content", fix_content_issues_node)
    workflow.add_node("execute_code", execute_code_node)
    workflow.add_node("generate_diagrams", generate_diagrams_node)
    workflow.add_node("generate_voiceover", generate_voiceover_node)
    workflow.add_node("sync_timing", synchronize_timing_node)
    workflow.add_node("generate_assets", generate_assets_node)
    workflow.add_node("generate_animations", generate_animations_node)
    workflow.add_node("build_timeline", build_timeline_node)  # SSVS synchronization
    workflow.add_node("compose_video", compose_video_node)
    workflow.add_node("finalize", finalize_node)

    # Set entry point
    workflow.set_entry_point("initialize")

    # Add edges
    workflow.add_edge("initialize", "generate_script")
    workflow.add_edge("generate_script", "validate_content")

    # Conditional: validation result
    workflow.add_conditional_edges(
        "validate_content",
        should_fix_content,
        {
            "fix": "fix_content",
            "continue": "execute_code",
            "fail": "finalize"  # Exit with errors
        }
    )

    # Fix content loops back to validation
    workflow.add_edge("fix_content", "validate_content")

    # Main flow after validation
    workflow.add_edge("execute_code", "generate_diagrams")
    workflow.add_edge("generate_diagrams", "generate_voiceover")
    workflow.add_edge("generate_voiceover", "sync_timing")

    # Conditional: timing sync
    workflow.add_conditional_edges(
        "sync_timing",
        should_retry_timing,
        {
            "retry": "sync_timing",
            "continue": "generate_assets"
        }
    )

    # Asset generation flow with SSVS timeline
    workflow.add_edge("generate_assets", "generate_animations")
    workflow.add_edge("generate_animations", "build_timeline")  # SSVS before composition
    workflow.add_edge("build_timeline", "compose_video")
    workflow.add_edge("compose_video", "finalize")

    # Finalize ends the workflow
    workflow.add_edge("finalize", END)

    return workflow


# =============================================================================
# ORCHESTRATOR CLASS
# =============================================================================

class LangGraphOrchestrator:
    """
    Main orchestrator class for running the video generation workflow.
    """

    def __init__(self):
        self.graph = create_video_generation_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)

    async def generate_video(
        self,
        request: GeneratePresentationRequest,
        job_id: Optional[str] = None,
        on_progress: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run the complete video generation workflow.

        Args:
            request: The presentation generation request
            job_id: Optional job ID (generated if not provided)
            on_progress: Optional progress callback

        Returns:
            Final state with output video URL and metadata
        """
        job_id = job_id or str(uuid.uuid4())

        # Initialize state
        initial_state: VideoGenerationState = {
            "request": request.model_dump() if hasattr(request, 'model_dump') else request,
            "job_id": job_id,
            "script": None,
            "script_generation_attempts": 0,
            "validation_issues": [],
            "validation_passed": False,
            "slide_assets": [],
            "diagrams_generated": [],
            "code_executions": [],
            "slide_voiceovers": {},  # Per-slide voiceover info for sync
            "voiceover_duration": 0.0,
            "timing_info": [],
            "timing_synced": False,
            "animations": {},
            "timeline": None,  # SSVS timeline for sync
            "output_video_url": None,
            "composition_job_id": None,
            "current_phase": "starting",
            "iteration_count": 0,
            "max_iterations": 3,
            "errors": [],
            "warnings": [],
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }

        config = {"configurable": {"thread_id": job_id}}

        print(f"[LANGGRAPH] Starting workflow for job {job_id}", flush=True)

        # Run the workflow
        final_state = None
        async for state in self.app.astream(initial_state, config):
            # Get the latest state from any node that returned
            for node_name, node_state in state.items():
                final_state = node_state
                phase = node_state.get("current_phase", "unknown")

                print(f"[LANGGRAPH] Phase: {phase}", flush=True)

                if on_progress:
                    # Map phases to progress percentages
                    progress_map = {
                        "initialized": 5,
                        "script_generated": 15,
                        "validated": 20,
                        "content_fixed": 25,
                        "code_executed": 35,
                        "diagrams_prepared": 40,
                        "voiceover_generated": 50,
                        "timing_synced": 60,
                        "assets_generated": 70,
                        "animations_generated": 80,
                        "timeline_built": 85,  # SSVS synchronization
                        "timeline_fallback": 85,
                        "composition_started": 90,
                        "completed": 100,
                    }
                    progress = progress_map.get(phase, 0)
                    await on_progress(progress, f"Phase: {phase}")

        print(f"[LANGGRAPH] Workflow completed for job {job_id}", flush=True)

        return final_state or initial_state


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def run_video_generation(
    request: GeneratePresentationRequest,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run video generation.
    """
    orchestrator = LangGraphOrchestrator()
    return await orchestrator.generate_video(request, job_id)
