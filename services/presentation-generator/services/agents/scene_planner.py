"""
Scene Planner Agent

Plans the content and timing cues for a single scene.
Creates a detailed blueprint of what should happen and when.

Supports multiple LLM providers via shared.llm_provider:
- Groq (Llama 3.3 - Ultra-fast, cheap)
- OpenAI (GPT-4o, GPT-4o-mini)
- DeepSeek, Mistral, etc.
"""

import json
import os
import re
from typing import Any, Dict, List


def strip_json_comments(json_str: str) -> str:
    """Remove Python/JS style comments from JSON string that LLMs sometimes add."""
    # Remove single-line comments (# ... or // ...)
    # Be careful not to remove # inside strings
    lines = json_str.split('\n')
    cleaned_lines = []
    for line in lines:
        # Find if there's a comment outside of strings
        in_string = False
        escape_next = False
        comment_start = -1
        for i, char in enumerate(line):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
            if not in_string and char == '#':
                comment_start = i
                break
            if not in_string and i < len(line) - 1 and line[i:i+2] == '//':
                comment_start = i
                break
        if comment_start >= 0:
            line = line[:comment_start].rstrip()
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name, get_provider_config
    USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    USE_SHARED_LLM = False
    print("[SCENE_PLANNER] Warning: shared.llm_provider not found, using direct OpenAI", flush=True)

from .base_agent import BaseAgent, AgentResult, TimingCue


SCENE_PLANNING_PROMPT = """You are a video scene planner. Your job is to create a detailed timing plan for a single slide in a coding tutorial video.

Given the slide content, create timing cues that specify EXACTLY when each visual element should appear relative to the voiceover.

CRITICAL RULES:
1. The voiceover text will be converted to speech - estimate ~2.5 words per second
2. Visual elements must appear BEFORE or AT THE SAME TIME they are mentioned in voiceover
3. Never say "as you can see" unless something is already visible
4. For code slides: code should START appearing when we begin talking about it

Input slide:
{slide_json}

Output a JSON object with this structure:
{{
  "scene_type": "title|content|code|code_demo|conclusion",
  "estimated_duration": <seconds based on voiceover length>,
  "voiceover_text": "<the narration text, potentially adjusted for better sync>",
  "timing_cues": [
    {{
      "timestamp": 0.0,
      "event_type": "show_title|show_text|show_bullet|show_code|start_typing|show_output|highlight_line",
      "target": "<what to show/highlight>",
      "duration": <optional, how long this element stays>,
      "description": "<what happens>"
    }}
  ],
  "visual_requirements": {{
    "needs_animation": true/false,
    "animation_type": "typing|highlight|none",
    "code_to_animate": "<code if applicable>",
    "expected_output": "<output if code_demo>"
  }},
  "sync_checkpoints": [
    {{
      "word_or_phrase": "<key phrase in voiceover>",
      "expected_time": <when this phrase should be said>,
      "visual_state": "<what should be visible at this moment>"
    }}
  ]
}}

Be precise with timestamps. Calculate them based on voiceover word count.

IMPORTANT: Output ONLY valid JSON. Do NOT include any comments (no # or // comments). Do NOT include explanations or calculations in the JSON - just the values.
"""


class ScenePlannerAgent(BaseAgent):
    """Plans content and timing cues for a single scene"""

    def __init__(self):
        super().__init__("SCENE_PLANNER")
        if USE_SHARED_LLM:
            self.client = get_llm_client()
            self.model = get_model_name("fast")  # Use fast model for scene planning
            config = get_provider_config()
            self.log(f"Using {config.name} provider with model {self.model}")
        else:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Cheaper default

    async def execute(self, state: Dict[str, Any]) -> AgentResult:
        """Plan a single scene with timing cues"""
        slide_data = state.get("slide_data", {})
        scene_index = state.get("scene_index", 0)

        self.log(f"Planning scene {scene_index}: {slide_data.get('title', 'Untitled')}")

        try:
            # Call GPT-4 to plan the scene
            prompt = SCENE_PLANNING_PROMPT.format(
                slide_json=json.dumps(slide_data, indent=2)
            )

            raw_content = None
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise video timing planner. Output only valid JSON with no comments."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=2000
                )
                raw_content = response.choices[0].message.content
            except Exception as api_error:
                # Groq may fail with json_validate_failed if model outputs comments
                # Retry without json_object format and strip comments ourselves
                if "json_validate_failed" in str(api_error) or "failed_generation" in str(api_error):
                    self.log(f"Scene {scene_index}: JSON validation failed, retrying without format constraint")
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a precise video timing planner. Output only valid JSON with no comments (no # or // comments)."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=2000
                    )
                    raw_content = response.choices[0].message.content
                    # Strip any comments the model may have added
                    raw_content = strip_json_comments(raw_content)
                else:
                    raise

            try:
                plan = json.loads(raw_content)
            except json.JSONDecodeError as je:
                self.log(f"Scene {scene_index}: JSON parse error, trying to clean and retry")
                # Try stripping comments first
                cleaned = strip_json_comments(raw_content or "")
                try:
                    plan = json.loads(cleaned)
                except json.JSONDecodeError:
                    # Try to extract JSON from response (sometimes wrapped in markdown)
                    json_match = re.search(r'\{[\s\S]*\}', cleaned)
                    if json_match:
                        try:
                            plan = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            return self._create_fallback_plan(slide_data, scene_index)
                    else:
                        return self._create_fallback_plan(slide_data, scene_index)

            # Validate and enhance the plan
            plan = self._validate_plan(plan, slide_data)

            # Calculate timing cues as proper objects
            timing_cues = []
            for cue_data in plan.get("timing_cues", []):
                timing_cues.append({
                    "timestamp": cue_data.get("timestamp", 0),
                    "event_type": cue_data.get("event_type", "show_text"),
                    "target": cue_data.get("target", ""),
                    "duration": cue_data.get("duration"),
                    "description": cue_data.get("description", "")
                })

            self.log(f"Scene {scene_index} planned: {len(timing_cues)} timing cues, ~{plan.get('estimated_duration', 0):.1f}s")

            return AgentResult(
                success=True,
                data={
                    "planned_content": plan,
                    "timing_cues": timing_cues,
                    "voiceover_text": plan.get("voiceover_text", slide_data.get("voiceover_text", "")),
                    "estimated_duration": plan.get("estimated_duration", 10),
                    "visual_requirements": plan.get("visual_requirements", {}),
                    "sync_checkpoints": plan.get("sync_checkpoints", [])
                }
            )

        except Exception as e:
            self.log(f"Error planning scene {scene_index}: {e}")
            # Fallback to basic plan
            return self._create_fallback_plan(slide_data, scene_index)

    def _validate_plan(self, plan: Dict[str, Any], slide_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix the plan"""
        # Ensure voiceover text exists
        if not plan.get("voiceover_text"):
            plan["voiceover_text"] = slide_data.get("voiceover_text", "")

        # Calculate duration from voiceover if not provided
        voiceover = plan.get("voiceover_text", "")
        word_count = len(voiceover.split())
        calculated_duration = word_count / 2.5  # ~150 words per minute

        if not plan.get("estimated_duration") or plan["estimated_duration"] < calculated_duration:
            plan["estimated_duration"] = calculated_duration + 1  # Add buffer

        # Ensure timing cues start at 0
        if plan.get("timing_cues") and plan["timing_cues"][0].get("timestamp", 0) > 0:
            # Insert initial cue
            plan["timing_cues"].insert(0, {
                "timestamp": 0,
                "event_type": "show_title" if slide_data.get("title") else "show_content",
                "target": slide_data.get("title", ""),
                "description": "Scene starts"
            })

        return plan

    def _create_fallback_plan(self, slide_data: Dict[str, Any], scene_index: int) -> AgentResult:
        """Create a basic fallback plan if GPT fails"""
        voiceover = slide_data.get("voiceover_text", "")
        word_count = len(voiceover.split())
        duration = max(word_count / 2.5, 5) + 1

        slide_type = slide_data.get("type", "content")

        timing_cues = [
            {
                "timestamp": 0,
                "event_type": "show_title",
                "target": slide_data.get("title", ""),
                "description": "Show slide title"
            }
        ]

        # Add code timing for code slides
        if slide_type in ["code", "code_demo"]:
            timing_cues.append({
                "timestamp": 1.0,
                "event_type": "start_typing",
                "target": "code",
                "duration": duration - 2,
                "description": "Start typing animation"
            })

        return AgentResult(
            success=True,
            data={
                "planned_content": {
                    "scene_type": slide_type,
                    "estimated_duration": duration,
                    "voiceover_text": voiceover
                },
                "timing_cues": timing_cues,
                "voiceover_text": voiceover,
                "estimated_duration": duration,
                "visual_requirements": {
                    "needs_animation": slide_type in ["code", "code_demo"],
                    "animation_type": "typing" if slide_type in ["code", "code_demo"] else "none"
                },
                "sync_checkpoints": []
            },
            warnings=["Using fallback plan due to planning error"]
        )
