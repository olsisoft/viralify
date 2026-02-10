"""
AI Video Planner Service
Uses GPT-4 to analyze prompts and generate video scene plans
- Supports script-based generation with Time, Visual, Audio structure
- Enforces exact duration
- Supports videos up to 30 minutes
"""

import json
import httpx
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
import os
import re


class SceneType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    AI_IMAGE = "ai_image"  # Generated with DALL-E
    TEXT = "text"  # Text overlay only


class ScriptSegment(BaseModel):
    """A segment of the script with timing, visual, and audio"""
    time_range: str  # "0:00-0:05"
    visual: str  # Description of what to show
    audio: str  # Voiceover text for this segment
    start_seconds: float = 0
    end_seconds: float = 0
    duration: float = 0


class MediaSource(str, Enum):
    PEXELS = "pexels"
    UNSPLASH = "unsplash"
    PIXABAY = "pixabay"
    DALLE = "dalle"
    UPLOADED = "uploaded"


class Scene(BaseModel):
    id: str
    order: int
    start_time: float  # seconds
    duration: float  # seconds
    scene_type: SceneType
    description: str  # What to search for or generate
    search_keywords: List[str]
    preferred_source: Optional[MediaSource] = None
    text_overlay: Optional[str] = None
    transition: str = "fade"  # fade, cut, dissolve, slide

    # Filled after asset fetching
    media_url: Optional[str] = None
    thumbnail_url: Optional[str] = None


class VideoProject(BaseModel):
    id: str
    title: str
    description: str
    duration: int  # total seconds
    format: str = "9:16"  # 9:16, 16:9, 1:1
    style: str = "cinematic"

    # Script and voiceover
    script: str
    voiceover_text: str
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    # Scenes
    scenes: List[Scene] = []

    # Music
    music_style: Optional[str] = None
    music_url: Optional[str] = None
    music_volume: float = 0.3  # 0-1

    # Output
    output_url: Optional[str] = None
    status: str = "draft"


class AIVideoPlannerService:
    """Plans video scenes using GPT-4"""

    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        self.model = "gpt-4o-mini"

    async def plan_video(
        self,
        prompt: str,
        duration: int = 30,
        style: str = "cinematic",
        format: str = "9:16",
        voice_style: str = "professional"
    ) -> VideoProject:
        """
        Analyze prompt and generate a complete video plan with scenes
        """

        system_prompt = """You are an expert video producer and scriptwriter for viral short-form content.
Your task is to create a detailed video plan from a user's prompt.

You must return a JSON object with this exact structure:
{
    "title": "Short catchy title",
    "description": "Brief description of the video",
    "script": "The narrative script for the video",
    "voiceover_text": "The exact text for the voiceover (natural, engaging)",
    "music_style": "Type of background music (e.g., 'upbeat electronic', 'calm piano', 'epic orchestral')",
    "scenes": [
        {
            "order": 1,
            "duration": 5,
            "scene_type": "video",
            "description": "Description of what should be shown",
            "search_keywords": ["keyword1", "keyword2", "keyword3"],
            "text_overlay": "Optional text to show on screen",
            "transition": "fade"
        }
    ]
}

Guidelines:
- Create scenes that total exactly the requested duration
- Each scene should be 3-7 seconds for short-form content
- Use "video" type for motion content, "ai_image" for unique/specific visuals
- search_keywords should be specific and searchable on stock sites
- voiceover_text should match the pacing of scenes
- Make content engaging, hook viewers in first 3 seconds
- End with a call-to-action or memorable conclusion
"""

        user_prompt = f"""Create a {duration}-second {style} video based on this prompt:

"{prompt}"

Video format: {format}
Voice style: {voice_style}

Return ONLY the JSON object, no other text."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "response_format": {"type": "json_object"}
                },
                timeout=60.0
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")

            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON response from API: {e}")

            if not data.get("choices") or len(data["choices"]) == 0:
                raise Exception("Empty choices in API response")

            content = data["choices"][0].get("message", {}).get("content", "")
            if not content:
                raise Exception("Empty content in API response")

            try:
                plan = json.loads(content)
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON in LLM response: {e}")

        # Convert to VideoProject
        import uuid
        project_id = str(uuid.uuid4())

        scenes = []
        current_time = 0.0

        for i, scene_data in enumerate(plan.get("scenes", [])):
            scene_id = f"{project_id}-scene-{i+1}"
            scene_duration = scene_data.get("duration", 5)

            scene = Scene(
                id=scene_id,
                order=scene_data.get("order", i + 1),
                start_time=current_time,
                duration=scene_duration,
                scene_type=SceneType(scene_data.get("scene_type", "video")),
                description=scene_data.get("description", ""),
                search_keywords=scene_data.get("search_keywords", []),
                text_overlay=scene_data.get("text_overlay"),
                transition=scene_data.get("transition", "fade")
            )
            scenes.append(scene)
            current_time += scene_duration

        project = VideoProject(
            id=project_id,
            title=plan.get("title", "Untitled Video"),
            description=plan.get("description", ""),
            duration=duration,
            format=format,
            style=style,
            script=plan.get("script", ""),
            voiceover_text=plan.get("voiceover_text", ""),
            music_style=plan.get("music_style"),
            scenes=scenes,
            status="planned"
        )

        return project

    async def regenerate_scene(
        self,
        scene: Scene,
        new_description: str
    ) -> Scene:
        """Regenerate a single scene with new description"""

        prompt = f"""Generate search keywords for this video scene:
Description: {new_description}

Return JSON: {{"search_keywords": ["keyword1", "keyword2", "keyword3"], "scene_type": "video" or "ai_image"}}"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                    "response_format": {"type": "json_object"}
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")

            data = response.json()
            result = json.loads(data["choices"][0]["message"]["content"])

        scene.description = new_description
        scene.search_keywords = result.get("search_keywords", [new_description])
        scene.scene_type = SceneType(result.get("scene_type", "video"))
        scene.media_url = None  # Reset to fetch new asset

        return scene

    async def generate_script_from_topic(
        self,
        topic: str,
        duration: int = 60,
        style: str = "educational",
        target_audience: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate a structured script from a topic.
        Returns a script with Time, Visual, and Audio columns.
        Supports durations up to 45 minutes (2700 seconds).
        """

        # Calculate number of segments based on duration
        # Short videos (< 60s): 5-8 second segments
        # Medium videos (1-5 min): 8-12 second segments
        # Long videos (5+ min): 12-20 second segments
        if duration <= 60:
            segment_duration = 6
        elif duration <= 300:
            segment_duration = 10
        else:
            segment_duration = 15

        num_segments = max(3, duration // segment_duration)
        print(f"Generating script: {duration}s duration, ~{num_segments} segments of ~{segment_duration}s each")

        system_prompt = f"""You are an expert viral video scriptwriter. Create a detailed script for a {duration}-second video.

CRITICAL: The total duration of all segments MUST equal EXACTLY {duration} seconds.

Return a JSON object with this EXACT structure:
{{
    "title": "Catchy video title",
    "hook": "Attention-grabbing opening line",
    "segments": [
        {{
            "time_range": "0:00-0:05",
            "visual": "Exact description of what to show on screen (be specific for stock footage search)",
            "audio": "The voiceover text for this segment"
        }},
        // ... more segments until you reach {duration} seconds
    ],
    "cta": "Call to action at the end",
    "music_mood": "Specific music mood (e.g., 'upbeat electronic', 'calm piano', 'epic orchestral', 'motivational corporate')",
    "hashtags": ["relevant", "hashtags"]
}}

IMPORTANT RULES:
1. First segment MUST start at 0:00
2. Last segment MUST end at {duration // 60}:{duration % 60:02d}
3. Segments must be continuous (no gaps)
4. Each segment: {segment_duration-3} to {segment_duration+5} seconds
5. Visual descriptions must be searchable on stock video/image sites
6. Audio must be natural speech, matching the visual timing
7. Hook viewers in first 3 seconds
8. Include pattern interrupts every 15-20 seconds for longer videos
9. End with a clear CTA

Style: {style}
Target audience: {target_audience}
"""

        user_prompt = f"""Create a {duration}-second viral video script about:

"{topic}"

Remember: Total duration must be EXACTLY {duration} seconds.
Generate approximately {num_segments} segments.
Return ONLY the JSON object."""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "response_format": {"type": "json_object"}
                },
                timeout=120.0
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.text}")

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            script_data = json.loads(content)

        # Parse and validate segments
        segments = []
        for seg in script_data.get("segments", []):
            time_range = seg.get("time_range", "0:00-0:05")
            start_str, end_str = time_range.split("-")

            # Parse time strings to seconds
            start_seconds = self._time_to_seconds(start_str)
            end_seconds = self._time_to_seconds(end_str)

            segment = ScriptSegment(
                time_range=time_range,
                visual=seg.get("visual", ""),
                audio=seg.get("audio", ""),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration=end_seconds - start_seconds
            )
            segments.append(segment)

        # Adjust segments to match exact duration
        segments = self._adjust_segments_duration(segments, duration)

        return {
            "title": script_data.get("title", ""),
            "hook": script_data.get("hook", ""),
            "segments": [s.model_dump() for s in segments],
            "cta": script_data.get("cta", ""),
            "music_mood": script_data.get("music_mood", "cinematic"),
            "hashtags": script_data.get("hashtags", []),
            "total_duration": duration
        }

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert time string (M:SS or MM:SS) to seconds"""
        parts = time_str.strip().split(":")
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(parts[0])

    def _seconds_to_time(self, seconds: float) -> str:
        """Convert seconds to time string (M:SS)"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def _adjust_segments_duration(
        self,
        segments: List[ScriptSegment],
        target_duration: int
    ) -> List[ScriptSegment]:
        """Adjust segment durations to match exact target duration"""

        if not segments:
            # Create default segments if none provided
            num_default_segments = max(3, target_duration // 10)
            seg_duration = target_duration / num_default_segments
            segments = []
            for i in range(num_default_segments):
                segments.append(ScriptSegment(
                    time_range=f"{self._seconds_to_time(i * seg_duration)}-{self._seconds_to_time((i+1) * seg_duration)}",
                    visual=f"Scene {i+1}",
                    audio=f"Part {i+1} of the video",
                    start_seconds=i * seg_duration,
                    end_seconds=(i+1) * seg_duration,
                    duration=seg_duration
                ))
            return segments

        # Calculate total current duration
        total_duration = sum(s.duration for s in segments)
        print(f"Script adjustment: {len(segments)} segments, total {total_duration}s, target {target_duration}s")

        if abs(total_duration - target_duration) < 0.5:
            print("Duration is close enough, no adjustment needed")
            return segments  # Close enough

        # Scale all segments proportionally
        scale_factor = target_duration / total_duration if total_duration > 0 else 1
        print(f"Scaling segments by factor {scale_factor:.2f}")

        current_time = 0.0
        adjusted = []

        for i, seg in enumerate(segments):
            new_duration = seg.duration * scale_factor

            # Ensure minimum duration of 3 seconds, max 60 seconds
            new_duration = max(3.0, min(60.0, new_duration))

            # For last segment, adjust to exactly hit target
            if i == len(segments) - 1:
                new_duration = max(3.0, target_duration - current_time)

            new_segment = ScriptSegment(
                time_range=f"{self._seconds_to_time(current_time)}-{self._seconds_to_time(current_time + new_duration)}",
                visual=seg.visual,
                audio=seg.audio,
                start_seconds=current_time,
                end_seconds=current_time + new_duration,
                duration=new_duration
            )
            adjusted.append(new_segment)
            current_time += new_duration

        final_duration = sum(s.duration for s in adjusted)
        print(f"After adjustment: {len(adjusted)} segments, total {final_duration}s")
        return adjusted

    async def script_to_video_project(
        self,
        script_data: Dict[str, Any],
        format: str = "9:16",
        voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    ) -> VideoProject:
        """Convert a script to a VideoProject for video generation"""

        import uuid
        project_id = str(uuid.uuid4())

        scenes = []
        voiceover_parts = []

        # Support both 'segments' and 'scenes' formats from different sources
        segments_data = script_data.get('segments', []) or script_data.get('scenes', [])
        print(f"Converting script to video project: {len(segments_data)} segments")

        for i, seg in enumerate(segments_data):
            scene_id = f"{project_id}-scene-{i+1}"

            # Extract keywords from visual description
            keywords = self._extract_keywords(seg.get("visual", ""))

            # Get duration from segment, fall back to calculating from time/time_range
            seg_duration = seg.get("duration")
            # Support both 'time_range' and 'time' fields
            time_range = seg.get("time_range") or seg.get("time", "0:00-0:05")

            if not seg_duration or seg_duration == 0:
                # Try to calculate from start/end seconds
                start_sec = seg.get("start_seconds", 0)
                end_sec = seg.get("end_seconds", 0)
                if end_sec > start_sec:
                    seg_duration = end_sec - start_sec
                else:
                    # Parse from time_range/time field
                    try:
                        start_str, end_str = time_range.split("-")
                        seg_duration = self._time_to_seconds(end_str) - self._time_to_seconds(start_str)
                    except:
                        seg_duration = 10  # Default 10 seconds

            print(f"  Scene {i+1}: duration={seg_duration}s, time={time_range}")

            scene = Scene(
                id=scene_id,
                order=i + 1,
                start_time=seg.get("start_seconds", 0),
                duration=seg_duration,
                scene_type=SceneType.VIDEO,
                description=seg.get("visual", ""),
                search_keywords=keywords,
                text_overlay=None,  # Could add key phrases here
                transition="fade" if i > 0 else "none"
            )
            scenes.append(scene)
            voiceover_parts.append(seg.get("audio", ""))

        total_scene_duration = sum(s.duration for s in scenes)
        print(f"Total scene duration: {total_scene_duration}s for {len(scenes)} scenes (target: {script_data.get('total_duration', 0)}s)")

        project = VideoProject(
            id=project_id,
            title=script_data.get("title", "Untitled"),
            description=script_data.get("hook", ""),
            duration=script_data.get("total_duration", 30),
            format=format,
            style="scripted",
            script="\n".join([f"{s.get('time_range') or s.get('time')}: {s.get('audio')}" for s in segments_data]),
            voiceover_text=" ".join(voiceover_parts),
            voice_id=voice_id,
            music_style=script_data.get("music_mood", "cinematic"),
            scenes=scenes,
            status="planned"
        )

        return project

    def _extract_keywords(self, visual_description: str) -> List[str]:
        """Extract searchable keywords from visual description"""
        # Remove common words and extract key phrases
        stop_words = {"a", "an", "the", "is", "are", "of", "to", "in", "on", "with", "for", "and", "or", "showing", "shows", "displayed", "text"}

        # Clean and split
        words = re.findall(r'\b\w+\b', visual_description.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Take top keywords
        return keywords[:5] if keywords else ["video", "content"]
