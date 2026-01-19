"""
Animation Agent

Creates animations timed precisely to audio timestamps.
Handles typing animations, reveals, transitions, and highlights.
"""

import os
import json
import math
import subprocess
import tempfile
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult, WordTimestamp


@dataclass
class AnimationKeyframe:
    """A keyframe in an animation sequence"""
    time: float  # seconds
    property: str  # what property to animate
    value: Any  # the value at this keyframe
    easing: str = "linear"  # easing function


@dataclass
class AnimationSequence:
    """A complete animation sequence for an element"""
    element_id: str
    element_type: str
    start_time: float
    duration: float
    keyframes: List[AnimationKeyframe]
    output_path: Optional[str] = None


class AnimationAgent(BaseAgent):
    """Creates precisely timed animations for visual elements"""

    def __init__(self):
        super().__init__("ANIMATION")
        self.typing_speed = float(os.getenv("TYPING_CHARS_PER_SEC", "30"))
        self.output_dir = os.getenv("ANIMATION_OUTPUT_DIR", "/tmp/animations")

    async def execute(self, state: Dict[str, Any]) -> AgentResult:
        """Create animations for a scene"""
        slide_data = state.get("slide_data", {})
        visual_elements = state.get("visual_elements", [])
        sync_map = state.get("sync_map", {})
        word_timestamps = state.get("word_timestamps", [])
        audio_duration = state.get("audio_duration", 0)
        scene_index = state.get("scene_index", 0)
        job_id = state.get("job_id", "unknown")

        self.log(f"Scene {scene_index}: Creating animations for {len(visual_elements)} elements")

        try:
            animations = []

            # Process each visual element that needs animation
            for i, element in enumerate(visual_elements):
                element_type = element.get("element_type", "")
                metadata = element.get("metadata", {})

                if metadata.get("needs_typing_animation"):
                    # Create typing animation for code
                    anim = await self._create_typing_animation(
                        element, sync_map, word_timestamps, job_id, scene_index, i
                    )
                    if anim:
                        animations.append(anim)

                elif element_type == "bullet":
                    # Create reveal animation for bullets
                    anim = self._create_reveal_animation(element, i)
                    animations.append(anim)

                elif element_type == "highlight":
                    # Create highlight animation
                    anim = self._create_highlight_animation(element, i)
                    animations.append(anim)

            # Create scene-level animations (transitions, etc.)
            scene_animation = self._create_scene_animation(
                slide_data, audio_duration, scene_index
            )

            self.log(f"Scene {scene_index}: Created {len(animations)} element animations")

            return AgentResult(
                success=True,
                data={
                    "animations": [
                        {
                            "element_id": anim.element_id,
                            "element_type": anim.element_type,
                            "start_time": anim.start_time,
                            "duration": anim.duration,
                            "keyframes": [
                                {
                                    "time": kf.time,
                                    "property": kf.property,
                                    "value": kf.value,
                                    "easing": kf.easing
                                }
                                for kf in anim.keyframes
                            ],
                            "output_path": anim.output_path
                        }
                        for anim in animations
                    ],
                    "scene_animation": scene_animation,
                    "total_animation_duration": audio_duration
                }
            )

        except Exception as e:
            self.log(f"Scene {scene_index}: Animation creation failed - {e}")
            return AgentResult(
                success=False,
                errors=[str(e)]
            )

    async def _create_typing_animation(
        self,
        element: Dict[str, Any],
        sync_map: Dict[str, Any],
        word_timestamps: List[Dict[str, Any]],
        job_id: str,
        scene_index: int,
        element_index: int
    ) -> Optional[AnimationSequence]:
        """Create a typing animation for code"""
        metadata = element.get("metadata", {})
        code = metadata.get("code", "")

        if not code:
            return None

        # Find when typing should start and end from sync map
        start_time = 0.5  # Default start
        end_time = 10.0  # Default end

        for point in sync_map.get("sync_points", []):
            if point.get("action") == "start_typing":
                start_time = point.get("time", start_time)
            elif point.get("action") == "end_typing":
                end_time = point.get("time", end_time)

        duration = end_time - start_time

        # Calculate typing speed to finish in time
        char_count = len(code)
        chars_per_second = char_count / duration if duration > 0 else self.typing_speed

        # Create keyframes for typing progress
        keyframes = []

        # Add keyframes at regular intervals for smooth animation
        num_keyframes = min(50, char_count)  # Cap at 50 keyframes
        interval = duration / num_keyframes

        for i in range(num_keyframes + 1):
            time = i * interval
            progress = min(1.0, i / num_keyframes)
            char_index = int(progress * char_count)

            keyframes.append(AnimationKeyframe(
                time=time,
                property="visible_chars",
                value=char_index,
                easing="linear"
            ))

        # Add cursor blinking keyframes
        cursor_keyframes = []
        blink_interval = 0.5
        for t in range(0, int(duration * 2)):
            cursor_keyframes.append(AnimationKeyframe(
                time=t * blink_interval,
                property="cursor_visible",
                value=(t % 2) == 0,
                easing="step"
            ))

        keyframes.extend(cursor_keyframes)

        # Sort keyframes by time
        keyframes.sort(key=lambda kf: (kf.time, kf.property))

        return AnimationSequence(
            element_id=f"scene_{scene_index}_code_{element_index}",
            element_type="typing",
            start_time=start_time,
            duration=duration,
            keyframes=keyframes
        )

    def _create_reveal_animation(
        self,
        element: Dict[str, Any],
        element_index: int
    ) -> AnimationSequence:
        """Create a reveal animation for bullet points"""
        start_time = element.get("start_time", 0)

        keyframes = [
            AnimationKeyframe(
                time=0,
                property="opacity",
                value=0,
                easing="ease-out"
            ),
            AnimationKeyframe(
                time=0.3,
                property="opacity",
                value=1,
                easing="ease-out"
            ),
            AnimationKeyframe(
                time=0,
                property="transform_y",
                value=20,
                easing="ease-out"
            ),
            AnimationKeyframe(
                time=0.3,
                property="transform_y",
                value=0,
                easing="ease-out"
            )
        ]

        return AnimationSequence(
            element_id=f"bullet_{element_index}",
            element_type="reveal",
            start_time=start_time,
            duration=0.3,
            keyframes=keyframes
        )

    def _create_highlight_animation(
        self,
        element: Dict[str, Any],
        element_index: int
    ) -> AnimationSequence:
        """Create a highlight animation for code lines"""
        start_time = element.get("start_time", 0)
        duration = element.get("duration", 2)

        keyframes = [
            # Fade in highlight
            AnimationKeyframe(
                time=0,
                property="highlight_opacity",
                value=0,
                easing="ease-in"
            ),
            AnimationKeyframe(
                time=0.2,
                property="highlight_opacity",
                value=0.3,
                easing="ease-in"
            ),
            # Pulse effect
            AnimationKeyframe(
                time=0.5,
                property="highlight_opacity",
                value=0.5,
                easing="ease-in-out"
            ),
            AnimationKeyframe(
                time=1.0,
                property="highlight_opacity",
                value=0.3,
                easing="ease-in-out"
            ),
            # Fade out
            AnimationKeyframe(
                time=duration - 0.2,
                property="highlight_opacity",
                value=0.3,
                easing="ease-out"
            ),
            AnimationKeyframe(
                time=duration,
                property="highlight_opacity",
                value=0,
                easing="ease-out"
            )
        ]

        return AnimationSequence(
            element_id=f"highlight_{element_index}",
            element_type="highlight",
            start_time=start_time,
            duration=duration,
            keyframes=keyframes
        )

    def _create_scene_animation(
        self,
        slide_data: Dict[str, Any],
        audio_duration: float,
        scene_index: int
    ) -> Dict[str, Any]:
        """Create scene-level animations (fade in/out, transitions)"""
        slide_type = slide_data.get("type", "content")

        # Determine transition style based on slide type
        if slide_type == "title":
            in_transition = "fade"
            in_duration = 0.8
        elif slide_type == "code" or slide_type == "code_demo":
            in_transition = "wipe_right"
            in_duration = 0.3
        else:
            in_transition = "fade"
            in_duration = 0.5

        return {
            "scene_id": f"scene_{scene_index}",
            "duration": audio_duration,
            "transitions": {
                "in": {
                    "type": in_transition,
                    "duration": in_duration,
                    "easing": "ease-out"
                },
                "out": {
                    "type": "fade",
                    "duration": 0.3,
                    "easing": "ease-in"
                }
            },
            "background": {
                "type": "gradient" if slide_type == "title" else "solid",
                "color": "#1a1a2e"
            }
        }

    async def render_animation_to_video(
        self,
        animation: AnimationSequence,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30
    ) -> Optional[str]:
        """Render an animation sequence to a video file using FFmpeg"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(
                self.output_dir,
                f"{animation.element_id}.mp4"
            )

            # For typing animations, create frame-by-frame video
            if animation.element_type == "typing":
                return await self._render_typing_animation(
                    animation, output_path, width, height, fps
                )

            return output_path

        except Exception as e:
            self.log(f"Animation render failed: {e}")
            return None

    async def _render_typing_animation(
        self,
        animation: AnimationSequence,
        output_path: str,
        width: int,
        height: int,
        fps: int
    ) -> str:
        """Render typing animation to video"""
        # This would use FFmpeg with text drawing
        # For now, return the animation data for compositor to handle
        animation.output_path = output_path
        return output_path
