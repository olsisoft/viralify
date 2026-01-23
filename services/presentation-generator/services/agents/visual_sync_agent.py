"""
Visual Sync Agent

Aligns visual elements to audio timestamps.
Generates actual slide images using SlideGeneratorService.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult, WordTimestamp, VisualElement

# Import slide generator for actual image generation
from models.presentation_models import Slide, SlideType, CodeBlock, PresentationStyle
from services.slide_generator import SlideGeneratorService
from services.typing_animator import TypingAnimatorService


@dataclass
class VisualCue:
    """A visual cue linked to audio timing"""
    element_type: str  # "title", "bullet", "code", "diagram", "output"
    content: str
    start_time: float
    end_time: float
    trigger_word: Optional[str] = None
    trigger_phrase: Optional[str] = None


class VisualSyncAgent(BaseAgent):
    """Aligns visual elements to audio timestamps and generates slide images/videos"""

    def __init__(self):
        super().__init__("VISUAL_SYNC")
        self.slide_generator = SlideGeneratorService()
        self.typing_animator = TypingAnimatorService()
        self.output_dir = Path(tempfile.gettempdir()) / "presentations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, state: Dict[str, Any]) -> AgentResult:
        """Generate slide visual (image or typing animation) and align to audio timestamps"""
        slide_data = state.get("slide_data", {})
        word_timestamps = state.get("word_timestamps", [])
        timing_cues = state.get("timing_cues", [])
        audio_duration = state.get("audio_duration", 0)
        scene_index = state.get("scene_index", 0)
        job_id = state.get("job_id", "unknown")
        style = state.get("style", "dark")
        target_audience = state.get("target_audience", "intermediate developers")
        target_career = state.get("target_career")  # Career for diagram focus (e.g., "data_engineer")

        slide_type = slide_data.get("type", "content")
        has_code = bool(slide_data.get("code"))

        self.log(f"Scene {scene_index}: Generating visual for {slide_type} slide (has_code={has_code})")

        try:
            # Convert word timestamps to objects if they're dicts
            word_ts = [
                WordTimestamp(**wt) if isinstance(wt, dict) else wt
                for wt in word_timestamps
            ]

            # Step 1: Generate visual based on slide type
            visual_path = None
            visual_type = "image"

            if slide_type in ["code", "code_demo"] and has_code:
                # Generate typing animation video for code slides
                visual_path, actual_duration = await self._generate_typing_animation(
                    slide_data, job_id, scene_index, style, audio_duration, target_audience, target_career
                )
                visual_type = "video"
                self.log(f"Scene {scene_index}: Created typing animation ({actual_duration:.1f}s)")
            else:
                # Generate static slide image for other slides
                visual_path = await self._generate_slide_image(
                    slide_data, job_id, scene_index, style, target_audience, target_career
                )
                visual_type = "image"

            # Step 2: Analyze timing cues and link to audio
            visual_cues = self._link_cues_to_audio(timing_cues, word_ts, slide_data)

            # Step 3: Create visual elements list
            visual_elements = []

            if visual_path:
                visual_elements.append(VisualElement(
                    element_type="slide_video" if visual_type == "video" else "slide_image",
                    file_path=visual_path,
                    url=visual_path,
                    start_time=0,
                    duration=audio_duration,
                    metadata={
                        "slide_type": slide_type,
                        "title": slide_data.get("title", ""),
                        "has_code": has_code,
                        "visual_type": visual_type
                    }
                ))

            # Step 4: Create sync map
            sync_map = self._create_sync_map(visual_cues, visual_elements, audio_duration)

            self.log(f"Scene {scene_index}: Created {visual_type} at {visual_path}")

            return AgentResult(
                success=True,
                data={
                    "visual_elements": [
                        {
                            "element_type": ve.element_type,
                            "file_path": ve.file_path,
                            "url": ve.url,
                            "start_time": ve.start_time,
                            "duration": ve.duration,
                            "metadata": ve.metadata
                        }
                        for ve in visual_elements
                    ],
                    "primary_visual_url": visual_path,
                    "primary_visual_type": visual_type,
                    "sync_map": sync_map,
                    "visual_cues": [
                        {
                            "element_type": vc.element_type,
                            "start_time": vc.start_time,
                            "end_time": vc.end_time,
                            "trigger_word": vc.trigger_word
                        }
                        for vc in visual_cues
                    ]
                }
            )

        except Exception as e:
            self.log(f"Scene {scene_index}: Visual sync failed - {e}")
            import traceback
            traceback.print_exc()
            return AgentResult(
                success=False,
                errors=[str(e)]
            )

    async def _generate_typing_animation(
        self,
        slide_data: Dict[str, Any],
        job_id: str,
        scene_index: int,
        style: str,
        target_duration: float,
        target_audience: str = "intermediate developers",
        target_career: Optional[str] = None
    ) -> tuple:
        """Generate typing animation video for code slides"""
        try:
            code = slide_data.get("code", "")
            language = slide_data.get("language", "python")
            title = slide_data.get("title", "")
            expected_output = slide_data.get("expected_output")
            slide_type = slide_data.get("type", "code")

            # Unescape literal \n to actual newlines (GPT sometimes double-escapes)
            if '\\n' in code:
                code = code.replace('\\n', '\n')
            if '\\t' in code:
                code = code.replace('\\t', '\t')

            # Get style colors
            style_colors = {
                "dark": {
                    "background": "#1e1e2e",
                    "text": "#cdd6f4",
                    "accent": "#89b4fa"
                },
                "light": {
                    "background": "#eff1f5",
                    "text": "#4c4f69",
                    "accent": "#1e66f5"
                }
            }
            colors = style_colors.get(style, style_colors["dark"])

            # Output path for the video
            output_path = str(self.output_dir / f"{job_id}_scene_{scene_index:03d}_typing.mp4")

            # Create typing animation
            # For code_demo slides, include execution output
            execution_output = expected_output if slide_type == "code_demo" else None

            video_path, actual_duration = await self.typing_animator.create_typing_animation(
                code=code,
                language=language,
                output_path=output_path,
                title=title,
                typing_speed="natural",
                target_duration=target_duration,
                execution_output=execution_output,
                background_color=colors["background"],
                text_color=colors["text"],
                accent_color=colors["accent"],
                pygments_style="monokai"
            )

            self.log(f"Scene {scene_index}: Typing animation created ({actual_duration:.1f}s)")
            return video_path, actual_duration

        except Exception as e:
            self.log(f"Scene {scene_index}: Typing animation failed - {e}")
            import traceback
            traceback.print_exc()
            # Fallback to static image
            static_path = await self._generate_slide_image(slide_data, job_id, scene_index, style, target_audience, target_career)
            return static_path, target_duration

    async def _generate_slide_image(
        self,
        slide_data: Dict[str, Any],
        job_id: str,
        scene_index: int,
        style: str,
        target_audience: str = "intermediate developers",
        target_career: Optional[str] = None
    ) -> Optional[str]:
        """Generate actual slide image using SlideGeneratorService"""
        try:
            # Convert slide_data dict to Slide model
            slide_type_str = slide_data.get("type", "content")
            slide_type_map = {
                "title": SlideType.TITLE,
                "content": SlideType.CONTENT,
                "code": SlideType.CODE,
                "code_demo": SlideType.CODE_DEMO,
                "conclusion": SlideType.CONCLUSION,
                "diagram": SlideType.DIAGRAM
            }
            slide_type = slide_type_map.get(slide_type_str, SlideType.CONTENT)

            # Build code blocks if code is present
            code_blocks = []
            if slide_data.get("code"):
                code_blocks.append(CodeBlock(
                    language=slide_data.get("language", "python"),
                    code=slide_data["code"],
                    filename=f"example.{slide_data.get('language', 'py')}",
                    expected_output=slide_data.get("expected_output")
                ))

            # Create Slide object
            slide = Slide(
                type=slide_type,
                title=slide_data.get("title", ""),
                subtitle=slide_data.get("subtitle"),
                content=slide_data.get("content"),
                bullet_points=slide_data.get("bullet_points", []),
                code_blocks=code_blocks,
                duration=slide_data.get("duration", 10),
                voiceover_text=slide_data.get("voiceover_text", "")
            )

            # Get presentation style
            style_map = {
                "dark": PresentationStyle.DARK,
                "light": PresentationStyle.LIGHT,
                "gradient": PresentationStyle.GRADIENT,
                "ocean": PresentationStyle.OCEAN
            }
            pres_style = style_map.get(style, PresentationStyle.DARK)

            # Generate the slide image with audience-based complexity and career-based focus
            image_bytes = await self.slide_generator.generate_slide_image(slide, pres_style, target_audience, target_career)

            # Save to file
            output_path = self.output_dir / f"{job_id}_scene_{scene_index:03d}.png"
            with open(output_path, "wb") as f:
                f.write(image_bytes)

            self.log(f"Scene {scene_index}: Generated slide image ({len(image_bytes)} bytes)")
            return str(output_path)

        except Exception as e:
            self.log(f"Scene {scene_index}: Slide image generation failed - {e}")
            import traceback
            traceback.print_exc()
            return None

    def _link_cues_to_audio(
        self,
        timing_cues: List[Dict[str, Any]],
        word_timestamps: List[WordTimestamp],
        slide_data: Dict[str, Any]
    ) -> List[VisualCue]:
        """Link timing cues to actual audio timestamps"""
        visual_cues = []

        if not word_timestamps:
            # No word timestamps, create basic cue
            return [VisualCue(
                element_type="slide",
                content=slide_data.get("title", ""),
                start_time=0,
                end_time=slide_data.get("duration", 10)
            )]

        # Build word index for phrase matching
        words_text = [wt.word.lower().strip(".,!?") for wt in word_timestamps]

        for cue in timing_cues:
            event_type = cue.get("event_type", "")
            target = cue.get("target", "")
            planned_timestamp = cue.get("timestamp", 0)

            # Try to find the trigger word/phrase in audio
            actual_start = planned_timestamp
            cue_duration = cue.get("duration") or 5  # Handle None explicitly
            actual_end = planned_timestamp + cue_duration

            # Search for related words in the transcript
            trigger_word = None
            if target:
                # Look for target word or related words
                target_words = target.lower().split()
                for i, word in enumerate(words_text):
                    if any(tw in word or word in tw for tw in target_words):
                        actual_start = word_timestamps[i].start
                        trigger_word = word_timestamps[i].word
                        break

            # Determine element type from event type
            element_type = self._event_to_element_type(event_type)

            visual_cues.append(VisualCue(
                element_type=element_type,
                content=target,
                start_time=actual_start,
                end_time=actual_end,
                trigger_word=trigger_word
            ))

        # If no cues, create defaults from slide structure
        if not visual_cues:
            visual_cues = self._create_default_cues(slide_data, word_timestamps)

        return visual_cues

    def _event_to_element_type(self, event_type: str) -> str:
        """Convert event type to visual element type"""
        mapping = {
            "show_title": "title",
            "show_text": "text",
            "show_bullet": "bullet",
            "show_code": "code",
            "start_typing": "code_animation",
            "show_output": "output",
            "highlight_line": "highlight",
            "show_diagram": "diagram",
            "show_image": "image"
        }
        return mapping.get(event_type, "text")

    def _create_default_cues(
        self,
        slide_data: Dict[str, Any],
        word_timestamps: List[WordTimestamp]
    ) -> List[VisualCue]:
        """Create default visual cues from slide structure"""
        cues = []
        slide_type = slide_data.get("type", "content")
        end_time = word_timestamps[-1].end if word_timestamps else 10

        # Title always shows first
        if slide_data.get("title"):
            cues.append(VisualCue(
                element_type="title",
                content=slide_data["title"],
                start_time=0,
                end_time=end_time
            ))

        # Content based on slide type
        if slide_type in ["code", "code_demo"]:
            code_content = slide_data.get("code", "")
            if code_content:
                cues.append(VisualCue(
                    element_type="code",
                    content=code_content,
                    start_time=0,
                    end_time=end_time
                ))

        elif slide_type == "content":
            bullets = slide_data.get("bullet_points", [])
            if bullets:
                bullet_interval = end_time / (len(bullets) + 1)
                for i, bullet in enumerate(bullets):
                    cues.append(VisualCue(
                        element_type="bullet",
                        content=bullet,
                        start_time=bullet_interval * (i + 1),
                        end_time=end_time
                    ))

        return cues

    def _create_sync_map(
        self,
        visual_cues: List[VisualCue],
        visual_elements: List[VisualElement],
        audio_duration: float
    ) -> Dict[str, Any]:
        """Create a complete sync map for the scene"""
        sync_points = []

        for cue in visual_cues:
            sync_points.append({
                "time": cue.start_time,
                "action": "show",
                "element_type": cue.element_type,
                "content": cue.content,
                "trigger_word": cue.trigger_word
            })

        # Sort by time
        sync_points.sort(key=lambda x: x["time"])

        return {
            "total_duration": audio_duration,
            "sync_points": sync_points,
            "visual_element_count": len(visual_elements)
        }
