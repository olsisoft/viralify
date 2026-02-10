"""
Manim Renderer Service
Generates mathematical and technical animations using Manim (3Blue1Brown's library).
"""

import os
import json
import uuid
import time
import subprocess
import tempfile
import shutil
from typing import Optional, List, Dict, Any
from pathlib import Path

from openai import AsyncOpenAI

from models.visual_models import (
    DiagramType,
    DiagramStyle,
    AnimationComplexity,
    RenderFormat,
    ManimAnimation,
    ManimScene,
    DiagramResult,
)


class ManimRenderer:
    """
    Renders complex animations using Manim.
    Generates Manim Python code from descriptions and executes it.
    """

    # Quality presets matching Manim CLI flags
    QUALITY_MAP = {
        "480p": "-ql",   # Low quality, fast render
        "720p": "-qm",   # Medium quality
        "1080p": "-qh",  # High quality
        "1440p": "-qp",  # Production quality
        "4k": "-qk",     # 4K quality
    }

    # Complexity to duration mapping (seconds)
    COMPLEXITY_DURATION = {
        AnimationComplexity.SIMPLE: (5, 10),
        AnimationComplexity.MODERATE: (10, 20),
        AnimationComplexity.COMPLEX: (20, 45),
        AnimationComplexity.CINEMATIC: (45, 90),
    }

    # Color schemes for Manim
    COLOR_SCHEMES = {
        DiagramStyle.DARK: {
            "background": "#1e1e1e",
            "primary": "#00d4ff",
            "secondary": "#ff6b6b",
            "tertiary": "#4ecdc4",
            "text": "#ffffff",
        },
        DiagramStyle.LIGHT: {
            "background": "#ffffff",
            "primary": "#3498db",
            "secondary": "#e74c3c",
            "tertiary": "#2ecc71",
            "text": "#2c3e50",
        },
        DiagramStyle.COLORFUL: {
            "background": "#1a1a2e",
            "primary": "#e94560",
            "secondary": "#0f3460",
            "tertiary": "#16c79a",
            "text": "#eaeaea",
        },
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        output_dir: str = "/tmp/viralify/animations"
    ):
        """Initialize the Manim renderer."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if manim is installed
        self.manim_available = self._check_manim()

    def _check_manim(self) -> bool:
        """Check if Manim is installed and available."""
        try:
            result = subprocess.run(
                ["manim", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    async def generate_manim_code(
        self,
        description: str,
        animation_type: DiagramType,
        complexity: AnimationComplexity = AnimationComplexity.MODERATE,
        style: DiagramStyle = DiagramStyle.DARK,
        context: Optional[str] = None,
        language: str = "en"
    ) -> ManimAnimation:
        """
        Generate Manim Python code from a description using GPT-4.
        """
        if not self.client:
            raise ValueError("OpenAI API key required for Manim code generation")

        colors = self.COLOR_SCHEMES.get(style, self.COLOR_SCHEMES[DiagramStyle.DARK])
        min_dur, max_dur = self.COMPLEXITY_DURATION[complexity]

        system_prompt = f"""You are a Manim animation expert (Community edition - manim-ce).
Generate complete, working Manim Python code for educational animations.

CRITICAL RULES:
1. Use `from manim import *` import
2. Create a single Scene class named `GeneratedScene`
3. Total duration: {min_dur}-{max_dur} seconds
4. Use self.play() for all animations
5. Include self.wait() between animations
6. Keep code simple and working - avoid complex custom classes
7. Color scheme:
   - Background: {colors['background']}
   - Primary: {colors['primary']}
   - Secondary: {colors['secondary']}
   - Text: {colors['text']}

ANIMATION TYPES:
- For algorithms: Use Rectangles/Circles for array elements, animate swaps
- For data structures: Use VGroup, arrows, labeled nodes
- For math: Use MathTex, Transform, ReplacementTransform
- For graphs: Use Graph, Vertices, Edges
- For code viz: Use Code class with syntax highlighting

OUTPUT FORMAT (JSON):
{{
    "title": "Animation title",
    "description": "What the animation shows",
    "complexity": "{complexity.value}",
    "estimated_duration": 15.0,
    "manim_code": "from manim import *\\n\\nclass GeneratedScene(Scene):\\n    def construct(self):\\n        ..."
}}

Make the animation visually engaging and educational.
Labels should be in {language}."""

        user_content = f"Create a {animation_type.value} animation showing: {description}"
        if context:
            user_content += f"\n\nContext: {context[:500]}"

        response = await self.client.chat.completions.create(
            model="gpt-4o",  # Use GPT-4 for complex code generation
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=3000
        )

        result = json.loads(response.choices[0].message.content)

        return ManimAnimation(
            title=result.get("title", "Generated Animation"),
            description=result.get("description", description),
            complexity=complexity,
            scenes=[
                ManimScene(
                    name="main",
                    description=result.get("description", ""),
                    duration_seconds=result.get("estimated_duration", 15.0)
                )
            ],
            background_color=colors["background"],
            manim_code=result.get("manim_code", "")
        )

    async def render(
        self,
        animation: ManimAnimation,
        format: RenderFormat = RenderFormat.MP4,
        resolution: str = "1080p",
        fps: int = 30
    ) -> DiagramResult:
        """
        Render a Manim animation to video file.
        """
        start_time = time.time()

        if not self.manim_available:
            return DiagramResult(
                success=False,
                diagram_type=DiagramType.ANIMATION,
                width=1920,
                height=1080,
                format=format,
                generation_time_ms=0,
                error="Manim is not installed. Install with: pip install manim"
            )

        if not animation.manim_code:
            return DiagramResult(
                success=False,
                diagram_type=DiagramType.ANIMATION,
                width=1920,
                height=1080,
                format=format,
                generation_time_ms=0,
                error="No Manim code provided"
            )

        # Create temp directory for rendering
        temp_dir = tempfile.mkdtemp(prefix="manim_")
        script_path = Path(temp_dir) / "scene.py"
        file_id = str(uuid.uuid4())[:8]

        try:
            # Write Manim code to file
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(animation.manim_code)

            # Build Manim command
            quality_flag = self.QUALITY_MAP.get(resolution, "-qh")
            output_file = self.output_dir / f"animation_{file_id}.mp4"

            cmd = [
                "manim",
                quality_flag,
                "--fps", str(fps),
                "--format", "mp4",
                "-o", str(output_file),
                str(script_path),
                "GeneratedScene"
            ]

            # Run Manim
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=temp_dir
            )

            generation_time = int((time.time() - start_time) * 1000)

            if result.returncode != 0:
                return DiagramResult(
                    success=False,
                    diagram_type=DiagramType.ANIMATION,
                    width=1920,
                    height=1080,
                    format=format,
                    generation_time_ms=generation_time,
                    error=f"Manim render failed: {result.stderr[:500]}"
                )

            # Find the output file (Manim may put it in media/ subdirectory)
            actual_output = self._find_output_file(temp_dir, output_file)

            if actual_output and actual_output.exists():
                # Move to final location
                final_path = self.output_dir / f"animation_{file_id}.mp4"
                shutil.move(str(actual_output), str(final_path))

                return DiagramResult(
                    success=True,
                    diagram_type=DiagramType.ANIMATION,
                    file_path=str(final_path),
                    width=1920 if resolution in ["1080p", "1440p", "4k"] else 1280,
                    height=1080 if resolution == "1080p" else 720,
                    format=RenderFormat.MP4,
                    generation_time_ms=generation_time,
                    metadata={
                        "title": animation.title,
                        "complexity": animation.complexity.value,
                        "duration_estimate": animation.scenes[0].duration_seconds if animation.scenes else 15.0
                    }
                )
            else:
                return DiagramResult(
                    success=False,
                    diagram_type=DiagramType.ANIMATION,
                    width=1920,
                    height=1080,
                    format=format,
                    generation_time_ms=generation_time,
                    error="Output file not found after render"
                )

        except subprocess.TimeoutExpired:
            generation_time = int((time.time() - start_time) * 1000)
            return DiagramResult(
                success=False,
                diagram_type=DiagramType.ANIMATION,
                width=1920,
                height=1080,
                format=format,
                generation_time_ms=generation_time,
                error="Manim render timed out (>5 minutes)"
            )
        except Exception as e:
            generation_time = int((time.time() - start_time) * 1000)
            return DiagramResult(
                success=False,
                diagram_type=DiagramType.ANIMATION,
                width=1920,
                height=1080,
                format=format,
                generation_time_ms=generation_time,
                error=str(e)
            )
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                pass

    def _find_output_file(self, temp_dir: str, expected_path: Path) -> Optional[Path]:
        """Find the actual output file from Manim render."""
        # Check expected path first
        if expected_path.exists():
            return expected_path

        # Search in media directory (Manim default)
        media_dir = Path(temp_dir) / "media"
        if media_dir.exists():
            for mp4_file in media_dir.rglob("*.mp4"):
                return mp4_file

        return None

    async def generate_and_render(
        self,
        description: str,
        animation_type: DiagramType,
        complexity: AnimationComplexity = AnimationComplexity.MODERATE,
        style: DiagramStyle = DiagramStyle.DARK,
        resolution: str = "1080p",
        fps: int = 30,
        context: Optional[str] = None,
        language: str = "en"
    ) -> DiagramResult:
        """
        Generate Manim code from description and render to video.
        """
        # Generate the Manim code
        animation = await self.generate_manim_code(
            description=description,
            animation_type=animation_type,
            complexity=complexity,
            style=style,
            context=context,
            language=language
        )

        # Render to video
        return await self.render(
            animation=animation,
            format=RenderFormat.MP4,
            resolution=resolution,
            fps=fps
        )


# Predefined Manim templates for common animations
class ManimTemplates:
    """Ready-to-use Manim code templates."""

    @staticmethod
    def binary_search(array: List[int], target: int) -> str:
        """Binary search algorithm visualization."""
        return f'''from manim import *

class GeneratedScene(Scene):
    def construct(self):
        # Configuration
        self.camera.background_color = "#1e1e1e"

        # Title
        title = Text("Binary Search", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Array
        array = {array}
        target = {target}

        # Create rectangles for array elements
        rects = VGroup()
        nums = VGroup()
        for i, val in enumerate(array):
            rect = Rectangle(width=0.8, height=0.8, color=BLUE, fill_opacity=0.3)
            num = Text(str(val), font_size=24, color=WHITE)
            rect.move_to(RIGHT * (i - len(array)/2 + 0.5) * 1.0)
            num.move_to(rect.get_center())
            rects.add(rect)
            nums.add(num)

        array_group = VGroup(rects, nums)
        array_group.move_to(ORIGIN)

        self.play(Create(array_group))
        self.wait(0.5)

        # Binary search animation
        left, right = 0, len(array) - 1
        while left <= right:
            mid = (left + right) // 2

            # Highlight current search range
            for i in range(len(array)):
                if left <= i <= right:
                    self.play(rects[i].animate.set_fill(YELLOW, opacity=0.5), run_time=0.3)

            # Highlight mid
            self.play(rects[mid].animate.set_fill(GREEN, opacity=0.7), run_time=0.5)
            self.wait(0.3)

            if array[mid] == target:
                # Found!
                self.play(rects[mid].animate.set_fill(GREEN, opacity=1.0))
                found_text = Text("Found!", font_size=36, color=GREEN)
                found_text.next_to(array_group, DOWN)
                self.play(Write(found_text))
                break
            elif array[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

            # Reset non-range elements
            for i in range(len(array)):
                if i < left or i > right:
                    self.play(rects[i].animate.set_fill(GRAY, opacity=0.2), run_time=0.2)

        self.wait(2)
'''

    @staticmethod
    def linked_list_operations() -> str:
        """Linked list insertion/deletion visualization."""
        return '''from manim import *

class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#1e1e1e"

        title = Text("Linked List", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))

        # Create initial nodes
        nodes = []
        arrows = []
        values = [1, 2, 3, 4]

        for i, val in enumerate(values):
            node = VGroup(
                Circle(radius=0.4, color=BLUE, fill_opacity=0.3),
                Text(str(val), font_size=24, color=WHITE)
            )
            node.move_to(LEFT * 3 + RIGHT * i * 2)
            nodes.append(node)

        # Create arrows
        for i in range(len(nodes) - 1):
            arrow = Arrow(
                nodes[i].get_right(),
                nodes[i+1].get_left(),
                color=WHITE,
                buff=0.1
            )
            arrows.append(arrow)

        # Animate creation
        for node in nodes:
            self.play(Create(node), run_time=0.5)
        for arrow in arrows:
            self.play(Create(arrow), run_time=0.3)

        self.wait(1)

        # Insert new node
        new_node = VGroup(
            Circle(radius=0.4, color=GREEN, fill_opacity=0.5),
            Text("5", font_size=24, color=WHITE)
        )
        new_node.move_to(DOWN * 2 + nodes[-1].get_center())

        insert_text = Text("Insert 5", font_size=24, color=GREEN)
        insert_text.to_edge(DOWN)
        self.play(Write(insert_text))
        self.play(Create(new_node))

        # Animate insertion
        new_arrow = Arrow(
            nodes[-1].get_right(),
            new_node.get_left(),
            color=GREEN,
            buff=0.1
        )
        self.play(Create(new_arrow))
        self.play(new_node.animate.move_to(nodes[-1].get_center() + RIGHT * 2))
        new_arrow_final = Arrow(
            nodes[-1].get_right(),
            new_node.get_left(),
            color=WHITE,
            buff=0.1
        )
        self.play(Transform(new_arrow, new_arrow_final))

        self.wait(2)
'''

    @staticmethod
    def sorting_bubble(array: List[int]) -> str:
        """Bubble sort visualization."""
        return f'''from manim import *

class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#1e1e1e"

        title = Text("Bubble Sort", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Array to sort
        array = {array}

        # Create bars
        bars = VGroup()
        max_val = max(array)
        for i, val in enumerate(array):
            bar = Rectangle(
                width=0.6,
                height=val / max_val * 3,
                color=BLUE,
                fill_opacity=0.7
            )
            bar.move_to(LEFT * 3 + RIGHT * i * 0.8)
            bar.align_to(ORIGIN, DOWN)
            bars.add(bar)

        self.play(Create(bars))
        self.wait(0.5)

        # Bubble sort with animation
        n = len(array)
        for i in range(n):
            for j in range(0, n-i-1):
                # Highlight comparison
                self.play(
                    bars[j].animate.set_color(YELLOW),
                    bars[j+1].animate.set_color(YELLOW),
                    run_time=0.2
                )

                if array[j] > array[j+1]:
                    # Swap
                    array[j], array[j+1] = array[j+1], array[j]
                    self.play(
                        bars[j].animate.move_to(bars[j+1].get_center()),
                        bars[j+1].animate.move_to(bars[j].get_center()),
                        run_time=0.3
                    )
                    bars[j], bars[j+1] = bars[j+1], bars[j]

                # Reset color
                self.play(
                    bars[j].animate.set_color(BLUE),
                    bars[j+1].animate.set_color(GREEN if j+1 >= n-i-1 else BLUE),
                    run_time=0.1
                )

        # Final state
        self.play(*[bar.animate.set_color(GREEN) for bar in bars])
        sorted_text = Text("Sorted!", font_size=36, color=GREEN)
        sorted_text.next_to(bars, DOWN)
        self.play(Write(sorted_text))
        self.wait(2)
'''

    @staticmethod
    def math_equation_transform(equation1: str, equation2: str) -> str:
        """Mathematical equation transformation."""
        return f'''from manim import *

class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#1e1e1e"

        # First equation
        eq1 = MathTex(r"{equation1}", font_size=72, color=WHITE)
        self.play(Write(eq1))
        self.wait(1)

        # Transform to second equation
        eq2 = MathTex(r"{equation2}", font_size=72, color=WHITE)
        self.play(TransformMatchingTex(eq1, eq2))
        self.wait(2)
'''

    @staticmethod
    def graph_traversal_bfs() -> str:
        """BFS graph traversal visualization."""
        return '''from manim import *

class GeneratedScene(Scene):
    def construct(self):
        self.camera.background_color = "#1e1e1e"

        title = Text("BFS Traversal", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))

        # Create graph
        vertices = [1, 2, 3, 4, 5, 6]
        edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6)]

        layout = {
            1: UP * 1.5,
            2: LEFT * 2,
            3: RIGHT * 2,
            4: LEFT * 3 + DOWN * 1.5,
            5: LEFT * 1 + DOWN * 1.5,
            6: RIGHT * 2 + DOWN * 1.5,
        }

        # Draw nodes
        nodes = {}
        for v in vertices:
            node = VGroup(
                Circle(radius=0.4, color=BLUE, fill_opacity=0.3),
                Text(str(v), font_size=24, color=WHITE)
            )
            node.move_to(layout[v])
            nodes[v] = node
            self.play(Create(node), run_time=0.3)

        # Draw edges
        edge_lines = []
        for u, v in edges:
            line = Line(
                nodes[u].get_center(),
                nodes[v].get_center(),
                color=WHITE
            )
            edge_lines.append(line)
            self.play(Create(line), run_time=0.2)

        self.wait(0.5)

        # BFS traversal
        queue = [1]
        visited = set([1])
        order = []

        while queue:
            current = queue.pop(0)
            order.append(current)

            # Highlight current node
            self.play(
                nodes[current][0].animate.set_fill(GREEN, opacity=0.7),
                run_time=0.5
            )

            # Find neighbors
            for u, v in edges:
                neighbor = v if u == current else (u if v == current else None)
                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    # Highlight edge
                    for line in edge_lines:
                        if (line.get_start() - nodes[current].get_center()).length() < 0.5:
                            self.play(line.animate.set_color(YELLOW), run_time=0.2)

        # Show traversal order
        order_text = Text(f"Order: {order}", font_size=24, color=WHITE)
        order_text.to_edge(DOWN)
        self.play(Write(order_text))
        self.wait(2)
'''
