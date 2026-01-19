# Integration Guide: Visual Generator & Curriculum Enforcer

This guide explains how to integrate the two new modules into the existing Viralify architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      COURSE GENERATOR (8007)                     │
│                                                                  │
│  ┌──────────────────┐    ┌────────────────────────────────────┐ │
│  │  Course Planner  │───>│  Curriculum Enforcer (NEW)         │ │
│  └──────────────────┘    │  - Validates lesson structure      │ │
│           │              │  - Auto-fixes missing phases       │ │
│           │              │  - Applies Education/Enterprise    │ │
│           ▼              └────────────────────────────────────┘ │
│  ┌──────────────────┐                                           │
│  │ Generate Outline │                                           │
│  └──────────────────┘                                           │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PRESENTATION GENERATOR (8006)                    │
│                                                                  │
│  ┌──────────────────┐    ┌────────────────────────────────────┐ │
│  │  Slide Generator │───>│  Visual Generator (NEW)            │ │
│  └──────────────────┘    │  - Detects diagram needs           │ │
│           │              │  - Generates Mermaid/Matplotlib    │ │
│           │              │  - Creates Manim animations        │ │
│           ▼              └────────────────────────────────────┘ │
│  ┌──────────────────┐                                           │
│  │  Audio + Sync    │                                           │
│  └──────────────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │   FFmpeg Comp    │                                           │
│  └──────────────────┘                                           │
└───────────────────────────────────────────────────────────────────┘
```

## Step 1: Install Modules

```bash
# From the viralify root directory
cd services

# Install Visual Generator dependencies
cd visual-generator
pip install -r requirements.txt
cd ..

# Install Curriculum Enforcer dependencies
cd curriculum-enforcer
pip install -r requirements.txt
cd ..

# Optional: Install Manim for animations
pip install manim
```

## Step 2: Integrate Curriculum Enforcer into Course Generator

### File: `services/course-generator/services/course_planner.py`

```python
# Add import at the top
import sys
sys.path.insert(0, '../curriculum-enforcer')

from curriculum_enforcer import (
    CurriculumEnforcerService,
    ContextType,
    LessonContent,
    EnforcementRequest
)

class CoursePlannerService:
    def __init__(self, ...):
        # ... existing init ...
        self.curriculum_enforcer = CurriculumEnforcerService()

    async def generate_course(
        self,
        request: CourseGenerationRequest,
        context_type: str = "education"  # NEW PARAMETER
    ):
        # Map string to enum
        ctx = ContextType(context_type) if context_type else ContextType.EDUCATION

        # ... existing outline generation ...

        # NEW: Enforce curriculum structure on each lecture
        for section in outline.sections:
            for lecture in section.lectures:
                lecture_content = LessonContent(
                    lesson_id=lecture.id,
                    title=lecture.title,
                    slides=lecture.slides
                )

                result = await self.curriculum_enforcer.enforce(
                    EnforcementRequest(
                        content=lecture_content,
                        context_type=ctx,
                        auto_fix=True
                    )
                )

                if result.restructured_content:
                    lecture.slides = result.restructured_content.slides
                    print(f"[Enforcer] Restructured: {lecture.title}")

        # ... rest of generation ...
```

### Add API Parameter

```python
# In main.py, update the generate endpoint
@app.post("/api/v1/courses/generate")
async def generate_course(
    request: CourseGenerationRequest,
    context_type: str = Query(default="education", enum=["education", "enterprise", "bootcamp", "tutorial"])
):
    # Pass context_type to planner
    ...
```

## Step 3: Integrate Visual Generator into Presentation Generator

### File: `services/presentation-generator/services/slide_generator.py`

```python
# Add import at the top
import sys
sys.path.insert(0, '../visual-generator')

from visual_generator import (
    VisualGeneratorService,
    VisualGenerationRequest,
    DiagramStyle
)

class SlideGeneratorService:
    def __init__(self, ...):
        # ... existing init ...
        self.visual_generator = VisualGeneratorService()

    async def generate_slide(self, slide_spec: dict, lesson_context: str):
        # ... existing slide generation ...

        # NEW: Check if visual is needed and generate
        visual_request = VisualGenerationRequest(
            content=slide_spec.get("content", "") + " " + slide_spec.get("voiceover", ""),
            slide_type=slide_spec.get("type", ""),
            lesson_context=lesson_context,
            style=DiagramStyle.DARK,
            width=1920,
            height=1080
        )

        visual_result = await self.visual_generator.generate(visual_request)

        if visual_result.success and visual_result.file_path:
            # Add visual asset to slide
            slide_spec["visual_asset"] = {
                "path": visual_result.file_path,
                "type": visual_result.visual_type.value,
                "renderer": visual_result.renderer_used,
                "duration": visual_result.duration_seconds
            }

            print(f"[Visual] Generated {visual_result.visual_type} for: {slide_spec.get('title')}")

        return slide_spec
```

### Update Compositor to Include Visuals

```python
# In presentation_compositor.py

async def compose_slide(self, slide: dict, audio_path: str):
    # Get visual asset if present
    visual_asset = slide.get("visual_asset")

    if visual_asset:
        visual_path = visual_asset["path"]
        if visual_asset["renderer"] == "manim":
            # It's a video, overlay it
            await self._overlay_video(visual_path, slide["duration"])
        else:
            # It's an image, add as background or overlay
            await self._add_visual_background(visual_path)

    # ... rest of composition ...
```

## Step 4: Update Docker Configuration

### Add to `docker-compose.yml`

```yaml
services:
  presentation-generator:
    # ... existing config ...
    volumes:
      - ./services/visual-generator:/app/visual-generator:ro
      - /tmp/viralify/visuals:/tmp/viralify/visuals
    environment:
      - VISUAL_OUTPUT_DIR=/tmp/viralify/visuals

  course-generator:
    # ... existing config ...
    volumes:
      - ./services/curriculum-enforcer:/app/curriculum-enforcer:ro
```

## Step 5: API Changes

### New Query Parameters

```
POST /api/v1/courses/generate
  ?context_type=education|enterprise|bootcamp|tutorial

POST /api/v1/presentations/generate/v3
  ?enable_visuals=true
  ?visual_style=dark|light|colorful
```

### New Endpoints (Optional)

```
# Visual Generator direct access
POST /api/v1/visuals/generate
GET /api/v1/visuals/types

# Curriculum templates
GET /api/v1/curriculum/templates
GET /api/v1/curriculum/templates/{context_type}/phases
```

## Example: Full Flow

```python
# 1. User requests a course with enterprise context
POST /api/v1/courses/generate?context_type=enterprise
{
    "topic": "Kubernetes for DevOps",
    "structure": {"sections": 3, "lectures_per_section": 2}
}

# 2. Course Generator creates outline
# 3. Curriculum Enforcer validates/restructures each lecture:
#    - Adds missing "hook" phase
#    - Ensures "ROI" section is present (enterprise requirement)
#    - Adds "action_items" at the end

# 4. For each lecture, Presentation Generator:
#    a. Generates slides
#    b. Visual Generator detects "Kubernetes architecture" slide
#    c. Generates Mermaid diagram showing pods, nodes, services
#    d. Compositor includes diagram in final video

# 5. Final output: Professional course with:
#    - Consistent pedagogical structure
#    - Real diagrams instead of text descriptions
#    - Business-focused flow (enterprise template)
```

## Troubleshooting

### Visual Generator Issues

```bash
# Check if mermaid.ink is accessible
curl https://mermaid.ink/img/pako:eNpLyUwvSizIUHBXKC4pysxLL84vSsxRcMzJL0rNBQBsxAoq

# Check Manim installation
manim --version

# Verify output directory permissions
ls -la /tmp/viralify/visuals
```

### Curriculum Enforcer Issues

```bash
# Check template loading
python -c "from curriculum_enforcer import get_all_templates; print([t.id for t in get_all_templates()])"

# Test validation
python services/curriculum-enforcer/examples/integration_example.py
```

## Performance Considerations

1. **Visual Generation**: Mermaid is fast (~1-2s), Matplotlib is fast (~1-2s), Manim is slow (~30-120s)
2. **Parallel Processing**: Generate visuals in parallel with audio
3. **Caching**: Cache generated diagrams by content hash
4. **Fallback**: If visual generation fails, continue with text-based slide

## Next Steps

1. Add visual generation to the V3 pipeline
2. Add context_type parameter to frontend course creation
3. Create visual assets cache/CDN
4. Add Manim templates for common algorithms
5. Implement visual regeneration on edit
