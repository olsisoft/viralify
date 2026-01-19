# Visual Generator Module

Generates diagrams, charts, and animations for educational video content.

## Features

- **DiagramDetector**: Analyzes content to detect if visualization is needed
- **MermaidRenderer**: Generates flowcharts, sequence diagrams, architecture diagrams via mermaid.ink API
- **MatplotlibRenderer**: Generates charts, plots, and data visualizations
- **ManimRenderer**: Generates complex mathematical and algorithmic animations

## Quick Start

```python
from visual_generator import VisualGeneratorService, VisualGenerationRequest

async with VisualGeneratorService() as generator:
    result = await generator.generate(
        VisualGenerationRequest(
            content="Kafka architecture with producers, brokers, and consumers",
            slide_type="concept",
            style="dark"
        )
    )

    if result.success:
        print(f"Generated: {result.file_path}")
        print(f"Type: {result.visual_type}")
        print(f"Renderer: {result.renderer_used}")
```

## Diagram Types

### Mermaid (Static Diagrams)
- `flowchart` - Process flows, decision trees
- `sequence` - API calls, message passing
- `class` - OOP class hierarchies
- `state` - State machines
- `er` - Database schemas
- `architecture` - System architecture
- `pie` - Distribution charts
- `mindmap` - Concept maps
- `timeline` - Historical events

### Matplotlib (Data Visualization)
- `line_chart` - Trends over time
- `bar_chart` - Category comparisons
- `scatter` - Correlations
- `histogram` - Distributions
- `heatmap` - Matrix data
- `box_plot` - Statistical summaries

### Manim (Animations)
- `algorithm` - Sorting, searching visualizations
- `data_structure` - Trees, graphs, linked lists
- `math` - Mathematical formulas and transforms
- `animation` - Custom animated explanations

## Integration with Presentation Generator

```python
# In presentation-generator/services/slide_generator.py

from visual_generator import VisualGeneratorService

class SlideGeneratorService:
    def __init__(self):
        self.visual_generator = VisualGeneratorService()

    async def generate_slide(self, slide_spec):
        # Generate visual if needed
        result = await self.visual_generator.generate_from_slide(
            slide_content=slide_spec,
            lesson_context=self.current_lesson_context
        )

        if result.success and result.file_path:
            slide_spec["visual_asset"] = result.file_path

        return slide_spec
```

## Configuration

```bash
# Required
export OPENAI_API_KEY=your_key_here

# Optional
export VISUAL_OUTPUT_DIR=/tmp/viralify/visuals
```

## Dependencies

```bash
pip install -r requirements.txt

# For Manim animations (optional)
pip install manim
```
