"""
Matplotlib Renderer Service
Generates charts and data visualizations using Matplotlib.
"""

import os
import json
import uuid
import time
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import numpy as np

from openai import AsyncOpenAI

from ..models.visual_models import (
    DiagramType,
    DiagramStyle,
    RenderFormat,
    MatplotlibChart,
    DataSeries,
    DiagramResult,
)


class MatplotlibRenderer:
    """
    Renders charts and data visualizations using Matplotlib.
    Can generate chart specifications from natural language.
    """

    # Style mappings
    STYLE_MAP = {
        DiagramStyle.DARK: "dark_background",
        DiagramStyle.LIGHT: "seaborn-v0_8-whitegrid",
        DiagramStyle.NEUTRAL: "seaborn-v0_8-muted",
        DiagramStyle.COLORFUL: "seaborn-v0_8-bright",
        DiagramStyle.MINIMAL: "seaborn-v0_8-white",
        DiagramStyle.CORPORATE: "seaborn-v0_8-paper",
    }

    # Color palettes
    PALETTES = {
        DiagramStyle.DARK: ["#00d4ff", "#ff6b6b", "#4ecdc4", "#ffe66d", "#95e1d3", "#f38181"],
        DiagramStyle.LIGHT: ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"],
        DiagramStyle.NEUTRAL: ["#5d6d7e", "#aab7b8", "#85929e", "#566573", "#7f8c8d", "#95a5a6"],
        DiagramStyle.COLORFUL: ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"],
        DiagramStyle.MINIMAL: ["#2c3e50", "#7f8c8d", "#bdc3c7", "#95a5a6", "#34495e", "#1abc9c"],
        DiagramStyle.CORPORATE: ["#2c3e50", "#3498db", "#27ae60", "#e67e22", "#8e44ad", "#16a085"],
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        output_dir: str = "/tmp/viralify/charts"
    ):
        """Initialize the Matplotlib renderer."""
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_from_description(
        self,
        description: str,
        chart_type: DiagramType,
        style: DiagramStyle = DiagramStyle.DARK,
        context: Optional[str] = None,
        language: str = "en"
    ) -> MatplotlibChart:
        """
        Generate chart specification from natural language description.
        """
        if not self.client:
            raise ValueError("OpenAI API key required for chart generation")

        system_prompt = f"""You are a data visualization expert. Generate chart specifications as JSON.

Output format:
{{
    "chart_type": "{chart_type.value}",
    "title": "Chart title",
    "x_label": "X axis label",
    "y_label": "Y axis label",
    "x_values": ["label1", "label2", ...] or [1, 2, 3, ...],
    "data_series": [
        {{"name": "Series 1", "values": [10, 20, 30, ...]}},
        {{"name": "Series 2", "values": [15, 25, 35, ...]}}
    ],
    "legend": true,
    "grid": true,
    "annotations": [
        {{"x": 0, "y": 10, "text": "Important point"}}
    ]
}}

Rules:
1. Generate realistic but illustrative sample data
2. Use 5-10 data points for clarity
3. Labels in {language}
4. Make data tell a clear story
5. Include 1-3 data series max"""

        user_content = f"Create a {chart_type.value} chart showing: {description}"
        if context:
            user_content += f"\nContext: {context[:500]}"

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=800
        )

        spec = json.loads(response.choices[0].message.content)

        # Convert to MatplotlibChart
        data_series = [
            DataSeries(
                name=ds.get("name", f"Series {i+1}"),
                values=ds.get("values", []),
                color=ds.get("color"),
                style=ds.get("style")
            )
            for i, ds in enumerate(spec.get("data_series", []))
        ]

        return MatplotlibChart(
            chart_type=chart_type,
            title=spec.get("title", "Chart"),
            x_label=spec.get("x_label"),
            y_label=spec.get("y_label"),
            x_values=spec.get("x_values"),
            data_series=data_series,
            legend=spec.get("legend", True),
            grid=spec.get("grid", True),
            annotations=spec.get("annotations"),
            style=self.STYLE_MAP.get(style, "dark_background")
        )

    def _setup_style(self, style: DiagramStyle, fig: Figure, ax: plt.Axes):
        """Apply visual style to the chart."""
        # Get colors
        palette = self.PALETTES.get(style, self.PALETTES[DiagramStyle.DARK])

        if style in [DiagramStyle.DARK, DiagramStyle.COLORFUL]:
            # Dark theme settings
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.grid(color='#333333', alpha=0.5)
        else:
            # Light theme adjustments
            ax.grid(color='#cccccc', alpha=0.5)

        return palette

    def render(
        self,
        chart: MatplotlibChart,
        style: DiagramStyle = DiagramStyle.DARK,
        format: RenderFormat = RenderFormat.PNG,
        width: int = 1920,
        height: int = 1080,
        dpi: int = 150
    ) -> DiagramResult:
        """
        Render a Matplotlib chart to an image file.
        """
        start_time = time.time()

        try:
            # Calculate figure size in inches
            fig_width = width / dpi
            fig_height = height / dpi

            # Apply matplotlib style
            plt.style.use(chart.style)

            # Create figure
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

            # Setup custom style
            palette = self._setup_style(style, fig, ax)

            # Get x values
            x_values = chart.x_values
            if x_values is None:
                if chart.data_series:
                    x_values = list(range(len(chart.data_series[0].values)))

            # Render based on chart type
            if chart.chart_type == DiagramType.LINE_CHART:
                self._render_line_chart(ax, chart, x_values, palette)
            elif chart.chart_type == DiagramType.BAR_CHART:
                self._render_bar_chart(ax, chart, x_values, palette)
            elif chart.chart_type == DiagramType.PIE_CHART:
                self._render_pie_chart(ax, chart, palette)
            elif chart.chart_type == DiagramType.SCATTER_PLOT:
                self._render_scatter_plot(ax, chart, x_values, palette)
            elif chart.chart_type == DiagramType.HISTOGRAM:
                self._render_histogram(ax, chart, palette)
            elif chart.chart_type == DiagramType.HEATMAP:
                self._render_heatmap(ax, chart, palette)
            elif chart.chart_type == DiagramType.BOX_PLOT:
                self._render_box_plot(ax, chart, palette)
            else:
                # Default to line chart
                self._render_line_chart(ax, chart, x_values, palette)

            # Set labels and title
            ax.set_title(chart.title, fontsize=18, fontweight='bold', pad=20)
            if chart.x_label:
                ax.set_xlabel(chart.x_label, fontsize=12)
            if chart.y_label:
                ax.set_ylabel(chart.y_label, fontsize=12)

            # Add grid
            if chart.grid:
                ax.grid(True, alpha=0.3)

            # Add legend
            if chart.legend and len(chart.data_series) > 1:
                legend = ax.legend(loc='upper right', framealpha=0.9)
                if style in [DiagramStyle.DARK, DiagramStyle.COLORFUL]:
                    legend.get_frame().set_facecolor('#2d2d2d')
                    for text in legend.get_texts():
                        text.set_color('white')

            # Add annotations
            if chart.annotations:
                for ann in chart.annotations:
                    ax.annotate(
                        ann.get("text", ""),
                        xy=(ann.get("x", 0), ann.get("y", 0)),
                        fontsize=10,
                        color='white' if style == DiagramStyle.DARK else 'black'
                    )

            # Tight layout
            plt.tight_layout()

            # Save to file
            file_id = str(uuid.uuid4())[:8]
            extension = format.value
            file_path = self.output_dir / f"chart_{file_id}.{extension}"

            fig.savefig(
                file_path,
                format=extension,
                dpi=dpi,
                bbox_inches='tight',
                facecolor=fig.get_facecolor(),
                edgecolor='none'
            )

            plt.close(fig)

            generation_time = int((time.time() - start_time) * 1000)

            return DiagramResult(
                success=True,
                diagram_type=chart.chart_type,
                file_path=str(file_path),
                file_url=None,
                width=width,
                height=height,
                format=format,
                generation_time_ms=generation_time,
                metadata={
                    "chart_title": chart.title,
                    "series_count": len(chart.data_series)
                }
            )

        except Exception as e:
            plt.close('all')
            generation_time = int((time.time() - start_time) * 1000)
            return DiagramResult(
                success=False,
                diagram_type=chart.chart_type,
                width=width,
                height=height,
                format=format,
                generation_time_ms=generation_time,
                error=str(e)
            )

    def _render_line_chart(
        self,
        ax: plt.Axes,
        chart: MatplotlibChart,
        x_values: List,
        palette: List[str]
    ):
        """Render a line chart."""
        for i, series in enumerate(chart.data_series):
            color = series.color or palette[i % len(palette)]
            line_style = series.style or '-'
            ax.plot(
                x_values[:len(series.values)],
                series.values,
                label=series.name,
                color=color,
                linestyle=line_style,
                linewidth=2,
                marker='o',
                markersize=6
            )

    def _render_bar_chart(
        self,
        ax: plt.Axes,
        chart: MatplotlibChart,
        x_values: List,
        palette: List[str]
    ):
        """Render a bar chart."""
        n_series = len(chart.data_series)
        bar_width = 0.8 / n_series
        x = np.arange(len(x_values))

        for i, series in enumerate(chart.data_series):
            color = series.color or palette[i % len(palette)]
            offset = (i - n_series / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                series.values[:len(x_values)],
                bar_width,
                label=series.name,
                color=color,
                alpha=0.9
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_values, rotation=45, ha='right')

    def _render_pie_chart(
        self,
        ax: plt.Axes,
        chart: MatplotlibChart,
        palette: List[str]
    ):
        """Render a pie chart."""
        if chart.data_series:
            series = chart.data_series[0]
            labels = chart.x_values or [f"Item {i+1}" for i in range(len(series.values))]
            colors = [palette[i % len(palette)] for i in range(len(series.values))]

            wedges, texts, autotexts = ax.pie(
                series.values,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=[0.02] * len(series.values)
            )

            # Style the text
            for text in texts + autotexts:
                text.set_fontsize(10)

    def _render_scatter_plot(
        self,
        ax: plt.Axes,
        chart: MatplotlibChart,
        x_values: List,
        palette: List[str]
    ):
        """Render a scatter plot."""
        for i, series in enumerate(chart.data_series):
            color = series.color or palette[i % len(palette)]
            ax.scatter(
                x_values[:len(series.values)],
                series.values,
                label=series.name,
                color=color,
                s=80,
                alpha=0.7
            )

    def _render_histogram(
        self,
        ax: plt.Axes,
        chart: MatplotlibChart,
        palette: List[str]
    ):
        """Render a histogram."""
        for i, series in enumerate(chart.data_series):
            color = series.color or palette[i % len(palette)]
            ax.hist(
                series.values,
                bins=10,
                label=series.name,
                color=color,
                alpha=0.7,
                edgecolor='white'
            )

    def _render_heatmap(
        self,
        ax: plt.Axes,
        chart: MatplotlibChart,
        palette: List[str]
    ):
        """Render a heatmap."""
        # Convert data series to 2D array
        data = np.array([s.values for s in chart.data_series])
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax)

        # Set labels
        if chart.x_values:
            ax.set_xticks(range(len(chart.x_values)))
            ax.set_xticklabels(chart.x_values, rotation=45, ha='right')
        y_labels = [s.name for s in chart.data_series]
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

    def _render_box_plot(
        self,
        ax: plt.Axes,
        chart: MatplotlibChart,
        palette: List[str]
    ):
        """Render a box plot."""
        data = [s.values for s in chart.data_series]
        labels = [s.name for s in chart.data_series]

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(palette[i % len(palette)])
            patch.set_alpha(0.7)

    async def generate_and_render(
        self,
        description: str,
        chart_type: DiagramType,
        style: DiagramStyle = DiagramStyle.DARK,
        format: RenderFormat = RenderFormat.PNG,
        width: int = 1920,
        height: int = 1080,
        context: Optional[str] = None,
        language: str = "en"
    ) -> DiagramResult:
        """
        Generate chart specification from description and render to image.
        """
        # Generate the chart specification
        chart = await self.generate_from_description(
            description=description,
            chart_type=chart_type,
            style=style,
            context=context,
            language=language
        )

        # Render to image
        return self.render(
            chart=chart,
            style=style,
            format=format,
            width=width,
            height=height
        )
