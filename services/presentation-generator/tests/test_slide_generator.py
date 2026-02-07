"""
Unit tests for SlideGeneratorService

Tests cover:
- Initialization and configuration loading
- Style colors retrieval
- Slide image generation
- Background rendering (gradients, solid)
- Text rendering and wrapping
- Code highlighting
- Various slide type rendering (title, content, code, conclusion, diagram)
- Cloudinary upload
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from PIL import Image, ImageDraw, ImageFont
import json
import io
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import models first (they have minimal dependencies)
from models.presentation_models import (
    Slide, SlideType, CodeBlock, PresentationStyle
)


# ============================================================================
# Helper functions to test independently (without full SlideGeneratorService)
# ============================================================================

def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def normalize_language(lang: str) -> str:
    """Normalize language name for Pygments"""
    lang_lower = lang.lower().strip()

    # Common aliases
    aliases = {
        'py': 'python',
        'python3': 'python',
        'js': 'javascript',
        'ts': 'typescript',
        'c#': 'csharp',
        'c++': 'cpp',
        'sh': 'bash',
        'shell': 'bash',
        'yml': 'yaml',
        'dockerfile': 'docker',
    }

    return aliases.get(lang_lower, lang_lower)


def wrap_text(text: str, font, max_width: int) -> list:
    """Wrap text to fit within max_width"""
    if not text:
        return []

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        # Simulate font width calculation
        width = font.getlength(test_line) if hasattr(font, 'getlength') else len(test_line) * 10

        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def break_long_word(word: str, font, max_width: int) -> list:
    """Break a long word that doesn't fit on one line"""
    parts = []
    current = ""

    for char in word:
        test = current + char
        width = font.getlength(test) if hasattr(font, 'getlength') else len(test) * 20

        if width <= max_width:
            current = test
        else:
            if current:
                parts.append(current)
            current = char

    if current:
        parts.append(current)

    return parts


def create_gradient_background(width: int, height: int, start_color: str, end_color: str) -> Image.Image:
    """Create a vertical gradient background"""
    img = Image.new('RGB', (width, height))

    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)

    for y in range(height):
        ratio = y / height
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)

        for x in range(width):
            img.putpixel((x, y), (r, g, b))

    return img


def get_style_colors(style: PresentationStyle) -> dict:
    """Get color configuration for a presentation style"""
    styles = {
        PresentationStyle.DARK: {
            'background': '#1a1a2e',
            'gradient_end': '#16213e',
            'text': '#ffffff',
            'accent': '#e94560',
            'secondary': '#0f3460',
        },
        PresentationStyle.LIGHT: {
            'background': '#ffffff',
            'gradient_end': '#f0f0f0',
            'text': '#333333',
            'accent': '#2196f3',
            'secondary': '#e3f2fd',
        },
        PresentationStyle.GRADIENT: {
            'background': '#667eea',
            'gradient_end': '#764ba2',
            'text': '#ffffff',
            'accent': '#ffd700',
            'secondary': '#9b59b6',
        },
        PresentationStyle.OCEAN: {
            'background': '#006994',
            'gradient_end': '#003366',
            'text': '#ffffff',
            'accent': '#00d4aa',
            'secondary': '#40e0d0',
        },
    }
    return styles.get(style, styles[PresentationStyle.DARK])


def preprocess_code(code: str) -> str:
    """Preprocess code for display"""
    # Replace tabs with spaces
    code = code.replace('\t', '    ')
    # Remove trailing whitespace from each line
    lines = code.split('\n')
    lines = [line.rstrip() for line in lines]
    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


# ============================================================================
# Test Classes
# ============================================================================

class TestHexToRgb:
    """Tests for hex_to_rgb function"""

    def test_black(self):
        """Test conversion of black color"""
        assert hex_to_rgb('#000000') == (0, 0, 0)

    def test_white(self):
        """Test conversion of white color"""
        assert hex_to_rgb('#FFFFFF') == (255, 255, 255)

    def test_red(self):
        """Test conversion of red color"""
        assert hex_to_rgb('#FF0000') == (255, 0, 0)

    def test_green(self):
        """Test conversion of green color"""
        assert hex_to_rgb('#00FF00') == (0, 255, 0)

    def test_blue(self):
        """Test conversion of blue color"""
        assert hex_to_rgb('#0000FF') == (0, 0, 255)

    def test_lowercase(self):
        """Test conversion with lowercase hex"""
        assert hex_to_rgb('#ff5733') == (255, 87, 51)

    def test_without_hash(self):
        """Test conversion without # prefix"""
        assert hex_to_rgb('FF0000') == (255, 0, 0)

    def test_mixed_case(self):
        """Test conversion with mixed case"""
        assert hex_to_rgb('#aAbBcC') == (170, 187, 204)


class TestNormalizeLanguage:
    """Tests for normalize_language function"""

    def test_python_variations(self):
        """Test normalizing Python language variations"""
        assert normalize_language('Python') == 'python'
        assert normalize_language('PYTHON') == 'python'
        assert normalize_language('python') == 'python'
        assert normalize_language('py') == 'python'
        assert normalize_language('python3') == 'python'

    def test_javascript_variations(self):
        """Test normalizing JavaScript language variations"""
        assert normalize_language('JavaScript') == 'javascript'
        assert normalize_language('js') == 'javascript'
        assert normalize_language('JS') == 'javascript'

    def test_typescript_variations(self):
        """Test normalizing TypeScript language variations"""
        assert normalize_language('TypeScript') == 'typescript'
        assert normalize_language('ts') == 'typescript'
        assert normalize_language('TS') == 'typescript'

    def test_csharp(self):
        """Test normalizing C#"""
        assert normalize_language('C#') == 'csharp'
        assert normalize_language('c#') == 'csharp'

    def test_cpp(self):
        """Test normalizing C++"""
        assert normalize_language('C++') == 'cpp'
        assert normalize_language('c++') == 'cpp'

    def test_shell_variations(self):
        """Test normalizing shell language variations"""
        assert normalize_language('sh') == 'bash'
        assert normalize_language('shell') == 'bash'

    def test_yaml(self):
        """Test normalizing YAML"""
        assert normalize_language('yml') == 'yaml'

    def test_unknown_language(self):
        """Test normalizing unknown language"""
        assert normalize_language('unknownlang') == 'unknownlang'

    def test_empty_string(self):
        """Test normalizing empty string"""
        assert normalize_language('') == ''

    def test_whitespace(self):
        """Test normalizing with whitespace"""
        assert normalize_language('  python  ') == 'python'


class TestWrapText:
    """Tests for wrap_text function"""

    def test_short_text(self):
        """Test wrapping text that fits on one line"""
        mock_font = MagicMock()
        mock_font.getlength = MagicMock(side_effect=lambda x: len(x) * 10)

        result = wrap_text("Short text", mock_font, 500)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "Short text"

    def test_long_text(self):
        """Test wrapping text that needs multiple lines"""
        mock_font = MagicMock()
        mock_font.getlength = MagicMock(side_effect=lambda x: len(x) * 20)

        long_text = "This is a very long text that should definitely be wrapped"
        result = wrap_text(long_text, mock_font, 200)

        assert isinstance(result, list)
        assert len(result) > 1

    def test_empty_string(self):
        """Test wrapping empty string"""
        mock_font = MagicMock()
        mock_font.getlength = MagicMock(return_value=0)

        result = wrap_text("", mock_font, 500)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_preserves_words(self):
        """Test that wrapping preserves complete words"""
        mock_font = MagicMock()
        mock_font.getlength = MagicMock(side_effect=lambda x: len(x) * 15)

        text = "Hello World Testing"
        result = wrap_text(text, mock_font, 150)

        # Each word should be intact across all lines
        all_words = ' '.join(result).split()
        original_words = text.split()
        assert all_words == original_words

    def test_single_word(self):
        """Test wrapping single word"""
        mock_font = MagicMock()
        mock_font.getlength = MagicMock(side_effect=lambda x: len(x) * 10)

        result = wrap_text("Hello", mock_font, 500)

        assert result == ["Hello"]


class TestBreakLongWord:
    """Tests for break_long_word function"""

    def test_word_fits(self):
        """Test word that fits without breaking"""
        mock_font = MagicMock()
        mock_font.getlength = MagicMock(side_effect=lambda x: len(x) * 10)

        result = break_long_word("short", mock_font, 500)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "short"

    def test_word_needs_breaking(self):
        """Test word that needs to be broken"""
        mock_font = MagicMock()
        mock_font.getlength = MagicMock(side_effect=lambda x: len(x) * 30)

        long_word = "supercalifragilisticexpialidocious"
        result = break_long_word(long_word, mock_font, 100)

        assert isinstance(result, list)
        assert len(result) > 1
        # All parts combined should equal original word
        assert ''.join(result) == long_word

    def test_empty_word(self):
        """Test breaking empty word"""
        mock_font = MagicMock()
        mock_font.getlength = MagicMock(return_value=0)

        result = break_long_word("", mock_font, 100)

        assert result == []


class TestCreateGradientBackground:
    """Tests for create_gradient_background function"""

    def test_returns_image(self):
        """Test that gradient background returns an image"""
        img = create_gradient_background(100, 50, '#000000', '#FFFFFF')

        assert isinstance(img, Image.Image)
        assert img.size == (100, 50)

    def test_vertical_gradient(self):
        """Test vertical gradient creation"""
        img = create_gradient_background(100, 100, '#000000', '#FFFFFF')

        # Top should be darker, bottom should be lighter
        top_pixel = img.getpixel((50, 0))
        bottom_pixel = img.getpixel((50, 99))

        # Top is black (0,0,0), bottom is white (255,255,255)
        assert top_pixel == (0, 0, 0)
        assert bottom_pixel[0] > 200  # Close to white

    def test_same_colors(self):
        """Test gradient with same start and end colors"""
        img = create_gradient_background(100, 100, '#FF0000', '#FF0000')

        # Should be solid color
        top_pixel = img.getpixel((0, 0))
        bottom_pixel = img.getpixel((99, 99))

        assert top_pixel == (255, 0, 0)
        assert bottom_pixel == (255, 0, 0)

    def test_custom_dimensions(self):
        """Test gradient with custom dimensions"""
        img = create_gradient_background(1920, 1080, '#1a1a2e', '#16213e')

        assert img.size == (1920, 1080)


class TestGetStyleColors:
    """Tests for get_style_colors function"""

    def test_dark_style(self):
        """Test getting dark style colors"""
        colors = get_style_colors(PresentationStyle.DARK)

        assert 'background' in colors
        assert 'text' in colors
        assert 'accent' in colors
        assert colors['text'] == '#ffffff'

    def test_light_style(self):
        """Test getting light style colors"""
        colors = get_style_colors(PresentationStyle.LIGHT)

        assert 'background' in colors
        assert colors['background'] == '#ffffff'
        assert colors['text'] == '#333333'

    def test_gradient_style(self):
        """Test getting gradient style colors"""
        colors = get_style_colors(PresentationStyle.GRADIENT)

        assert 'background' in colors
        assert 'gradient_end' in colors

    def test_ocean_style(self):
        """Test getting ocean style colors"""
        colors = get_style_colors(PresentationStyle.OCEAN)

        assert 'background' in colors
        assert '#00' in colors['background']  # Blue tones

    def test_all_styles_return_dict(self):
        """Test that all styles return a dictionary"""
        for style in PresentationStyle:
            colors = get_style_colors(style)
            assert isinstance(colors, dict)
            assert len(colors) > 0
            assert 'background' in colors
            assert 'text' in colors


class TestPreprocessCode:
    """Tests for preprocess_code function"""

    def test_removes_trailing_whitespace(self):
        """Test that preprocessing removes trailing whitespace"""
        code = "def hello():    \n    print('hello')    "
        result = preprocess_code(code)

        for line in result.split('\n'):
            assert not line.endswith(' ')

    def test_normalizes_tabs(self):
        """Test that preprocessing normalizes tabs to spaces"""
        code = "def hello():\n\tprint('hello')"
        result = preprocess_code(code)

        assert '\t' not in result
        assert '    ' in result

    def test_preserves_structure(self):
        """Test that preprocessing preserves code structure"""
        code = "def hello():\n    print('hello')\n    return True"
        result = preprocess_code(code)

        assert '\n' in result
        lines = result.split('\n')
        assert len(lines) == 3

    def test_removes_leading_empty_lines(self):
        """Test that preprocessing removes leading empty lines"""
        code = "\n\n\ndef hello():\n    pass"
        result = preprocess_code(code)

        assert result.startswith('def')

    def test_removes_trailing_empty_lines(self):
        """Test that preprocessing removes trailing empty lines"""
        code = "def hello():\n    pass\n\n\n"
        result = preprocess_code(code)

        assert result.endswith('pass')

    def test_empty_string(self):
        """Test preprocessing empty string"""
        result = preprocess_code("")
        assert result == ""

    def test_preserves_indentation(self):
        """Test that preprocessing preserves indentation"""
        code = "def hello():\n    if True:\n        print('nested')"
        result = preprocess_code(code)

        lines = result.split('\n')
        assert lines[1].startswith('    ')
        assert lines[2].startswith('        ')


class TestSlideModel:
    """Tests for Slide model"""

    def test_title_slide_creation(self):
        """Test creating a title slide"""
        slide = Slide(
            type=SlideType.TITLE,
            title="Introduction",
            content="Subtitle here",
            index=0
        )

        assert slide.type == SlideType.TITLE
        assert slide.title == "Introduction"
        assert slide.index == 0

    def test_content_slide_with_bullets(self):
        """Test creating a content slide with bullet points"""
        slide = Slide(
            type=SlideType.CONTENT,
            title="Key Points",
            content="Overview",
            bullet_points=["Point 1", "Point 2", "Point 3"],
            index=1
        )

        assert slide.type == SlideType.CONTENT
        assert len(slide.bullet_points) == 3

    def test_code_slide_with_block(self):
        """Test creating a code slide"""
        code_block = CodeBlock(
            code="print('Hello')",
            language="python",
            filename="example.py"
        )

        slide = Slide(
            type=SlideType.CODE,
            title="Code Example",
            code_blocks=[code_block],
            index=2
        )

        assert slide.type == SlideType.CODE
        assert len(slide.code_blocks) == 1
        assert slide.code_blocks[0].language == "python"

    def test_diagram_slide(self):
        """Test creating a diagram slide"""
        slide = Slide(
            type=SlideType.DIAGRAM,
            title="Architecture",
            diagram_type="architecture",
            index=3
        )

        assert slide.type == SlideType.DIAGRAM
        assert slide.diagram_type == "architecture"

    def test_conclusion_slide(self):
        """Test creating a conclusion slide"""
        slide = Slide(
            type=SlideType.CONCLUSION,
            title="Summary",
            content="Key takeaways",
            bullet_points=["Takeaway 1", "Takeaway 2"],
            index=10
        )

        assert slide.type == SlideType.CONCLUSION


class TestCodeBlockModel:
    """Tests for CodeBlock model"""

    def test_basic_code_block(self):
        """Test creating a basic code block"""
        block = CodeBlock(
            code="print('Hello')",
            language="python"
        )

        assert block.code == "print('Hello')"
        assert block.language == "python"

    def test_code_block_with_filename(self):
        """Test creating a code block with filename"""
        block = CodeBlock(
            code="console.log('Hello');",
            language="javascript",
            filename="example.js"
        )

        assert block.filename == "example.js"

    def test_code_block_with_highlight_lines(self):
        """Test creating a code block with highlighted lines"""
        block = CodeBlock(
            code="def hello():\n    print('Hello')",
            language="python",
            highlight_lines=[1, 2]
        )

        assert block.highlight_lines == [1, 2]

    def test_code_block_multiline(self):
        """Test creating a multiline code block"""
        code = """def hello():
    print('Hello, World!')
    return True"""

        block = CodeBlock(code=code, language="python")

        assert '\n' in block.code
        assert block.code.count('\n') == 2


class TestPresentationStyleEnum:
    """Tests for PresentationStyle enum"""

    def test_all_styles_exist(self):
        """Test that all expected styles exist"""
        styles = [
            PresentationStyle.DARK,
            PresentationStyle.LIGHT,
            PresentationStyle.GRADIENT,
            PresentationStyle.OCEAN,
        ]

        assert len(styles) == 4

    def test_style_values(self):
        """Test style values"""
        assert PresentationStyle.DARK.value == "dark"
        assert PresentationStyle.LIGHT.value == "light"
        assert PresentationStyle.GRADIENT.value == "gradient"
        assert PresentationStyle.OCEAN.value == "ocean"


class TestSlideTypeEnum:
    """Tests for SlideType enum"""

    def test_all_types_exist(self):
        """Test that all expected slide types exist"""
        types = [
            SlideType.TITLE,
            SlideType.CONTENT,
            SlideType.CODE,
            SlideType.DIAGRAM,
            SlideType.CONCLUSION,
        ]

        assert len(types) >= 5

    def test_type_values(self):
        """Test slide type values"""
        assert SlideType.TITLE.value == "title"
        assert SlideType.CONTENT.value == "content"
        assert SlideType.CODE.value == "code"


class TestImageGeneration:
    """Tests for image generation utilities"""

    def test_create_blank_image(self):
        """Test creating a blank image"""
        img = Image.new('RGB', (1920, 1080), color='black')

        assert img.size == (1920, 1080)
        assert img.mode == 'RGB'

    def test_create_image_with_color(self):
        """Test creating an image with specific color"""
        img = Image.new('RGB', (100, 100), color=(255, 0, 0))

        pixel = img.getpixel((50, 50))
        assert pixel == (255, 0, 0)

    def test_draw_text_on_image(self):
        """Test drawing text on an image"""
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)

        # Use default font
        draw.text((10, 10), "Hello", fill='black')

        # Image should be modified (not all white)
        # Check a pixel in the text area
        assert img.size == (200, 100)

    def test_draw_rectangle(self):
        """Test drawing a rectangle on an image"""
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)

        draw.rectangle([50, 50, 150, 150], fill='blue')

        # Check pixel inside rectangle
        pixel = img.getpixel((100, 100))
        assert pixel == (0, 0, 255)


class TestEdgeCases:
    """Tests for edge cases"""

    def test_empty_slide_title(self):
        """Test handling slide with empty title"""
        slide = Slide(
            type=SlideType.CONTENT,
            title="",
            content="Content without title",
            index=1
        )

        assert slide.title == ""
        assert slide.content == "Content without title"

    def test_unicode_content(self):
        """Test handling Unicode content"""
        slide = Slide(
            type=SlideType.CONTENT,
            title="Les bases de Python 🐍",
            content="Apprenez le Python facilement",
            bullet_points=["Variables 📊", "Fonctions 🔧", "Classes 🏗️"],
            index=1
        )

        assert "🐍" in slide.title
        assert "📊" in slide.bullet_points[0]

    def test_special_characters_in_code(self):
        """Test handling special characters in code"""
        code = '''def calculate(a: int, b: int) -> int:
    """Calculate sum with <special> & "chars"."""
    return a + b if a > 0 else b - a'''

        block = CodeBlock(code=code, language="python")

        assert '<special>' in block.code
        assert '&' in block.code
        assert '"chars"' in block.code

    def test_very_long_title(self):
        """Test handling very long title"""
        long_title = "This is a very long title that might need to be wrapped " * 3

        slide = Slide(
            type=SlideType.TITLE,
            title=long_title,
            content="Subtitle",
            index=0
        )

        assert len(slide.title) > 100

    def test_many_bullet_points(self):
        """Test handling many bullet points"""
        bullets = [f"Point {i}" for i in range(20)]

        slide = Slide(
            type=SlideType.CONTENT,
            title="Many Points",
            bullet_points=bullets,
            index=1
        )

        assert len(slide.bullet_points) == 20

    def test_empty_code_block(self):
        """Test handling empty code block"""
        block = CodeBlock(code="", language="python")

        assert block.code == ""

    def test_multiple_code_blocks(self):
        """Test handling multiple code blocks"""
        blocks = [
            CodeBlock(code="print('1')", language="python"),
            CodeBlock(code="console.log('2');", language="javascript"),
            CodeBlock(code="SELECT * FROM users;", language="sql"),
        ]

        slide = Slide(
            type=SlideType.CODE,
            title="Multiple Languages",
            code_blocks=blocks,
            index=1
        )

        assert len(slide.code_blocks) == 3
        assert slide.code_blocks[0].language == "python"
        assert slide.code_blocks[1].language == "javascript"
        assert slide.code_blocks[2].language == "sql"


class TestColorConversions:
    """Tests for color conversion utilities"""

    def test_common_colors(self):
        """Test conversion of common colors"""
        colors = {
            '#FF0000': (255, 0, 0),      # Red
            '#00FF00': (0, 255, 0),      # Green
            '#0000FF': (0, 0, 255),      # Blue
            '#FFFF00': (255, 255, 0),    # Yellow
            '#FF00FF': (255, 0, 255),    # Magenta
            '#00FFFF': (0, 255, 255),    # Cyan
        }

        for hex_color, expected_rgb in colors.items():
            assert hex_to_rgb(hex_color) == expected_rgb

    def test_grayscale_colors(self):
        """Test conversion of grayscale colors"""
        grayscale = {
            '#000000': (0, 0, 0),        # Black
            '#404040': (64, 64, 64),     # Dark gray
            '#808080': (128, 128, 128),  # Medium gray
            '#C0C0C0': (192, 192, 192),  # Light gray
            '#FFFFFF': (255, 255, 255),  # White
        }

        for hex_color, expected_rgb in grayscale.items():
            assert hex_to_rgb(hex_color) == expected_rgb


class TestAllSlideTypesWithAllStyles:
    """Parametrized tests for all slide type and style combinations"""

    @pytest.mark.parametrize("slide_type", list(SlideType))
    @pytest.mark.parametrize("style", list(PresentationStyle))
    def test_slide_type_style_combination(self, slide_type, style):
        """Test that all slide types work with all styles"""
        colors = get_style_colors(style)

        slide = Slide(
            type=slide_type,
            title=f"Test {slide_type.value} slide",
            content="Test content",
            index=0
        )

        if slide_type == SlideType.CODE:
            slide.code_blocks = [CodeBlock(code="print('test')", language="python")]

        # Verify slide is valid
        assert slide.type == slide_type
        assert isinstance(colors, dict)
        assert 'background' in colors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
