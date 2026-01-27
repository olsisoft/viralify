"""
Practical Focus Configuration

Defines the slide type distribution and instructions based on
the practical focus level (theoretical, balanced, practical).
"""

from typing import Dict, Optional


PRACTICAL_FOCUS_CONFIG: Dict[str, dict] = {
    "theoretical": {
        "name": "Théorique (concepts)",
        "aliases": ["théorique", "theoretical", "concepts", "théorique (concepts)"],
        "slide_ratio": {
            "content": 0.50,      # 50% explanation slides
            "diagram": 0.25,      # 25% diagrams
            "code": 0.15,         # 15% code examples
            "code_demo": 0.05,    # 5% live demos
            "conclusion": 0.05,   # 5% summary
        },
        "instructions": """
═══════════════════════════════════════════════════════════════════════════════
                    THEORETICAL FOCUS - CONCEPTUAL LEARNING
═══════════════════════════════════════════════════════════════════════════════

This course emphasizes UNDERSTANDING over DOING. Follow these guidelines:

📚 CONTENT STRUCTURE:
- Prioritize deep conceptual understanding over hands-on practice
- Each concept should be explained thoroughly with WHY and HOW it works
- Use diagrams to visualize abstract concepts
- Code examples should ILLUSTRATE concepts, not be the main focus

📊 SLIDE TYPE REQUIREMENTS:
- 50% content slides (conceptual explanations)
- 25% diagram slides (visualizations)
- 15% code slides (illustrative examples)
- 5% code_demo slides (brief demonstrations)
- 5% conclusion slides

⚠️ IMPORTANT:
- Every code example MUST be preceded by conceptual explanation
- Include 'Why this works' sections BEFORE 'How to do it'
- Focus on mental models and understanding patterns
- Code should illustrate concepts, not be learned for its own sake
""",
    },
    "balanced": {
        "name": "Équilibré (50/50)",
        "aliases": ["équilibré", "balanced", "50/50", "équilibré (50/50)", "mixed"],
        "slide_ratio": {
            "content": 0.35,      # 35% explanation slides
            "diagram": 0.20,      # 20% diagrams
            "code": 0.25,         # 25% code examples
            "code_demo": 0.15,    # 15% live demos
            "conclusion": 0.05,   # 5% summary
        },
        "instructions": """
═══════════════════════════════════════════════════════════════════════════════
                    BALANCED FOCUS - THEORY + PRACTICE (50/50)
═══════════════════════════════════════════════════════════════════════════════

This course balances UNDERSTANDING with DOING. Follow these guidelines:

📚 CONTENT STRUCTURE:
- Equal emphasis on understanding concepts AND applying them
- Each concept: first explain (content slide), then show (code slide)
- Alternate between theory and practice throughout each section
- Diagrams should bridge theory and implementation

📊 SLIDE TYPE REQUIREMENTS:
- 35% content slides (explanations)
- 20% diagram slides (visualizations)
- 25% code slides (examples)
- 15% code_demo slides (demonstrations with output)
- 5% conclusion slides

⚠️ IMPORTANT:
- For every concept: explain WHY, then show HOW
- Include both 'why it works' and 'how to use it'
- Code examples should reinforce theoretical concepts
""",
    },
    "practical": {
        "name": "Très pratique (projets)",
        "aliases": ["pratique", "practical", "hands-on", "projets", "très pratique", "très pratique (projets)"],
        "slide_ratio": {
            "content": 0.20,      # 20% brief explanations
            "diagram": 0.10,      # 10% architecture diagrams
            "code": 0.35,         # 35% code examples
            "code_demo": 0.30,    # 30% live demos with output
            "conclusion": 0.05,   # 5% summary
        },
        "instructions": """
═══════════════════════════════════════════════════════════════════════════════
                    PRACTICAL FOCUS - HANDS-ON PROJECTS
═══════════════════════════════════════════════════════════════════════════════

This course emphasizes DOING over theoretical explanations. Follow these guidelines:

📚 CONTENT STRUCTURE:
- Prioritize learning by DOING over theoretical explanations
- Start with a BRIEF concept intro, then IMMEDIATELY show code
- Every lecture should include EXECUTABLE code examples
- Use code_demo slides to show REAL output and results
- Build towards a mini-project in each section

📊 SLIDE TYPE REQUIREMENTS:
- 20% content slides (brief context only)
- 10% diagram slides (architecture/flow only)
- 35% code slides (hands-on examples)
- 30% code_demo slides (with real output)
- 5% conclusion slides

⚠️ IMPORTANT:
- Minimum 65% of slides should be code or code_demo
- Theory should be MINIMAL - just enough context to understand the code
- Focus on 'how to build' rather than 'why it works'
- Include common errors and debugging tips
- Show REAL-WORLD use cases and practical applications
- Every section should end with a working mini-example
""",
    },
}


def parse_practical_focus(value: Optional[str]) -> str:
    """
    Parse practical focus value to normalized key.

    Args:
        value: Raw practical focus value (e.g., "théorique", "balanced", etc.)

    Returns:
        Normalized key: "theoretical", "balanced", or "practical"
    """
    if not value:
        return "balanced"

    value_lower = value.lower().strip()

    for level_key, level_config in PRACTICAL_FOCUS_CONFIG.items():
        if value_lower in [alias.lower() for alias in level_config["aliases"]]:
            return level_key

    return "balanced"


def get_practical_focus_instructions(practical_focus: Optional[str]) -> str:
    """
    Get instructions for the practical focus level.

    Args:
        practical_focus: The practical focus level

    Returns:
        Instructions string for the LLM prompt
    """
    level = parse_practical_focus(practical_focus)
    config = PRACTICAL_FOCUS_CONFIG.get(level, PRACTICAL_FOCUS_CONFIG["balanced"])
    return config["instructions"]


def get_practical_focus_slide_ratio(practical_focus: Optional[str]) -> Dict[str, float]:
    """
    Get slide type ratio for the practical focus level.

    Args:
        practical_focus: The practical focus level

    Returns:
        Dictionary with slide type ratios
    """
    level = parse_practical_focus(practical_focus)
    config = PRACTICAL_FOCUS_CONFIG.get(level, PRACTICAL_FOCUS_CONFIG["balanced"])
    return config["slide_ratio"]


def get_practical_focus_name(practical_focus: Optional[str]) -> str:
    """
    Get the display name for the practical focus level.

    Args:
        practical_focus: The practical focus level

    Returns:
        Display name (e.g., "Équilibré (50/50)")
    """
    level = parse_practical_focus(practical_focus)
    config = PRACTICAL_FOCUS_CONFIG.get(level, PRACTICAL_FOCUS_CONFIG["balanced"])
    return config["name"]
