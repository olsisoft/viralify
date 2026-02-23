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
            "content": 0.50,  # 50% explanation slides
            "diagram": 0.25,  # 25% diagrams
            "code": 0.15,  # 15% code examples
            "code_demo": 0.05,  # 5% live demos
            "conclusion": 0.05,  # 5% summary
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

💻 CODE STYLE FOR THEORETICAL FOCUS:
- Prefer PSEUDOCODE or SHORT illustrative snippets (5-15 lines max)
- For architecture/design courses: use pseudocode or simplified skeleton code
- Every line of code MUST have a comment explaining its PURPOSE
- Structure code as: concept illustration, NOT production implementation
- Show the PATTERN, not the full implementation details
- Use "..." or "# implementation details" to abbreviate non-essential parts
- Focus on the KEY lines that demonstrate the concept being taught
- Example structure:
  ```
  # PATTERN: Observer - notify all listeners when state changes
  class EventBus:
      def __init__(self):
          self.listeners = {}  # topic -> list of callbacks

      def subscribe(self, topic, callback):
          # Register a listener for a specific topic
          self.listeners.setdefault(topic, []).append(callback)

      def publish(self, topic, data):
          # Notify all registered listeners
          for callback in self.listeners.get(topic, []):
              callback(data)  # Each listener receives the event data
  ```

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
            "content": 0.35,  # 35% explanation slides
            "diagram": 0.20,  # 20% diagrams
            "code": 0.25,  # 25% code examples
            "code_demo": 0.15,  # 15% live demos
            "conclusion": 0.05,  # 5% summary
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

💻 CODE STYLE FOR BALANCED FOCUS:
- Write FUNCTIONAL, well-structured code (15-30 lines per slide)
- Each code block must be COMPLETE and self-contained (no missing pieces)
- Split code into logical segments: one concept per code slide
- Add comments for KEY lines (not every line, but important ones)
- Include brief docstrings for functions and classes
- Show both the implementation AND a usage example where relevant
- Code must be progressively built: slide N+1 extends what slide N introduced
- Example structure:
  ```
  from dataclasses import dataclass
  from typing import List

  @dataclass
  class Order:
      \"\"\"Represents a customer order.\"\"\"
      id: str
      items: List[str]
      total: float

      def apply_discount(self, percent: float) -> float:
          \"\"\"Apply a percentage discount and return new total.\"\"\"
          self.total *= (1 - percent / 100)
          return self.total

  # Usage
  order = Order(id="ORD-001", items=["Widget", "Gadget"], total=99.90)
  print(f"After 10% discount: {order.apply_discount(10):.2f}€")
  ```

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
            "content": 0.20,  # 20% brief explanations
            "diagram": 0.10,  # 10% architecture diagrams
            "code": 0.35,  # 35% code examples
            "code_demo": 0.30,  # 30% live demos with output
            "conclusion": 0.05,  # 5% summary
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

💻 CODE STYLE FOR PRACTICAL FOCUS:
- Write COMPLETE, PRODUCTION-READY, EXECUTABLE code (20-40 lines per slide)
- Code MUST run as-is without modifications — no placeholders, no "..."
- Include ALL imports, ALL setup code, ALL error handling
- Build a REAL mini-project across the lecture: each slide adds a piece
- Include expected OUTPUT as comments at the end of code_demo slides
- Show REAL-WORLD patterns: API calls, file I/O, database queries, etc.
- Include common MISTAKES and how to FIX them (before/after)
- Add debugging tips as comments where relevant
- Example structure for code_demo:
  ```
  import requests
  from dataclasses import dataclass
  from typing import Optional

  @dataclass
  class WeatherData:
      city: str
      temperature: float
      description: str

  def get_weather(city: str, api_key: str) -> Optional[WeatherData]:
      \"\"\"Fetch current weather for a city.\"\"\"
      url = f"https://api.weather.com/v1/current?city={city}"
      try:
          response = requests.get(url, headers={"X-API-Key": api_key}, timeout=10)
          response.raise_for_status()
          data = response.json()
          return WeatherData(
              city=city,
              temperature=data["main"]["temp"],
              description=data["weather"][0]["description"],
          )
      except requests.RequestException as e:
          print(f"Error fetching weather: {e}")
          return None

  # Usage
  weather = get_weather("Paris", api_key="your-api-key")
  if weather:
      print(f"{weather.city}: {weather.temperature}°C - {weather.description}")
  # Output: Paris: 18.5°C - partly cloudy
  ```

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


_CODE_STYLE_GUIDELINES: Dict[str, str] = {
    "theoretical": """CODE GENERATION RULES (THEORETICAL FOCUS):
- Write SHORT illustrative snippets: 5-15 lines max per code block
- For architecture/design topics: use pseudocode or skeleton code
- Comment EVERY significant line to explain its purpose
- Use "..." or "# (implementation details)" to skip non-essential code
- Focus on showing the PATTERN, not production-ready implementation
- Variable/function names should clearly convey the concept being taught
- For non-programming topics (architecture, design patterns): prefer diagrams over code
- NEVER generate empty function bodies or placeholder-only code
- Even short code MUST demonstrate a real concept with real logic""",
    "balanced": """CODE GENERATION RULES (BALANCED FOCUS):
- Write COMPLETE, well-structured code: 15-30 lines per code block
- Each code block must be self-contained and functional
- Add docstrings for functions/classes, comments for key logic
- Include a brief usage example after each function/class definition
- Split complex implementations across multiple slides progressively
- NEVER generate empty function bodies or placeholder-only code
- Every code block MUST contain real logic, real operations, real output""",
    "practical": """CODE GENERATION RULES (PRACTICAL FOCUS):
- Write COMPLETE, EXECUTABLE, production-quality code: 20-40 lines per block
- Code MUST run as-is: include ALL imports, setup, error handling
- Build a real mini-project across slides: each slide adds functionality
- Include expected output as comments for code_demo slides
- Show common mistakes with before/after corrections
- Add debugging tips as inline comments
- NEVER generate empty function bodies or placeholder-only code
- Every code block MUST be copy-pasteable and runnable""",
}


def get_code_style_for_focus(practical_focus: Optional[str]) -> str:
    """
    Get code generation style guidelines for the practical focus level.

    These guidelines control HOW code is written (length, comments, completeness)
    as opposed to the slide ratio which controls HOW MUCH code appears.

    Args:
        practical_focus: The practical focus level

    Returns:
        Code style guidelines string for the LLM prompt
    """
    level = parse_practical_focus(practical_focus)
    return _CODE_STYLE_GUIDELINES.get(level, _CODE_STYLE_GUIDELINES["balanced"])


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
