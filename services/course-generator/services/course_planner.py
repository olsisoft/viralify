"""
Course Planner Service

Uses LLM to generate structured course curricula/outlines.
Now integrates adaptive element suggestion based on profile category.

Supports multiple providers via shared.llm_provider:
- OpenAI, DeepSeek, Groq, Mistral, Together AI, xAI Grok
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import (
        get_llm_client,
        get_model_name,
        get_provider_config,
    )
    from shared.training_logger import (
        log_training_example,
        TaskType,
    )
    USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    USE_SHARED_LLM = False
    log_training_example = None  # Fallback: no logging
    print("[COURSE_PLANNER] Warning: shared.llm_provider not found, using direct OpenAI", flush=True)

from models.course_models import (
    PreviewOutlineRequest,
    CourseOutline,
    Section,
    Lecture,
    DifficultyLevel,
    ProfileCategory,
    CourseContext,
)
from services.element_suggester import ElementSuggester
from models.lesson_elements import LessonElementType
from agents.pedagogical_graph import get_pedagogical_agent

# Pre-LLM document structure extraction
from extractors.integration import (
    StructureAwareConstraints,
    get_adaptive_constraints,
    validate_output_against_constraints,
    validate_source_references,
    SOURCE_REFERENCE_PROMPT,
)

# Post-generation validation
from validators.post_generation_validator import (
    validate_curriculum,
    CurriculumCorrector,
    ValidationReport,
)


# Language code to full name mapping for content generation
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French (Français)",
    "es": "Spanish (Español)",
    "de": "German (Deutsch)",
    "pt": "Portuguese (Português)",
    "it": "Italian (Italiano)",
    "nl": "Dutch (Nederlands)",
    "pl": "Polish (Polski)",
    "ru": "Russian (Русский)",
    "zh": "Chinese (中文)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "ar": "Arabic (العربية)",
    "hi": "Hindi (हिन्दी)",
    "tr": "Turkish (Türkçe)",
}


class CoursePlanner:
    """Service for planning course curricula using GPT-4"""

    # Token limits to prevent API errors
    MAX_RAG_CONTEXT_TOKENS = 10000  # Max tokens for RAG context (increased for deeper integration)
    MAX_TOTAL_PROMPT_TOKENS = 100000  # Safety limit for total prompt

    # Profile-based element weights
    # These weights determine the emphasis on different content types
    PROFILE_ELEMENT_WEIGHTS = {
        ProfileCategory.TECH: {
            "code": 0.9,         # Developers need lots of code
            "diagram": 0.7,     # Architecture diagrams are useful
            "demo": 0.8,        # Live coding demos
            "theory": 0.5,      # Some theory but not too much
        },
        ProfileCategory.BUSINESS: {
            "code": 0.2,         # Minimal code
            "diagram": 0.8,     # Process and strategy diagrams
            "case_study": 0.9,  # Business case studies
            "theory": 0.7,      # Concepts and frameworks
        },
        ProfileCategory.CREATIVE: {
            "code": 0.3,         # Some code for digital tools
            "diagram": 0.5,     # Visual examples
            "demo": 0.9,        # Technique demonstrations
            "theory": 0.4,      # Less theory, more practice
        },
        ProfileCategory.HEALTH: {
            "code": 0.1,         # Almost no code
            "diagram": 0.8,     # Anatomy/process diagrams
            "demo": 0.7,        # Exercise demonstrations
            "theory": 0.6,      # Scientific background
        },
        ProfileCategory.EDUCATION: {
            "code": 0.4,         # Varies by subject
            "diagram": 0.7,     # Visual aids
            "demo": 0.6,        # Examples
            "theory": 0.8,      # Conceptual depth
        },
        ProfileCategory.LIFESTYLE: {
            "code": 0.1,         # Minimal code
            "diagram": 0.5,     # Visual guides
            "demo": 0.8,        # Practical demonstrations
            "theory": 0.4,      # Less theory
        },
    }

    # Practical focus levels and their modifiers
    # These modify the base PROFILE_ELEMENT_WEIGHTS based on user preference
    PRACTICAL_FOCUS_LEVELS = {
        "theoretical": {
            "name": "Théorique (concepts)",
            "aliases": ["théorique", "theoretical", "concepts", "théorique (concepts)"],
            "modifiers": {
                "code": 0.5,      # Reduce code by 50%
                "demo": 0.4,      # Reduce demos by 60%
                "diagram": 1.2,   # Increase diagrams by 20%
                "theory": 1.5,    # Increase theory by 50%
            },
            "slide_ratio": {
                "content": 0.50,      # 50% explanation slides
                "diagram": 0.25,      # 25% diagrams
                "code": 0.15,         # 15% code examples
                "code_demo": 0.05,    # 5% live demos
                "conclusion": 0.05,   # 5% summary
            },
            "instructions": """
THEORETICAL FOCUS - CONCEPTUAL LEARNING:
- Prioritize deep conceptual understanding over hands-on practice
- Each concept should be explained thoroughly with WHY and HOW it works
- Use diagrams to visualize abstract concepts
- Code examples should illustrate concepts, not be the main focus
- Include theoretical foundations, principles, and frameworks
- Minimum 60% of content should be conceptual explanations
- Code slides should explain the THEORY behind the code
- Focus on mental models and understanding patterns
""",
        },
        "balanced": {
            "name": "Équilibré (50/50)",
            "aliases": ["équilibré", "balanced", "50/50", "équilibré (50/50)", "mixed"],
            "modifiers": {
                "code": 1.0,      # Keep as-is
                "demo": 1.0,      # Keep as-is
                "diagram": 1.0,   # Keep as-is
                "theory": 1.0,    # Keep as-is
            },
            "slide_ratio": {
                "content": 0.35,      # 35% explanation slides
                "diagram": 0.20,      # 20% diagrams
                "code": 0.25,         # 25% code examples
                "code_demo": 0.15,    # 15% live demos
                "conclusion": 0.05,   # 5% summary
            },
            "instructions": """
BALANCED FOCUS - THEORY + PRACTICE:
- Equal emphasis on understanding concepts AND applying them
- Each concept: first explain (content slide), then show (code slide)
- Alternate between theory and practice throughout each section
- Diagrams should bridge theory and implementation
- Code examples should reinforce theoretical concepts
- 50% theory/explanation, 50% practical/code
- Include both "why it works" and "how to use it"
""",
        },
        "practical": {
            "name": "Très pratique (projets)",
            "aliases": ["pratique", "practical", "hands-on", "projets", "très pratique", "très pratique (projets)"],
            "modifiers": {
                "code": 1.8,      # Increase code by 80%
                "demo": 1.6,      # Increase demos by 60%
                "diagram": 0.7,   # Reduce diagrams by 30%
                "theory": 0.5,    # Reduce theory by 50%
            },
            "slide_ratio": {
                "content": 0.20,      # 20% brief explanations
                "diagram": 0.10,      # 10% architecture diagrams
                "code": 0.35,         # 35% code examples
                "code_demo": 0.30,    # 30% live demos with output
                "conclusion": 0.05,   # 5% summary
            },
            "instructions": """
PRACTICAL FOCUS - HANDS-ON PROJECTS:
- Prioritize learning by DOING over theoretical explanations
- Start with a brief concept intro, then immediately show code
- Every lecture should include executable code examples
- Use code_demo slides to show real output and results
- Build towards a mini-project in each section
- Minimum 65% of content should be code or code_demo slides
- Theory should be minimal - just enough context to understand the code
- Focus on "how to build" rather than "why it works"
- Include common errors and debugging tips
- Show real-world use cases and practical applications
""",
        },
    }

    @classmethod
    def parse_practical_focus(cls, value: Optional[str]) -> str:
        """
        Parse practical focus value from user input to normalized key.

        Returns: 'theoretical', 'balanced', or 'practical'
        """
        if not value:
            return "balanced"

        value_lower = value.lower().strip()

        for level_key, level_config in cls.PRACTICAL_FOCUS_LEVELS.items():
            if value_lower in [alias.lower() for alias in level_config["aliases"]]:
                return level_key

        # Default to balanced if not recognized
        return "balanced"

    @classmethod
    def get_practical_focus_config(cls, practical_focus: Optional[str]) -> dict:
        """Get the full configuration for a practical focus level."""
        level = cls.parse_practical_focus(practical_focus)
        return cls.PRACTICAL_FOCUS_LEVELS.get(level, cls.PRACTICAL_FOCUS_LEVELS["balanced"])

    @classmethod
    def get_adjusted_weights(
        cls,
        category: ProfileCategory,
        practical_focus: Optional[str]
    ) -> dict:
        """
        Get element weights adjusted for both category and practical focus.

        Combines base PROFILE_ELEMENT_WEIGHTS with PRACTICAL_FOCUS modifiers.
        """
        base_weights = cls.PROFILE_ELEMENT_WEIGHTS.get(
            category,
            cls.PROFILE_ELEMENT_WEIGHTS[ProfileCategory.EDUCATION]
        )

        focus_config = cls.get_practical_focus_config(practical_focus)
        modifiers = focus_config["modifiers"]

        adjusted = {}
        for key, base_value in base_weights.items():
            modifier = modifiers.get(key, 1.0)
            adjusted[key] = min(1.0, base_value * modifier)  # Cap at 1.0

        return adjusted

    def __init__(self, openai_api_key: Optional[str] = None):
        # Use shared LLM provider if available
        if USE_SHARED_LLM:
            self.client = get_llm_client()
            self.model = get_model_name("quality")
            config = get_provider_config()
            print(f"[COURSE_PLANNER] Using {config.name} provider with model {self.model}", flush=True)
        else:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=openai_api_key,
                timeout=120.0,
                max_retries=2
            ) if openai_api_key else AsyncOpenAI(timeout=120.0, max_retries=2)
            self.model = "gpt-4-turbo-preview"
            print(f"[COURSE_PLANNER] Using direct OpenAI with model {self.model}", flush=True)

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Initialize element suggester for adaptive content
        self.element_suggester = ElementSuggester(openai_api_key)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit"""
        if not text:
            return ""

        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens - 30]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        return truncated_text + "\n\n[... content truncated for length ...]"

    async def generate_outline(self, request: PreviewOutlineRequest) -> CourseOutline:
        """Generate a complete course outline from the request with adaptive elements"""
        print(f"[PLANNER] Generating outline for: {request.topic}", flush=True)

        # Check if RAG context is available (must have substantial content)
        has_rag_context = bool(request.rag_context and len(request.rag_context) > 100)

        if has_rag_context:
            print(f"[PLANNER] 🔒 RAG MODE: Using documents ({len(request.rag_context)} chars)", flush=True)
            temperature = 0.3  # Lower temperature for stricter document adherence
        else:
            print(f"[PLANNER] 📝 STANDARD MODE: Creating from topic", flush=True)
            temperature = 0.7  # Higher temperature for creative curriculum design

        # Build the prompt (structure differs based on RAG mode)
        prompt = self._build_curriculum_prompt(request)

        # Call GPT-4
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt(has_source_documents=has_rag_context)
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=4000
        )

        # Parse response
        content = response.choices[0].message.content
        try:
            outline_data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[PLANNER] ❌ JSON parsing failed: {e}", flush=True)
            print(f"[PLANNER] Raw response length: {len(content)} chars", flush=True)
            print(f"[PLANNER] Raw response (first 500 chars): {content[:500]}", flush=True)
            print(f"[PLANNER] Raw response (last 500 chars): {content[-500:]}", flush=True)
            raise

        # Validate source references in RAG mode
        if has_rag_context:
            ref_validation = validate_source_references(outline_data, request.rag_context)
            print(f"[PLANNER] 🔗 Source Reference Validation: {ref_validation.coverage:.0%} coverage ({ref_validation.valid_count}/{ref_validation.total})", flush=True)
            if not ref_validation.valid:
                print(f"[PLANNER] ⚠️ Invalid source references found:", flush=True)
                for invalid_ref in ref_validation.invalid_refs[:5]:  # Show max 5
                    print(f"[PLANNER]    - {invalid_ref}", flush=True)
                if len(ref_validation.invalid_refs) > 5:
                    print(f"[PLANNER]    ... and {len(ref_validation.invalid_refs) - 5} more", flush=True)
            else:
                print(f"[PLANNER] ✅ All source references validated", flush=True)

        # Log successful LLM response for training data collection
        system_prompt = self._get_system_prompt(has_source_documents=has_rag_context)
        if log_training_example:
            log_training_example(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response=content,
                task_type=TaskType.COURSE_OUTLINE,
                model=self.model,
                input_tokens=getattr(response.usage, 'prompt_tokens', None),
                output_tokens=getattr(response.usage, 'completion_tokens', None),
                metadata={
                    "topic": request.topic,
                    "language": getattr(request, 'language', 'en'),
                    "difficulty_start": request.difficulty_start.value,
                    "difficulty_end": request.difficulty_end.value,
                    "has_rag": has_rag_context,
                    "section_count": len(outline_data.get("sections", [])),
                }
            )

        # Convert to CourseOutline
        outline = self._parse_outline(outline_data, request)

        print(f"[PLANNER] Generated: {outline.section_count} sections, {outline.total_lectures} lectures", flush=True)

        # Post-generation validation
        if has_rag_context:
            # Get expected structure from constraints if available
            expected_sections = None
            expected_lectures = None
            if hasattr(request, '_structure_constraints') and request._structure_constraints:
                constraints = request._structure_constraints
                if constraints.is_from_documents:
                    expected_sections = constraints.section_count
                    expected_lectures = constraints.lectures_per_section[0] if constraints.lectures_per_section else None

            # Validate the generated curriculum
            validation_report = validate_curriculum(
                curriculum=outline_data,
                document_text=request.rag_context,
                expected_sections=expected_sections,
                expected_lectures=expected_lectures,
                strict=True
            )

            # Log validation results
            print(f"[PLANNER] ╔════════════════════════════════════════════════════════╗", flush=True)
            print(f"[PLANNER] ║         POST-GENERATION VALIDATION                      ║", flush=True)
            print(f"[PLANNER] ╚════════════════════════════════════════════════════════╝", flush=True)
            print(f"[PLANNER] Valid: {'✅ Yes' if validation_report.is_valid else '❌ No'}", flush=True)
            print(f"[PLANNER] Overall Score: {validation_report.overall_score:.0f}/100", flush=True)
            print(f"[PLANNER] Structure Score: {validation_report.structure_score:.0f}/100", flush=True)
            print(f"[PLANNER] Source Ref Score: {validation_report.source_reference_score:.0f}/100", flush=True)
            print(f"[PLANNER] Content Score: {validation_report.content_score:.0f}/100", flush=True)

            # Log issues by severity
            errors = [i for i in validation_report.issues if i.severity.value == "error"]
            warnings = [i for i in validation_report.issues if i.severity.value == "warning"]
            infos = [i for i in validation_report.issues if i.severity.value == "info"]

            if errors:
                print(f"[PLANNER] ❌ Errors ({len(errors)}):", flush=True)
                for issue in errors[:5]:  # Show max 5
                    print(f"[PLANNER]    - [{issue.category.value}] {issue.message}", flush=True)
                if len(errors) > 5:
                    print(f"[PLANNER]    ... and {len(errors) - 5} more errors", flush=True)

            if warnings:
                print(f"[PLANNER] ⚠️ Warnings ({len(warnings)}):", flush=True)
                for issue in warnings[:3]:  # Show max 3
                    print(f"[PLANNER]    - [{issue.category.value}] {issue.message}", flush=True)
                if len(warnings) > 3:
                    print(f"[PLANNER]    ... and {len(warnings) - 3} more warnings", flush=True)

            # Auto-correct if validation failed
            if not validation_report.is_valid:
                print(f"[PLANNER] 🔧 Attempting auto-correction...", flush=True)
                corrector = CurriculumCorrector(request.rag_context)
                corrected_data, corrections = corrector.correct(outline_data, validation_report)

                if corrections:
                    print(f"[PLANNER] ✅ Applied {len(corrections)} corrections:", flush=True)
                    for correction in corrections[:5]:
                        print(f"[PLANNER]    - {correction}", flush=True)
                    if len(corrections) > 5:
                        print(f"[PLANNER]    ... and {len(corrections) - 5} more corrections", flush=True)

                    # Re-parse the corrected outline
                    outline = self._parse_outline(corrected_data, request)
                    print(f"[PLANNER] 🔄 Re-parsed corrected outline: {outline.section_count} sections, {outline.total_lectures} lectures", flush=True)
                else:
                    print(f"[PLANNER] ⚠️ No automatic corrections could be applied", flush=True)
            else:
                print(f"[PLANNER] ✅ Validation passed - no corrections needed", flush=True)

        # Add adaptive elements to each lecture based on profile category
        if request.context and request.context.category:
            outline = await self._add_adaptive_elements(outline, request)
            print(f"[PLANNER] Added adaptive elements based on {request.context.category.value} profile", flush=True)

        return outline

    async def _add_adaptive_elements(
        self,
        outline: CourseOutline,
        request: PreviewOutlineRequest
    ) -> CourseOutline:
        """
        Add adaptive lesson elements to each lecture based on profile category.

        This uses the ElementSuggester to analyze each lecture topic and suggest
        the most relevant elements, then assigns profile-based weights.
        """
        category = request.context.category if request.context else ProfileCategory.EDUCATION
        element_weights = self.PROFILE_ELEMENT_WEIGHTS.get(category, self.PROFILE_ELEMENT_WEIGHTS[ProfileCategory.EDUCATION])

        print(f"[PLANNER] Suggesting elements for {category.value} profile", flush=True)

        # Get general topic suggestions once (cached)
        try:
            topic_suggestions = await self.element_suggester.suggest_elements(
                topic=request.topic,
                description=request.description,
                category=category,
                context=request.context
            )
            # Extract high-confidence elements (score > 0.5)
            suggested_elements = [
                s[0].value for s in topic_suggestions
                if s[1] > 0.5
            ]
        except Exception as e:
            print(f"[PLANNER] Element suggestion error: {e}, using defaults", flush=True)
            suggested_elements = self._get_default_elements_for_category(category)

        # Apply elements to each lecture
        for section in outline.sections:
            for lecture in section.lectures:
                # Assign suggested elements to lecture
                lecture.lesson_elements = suggested_elements[:6]  # Limit to 6 elements max
                lecture.element_weights = element_weights

        print(f"[PLANNER] Applied {len(suggested_elements)} elements to all lectures", flush=True)
        return outline

    def _get_default_elements_for_category(self, category: ProfileCategory) -> List[str]:
        """Get default elements for a category if AI suggestion fails"""
        defaults = {
            ProfileCategory.TECH: [
                "code_demo", "architecture_diagram", "debug_tips", "terminal_output", "code_typing"
            ],
            ProfileCategory.BUSINESS: [
                "case_study", "framework_template", "roi_metrics", "action_checklist", "market_analysis"
            ],
            ProfileCategory.CREATIVE: [
                "before_after", "technique_demo", "tool_tutorial", "creative_exercise"
            ],
            ProfileCategory.HEALTH: [
                "exercise_demo", "safety_warning", "body_diagram", "progression_plan"
            ],
            ProfileCategory.EDUCATION: [
                "memory_aid", "practice_problem", "multiple_explanations", "summary_card"
            ],
            ProfileCategory.LIFESTYLE: [
                "daily_routine", "reflection_exercise", "goal_setting", "habit_tracker"
            ],
        }
        return defaults.get(category, defaults[ProfileCategory.EDUCATION])

    async def generate_outline_with_agent(
        self,
        request: PreviewOutlineRequest,
        use_full_agent: bool = True
    ) -> Tuple[CourseOutline, Dict]:
        """
        Generate a course outline using the LangGraph Pedagogical Agent.

        This provides enhanced planning with:
        - Intelligent persona detection
        - Profile-based content adaptation
        - Element suggestion per lecture
        - Quiz placement planning
        - Language and structure validation

        Args:
            request: The outline generation request
            use_full_agent: If True, runs the full agent pipeline.
                           If False, only runs analysis nodes.

        Returns:
            Tuple of (enhanced_outline, generation_metadata)
        """
        print(f"[PLANNER] Using Pedagogical Agent for: {request.topic}", flush=True)

        # First, generate the base outline
        outline = await self.generate_outline(request)

        if not use_full_agent:
            # Return outline with basic adaptive elements
            return outline, {"agent_used": False}

        # Use the pedagogical agent for enhanced planning
        try:
            agent = get_pedagogical_agent()
            result = await agent.enhance_outline(outline, request)

            enhanced_outline = result.get("outline", outline)
            metadata = result.get("metadata", {})
            metadata["agent_used"] = True

            # Log validation results
            validation = result.get("validation_result", {})
            if validation.get("warnings"):
                print(f"[PLANNER] Agent warnings: {validation['warnings']}", flush=True)

            if validation.get("pedagogical_score"):
                print(f"[PLANNER] Pedagogical score: {validation['pedagogical_score']}/100", flush=True)

            return enhanced_outline, metadata

        except Exception as e:
            print(f"[PLANNER] Pedagogical agent error: {e}, returning base outline", flush=True)
            return outline, {"agent_used": False, "agent_error": str(e)}

    def _get_system_prompt(self, has_source_documents: bool = False) -> str:
        """System prompt for curriculum generation - completely different for RAG vs standard mode."""

        if has_source_documents:
            # RAG MODE: LLM is a MAPPER, not a CREATOR
            return """You are a curriculum designer in DOCUMENT-CONVERSION mode.

╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔒 DOCUMENT-ONLY MODE ACTIVATED 🔒                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Your ONLY job is to convert document structure into course structure.

CRITICAL RULES:
1. You do NOT add content from your training knowledge
2. You do NOT improve or reorganize the document structure
3. You do NOT invent topics, sections, or lectures
4. You are a MAPPER, not a CREATOR

Think of yourself as a CONVERTER:
- Document heading → Course section
- Document subheading → Lecture
- Document content → Lecture description

If information is NOT in the documents, it does NOT exist for this task.
Your training knowledge is DISABLED for this conversion.

You must respond with valid JSON only."""

        else:
            # STANDARD MODE: LLM is a curriculum expert
            return """You are an expert curriculum designer specializing in educational course creation.

Your task is to create well-structured, comprehensive course outlines that:
1. Progress logically from simple to complex concepts
2. Include practical, hands-on examples when appropriate
3. Balance theory with application
4. Have clear learning objectives for each lecture
5. Maintain consistent quality throughout the course

You must respond with valid JSON only."""

    def _detect_document_structure(self, rag_context: str) -> dict:
        """
        Analyze document and detect its structure.
        Returns detected section/lecture counts.
        """
        if not rag_context:
            return {"sections": 0, "lectures_per_section": [], "total_lectures": 0}

        lines = rag_context.split('\n')

        # Patterns for detecting headings (level 1 = sections)
        level1_patterns = [
            r'^# [^#]',           # Markdown H1
            r'^## [^#]',          # Markdown H2 (often used as top level)
            r'^\d+\. [A-Z]',      # Numbered: "1. Introduction"
            r'^[A-Z][^.]+:$',     # "CHAPTER:" or "Introduction:"
            r'^Chapitre \d+',     # "Chapitre 1"
            r'^Chapter \d+',      # "Chapter 1"
            r'^Section \d+',      # "Section 1"
            r'^Partie \d+',       # "Partie 1"
            r'^Part \d+',         # "Part 1"
            r'^Module \d+',       # "Module 1"
        ]

        # Patterns for detecting subheadings (level 2 = lectures)
        level2_patterns = [
            r'^### [^#]',          # Markdown H3
            r'^#### [^#]',         # Markdown H4
            r'^\d+\.\d+\.?\s+[A-Z]', # Numbered: "1.1 Concept" or "1.1. Concept"
            r'^[a-z]\)\s+',        # "a) Point"
            r'^\s*[-•]\s+[A-Z]',   # Bullet with capital letter
        ]

        sections = []
        current_section_lectures = 0
        in_section = False

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check level 1 (new section)
            is_level1 = False
            for pattern in level1_patterns:
                if re.match(pattern, line_stripped):
                    is_level1 = True
                    break

            if is_level1:
                # Save previous section's lecture count
                if in_section and current_section_lectures > 0:
                    sections.append(current_section_lectures)
                elif in_section:
                    sections.append(1)  # Section with no subheadings = 1 lecture
                current_section_lectures = 0
                in_section = True
                continue

            # Check level 2 (lecture within section)
            if in_section:
                for pattern in level2_patterns:
                    if re.match(pattern, line_stripped):
                        current_section_lectures += 1
                        break

        # Don't forget last section
        if in_section:
            if current_section_lectures > 0:
                sections.append(current_section_lectures)
            else:
                sections.append(1)

        # If no structure detected, return zeros
        if not sections:
            return {"sections": 0, "lectures_per_section": [], "total_lectures": 0}

        return {
            "sections": len(sections),
            "lectures_per_section": sections,
            "total_lectures": sum(sections),
            "avg_lectures": sum(sections) / len(sections) if sections else 0
        }

    def _build_curriculum_prompt(self, request: PreviewOutlineRequest) -> str:
        """Build the prompt for curriculum generation.

        CRITICAL: When RAG context is present, documents and rules come FIRST
        to prevent the LLM from activating its training knowledge on the topic.
        """
        # Get language name for prompt
        content_language = getattr(request, 'language', 'en')
        language_name = LANGUAGE_NAMES.get(content_language, content_language)

        # Check if RAG mode
        has_rag = bool(request.rag_context and len(request.rag_context) > 100)

        if has_rag:
            # RAG MODE: Documents and rules FIRST, then topic
            return self._build_rag_curriculum_prompt(request, language_name, content_language)
        else:
            # STANDARD MODE: Normal prompt structure
            return self._build_standard_curriculum_prompt(request, language_name, content_language)

    def _build_rag_curriculum_prompt(
        self,
        request: PreviewOutlineRequest,
        language_name: str,
        content_language: str
    ) -> str:
        """Build curriculum prompt for RAG mode - documents FIRST, topic LAST."""

        # Truncate RAG context if needed
        rag_context = request.rag_context
        context_tokens = self.count_tokens(rag_context)
        if context_tokens > self.MAX_RAG_CONTEXT_TOKENS:
            print(f"[PLANNER] RAG context too large ({context_tokens} tokens), truncating to {self.MAX_RAG_CONTEXT_TOKENS}", flush=True)
            rag_context = self.truncate_to_tokens(rag_context, self.MAX_RAG_CONTEXT_TOKENS)

        # Get adaptive constraints using pre-LLM structure extraction
        user_sections = request.structure.number_of_sections
        user_lectures = request.structure.lectures_per_section

        constraints = get_adaptive_constraints(
            rag_context=rag_context,
            target_sections=user_sections,
            target_lectures=user_lectures
        )

        # Store constraints on request for later validation
        request._structure_constraints = constraints

        # Log structure analysis results
        print(f"[PLANNER] ╔════════════════════════════════════════════════════════╗", flush=True)
        print(f"[PLANNER] ║         STRUCTURE ANALYSIS COMPLETE                     ║", flush=True)
        print(f"[PLANNER] ╚════════════════════════════════════════════════════════╝", flush=True)
        print(f"[PLANNER] Source: {'Documents' if constraints.is_from_documents else 'User targets'}", flush=True)
        print(f"[PLANNER] Sections: {constraints.section_count}", flush=True)
        print(f"[PLANNER] Lectures: {constraints.lectures_per_section}", flush=True)
        print(f"[PLANNER] Confidence: {constraints.confidence:.0%}", flush=True)

        if constraints.warnings:
            for warning in constraints.warnings:
                print(f"[PLANNER] ⚠️ {warning}", flush=True)

        # Build structure guidance based on constraints
        if constraints.is_from_documents:
            # Structure detected from documents - use it
            structure_guidance = self._build_structure_guidance_from_constraints(constraints, user_sections, user_lectures)
        else:
            # No clear structure detected - use flexible guidance
            structure_guidance = f"""
STRUCTURE GUIDANCE:
• No clear document structure detected
• Analyze the document content and create logical sections
• Target approximately {user_sections} sections with ~{user_lectures} lectures each
• Group related content together
• Do NOT invent topics - only organize what's in the documents"""

        # Build context section if available
        context_section = self._build_context_section(request.context)

        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     🔒 STEP 1: READ THESE RULES BEFORE ANYTHING ELSE 🔒                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

You are in DOCUMENT-ONLY mode. Your training knowledge is DISABLED.

ABSOLUTE RULES:
┌─────────────────────────────────────────────────────────────────────────────┐
│  ✅ Information IN the documents below    → You MAY use it                  │
│  ❌ Information NOT IN the documents      → You MUST NOT use it             │
│  ❌ Your training knowledge               → FORBIDDEN for this task         │
└─────────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║     📄 STEP 2: READ THE SOURCE DOCUMENTS                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

<source_documents>
{rag_context}
</source_documents>

{SOURCE_REFERENCE_PROMPT}

╔══════════════════════════════════════════════════════════════════════════════╗
║     🔄 STEP 3: MAPPING ALGORITHM (FOLLOW EXACTLY)                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Your job is to CONVERT document structure → course structure:

MAPPING RULES:
• Document heading/chapter → Course SECTION
• Document subheading → LECTURE in that section
• Document paragraph content → Lecture description
• Keep the SAME ORDER as the documents
• Keep titles SIMILAR to document headings (translate if needed)

WHAT YOU MUST NOT DO:
❌ Add "Introduction" section if not in documents
❌ Add "Conclusion" section if not in documents
❌ Add "Best Practices" from your knowledge
❌ Reorganize or merge document sections
❌ Invent topics not mentioned in documents
❌ Add examples not present in documents

╔══════════════════════════════════════════════════════════════════════════════╗
║     📊 STEP 4: STRUCTURE REQUIREMENTS (PRE-ANALYZED FROM DOCUMENTS)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

{constraints.structure_prompt if constraints.is_from_documents else structure_guidance}

╔══════════════════════════════════════════════════════════════════════════════╗
║     📋 STEP 5: CREATE THE COURSE (from documents only)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

TASK DETAILS:
• Topic: {request.topic}
• Description: {request.description or 'See documents above'}
• Language: {language_name} (ALL content must be in this language)

{context_section}

DIFFICULTY PROGRESSION:
• Start: {request.difficulty_start.value}
• End: {request.difficulty_end.value}

╔══════════════════════════════════════════════════════════════════════════════╗
║     ✅ STEP 6: OUTPUT JSON                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generate JSON with this structure:
{{
    "title": "Course title (use document title if available)",
    "description": "Description based on document content",
    "target_audience": "Based on document content or general audience",
    "context_summary": "Summary of what documents cover",
    "sections": [
        {{
            "title": "Section title (from document heading)",
            "source_reference": "REQUIRED: ## Exact Document Heading (line N)",
            "description": "From document content",
            "lectures": [
                {{
                    "title": "Lecture title (from document subheading)",
                    "source_reference": "REQUIRED: ### Exact Document Subheading (line N)",
                    "description": "From document content for this section",
                    "objectives": ["Based on document content", "..."],
                    "difficulty": "beginner|intermediate|advanced",
                    "has_practical_content": true/false,
                    "key_concepts": ["From documents"]
                }}
            ]
        }}
    ]
}}

FINAL CHECK before outputting:
□ Section count = {constraints.section_count} (from documents, NOT user's {user_sections})
□ Lecture counts = {constraints.lectures_per_section} (from documents)
□ Every section title is traceable to a document heading
□ Every lecture covers content that EXISTS in documents
□ I did NOT add any topic from my training knowledge
□ The order matches the document order"""

    def _build_structure_guidance_from_constraints(
        self,
        constraints: StructureAwareConstraints,
        user_sections: int,
        user_lectures: int
    ) -> str:
        """Build structure guidance based on pre-LLM extracted constraints."""

        doc_sections = constraints.section_count
        lectures_per_section = constraints.lectures_per_section
        total_lectures = sum(lectures_per_section)

        # Build lecture counts display
        lecture_counts_display = ""
        for i, count in enumerate(lectures_per_section, 1):
            lecture_counts_display += f"   Section {i}: {count} lectures\n"

        # Determine if there's a mismatch
        mismatch_warning = ""
        if doc_sections != user_sections:
            mismatch_warning = f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚠️  MISMATCH DETECTED                                                       │
│  User requested: {user_sections} sections, {user_lectures} lectures each                       │
│  Documents have: {doc_sections} sections, varying lectures                              │
│                                                                             │
│  → FOLLOW DOCUMENTS, NOT USER REQUEST                                       │
│  → Create {doc_sections} sections (NOT {user_sections})                                          │
│  → Lecture counts vary per section (NOT {user_lectures} each)                            │
└─────────────────────────────────────────────────────────────────────────────┘
"""

        return f"""
DETECTED DOCUMENT STRUCTURE (PRE-LLM ANALYSIS):
══════════════════════════════════════════════════════════════════════════════
  Documents contain: {doc_sections} main sections/chapters
  Total lectures: {total_lectures}
  Detection confidence: {constraints.confidence:.0%}

  Lecture distribution:
{lecture_counts_display}
══════════════════════════════════════════════════════════════════════════════
{mismatch_warning}
INSTRUCTION: Create EXACTLY {doc_sections} sections following the document structure.
             Do NOT create {user_sections} sections.
             Do NOT add sections to reach {user_sections}.
             Do NOT merge sections to reduce count.

RULE: If you find yourself wanting to add a section to reach {user_sections},
      STOP. That would be invention. Stick to {doc_sections} sections."""

    def _build_standard_curriculum_prompt(
        self,
        request: PreviewOutlineRequest,
        language_name: str,
        content_language: str
    ) -> str:
        """Build curriculum prompt for standard mode (no RAG) - normal structure."""

        difficulty_progression = self._get_difficulty_progression(
            request.difficulty_start,
            request.difficulty_end
        )

        structure_info = ""
        if request.structure.random_structure:
            structure_info = f"""
Let the course structure emerge naturally from the topic.
Target duration: {request.structure.total_duration_minutes} minutes.
Choose an appropriate number of sections and lectures based on what makes pedagogical sense for this duration."""
        else:
            total_lectures = request.structure.number_of_sections * request.structure.lectures_per_section
            lecture_duration = request.structure.total_duration_minutes // max(total_lectures, 1)
            structure_info = f"""
STRICT COURSE STRUCTURE REQUIREMENTS:
- Total Duration: {request.structure.total_duration_minutes} minutes
- EXACTLY {request.structure.number_of_sections} sections
- EXACTLY {request.structure.lectures_per_section} lectures per section
- Total lectures: {total_lectures}
- Target duration per lecture: ~{lecture_duration} minutes"""

        # Build context section if available
        context_section = self._build_context_section(request.context)

        # Get category-specific instructions
        category_instructions = ""
        if request.context:
            category_instructions = self._get_category_specific_instructions(request.context.category)

        # Build keywords section if available
        keywords_section = self._build_keywords_section(getattr(request, 'keywords', None))

        return f"""Create a comprehensive course outline for the following:

**CRITICAL: ALL CONTENT MUST BE IN {language_name.upper()}**

TOPIC: {request.topic}
{f'DESCRIPTION: {request.description}' if request.description else ''}
CONTENT LANGUAGE: {language_name} (code: {content_language})

{context_section}
{keywords_section}

DIFFICULTY PROGRESSION:
- Starting Level: {request.difficulty_start.value}
- Ending Level: {request.difficulty_end.value}
{difficulty_progression}

{structure_info}

Generate a JSON response with this structure:
{{
    "title": "Course Title",
    "description": "A compelling course description (2-3 sentences)",
    "target_audience": "Description of ideal learner",
    "context_summary": "Brief summary of the course context and focus",
    "sections": [
        {{
            "title": "Section Title",
            "description": "Section overview",
            "lectures": [
                {{
                    "title": "Lecture Title",
                    "description": "What will be covered",
                    "objectives": ["Objective 1", "Objective 2", "Objective 3"],
                    "difficulty": "beginner|intermediate|advanced|very_advanced|expert",
                    "has_practical_content": true/false,
                    "key_concepts": ["concept1", "concept2"]
                }}
            ]
        }}
    ]
}}

Requirements:
1. Follow the structure requirements above
2. **LANGUAGE: Write ALL content in {language_name}**
3. Each section should have a clear theme
4. Lectures should build upon each other
5. Include 3-5 specific learning objectives per lecture
6. Ensure smooth difficulty progression
7. Make titles engaging and specific
{category_instructions}"""

    def _build_rag_section(self, rag_context: Optional[str]) -> str:
        """Build the RAG source documents section of the prompt with STRICT structure adherence"""
        if not rag_context:
            return ""

        # Check token count and truncate if necessary
        context_tokens = self.count_tokens(rag_context)
        if context_tokens > self.MAX_RAG_CONTEXT_TOKENS:
            print(f"[PLANNER] RAG context too large ({context_tokens} tokens), truncating to {self.MAX_RAG_CONTEXT_TOKENS}", flush=True)
            rag_context = self.truncate_to_tokens(rag_context, self.MAX_RAG_CONTEXT_TOKENS)

        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     ⚠️  CRITICAL: 100% RAG-BASED COURSE STRUCTURE (NO EXCEPTIONS)  ⚠️         ║
╚══════════════════════════════════════════════════════════════════════════════╝

The user has uploaded source documents. Your ONLY job is to transform their content
into a course structure. You are NOT allowed to add your own knowledge or topics.

---BEGIN SOURCE DOCUMENTS---
{rag_context}
---END SOURCE DOCUMENTS---

╔══════════════════════════════════════════════════════════════════════════════╗
║              STRICT RULES - FAILURE TO FOLLOW = UNUSABLE COURSE              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🚨 RULE 1: STRUCTURE MUST MATCH DOCUMENTS EXACTLY
   - Look at the "DOCUMENT STRUCTURE" section above
   - Each heading/chapter in documents → becomes a SECTION in the course
   - Each subheading/subtopic in documents → becomes a LECTURE in the course
   - The ORDER of sections/lectures must match the document order
   - DO NOT reorder, merge, split, or rename topics arbitrarily

🚨 RULE 2: ZERO INVENTION POLICY
   ❌ FORBIDDEN: Adding topics NOT in the source documents
   ❌ FORBIDDEN: Creating "Introduction" or "Conclusion" sections not in documents
   ❌ FORBIDDEN: Adding "best practices" or "tips" not mentioned in documents
   ❌ FORBIDDEN: Expanding with your own knowledge
   ✅ REQUIRED: Every section title must be traceable to a document heading
   ✅ REQUIRED: Every lecture must cover content that EXISTS in documents

🚨 RULE 3: MAPPING ALGORITHM (FOLLOW EXACTLY)
   Step 1: Read the DOCUMENT STRUCTURE at the top of the source
   Step 2: Each Level-1 heading → Course Section
   Step 3: Each Level-2 heading under it → Lectures in that Section
   Step 4: If no Level-2, split Level-1 content into logical lectures
   Step 5: Lecture descriptions = summarize ONLY what's in that document section

🚨 RULE 4: VERIFICATION CHECKLIST (MENTAL CHECK BEFORE OUTPUT)
   □ Does every section title appear in the document structure?
   □ Does every lecture cover content that exists in the documents?
   □ Did I add any topic that is NOT in the source documents?
   □ Is the order consistent with the document order?
   If any answer is NO → REVISE before outputting.

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE CORRECT MAPPING:

Document Structure:
┌── Chapter 1: Introduction to Python
    ├── 1.1 What is Python?
    ├── 1.2 Installing Python
┌── Chapter 2: Variables and Data Types
    ├── 2.1 Variables
    ├── 2.2 Numbers
    ├── 2.3 Strings

Correct Course Structure:
Section 1: "Introduction to Python" (from Chapter 1)
  - Lecture 1: "What is Python?" (from 1.1)
  - Lecture 2: "Installing Python" (from 1.2)
Section 2: "Variables and Data Types" (from Chapter 2)
  - Lecture 3: "Variables" (from 2.1)
  - Lecture 4: "Numbers" (from 2.2)
  - Lecture 5: "Strings" (from 2.3)

❌ WRONG: Adding "Section 3: Best Practices" (not in documents!)
❌ WRONG: Adding "Lecture: Python History" (not in documents!)
❌ WRONG: Renaming "Variables" to "Understanding Variables in Depth"
═══════════════════════════════════════════════════════════════════════════════

The user chose THESE specific documents because they contain the EXACT content
they want to teach. A course that ignores document structure is USELESS.
"""

    def _build_context_section(self, context: Optional[CourseContext]) -> str:
        """Build the context section of the prompt"""
        if not context:
            return ""

        lines = ["CREATOR & AUDIENCE CONTEXT:"]
        lines.append(f"- Category: {context.category.value}")
        lines.append(f"- Niche: {context.profile_niche}")
        lines.append(f"- Communication Tone: {context.profile_tone}")
        lines.append(f"- Audience Level: {context.profile_audience_level}")
        lines.append(f"- Language Complexity: {context.profile_language_level}")
        lines.append(f"- Primary Goal: {context.profile_primary_goal}")

        if context.profile_audience_description:
            lines.append(f"- Target Audience: {context.profile_audience_description}")

        if context.context_answers:
            lines.append("\nSPECIFIC CONTEXT:")
            for key, value in context.context_answers.items():
                # Skip practical_focus here - we'll handle it specially below
                if key == "practical_focus":
                    continue
                # Convert snake_case to human-readable
                readable_key = key.replace("_", " ").title()
                lines.append(f"- {readable_key}: {value}")

        if context.specific_tools:
            lines.append(f"\nTools/Technologies: {context.specific_tools}")

        # Get practical focus from context.practical_focus OR context_answers
        practical_focus_value = context.practical_focus
        if not practical_focus_value and context.context_answers:
            practical_focus_value = context.context_answers.get("practical_focus")

        if practical_focus_value:
            # Get the full configuration for this practical focus level
            focus_config = self.get_practical_focus_config(practical_focus_value)
            focus_level = self.parse_practical_focus(practical_focus_value)
            slide_ratio = focus_config["slide_ratio"]

            lines.append(f"\n{'='*70}")
            lines.append(f"PRACTICAL FOCUS LEVEL: {focus_config['name'].upper()}")
            lines.append(f"{'='*70}")
            lines.append(focus_config["instructions"])

            # Add slide type distribution requirements
            lines.append("\nREQUIRED SLIDE TYPE DISTRIBUTION:")
            lines.append(f"- Content/explanation slides: {int(slide_ratio['content']*100)}%")
            lines.append(f"- Diagram slides: {int(slide_ratio['diagram']*100)}%")
            lines.append(f"- Code slides: {int(slide_ratio['code']*100)}%")
            lines.append(f"- Code demo slides (with output): {int(slide_ratio['code_demo']*100)}%")
            lines.append(f"- Conclusion slides: {int(slide_ratio['conclusion']*100)}%")

            # Add specific guidance based on level
            if focus_level == "theoretical":
                lines.append("\n⚠️ IMPORTANT: This is a THEORY-FOCUSED course.")
                lines.append("- Every code example must be preceded by conceptual explanation")
                lines.append("- Include 'Why this works' sections before 'How to do it'")
                lines.append("- Use diagrams to explain abstract concepts")
                lines.append("- Code should illustrate concepts, not be learned for its own sake")
            elif focus_level == "practical":
                lines.append("\n⚠️ IMPORTANT: This is a HANDS-ON PRACTICAL course.")
                lines.append("- Minimize theory - get to code quickly")
                lines.append("- Every section should have a mini-project or exercise")
                lines.append("- Show real output using code_demo slides")
                lines.append("- Include debugging tips and common errors")
                lines.append("- Focus on 'how to build' over 'how it works'")

            print(f"[PLANNER] 🎯 Practical focus: {focus_config['name']} (level: {focus_level})", flush=True)

        if context.expected_outcome:
            lines.append(f"\nExpected Outcome: {context.expected_outcome}")

        return "\n".join(lines)

    def _build_keywords_section(self, keywords: Optional[list]) -> str:
        """Build the keywords section of the prompt"""
        if not keywords or len(keywords) == 0:
            return ""

        keywords_str = ", ".join(keywords[:5])  # Limit to 5 keywords
        return f"""
FOCUS KEYWORDS:
The user has specified the following keywords to focus on in this course: {keywords_str}

IMPORTANT: These keywords represent key technologies, concepts, or tools that MUST be prominently featured in the course content. Ensure that:
1. Each keyword is covered in at least one dedicated lecture or section
2. The keywords are mentioned in the course description and learning objectives
3. Practical examples specifically use or reference these keywords
4. The course structure reflects the importance of these focus areas
"""

    def _get_category_specific_instructions(self, category: ProfileCategory) -> str:
        """Get category-specific curriculum instructions"""
        instructions = {
            ProfileCategory.TECH: """
8. Include practical code exercises and projects
9. Follow industry best practices and patterns
10. Build progressively from basics to advanced concepts
11. Include debugging tips and common pitfalls""",

            ProfileCategory.BUSINESS: """
8. Include real-world case studies and examples
9. Provide actionable frameworks and templates
10. Add measurable success metrics where applicable
11. Focus on practical application over theory""",

            ProfileCategory.HEALTH: """
8. Include step-by-step demonstrations
9. Add safety warnings where appropriate
10. Provide progressive difficulty in exercises
11. Include rest and recovery guidance""",

            ProfileCategory.CREATIVE: """
8. Include hands-on creative projects
9. Provide technique demonstrations
10. Allow for creative exploration
11. Include critique and improvement sections""",

            ProfileCategory.EDUCATION: """
8. Include assessment checkpoints
9. Provide multiple explanation approaches
10. Add practice exercises and examples
11. Include memory aids and summaries""",

            ProfileCategory.LIFESTYLE: """
8. Include actionable daily habits
9. Provide reflection exercises
10. Add milestone celebrations
11. Include accountability check-ins""",
        }
        return instructions.get(category, "")

    def _get_difficulty_criteria(self) -> dict:
        """Define precise criteria for each difficulty level"""
        return {
            DifficultyLevel.BEGINNER: {
                "name": "Beginner (Débutant)",
                "prerequisites": "No prior knowledge required",
                "vocabulary": "Simple, everyday language. Define ALL technical terms when first introduced.",
                "concepts": "One concept at a time. Maximum 2-3 new concepts per lecture.",
                "examples": "Real-world analogies and relatable examples. Step-by-step explanations.",
                "code_complexity": "Simple, short code snippets (5-15 lines). No advanced patterns.",
                "pace": "Slow pace with frequent recaps and summaries.",
                "assumptions": "Assume learner has NEVER seen this topic before.",
                "indicators": [
                    "Uses 'What is X?' type explanations",
                    "Provides definitions for basic terms",
                    "Uses analogies to familiar concepts",
                    "Breaks down every step explicitly",
                    "No assumed background knowledge"
                ]
            },
            DifficultyLevel.INTERMEDIATE: {
                "name": "Intermediate (Intermédiaire)",
                "prerequisites": "Basic understanding of fundamentals",
                "vocabulary": "Technical terms used but still explained. Industry jargon introduced gradually.",
                "concepts": "Multiple related concepts. Build on prior knowledge.",
                "examples": "Practical, real-world scenarios. Some complexity in examples.",
                "code_complexity": "Moderate code (15-40 lines). Common patterns and best practices.",
                "pace": "Moderate pace. Less hand-holding, more practice.",
                "assumptions": "Learner knows basics but needs guidance on application.",
                "indicators": [
                    "Uses 'How to apply X' type content",
                    "References foundational concepts without re-explaining",
                    "Introduces common patterns and conventions",
                    "Expects learner to follow multi-step processes",
                    "Some problem-solving required"
                ]
            },
            DifficultyLevel.ADVANCED: {
                "name": "Advanced (Avancé)",
                "prerequisites": "Solid understanding of core concepts and practical experience",
                "vocabulary": "Technical language assumed. Industry terminology without explanation.",
                "concepts": "Complex, interconnected concepts. Edge cases and nuances.",
                "examples": "Production-level scenarios. Performance and scalability considerations.",
                "code_complexity": "Complex code (40-100+ lines). Design patterns, optimization.",
                "pace": "Fast pace. Focus on depth, not basics.",
                "assumptions": "Learner has hands-on experience and seeks mastery.",
                "indicators": [
                    "Uses 'Why X works this way' explanations",
                    "Discusses trade-offs and alternatives",
                    "Covers edge cases and error handling",
                    "Performance optimization techniques",
                    "Architecture and design decisions"
                ]
            },
            DifficultyLevel.VERY_ADVANCED: {
                "name": "Very Advanced (Très Avancé)",
                "prerequisites": "Deep expertise in the field",
                "vocabulary": "Expert-level terminology. Assumes familiarity with advanced concepts.",
                "concepts": "Cutting-edge topics. Research-level content. System design.",
                "examples": "Enterprise-scale problems. Complex system interactions.",
                "code_complexity": "Complex systems (100+ lines). Low-level optimizations.",
                "pace": "Expert pace. Deep technical dives.",
                "assumptions": "Learner is a practitioner seeking specialized knowledge.",
                "indicators": [
                    "Explores internal workings and implementation details",
                    "Discusses limitations and workarounds",
                    "System-level thinking and architecture",
                    "Benchmarking and profiling",
                    "Integration with other advanced systems"
                ]
            },
            DifficultyLevel.EXPERT: {
                "name": "Expert",
                "prerequisites": "Years of professional experience, deep domain expertise",
                "vocabulary": "Highly specialized terminology. Academic/research language.",
                "concepts": "Frontier knowledge. Original research. Novel approaches.",
                "examples": "Unique, complex scenarios. State-of-the-art solutions.",
                "code_complexity": "Research-grade code. Novel algorithms. Theoretical foundations.",
                "pace": "Expert-only pace. No basics covered.",
                "assumptions": "Learner is pushing the boundaries of the field.",
                "indicators": [
                    "Discusses open research problems",
                    "Compares multiple advanced approaches",
                    "Mathematical or theoretical foundations",
                    "Contributes to or critiques existing solutions",
                    "Novel techniques and methodologies"
                ]
            }
        }

    def _get_difficulty_progression(
        self,
        start: DifficultyLevel,
        end: DifficultyLevel
    ) -> str:
        """Generate detailed guidance for difficulty progression with specific criteria"""
        criteria = self._get_difficulty_criteria()
        levels = list(DifficultyLevel)
        start_idx = levels.index(start)
        end_idx = levels.index(end)

        # Build detailed criteria section
        relevant_levels = levels[start_idx:end_idx + 1]

        criteria_text = "\n\n=== DIFFICULTY LEVEL CRITERIA ===\n"
        for level in relevant_levels:
            c = criteria[level]
            criteria_text += f"""
**{c['name']}**
- Prerequisites: {c['prerequisites']}
- Vocabulary: {c['vocabulary']}
- Concepts: {c['concepts']}
- Examples: {c['examples']}
- Code complexity: {c['code_complexity']}
- Pace: {c['pace']}
- Key indicators: {', '.join(c['indicators'][:3])}
"""

        if start_idx == end_idx:
            # Single level - strict adherence
            c = criteria[start]
            return f"""{criteria_text}
=== DIFFICULTY REQUIREMENT ===
ALL content MUST strictly match the **{c['name']}** level:
- {c['vocabulary']}
- {c['concepts']}
- {c['code_complexity']}
- {c['pace']}

DO NOT include content from higher difficulty levels.
VERIFY each lecture matches these criteria before including it."""
        else:
            # Progression
            level_names = [criteria[l]['name'] for l in relevant_levels]
            return f"""{criteria_text}
=== DIFFICULTY PROGRESSION ===
The course MUST progress through: {' → '.join(level_names)}

- First {100 // len(relevant_levels)}% of lectures: {criteria[relevant_levels[0]]['name']} level
- Last {100 // len(relevant_levels)}% of lectures: {criteria[relevant_levels[-1]]['name']} level
- Each lecture MUST specify its difficulty level
- Gradual transition between levels - no sudden jumps"""

    def _parse_outline(
        self,
        data: dict,
        request: PreviewOutlineRequest
    ) -> CourseOutline:
        """Parse GPT-4 response into CourseOutline model.

        In RAG mode: Accept whatever structure the LLM created from documents
        In standard mode: Enforce exact structure requirements
        """
        sections = []

        # Get expected structure
        expected_sections = request.structure.number_of_sections
        expected_lectures_per_section = request.structure.lectures_per_section
        is_random = request.structure.random_structure

        # Check if RAG mode - documents define structure, not user requirements
        is_rag_mode = bool(request.rag_context and len(request.rag_context) > 100)

        # Calculate duration per lecture based on ACTUAL structure (not expected)
        total_lectures_generated = sum(
            len(s.get("lectures", []))
            for s in data.get("sections", [])
        )
        if total_lectures_generated > 0:
            duration_per_lecture = (
                request.structure.total_duration_minutes * 60 // total_lectures_generated
            )
        else:
            # Fallback if no lectures
            total_lectures = expected_sections * expected_lectures_per_section
            duration_per_lecture = (
                request.structure.total_duration_minutes * 60 // max(total_lectures, 1)
            )

        raw_sections = data.get("sections", [])

        for sec_idx, sec_data in enumerate(raw_sections):
            lectures = []
            raw_lectures = sec_data.get("lectures", [])

            for lec_idx, lec_data in enumerate(raw_lectures):
                # Parse difficulty
                difficulty_str = lec_data.get("difficulty", "intermediate")
                try:
                    difficulty = DifficultyLevel(difficulty_str)
                except ValueError:
                    difficulty = DifficultyLevel.INTERMEDIATE

                lecture = Lecture(
                    title=lec_data.get("title", f"Lecture {lec_idx + 1}"),
                    description=lec_data.get("description", ""),
                    objectives=lec_data.get("objectives", []),
                    difficulty=difficulty,
                    duration_seconds=duration_per_lecture,
                    order=lec_idx
                )
                lectures.append(lecture)

            section = Section(
                title=sec_data.get("title", f"Section {sec_idx + 1}"),
                description=sec_data.get("description", ""),
                order=sec_idx,
                lectures=lectures
            )
            sections.append(section)

        # Enforce exact structure ONLY in standard mode (not RAG, not random)
        # In RAG mode, documents define the structure
        if not is_random and not is_rag_mode:
            sections = self._enforce_structure(
                sections,
                expected_sections,
                expected_lectures_per_section,
                duration_per_lecture,
                request
            )
        elif is_rag_mode:
            actual_sections = len(sections)
            actual_lectures = sum(len(s.lectures) for s in sections)
            print(f"[PLANNER] RAG mode: Accepting document-defined structure ({actual_sections} sections, {actual_lectures} lectures)", flush=True)

            # Post-LLM validation against detected structure
            if hasattr(request, '_structure_constraints') and request._structure_constraints:
                constraints = request._structure_constraints
                if constraints.is_from_documents:
                    # Validate output against pre-analyzed structure
                    validation = validate_output_against_constraints(
                        {"sections": [{"lectures": s.lectures} for s in sections]},
                        constraints
                    )
                    if not validation["valid"]:
                        print(f"[PLANNER] ⚠️ Structure validation issues:", flush=True)
                        for issue in validation["issues"]:
                            print(f"[PLANNER]    - {issue}", flush=True)
                    else:
                        print(f"[PLANNER] ✅ Structure validated against document analysis", flush=True)

        # Extract category from context if available
        category = None
        context_summary = data.get("context_summary", "")
        if request.context:
            category = request.context.category
            if not context_summary:
                context_summary = f"{request.context.profile_niche} - {request.context.profile_tone}"

        return CourseOutline(
            title=data.get("title", request.topic),
            description=data.get("description", f"A course about {request.topic}"),
            target_audience=data.get("target_audience", ""),
            category=category,
            context_summary=context_summary,
            language=getattr(request, 'language', 'en'),  # Pass through the content language
            difficulty_start=request.difficulty_start,
            difficulty_end=request.difficulty_end,
            total_duration_minutes=request.structure.total_duration_minutes,
            sections=sections
        )

    def _enforce_structure(
        self,
        sections: list,
        expected_sections: int,
        expected_lectures_per_section: int,
        duration_per_lecture: int,
        request: PreviewOutlineRequest
    ) -> list:
        """Enforce exact structure by padding or trimming sections and lectures"""
        from models.course_models import Section, Lecture

        # Handle sections count
        current_sections = len(sections)
        print(f"[PLANNER] Structure check: got {current_sections} sections, expected {expected_sections}", flush=True)

        if current_sections > expected_sections:
            # Trim extra sections
            sections = sections[:expected_sections]
            print(f"[PLANNER] Trimmed to {expected_sections} sections", flush=True)
        elif current_sections < expected_sections:
            # Pad with additional sections
            for i in range(current_sections, expected_sections):
                new_section = Section(
                    title=f"Section {i + 1}: Additional Topics",
                    description=f"Continuation of {request.topic} - advanced concepts",
                    order=i,
                    lectures=[]
                )
                sections.append(new_section)
            print(f"[PLANNER] Padded to {expected_sections} sections", flush=True)

        # Handle lectures per section
        for sec_idx, section in enumerate(sections):
            current_lectures = len(section.lectures)

            if current_lectures > expected_lectures_per_section:
                # Trim extra lectures
                section.lectures = section.lectures[:expected_lectures_per_section]
            elif current_lectures < expected_lectures_per_section:
                # Pad with additional lectures
                for lec_idx in range(current_lectures, expected_lectures_per_section):
                    # Determine difficulty based on position
                    progress = (sec_idx * expected_lectures_per_section + lec_idx) / (expected_sections * expected_lectures_per_section)
                    if progress < 0.33:
                        difficulty = request.difficulty_start
                    elif progress < 0.66:
                        difficulty = DifficultyLevel.INTERMEDIATE
                    else:
                        difficulty = request.difficulty_end

                    new_lecture = Lecture(
                        title=f"Lecture {lec_idx + 1}: Advanced Concepts",
                        description=f"Deep dive into {request.topic} concepts",
                        objectives=[
                            "Understand advanced concepts",
                            "Apply knowledge to real scenarios",
                            "Build practical skills"
                        ],
                        difficulty=difficulty,
                        duration_seconds=duration_per_lecture,
                        order=lec_idx
                    )
                    section.lectures.append(new_lecture)

            # Update lecture order
            for lec_idx, lecture in enumerate(section.lectures):
                lecture.order = lec_idx

        print(f"[PLANNER] Final structure: {len(sections)} sections, {sum(len(s.lectures) for s in sections)} total lectures", flush=True)
        return sections

    async def generate_lecture_prompt(
        self,
        lecture: Lecture,
        section: Section,
        outline: CourseOutline,
        lesson_elements: dict,
        position: int,
        total: int,
        rag_context: Optional[str] = None,
        programming_language: Optional[str] = None
    ) -> str:
        """Generate the prompt for a specific lecture to be sent to presentation-generator"""
        elements_text = []
        if lesson_elements.get("concept_intro", True):
            elements_text.append("- Start with a concept introduction slide explaining the theory")
        if lesson_elements.get("diagram_schema", True):
            elements_text.append("- Include visual diagrams or schemas to illustrate concepts (MANDATORY: at least 1-2 diagrams)")
        if lesson_elements.get("code_typing", True):
            code_lang = programming_language or "the appropriate language"
            elements_text.append(f"- Show code with typing animation (CODE_DEMO slides) - IMPORTANT: Include 2-4 code examples in {code_lang}")
            elements_text.append(f"- Each code example should build progressively from simple to more complex")
            elements_text.append(f"- Include comments in the code to explain key concepts")
        if lesson_elements.get("code_execution", False):
            elements_text.append("- Execute code and show the output (include expected output)")
        if lesson_elements.get("voiceover_explanation", True):
            elements_text.append("- Include detailed voiceover explanation during code")
        if lesson_elements.get("curriculum_slide", True):
            elements_text.append("- Start with a curriculum slide showing course position")

        elements_str = "\n".join(elements_text)

        # Build context section
        context_section = ""
        if outline.category:
            context_section = f"- Category: {outline.category.value}\n"
        if outline.context_summary:
            context_section += f"- Context: {outline.context_summary}\n"

        # Get language name for prompt
        content_language = outline.language or "en"
        language_name = LANGUAGE_NAMES.get(content_language, content_language)

        return f"""Create a video presentation for Lecture {position}/{total} in the course "{outline.title}".

**CRITICAL: ALL CONTENT MUST BE IN {language_name.upper()}**
- All titles, subtitles, and text content must be in {language_name}
- All voiceover narration text must be in {language_name}
- All bullet points and explanations must be in {language_name}
- Code comments SHOULD be in {language_name} for educational clarity
- Only code syntax/keywords remain in the programming language

COURSE CONTEXT:
- Course: {outline.title}
- Target Audience: {outline.target_audience}
- Content Language: {language_name} (code: {content_language})
{context_section}
SECTION: {section.title}
{section.description}

LECTURE: {lecture.title}
{lecture.description}

LEARNING OBJECTIVES:
{chr(10).join(f'- {obj}' for obj in lecture.objectives)}

DIFFICULTY LEVEL: {lecture.difficulty.value}
{self._get_difficulty_requirements(lecture.difficulty)}

TARGET DURATION: {lecture.duration_seconds} seconds

LESSON ELEMENTS TO INCLUDE:
{elements_str}

SLIDE STRUCTURE:
1. CURRICULUM - Show this lecture's position in the course (Section {section.order + 1}, Lecture {lecture.order + 1})
2. Follow with requested elements in logical order
3. End with a conclusion summarizing key takeaways

PROGRAMMING LANGUAGE/TOOLS: {programming_language or 'Not specified - use appropriate language based on topic'}

IMPORTANT REQUIREMENTS:
- This is lecture {position} of {total} in the course
- **LANGUAGE: Write ALL content in {language_name}** - this is MANDATORY
- STRICTLY MATCH the {lecture.difficulty.value} difficulty level as defined above
- CODE REQUIREMENT: Include MULTIPLE code examples (minimum 2-3) that progressively build understanding
- Each code example should demonstrate a specific concept from the learning objectives
- DIAGRAM REQUIREMENT: Include at least 1-2 visual diagrams/schemas to illustrate complex concepts
- Voiceover should be engaging and educational, explaining the code line by line (in {language_name})
- Adapt the content tone to match the course context
- Focus on the specific learning objectives listed above
- After each code block, pause to allow learner comprehension
{self._build_lecture_rag_section(rag_context)}"""

    def _get_difficulty_requirements(self, difficulty: DifficultyLevel) -> str:
        """Generate specific difficulty requirements for lecture content"""
        criteria = self._get_difficulty_criteria()
        c = criteria.get(difficulty, criteria[DifficultyLevel.INTERMEDIATE])

        return f"""
=== DIFFICULTY REQUIREMENTS FOR {c['name'].upper()} ===
- VOCABULARY: {c['vocabulary']}
- CONCEPTS: {c['concepts']}
- EXAMPLES: {c['examples']}
- CODE: {c['code_complexity']}
- PACE: {c['pace']}

Your content MUST demonstrate these indicators:
{chr(10).join(f'  • {ind}' for ind in c['indicators'])}

DO NOT include content appropriate for higher difficulty levels."""

    def _build_lecture_rag_section(self, rag_context: Optional[str]) -> str:
        """Build RAG context section for lecture prompt with deep integration instructions"""
        if not rag_context:
            return ""

        # Increased token limit for better RAG integration (was 3000, now 5000)
        max_lecture_context = 5000
        context_tokens = self.count_tokens(rag_context)
        if context_tokens > max_lecture_context:
            print(f"[PLANNER] Lecture RAG context too large ({context_tokens} tokens), truncating to {max_lecture_context}", flush=True)
            rag_context = self.truncate_to_tokens(rag_context, max_lecture_context)

        return f"""
=== CRITICAL: SOURCE DOCUMENT CONTENT (MUST USE) ===

The following content has been extracted from the user's uploaded documents (PDFs, videos, URLs, etc.).
This is the PRIMARY SOURCE for this lecture. You MUST integrate this content deeply.

---BEGIN SOURCE CONTENT---
{rag_context}
---END SOURCE CONTENT---

=== MANDATORY RAG INTEGRATION REQUIREMENTS ===

1. **CONTENT PRIORITY**: The source documents are your PRIMARY reference. Build the lecture content AROUND the information in these documents.

2. **DIRECT INFORMATION**: Include specific facts, figures, definitions, and examples found in the source documents:
   - Use exact terminology from the documents
   - Include specific numbers, statistics, or data mentioned
   - Reference frameworks, models, or methodologies described
   - Use examples and case studies from the documents

3. **SLIDE CONTENT REQUIREMENTS**:
   - Each slide MUST contain at least one piece of information from the source documents
   - Quote or paraphrase key concepts directly from the documents
   - If the document mentions a list or steps, include them in your slides
   - Include any diagrams, charts, or visual concepts described in the documents

4. **VOICEOVER REQUIREMENTS**:
   - Explain concepts using the same language and terminology as the source documents
   - Reference "as mentioned in the course materials" or "according to the source content"
   - Expand on document content with additional context, but keep the core information intact

5. **WHAT TO INCLUDE FROM DOCUMENTS**:
   - Key definitions and explanations
   - Step-by-step processes or procedures
   - Best practices and recommendations
   - Common mistakes or warnings mentioned
   - Real-world examples or case studies
   - Technical specifications or requirements

6. **ACCURACY**: Do NOT invent information. If the documents don't cover something, either:
   - Skip that topic, OR
   - Clearly indicate it's supplementary information

The user uploaded these documents specifically to create a course based on their content.
Failing to use this content deeply would make the generated course useless to them."""
