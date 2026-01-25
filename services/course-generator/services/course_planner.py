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
    USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    USE_SHARED_LLM = False
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

        # Check if RAG context is available
        has_rag_context = bool(request.rag_context)
        if has_rag_context:
            print(f"[PLANNER] Using RAG context ({len(request.rag_context)} chars)", flush=True)

        # Build the prompt
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
            temperature=0.7,
            max_tokens=4000
        )

        # Parse response
        content = response.choices[0].message.content
        outline_data = json.loads(content)

        # Convert to CourseOutline
        outline = self._parse_outline(outline_data, request)

        print(f"[PLANNER] Generated: {outline.section_count} sections, {outline.total_lectures} lectures", flush=True)

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
        """System prompt for curriculum generation"""
        base_prompt = """You are an expert curriculum designer specializing in educational course creation.

Your task is to create well-structured, comprehensive course outlines that:
1. Progress logically from simple to complex concepts
2. Include practical, hands-on examples when appropriate
3. Balance theory with application
4. Have clear learning objectives for each lecture
5. Maintain consistent quality throughout the course"""

        if has_source_documents:
            base_prompt += """

=== CRITICAL: SOURCE DOCUMENT MODE ===

The user has uploaded source documents (PDFs, videos, URLs, articles) that contain the EXACT content they want to teach.
Your PRIMARY task is to transform their document content into a structured course.

YOU MUST:
1. **USE THE DOCUMENTS AS YOUR MAIN SOURCE**: The course outline must be based on the content in the documents, not generic knowledge.
2. **MAP DOCUMENT SECTIONS TO COURSE SECTIONS**: Each major topic in the documents should become a course section.
3. **EXTRACT SPECIFIC CONTENT**: Include specific facts, figures, examples, and explanations from the documents.
4. **MAINTAIN FIDELITY**: Do not add topics that are not in the source documents unless absolutely necessary for context.
5. **COVER EVERYTHING**: Ensure ALL major topics from the documents are included in the curriculum.

The user chose these specific documents because they contain the exact knowledge they want to share.
A course that ignores the document content is USELESS to the user."""

        base_prompt += "\n\nYou must respond with valid JSON only."
        return base_prompt

    def _build_curriculum_prompt(self, request: PreviewOutlineRequest) -> str:
        """Build the prompt for curriculum generation"""
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
STRICT COURSE STRUCTURE REQUIREMENTS (MUST BE FOLLOWED EXACTLY):
- Total Duration: {request.structure.total_duration_minutes} minutes
- EXACTLY {request.structure.number_of_sections} sections (no more, no less)
- EXACTLY {request.structure.lectures_per_section} lectures per section
- Total lectures: {total_lectures}
- Target duration per lecture: ~{lecture_duration} minutes

IMPORTANT: You MUST create exactly {request.structure.number_of_sections} sections with exactly {request.structure.lectures_per_section} lectures each. Do not deviate from this structure."""

        # Build context section if available
        context_section = self._build_context_section(request.context)

        # Get category-specific instructions
        category_instructions = ""
        if request.context:
            category_instructions = self._get_category_specific_instructions(request.context.category)

        # Build RAG context section if available
        rag_section = self._build_rag_section(request.rag_context)

        # Build keywords section if available
        keywords_section = self._build_keywords_section(getattr(request, 'keywords', None))

        # Get language name for prompt
        content_language = getattr(request, 'language', 'en')
        language_name = LANGUAGE_NAMES.get(content_language, content_language)

        return f"""Create a comprehensive course outline for the following:

**CRITICAL: ALL CONTENT MUST BE IN {language_name.upper()}**
- Course title must be in {language_name}
- Course description must be in {language_name}
- All section titles and descriptions must be in {language_name}
- All lecture titles, descriptions, and objectives must be in {language_name}
- Target audience description must be in {language_name}

TOPIC: {request.topic}
{f'DESCRIPTION: {request.description}' if request.description else ''}
CONTENT LANGUAGE: {language_name} (code: {content_language})

{context_section}
{keywords_section}
{rag_section}
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
1. CRITICAL: Follow the exact structure requirements above - the exact number of sections and lectures per section MUST be respected
2. **LANGUAGE: Write ALL content in {language_name}** - this is MANDATORY
3. Each section should have a clear theme
4. Lectures within a section should build upon each other
5. Include 3-5 specific learning objectives per lecture (in {language_name})
6. Ensure smooth difficulty progression throughout the course
7. Make titles engaging and specific (avoid generic names) - in {language_name}
8. Each lecture should be self-contained but connected to the overall narrative
9. Adapt content to the specified audience and communication tone
{category_instructions}"""

    def _build_rag_section(self, rag_context: Optional[str]) -> str:
        """Build the RAG source documents section of the prompt with deep integration"""
        if not rag_context:
            return ""

        # Check token count and truncate if necessary
        context_tokens = self.count_tokens(rag_context)
        if context_tokens > self.MAX_RAG_CONTEXT_TOKENS:
            print(f"[PLANNER] RAG context too large ({context_tokens} tokens), truncating to {self.MAX_RAG_CONTEXT_TOKENS}", flush=True)
            rag_context = self.truncate_to_tokens(rag_context, self.MAX_RAG_CONTEXT_TOKENS)

        return f"""
=== CRITICAL: USER'S SOURCE DOCUMENTS (PRIMARY CONTENT SOURCE) ===

The user has uploaded documents (PDFs, videos, URLs, etc.) that contain the content they want to teach.
This content is your PRIMARY AND MANDATORY source for creating the course curriculum.

---BEGIN SOURCE CONTENT---
{rag_context}
---END SOURCE CONTENT---

=== MANDATORY OUTLINE REQUIREMENTS BASED ON SOURCE DOCUMENTS ===

1. **STRUCTURE THE COURSE AROUND THE DOCUMENTS**:
   - Analyze the source content and identify all major topics/chapters
   - Create sections that DIRECTLY map to the topics in the documents
   - Each lecture should cover specific content from the documents

2. **SECTION CREATION RULES**:
   - Each section title should reflect a major topic from the source documents
   - Section descriptions must reference what will be covered from the source material

3. **LECTURE CREATION RULES**:
   - Each lecture MUST be based on specific content from the source documents
   - Lecture titles should match topics/subtopics found in the documents
   - Lecture descriptions should summarize what document content will be covered
   - Learning objectives MUST be derived from the source document content

4. **COVERAGE REQUIREMENTS**:
   - ALL major topics from the source documents must appear in the curriculum
   - Do NOT add topics that are NOT in the source documents (unless essential for context)
   - The course outline should essentially be a structured reorganization of the source content

5. **WHAT TO EXTRACT FROM DOCUMENTS**:
   - Main chapters/sections → Course sections
   - Subtopics/subsections → Individual lectures
   - Key concepts → Learning objectives
   - Examples mentioned → Practical content flags

The user uploaded these specific documents because they want a course based on THIS content.
Do NOT create a generic course - create one that teaches exactly what's in the documents.
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
                # Convert snake_case to human-readable
                readable_key = key.replace("_", " ").title()
                lines.append(f"- {readable_key}: {value}")

        if context.specific_tools:
            lines.append(f"\nTools/Technologies: {context.specific_tools}")

        if context.practical_focus:
            lines.append(f"Practical Focus: {context.practical_focus}")

        if context.expected_outcome:
            lines.append(f"Expected Outcome: {context.expected_outcome}")

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
        """Parse GPT-4 response into CourseOutline model and enforce structure"""
        sections = []

        # Get expected structure
        expected_sections = request.structure.number_of_sections
        expected_lectures_per_section = request.structure.lectures_per_section
        is_random = request.structure.random_structure

        # Calculate duration per lecture based on expected structure
        if is_random:
            total_lectures_generated = sum(
                len(s.get("lectures", []))
                for s in data.get("sections", [])
            )
            duration_per_lecture = (
                request.structure.total_duration_minutes * 60 // max(total_lectures_generated, 1)
            )
        else:
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

        # Enforce exact structure if not random
        if not is_random:
            sections = self._enforce_structure(
                sections,
                expected_sections,
                expected_lectures_per_section,
                duration_per_lecture,
                request
            )

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
