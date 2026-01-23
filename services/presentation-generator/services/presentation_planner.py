"""
Presentation Planner Service

Uses GPT-4 to generate a structured presentation script from a topic prompt.
Enhanced with TechPromptBuilder for domain-specific, professional-grade content.
"""
import json
import os
from typing import Optional, List
import httpx
from openai import AsyncOpenAI

from models.presentation_models import (
    PresentationScript,
    Slide,
    SlideType,
    CodeBlock,
    GeneratePresentationRequest,
)
from services.tech_prompt_builder import (
    TechPromptBuilder,
    AudienceLevel,
)
from models.tech_domains import (
    TechDomain,
    TechCareer,
    CodeLanguage,
)


PLANNING_SYSTEM_PROMPT = """You are an expert technical presenter and course creator. Your task is to create a structured presentation script for a coding tutorial.

You will receive:
- A topic/prompt describing what to teach
- The target programming language
- The CONTENT LANGUAGE (e.g., 'en', 'fr', 'es') - ALL text content MUST be in this language
- The target duration in seconds
- The target audience level

CRITICAL LANGUAGE COMPLIANCE:
- ALL voiceover_text, titles, subtitles, bullet_points, content, and notes MUST be written in the specified content language
- Code comments can be in the content language for educational purposes
- Variable names and syntax keywords stay in the programming language (usually English)
- If content language is 'fr', write in French. If 'es', write in Spanish. If 'de', write in German. etc.
- NEVER mix languages in the content - be consistent throughout

GRAMMAR AND STYLE (especially for non-English):
- Write with PERFECT grammar - no spelling or grammatical errors
- For FRENCH: Use proper accents (é, è, ê, à, ù, ç), correct agreements, proper verb conjugations
- For FRENCH: Write natural, professional French - not literal translations from English
- Use formal register appropriate for educational content

Your output MUST be a valid JSON object with this structure:
{
  "title": "Presentation Title",
  "description": "Brief description of what viewers will learn",
  "target_audience": "Who this is for",
  "language": "python",
  "total_duration": 300,
  "slides": [
    {
      "type": "title|content|code|code_demo|diagram|conclusion",
      "title": "Slide Title",
      "subtitle": "Optional subtitle",
      "content": "Main text content (markdown supported)",
      "bullet_points": ["Point 1", "Point 2"],
      "code_blocks": [
        {
          "language": "python",
          "code": "actual working code here",
          "filename": "example.py",
          "highlight_lines": [1, 3],
          "expected_output": "what this code outputs"
        }
      ],
      "duration": 10.0,
      "voiceover_text": "What the narrator should say for this slide",
      "notes": "Speaker notes"
    }
  ],
  "code_context": {}
}

SYNC ANCHORS FOR TIMELINE PRECISION:
Include [SYNC:slide_XXX] markers at the START of each slide's voiceover_text.
Example: "[SYNC:slide_001] Welcome to this tutorial on Python basics."
This enables precise audio-video synchronization.

IMPORTANT RULES:
1. Start with a "title" slide introducing the topic
2. Use "content" slides for explanations with bullet points
3. Use "code" slides to show code snippets with syntax highlighting
4. Use "code_demo" slides when you want to show code being executed
5. End with a "conclusion" slide summarizing key points
6. The voiceover_text should be natural, conversational narration
7. All code MUST be syntactically correct and functional
8. Duration should be based on voiceover length (~150 words per minute)
9. Include 8-15 slides for a 5-minute presentation
10. Progress from simple to complex concepts
11. Each slide's voiceover_text MUST start with [SYNC:slide_XXX] where XXX is the slide index (001, 002, etc.)

SLIDE TYPE GUIDELINES:
- title: Main presentation title with subtitle (MUST have voiceover_text introducing the topic)
- content: Bullet points explaining concepts (3-5 points)
- code: Code snippet with explanation (include voiceover explaining line by line)
- code_demo: Code that will be executed with expected output
- diagram: Visual diagram to explain architecture, processes, or comparisons. MUST include:
  * "diagram_type": one of "flowchart", "architecture", "process", "comparison", "hierarchy"
  * "content": detailed description of what the diagram should show (nodes, connections, labels)
  * Use diagrams when explaining: system architecture, data flow, step-by-step processes, comparisons between options
- conclusion: Summary slide with bullet_points listing 3-5 key takeaways. MUST include bullet_points array AND voiceover_text that summarizes what was learned and thanks the viewer.

DIAGRAM USAGE (IMPORTANT):
- Include at least 1-2 diagram slides for topics involving architecture, workflows, or comparisons
- For "How X works" topics: use a flowchart or process diagram
- For "X vs Y" topics: use a comparison diagram
- For system/architecture topics: use an architecture diagram

CRITICAL: EVERY slide MUST have a non-empty voiceover_text field. The conclusion slide voiceover should recap the key points and end with a natural closing like "Thanks for watching!" or "That's it for this tutorial!".

Output ONLY valid JSON, no markdown code blocks or additional text."""


# Enhanced prompt with visual-audio alignment validation and sync anchors
VALIDATED_PLANNING_PROMPT = """You are an expert technical presenter creating VIDEO presentations. Your task is to create a structured presentation script where the voiceover PERFECTLY MATCHES the visuals.

CRITICAL LANGUAGE COMPLIANCE:
- ALL voiceover_text, titles, subtitles, bullet_points, content, and notes MUST be written in the specified CONTENT LANGUAGE
- Code comments can be in the content language for educational purposes
- Variable names and syntax keywords stay in the programming language (usually English)
- If content language is 'fr', write in French. If 'es', write in Spanish. If 'de', write in German. etc.
- NEVER mix languages in the content - be consistent throughout

GRAMMAR AND STYLE REQUIREMENTS (especially for non-English):
- Write with PERFECT grammar - no spelling or grammatical errors
- For FRENCH (fr): Use proper accents (é, è, ê, à, ù, ç, etc.), correct article agreements (le/la/les), proper verb conjugations
- For FRENCH: Use natural, professional French - not literal translations from English
- For FRENCH: Avoid anglicisms when French alternatives exist (e.g., "logiciel" not "software")
- Use formal/professional register appropriate for educational content
- Proofread all text for language-specific errors

SYNC ANCHORS FOR TIMELINE PRECISION:
To enable millisecond-precision audio-video synchronization, include [SYNC:slide_id] markers in the voiceover_text.
These markers indicate when a slide should appear based on the narration timing.

Example:
- "Welcome to this tutorial. [SYNC:slide_001] Let's start by understanding the basics."
- The slide_001 will appear exactly when the word after [SYNC:slide_001] is spoken.

Rules for sync anchors:
1. Place [SYNC:slide_id] at the BEGINNING of text content specific to that slide
2. The slide_id MUST match the "id" field of the corresponding slide
3. Every slide should have its sync anchor in the combined voiceover text
4. Do NOT include the sync anchor brackets in the actual spoken text (they are markers only)

CRITICAL VISUAL-AUDIO ALIGNMENT RULES:
1. NEVER say "as you can see", "look at this diagram", "notice here" unless the slide VISUALLY shows exactly what you're describing
2. For "code" slides: The voiceover must describe what the CODE LOOKS LIKE, not what happens when it runs
3. For "code_demo" slides: You CAN describe execution and output because it will be shown
4. For "diagram" slides: Provide detailed diagram_description so it can be generated
5. For "content" slides: Only reference bullet points that are actually visible

TIMING ALIGNMENT:
- Calculate duration based on: max(voiceover_time, animation_time, visual_time)
- Code typing animation: ~30 characters/second
- Voiceover: ~150 words/minute (2.5 words/second)
- Add 2 seconds buffer per slide for transitions

SLIDE STRUCTURE:
{
  "title": "Presentation Title",
  "description": "Brief description",
  "target_audience": "Who this is for",
  "language": "python",
  "total_duration": 300,
  "slides": [
    {
      "id": "slide_001",
      "type": "title|content|code|code_demo|diagram|conclusion",
      "title": "Slide Title",
      "subtitle": "Optional subtitle",
      "content": "Main text content",
      "bullet_points": ["Point 1", "Point 2"],
      "code_blocks": [
        {
          "language": "python",
          "code": "print('hello')",
          "filename": "example.py",
          "highlight_lines": [1],
          "expected_output": "hello",
          "show_execution": true
        }
      ],
      "diagram_description": "For diagram slides: detailed description of what to draw",
      "diagram_type": "flowchart|architecture|sequence|mindmap|comparison",
      "duration": 15.0,
      "voiceover_text": "[SYNC:slide_001] Narration that matches EXACTLY what's visible",
      "visual_cues": ["code appears", "output shows"],
      "timing_notes": "Wait for code typing to complete before mentioning output"
    }
  ]
}

FORBIDDEN PHRASES (unless slide type matches):
- "this diagram shows..." (only on diagram slides)
- "as you can see the output..." (only on code_demo slides)
- "watch as..." (only on animation slides)
- "here we have a chart..." (only if chart is actually shown)

REQUIRED FOR EACH SLIDE TYPE:
- title: voiceover_text introducing topic (with [SYNC:slide_id] at start)
- content: bullet_points array with items mentioned in voiceover
- code: code_blocks with actual code, voiceover describes the code structure
- code_demo: code_blocks with expected_output, voiceover can mention execution
- diagram: diagram_description (50+ words), diagram_type, voiceover describes the diagram
- conclusion: bullet_points summary, voiceover recap

Output ONLY valid JSON."""


class PresentationPlannerService:
    """Service for planning presentation structure using GPT-4"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=120.0,  # 2 minutes timeout for GPT-4 calls
            max_retries=2
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        # Initialize the tech prompt builder for enhanced code/diagram generation
        self.prompt_builder = TechPromptBuilder()

    async def generate_script(
        self,
        request: GeneratePresentationRequest,
        on_progress: Optional[callable] = None
    ) -> PresentationScript:
        """
        Generate a complete presentation script from a topic prompt.

        Args:
            request: The presentation generation request
            on_progress: Optional callback for progress updates

        Returns:
            PresentationScript with all slides defined
        """
        if on_progress:
            await on_progress(0, "Analyzing topic...")

        # Build the user prompt
        user_prompt = self._build_prompt(request)

        if on_progress:
            await on_progress(10, "Generating presentation structure...")

        # Call GPT-4 to generate the script
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )

        if on_progress:
            await on_progress(80, "Parsing presentation script...")

        # Parse the response
        content = response.choices[0].message.content
        script_data = json.loads(content)

        # Ensure sync anchors are present for timeline composition
        script_data = self._ensure_sync_anchors(script_data)

        # Convert to PresentationScript
        script = self._parse_script(script_data, request)

        if on_progress:
            await on_progress(100, "Script generation complete")

        return script

    # Language code to full name mapping
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
    }

    def _get_language_name(self, code: str) -> str:
        """Get full language name from code"""
        return self.LANGUAGE_NAMES.get(code.lower(), code)

    def _map_audience_level(self, audience: str) -> AudienceLevel:
        """Map audience string to AudienceLevel enum."""
        mapping = {
            "beginner": AudienceLevel.BEGINNER,
            "absolute beginner": AudienceLevel.ABSOLUTE_BEGINNER,
            "débutant": AudienceLevel.BEGINNER,
            "débutant absolu": AudienceLevel.ABSOLUTE_BEGINNER,
            "intermediate": AudienceLevel.INTERMEDIATE,
            "intermédiaire": AudienceLevel.INTERMEDIATE,
            "advanced": AudienceLevel.ADVANCED,
            "avancé": AudienceLevel.ADVANCED,
            "expert": AudienceLevel.EXPERT,
        }
        return mapping.get(audience.lower(), AudienceLevel.INTERMEDIATE)

    def _detect_domain(self, topic: str, language: str) -> Optional[TechDomain]:
        """Detect tech domain from topic and programming language."""
        topic_lower = topic.lower()

        # Domain detection keywords
        domain_keywords = {
            TechDomain.DATA_ENGINEERING: ["data pipeline", "etl", "data lake", "spark", "airflow", "data warehouse"],
            TechDomain.MACHINE_LEARNING: ["machine learning", "ml", "neural", "deep learning", "tensorflow", "pytorch", "model training"],
            TechDomain.DEVOPS: ["devops", "ci/cd", "deployment", "infrastructure", "terraform", "ansible"],
            TechDomain.KUBERNETES: ["kubernetes", "k8s", "helm", "kubectl", "pod", "deployment"],
            TechDomain.CLOUD_AWS: ["aws", "amazon", "lambda", "s3", "ec2", "dynamodb"],
            TechDomain.CLOUD_AZURE: ["azure", "microsoft cloud", "azure function"],
            TechDomain.CLOUD_GCP: ["gcp", "google cloud", "bigquery", "cloud run"],
            TechDomain.CYBERSECURITY: ["security", "authentication", "encryption", "vulnerability", "pentest"],
            TechDomain.BLOCKCHAIN: ["blockchain", "smart contract", "solidity", "web3", "ethereum"],
            TechDomain.QUANTUM_COMPUTING: ["quantum", "qubit", "qiskit", "cirq"],
            TechDomain.WEB_FRONTEND: ["react", "vue", "angular", "frontend", "css", "html"],
            TechDomain.WEB_BACKEND: ["api", "rest", "graphql", "backend", "server"],
            TechDomain.MOBILE_DEVELOPMENT: ["mobile", "ios", "android", "flutter", "react native"],
            TechDomain.DATABASES: ["database", "sql", "mongodb", "postgres", "mysql", "redis"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in topic_lower for kw in keywords):
                return domain

        return None

    def _detect_code_language(self, language: str) -> Optional[CodeLanguage]:
        """Map programming language string to CodeLanguage enum."""
        mapping = {
            "python": CodeLanguage.PYTHON,
            "javascript": CodeLanguage.JAVASCRIPT,
            "typescript": CodeLanguage.TYPESCRIPT,
            "java": CodeLanguage.JAVA,
            "go": CodeLanguage.GO,
            "golang": CodeLanguage.GO,
            "rust": CodeLanguage.RUST,
            "c++": CodeLanguage.CPP,
            "cpp": CodeLanguage.CPP,
            "c#": CodeLanguage.CSHARP,
            "csharp": CodeLanguage.CSHARP,
            "kotlin": CodeLanguage.KOTLIN,
            "swift": CodeLanguage.SWIFT,
            "ruby": CodeLanguage.RUBY,
            "php": CodeLanguage.PHP,
            "scala": CodeLanguage.SCALA,
            "r": CodeLanguage.R,
            "sql": CodeLanguage.SQL,
            "bash": CodeLanguage.BASH,
            "shell": CodeLanguage.BASH,
            "terraform": CodeLanguage.TERRAFORM,
            "yaml": CodeLanguage.YAML,
            "dockerfile": CodeLanguage.DOCKERFILE,
            "solidity": CodeLanguage.SOLIDITY,
        }
        return mapping.get(language.lower())

    def _build_prompt(self, request: GeneratePresentationRequest) -> str:
        """Build the prompt for GPT-4 with enhanced code quality standards."""
        minutes = request.duration // 60
        seconds = request.duration % 60

        duration_str = f"{minutes} minutes"
        if seconds > 0:
            duration_str += f" {seconds} seconds"

        # Get content language name
        content_lang = getattr(request, 'content_language', 'en')
        content_lang_name = self._get_language_name(content_lang)

        # Detect domain and audience level for enhanced prompts
        domain = self._detect_domain(request.topic, request.language)
        audience_level = self._map_audience_level(request.target_audience)
        code_language = self._detect_code_language(request.language)

        # Build enhanced code quality prompt using TechPromptBuilder
        code_languages = [code_language] if code_language else None

        enhanced_code_prompt = self.prompt_builder.build_code_prompt(
            topic=request.topic,
            domain=domain,
            audience_level=audience_level,
            languages=code_languages,
            content_language=content_lang
        )

        return f"""Create a presentation script for the following:

TOPIC: {request.topic}

PARAMETERS:
- Programming Language: {request.language}
- CONTENT LANGUAGE: {content_lang_name} (code: {content_lang})
- Target Duration: {duration_str} ({request.duration} seconds total)
- Target Audience: {request.target_audience}
- Visual Style: {request.style.value}

IMPORTANT: ALL text content (titles, subtitles, voiceover_text, bullet_points, content) MUST be written in {content_lang_name}.
- Code syntax and keywords stay in the programming language
- Code comments SHOULD be in {content_lang_name} for educational clarity

{enhanced_code_prompt}

Please create a well-structured, educational presentation that:
1. Introduces the topic clearly in {content_lang_name}
2. Explains concepts progressively with the teaching style appropriate for {request.target_audience}
3. Includes practical code examples that are 100% functional and follow ALL quality standards above
4. Has natural, engaging narration text in {content_lang_name}
5. Ends with a clear summary in {content_lang_name}

The presentation should feel like a high-quality tutorial from platforms like Udemy or Coursera."""

    def _parse_script(
        self,
        data: dict,
        request: GeneratePresentationRequest
    ) -> PresentationScript:
        """Parse GPT-4 response into PresentationScript"""
        slides = []

        for slide_data in data.get("slides", []):
            # Parse code blocks
            code_blocks = []
            for cb_data in slide_data.get("code_blocks", []):
                code_blocks.append(CodeBlock(
                    language=cb_data.get("language", request.language),
                    code=cb_data.get("code", ""),
                    filename=cb_data.get("filename"),
                    highlight_lines=cb_data.get("highlight_lines", []),
                    execution_order=cb_data.get("execution_order", 0),
                    expected_output=cb_data.get("expected_output"),
                    show_line_numbers=cb_data.get("show_line_numbers", True)
                ))

            # Parse slide type
            slide_type_str = slide_data.get("type", "content")
            try:
                slide_type = SlideType(slide_type_str)
            except ValueError:
                slide_type = SlideType.CONTENT

            # For diagram slides, use diagram_description as content if content is empty
            content = slide_data.get("content")

            # Handle case where GPT returns content as a dict (e.g., for diagrams)
            if isinstance(content, dict):
                # Extract description from dict or convert to string
                content = content.get("description") or content.get("diagram_description") or str(content)

            if slide_type == SlideType.DIAGRAM and not content:
                content = slide_data.get("diagram_description", "")

            slides.append(Slide(
                type=slide_type,
                title=slide_data.get("title"),
                subtitle=slide_data.get("subtitle"),
                content=content,
                bullet_points=slide_data.get("bullet_points", []),
                code_blocks=code_blocks,
                duration=slide_data.get("duration", 10.0),
                voiceover_text=slide_data.get("voiceover_text", ""),
                transition=slide_data.get("transition", "fade"),
                notes=slide_data.get("notes"),
                diagram_type=slide_data.get("diagram_type"),
                index=len(slides)
            ))

        return PresentationScript(
            title=data.get("title", "Untitled Presentation"),
            description=data.get("description", ""),
            target_audience=data.get("target_audience", request.target_audience),
            language=data.get("language", request.language),
            total_duration=data.get("total_duration", request.duration),
            slides=slides,
            code_context=data.get("code_context", {})
        )

    async def generate_script_with_validation(
        self,
        request: GeneratePresentationRequest,
        on_progress: Optional[callable] = None
    ) -> PresentationScript:
        """
        Generate a presentation script with enhanced visual-audio alignment.

        Uses the VALIDATED_PLANNING_PROMPT which enforces strict rules about
        matching voiceover content with visual elements.

        Args:
            request: The presentation generation request
            on_progress: Optional callback for progress updates

        Returns:
            PresentationScript with validated visual-audio alignment
        """
        if on_progress:
            await on_progress(0, "Analyzing topic with validation rules...")

        user_prompt = self._build_validated_prompt(request)

        if on_progress:
            await on_progress(10, "Generating validated presentation structure...")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": VALIDATED_PLANNING_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,  # Lower temperature for more consistent output
            max_tokens=4000,  # Max tokens for detailed descriptions
            response_format={"type": "json_object"}
        )

        if on_progress:
            await on_progress(80, "Parsing validated script...")

        content = response.choices[0].message.content
        script_data = json.loads(content)

        # Post-process to ensure IDs and calculate timing
        script_data = self._post_process_validated_script(script_data)

        script = self._parse_script(script_data, request)

        if on_progress:
            await on_progress(100, "Validated script generation complete")

        return script

    def _build_validated_prompt(self, request: GeneratePresentationRequest) -> str:
        """Build prompt for validated script generation with enhanced code quality."""
        minutes = request.duration // 60
        seconds = request.duration % 60
        duration_str = f"{minutes} minutes" + (f" {seconds} seconds" if seconds > 0 else "")

        # Get content language name
        content_lang = getattr(request, 'content_language', 'en')
        content_lang_name = self._get_language_name(content_lang)

        # Detect domain and audience level for enhanced prompts
        domain = self._detect_domain(request.topic, request.language)
        audience_level = self._map_audience_level(request.target_audience)
        code_language = self._detect_code_language(request.language)

        # Build enhanced code quality prompt using TechPromptBuilder
        code_languages = [code_language] if code_language else None

        enhanced_code_prompt = self.prompt_builder.build_code_prompt(
            topic=request.topic,
            domain=domain,
            audience_level=audience_level,
            languages=code_languages,
            content_language=content_lang
        )

        return f"""Create a VIDEO presentation script for:

TOPIC: {request.topic}

PARAMETERS:
- Programming Language: {request.language}
- CONTENT LANGUAGE: {content_lang_name} (code: {content_lang})
- Target Duration: {duration_str} ({request.duration} seconds total)
- Target Audience: {request.target_audience}
- Visual Style: {request.style.value}

IMPORTANT LANGUAGE REQUIREMENT:
ALL text content (titles, subtitles, voiceover_text, bullet_points, content, notes) MUST be written in {content_lang_name}.
Code syntax stays in the programming language, but code comments SHOULD be in {content_lang_name}.

{enhanced_code_prompt}

REQUIREMENTS:
1. Every visual reference in voiceover MUST have corresponding visual content
2. Code slides: describe the code structure, NOT the execution
3. Code demo slides: CAN describe execution since output will be shown
4. Diagram slides: provide detailed diagram_description for generation
5. Calculate duration = max(voiceover_time, code_length/30, bullet_count*2) + 2s buffer
6. ALL text content MUST be in {content_lang_name}
7. ALL code must follow the CODE QUALITY STANDARDS above

Create a well-structured tutorial in {content_lang_name} where the viewer sees EXACTLY what the narrator describes."""

    def _ensure_sync_anchors(self, script_data: dict) -> dict:
        """
        Ensure all slides have sync anchors in their voiceover text.

        If a slide's voiceover doesn't have a sync anchor, add one at the beginning.
        This enables timeline-based composition for precise audio-video sync.
        """
        import re

        slides = script_data.get("slides", [])

        for i, slide in enumerate(slides):
            slide_id = slide.get("id", f"slide_{i:03d}")
            voiceover = slide.get("voiceover_text", "")

            # Check if sync anchor already exists
            sync_pattern = r'\[SYNC:[\w_]+\]'
            has_sync = bool(re.search(sync_pattern, voiceover))

            if not has_sync and voiceover:
                # Add sync anchor at the beginning
                slide["voiceover_text"] = f"[SYNC:{slide_id}] {voiceover}"

        script_data["slides"] = slides
        return script_data

    def _post_process_validated_script(self, script_data: dict) -> dict:
        """Post-process the script to ensure consistency"""
        slides = script_data.get("slides", [])

        for i, slide in enumerate(slides):
            # Ensure ID exists
            if "id" not in slide:
                slide["id"] = f"slide_{i:03d}"

            # Calculate proper duration based on content
            voiceover = slide.get("voiceover_text", "")
            # Remove sync anchors for word count
            import re
            clean_voiceover = re.sub(r'\[SYNC:[\w_]+\]', '', voiceover).strip()
            word_count = len(clean_voiceover.split())
            voiceover_duration = word_count / 2.5  # 150 words/minute

            # Code animation duration
            animation_duration = 0
            for block in slide.get("code_blocks", []):
                code_len = len(block.get("code", ""))
                animation_duration = max(animation_duration, code_len / 30.0)

            # Visual reading time
            bullets = slide.get("bullet_points", [])
            visual_duration = len(bullets) * 2.0

            # Set duration to max + buffer
            calculated = max(voiceover_duration, animation_duration, visual_duration) + 2.0

            # Only override if significantly different
            current = slide.get("duration", 10.0)
            if abs(current - calculated) > 5:
                slide["duration"] = round(calculated, 1)

        script_data["slides"] = slides

        # Ensure sync anchors are present
        script_data = self._ensure_sync_anchors(script_data)

        return script_data

    async def refine_slide(
        self,
        slide: Slide,
        feedback: str,
        language: str
    ) -> Slide:
        """
        Refine a single slide based on feedback.

        Args:
            slide: The slide to refine
            feedback: User feedback for improvement
            language: Programming language context

        Returns:
            Refined Slide object
        """
        prompt = f"""Refine this presentation slide based on the feedback:

CURRENT SLIDE:
{json.dumps(slide.model_dump(), indent=2)}

FEEDBACK:
{feedback}

LANGUAGE: {language}

Return the improved slide as JSON with the same structure.
Output ONLY valid JSON."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a presentation expert. Improve slides based on feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        slide_data = json.loads(response.choices[0].message.content)

        # Parse code blocks
        code_blocks = []
        for cb_data in slide_data.get("code_blocks", []):
            code_blocks.append(CodeBlock(
                language=cb_data.get("language", language),
                code=cb_data.get("code", ""),
                filename=cb_data.get("filename"),
                highlight_lines=cb_data.get("highlight_lines", []),
                expected_output=cb_data.get("expected_output")
            ))

        slide_type_str = slide_data.get("type", slide.type.value)
        try:
            slide_type = SlideType(slide_type_str)
        except ValueError:
            slide_type = slide.type

        # For diagram slides, use diagram_description as content if content is empty
        content = slide_data.get("content")
        if slide_type == SlideType.DIAGRAM and not content:
            content = slide_data.get("diagram_description", "")

        return Slide(
            id=slide.id,
            type=slide_type,
            title=slide_data.get("title", slide.title),
            subtitle=slide_data.get("subtitle"),
            content=content,
            bullet_points=slide_data.get("bullet_points", []),
            code_blocks=code_blocks,
            duration=slide_data.get("duration", slide.duration),
            voiceover_text=slide_data.get("voiceover_text", slide.voiceover_text),
            transition=slide_data.get("transition", slide.transition),
            notes=slide_data.get("notes"),
            diagram_type=slide_data.get("diagram_type"),
            index=getattr(slide, 'index', 0)
        )
