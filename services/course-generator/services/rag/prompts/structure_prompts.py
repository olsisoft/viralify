"""
Structure Extraction Prompts

Prompt builders for extracting document structure (TOC, headings, sections)
when explicit structure is not available.
"""

from typing import List

from .base_prompt import BasePromptBuilder, PromptExample, PromptSection


class StructureExtractionPromptBuilder(BasePromptBuilder):
    """
    Builder for document structure extraction prompts.

    Used when documents lack explicit TOC/headings and we need to
    infer the logical structure from content.

    Usage:
        builder = StructureExtractionPromptBuilder(
            source_type="youtube",
            has_chapters=False,
            content_language="en"
        )
        system_prompt = builder.build()
    """

    def __init__(
        self,
        source_type: str = "document",
        has_chapters: bool = False,
        content_language: str = "en",
    ):
        """
        Initialize the structure extraction prompt builder.

        Args:
            source_type: Type of source (document, youtube, url)
            has_chapters: Whether the source has explicit chapters
            content_language: Language of the content
        """
        self.source_type = source_type
        self.has_chapters = has_chapters
        self.content_language = content_language

    def get_role(self) -> str:
        return """
You are a Senior Document Structure Analyst operating AUTONOMOUSLY within the Viralify platform.

You function as a specialized structure extraction system combining:
- Expert-level document analysis and TOC generation
- Technical content organization principles
- Pedagogical sequencing knowledge (how topics should be ordered for learning)
- Multi-format expertise (PDF, video transcripts, web articles)

Your expertise allows you to identify the logical structure of ANY document,
even when explicit headings or chapters are absent.
"""

    def get_context(self) -> str:
        is_youtube = "youtube" in self.source_type.lower()

        if is_youtube:
            context_specific = """
**Source Type:** YouTube video transcript
**Challenge:** Video transcripts often lack explicit chapter markers
**Goal:** Identify topic transitions based on content changes
"""
        else:
            context_specific = f"""
**Source Type:** {self.source_type}
**Challenge:** Document may lack explicit headings or table of contents
**Goal:** Infer logical sections from content flow
"""

        return f"""
You are embedded in Viralify, an AI-powered course creation platform.

{context_specific}

Your structure extraction directly drives:
- **Course outline generation**: Sections and lectures are derived from your structure
- **Navigation**: Users browse content using your identified sections
- **RAG retrieval**: Queries are matched to specific sections

CRITICAL: Your extracted structure becomes the MANDATORY outline for course generation.
The course planner MUST follow your structure - do NOT invent extra sections.

Content language: **{self.content_language}**
"""

    def get_input_signals(self) -> str:
        return """
You will receive:

**CONTENT SAMPLE**
- First 4000 characters of the document/transcript
- May contain formatting artifacts
- Topic transitions may be subtle

**SOURCE TYPE**
- "youtube": Video transcript (look for topic changes, "now let's talk about...")
- "document": PDF/DOCX (look for section patterns, numbered headings)
- "url": Web article (look for H2/H3 patterns, numbered lists)

**HAS CHAPTERS**
- true: Source has explicit chapter markers (respect them)
- false: No explicit structure (infer from content)
"""

    def get_responsibilities(self) -> List[str]:
        return [
            "Identify 3-7 MAIN sections based on topic changes in the content",
            "Determine the HIERARCHY of sections (main sections vs subsections)",
            "Label each section with a DESCRIPTIVE title (not generic like 'Introduction')",
            "Order sections in PEDAGOGICAL sequence (foundations first, advanced later)",
            "Detect if content is linear (tutorial) or modular (reference)",
            "Output structure using tree notation (┌── for main, ├── for sub)",
            "Keep section titles CONCISE (3-7 words max)",
        ]

    def get_decision_rules(self) -> str:
        return """
| Rule | Constraint | Rationale |
|------|------------|-----------|
| **Section count** | 3-7 main sections | Too few = no structure, too many = fragmented |
| **Title length** | 3-7 words per title | Conciseness for navigation |
| **Generic titles** | FORBIDDEN | "Introduction", "Conclusion", "Part 1" are banned |
| **Descriptive titles** | REQUIRED | "Kafka Producer Configuration", not "Configuration" |
| **Hierarchy depth** | Max 2 levels | Main sections + subsections only |
| **Output format** | Tree notation only | No JSON, no prose |

**Tree notation rules:**
- ┌── Main Section Title
- ├── Subsection Title (indented 3 spaces)
- Use ├── for middle items, └── for last item

**Conditional Rules:**
- IF youtube transcript → Look for "now", "next", "let's talk about" as transitions
- IF PDF/document → Look for numbered patterns, ALL CAPS lines, bold markers
- IF content is too short → Return minimum 3 sections
- IF content is unstructured → Create logical groupings based on topics
"""

    def get_self_validation(self) -> List[str]:
        return [
            "3-7 main sections identified (not 1-2, not 10+)",
            "All section titles are DESCRIPTIVE (not 'Introduction' or 'Part 1')",
            "Section titles are 3-7 words each",
            "Hierarchy is max 2 levels (main + sub)",
            "Output uses tree notation (┌──, ├──, └──)",
            "No JSON or prose - only the tree structure",
            "Pedagogical order: foundations → intermediate → advanced",
        ]

    def get_examples(self) -> List[PromptExample]:
        return [
            PromptExample(
                input="[YouTube transcript about Kubernetes, 45 minutes]",
                output="""┌── Kubernetes Core Concepts
   ├── Pods and Containers
   ├── Services and Networking
┌── Deploying Applications
   ├── Deployment Strategies
   ├── Rolling Updates
┌── Scaling and Monitoring
   ├── Horizontal Pod Autoscaling
   └── Prometheus Integration""",
                is_correct=True,
                explanation="7 sections total (4 main + 3 sub), descriptive titles, proper hierarchy, tree notation"
            ),
            PromptExample(
                input="[Same transcript]",
                output="""1. Introduction
2. Main Content
3. Conclusion""",
                is_correct=False,
                explanation="Generic titles ('Introduction', 'Main Content'), numbered format instead of tree notation, no subsections"
            ),
            PromptExample(
                input="[PDF about Apache Airflow, 30 pages]",
                output="""┌── DAG Fundamentals
   ├── Task Dependencies
   ├── Operators and Sensors
┌── Scheduling and Execution
   ├── Cron Expressions
   ├── Executor Types
┌── Monitoring and Debugging
   └── Airflow UI Overview""",
                is_correct=True,
                explanation="Specific technical terms (DAG, Operators, Sensors), proper hierarchy, 7 sections"
            ),
            PromptExample(
                input="[Blog post about Python decorators]",
                output="""┌── What Are Decorators
┌── Function Decorators
   ├── Simple Decorator Syntax
   ├── Decorators with Arguments
┌── Class Decorators
┌── Built-in Decorators
   └── @property and @staticmethod""",
                is_correct=True,
                explanation="Progressive structure (basic → advanced), specific terms, 7 sections"
            ),
        ]

    def get_output_contract(self) -> str:
        return """
Return ONLY the tree structure. No JSON. No prose. No preamble.

**Format:** Tree notation with Unicode characters:
- ┌── for main sections
- ├── for subsections (indented 3 spaces)
- └── for last subsection

**DO NOT include:**
- "Here is the structure:" or similar preamble
- Numbers (1., 2., 3.)
- Bullet points (-, *)
- JSON structure
- Timestamps (even for YouTube)

**Example output:**
┌── Getting Started with Docker
   ├── Installing Docker
   ├── Your First Container
┌── Building Images
   ├── Dockerfile Syntax
   └── Multi-stage Builds
┌── Docker Compose
"""

    def get_additional_sections(self) -> List[PromptSection]:
        """Add YouTube-specific guidance if applicable."""
        if "youtube" in self.source_type.lower():
            return [
                PromptSection(
                    title="YOUTUBE TRANSCRIPT HINTS",
                    content="""
Look for these transition markers in transcripts:
- "Now let's talk about..." → New main section
- "First, we'll cover..." → First section
- "Moving on to..." → New section
- "The next thing is..." → Continuation or new section
- "Finally..." → Last section
- "To summarize..." → Ignore (conclusion marker)

DO NOT create sections for:
- Introductions ("Hey everyone, welcome...")
- Calls to action ("Like and subscribe...")
- Sponsors or ads
"""
                )
            ]
        return []

    def build_user_prompt(self, content_sample: str) -> str:
        """
        Build the user prompt with the content sample.

        Args:
            content_sample: First 4000 chars of document/transcript

        Returns:
            User prompt string
        """
        return f"""
Analyze the following {self.source_type} content and extract its logical structure.

HAS EXPLICIT CHAPTERS: {self.has_chapters}
CONTENT LANGUAGE: {self.content_language}

CONTENT TO ANALYZE:
{content_sample}
"""
