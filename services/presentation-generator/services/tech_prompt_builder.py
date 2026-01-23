"""
Tech Prompt Builder - Contextual Prompt Generation for IT Education

Builds dynamic, high-quality prompts based on:
- User profile (career, experience level)
- Course context (domain, topic, keywords)
- Technical requirements (languages, tools)
- Quality standards (clean code, testable, professional)

This module ensures all generated content meets enterprise-grade standards.
"""

import os
from typing import Optional, List, Dict, Any
from enum import Enum

from models.tech_domains import (
    TechCareer,
    TechDomain,
    CodeLanguage,
    get_career_display_name,
    get_domain_display_name,
    get_language_display_name,
    CAREER_DOMAIN_MAP,
    DOMAIN_LANGUAGE_MAP,
)


class AudienceLevel(str, Enum):
    """Audience expertise levels"""
    ABSOLUTE_BEGINNER = "absolute_beginner"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TechPromptBuilder:
    """
    Builds contextual prompts for code and diagram generation.

    Adapts tone, complexity, and examples based on:
    - Target audience level
    - Specific tech domain
    - Career path context
    - Required technologies
    """

    def __init__(self):
        # Code quality standards - always included
        self.code_standards = self._build_code_standards()
        self.diagram_standards = self._build_diagram_standards()

        # Domain-specific expertise contexts
        self.domain_contexts = self._build_domain_contexts()

        # Level-appropriate teaching styles
        self.teaching_styles = self._build_teaching_styles()

    def build_code_prompt(
        self,
        topic: str,
        domain: Optional[TechDomain] = None,
        career: Optional[TechCareer] = None,
        audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE,
        languages: Optional[List[CodeLanguage]] = None,
        tools: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        content_language: str = "en"
    ) -> str:
        """
        Build a comprehensive prompt for code generation.

        Args:
            topic: Main topic of the course/presentation
            domain: Tech domain (e.g., DATA_ENGINEERING, DEVOPS)
            career: Target career path
            audience_level: Expertise level of the audience
            languages: Programming languages to use
            tools: Specific tools/technologies to cover
            keywords: Important keywords to include
            content_language: Language for text content

        Returns:
            Complete system prompt for code generation
        """
        sections = []

        # 1. ROLE - Based on career and audience level
        role = self._build_role_section(domain, career, audience_level)
        sections.append(f"# ROLE\n{role}")

        # 2. CONTEXT - Based on topic and domain
        context = self._build_context_section(topic, domain, keywords)
        sections.append(f"# CONTEXT\n{context}")

        # 3. AUDIENCE - Teaching style adaptation
        audience = self._build_audience_section(audience_level)
        sections.append(f"# AUDIENCE\n{audience}")

        # 4. TECHNICAL REQUIREMENTS - Languages and tools
        tech_req = self._build_tech_requirements(languages, tools, domain)
        sections.append(f"# TECHNICAL REQUIREMENTS\n{tech_req}")

        # 5. CODE QUALITY STANDARDS - Always included
        sections.append(f"# CODE QUALITY STANDARDS (MANDATORY)\n{self.code_standards}")

        # 6. EXAMPLES - Language-specific good vs bad code
        examples = self._build_code_examples(languages, audience_level)
        if examples:
            sections.append(f"# CODE EXAMPLES\n{examples}")

        # 7. LANGUAGE - Content language requirements
        lang_req = self._build_language_requirements(content_language)
        sections.append(f"# CONTENT LANGUAGE\n{lang_req}")

        return "\n\n".join(sections)

    def build_diagram_prompt(
        self,
        description: str,
        domain: Optional[TechDomain] = None,
        diagram_type: str = "architecture",
        audience_level: AudienceLevel = AudienceLevel.INTERMEDIATE,
        style: str = "dark",
        content_language: str = "en"
    ) -> str:
        """
        Build a comprehensive prompt for diagram generation.

        Args:
            description: What the diagram should show
            domain: Tech domain for context
            diagram_type: Type of diagram
            audience_level: Expertise level (affects complexity)
            style: Visual style
            content_language: Language for labels

        Returns:
            Complete system prompt for diagram generation
        """
        sections = []

        # 1. ROLE
        role = f"""You are a Senior Solutions Architect and Technical Illustrator with expertise in:
- Enterprise architecture patterns
- System design and diagramming
- Visual communication of complex technical concepts
- {get_domain_display_name(domain) if domain else 'Multi-domain IT systems'}"""
        sections.append(f"# ROLE\n{role}")

        # 2. CONTEXT
        context = f"""Creating a {diagram_type} diagram for: {description}

Domain focus: {get_domain_display_name(domain) if domain else 'General IT'}
Audience level: {audience_level.value.replace('_', ' ').title()}"""
        sections.append(f"# CONTEXT\n{context}")

        # 3. DIAGRAM STANDARDS
        sections.append(f"# DIAGRAM QUALITY STANDARDS (MANDATORY)\n{self.diagram_standards}")

        # 4. COMPLEXITY GUIDELINES
        complexity = self._get_diagram_complexity(audience_level)
        sections.append(f"# COMPLEXITY GUIDELINES\n{complexity}")

        # 5. LANGUAGE
        lang_req = f"""All labels and text must be in {self._get_language_name(content_language)}.
Technical terms that are universally understood (API, REST, HTTP, etc.) can remain in English."""
        sections.append(f"# LABEL LANGUAGE\n{lang_req}")

        return "\n\n".join(sections)

    def _build_role_section(
        self,
        domain: Optional[TechDomain],
        career: Optional[TechCareer],
        audience_level: AudienceLevel
    ) -> str:
        """Build the role section of the prompt."""

        # Determine expertise level for the teacher
        teacher_level = self._get_teacher_level(audience_level)

        # Build domain expertise
        domain_expertise = ""
        if domain:
            domain_expertise = f"with deep expertise in {get_domain_display_name(domain)}"

        # Build career context
        career_context = ""
        if career:
            career_context = f"\nYour content targets {get_career_display_name(career)} professionals."

        # Teaching persona based on audience
        teaching_persona = self._get_teaching_persona(audience_level)

        return f"""You are a {teacher_level} Software Engineer and Technical Educator {domain_expertise}.

{teaching_persona}
{career_context}

Your code must be:
- Production-ready and enterprise-grade
- Following industry best practices
- Clear enough for the target audience to understand
- Well-documented with meaningful comments"""

    def _build_context_section(
        self,
        topic: str,
        domain: Optional[TechDomain],
        keywords: Optional[List[str]]
    ) -> str:
        """Build the context section."""

        context = f"Topic: {topic}\n"

        if domain:
            context += f"Domain: {get_domain_display_name(domain)}\n"

            # Add domain-specific context
            if domain in self.domain_contexts:
                context += f"\nDomain-specific considerations:\n{self.domain_contexts[domain]}"

        if keywords:
            context += f"\nKey concepts to cover: {', '.join(keywords)}"

        return context

    def _build_audience_section(self, audience_level: AudienceLevel) -> str:
        """Build audience-specific guidelines."""
        return self.teaching_styles.get(audience_level, self.teaching_styles[AudienceLevel.INTERMEDIATE])

    def _build_tech_requirements(
        self,
        languages: Optional[List[CodeLanguage]],
        tools: Optional[List[str]],
        domain: Optional[TechDomain]
    ) -> str:
        """Build technical requirements section."""

        requirements = []

        if languages:
            lang_names = [get_language_display_name(lang) for lang in languages]
            requirements.append(f"Programming languages: {', '.join(lang_names)}")
        elif domain and domain in DOMAIN_LANGUAGE_MAP:
            # Use domain default languages
            default_langs = DOMAIN_LANGUAGE_MAP[domain][:3]  # Top 3
            lang_names = [get_language_display_name(lang) for lang in default_langs]
            requirements.append(f"Recommended languages for this domain: {', '.join(lang_names)}")

        if tools:
            requirements.append(f"Tools/Technologies to cover: {', '.join(tools)}")

        if not requirements:
            return "Use appropriate languages and tools for the topic."

        return "\n".join(requirements)

    def _build_code_standards(self) -> str:
        """Build mandatory code quality standards."""
        return """ALL code must adhere to these standards:

1. NAMING CONVENTIONS:
   - Variables: descriptive, no single letters (except loop indices)
   - Functions: verb phrases (get_user, calculate_total, validate_input)
   - Classes: noun phrases in PascalCase
   - Constants: SCREAMING_SNAKE_CASE

2. STRUCTURE:
   - Functions: max 20 lines, single responsibility
   - Classes: max 200 lines, cohesive
   - Files: max 400 lines
   - Nesting: max 3 levels deep

3. TESTABILITY:
   - Pure functions where possible
   - Dependency injection for external services
   - No global state
   - Clear input/output contracts

4. DOCUMENTATION:
   - Docstrings for all public functions/methods
   - Include: purpose, args, returns, examples
   - Inline comments only for non-obvious logic

5. ERROR HANDLING:
   - Specific exceptions (not bare except)
   - Meaningful error messages
   - Fail fast, recover gracefully
   - Log errors appropriately

6. TYPE SAFETY:
   - Full type hints (Python 3.10+ style)
   - Generic types where appropriate
   - Optional types for nullable values

7. PATTERNS:
   - Apply appropriate design patterns
   - Factory for object creation
   - Strategy for interchangeable algorithms
   - Repository for data access

8. ANTI-PATTERNS TO AVOID:
   - Magic numbers/strings (use constants)
   - Deep nesting (use early returns)
   - God classes/functions
   - Copy-paste code (DRY principle)
   - Premature optimization

BAD CODE EXAMPLE (NEVER DO THIS):
```python
def p(d, x):
    r = []
    for i in d:
        if i > x:
            r.append(i*2)
    return r
```

GOOD CODE EXAMPLE (ALWAYS DO THIS):
```python
from typing import List

def filter_and_double_above_threshold(
    numbers: List[float],
    threshold: float
) -> List[float]:
    \"\"\"
    Filter numbers above threshold and double them.

    Args:
        numbers: List of numbers to process
        threshold: Minimum value to include (exclusive)

    Returns:
        List of doubled values for numbers above threshold

    Example:
        >>> filter_and_double_above_threshold([1, 5, 10], 4)
        [10, 20]
    \"\"\"
    return [num * 2 for num in numbers if num > threshold]
```"""

    def _build_diagram_standards(self) -> str:
        """Build mandatory diagram quality standards."""
        return """ALL diagrams must adhere to these standards:

1. PROFESSIONAL APPEARANCE:
   - Clean, enterprise-grade look
   - Consistent spacing and alignment
   - Professional color palette
   - High resolution output

2. READABILITY:
   - Maximum 12-15 nodes for clarity
   - Clear, concise labels (2-4 words)
   - Logical grouping with clusters
   - Left-to-right or top-to-bottom flow

3. CLARITY:
   - Each element serves a purpose
   - No decorative-only components
   - Clear connection meanings
   - Edge labels for non-obvious flows

4. CONSISTENCY:
   - Uniform icon style (same provider)
   - Consistent naming convention
   - Aligned elements where logical
   - Balanced visual weight

5. COMPLETENESS:
   - All connections shown
   - No orphan nodes
   - Entry and exit points clear
   - Legend if using custom symbols

BAD DIAGRAM (NEVER):
- Too many nodes (>15)
- Cryptic labels (A, B, C)
- Crossing lines everywhere
- Mixed icon styles
- No logical grouping

GOOD DIAGRAM (ALWAYS):
- Focused scope (5-12 nodes)
- Descriptive labels ("User API", "Order DB")
- Clean flow with minimal crossings
- Consistent provider icons
- Logical clusters (Frontend, Backend, Data)"""

    def _build_domain_contexts(self) -> Dict[TechDomain, str]:
        """Build domain-specific context snippets."""
        return {
            TechDomain.DATA_ENGINEERING: """
- Focus on data pipelines, ETL/ELT patterns
- Consider scalability and data quality
- Include error handling for data issues
- Show idempotent operations where relevant
- Consider batch vs streaming trade-offs""",

            TechDomain.MACHINE_LEARNING: """
- Include data preprocessing considerations
- Show model training/inference separation
- Consider reproducibility (random seeds, versioning)
- Include evaluation metrics
- Handle edge cases in predictions""",

            TechDomain.DEVOPS: """
- Focus on automation and repeatability
- Include error handling and rollback
- Consider security best practices
- Show logging and monitoring hooks
- Infrastructure as Code patterns""",

            TechDomain.CLOUD_AWS: """
- Use AWS SDK best practices
- Include IAM considerations
- Handle AWS-specific errors
- Consider cost optimization
- Show proper resource cleanup""",

            TechDomain.KUBERNETES: """
- Follow K8s manifest best practices
- Include resource limits
- Show health checks (liveness/readiness)
- Consider security contexts
- Include proper labels and annotations""",

            TechDomain.CYBERSECURITY: """
- Never show actual credentials or secrets
- Include input validation
- Demonstrate secure coding patterns
- Consider authentication/authorization
- Show proper error handling without info leakage""",

            TechDomain.BLOCKCHAIN: """
- Include gas optimization considerations
- Show security patterns (reentrancy guards)
- Consider upgrade patterns
- Include event emissions
- Show proper access control""",

            TechDomain.QUANTUM_COMPUTING: """
- Explain quantum concepts clearly
- Show circuit diagrams when relevant
- Include measurement considerations
- Explain superposition/entanglement
- Show classical-quantum interface""",
        }

    def _build_teaching_styles(self) -> Dict[AudienceLevel, str]:
        """Build audience-appropriate teaching styles."""
        return {
            AudienceLevel.ABSOLUTE_BEGINNER: """
AUDIENCE: Absolute beginners with no programming experience

Teaching approach:
- Explain EVERY concept, assume nothing
- Use real-world analogies extensively
- Break down into smallest possible steps
- Repeat key concepts multiple times
- Use extremely simple examples first
- Avoid jargon or define all terms
- Celebrate small victories
- Include "why" for everything""",

            AudienceLevel.BEGINNER: """
AUDIENCE: Beginners with basic programming knowledge

Teaching approach:
- Explain new concepts thoroughly
- Build on basic programming knowledge
- Use practical, relatable examples
- Introduce terminology gradually
- Show common mistakes and how to avoid them
- Provide step-by-step instructions
- Connect to concepts they likely know""",

            AudienceLevel.INTERMEDIATE: """
AUDIENCE: Intermediate developers with solid fundamentals

Teaching approach:
- Assume familiarity with basics
- Focus on best practices and patterns
- Explain the "why" behind decisions
- Show trade-offs between approaches
- Include real-world scenarios
- Mention edge cases and gotchas
- Reference related advanced topics""",

            AudienceLevel.ADVANCED: """
AUDIENCE: Advanced developers with significant experience

Teaching approach:
- Assume strong technical foundation
- Focus on optimization and edge cases
- Discuss architectural trade-offs
- Include performance considerations
- Show advanced patterns and techniques
- Reference industry standards
- Discuss scalability implications""",

            AudienceLevel.EXPERT: """
AUDIENCE: Expert developers and architects

Teaching approach:
- Peer-to-peer technical discussion
- Focus on cutting-edge techniques
- Deep dive into internals
- Discuss research and emerging patterns
- Challenge conventional approaches
- Include performance benchmarks
- Reference academic/industry papers""",
        }

    def _build_code_examples(
        self,
        languages: Optional[List[CodeLanguage]],
        audience_level: AudienceLevel
    ) -> str:
        """Build language-specific code examples."""

        if not languages:
            return ""

        examples = []

        for lang in languages[:2]:  # Max 2 examples
            example = self._get_language_example(lang, audience_level)
            if example:
                examples.append(example)

        return "\n\n".join(examples)

    def _get_language_example(
        self,
        lang: CodeLanguage,
        level: AudienceLevel
    ) -> Optional[str]:
        """Get a language-specific example."""

        # Python example
        if lang in [CodeLanguage.PYTHON, CodeLanguage.QISKIT]:
            return '''PYTHON CODE STYLE:
```python
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Represents a task in the system."""
    id: str
    title: str
    status: Status = Status.PENDING

    def complete(self) -> None:
        """Mark task as completed."""
        self.status = Status.COMPLETED

def get_pending_tasks(tasks: List[Task]) -> List[Task]:
    """
    Filter tasks to return only pending ones.

    Args:
        tasks: List of tasks to filter

    Returns:
        List of tasks with PENDING status
    """
    return [task for task in tasks if task.status == Status.PENDING]
```'''

        # TypeScript example
        if lang in [CodeLanguage.TYPESCRIPT, CodeLanguage.JAVASCRIPT]:
            return '''TYPESCRIPT CODE STYLE:
```typescript
interface User {
  id: string;
  email: string;
  createdAt: Date;
}

interface UserRepository {
  findById(id: string): Promise<User | null>;
  save(user: User): Promise<void>;
}

class UserService {
  constructor(private readonly repository: UserRepository) {}

  async getUserOrThrow(id: string): Promise<User> {
    const user = await this.repository.findById(id);

    if (!user) {
      throw new NotFoundError(`User ${id} not found`);
    }

    return user;
  }
}
```'''

        # Go example
        if lang == CodeLanguage.GO:
            return '''GO CODE STYLE:
```go
package user

import (
    "context"
    "errors"
)

// ErrUserNotFound is returned when a user cannot be found.
var ErrUserNotFound = errors.New("user not found")

// User represents a user in the system.
type User struct {
    ID    string
    Email string
}

// Repository defines the interface for user storage.
type Repository interface {
    FindByID(ctx context.Context, id string) (*User, error)
}

// Service handles user business logic.
type Service struct {
    repo Repository
}

// NewService creates a new user service.
func NewService(repo Repository) *Service {
    return &Service{repo: repo}
}

// GetUser retrieves a user by ID.
func (s *Service) GetUser(ctx context.Context, id string) (*User, error) {
    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        return nil, fmt.Errorf("failed to get user: %w", err)
    }
    return user, nil
}
```'''

        return None

    def _get_teacher_level(self, audience_level: AudienceLevel) -> str:
        """Get appropriate teacher seniority based on audience."""
        mapping = {
            AudienceLevel.ABSOLUTE_BEGINNER: "Patient Senior",
            AudienceLevel.BEGINNER: "Experienced Senior",
            AudienceLevel.INTERMEDIATE: "Staff",
            AudienceLevel.ADVANCED: "Principal",
            AudienceLevel.EXPERT: "Distinguished Fellow",
        }
        return mapping.get(audience_level, "Senior")

    def _get_teaching_persona(self, audience_level: AudienceLevel) -> str:
        """Get teaching persona based on audience level."""
        personas = {
            AudienceLevel.ABSOLUTE_BEGINNER:
                "You are patient, encouraging, and explain concepts as if teaching someone their very first program. "
                "Use analogies to everyday life. Celebrate progress.",

            AudienceLevel.BEGINNER:
                "You are supportive and thorough, building confidence while teaching proper foundations. "
                "Connect new concepts to what beginners likely already know.",

            AudienceLevel.INTERMEDIATE:
                "You are professional and practical, focusing on real-world applications and best practices. "
                "You challenge learners to think about edge cases and trade-offs.",

            AudienceLevel.ADVANCED:
                "You are technically rigorous, discussing advanced patterns and optimizations. "
                "You treat learners as capable developers ready for complex challenges.",

            AudienceLevel.EXPERT:
                "You engage as a technical peer, discussing cutting-edge techniques and research. "
                "You challenge assumptions and explore the boundaries of current best practices.",
        }
        return personas.get(audience_level, personas[AudienceLevel.INTERMEDIATE])

    def _get_diagram_complexity(self, audience_level: AudienceLevel) -> str:
        """Get diagram complexity guidelines based on audience."""
        guidelines = {
            AudienceLevel.ABSOLUTE_BEGINNER: """
- Maximum 5-7 components
- One concept per diagram
- Very simple flows
- Extensive labels
- No advanced patterns""",

            AudienceLevel.BEGINNER: """
- Maximum 8-10 components
- Simple groupings allowed
- Clear, linear flows
- Helpful annotations
- Basic patterns only""",

            AudienceLevel.INTERMEDIATE: """
- Maximum 10-12 components
- Logical clustering
- Multiple flows acceptable
- Technical labels
- Common patterns shown""",

            AudienceLevel.ADVANCED: """
- Maximum 12-15 components
- Complex groupings
- Multiple interaction patterns
- Technical detail
- Advanced patterns welcome""",

            AudienceLevel.EXPERT: """
- Complexity as needed (still readable)
- Full architectural detail
- All relevant components
- Expert-level patterns
- Cross-cutting concerns shown""",
        }
        return guidelines.get(audience_level, guidelines[AudienceLevel.INTERMEDIATE])

    def _build_language_requirements(self, content_language: str) -> str:
        """Build content language requirements."""
        lang_name = self._get_language_name(content_language)

        return f"""All text content must be in {lang_name}:
- Titles and headings
- Explanations and narration
- Comments in code (educational comments)
- Diagram labels (except universal tech terms)

Code syntax and keywords remain in their native language.
Variable names should be in English for code readability.
Technical terms universally known (API, REST, HTTP, JSON) can stay in English."""

    def _get_language_name(self, code: str) -> str:
        """Get full language name from code."""
        names = {
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
        return names.get(code.lower(), code)


# Singleton instance for easy import
prompt_builder = TechPromptBuilder()
