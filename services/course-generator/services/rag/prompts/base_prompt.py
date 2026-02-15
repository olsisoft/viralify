"""
Base Prompt Builder

Abstract base class for building well-structured prompts following
the Viralify prompt engineering pattern. All prompts MUST include:

1. ROLE - Who the AI is acting as (expertise + autonomy)
2. CONTEXT - Viralify platform context
3. INPUT SIGNALS - What data will be received
4. RESPONSIBILITIES - Explicit actions to take
5. DECISION RULES - Hard constraints (MUST, not should)
6. SELF-VALIDATION - Checklist before output
7. EXAMPLES - Few-shot with correct and incorrect examples
8. OUTPUT CONTRACT - Exact format specification
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PromptSection:
    """A section of a structured prompt."""
    title: str
    content: str


@dataclass
class PromptExample:
    """
    An example for few-shot learning in prompts.

    Attributes:
        input: The example input/scenario
        output: The expected output
        is_correct: True for correct examples (✅), False for incorrect (❌)
        explanation: Why this example is correct/incorrect
    """
    input: str
    output: str
    is_correct: bool
    explanation: str


class BasePromptBuilder(ABC):
    """
    Abstract base class for building well-structured prompts.

    Implements the Viralify prompt engineering pattern with:
    - Primacy effect: Critical instructions first (ROLE)
    - Recency effect: Verification checklist last (OUTPUT CONTRACT)
    - Visual separators for easy scanning
    - Hard constraints (MUST) not soft suggestions (should)

    Usage:
        class MyPromptBuilder(BasePromptBuilder):
            def get_role(self) -> str:
                return "You are a Senior..."
            # ... implement all abstract methods

        prompt = MyPromptBuilder().build()
    """

    # Visual separators for prompt sections
    HEADER_BOX = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                    {title:^42}                                ║
╚══════════════════════════════════════════════════════════════════════════════╝"""

    SECTION_BOX = """
┌──────────────────────────────────────────────────────────────────────────────┐
│                                  {title:^42}                                  │
└──────────────────────────────────────────────────────────────────────────────┘"""

    SEPARATOR = "─" * 80

    @abstractmethod
    def get_role(self) -> str:
        """
        Define the AI's role and expertise.

        Should include:
        - Title: "You are a Senior [EXPERTISE] Agent"
        - Autonomy: "operating AUTONOMOUSLY"
        - Combined skills: "combining [SKILL_1], [SKILL_2], [SKILL_3]"

        Example:
            return '''
            You are a Senior Content Analysis Agent operating AUTONOMOUSLY.
            You function as a specialized document intelligence system combining:
            - Expert-level reading comprehension
            - Technical domain knowledge
            - Multi-language proficiency
            '''
        """
        pass

    @abstractmethod
    def get_context(self) -> str:
        """
        Define the platform context.

        Should include:
        - Platform: "Embedded in Viralify, an AI-powered course creation platform"
        - Output usage: "Your output drives [DOWNSTREAM_SYSTEM]"
        - Target audience: "Content is consumed by [TARGET_LEARNERS]"

        Example:
            return '''
            You are embedded in Viralify, similar to Udemy/Coursera.
            Your summaries directly drive:
            - Course structure generation
            - RAG retrieval scoring
            Target learners: Professional software engineers.
            '''
        """
        pass

    @abstractmethod
    def get_input_signals(self) -> str:
        """
        Define the inputs the model will receive.

        Should be structured with types:
            return '''
            You will receive:
            - **Document content**: Raw text (str)
            - **Document type**: pdf, docx, youtube (enum)
            - **Target language**: en, fr, es (ISO 639-1)
            '''
        """
        pass

    @abstractmethod
    def get_responsibilities(self) -> List[str]:
        """
        Define explicit actions the model must take.

        Should be action verbs (imperative):
        - "Identify the PRIMARY subject"
        - "Extract 3-5 KEY concepts"
        - "Generate a 2-3 sentence summary"

        Returns:
            List of responsibility strings
        """
        pass

    @abstractmethod
    def get_decision_rules(self) -> str:
        """
        Define hard constraints (MUST, not should).

        Best format is a table:
            return '''
            | Rule | Constraint | Rationale |
            |------|------------|-----------|
            | Length | 2-3 sentences MAXIMUM | UI display |
            | Language | MUST match target_language | User preference |
            '''

        Use conditional rules where appropriate:
        - "IF requires_code=true → code_weight ≥ 0.6"
        """
        pass

    @abstractmethod
    def get_self_validation(self) -> List[str]:
        """
        Define checklist items to verify before output.

        Each item should be checkable:
        - "Summary is 2-3 sentences (not more)"
        - "No speculation or invented content"

        Returns:
            List of validation check strings
        """
        pass

    @abstractmethod
    def get_examples(self) -> List[PromptExample]:
        """
        Provide few-shot examples (correct AND incorrect).

        Returns:
            List of PromptExample objects

        Best practice: Include at least one ✅ and one ❌ example
        """
        pass

    @abstractmethod
    def get_output_contract(self) -> str:
        """
        Define the exact output format.

        Should specify:
        - Format: JSON, plain text, markdown
        - Schema: Exact fields and types
        - Example: Concrete output example

        Example:
            return '''
            Return ONLY the summary text, no JSON, no preamble.

            Example output:
            "This course teaches you to build data pipelines..."
            '''
        """
        pass

    def get_additional_sections(self) -> List[PromptSection]:
        """
        Override to add custom sections between standard ones.

        Returns:
            List of PromptSection objects to insert
        """
        return []

    def build(self) -> str:
        """
        Assemble all sections into the final prompt.

        Order (optimized for LLM attention):
        1. ROLE (primacy - first thing processed)
        2. CONTEXT
        3. INPUT SIGNALS
        4. RESPONSIBILITIES
        5. DECISION RULES
        6. EXAMPLES
        7. [Additional sections]
        8. SELF-VALIDATION
        9. OUTPUT CONTRACT (recency - last thing remembered)

        Returns:
            Complete prompt string
        """
        sections = []

        # 1. ROLE (primacy effect - critical instructions first)
        sections.append(self.HEADER_BOX.format(title="ROLE"))
        sections.append(self.get_role())
        sections.append("")

        # 2. CONTEXT
        sections.append(self.SECTION_BOX.format(title="CONTEXT"))
        sections.append(self.get_context())
        sections.append("")

        # 3. INPUT SIGNALS
        sections.append(f"### INPUT SIGNALS")
        sections.append(self.get_input_signals())
        sections.append("")

        # 4. RESPONSIBILITIES
        sections.append(f"### RESPONSIBILITIES")
        responsibilities = self.get_responsibilities()
        for i, resp in enumerate(responsibilities, 1):
            sections.append(f"{i}. {resp}")
        sections.append("")

        # 5. DECISION RULES
        sections.append(f"### DECISION RULES (HARD CONSTRAINTS - NOT SUGGESTIONS)")
        sections.append(self.get_decision_rules())
        sections.append("")

        # 6. EXAMPLES
        sections.append(f"### EXAMPLES")
        examples = self.get_examples()
        for ex in examples:
            icon = "✅" if ex.is_correct else "❌"
            status = "CORRECT" if ex.is_correct else "INCORRECT"
            sections.append(f"""
{icon} **{status}**
**Input:** {ex.input}
**Output:** {ex.output}
**Why:** {ex.explanation}
""")
        sections.append("")

        # 7. Additional sections (optional)
        for section in self.get_additional_sections():
            sections.append(f"### {section.title.upper()}")
            sections.append(section.content)
            sections.append("")

        # 8. SELF-VALIDATION
        sections.append(f"### SELF-VALIDATION (verify before output)")
        checks = self.get_self_validation()
        for check in checks:
            sections.append(f"- [ ] {check}")
        sections.append("")

        # 9. OUTPUT CONTRACT (recency effect - last thing processed)
        sections.append(self.SEPARATOR)
        sections.append(f"### OUTPUT CONTRACT")
        sections.append(self.get_output_contract())

        return "\n".join(sections)

    def build_user_prompt(self, **kwargs) -> str:
        """
        Build the user prompt with provided data.

        Override this method to create the user message that
        accompanies the system prompt.

        Args:
            **kwargs: Data to include in the user prompt

        Returns:
            User prompt string
        """
        return str(kwargs)


class SimplePromptBuilder(BasePromptBuilder):
    """
    A simple prompt builder that can be configured without subclassing.

    Usage:
        builder = SimplePromptBuilder(
            role="You are a document summarizer...",
            context="Embedded in Viralify...",
            input_signals="You will receive...",
            responsibilities=["Extract key points", "Summarize"],
            decision_rules="| Rule | Constraint |...",
            self_validation=["Check 1", "Check 2"],
            examples=[PromptExample(...)],
            output_contract="Return JSON..."
        )
        prompt = builder.build()
    """

    def __init__(
        self,
        role: str,
        context: str,
        input_signals: str,
        responsibilities: List[str],
        decision_rules: str,
        self_validation: List[str],
        examples: List[PromptExample],
        output_contract: str,
        additional_sections: Optional[List[PromptSection]] = None,
    ):
        self._role = role
        self._context = context
        self._input_signals = input_signals
        self._responsibilities = responsibilities
        self._decision_rules = decision_rules
        self._self_validation = self_validation
        self._examples = examples
        self._output_contract = output_contract
        self._additional_sections = additional_sections or []

    def get_role(self) -> str:
        return self._role

    def get_context(self) -> str:
        return self._context

    def get_input_signals(self) -> str:
        return self._input_signals

    def get_responsibilities(self) -> List[str]:
        return self._responsibilities

    def get_decision_rules(self) -> str:
        return self._decision_rules

    def get_self_validation(self) -> List[str]:
        return self._self_validation

    def get_examples(self) -> List[PromptExample]:
        return self._examples

    def get_output_contract(self) -> str:
        return self._output_contract

    def get_additional_sections(self) -> List[PromptSection]:
        return self._additional_sections
