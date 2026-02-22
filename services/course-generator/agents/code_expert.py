"""
Code Expert Agent

Generates production-quality, executable code for course lectures.
Uses the enriched prompt from TechnicalReviewerAgent to ensure code
matches the learner profile and course requirements.
"""
import json
from typing import Dict, Any, Optional

from agents.base import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentResult,
    CourseGenerationState,
    CodeBlockState,
)


# Default system prompt if code_expert_prompt not available
DEFAULT_CODE_EXPERT_PROMPT = """### ROLE
You are "Olsisoft Senior Architect", a software engineering expert with 20 years of experience.
Your goal is to produce "Production-Ready" quality code for a video training course.

### GOLDEN RULES
1. NO PSEUDO-CODE: Complete code with ALL imports at the top.
2. NO PLACEHOLDERS: No "TODO", "...", "pass #", "implement here", "your code here".
3. REAL COMPLEXITY: Code MUST match the learner's level (see LEVEL REQUIREMENTS below).
4. EXECUTABILITY: Code MUST run immediately without modifications. Test mentally: does this run?
5. PEDAGOGY: Explicit variable names, docstrings explaining "why" not just "what".
6. SELF-CONTAINED: No external dependencies that aren't imported. No references to undefined variables.

### LEVEL REQUIREMENTS
Adapt complexity to the learner's level:
| Level | Lines | Functions | Classes | Error handling | Patterns |
|-------|-------|-----------|---------|----------------|----------|
| beginner | 10-25 | 0-1 | 0 | Basic try/except | Simple, linear flow |
| intermediate | 20-50 | 2-3 | 0-1 | Specific exceptions | Functions, data structures |
| advanced | 40-80 | 3-5 | 1-2 | Custom exceptions | Design patterns, decorators |
| expert | 60-120 | 5+ | 2+ | Full error chain | Architecture, async, typing |

### COMMON MISTAKES TO AVOID (CRITICAL)
- DO NOT write "Hello World" programs for intermediate+ levels
- DO NOT use single-letter variable names (except i, j, k in loops)
- DO NOT write code that only prints strings — demonstrate real logic
- DO NOT hardcode values that should be parameters
- DO NOT import modules you don't use
- DO NOT write functions that do nothing meaningful
- For BEGINNER level: still include real logic, not just print statements

### EXAMPLE OF BAD CODE (NEVER DO THIS)
```python
# BAD: Too simple, no real logic, hardcoded values
def greet(name):
    print(f"Hello {name}")
greet("Alice")
```

### EXAMPLE OF GOOD CODE (DO THIS)
```python
# GOOD: Real logic, error handling, clear naming, educational
from dataclasses import dataclass
from typing import List

@dataclass
class Student:
    \"\"\"Represents a student with grades for GPA calculation.\"\"\"
    name: str
    grades: List[float]

    @property
    def gpa(self) -> float:
        \"\"\"Calculate the Grade Point Average.\"\"\"
        if not self.grades:
            return 0.0
        return round(sum(self.grades) / len(self.grades), 2)

# Create students and demonstrate GPA calculation
students = [
    Student("Alice", [3.8, 3.5, 4.0]),
    Student("Bob", [2.9, 3.1, 3.3]),
]
for student in students:
    print(f"{student.name}: GPA = {student.gpa}")
```

### OUTPUT FORMAT (JSON)
{
  "code_block": "the complete, executable code",
  "explanation": "step-by-step explanation for the voiceover (explain each section of the code)",
  "execution_command": "command to execute (e.g., python example.py)",
  "expected_output": "exact terminal output when executed",
  "complexity_score": 1-10,
  "patterns_used": ["patterns used"]
}
"""


class CodeExpertAgent(BaseAgent):
    """
    Generates production-quality code for course slides.

    This agent:
    1. Analyzes the concept to be demonstrated
    2. Considers the learner's level (persona)
    3. Generates complete, executable code
    4. Provides explanation for voiceover
    5. Specifies expected output for validation
    """

    # Code generation needs the quality model for accurate, complex code
    MODEL_TIER = "quality"

    def __init__(self):
        super().__init__(AgentType.CODE_EXPERT)

    async def process(self, state: CourseGenerationState) -> CourseGenerationState:
        """
        Process all code blocks that need generation.

        This method processes the current_code_block in the state.
        For batch processing, the orchestrator should call this
        multiple times with different code blocks.

        Args:
            state: Current course generation state

        Returns:
            State updated with generated code
        """
        code_block = state.get("current_code_block")

        if not code_block:
            self.log("No code block to process")
            return state

        self.log(f"Generating code for concept: {code_block.get('concept', 'unknown')}")

        # Get the enriched prompt from TechnicalReviewer
        system_prompt = state.get("code_expert_prompt", DEFAULT_CODE_EXPERT_PROMPT)

        # Generate code
        result = await self._generate_code(
            concept=code_block.get("concept", ""),
            language=code_block.get("language", "python"),
            persona_level=code_block.get("persona_level", state.get("persona_level", "intermediate")),
            rag_context=state.get("rag_context"),
            system_prompt=system_prompt,
        )

        if result.success:
            # Update code block with generated code
            code_block["refined_code"] = result.data.get("code_block", "")
            code_block["expected_output"] = result.data.get("expected_output", "")
            code_block["complexity_score"] = result.data.get("complexity_score", 5)
            code_block["review_status"] = "pending"  # Ready for CodeReviewer
            code_block["patterns_used"] = result.data.get("patterns_used", [])

            state["current_code_block"] = code_block
            state["code_blocks_processed"] = state.get("code_blocks_processed", 0) + 1

            self.log(f"Code generated successfully. Complexity: {code_block['complexity_score']}/10")
        else:
            code_block["review_status"] = "failed"
            code_block["rejection_reasons"] = result.errors
            state["errors"] = state.get("errors", []) + result.errors

            self.log(f"Code generation failed: {result.errors}")

        # Add to agent history
        self.add_to_history(
            state,
            AgentStatus.COMPLETED if result.success else AgentStatus.FAILED,
            result
        )

        return state

    async def _generate_code(
        self,
        concept: str,
        language: str,
        persona_level: str,
        rag_context: Optional[str],
        system_prompt: str,
    ) -> AgentResult:
        """
        Generate production-quality code using GPT-4.

        Args:
            concept: The concept to demonstrate
            language: Programming language
            persona_level: Learner's level
            rag_context: Optional context from documents
            system_prompt: Enriched system prompt

        Returns:
            AgentResult with generated code
        """
        # Build user prompt
        user_prompt = self._build_user_prompt(
            concept, language, persona_level, rag_context
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent code
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            # Validate the response has required fields
            required_fields = ["code_block", "explanation"]
            missing = [f for f in required_fields if not result.get(f)]

            if missing:
                return AgentResult(
                    success=False,
                    errors=[f"Response missing required fields: {missing}"]
                )

            # Check for pseudo-code indicators
            code = result.get("code_block", "")
            pseudo_indicators = ["TODO", "...", "implement here", "your code here", "pass  #"]

            for indicator in pseudo_indicators:
                if indicator.lower() in code.lower():
                    return AgentResult(
                        success=False,
                        errors=[f"Code contains pseudo-code indicator: '{indicator}'"],
                        retry_needed=True,
                        retry_prompt=f"The code contains '{indicator}' which is not allowed. "
                                    "Generate complete, executable code without placeholders."
                    )

            # Check minimum code length for non-trivial code
            if len(code.strip()) < 50:
                return AgentResult(
                    success=False,
                    errors=["Code is too short to be production-quality"],
                    retry_needed=True,
                    retry_prompt="The generated code is too short. "
                                "Provide a more complete implementation with proper structure."
                )

            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "model": self.model,
                    "concept": concept,
                    "language": language,
                }
            )

        except json.JSONDecodeError as e:
            return AgentResult(
                success=False,
                errors=[f"Failed to parse JSON response: {str(e)}"],
                retry_needed=True
            )
        except Exception as e:
            return AgentResult(
                success=False,
                errors=[f"Code generation error: {str(e)}"]
            )

    # Level-specific generation instructions
    LEVEL_INSTRUCTIONS = {
        "beginner": (
            "The learner is a BEGINNER. Write simple, clear code that:\n"
            "- Uses basic constructs (variables, loops, conditionals, functions)\n"
            "- Has extensive comments explaining each step\n"
            "- Avoids advanced patterns (decorators, comprehensions, OOP)\n"
            "- Still demonstrates REAL logic (not just print statements)\n"
            "- Target: 10-25 lines of meaningful code"
        ),
        "intermediate": (
            "The learner is INTERMEDIATE. Write production-style code that:\n"
            "- Uses functions, data structures, and error handling\n"
            "- Demonstrates the concept with real-world use cases\n"
            "- Includes type hints and docstrings\n"
            "- Shows proper code organization\n"
            "- Target: 20-50 lines with 2-3 functions"
        ),
        "advanced": (
            "The learner is ADVANCED. Write professional code that:\n"
            "- Uses design patterns, classes, and decorators\n"
            "- Includes custom exceptions and proper error chains\n"
            "- Demonstrates architectural thinking\n"
            "- Shows testing or validation patterns\n"
            "- Target: 40-80 lines with classes and multiple functions"
        ),
        "expert": (
            "The learner is an EXPERT. Write enterprise-grade code that:\n"
            "- Uses advanced patterns (async, metaclasses, protocols, generics)\n"
            "- Demonstrates system design principles\n"
            "- Includes comprehensive typing and documentation\n"
            "- Shows performance considerations\n"
            "- Target: 60-120 lines with full architecture"
        ),
    }

    def _build_user_prompt(
        self,
        concept: str,
        language: str,
        persona_level: str,
        rag_context: Optional[str]
    ) -> str:
        """Build the user prompt for code generation"""
        level_instructions = self.LEVEL_INSTRUCTIONS.get(
            persona_level.lower(),
            self.LEVEL_INSTRUCTIONS["intermediate"]
        )

        prompt_parts = [
            f"### CONCEPT TO DEMONSTRATE",
            f"{concept}",
            "",
            f"### PROGRAMMING LANGUAGE",
            f"{language}",
            "",
            f"### LEARNER LEVEL: {persona_level.upper()}",
            level_instructions,
            "",
        ]

        if rag_context:
            prompt_parts.extend([
                "### DOCUMENT CONTEXT (RAG)",
                f"{rag_context[:2000]}",  # Limit RAG context
                "",
            ])

        prompt_parts.extend([
            "### INSTRUCTIONS",
            f"1. Generate a complete, executable {language} code block that demonstrates '{concept}'.",
            f"2. The code MUST match the {persona_level} level described above.",
            "3. Include appropriate error handling (try/except with specific exception types).",
            "4. Provide the EXACT expected terminal output when the code is executed.",
            "5. The code must be SELF-CONTAINED: all imports, all definitions, runnable as-is.",
            "6. Variable names must be descriptive (no single letters except loop counters).",
            "7. Include comments explaining the 'why', not just the 'what'.",
            "",
            "Respond in JSON according to the specified format.",
        ])

        return "\n".join(prompt_parts)

    async def generate_for_slide(
        self,
        slide_type: str,
        slide_title: str,
        concept: str,
        language: str,
        persona_level: str,
        rag_context: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AgentResult:
        """
        Convenience method to generate code for a specific slide.

        Args:
            slide_type: Type of slide (code, code_demo)
            slide_title: Title of the slide
            concept: Concept to demonstrate
            language: Programming language
            persona_level: Learner level
            rag_context: Optional RAG context
            system_prompt: Optional custom system prompt

        Returns:
            AgentResult with generated code
        """
        self.log(f"Generating code for slide: {slide_title} ({slide_type})")

        # Adjust prompt based on slide type
        if slide_type == "code_demo":
            # For demos, emphasize execution output
            concept = f"{concept}\n\nIMPORTANT: This is a DEMO slide. The code will be executed and the output shown. Make sure the output is meaningful and demonstrates the concept clearly."

        return await self._generate_code(
            concept=concept,
            language=language,
            persona_level=persona_level,
            rag_context=rag_context,
            system_prompt=system_prompt or DEFAULT_CODE_EXPERT_PROMPT,
        )

    async def refine_code(
        self,
        original_code: str,
        feedback: str,
        language: str,
        persona_level: str,
    ) -> AgentResult:
        """
        Refine code based on reviewer feedback.

        Args:
            original_code: The code to refine
            feedback: Feedback from CodeReviewer
            language: Programming language
            persona_level: Learner level

        Returns:
            AgentResult with refined code
        """
        self.log(f"Refining code based on feedback")

        refinement_prompt = f"""### ROLE
You are an expert code refactorer. You must improve the following code
based on the reviewer's feedback.

### ORIGINAL CODE
```{language}
{original_code}
```

### REVIEWER FEEDBACK
{feedback}

### INSTRUCTIONS
1. Fix ALL issues mentioned in the feedback.
2. Keep the same pedagogical objective.
3. Improve quality without changing the core logic.
4. Ensure the code is still executable.

### OUTPUT FORMAT (JSON)
{{
  "code_block": "the improved code",
  "explanation": "what was improved",
  "execution_command": "command to execute",
  "expected_output": "expected output",
  "complexity_score": 1-10,
  "improvements_made": ["list of improvements"]
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = json.loads(content)

            return AgentResult(
                success=True,
                data=result,
                metadata={"refined": True}
            )

        except Exception as e:
            return AgentResult(
                success=False,
                errors=[f"Code refinement error: {str(e)}"]
            )


def create_code_expert() -> CodeExpertAgent:
    """Factory function to create a CodeExpertAgent"""
    return CodeExpertAgent()
