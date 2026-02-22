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
1. NO PSEUDO-CODE: Complete code with imports.
2. NO "TODO": No placeholders.
3. REAL COMPLEXITY: Code adapted to the learner's level.
4. EXECUTABILITY: Code must run immediately.
5. PEDAGOGY: Explicit names, docstrings explaining the "why".

### OUTPUT FORMAT (JSON)
{
  "code_block": "the complete code",
  "explanation": "explanation for the voiceover",
  "execution_command": "command to execute",
  "expected_output": "expected output",
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

    def _build_user_prompt(
        self,
        concept: str,
        language: str,
        persona_level: str,
        rag_context: Optional[str]
    ) -> str:
        """Build the user prompt for code generation"""
        prompt_parts = [
            f"### CONCEPT TO DEMONSTRATE",
            f"{concept}",
            "",
            f"### PROGRAMMING LANGUAGE",
            f"{language}",
            "",
            f"### LEARNER LEVEL",
            f"{persona_level}",
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
            "1. Generate a complete, executable code block.",
            "2. Code must be adapted to the learner's level.",
            "3. Include appropriate error handling.",
            "4. Provide the expected terminal output.",
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
