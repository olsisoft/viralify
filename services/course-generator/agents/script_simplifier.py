"""
Script Simplifier Agent

Responsible for simplifying lecture scripts when media generation fails.
This agent is part of the recovery loop in the Production subgraph.

Recovery strategies:
1. simplify_script: Reduce text complexity, shorten sentences
2. reduce_animations: Remove typing animations, simplify code demos
3. shorten_content: Cut content length while preserving key points
"""
import os
import json
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from agents.base import AgentType, BaseAgent, AgentResult
from agents.state import ProductionState, RecoveryStrategy


class ScriptSimplifierAgent(BaseAgent):
    """
    Agent for simplifying lecture scripts when media generation fails.

    Uses GPT-4 to:
    - Shorten voiceover scripts
    - Reduce code block complexity
    - Remove or simplify animations
    - Preserve core learning objectives
    """

    def __init__(self):
        super().__init__(AgentType.SCRIPT_WRITER)
        self.name = "script_simplifier"

    async def process(self, state: ProductionState) -> ProductionState:
        """
        Process the state and simplify the script based on recovery strategy.

        Args:
            state: Current production state

        Returns:
            Updated state with simplified content
        """
        strategy = state.get("recovery_strategy")
        lecture_plan = state.get("lecture_plan", {})
        last_error = state.get("last_media_error", "Unknown error")

        self.log(f"Simplifying lecture '{lecture_plan.get('title', 'Unknown')}' "
                 f"with strategy: {strategy}")
        self.log(f"Last error: {last_error}")

        if strategy == RecoveryStrategy.SIMPLIFY_SCRIPT:
            state = await self._simplify_script(state, last_error)
        elif strategy == RecoveryStrategy.REDUCE_ANIMATIONS:
            state = await self._reduce_animations(state, last_error)
        else:
            # Default: just increment version
            state["script_version"] = state.get("script_version", 1) + 1

        state["simplification_applied"] = True
        state["recovery_attempts"] = state.get("recovery_attempts", 0) + 1

        return state

    async def _simplify_script(
        self,
        state: ProductionState,
        error_context: str
    ) -> ProductionState:
        """
        Simplify the voiceover script.

        Strategies:
        - Shorten long sentences
        - Remove complex vocabulary
        - Reduce total length by ~30%
        - Keep key learning points
        """
        lecture_plan = state.get("lecture_plan", {})
        current_script = state.get("voiceover_script") or lecture_plan.get("description", "")
        objectives = lecture_plan.get("objectives", [])
        current_complexity = state.get("script_complexity_score", 5)

        if not current_script:
            self.log("No script to simplify")
            return state

        prompt = f"""You are a script editor. The following voiceover script caused a media generation error.
Simplify it while preserving the key learning objectives.

ERROR CONTEXT: {error_context}

CURRENT SCRIPT:
{current_script}

LEARNING OBJECTIVES TO PRESERVE:
{json.dumps(objectives, indent=2)}

SIMPLIFICATION RULES:
1. Reduce total length by approximately 30%
2. Use shorter, simpler sentences
3. Remove repetitive explanations
4. Keep all key learning points from the objectives
5. Use simpler vocabulary
6. Remove any complex technical jargon unless essential

Return ONLY the simplified script, no explanations."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a script simplification assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            simplified_script = response.choices[0].message.content.strip()

            # Update state
            state["voiceover_script"] = simplified_script
            state["script_version"] = state.get("script_version", 1) + 1
            state["script_complexity_score"] = max(1, current_complexity - 2)

            # Also update the lecture plan
            lecture_plan["voiceover_script"] = simplified_script
            state["lecture_plan"] = lecture_plan

            original_len = len(current_script)
            new_len = len(simplified_script)
            reduction = ((original_len - new_len) / original_len) * 100

            self.log(f"Script simplified: {original_len} -> {new_len} chars "
                     f"({reduction:.1f}% reduction)")

        except Exception as e:
            self.log(f"Script simplification failed: {e}")
            state["errors"] = state.get("errors", []) + [f"Script simplification failed: {str(e)}"]

        return state

    async def _reduce_animations(
        self,
        state: ProductionState,
        error_context: str
    ) -> ProductionState:
        """
        Reduce or remove animations from the lecture.

        Strategies:
        - Remove typing animations
        - Use static code blocks instead of animated
        - Simplify diagram animations
        """
        lecture_plan = state.get("lecture_plan", {})

        self.log("Reducing animations")

        # Set flag to disable animations in media generation
        state["animations_disabled"] = True

        # Simplify code blocks to static versions
        code_blocks = state.get("generated_code_blocks", [])
        for block in code_blocks:
            # Mark code blocks for static rendering
            block["use_static_rendering"] = True

        state["generated_code_blocks"] = code_blocks
        state["script_version"] = state.get("script_version", 1) + 1

        # Also simplify the script slightly
        current_script = state.get("voiceover_script") or lecture_plan.get("description", "")
        if current_script and "timeout" in error_context.lower():
            # If timeout, also reduce script length
            state = await self._quick_shorten(state, current_script)

        self.log("Animations reduced/disabled")

        return state

    async def _quick_shorten(
        self,
        state: ProductionState,
        script: str
    ) -> ProductionState:
        """
        Quick shortening of script for timeout errors.
        Reduces by ~20% without full LLM call.
        """
        # Simple heuristic: remove parenthetical content and shorten sentences
        import re

        # Remove parenthetical content
        shortened = re.sub(r'\([^)]+\)', '', script)

        # Remove extra whitespace
        shortened = re.sub(r'\s+', ' ', shortened).strip()

        # If still too long, truncate at sentence boundaries
        target_len = int(len(script) * 0.8)
        if len(shortened) > target_len:
            sentences = shortened.split('. ')
            result = []
            current_len = 0
            for sentence in sentences:
                if current_len + len(sentence) < target_len:
                    result.append(sentence)
                    current_len += len(sentence) + 2
                else:
                    break
            shortened = '. '.join(result)
            if not shortened.endswith('.'):
                shortened += '.'

        state["voiceover_script"] = shortened

        lecture_plan = state.get("lecture_plan", {})
        lecture_plan["voiceover_script"] = shortened
        state["lecture_plan"] = lecture_plan

        self.log(f"Quick shortened: {len(script)} -> {len(shortened)} chars")

        return state

    async def simplify_code_block(
        self,
        code: str,
        language: str,
        error_context: str
    ) -> str:
        """
        Simplify a code block that's causing issues.

        Args:
            code: Original code
            language: Programming language
            error_context: What error occurred

        Returns:
            Simplified code
        """
        prompt = f"""Simplify this {language} code. It caused an error during processing.

ERROR: {error_context}

ORIGINAL CODE:
```{language}
{code}
```

SIMPLIFICATION RULES:
1. Remove any complex patterns
2. Use simpler, more direct logic
3. Reduce nesting depth
4. Keep functionality but simplify implementation
5. Ensure the code still demonstrates the concept

Return ONLY the simplified code, no explanations."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code simplification assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000,
            )

            result = response.choices[0].message.content.strip()

            # Extract code from markdown if present
            if f"```{language}" in result:
                start = result.find(f"```{language}") + len(f"```{language}")
                end = result.find("```", start)
                result = result[start:end].strip()
            elif "```" in result:
                start = result.find("```") + 3
                end = result.find("```", start)
                result = result[start:end].strip()

            return result

        except Exception as e:
            self.log(f"Code simplification failed: {e}")
            return code  # Return original on failure


# =============================================================================
# FACTORY
# =============================================================================

def get_script_simplifier() -> ScriptSimplifierAgent:
    """Get a ScriptSimplifierAgent instance"""
    return ScriptSimplifierAgent()
