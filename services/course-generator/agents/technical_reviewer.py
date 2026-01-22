"""
Technical Reviewer Agent

Reviews and enriches prompts to ensure all frontend configuration choices
are properly included before sending to the LLM. This agent acts as a
"gatekeeper" to prevent configuration drift.
"""
from typing import Dict, List, Any

from agents.base import (
    BaseAgent,
    AgentType,
    AgentStatus,
    CourseGenerationState,
)


# Mapping of lesson element keys to their prompt requirements
ELEMENT_PROMPT_REQUIREMENTS = {
    "concept_intro": {
        "description": "Clear concept introduction slide",
        "prompt_addition": "Include a dedicated concept introduction slide that explains the main idea before diving into details.",
        "slide_type": "concept_intro",
    },
    "diagram_schema": {
        "description": "Visual diagrams and schemas",
        "prompt_addition": "Include architecture diagrams, flowcharts, or visual schemas to illustrate complex concepts. Use Mermaid syntax for diagrams.",
        "slide_type": "diagram",
    },
    "code_typing": {
        "description": "Animated code typing demonstrations",
        "prompt_addition": "Include code slides with complete, executable code that will be shown with a typing animation. Code must be production-quality, not pseudo-code.",
        "slide_type": "code",
    },
    "code_execution": {
        "description": "Live code execution with output",
        "prompt_addition": "Include code_demo slides where code is executed in a sandbox and the terminal output is captured and shown. Include expected_output field.",
        "slide_type": "code_demo",
    },
    "voiceover_explanation": {
        "description": "Voiceover narration for slides",
        "prompt_addition": "Every slide MUST have a voiceover_text field with natural, educational narration in the specified content language.",
        "required_field": "voiceover_text",
    },
    "curriculum_slide": {
        "description": "Course progress/curriculum indicator",
        "prompt_addition": "Include curriculum position indicator showing where this lecture fits in the overall course structure.",
        "slide_type": "curriculum",
    },
}

# Quiz frequency to prompt mapping
QUIZ_FREQUENCY_PROMPTS = {
    "per_lecture": "Generate 3-5 quiz questions at the end of EACH lecture to assess understanding.",
    "per_section": "Generate 5-8 quiz questions at the end of EACH SECTION to review all lectures in that section.",
    "end_only": "Generate a comprehensive final quiz of 10-15 questions covering all course material at the end.",
    "custom": "Generate quiz questions at the specified interval throughout the course.",
}

# Profile category specific requirements
CATEGORY_REQUIREMENTS = {
    "tech": {
        "code_style": "Use industry-standard patterns: SOLID principles, clean code, proper error handling, type hints.",
        "examples": "Examples should reflect real-world scenarios: API integrations, database operations, async patterns.",
        "terminology": "Use correct technical terminology. Explain jargon when first introduced.",
    },
    "business": {
        "code_style": "Focus on business logic clarity. Use meaningful variable names that reflect business concepts.",
        "examples": "Use business case studies: CRM systems, inventory management, reporting dashboards.",
        "terminology": "Balance technical accuracy with business-friendly explanations.",
    },
    "creative": {
        "code_style": "Emphasize creative applications: visualizations, generative art, interactive experiences.",
        "examples": "Show creative coding examples: animations, visual effects, interactive media.",
        "terminology": "Make technical concepts accessible to creative professionals.",
    },
    "education": {
        "code_style": "Prioritize readability and step-by-step progression over optimization.",
        "examples": "Use educational examples that build on each other progressively.",
        "terminology": "Define all terms. Assume no prior knowledge unless difficulty indicates otherwise.",
    },
}

# Difficulty level prompt additions
DIFFICULTY_PROMPTS = {
    "beginner": {
        "code_complexity": "Use simple, straightforward code. No advanced patterns. Lots of comments explaining each line.",
        "concept_depth": "Explain concepts from first principles. Use analogies and simple examples.",
        "assumed_knowledge": "Assume no prior knowledge of the topic. Explain everything.",
    },
    "intermediate": {
        "code_complexity": "Use standard patterns and best practices. Include error handling. Moderate comments.",
        "concept_depth": "Build on fundamental knowledge. Introduce some advanced concepts.",
        "assumed_knowledge": "Assume basic familiarity with the language and general programming concepts.",
    },
    "advanced": {
        "code_complexity": "Use advanced patterns: design patterns, optimization, advanced language features.",
        "concept_depth": "Deep dive into internals and edge cases. Discuss trade-offs.",
        "assumed_knowledge": "Assume solid programming experience and familiarity with common patterns.",
    },
    "expert": {
        "code_complexity": "Production-grade code: performance optimized, fully tested, deployment-ready.",
        "concept_depth": "Expert-level analysis: architecture decisions, scalability, system design.",
        "assumed_knowledge": "Assume professional experience. Focus on nuances and advanced techniques.",
    },
}


class TechnicalReviewerAgent(BaseAgent):
    """
    Reviews configuration and enriches prompts to include all frontend choices.

    This agent ensures that:
    1. All enabled lesson elements are reflected in the generation prompts
    2. Quiz configuration is properly communicated to the LLM
    3. Profile category and difficulty affect code style and explanations
    4. Language requirements are clearly specified
    5. No configuration option is "lost" before reaching the LLM
    """

    def __init__(self):
        super().__init__(AgentType.TECHNICAL_REVIEWER)

    async def process(self, state: CourseGenerationState) -> CourseGenerationState:
        """
        Review configuration and generate prompt enrichments.

        Args:
            state: Current course generation state

        Returns:
            State updated with prompt enrichments and review results
        """
        self.log(f"Reviewing configuration for job: {state.get('job_id', 'unknown')}")

        prompt_enrichments: Dict[str, str] = {}
        warnings: List[str] = []
        suggestions: List[str] = []

        # 1. Build lesson elements prompt section
        elements_prompt = self._build_elements_prompt(
            state.get("lesson_elements", {}),
            warnings
        )
        prompt_enrichments["lesson_elements"] = elements_prompt

        # 2. Build quiz configuration prompt section
        quiz_prompt = self._build_quiz_prompt(
            state.get("quiz_config", {}),
            warnings
        )
        prompt_enrichments["quiz"] = quiz_prompt

        # 3. Build category-specific prompt section
        category = state.get("profile_category", "education")
        category_prompt = self._build_category_prompt(category)
        prompt_enrichments["category"] = category_prompt

        # 4. Build difficulty-based prompt section
        diff_start = state.get("difficulty_start", "beginner").lower()
        diff_end = state.get("difficulty_end", "intermediate").lower()
        difficulty_prompt = self._build_difficulty_prompt(diff_start, diff_end)
        prompt_enrichments["difficulty"] = difficulty_prompt

        # 5. Build language requirements prompt section
        content_lang = state.get("content_language", "en")
        prog_lang = state.get("programming_language", "python")
        language_prompt = self._build_language_prompt(content_lang, prog_lang)
        prompt_enrichments["language"] = language_prompt

        # 6. Build structure requirements prompt section
        structure = state.get("structure", {})
        structure_prompt = self._build_structure_prompt(structure)
        prompt_enrichments["structure"] = structure_prompt

        # 7. Build the complete code expert prompt
        code_expert_prompt = self._build_code_expert_prompt(
            state, prompt_enrichments
        )
        state["code_expert_prompt"] = code_expert_prompt

        # 8. Generate suggestions for better configuration
        suggestions = self._generate_suggestions(state)

        # Update state
        state["config_reviewed"] = True
        state["prompt_enrichments"] = prompt_enrichments
        state["config_warnings"] = warnings
        state["config_suggestions"] = suggestions
        state["warnings"] = state.get("warnings", []) + warnings

        # Add to agent history
        self.add_to_history(state, AgentStatus.COMPLETED)

        self.log(f"Configuration reviewed. Generated {len(prompt_enrichments)} prompt sections.")
        if warnings:
            self.log(f"Warnings: {warnings}")

        return state

    def _build_elements_prompt(
        self,
        elements: Dict[str, Any],
        warnings: List[str]
    ) -> str:
        """Build prompt section for lesson elements"""
        lines = ["### REQUIRED LESSON ELEMENTS"]
        lines.append("The following elements MUST be included in the course:")
        lines.append("")

        enabled_elements = []
        for key, enabled in elements.items():
            if enabled and key in ELEMENT_PROMPT_REQUIREMENTS:
                req = ELEMENT_PROMPT_REQUIREMENTS[key]
                enabled_elements.append(f"- **{req['description']}**: {req['prompt_addition']}")

        if enabled_elements:
            lines.extend(enabled_elements)
        else:
            warnings.append("No lesson elements are enabled. Using defaults.")
            lines.append("- Include concept introductions")
            lines.append("- Include voiceover narration")

        return "\n".join(lines)

    def _build_quiz_prompt(
        self,
        quiz_config: Dict[str, Any],
        warnings: List[str]
    ) -> str:
        """Build prompt section for quiz configuration"""
        if not quiz_config.get("enabled", False):
            return "### QUIZ: Disabled for this course."

        frequency = quiz_config.get("frequency", "per_section")
        freq_prompt = QUIZ_FREQUENCY_PROMPTS.get(
            frequency,
            QUIZ_FREQUENCY_PROMPTS["per_section"]
        )

        lines = ["### QUIZ REQUIREMENTS"]
        lines.append(freq_prompt)
        lines.append("")

        # Question types
        q_types = quiz_config.get("question_types", ["mcq", "true_false"])
        if q_types:
            lines.append(f"Question types to include: {', '.join(q_types)}")

        lines.append("")
        lines.append("Quiz format: Each question must have:")
        lines.append("- Clear question text")
        lines.append("- 4 answer options (for MCQ)")
        lines.append("- Correct answer indicated")
        lines.append("- Brief explanation of why the answer is correct")

        return "\n".join(lines)

    def _build_category_prompt(self, category: str) -> str:
        """Build prompt section for profile category"""
        cat_lower = category.lower() if category else "education"
        requirements = CATEGORY_REQUIREMENTS.get(
            cat_lower,
            CATEGORY_REQUIREMENTS["education"]
        )

        lines = [f"### PROFILE CATEGORY: {cat_lower.upper()}"]
        lines.append("")
        for key, value in requirements.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")

        return "\n".join(lines)

    def _build_difficulty_prompt(self, diff_start: str, diff_end: str) -> str:
        """Build prompt section for difficulty progression"""
        start_reqs = DIFFICULTY_PROMPTS.get(diff_start, DIFFICULTY_PROMPTS["beginner"])
        end_reqs = DIFFICULTY_PROMPTS.get(diff_end, DIFFICULTY_PROMPTS["intermediate"])

        lines = [f"### DIFFICULTY PROGRESSION: {diff_start.upper()} → {diff_end.upper()}"]
        lines.append("")
        lines.append("**Starting Level Requirements:**")
        for key, value in start_reqs.items():
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        lines.append("")
        lines.append("**Ending Level Requirements:**")
        for key, value in end_reqs.items():
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")

        lines.append("")
        lines.append("**Progression Strategy:** Start with simpler concepts and code, "
                    "gradually increase complexity throughout the course.")

        return "\n".join(lines)

    def _build_language_prompt(self, content_lang: str, prog_lang: str) -> str:
        """Build prompt section for language requirements"""
        lang_names = {
            "en": "English",
            "fr": "French (Français)",
            "es": "Spanish (Español)",
            "de": "German (Deutsch)",
            "pt": "Portuguese (Português)",
            "it": "Italian (Italiano)",
            "zh": "Chinese (中文)",
        }

        content_lang_name = lang_names.get(content_lang, content_lang)

        lines = ["### LANGUAGE REQUIREMENTS"]
        lines.append("")
        lines.append(f"**Content Language:** {content_lang_name} (code: {content_lang})")
        lines.append(f"**Programming Language:** {prog_lang}")
        lines.append("")
        lines.append("CRITICAL RULES:")
        lines.append(f"- ALL text content (titles, descriptions, voiceover, explanations) MUST be in {content_lang_name}")
        lines.append(f"- Code comments SHOULD be in {content_lang_name} for educational clarity")
        lines.append(f"- Variable/function names follow {prog_lang} conventions (usually English)")
        lines.append(f"- Use proper grammar and spelling for {content_lang_name}")

        if content_lang == "fr":
            lines.append("")
            lines.append("**French-specific:** Use proper accents (é, è, ê, à, ç), "
                        "correct article agreements, proper verb conjugations.")

        return "\n".join(lines)

    def _build_structure_prompt(self, structure: Dict[str, Any]) -> str:
        """Build prompt section for course structure"""
        sections = structure.get("number_of_sections", 4)
        lectures = structure.get("lectures_per_section", 3)
        duration = structure.get("total_duration_minutes", 60)

        total_lectures = sections * lectures
        avg_lecture_min = duration / total_lectures if total_lectures > 0 else 10

        lines = ["### COURSE STRUCTURE"]
        lines.append("")
        lines.append(f"- **Total Duration:** {duration} minutes")
        lines.append(f"- **Sections:** {sections}")
        lines.append(f"- **Lectures per Section:** {lectures}")
        lines.append(f"- **Total Lectures:** {total_lectures}")
        lines.append(f"- **Average Lecture Duration:** {avg_lecture_min:.1f} minutes")
        lines.append("")
        lines.append("Each lecture should be self-contained but build on previous material.")

        return "\n".join(lines)

    def _build_code_expert_prompt(
        self,
        state: CourseGenerationState,
        enrichments: Dict[str, str]
    ) -> str:
        """
        Build the complete prompt for the CodeExpert agent.

        This combines all enrichments into a comprehensive system prompt.
        """
        topic = state.get("topic", "Unknown topic")
        persona = state.get("persona_level", state.get("difficulty_start", "intermediate"))

        prompt = f"""### ROLE
Tu es "Olsisoft Senior Architect", un expert en ingénierie logicielle avec 20 ans d'expérience.
Ton objectif est de produire du code de qualité "Production-Ready" pour un cours vidéo de haut niveau.
Tu ne tolères pas le code trivial, incomplet ou non testé.

### CONTEXTE DU COURS
**Sujet:** {topic}
**Niveau de l'apprenant:** {persona}
**Description:** {state.get('description', 'N/A')}

{enrichments.get('language', '')}

{enrichments.get('difficulty', '')}

{enrichments.get('category', '')}

{enrichments.get('lesson_elements', '')}

{enrichments.get('quiz', '')}

{enrichments.get('structure', '')}

### RÈGLES D'OR DE GÉNÉRATION DE CODE
1. **INTERDICTION DU PSEUDO-CODE:** Tout code généré doit être complet, incluant les imports et une structure cohérente.
2. **PAS DE "TODO":** Aucun commentaire "TODO: implement here" ou "..." à compléter.
3. **COMPLEXITÉ RÉELLE:** Le code doit correspondre au niveau du Persona. Un Senior reçoit des Design Patterns, pas un print("hello").
4. **EXÉCUTABILITÉ:** Le code doit pouvoir tourner immédiatement dans une sandbox Python/Node/etc.
5. **PÉDAGOGIE PAR LE CODE:** Noms de variables explicites, docstrings qui expliquent le "POURQUOI".

### PROCESSUS DE RÉFLEXION
Étape 1: Analyse le concept à démontrer et le niveau de l'apprenant.
Étape 2: Définis une structure de code qui respecte les principes SOLID.
Étape 3: Rédige le code complet avec gestion d'erreurs appropriée.
Étape 4: Prépare l'output attendu du terminal.

### FORMAT DE SORTIE
Tu dois répondre UNIQUEMENT sous ce format JSON:
{{
  "code_block": "le code complet ici",
  "explanation": "explication courte pour le script vocal",
  "execution_command": "la commande pour lancer le code",
  "expected_output": "ce que le terminal doit afficher",
  "complexity_score": 1-10,
  "patterns_used": ["liste des patterns/principes utilisés"]
}}
"""
        return prompt

    def _generate_suggestions(self, state: CourseGenerationState) -> List[str]:
        """Generate suggestions for improving the configuration"""
        suggestions = []

        elements = state.get("lesson_elements", {})
        topic = (state.get("topic") or "").lower()

        # Suggest diagrams for architecture topics
        arch_keywords = ["architecture", "system", "design", "pattern", "microservice"]
        if any(kw in topic for kw in arch_keywords) and not elements.get("diagram_schema"):
            suggestions.append(
                "Consider enabling 'diagram_schema' for architecture-related topics."
            )

        # Suggest code execution for programming topics
        code_keywords = ["programming", "coding", "tutorial", "how to", "build"]
        if any(kw in topic for kw in code_keywords):
            if not elements.get("code_typing"):
                suggestions.append(
                    "Consider enabling 'code_typing' for programming tutorials."
                )
            if not elements.get("code_execution"):
                suggestions.append(
                    "Consider enabling 'code_execution' to show code output."
                )

        # Suggest quizzes for longer courses
        structure = state.get("structure", {})
        if structure.get("total_duration_minutes", 0) > 30:
            quiz_config = state.get("quiz_config", {})
            if not quiz_config.get("enabled"):
                suggestions.append(
                    "Consider enabling quizzes for courses longer than 30 minutes."
                )

        return suggestions


def create_technical_reviewer() -> TechnicalReviewerAgent:
    """Factory function to create a TechnicalReviewerAgent"""
    return TechnicalReviewerAgent()
