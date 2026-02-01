"""
NEXUS Multi-Agent Orchestrator
Orchestration d'agents spécialisés avec feedback loop

Innovation: 5 agents spécialisés qui collaborent avec validation croisée
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import time

from models.data_models import (
    ArchitectureDNA, CognitiveBlueprint, CognitiveStep,
    CodeSegment, ExecutionResult, NexusRequest,
    ComponentType, TargetAudience, CodeVerbosity
)


logger = logging.getLogger(__name__)


# =============================================================================
# AGENT ROLES
# =============================================================================

class AgentRole(Enum):
    """Rôles des agents spécialisés"""
    ARCHITECT = "architect"       # Décisions de structure
    CODER = "coder"              # Génération du code
    REVIEWER = "reviewer"        # Review qualité et pédagogie
    EXECUTOR = "executor"        # Exécution et validation
    NARRATOR = "narrator"        # Synchronisation narration


# =============================================================================
# AGENT MESSAGE PROTOCOL
# =============================================================================

@dataclass
class AgentMessage:
    """Message entre agents"""
    from_agent: AgentRole
    to_agent: AgentRole
    message_type: str  # "request", "response", "feedback", "validation"
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "from": self.from_agent.value,
            "to": self.to_agent.value,
            "type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentContext:
    """Contexte partagé entre agents"""
    request: NexusRequest
    dna: ArchitectureDNA
    blueprint: CognitiveBlueprint
    generated_segments: List[CodeSegment] = field(default_factory=list)
    execution_results: List[ExecutionResult] = field(default_factory=list)
    conversation_history: List[AgentMessage] = field(default_factory=list)
    
    # Métriques de collaboration
    iteration_count: int = 0
    total_feedback_loops: int = 0
    
    def add_message(self, msg: AgentMessage):
        self.conversation_history.append(msg)


# =============================================================================
# BASE AGENT
# =============================================================================

class BaseAgent(ABC):
    """Agent de base avec capacités communes"""
    
    def __init__(self, llm_provider, role: AgentRole):
        self.llm = llm_provider
        self.role = role
        self.name = f"{role.value.capitalize()}Agent"
    
    @abstractmethod
    def process(self, context: AgentContext, step: CognitiveStep) -> Dict[str, Any]:
        """Traite une étape cognitive et retourne son résultat"""
        pass
    
    def send_message(self, context: AgentContext, to: AgentRole, 
                     msg_type: str, content: Dict) -> AgentMessage:
        """Envoie un message à un autre agent"""
        msg = AgentMessage(
            from_agent=self.role,
            to_agent=to,
            message_type=msg_type,
            content=content,
        )
        context.add_message(msg)
        return msg
    
    def _get_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """Helper pour appel LLM"""
        from providers.llm_provider import LLMMessage
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        response = self.llm.generate(messages)
        return response.content
    
    def _get_llm_json(self, system_prompt: str, user_prompt: str) -> Dict:
        """Helper pour appel LLM avec réponse JSON"""
        from providers.llm_provider import LLMMessage
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        
        return self.llm.generate_json(messages)


# =============================================================================
# ARCHITECT AGENT
# =============================================================================

class ArchitectAgent(BaseAgent):
    """
    Agent Architecte: Décisions de structure et patterns.
    
    Responsabilités:
    - Décider de la structure du code pour chaque étape
    - Choisir les patterns à appliquer
    - Définir les interfaces et contrats
    """
    
    SYSTEM_PROMPT = """You are a senior software architect. Your role is to make structural decisions.

For each code component, you decide:
1. The exact structure and organization
2. Which design patterns to apply
3. The interfaces and contracts
4. How it fits with existing components

Be precise and practical. Your decisions guide the Coder agent."""

    def __init__(self, llm_provider):
        super().__init__(llm_provider, AgentRole.ARCHITECT)
    
    def process(self, context: AgentContext, step: CognitiveStep) -> Dict[str, Any]:
        """Produit les décisions architecturales pour une étape"""
        
        # Ne traiter que les étapes de design et implémentation
        if step.code_component == "none":
            return {"skip": True, "reason": "No code component needed"}
        
        prompt = f"""Make architectural decisions for this component:

Component Type: {step.code_component}
Thought: {step.thought}
Decision: {step.decision}

Project Context:
- Language: {context.dna.language}
- Framework: {context.dna.framework}
- Patterns: {[p.value for p in context.dna.patterns]}
- Existing entities: {[e.name for e in context.dna.entities]}

Already generated components:
{[s.filename for s in context.generated_segments]}

Provide JSON:
{{
  "filename": "path/to/file.py",
  "structure": {{
    "imports": ["module1", "module2"],
    "classes": ["ClassName"],
    "functions": ["func1", "func2"],
    "patterns_applied": ["pattern1"]
  }},
  "interfaces": ["def method(arg: Type) -> ReturnType"],
  "dependencies": ["other_component"],
  "notes_for_coder": "Implementation hints"
}}"""

        result = self._get_llm_json(self.SYSTEM_PROMPT, prompt)
        
        # Envoyer au Coder
        self.send_message(
            context, 
            AgentRole.CODER, 
            "request",
            {"architecture": result, "step": step.to_dict()}
        )
        
        return result


# =============================================================================
# CODER AGENT
# =============================================================================

class CoderAgent(BaseAgent):
    """
    Agent Codeur: Génération du code source.
    
    Responsabilités:
    - Implémenter le code selon les décisions de l'Architecte
    - Adapter le style au public cible
    - Ajouter les commentaires appropriés
    """
    
    VERBOSITY_STYLES = {
        CodeVerbosity.MINIMAL: "Write minimal, clean code. Few comments, only essential.",
        CodeVerbosity.STANDARD: "Write clean code with key comments explaining important parts.",
        CodeVerbosity.VERBOSE: "Write highly commented code. Explain every significant line for learning.",
        CodeVerbosity.PRODUCTION: "Write production-ready code with error handling, logging, and docstrings.",
    }
    
    AUDIENCE_STYLES = {
        TargetAudience.DEVELOPER: "Write practical, production-ready code a developer would use.",
        TargetAudience.ARCHITECT: "Focus on structure and patterns. Code can be more schematic.",
        TargetAudience.STUDENT: "Write educational code. Explain concepts. Avoid advanced shortcuts.",
        TargetAudience.TECHNICAL_LEAD: "Balance practical implementation with clear architecture.",
    }
    
    def __init__(self, llm_provider):
        super().__init__(llm_provider, AgentRole.CODER)
    
    def process(self, context: AgentContext, step: CognitiveStep,
                architecture: Dict = None) -> Dict[str, Any]:
        """Génère le code pour une étape"""
        
        if not architecture or architecture.get("skip"):
            return {"skip": True}
        
        verbosity_style = self.VERBOSITY_STYLES.get(
            context.request.verbosity, 
            self.VERBOSITY_STYLES[CodeVerbosity.STANDARD]
        )
        
        audience_style = self.AUDIENCE_STYLES.get(
            context.request.target_audience,
            self.AUDIENCE_STYLES[TargetAudience.STUDENT]
        )
        
        system_prompt = f"""You are an expert {context.dna.language} developer and educator.

Style Guidelines:
- {verbosity_style}
- {audience_style}

Framework: {context.dna.framework}

Generate clean, working code that can be executed."""

        prompt = f"""Generate code based on this architecture:

Architecture Decision:
{architecture}

Cognitive Step:
- Thought: {step.thought}
- Reasoning: {step.reasoning}
- Decision: {step.decision}

Requirements:
1. Follow the structure exactly
2. Make it educational and clear
3. Include appropriate comments
4. Ensure it's syntactically correct

Output JSON:
{{
  "code": "the complete code as a string",
  "explanation": "What this code does (for narration)",
  "key_concepts": ["concept1", "concept2"],
  "common_mistakes": ["mistake1 to avoid"],
  "expected_output": "What running this should show"
}}"""

        result = self._get_llm_json(system_prompt, prompt)
        
        # Créer le CodeSegment
        segment = CodeSegment(
            id="",
            cognitive_step_id=step.id,
            code=result.get("code", ""),
            language=context.dna.language,
            filename=architecture.get("filename", "code.py"),
            component_type=self._map_component_type(step.code_component),
            explanation=result.get("explanation", ""),
            key_concepts=result.get("key_concepts", []),
            common_mistakes=result.get("common_mistakes", []) if context.request.show_mistakes else [],
            expected_output=result.get("expected_output", ""),
        )
        
        # Envoyer au Reviewer
        self.send_message(
            context,
            AgentRole.REVIEWER,
            "request",
            {"segment": segment.to_dict(), "architecture": architecture}
        )
        
        return {"segment": segment, "raw_result": result}
    
    def _map_component_type(self, component_str: str) -> ComponentType:
        """Mappe une string vers ComponentType"""
        mapping = {
            "model": ComponentType.MODEL,
            "models": ComponentType.MODEL,
            "repository": ComponentType.REPOSITORY,
            "service": ComponentType.SERVICE,
            "controller": ComponentType.CONTROLLER,
            "api": ComponentType.API_ENDPOINT,
            "api_endpoint": ComponentType.API_ENDPOINT,
            "routes": ComponentType.API_ENDPOINT,
            "middleware": ComponentType.MIDDLEWARE,
            "utility": ComponentType.UTILITY,
            "utils": ComponentType.UTILITY,
            "config": ComponentType.CONFIG,
            "test": ComponentType.TEST,
            "database": ComponentType.DATABASE,
        }
        return mapping.get(component_str.lower(), ComponentType.UTILITY)


# =============================================================================
# REVIEWER AGENT
# =============================================================================

class ReviewerAgent(BaseAgent):
    """
    Agent Reviewer: Validation qualité et pédagogique.
    
    Responsabilités:
    - Vérifier la qualité du code
    - Valider l'aspect pédagogique
    - Suggérer des améliorations
    - Décider si le code passe ou doit être régénéré
    """
    
    SYSTEM_PROMPT = """You are a senior code reviewer and education specialist.

Your job is to review code for:
1. CORRECTNESS: Will it run without errors?
2. QUALITY: Is it clean and well-structured?
3. PEDAGOGY: Is it educational and clear for the target audience?
4. CONSISTENCY: Does it match the architectural decisions?

Be constructive but strict. Quality matters."""

    def __init__(self, llm_provider):
        super().__init__(llm_provider, AgentRole.REVIEWER)
    
    def process(self, context: AgentContext, segment: CodeSegment,
                architecture: Dict = None) -> Dict[str, Any]:
        """Review un segment de code"""
        
        prompt = f"""Review this code segment:

```{segment.language}
{segment.code}
```

Context:
- Target audience: {context.request.target_audience.value}
- Skill level: {context.request.skill_level}
- Expected patterns: {[p.value for p in context.dna.patterns]}
- Framework: {context.dna.framework}

Architecture Expected:
{architecture}

Rate each aspect 1-10 and provide feedback:

Output JSON:
{{
  "correctness_score": 8,
  "quality_score": 7,
  "pedagogy_score": 9,
  "consistency_score": 8,
  "overall_pass": true,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1"],
  "must_fix": ["critical issue that requires regeneration"],
  "feedback_for_coder": "Specific improvement instructions"
}}

Set overall_pass to false if any score < 6 or there are must_fix issues."""

        result = self._get_llm_json(self.SYSTEM_PROMPT, prompt)
        
        # Si échec, demander une régénération
        if not result.get("overall_pass", True):
            self.send_message(
                context,
                AgentRole.CODER,
                "feedback",
                {
                    "regenerate": True,
                    "issues": result.get("must_fix", []),
                    "feedback": result.get("feedback_for_coder", ""),
                }
            )
            context.total_feedback_loops += 1
        else:
            # Envoyer à l'Executor pour validation runtime
            self.send_message(
                context,
                AgentRole.EXECUTOR,
                "request",
                {"segment": segment.to_dict()}
            )
        
        return result


# =============================================================================
# EXECUTOR AGENT
# =============================================================================

class ExecutorAgent(BaseAgent):
    """
    Agent Executor: Exécution et validation du code.
    
    Responsabilités:
    - Exécuter le code dans un sandbox
    - Capturer les outputs
    - Valider que le code fonctionne
    """
    
    def __init__(self, llm_provider, sandbox_enabled: bool = True):
        super().__init__(llm_provider, AgentRole.EXECUTOR)
        self.sandbox_enabled = sandbox_enabled
    
    def process(self, context: AgentContext, segment: CodeSegment) -> ExecutionResult:
        """Exécute un segment de code"""
        
        if not self.sandbox_enabled:
            # Mode dry-run: validation syntaxique uniquement
            return self._validate_syntax(segment)
        
        # Exécution réelle
        return self._execute_in_sandbox(context, segment)
    
    def _validate_syntax(self, segment: CodeSegment) -> ExecutionResult:
        """Validation syntaxique sans exécution"""
        import ast
        
        if segment.language.lower() != "python":
            # Pour les autres langages, on fait confiance au Reviewer
            return ExecutionResult(
                segment_id=segment.id,
                success=True,
                output="[Syntax validation skipped for non-Python]",
            )
        
        try:
            ast.parse(segment.code)
            return ExecutionResult(
                segment_id=segment.id,
                success=True,
                output="[Syntax valid]",
            )
        except SyntaxError as e:
            return ExecutionResult(
                segment_id=segment.id,
                success=False,
                output="",
                error=f"Syntax error: {e}",
            )
    
    def _execute_in_sandbox(self, context: AgentContext, 
                            segment: CodeSegment) -> ExecutionResult:
        """Exécute le code dans un sandbox isolé"""
        import subprocess
        import tempfile
        import os
        
        if segment.language.lower() != "python":
            return self._validate_syntax(segment)
        
        # Créer un fichier temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(segment.code)
            temp_path = f.name
        
        try:
            # Exécuter avec timeout
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=10,  # 10 secondes max
                cwd=tempfile.gettempdir(),
            )
            
            success = result.returncode == 0
            output = result.stdout or "[No output]"
            error = result.stderr if not success else ""
            
            return ExecutionResult(
                segment_id=segment.id,
                success=success,
                output=output,
                error=error,
                execution_time_ms=0,  # TODO: mesurer
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                segment_id=segment.id,
                success=False,
                output="",
                error="Execution timeout (10s)",
            )
        except Exception as e:
            return ExecutionResult(
                segment_id=segment.id,
                success=False,
                output="",
                error=str(e),
            )
        finally:
            # Nettoyer
            try:
                os.unlink(temp_path)
            except:
                pass


# =============================================================================
# NARRATOR AGENT
# =============================================================================

class NarratorAgent(BaseAgent):
    """
    Agent Narrateur: Synchronisation avec la narration.
    
    Responsabilités:
    - Générer le script de narration pour chaque segment
    - Synchroniser le timing
    - Adapter le ton au public
    """
    
    SYSTEM_PROMPT = """You are an expert coding instructor creating narration for video lessons.

Your narration should:
1. Be NATURAL and CONVERSATIONAL - speak like a teacher explaining to students
2. Explain what each part of the code does and WHY
3. Weave key concepts naturally into your explanation (NEVER list them like "Key concepts: X, Y, Z")
4. Mention pitfalls naturally (e.g., "Be careful not to..." instead of "Common mistakes: ...")
5. Flow smoothly with transitions like "Now let's see...", "Notice how...", "This is important because..."

CRITICAL STYLE RULES:
- NEVER use bullet-point style speech or lists in narration
- NEVER say "Key concepts:", "Key metrics:", "Common mistakes:", or similar headers
- NEVER end with a summary list of concepts
- Speak as if you're explaining to a student sitting next to you
- Use contractions (it's, we're, let's) for natural flow
- Integrate ALL concepts naturally into your flowing explanation"""

    def __init__(self, llm_provider):
        super().__init__(llm_provider, AgentRole.NARRATOR)
    
    def process(self, context: AgentContext, segment: CodeSegment,
                previous_segments: List[CodeSegment] = None) -> Dict[str, Any]:
        """Génère la narration pour un segment"""
        
        prev_context = ""
        if previous_segments:
            prev_context = f"Previous topics: {[s.key_concepts for s in previous_segments[-3:]]}"
        
        prompt = f"""Generate natural, conversational narration for this code segment.

Code to explain:
```{segment.language}
{segment.code}
```

What this code does: {segment.explanation}

Concepts to weave into your explanation (integrate naturally, DO NOT list): {segment.key_concepts}

Pitfalls learners should avoid (mention naturally if relevant): {segment.common_mistakes}

{prev_context}

Target audience: {context.request.target_audience.value}
Skill level: {context.request.skill_level}

IMPORTANT: Write as a teacher naturally explaining code. NO lists, NO "Key concepts:", NO mechanical enumeration.
Just flowing, conversational explanation that covers all concepts organically.

Output JSON:
{{
  "narration_script": "Natural flowing narration (NO bullet points or concept lists)",
  "duration_estimate_seconds": 30,
  "emphasis_points": ["word or phrase to emphasize"],
  "pause_after": ["concept after which to pause briefly"],
  "transition_to_next": "Transition phrase to next segment"
}}"""

        result = self._get_llm_json(self.SYSTEM_PROMPT, prompt)
        
        # Mettre à jour le segment
        segment.narration_script = result.get("narration_script", "")
        segment.duration_seconds = result.get("duration_estimate_seconds", 30)
        
        return result


# =============================================================================
# MULTI-AGENT ORCHESTRATOR
# =============================================================================

class MultiAgentOrchestrator:
    """
    Orchestrateur multi-agents avec feedback loop.
    
    Coordonne les 5 agents spécialisés:
    1. Architect → décisions de structure
    2. Coder → génération du code
    3. Reviewer → validation qualité
    4. Executor → exécution et tests
    5. Narrator → synchronisation narration
    
    Le feedback loop permet des itérations jusqu'à qualité satisfaisante.
    """
    
    MAX_ITERATIONS = 3  # Max tentatives par segment
    
    def __init__(self, llm_provider, sandbox_enabled: bool = True):
        self.llm = llm_provider
        
        # Initialiser les agents
        self.architect = ArchitectAgent(llm_provider)
        self.coder = CoderAgent(llm_provider)
        self.reviewer = ReviewerAgent(llm_provider)
        self.executor = ExecutorAgent(llm_provider, sandbox_enabled)
        self.narrator = NarratorAgent(llm_provider)
    
    def orchestrate(self, context: AgentContext) -> List[CodeSegment]:
        """
        Orchestre la génération complète.
        
        Pour chaque étape cognitive du blueprint, coordonne les agents.
        """
        logger.info(f"Starting multi-agent orchestration with {len(context.blueprint.all_steps)} steps")
        
        segments = []
        previous_segments = []
        
        for step in context.blueprint.all_steps:
            # Ignorer les étapes sans code
            if step.code_component == "none":
                continue
            
            segment = self._process_step_with_feedback_loop(
                context, step, previous_segments
            )
            
            if segment:
                segments.append(segment)
                previous_segments.append(segment)
                context.generated_segments.append(segment)
        
        # Assigner les ordres d'affichage
        for i, segment in enumerate(segments):
            segment.display_order = i
        
        logger.info(f"Orchestration complete: {len(segments)} segments, "
                   f"{context.total_feedback_loops} feedback loops")
        
        return segments
    
    def _process_step_with_feedback_loop(self, context: AgentContext,
                                          step: CognitiveStep,
                                          previous_segments: List[CodeSegment]) -> Optional[CodeSegment]:
        """
        Traite une étape avec le feedback loop.
        
        Flow:
        Architect → Coder → Reviewer → (feedback loop si échec) → Executor → Narrator
        """
        logger.debug(f"Processing step: {step.thought[:50]}...")
        
        iteration = 0
        segment = None
        architecture = None
        feedback = None
        
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            context.iteration_count += 1
            
            # 1. Architect (seulement première itération)
            if architecture is None:
                architecture = self.architect.process(context, step)
                if architecture.get("skip"):
                    return None
            
            # 2. Coder (avec feedback si itération > 1)
            coder_result = self.coder.process(context, step, architecture)
            if coder_result.get("skip"):
                return None
            
            segment = coder_result.get("segment")
            
            # Appliquer le feedback de l'itération précédente
            if feedback and segment:
                segment = self._apply_feedback(context, segment, feedback)
            
            # 3. Reviewer
            review_result = self.reviewer.process(context, segment, architecture)
            
            if review_result.get("overall_pass", True):
                # Succès ! Passer à l'exécution
                break
            else:
                # Échec, préparer le feedback pour la prochaine itération
                feedback = {
                    "issues": review_result.get("must_fix", []),
                    "suggestions": review_result.get("suggestions", []),
                    "feedback": review_result.get("feedback_for_coder", ""),
                }
                logger.debug(f"Feedback loop {iteration}: {feedback['issues']}")
        
        if segment:
            # 4. Executor
            exec_result = self.executor.process(context, segment)
            context.execution_results.append(exec_result)
            
            if not exec_result.success:
                logger.warning(f"Execution failed: {exec_result.error}")
                segment.is_executable = False
            
            # 5. Narrator
            narration_result = self.narrator.process(context, segment, previous_segments)
        
        return segment
    
    def _apply_feedback(self, context: AgentContext, segment: CodeSegment,
                        feedback: Dict) -> CodeSegment:
        """Applique le feedback du reviewer pour améliorer le code"""
        
        prompt = f"""Improve this code based on feedback:

Original code:
```{segment.language}
{segment.code}
```

Feedback:
- Issues to fix: {feedback.get('issues', [])}
- Suggestions: {feedback.get('suggestions', [])}
- Specific feedback: {feedback.get('feedback', '')}

Output the improved code only, no explanation:"""

        from providers.llm_provider import LLMMessage
        
        messages = [
            LLMMessage(role="system", content=f"Expert {segment.language} developer. Code only."),
            LLMMessage(role="user", content=prompt),
        ]
        
        response = self.llm.generate(messages)
        improved_code = response.content.strip()
        
        # Nettoyer les backticks si présents
        if improved_code.startswith("```"):
            lines = improved_code.split("\n")
            improved_code = "\n".join(lines[1:-1])
        
        segment.code = improved_code
        return segment
