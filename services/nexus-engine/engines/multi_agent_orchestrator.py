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

    CONCEPT_LESSON_PROMPT = """You are planning a simple CONCEPT lesson (not a project).

For concept lessons like loops, functions, classes, etc., architecture should be MINIMAL:
- NO frameworks (no Flask, Django, FastAPI)
- NO complex patterns (no MVC, Repository, Service Layer)
- Just simple standalone scripts demonstrating the concept
- Focus on teaching the concept, not building an application

Your decisions guide the Coder to create simple, educational examples."""

    # Concept keywords for detecting concept lessons
    CONCEPT_KEYWORDS = {
        "loop": ["boucle", "loop", "for", "while", "itération", "iterate", "range"],
        "function": ["fonction", "function", "def", "paramètre", "return"],
        "class": ["classe", "class", "objet", "object", "oop", "héritage"],
        "list": ["liste", "list", "array", "tableau", "append"],
        "dictionary": ["dictionnaire", "dictionary", "dict", "clé", "key"],
        "condition": ["condition", "if", "else", "elif", "boolean"],
    }

    def __init__(self, llm_provider):
        super().__init__(llm_provider, AgentRole.ARCHITECT)

    def _is_concept_lesson(self, topic: str) -> bool:
        """Detect if this is a concept lesson"""
        topic_lower = topic.lower()
        for keywords in self.CONCEPT_KEYWORDS.values():
            if any(kw in topic_lower for kw in keywords):
                return True
        return False

    def process(self, context: AgentContext, step: CognitiveStep) -> Dict[str, Any]:
        """Produit les décisions architecturales pour une étape"""

        # Ne traiter que les étapes de design et implémentation
        if step.code_component == "none":
            return {"skip": True, "reason": "No code component needed"}

        topic = context.request.project_description
        is_concept = self._is_concept_lesson(topic)

        if is_concept:
            # For concept lessons, return minimal architecture
            prompt = f"""Plan a SIMPLE example for this concept lesson: {topic}

Component: {step.code_component}
Thought: {step.thought}
Decision: {step.decision}

IMPORTANT: This is a CONCEPT lesson, NOT a project.
- NO frameworks (Flask, Django, etc.)
- NO complex patterns (MVC, Repository, etc.)
- Just a simple standalone script

Provide JSON:
{{
  "filename": "example.py",
  "structure": {{
    "imports": [],
    "classes": [],
    "functions": [],
    "patterns_applied": ["none - simple script"]
  }},
  "interfaces": [],
  "dependencies": [],
  "notes_for_coder": "Keep it simple. Demonstrate {topic} with basic examples and print() statements."
}}"""
            result = self._get_llm_json(self.CONCEPT_LESSON_PROMPT, prompt)
        else:
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
    
    # Concept keywords for detecting concept lessons
    CONCEPT_KEYWORDS = {
        "loop": ["boucle", "loop", "for", "while", "itération", "iterate", "range"],
        "function": ["fonction", "function", "def", "paramètre", "argument", "return"],
        "class": ["classe", "class", "objet", "object", "oop", "héritage", "inheritance"],
        "list": ["liste", "list", "array", "tableau", "append", "index"],
        "dictionary": ["dictionnaire", "dictionary", "dict", "clé", "key", "value"],
        "string": ["chaîne", "string", "caractère", "character", "text"],
        "variable": ["variable", "assignation", "assignment", "type"],
        "condition": ["condition", "if", "else", "elif", "comparaison", "boolean"],
        "exception": ["exception", "try", "except", "error", "erreur"],
        "file": ["fichier", "file", "read", "write", "open", "close"],
    }

    def _detect_concept_lesson(self, topic: str) -> tuple:
        """Detect if topic is a concept lesson and return concept type"""
        topic_lower = topic.lower()
        for concept_type, keywords in self.CONCEPT_KEYWORDS.items():
            if any(kw in topic_lower for kw in keywords):
                return True, concept_type
        return False, None

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

        # Detect if this is a concept lesson
        topic = context.request.project_description
        is_concept, concept_type = self._detect_concept_lesson(topic)

        # Build concept-specific constraints
        concept_constraints = ""
        if is_concept:
            concept_constraints = f"""
CRITICAL CONSTRAINT - CONCEPT LESSON:
This is a lesson teaching "{topic}". Your code MUST:
1. ONLY demonstrate the concept: {concept_type}
2. Use SIMPLE, beginner-friendly examples (no Flask, no APIs, no complex patterns)
3. DO NOT introduce advanced concepts that haven't been explained
4. Keep it focused: if the topic is loops, show loops - not classes or decorators
5. Examples should be self-contained and runnable
6. Use print() to show output so students understand what's happening

BAD example for "for loops": Using Flask, defining classes, complex data structures
GOOD example for "for loops": Simple iteration over lists, range(), basic patterns
"""

        system_prompt = f"""You are an expert {context.dna.language} developer and educator.

LESSON TOPIC: {topic}

Style Guidelines:
- {verbosity_style}
- {audience_style}
{concept_constraints}
Framework: {context.dna.framework if not is_concept else "None (pure Python for concept lessons)"}

Generate clean, working code that can be executed."""

        prompt = f"""Generate code for this lesson: {topic}

Architecture Decision:
{architecture}

Cognitive Step:
- Thought: {step.thought}
- Reasoning: {step.reasoning}
- Decision: {step.decision}

Requirements:
1. The code MUST demonstrate "{topic}" specifically
2. Keep it educational, focused, and simple
3. Include helpful comments
4. Ensure it's syntactically correct and runnable
5. Use print() to show output

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
5. TOPIC FOCUS: Does it demonstrate the lesson topic directly?

Be constructive but strict. Quality matters.
Code that doesn't demonstrate the lesson topic should be REJECTED."""

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
# TOPIC GUARD AGENT - Validates code matches lesson topic
# =============================================================================

class TopicGuardAgent(BaseAgent):
    """
    Agent de Garde Pédagogique: Valide la cohérence avec le topic.

    Responsabilités:
    - Vérifier que le code correspond au sujet demandé
    - S'assurer que les concepts sont introduits dans le bon ordre
    - Détecter et rejeter le code hors-sujet
    - Valider la progression pédagogique
    """

    SYSTEM_PROMPT = """You are a pedagogical quality guard. Your role is to ensure code matches the lesson topic.

Your job is to validate:
1. TOPIC RELEVANCE: Does the code actually demonstrate the requested topic?
2. CONCEPT FOCUS: Is the code focused on the main concept, not tangential topics?
3. PREREQUISITE CHECK: Are concepts used before being introduced?
4. PEDAGOGICAL ORDER: Does the code build understanding step by step?

Be strict. If the topic is "for loops in Python" and the code shows classes or decorators, REJECT it.
If the topic is "API endpoints" and the code shows database models only, REJECT it.
The code MUST directly demonstrate the topic, not just be tangentially related."""

    # Keywords for common programming concepts
    CONCEPT_KEYWORDS = {
        "loop": ["for", "while", "iteration", "iterate", "range", "enumerate", "loop"],
        "boucle": ["for", "while", "itération", "itérer", "range", "enumerate", "boucle"],
        "function": ["def", "function", "return", "parameter", "argument", "call"],
        "fonction": ["def", "fonction", "retour", "paramètre", "argument", "appel"],
        "class": ["class", "object", "instance", "__init__", "method", "attribute", "self"],
        "classe": ["class", "objet", "instance", "__init__", "méthode", "attribut", "self"],
        "list": ["list", "append", "extend", "index", "slice", "[]"],
        "liste": ["liste", "append", "extend", "index", "slice", "[]"],
        "dict": ["dict", "dictionary", "key", "value", "{}", "items", "keys", "values"],
        "dictionnaire": ["dict", "dictionnaire", "clé", "valeur", "{}", "items", "keys", "values"],
        "api": ["endpoint", "route", "request", "response", "GET", "POST", "REST", "HTTP"],
        "exception": ["try", "except", "raise", "error", "exception", "finally"],
        "file": ["open", "read", "write", "file", "with", "close", "path"],
        "fichier": ["open", "read", "write", "fichier", "with", "close", "path"],
        "async": ["async", "await", "asyncio", "coroutine", "concurrent"],
        "decorator": ["@", "decorator", "wrapper", "functools"],
        "comprehension": ["comprehension", "[x for", "{x:", "(x for"],
        "lambda": ["lambda", "anonymous", "inline function"],
        "recursion": ["recursion", "recursive", "base case", "call itself"],
        "récursion": ["récursion", "récursif", "cas de base"],
        "inheritance": ["inherit", "parent", "child", "super", "override", "extends"],
        "héritage": ["hériter", "parent", "enfant", "super", "surcharge"],
        "module": ["import", "from", "module", "package", "__init__"],
        "string": ["string", "str", "format", "f-string", "split", "join", "replace"],
        "chaîne": ["chaîne", "str", "format", "f-string", "split", "join", "replace"],
    }

    # Frameworks/patterns FORBIDDEN in simple concept lessons
    FORBIDDEN_IN_CONCEPTS = [
        "flask", "django", "fastapi", "tornado", "bottle", "pyramid",  # Web frameworks
        "sqlalchemy", "peewee", "mongoengine",  # ORMs
        "celery", "redis", "rabbitmq",  # Task queues
        "kubernetes", "docker",  # Infrastructure
        "@app.route", "render_template", "Blueprint",  # Flask patterns
        "APIRouter", "Depends",  # FastAPI patterns
    ]

    # Simple concept lesson indicators
    SIMPLE_CONCEPT_TOPICS = [
        "boucle", "loop", "for", "while",
        "fonction", "function", "def",
        "classe", "class", "objet", "object",
        "liste", "list", "tableau", "array",
        "dictionnaire", "dictionary", "dict",
        "condition", "if", "else",
        "variable", "type", "string", "int", "float",
        "tuple", "set", "ensemble",
        "exception", "try", "except",
        "fichier", "file", "open", "read", "write",
        "module", "import",
        "comprehension", "compréhension",
        "lambda", "récursion", "recursion",
    ]

    def __init__(self, llm_provider):
        super().__init__(llm_provider, AgentRole.REVIEWER)  # Reuse REVIEWER role
        self.name = "TopicGuardAgent"

    # Frameworks that indicate a framework-specific lesson (not a simple concept)
    FRAMEWORK_TOPICS = [
        # Python web
        "flask", "django", "fastapi", "tornado", "bottle", "pyramid", "starlette",
        # Java
        "spring", "springboot", "j2ee", "jee", "servlet", "jsp", "struts", "hibernate",
        # JavaScript/Node
        "express", "nestjs", "nextjs", "nuxt", "react", "vue", "angular", "svelte",
        # .NET
        "asp.net", "dotnet", ".net", "blazor", "entity framework",
        # PHP
        "laravel", "symfony", "codeigniter", "yii",
        # Ruby
        "rails", "sinatra",
        # Go
        "gin", "echo", "fiber",
        # Databases/ORMs
        "sqlalchemy", "mongoose", "prisma", "sequelize",
        # DevOps/Cloud
        "kubernetes", "docker", "aws", "azure", "gcp", "terraform", "ansible",
        # Mobile
        "flutter", "react native", "swift", "kotlin",
    ]

    def _is_simple_concept_lesson(self, topic: str) -> bool:
        """Check if this is a simple concept lesson (not a project or framework-specific)."""
        topic_lower = topic.lower()

        # If topic mentions a specific framework, it's NOT a simple concept lesson
        for framework in self.FRAMEWORK_TOPICS:
            if framework in topic_lower:
                return False

        # Check if topic matches simple concept patterns
        for concept in self.SIMPLE_CONCEPT_TOPICS:
            if concept in topic_lower:
                # Make sure it's not a project (e.g., "API avec Flask")
                project_indicators = ["projet", "project", "application", "app", "api", "web", "site", "plateforme"]
                if not any(ind in topic_lower for ind in project_indicators):
                    return True
        return False

    def _detect_forbidden_frameworks(self, code: str) -> List[str]:
        """Detect forbidden frameworks/patterns in code."""
        code_lower = code.lower()
        found = []
        for pattern in self.FORBIDDEN_IN_CONCEPTS:
            if pattern.lower() in code_lower:
                found.append(pattern)
        return found

    def extract_topic_keywords(self, description: str) -> List[str]:
        """Extract expected keywords from the topic description."""
        description_lower = description.lower()
        keywords = []

        for concept, concept_keywords in self.CONCEPT_KEYWORDS.items():
            if concept in description_lower:
                keywords.extend(concept_keywords)

        # Also extract explicit keywords from the description
        # e.g., "boucles for" → ["for", "boucle"]
        words = description_lower.split()
        for word in words:
            if len(word) > 2 and word not in ["les", "des", "une", "the", "and", "for", "with"]:
                keywords.append(word)

        return list(set(keywords))

    def check_code_matches_topic(self, code: str, topic_keywords: List[str]) -> float:
        """Quick heuristic check if code contains topic keywords."""
        code_lower = code.lower()
        matches = 0

        for keyword in topic_keywords:
            if keyword in code_lower:
                matches += 1

        if not topic_keywords:
            return 1.0

        return matches / len(topic_keywords)

    def process(self, context: AgentContext, segment: CodeSegment) -> Dict[str, Any]:
        """Validate that the code matches the lesson topic."""

        # Extract topic from request
        topic = context.request.project_description
        lesson_context = context.request.lesson_context

        # FIRST: Check for forbidden frameworks in concept lessons
        is_concept_lesson = self._is_simple_concept_lesson(topic)
        if is_concept_lesson:
            forbidden_found = self._detect_forbidden_frameworks(segment.code)
            if forbidden_found:
                logger.warning(f"[TOPIC_GUARD] REJECTED: Forbidden frameworks in concept lesson: {forbidden_found}")
                return {
                    "topic_match": False,
                    "score": 0.0,
                    "issues": [f"Code uses frameworks ({', '.join(forbidden_found)}) but topic '{topic}' is a simple concept lesson. Use simple examples with print() instead."],
                    "verdict": "FAIL",
                    "revision_instructions": f"Remove all frameworks. Write simple standalone code that demonstrates {topic} using print() statements only.",
                }

        # Extract expected keywords
        topic_keywords = self.extract_topic_keywords(topic)

        # Quick heuristic check
        keyword_match_score = self.check_code_matches_topic(segment.code, topic_keywords)

        # If heuristic shows good match (>0.3), likely OK
        if keyword_match_score >= 0.3:
            logger.info(f"[TOPIC_GUARD] Quick check passed: {keyword_match_score:.1%} keyword match")
            return {
                "topic_match": True,
                "score": keyword_match_score,
                "issues": [],
            }

        # LLM validation for uncertain cases
        prompt = f"""Validate if this code matches the lesson topic.

LESSON TOPIC: {topic}
LESSON CONTEXT: {lesson_context}

CODE TO VALIDATE:
```{segment.language}
{segment.code}
```

EXPECTED KEYWORDS: {topic_keywords}

Questions to answer:
1. Does this code directly demonstrate the topic "{topic}"?
2. Is the code focused on the main concept or does it drift to other topics?
3. Are there concepts used that weren't introduced (prerequisites missing)?

Output JSON:
{{
  "topic_match": true/false,
  "relevance_score": 0.0-1.0,
  "main_concept_demonstrated": "the concept this code actually shows",
  "expected_concept": "{topic}",
  "issues": ["list of problems"],
  "off_topic_elements": ["things that shouldn't be here"],
  "missing_elements": ["things that should be demonstrated but aren't"],
  "verdict": "PASS" or "FAIL" or "NEEDS_REVISION",
  "revision_instructions": "How to fix if NEEDS_REVISION"
}}"""

        result = self._get_llm_json(self.SYSTEM_PROMPT, prompt)

        topic_match = result.get("topic_match", True)
        relevance_score = result.get("relevance_score", 0.5)
        verdict = result.get("verdict", "PASS")

        logger.info(f"[TOPIC_GUARD] LLM validation: {verdict}, score={relevance_score:.1%}")

        if verdict == "FAIL":
            logger.warning(f"[TOPIC_GUARD] Code rejected: {result.get('issues', [])}")

        return result


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
2. STAY FOCUSED ON THE LESSON TOPIC - every explanation should relate back to the main topic
3. Explain what each part of the code does and WHY it relates to the lesson topic
4. Weave key concepts naturally into your explanation (NEVER list them like "Key concepts: X, Y, Z")
5. Mention pitfalls naturally (e.g., "Be careful not to..." instead of "Common mistakes: ...")
6. Flow smoothly with transitions like "Now let's see...", "Notice how...", "This is important because..."

CRITICAL STYLE RULES:
- NEVER use bullet-point style speech or lists in narration
- NEVER say "Key concepts:", "Key metrics:", "Common mistakes:", or similar headers
- NEVER end with a summary list of concepts
- ALWAYS reference the lesson topic in your narration (e.g., "This is how for loops work...")
- Speak as if you're explaining to a student sitting next to you
- Use contractions (it's, we're, let's) for natural flow
- Integrate ALL concepts naturally into your flowing explanation
- Keep the focus on demonstrating the LESSON TOPIC, not tangential topics"""

    def __init__(self, llm_provider):
        super().__init__(llm_provider, AgentRole.NARRATOR)
    
    def process(self, context: AgentContext, segment: CodeSegment,
                previous_segments: List[CodeSegment] = None) -> Dict[str, Any]:
        """Génère la narration pour un segment"""

        # Extract lesson topic for focus
        lesson_topic = context.request.project_description
        lesson_context = context.request.lesson_context

        prev_context = ""
        if previous_segments:
            prev_context = f"Previous topics covered: {[s.key_concepts for s in previous_segments[-3:]]}"

        prompt = f"""Generate natural, conversational narration for this code segment.

LESSON TOPIC: {lesson_topic}
LESSON CONTEXT: {lesson_context}

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

CRITICAL REQUIREMENTS:
1. Write as a teacher naturally explaining code
2. NO lists, NO "Key concepts:", NO mechanical enumeration
3. ALWAYS relate your explanation back to the lesson topic "{lesson_topic}"
4. Start by mentioning what aspect of "{lesson_topic}" this code demonstrates
5. Keep the focus on the lesson topic, not tangential concepts
6. Just flowing, conversational explanation that covers all concepts organically

Output JSON:
{{
  "narration_script": "Natural flowing narration focused on {lesson_topic}",
  "duration_estimate_seconds": 30,
  "emphasis_points": ["word or phrase to emphasize"],
  "pause_after": ["concept after which to pause briefly"],
  "transition_to_next": "Transition phrase to next segment"
}}"""

        result = self._get_llm_json(self.SYSTEM_PROMPT, prompt)

        # Mettre à jour le segment
        narration = result.get("narration_script", "")
        segment.narration_script = narration
        segment.duration_seconds = result.get("duration_estimate_seconds", 30)

        logger.info(f"[NARRATOR] Generated narration: {len(narration)} chars")
        if narration:
            logger.info(f"[NARRATOR] Preview: {narration[:100]}...")

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
        self.topic_guard = TopicGuardAgent(llm_provider)  # NEW: Topic validation
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
            
            # 3. Reviewer (quality check)
            review_result = self.reviewer.process(context, segment, architecture)

            if not review_result.get("overall_pass", True):
                # Quality failed, prepare feedback for next iteration
                feedback = {
                    "issues": review_result.get("must_fix", []),
                    "suggestions": review_result.get("suggestions", []),
                    "feedback": review_result.get("feedback_for_coder", ""),
                }
                logger.debug(f"Feedback loop {iteration} (quality): {feedback['issues']}")
                continue

            # 4. TopicGuard (topic relevance check) - NEW
            topic_result = self.topic_guard.process(context, segment)
            topic_verdict = topic_result.get("verdict", "PASS")

            if topic_verdict == "FAIL":
                # Topic mismatch - code doesn't demonstrate the lesson topic
                feedback = {
                    "issues": [f"Code does not demonstrate the topic: {context.request.project_description}"],
                    "suggestions": topic_result.get("missing_elements", []),
                    "feedback": topic_result.get("revision_instructions", "Rewrite code to focus on the lesson topic"),
                }
                logger.warning(f"[TOPIC_GUARD] Code rejected, regenerating. Issues: {topic_result.get('issues', [])}")
                continue
            elif topic_verdict == "NEEDS_REVISION":
                # Partial match - needs minor adjustments
                feedback = {
                    "issues": topic_result.get("off_topic_elements", []),
                    "suggestions": topic_result.get("missing_elements", []),
                    "feedback": topic_result.get("revision_instructions", ""),
                }
                logger.info(f"[TOPIC_GUARD] Code needs revision: {topic_result.get('revision_instructions', '')}")
                continue

            # All checks passed!
            logger.info(f"[TOPIC_GUARD] Code approved: {topic_result.get('relevance_score', 1.0):.0%} relevance")
            break
        
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

        topic = context.request.project_description
        is_topic_issue = "topic" in str(feedback.get('issues', [])).lower()

        if is_topic_issue:
            # Topic mismatch - need complete rewrite focused on topic
            prompt = f"""REWRITE this code completely. The current code is OFF-TOPIC.

LESSON TOPIC: {topic}

The code must ONLY demonstrate "{topic}".
- If topic is about loops: show simple for/while loops with print()
- If topic is about functions: show simple function definitions
- If topic is about classes: show simple class examples
- NO Flask, NO APIs, NO complex patterns unless the topic is specifically about them

Current code (DO NOT USE THIS - it's wrong):
```{segment.language}
{segment.code}
```

Issues: {feedback.get('issues', [])}

Write NEW code that:
1. Directly demonstrates {topic}
2. Is simple and educational
3. Uses print() to show output
4. Is beginner-friendly

Output ONLY the new code, no explanations:"""
        else:
            prompt = f"""Improve this code based on feedback:

LESSON TOPIC: {topic}

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

        system_content = f"Expert {segment.language} developer. Code only. Keep code focused on: {topic}"
        messages = [
            LLMMessage(role="system", content=system_content),
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
