"""
NEXUS Cognitive Decomposition Algorithm (CDA)
L'algorithme de décomposition unique qui transforme une description en Architecture DNA

Innovation clé: Décomposition en 4 phases cognitives qui simulent
le raisonnement d'un architecte senior.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import re

from models.data_models import (
    ArchitectureDNA, DomainEntity, EntityRelation, BusinessFlow,
    CognitiveBlueprint, CognitiveStep, NexusRequest,
    PatternType, ComponentType, TargetAudience, CodeVerbosity
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS SPÉCIALISÉS POUR LA DÉCOMPOSITION COGNITIVE
# =============================================================================

DOMAIN_EXTRACTION_PROMPT = """Analyze this project description and extract the domain model.

Project: "{description}"
Context: {context}

Think like a senior domain architect. Extract:

1. ENTITIES: Core business objects with their attributes and types
2. RELATIONS: How entities relate to each other
3. BEHAVIORS: Key actions/methods for each entity
4. CONSTRAINTS: Business rules and validations

Output JSON:
{{
  "entities": [
    {{
      "name": "EntityName",
      "description": "What this entity represents",
      "attributes": [
        {{"name": "attr_name", "type": "string|int|float|bool|datetime|reference", "constraints": "required|unique|nullable"}}
      ],
      "behaviors": ["action1", "action2"],
      "constraints": ["rule1", "rule2"],
      "dependencies": ["OtherEntity"]
    }}
  ],
  "relations": [
    {{
      "source": "Entity1",
      "target": "Entity2", 
      "relation_type": "has_many|belongs_to|has_one|many_to_many",
      "attributes": {{}}
    }}
  ],
  "business_rules": ["Global rule 1", "Global rule 2"]
}}

Be thorough but realistic. JSON only."""


FLOW_EXTRACTION_PROMPT = """Based on this domain, identify the key business flows.

Domain: {domain_name}
Entities: {entities}

Extract the main user journeys/use cases. For each flow:
- What triggers it
- The steps involved
- Which entities participate
- What can go wrong

Output JSON:
{{
  "flows": [
    {{
      "name": "FlowName",
      "description": "What this flow accomplishes",
      "steps": [
        {{"order": 1, "action": "User does X", "entity": "Entity", "result": "Y happens"}}
      ],
      "actors": ["Entity1", "Entity2"],
      "preconditions": ["State required before"],
      "postconditions": ["State after completion"],
      "error_scenarios": [
        {{"trigger": "When X fails", "handling": "Do Y"}}
      ]
    }}
  ]
}}

JSON only."""


ARCHITECTURE_DECISION_PROMPT = """Design the technical architecture for this project.

Domain: {domain_name}
Complexity: {complexity}
Language: {language}
Target Audience: {audience}
Skill Level: {skill_level}

Select appropriate:
1. PATTERNS: Which design patterns fit this domain
2. LAYERS: How to structure the code
3. FRAMEWORK: Best framework for this language/domain
4. DEPENDENCIES: Required packages

Consider the skill level:
- Beginner: Simple patterns, minimal layers, popular frameworks
- Intermediate: Standard patterns, clear separation
- Advanced: Sophisticated patterns, clean architecture
- Expert: DDD, CQRS if needed, optimal choices

Output JSON:
{{
  "patterns": ["pattern1", "pattern2"],
  "layers": ["layer1", "layer2"],
  "framework": "framework_name",
  "dependencies": {{"package": "version"}},
  "rationale": "Why these choices"
}}

JSON only."""


COGNITIVE_PLANNING_PROMPT = """Plan the implementation steps for teaching this project.

Architecture DNA:
- Entities: {entities}
- Flows: {flows}
- Patterns: {patterns}
- Language: {language}

Time Budget: {time_budget} seconds
Target: {audience}
Verbosity: {verbosity}

Create a cognitive plan - how would an expert explain building this step by step?

For each step, provide:
- What you're thinking (the "why")
- The decision you make
- What code component this produces

Output JSON:
{{
  "analysis_phase": [
    {{
      "thought": "First I need to understand...",
      "reasoning": "Because...",
      "decision": "I will...",
      "code_component": "none|model|service|etc",
      "duration_seconds": 20,
      "narration_cue": "Let's start by analyzing..."
    }}
  ],
  "design_phase": [...],
  "implementation_phase": [...],
  "validation_phase": [...]
}}

Keep total duration close to {time_budget} seconds.
JSON only."""


# =============================================================================
# COGNITIVE DECOMPOSITION ALGORITHM
# =============================================================================

class CognitiveDecompositionAlgorithm:
    """
    Algorithme de Décomposition Cognitive (CDA)
    
    Innovation unique: Simule le processus de réflexion d'un architecte senior
    en 4 phases cognitives:
    
    1. PERCEPTION: Comprendre le domaine et extraire les entités
    2. ANALYSIS: Identifier les flux et règles métier
    3. SYNTHESIS: Choisir l'architecture et les patterns
    4. PLANNING: Créer le plan d'implémentation cognitif
    
    La clé est que chaque phase alimente la suivante avec un contexte
    enrichi, créant une chaîne de raisonnement cohérente.
    """
    
    # Matrices de décision propriétaires
    PATTERN_AFFINITY_MATRIX = {
        # domain_keyword -> patterns recommandés
        "ecommerce": [PatternType.REPOSITORY, PatternType.SERVICE_LAYER, PatternType.MVC],
        "e-commerce": [PatternType.REPOSITORY, PatternType.SERVICE_LAYER, PatternType.MVC],
        "shop": [PatternType.REPOSITORY, PatternType.SERVICE_LAYER],
        "cart": [PatternType.REPOSITORY, PatternType.SERVICE_LAYER],
        "auth": [PatternType.REPOSITORY, PatternType.SERVICE_LAYER],
        "api": [PatternType.REPOSITORY, PatternType.SERVICE_LAYER, PatternType.MVC],
        "crud": [PatternType.REPOSITORY, PatternType.MVC],
        "blog": [PatternType.REPOSITORY, PatternType.MVC],
        "social": [PatternType.REPOSITORY, PatternType.SERVICE_LAYER, PatternType.EVENT_SOURCING],
        "banking": [PatternType.DOMAIN_DRIVEN, PatternType.CQRS, PatternType.EVENT_SOURCING],
        "trading": [PatternType.CQRS, PatternType.EVENT_SOURCING],
        "microservice": [PatternType.MICROSERVICES, PatternType.DOMAIN_DRIVEN],
    }
    
    FRAMEWORK_MATRIX = {
        # (language, complexity, audience) -> framework
        ("python", "low", TargetAudience.STUDENT): "flask",
        ("python", "low", TargetAudience.DEVELOPER): "flask",
        ("python", "medium", TargetAudience.STUDENT): "flask",
        ("python", "medium", TargetAudience.DEVELOPER): "fastapi",
        ("python", "high", TargetAudience.DEVELOPER): "fastapi",
        ("python", "high", TargetAudience.ARCHITECT): "fastapi",
        ("javascript", "low", TargetAudience.STUDENT): "express",
        ("javascript", "medium", TargetAudience.DEVELOPER): "express",
        ("javascript", "high", TargetAudience.DEVELOPER): "nestjs",
        ("java", "low", TargetAudience.STUDENT): "spring-boot",
        ("java", "medium", TargetAudience.DEVELOPER): "spring-boot",
        ("java", "high", TargetAudience.ARCHITECT): "spring-boot",
        ("go", "low", TargetAudience.DEVELOPER): "gin",
        ("go", "medium", TargetAudience.DEVELOPER): "gin",
        ("go", "high", TargetAudience.DEVELOPER): "fiber",
    }
    
    LAYER_TEMPLATES = {
        "simple": ["models", "routes"],
        "standard": ["models", "services", "routes"],
        "layered": ["models", "repositories", "services", "controllers"],
        "clean": ["domain/entities", "domain/repositories", "application/services", 
                  "infrastructure/persistence", "presentation/api"],
        "ddd": ["domain/entities", "domain/value_objects", "domain/aggregates",
                "domain/repositories", "application/commands", "application/queries",
                "infrastructure/persistence", "presentation/api"],
    }
    
    def __init__(self, llm_provider):
        """
        Args:
            llm_provider: Provider LLM (interface agnostique)
        """
        self.llm = llm_provider
        
        # Cache pour optimisation
        self._domain_cache: Dict[str, Dict] = {}
    
    def decompose(self, request: NexusRequest) -> Tuple[ArchitectureDNA, CognitiveBlueprint]:
        """
        Point d'entrée principal: Décompose une requête en DNA + Blueprint.
        
        Returns:
            Tuple (ArchitectureDNA, CognitiveBlueprint)
        """
        logger.info(f"Starting cognitive decomposition for: {request.project_description[:50]}...")
        
        # Phase 1: PERCEPTION - Extraire le domaine
        domain_data = self._phase_perception(request)
        
        # Phase 2: ANALYSIS - Identifier les flux
        flows_data = self._phase_analysis(request, domain_data)
        
        # Phase 3: SYNTHESIS - Choisir l'architecture
        architecture_data = self._phase_synthesis(request, domain_data, flows_data)
        
        # Phase 4: PLANNING - Créer le blueprint cognitif
        blueprint_data = self._phase_planning(request, domain_data, flows_data, architecture_data)
        
        # Construire les objets finaux
        dna = self._build_architecture_dna(request, domain_data, flows_data, architecture_data)
        blueprint = self._build_cognitive_blueprint(dna, blueprint_data)
        
        logger.info(f"Decomposition complete: {len(dna.entities)} entities, "
                   f"{len(dna.flows)} flows, {len(blueprint.all_steps)} cognitive steps")
        
        return dna, blueprint
    
    # =========================================================================
    # PHASE 1: PERCEPTION
    # =========================================================================
    
    def _phase_perception(self, request: NexusRequest) -> Dict:
        """
        Phase de perception: Comprendre le domaine.

        Utilise une combinaison de:
        1. Analyse heuristique (patterns connus)
        2. Extraction LLM (pour les détails)

        Pour les leçons sur des CONCEPTS (loops, functions, etc.), on skip
        l'extraction d'entités et on se concentre sur le concept.
        """
        logger.debug("Phase 1: Perception - Domain extraction")

        # D'abord, analyse heuristique rapide
        heuristic_hints = self._heuristic_domain_analysis(request.project_description)

        # For CONCEPT lessons, return simplified result
        if heuristic_hints.get("is_concept_lesson"):
            concept_type = heuristic_hints.get("concept_type", "programming")
            logger.info(f"[CDA] Concept lesson detected: {concept_type} - skipping entity extraction")

            return {
                "entities": [],  # No entities for concept lessons
                "relations": [],
                "business_rules": [],
                "_heuristic_hints": heuristic_hints,
                "_is_concept_lesson": True,
                "_concept_type": concept_type,
                "_focus_topic": request.project_description,
            }

        # For PROJECT lessons, do full entity extraction
        from providers.llm_provider import LLMMessage

        prompt = DOMAIN_EXTRACTION_PROMPT.format(
            description=request.project_description,
            context=f"Lesson: {request.lesson_context}\nHints: {heuristic_hints}"
        )

        messages = [
            LLMMessage(role="system", content="You are a senior domain architect. JSON output only."),
            LLMMessage(role="user", content=prompt),
        ]

        result = self.llm.generate_json(messages)

        # Enrichir avec les hints heuristiques
        result["_heuristic_hints"] = heuristic_hints

        return result
    
    def _heuristic_domain_analysis(self, description: str) -> Dict:
        """
        Analyse heuristique rapide basée sur des patterns connus.

        Détecte si c'est une leçon sur un CONCEPT (boucles, fonctions, etc.)
        ou un PROJET (e-commerce, blog, etc.)

        C'est une partie de la "sauce secrète" - des connaissances
        encodées sur les domaines courants.
        """
        description_lower = description.lower()
        hints = {
            "detected_domain": None,
            "suggested_entities": [],
            "suggested_patterns": [],
            "is_concept_lesson": False,  # NEW: Flag for concept vs project
            "concept_type": None,        # NEW: Type of concept if detected
        }

        # FIRST: Check if this is a CONCEPT lesson (not a project)
        concept_keywords = {
            "loop": ["boucle", "loop", "for loop", "while loop", "for", "while", "iterate", "iteration", "itération"],
            "function": ["fonction", "function", "def", "méthode", "method", "paramètre", "parameter", "return"],
            "class": ["classe", "class", "objet", "object", "héritage", "inheritance", "oop", "orienté objet"],
            "list": ["liste", "list", "array", "tableau", "append", "index"],
            "dict": ["dictionnaire", "dictionary", "dict", "hashmap", "clé", "key", "value", "valeur"],
            "string": ["chaîne", "string", "texte", "text", "caractère", "character"],
            "file": ["fichier", "file", "open", "read", "write", "io"],
            "exception": ["exception", "error", "erreur", "try", "except", "catch"],
            "decorator": ["décorateur", "decorator", "@"],
            "generator": ["générateur", "generator", "yield"],
            "async": ["async", "await", "asynchrone", "asyncio", "coroutine"],
            "recursion": ["récursion", "recursion", "récursif", "recursive"],
            "comprehension": ["compréhension", "comprehension", "list comprehension"],
            "lambda": ["lambda", "fonction anonyme", "anonymous function"],
            "module": ["module", "import", "package"],
            "api": ["api", "endpoint", "rest", "http", "request", "response"],
            "database": ["database", "sql", "base de données", "query", "requête"],
            "regex": ["regex", "expression régulière", "regular expression", "pattern matching"],
            "testing": ["test", "unittest", "pytest", "testing", "tdd"],
        }

        for concept_type, keywords in concept_keywords.items():
            if any(kw in description_lower for kw in keywords):
                hints["is_concept_lesson"] = True
                hints["concept_type"] = concept_type
                hints["detected_domain"] = f"concept:{concept_type}"
                logger.info(f"[CDA] Detected CONCEPT lesson: {concept_type}")
                break

        # If concept lesson, don't look for project entities
        if hints["is_concept_lesson"]:
            return hints

        # PROJECT detection (original logic)
        domain_keywords = {
            "e-commerce": ["ecommerce", "e-commerce", "shop", "store", "cart", "checkout", "product", "order"],
            "blog": ["blog", "post", "article", "comment", "author"],
            "social": ["social", "friend", "follow", "feed", "post", "like", "share"],
            "banking": ["bank", "account", "transaction", "transfer", "balance"],
            "education": ["course", "student", "lesson", "quiz", "grade", "enrollment"],
            "healthcare": ["patient", "doctor", "appointment", "prescription", "diagnosis"],
            "project_management": ["project", "task", "milestone", "team", "sprint"],
            "inventory": ["inventory", "stock", "warehouse", "product", "supplier"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in description_lower for kw in keywords):
                hints["detected_domain"] = domain
                hints["suggested_entities"] = self._get_domain_entities(domain)
                break

        # Patterns suggérés
        for keyword, patterns in self.PATTERN_AFFINITY_MATRIX.items():
            if keyword in description_lower:
                hints["suggested_patterns"].extend(patterns)

        # Dédupliquer
        hints["suggested_patterns"] = list(set(hints["suggested_patterns"]))

        return hints
    
    def _get_domain_entities(self, domain: str) -> List[Dict]:
        """Retourne les entités typiques pour un domaine connu."""
        domain_entities = {
            "e-commerce": [
                {"name": "User", "core": True},
                {"name": "Product", "core": True},
                {"name": "Category", "core": False},
                {"name": "Cart", "core": True},
                {"name": "CartItem", "core": True},
                {"name": "Order", "core": True},
                {"name": "OrderItem", "core": True},
                {"name": "Payment", "core": True},
                {"name": "Address", "core": False},
            ],
            "blog": [
                {"name": "User", "core": True},
                {"name": "Post", "core": True},
                {"name": "Category", "core": False},
                {"name": "Tag", "core": False},
                {"name": "Comment", "core": True},
            ],
            "social": [
                {"name": "User", "core": True},
                {"name": "Post", "core": True},
                {"name": "Comment", "core": True},
                {"name": "Like", "core": True},
                {"name": "Follow", "core": True},
                {"name": "Notification", "core": False},
            ],
            "education": [
                {"name": "User", "core": True},
                {"name": "Course", "core": True},
                {"name": "Lesson", "core": True},
                {"name": "Enrollment", "core": True},
                {"name": "Quiz", "core": False},
                {"name": "Grade", "core": False},
            ],
        }
        return domain_entities.get(domain, [])
    
    # =========================================================================
    # PHASE 2: ANALYSIS
    # =========================================================================
    
    def _phase_analysis(self, request: NexusRequest, domain_data: Dict) -> Dict:
        """
        Phase d'analyse: Identifier les flux métier.
        """
        logger.debug("Phase 2: Analysis - Flow extraction")
        
        entities = [e["name"] for e in domain_data.get("entities", [])]
        
        from providers.llm_provider import LLMMessage
        
        prompt = FLOW_EXTRACTION_PROMPT.format(
            domain_name=request.project_description,
            entities=", ".join(entities),
        )
        
        messages = [
            LLMMessage(role="system", content="You are a business analyst expert. JSON output only."),
            LLMMessage(role="user", content=prompt),
        ]
        
        result = self.llm.generate_json(messages)
        return result
    
    # =========================================================================
    # PHASE 3: SYNTHESIS
    # =========================================================================
    
    def _phase_synthesis(self, request: NexusRequest, 
                         domain_data: Dict, 
                         flows_data: Dict) -> Dict:
        """
        Phase de synthèse: Choisir l'architecture.
        
        Combine:
        1. Décisions basées sur les matrices propriétaires
        2. Raffinement LLM si nécessaire
        """
        logger.debug("Phase 3: Synthesis - Architecture decision")
        
        # Calcul de complexité
        num_entities = len(domain_data.get("entities", []))
        num_flows = len(flows_data.get("flows", []))
        complexity = self._calculate_complexity(num_entities, num_flows, request)
        
        # Sélection de framework via matrice
        framework = self._select_framework(request.language, complexity, request.target_audience)
        
        # Sélection de patterns
        patterns = self._select_patterns(domain_data, complexity, request)
        
        # Sélection de layers
        layers = self._select_layers(complexity, request)
        
        # Affiner avec LLM si complexité élevée
        if complexity == "high":
            refined = self._refine_architecture_with_llm(
                request, domain_data, flows_data, framework, patterns, layers
            )
            if refined:
                return refined
        
        return {
            "patterns": patterns,
            "layers": layers,
            "framework": framework,
            "dependencies": self._get_default_dependencies(request.language, framework),
            "complexity": complexity,
        }
    
    def _calculate_complexity(self, num_entities: int, num_flows: int, 
                              request: NexusRequest) -> str:
        """Calcule le niveau de complexité."""
        score = num_entities * 0.3 + num_flows * 0.3
        
        if request.target_audience in [TargetAudience.ARCHITECT, TargetAudience.TECHNICAL_LEAD]:
            score += 0.2
        
        if request.verbosity >= CodeVerbosity.PRODUCTION:
            score += 0.1
        
        if score < 2:
            return "low"
        elif score < 4:
            return "medium"
        else:
            return "high"
    
    def _select_framework(self, language: str, complexity: str, 
                          audience: TargetAudience) -> str:
        """Sélectionne le framework optimal."""
        key = (language.lower(), complexity, audience)
        
        # Chercher exactement
        if key in self.FRAMEWORK_MATRIX:
            return self.FRAMEWORK_MATRIX[key]
        
        # Fallback: chercher avec audience par défaut
        for aud in [TargetAudience.DEVELOPER, TargetAudience.STUDENT]:
            fallback_key = (language.lower(), complexity, aud)
            if fallback_key in self.FRAMEWORK_MATRIX:
                return self.FRAMEWORK_MATRIX[fallback_key]
        
        # Défauts par langage
        defaults = {
            "python": "flask",
            "javascript": "express",
            "typescript": "express",
            "java": "spring-boot",
            "go": "gin",
            "rust": "actix-web",
            "php": "laravel",
            "ruby": "rails",
            "csharp": "aspnet-core",
        }
        return defaults.get(language.lower(), "custom")
    
    def _select_patterns(self, domain_data: Dict, complexity: str, 
                         request: NexusRequest) -> List[str]:
        """Sélectionne les patterns appropriés."""
        patterns = set()
        
        # Patterns suggérés par l'analyse heuristique
        hints = domain_data.get("_heuristic_hints", {})
        for p in hints.get("suggested_patterns", []):
            patterns.add(p.value if isinstance(p, PatternType) else p)
        
        # Patterns par défaut selon complexité
        if complexity == "low":
            patterns.update(["mvc", "repository"])
        elif complexity == "medium":
            patterns.update(["repository", "service_layer", "mvc"])
        else:
            patterns.update(["repository", "service_layer", "clean_architecture"])
        
        # Ajuster selon audience
        if request.target_audience == TargetAudience.ARCHITECT:
            patterns.add("clean_architecture")
        
        return list(patterns)[:4]  # Max 4 patterns
    
    def _select_layers(self, complexity: str, request: NexusRequest) -> List[str]:
        """Sélectionne la structure en couches."""
        if request.skill_level in ["beginner", "novice"]:
            return self.LAYER_TEMPLATES["simple"]
        elif complexity == "low":
            return self.LAYER_TEMPLATES["standard"]
        elif complexity == "medium":
            return self.LAYER_TEMPLATES["layered"]
        else:
            if request.target_audience == TargetAudience.ARCHITECT:
                return self.LAYER_TEMPLATES["ddd"]
            return self.LAYER_TEMPLATES["clean"]
    
    def _get_default_dependencies(self, language: str, framework: str) -> Dict[str, str]:
        """Retourne les dépendances par défaut pour un framework."""
        deps_map = {
            ("python", "flask"): {
                "flask": ">=2.0.0",
                "flask-sqlalchemy": ">=3.0.0",
                "python-dotenv": ">=1.0.0",
            },
            ("python", "fastapi"): {
                "fastapi": ">=0.100.0",
                "uvicorn": ">=0.22.0",
                "sqlalchemy": ">=2.0.0",
                "pydantic": ">=2.0.0",
            },
            ("javascript", "express"): {
                "express": "^4.18.0",
                "mongoose": "^7.0.0",
                "dotenv": "^16.0.0",
            },
        }
        return deps_map.get((language.lower(), framework), {})
    
    def _refine_architecture_with_llm(self, request: NexusRequest,
                                       domain_data: Dict,
                                       flows_data: Dict,
                                       framework: str,
                                       patterns: List[str],
                                       layers: List[str]) -> Optional[Dict]:
        """Raffinement LLM pour cas complexes."""
        from providers.llm_provider import LLMMessage
        
        prompt = ARCHITECTURE_DECISION_PROMPT.format(
            domain_name=request.project_description,
            complexity="high",
            language=request.language,
            audience=request.target_audience.value,
            skill_level=request.skill_level,
        )
        
        messages = [
            LLMMessage(role="system", content="Senior software architect. JSON only."),
            LLMMessage(role="user", content=prompt),
        ]
        
        try:
            result = self.llm.generate_json(messages)
            result["complexity"] = "high"
            return result
        except Exception as e:
            logger.warning(f"LLM refinement failed: {e}")
            return None
    
    # =========================================================================
    # PHASE 4: PLANNING
    # =========================================================================
    
    def _phase_planning(self, request: NexusRequest,
                        domain_data: Dict,
                        flows_data: Dict,
                        architecture_data: Dict) -> Dict:
        """
        Phase de planification: Créer le blueprint cognitif.

        Pour les leçons sur des CONCEPTS, génère des étapes focalisées sur
        la démonstration progressive du concept.
        """
        logger.debug("Phase 4: Planning - Cognitive blueprint")

        from providers.llm_provider import LLMMessage

        # Check if this is a concept lesson
        is_concept_lesson = domain_data.get("_is_concept_lesson", False)
        concept_type = domain_data.get("_concept_type", "")
        focus_topic = domain_data.get("_focus_topic", request.project_description)

        if is_concept_lesson:
            # CONCEPT LESSON: Generate focused demonstration steps
            logger.info(f"[CDA] Planning CONCEPT lesson for: {concept_type}")

            concept_prompt = f"""Plan implementation steps for teaching the programming concept: "{focus_topic}"

Language: {request.language}
Time Budget: {request.allocated_time_seconds} seconds
Target: {request.target_audience.value}
Skill Level: {request.skill_level}
Verbosity: {request.verbosity.value}

IMPORTANT: This is a CONCEPT lesson, NOT a project.
- Focus ONLY on demonstrating "{focus_topic}"
- Each code segment MUST directly demonstrate the concept
- DO NOT include unrelated code (no classes if teaching loops, no decorators if teaching functions)
- Build understanding progressively: simple example → variations → practical use

Create cognitive steps that:
1. Introduce the concept with the simplest possible example
2. Show variations and common patterns
3. Demonstrate practical real-world usage
4. Highlight common mistakes to avoid

Output JSON:
{{
  "analysis_phase": [
    {{
      "thought": "What is the core concept to teach?",
      "reasoning": "Why this approach works best",
      "decision": "Start with...",
      "code_component": "utility",
      "duration_seconds": 30,
      "narration_cue": "Let's understand {concept_type}..."
    }}
  ],
  "implementation_phase": [
    {{
      "thought": "Show the basic syntax/usage",
      "reasoning": "Students need to see the simplest form first",
      "decision": "Create a basic {concept_type} example",
      "code_component": "utility",
      "duration_seconds": 60,
      "narration_cue": "Here's how {concept_type} works..."
    }},
    {{
      "thought": "Show variations and patterns",
      "reasoning": "Build on the basic understanding",
      "decision": "Demonstrate common patterns",
      "code_component": "utility",
      "duration_seconds": 60,
      "narration_cue": "Now let's see some variations..."
    }},
    {{
      "thought": "Practical application",
      "reasoning": "Connect theory to practice",
      "decision": "Real-world example using {concept_type}",
      "code_component": "utility",
      "duration_seconds": 60,
      "narration_cue": "In practice, you might use {concept_type} like this..."
    }}
  ],
  "validation_phase": []
}}

Keep total duration close to {request.allocated_time_seconds} seconds.
JSON only."""

            messages = [
                LLMMessage(role="system", content="Expert coding instructor. JSON only. Focus on the specific concept."),
                LLMMessage(role="user", content=concept_prompt),
            ]

            result = self.llm.generate_json(messages)
            return result

        # PROJECT LESSON: Original logic
        entities = [e["name"] for e in domain_data.get("entities", [])]
        flows = [f["name"] for f in flows_data.get("flows", [])]

        prompt = COGNITIVE_PLANNING_PROMPT.format(
            entities=", ".join(entities),
            flows=", ".join(flows),
            patterns=", ".join(architecture_data.get("patterns", [])),
            language=request.language,
            time_budget=request.allocated_time_seconds,
            audience=request.target_audience.value,
            verbosity=request.verbosity.value,
        )

        messages = [
            LLMMessage(role="system", content="Expert coding instructor. JSON only."),
            LLMMessage(role="user", content=prompt),
        ]

        result = self.llm.generate_json(messages)
        return result
    
    # =========================================================================
    # BUILDERS
    # =========================================================================
    
    def _build_architecture_dna(self, request: NexusRequest,
                                 domain_data: Dict,
                                 flows_data: Dict,
                                 architecture_data: Dict) -> ArchitectureDNA:
        """Construit l'objet ArchitectureDNA."""
        
        # Convertir les entités
        entities = []
        for e_data in domain_data.get("entities", []):
            entity = DomainEntity(
                name=e_data.get("name", ""),
                description=e_data.get("description", ""),
                attributes=e_data.get("attributes", []),
                behaviors=e_data.get("behaviors", []),
                constraints=e_data.get("constraints", []),
                dependencies=e_data.get("dependencies", []),
            )
            entities.append(entity)
        
        # Convertir les relations
        relations = []
        for r_data in domain_data.get("relations", []):
            relation = EntityRelation(
                source=r_data.get("source", ""),
                target=r_data.get("target", ""),
                relation_type=r_data.get("relation_type", "has_many"),
                attributes=r_data.get("attributes", {}),
            )
            relations.append(relation)
        
        # Convertir les flows
        flows = []
        for f_data in flows_data.get("flows", []):
            flow = BusinessFlow(
                name=f_data.get("name", ""),
                description=f_data.get("description", ""),
                steps=f_data.get("steps", []),
                actors=f_data.get("actors", []),
                preconditions=f_data.get("preconditions", []),
                postconditions=f_data.get("postconditions", []),
                error_scenarios=f_data.get("error_scenarios", []),
            )
            flows.append(flow)
        
        # Convertir les patterns
        patterns = []
        for p in architecture_data.get("patterns", []):
            try:
                patterns.append(PatternType(p))
            except ValueError:
                pass
        
        # Calculer la complexité
        complexity = len(entities) * 0.1 + len(flows) * 0.1
        complexity = min(1.0, complexity)
        
        return ArchitectureDNA(
            id="",
            project_name=request.project_description[:50],
            domain_description=request.project_description,
            entities=entities,
            relations=relations,
            flows=flows,
            business_rules=domain_data.get("business_rules", []),
            patterns=patterns,
            layers=architecture_data.get("layers", []),
            language=request.language,
            framework=architecture_data.get("framework", ""),
            dependencies=architecture_data.get("dependencies", {}),
            complexity_level=complexity,
        )
    
    def _build_cognitive_blueprint(self, dna: ArchitectureDNA,
                                    blueprint_data: Dict) -> CognitiveBlueprint:
        """Construit l'objet CognitiveBlueprint."""
        
        def build_steps(phase_data: List[Dict]) -> List[CognitiveStep]:
            steps = []
            for s_data in phase_data:
                step = CognitiveStep(
                    id="",
                    thought=s_data.get("thought", ""),
                    reasoning=s_data.get("reasoning", ""),
                    decision=s_data.get("decision", ""),
                    code_component=s_data.get("code_component", "none"),
                    estimated_duration_seconds=s_data.get("duration_seconds", 30),
                    narration_cue=s_data.get("narration_cue", ""),
                )
                steps.append(step)
            return steps
        
        return CognitiveBlueprint(
            id="",
            dna_id=dna.id,
            analysis_phase=build_steps(blueprint_data.get("analysis_phase", [])),
            design_phase=build_steps(blueprint_data.get("design_phase", [])),
            implementation_phase=build_steps(blueprint_data.get("implementation_phase", [])),
            validation_phase=build_steps(blueprint_data.get("validation_phase", [])),
        )
