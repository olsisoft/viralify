"""
NEXUS Data Models
Neural Execution & Understanding Synthesis

Structures de données innovantes pour la génération de code pédagogique contextuel.
Inclut l'Architecture DNA et le Cognitive Blueprint.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Dict, Optional, Any, Set, Tuple
from datetime import datetime
import json
import uuid
import hashlib


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class TargetAudience(Enum):
    """Public cible pour le code généré"""
    DEVELOPER = "developer"           # Code production-ready, gestion d'erreurs
    ARCHITECT = "architect"           # Mécanique, patterns, structure
    STUDENT = "student"               # Pédagogique, commenté, progressif
    TECHNICAL_LEAD = "technical_lead" # Balance entre les deux


class CodeVerbosity(IntEnum):
    """Niveau de verbosité du code"""
    MINIMAL = 1        # Squelette, essentiel uniquement
    STANDARD = 2       # Code propre, commentaires clés
    VERBOSE = 3        # Très commenté, explicatif
    PRODUCTION = 4     # Production-ready, error handling, logs


class ExecutionMode(Enum):
    """Mode d'exécution du code"""
    DRY_RUN = "dry_run"           # Pas d'exécution, validation syntaxique
    SANDBOX = "sandbox"           # Exécution isolée
    LIVE_DEMO = "live_demo"       # Exécution avec capture output
    INTERACTIVE = "interactive"   # Terminal interactif simulé


class ComponentType(Enum):
    """Types de composants architecturaux"""
    MODEL = "model"
    REPOSITORY = "repository"
    SERVICE = "service"
    CONTROLLER = "controller"
    API_ENDPOINT = "api_endpoint"
    MIDDLEWARE = "middleware"
    UTILITY = "utility"
    CONFIG = "config"
    TEST = "test"
    DATABASE = "database"
    MIGRATION = "migration"
    FRONTEND = "frontend"


class PatternType(Enum):
    """Design patterns reconnus"""
    SINGLETON = "singleton"
    FACTORY = "factory"
    REPOSITORY = "repository"
    SERVICE_LAYER = "service_layer"
    MVC = "mvc"
    MVVM = "mvvm"
    CLEAN_ARCHITECTURE = "clean_architecture"
    CQRS = "cqrs"
    EVENT_SOURCING = "event_sourcing"
    DOMAIN_DRIVEN = "domain_driven"
    MICROSERVICES = "microservices"
    MONOLITH = "monolith"


# =============================================================================
# ARCHITECTURE DNA - Représentation unique d'un projet
# =============================================================================

@dataclass
class EntityRelation:
    """Relation entre deux entités du domaine"""
    source: str
    target: str
    relation_type: str  # "has_many", "belongs_to", "has_one", "many_to_many"
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "attributes": self.attributes,
        }


@dataclass
class DomainEntity:
    """Entité du domaine métier"""
    name: str
    description: str
    attributes: List[Dict[str, str]]  # [{"name": "id", "type": "int", "constraints": "primary_key"}]
    behaviors: List[str]              # Méthodes/actions de l'entité
    constraints: List[str]            # Règles métier
    dependencies: List[str] = field(default_factory=list)
    
    @property
    def complexity_score(self) -> float:
        """Score de complexité de l'entité (0-1)"""
        attr_score = min(1.0, len(self.attributes) / 10)
        behavior_score = min(1.0, len(self.behaviors) / 8)
        constraint_score = min(1.0, len(self.constraints) / 5)
        dep_score = min(1.0, len(self.dependencies) / 5)
        return (attr_score + behavior_score * 1.5 + constraint_score * 1.2 + dep_score) / 4.7
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "attributes": self.attributes,
            "behaviors": self.behaviors,
            "constraints": self.constraints,
            "dependencies": self.dependencies,
            "complexity_score": self.complexity_score,
        }


@dataclass
class BusinessFlow:
    """Flux métier (use case)"""
    name: str
    description: str
    steps: List[Dict[str, Any]]       # Étapes ordonnées du flux
    actors: List[str]                 # Entités impliquées
    preconditions: List[str]
    postconditions: List[str]
    error_scenarios: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "actors": self.actors,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "error_scenarios": self.error_scenarios,
        }


@dataclass
class ArchitectureDNA:
    """
    Représentation unique et complète d'un projet.
    
    L'Architecture DNA encode :
    - Le domaine métier (entités, relations)
    - Les flux métier (use cases)
    - Les patterns architecturaux
    - Les contraintes techniques
    
    Cette représentation est la "sauce secrète" - elle capture
    l'essence du projet d'une manière structurée et unique.
    """
    id: str
    project_name: str
    domain_description: str
    
    # Domain Model
    entities: List[DomainEntity] = field(default_factory=list)
    relations: List[EntityRelation] = field(default_factory=list)
    
    # Business Logic
    flows: List[BusinessFlow] = field(default_factory=list)
    business_rules: List[str] = field(default_factory=list)
    
    # Architecture
    patterns: List[PatternType] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)  # ["presentation", "application", "domain", "infrastructure"]
    
    # Technical
    language: str = "python"
    framework: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)  # {"package": "version"}
    
    # Metadata
    complexity_level: float = 0.5  # 0-1
    estimated_loc: int = 0         # Lines of code estimées
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_dna_hash()
    
    def _generate_dna_hash(self) -> str:
        """Génère un hash unique basé sur le contenu"""
        content = f"{self.project_name}{self.domain_description}"
        content += "".join(e.name for e in self.entities)
        content += "".join(f.name for f in self.flows)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def entity_graph(self) -> Dict[str, List[str]]:
        """Graphe des dépendances entre entités"""
        graph = {e.name: [] for e in self.entities}
        for rel in self.relations:
            if rel.source in graph:
                graph[rel.source].append(rel.target)
        return graph
    
    def topological_order(self) -> List[str]:
        """Ordre topologique des entités (pour génération ordonnée)"""
        graph = self.entity_graph
        in_degree = {name: 0 for name in graph}
        
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in graph.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return result
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "project_name": self.project_name,
            "domain_description": self.domain_description,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "flows": [f.to_dict() for f in self.flows],
            "business_rules": self.business_rules,
            "patterns": [p.value for p in self.patterns],
            "layers": self.layers,
            "language": self.language,
            "framework": self.framework,
            "dependencies": self.dependencies,
            "complexity_level": self.complexity_level,
            "estimated_loc": self.estimated_loc,
        }


# =============================================================================
# COGNITIVE BLUEPRINT - Plan de génération cognitif
# =============================================================================

@dataclass
class CognitiveStep:
    """
    Étape cognitive dans le processus de génération.
    
    Représente une "pensée" de l'architecte :
    - Ce qu'il décide
    - Pourquoi
    - Comment ça se traduit en code
    """
    id: str
    thought: str              # "Je dois créer un modèle User avec authentification"
    reasoning: str            # "Car le e-commerce nécessite des comptes utilisateurs"
    decision: str             # "J'utilise le pattern Repository pour l'accès données"
    code_component: str       # Type de code à générer
    dependencies: List[str] = field(default_factory=list)  # IDs des steps précédents
    estimated_duration_seconds: int = 30
    
    # Pour la synchronisation narration
    narration_cue: str = ""   # Phrase qui déclenche cette étape
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "thought": self.thought,
            "reasoning": self.reasoning,
            "decision": self.decision,
            "code_component": self.code_component,
            "dependencies": self.dependencies,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "narration_cue": self.narration_cue,
        }


@dataclass
class CognitiveBlueprint:
    """
    Plan de génération basé sur le raisonnement cognitif.
    
    C'est le "plan de pensée" de l'agent - comment il va
    aborder le problème étape par étape, comme un expert humain.
    """
    id: str
    dna_id: str               # Référence à l'Architecture DNA
    
    # Phases de réflexion
    analysis_phase: List[CognitiveStep] = field(default_factory=list)
    design_phase: List[CognitiveStep] = field(default_factory=list)
    implementation_phase: List[CognitiveStep] = field(default_factory=list)
    validation_phase: List[CognitiveStep] = field(default_factory=list)
    
    # Timing
    total_duration_seconds: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:12]
        self._calculate_duration()
    
    def _calculate_duration(self):
        all_steps = (
            self.analysis_phase + 
            self.design_phase + 
            self.implementation_phase + 
            self.validation_phase
        )
        self.total_duration_seconds = sum(s.estimated_duration_seconds for s in all_steps)
    
    @property
    def all_steps(self) -> List[CognitiveStep]:
        return (
            self.analysis_phase + 
            self.design_phase + 
            self.implementation_phase + 
            self.validation_phase
        )
    
    def get_step_by_id(self, step_id: str) -> Optional[CognitiveStep]:
        for step in self.all_steps:
            if step.id == step_id:
                return step
        return None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "dna_id": self.dna_id,
            "analysis_phase": [s.to_dict() for s in self.analysis_phase],
            "design_phase": [s.to_dict() for s in self.design_phase],
            "implementation_phase": [s.to_dict() for s in self.implementation_phase],
            "validation_phase": [s.to_dict() for s in self.validation_phase],
            "total_duration_seconds": self.total_duration_seconds,
        }


# =============================================================================
# CODE SEGMENT - Unité de code générée
# =============================================================================

@dataclass
class CodeSegment:
    """
    Segment de code généré avec métadonnées de synchronisation.
    """
    id: str
    cognitive_step_id: str    # Lien vers l'étape cognitive
    
    # Code
    code: str
    language: str
    filename: str
    component_type: ComponentType
    
    # Pédagogie
    explanation: str          # Explication pour le narrateur
    key_concepts: List[str]   # Concepts illustrés
    common_mistakes: List[str] = field(default_factory=list)  # Erreurs à éviter
    
    # Synchronisation
    display_order: int = 0
    duration_seconds: int = 30
    narration_script: str = ""
    
    # Validation
    is_executable: bool = True
    expected_output: str = ""
    validation_command: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
    
    @property
    def line_count(self) -> int:
        return len(self.code.strip().split('\n'))
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "cognitive_step_id": self.cognitive_step_id,
            "code": self.code,
            "language": self.language,
            "filename": self.filename,
            "component_type": self.component_type.value,
            "explanation": self.explanation,
            "key_concepts": self.key_concepts,
            "common_mistakes": self.common_mistakes,
            "display_order": self.display_order,
            "duration_seconds": self.duration_seconds,
            "narration_script": self.narration_script,
            "is_executable": self.is_executable,
            "expected_output": self.expected_output,
            "line_count": self.line_count,
        }


# =============================================================================
# EXECUTION RESULT - Résultat d'exécution
# =============================================================================

@dataclass
class ExecutionResult:
    """Résultat de l'exécution d'un segment de code"""
    segment_id: str
    success: bool
    output: str
    error: str = ""
    execution_time_ms: int = 0
    screenshots: List[str] = field(default_factory=list)  # Paths vers captures
    
    def to_dict(self) -> Dict:
        return {
            "segment_id": self.segment_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "screenshots": self.screenshots,
        }


# =============================================================================
# NEXUS REQUEST & RESPONSE
# =============================================================================

@dataclass
class NexusRequest:
    """Requête de génération de code"""
    # Contexte
    project_description: str     # "une plateforme e-commerce"
    lesson_context: str          # Contexte de la leçon
    skill_level: str             # "beginner", "intermediate", "advanced", "expert"
    
    # Configuration
    language: str = "python"
    target_audience: TargetAudience = TargetAudience.STUDENT
    verbosity: CodeVerbosity = CodeVerbosity.STANDARD
    execution_mode: ExecutionMode = ExecutionMode.SANDBOX
    
    # Contraintes temporelles
    allocated_time_seconds: int = 300  # Temps alloué pour cette partie
    max_segments: int = 10
    
    # Options
    show_mistakes: bool = True         # Montrer les erreurs courantes
    show_evolution: bool = False       # Montrer v1 → v2 → v3
    include_tests: bool = False
    
    # Metadata
    request_id: str = ""
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())[:12]
    
    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "project_description": self.project_description,
            "lesson_context": self.lesson_context,
            "skill_level": self.skill_level,
            "language": self.language,
            "target_audience": self.target_audience.value,
            "verbosity": self.verbosity.value,
            "execution_mode": self.execution_mode.value,
            "allocated_time_seconds": self.allocated_time_seconds,
            "max_segments": self.max_segments,
            "show_mistakes": self.show_mistakes,
            "show_evolution": self.show_evolution,
            "include_tests": self.include_tests,
        }


@dataclass
class NexusResponse:
    """Réponse complète de NEXUS"""
    request_id: str
    
    # Artefacts générés
    architecture_dna: ArchitectureDNA
    cognitive_blueprint: CognitiveBlueprint
    code_segments: List[CodeSegment]
    execution_results: List[ExecutionResult] = field(default_factory=list)
    
    # Métadonnées
    total_duration_seconds: int = 0
    total_lines_of_code: int = 0
    generation_time_ms: int = 0
    
    # Pour l'assembleur vidéo
    sync_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        self.total_duration_seconds = sum(s.duration_seconds for s in self.code_segments)
        self.total_lines_of_code = sum(s.line_count for s in self.code_segments)
    
    def get_segments_ordered(self) -> List[CodeSegment]:
        """Retourne les segments dans l'ordre d'affichage"""
        return sorted(self.code_segments, key=lambda s: s.display_order)
    
    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "architecture_dna": self.architecture_dna.to_dict(),
            "cognitive_blueprint": self.cognitive_blueprint.to_dict(),
            "code_segments": [s.to_dict() for s in self.get_segments_ordered()],
            "execution_results": [r.to_dict() for r in self.execution_results],
            "total_duration_seconds": self.total_duration_seconds,
            "total_lines_of_code": self.total_lines_of_code,
            "generation_time_ms": self.generation_time_ms,
            "sync_metadata": self.sync_metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
