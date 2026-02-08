"""
Code Pipeline Models

Modèles pour la spécification et génération de code cohérent avec le voiceover.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pydantic import BaseModel


class CodeLanguage(str, Enum):
    """Langages supportés pour la génération de code"""
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    CSHARP = "csharp"
    KOTLIN = "kotlin"
    SCALA = "scala"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    CPP = "cpp"
    C = "c"
    SQL = "sql"
    BASH = "bash"
    PSEUDOCODE = "pseudocode"


class CodePurpose(str, Enum):
    """Type de code à générer"""
    TRANSFORMER = "transformer"        # Transformation de données
    VALIDATOR = "validator"            # Validation de données
    PROCESSOR = "processor"            # Traitement de données
    CALCULATOR = "calculator"          # Calcul/algorithme
    CONNECTOR = "connector"            # Connexion à un service
    HANDLER = "handler"                # Gestionnaire d'événements
    PARSER = "parser"                  # Parsing de données
    SERIALIZER = "serializer"          # Sérialisation
    ALGORITHM = "algorithm"            # Algorithme générique
    PATTERN_DEMO = "pattern_demo"      # Démonstration d'un pattern
    API_CLIENT = "api_client"          # Client API
    DATA_STRUCTURE = "data_structure"  # Structure de données


class TechnologyEcosystem(str, Enum):
    """Écosystème technologique - contexte où le code s'exécute"""
    # Messaging & Streaming
    KAFKA = "kafka"                    # Apache Kafka (Connect, Streams, etc.)
    RABBITMQ = "rabbitmq"              # RabbitMQ
    PULSAR = "pulsar"                  # Apache Pulsar
    ACTIVEMQ = "activemq"              # ActiveMQ

    # ESB & Integration
    MULESOFT = "mulesoft"              # MuleSoft Anypoint
    TALEND = "talend"                  # Talend
    APACHE_CAMEL = "apache_camel"      # Apache Camel
    SPRING_INTEGRATION = "spring_integration"  # Spring Integration
    BOOMI = "boomi"                    # Dell Boomi
    INFORMATICA = "informatica"        # Informatica

    # Cloud
    AWS = "aws"                        # AWS (Lambda, Step Functions, etc.)
    GCP = "gcp"                        # Google Cloud
    AZURE = "azure"                    # Microsoft Azure

    # Frameworks
    SPRING = "spring"                  # Spring Framework (Boot, MVC, etc.)
    QUARKUS = "quarkus"                # Quarkus
    MICRONAUT = "micronaut"            # Micronaut
    DJANGO = "django"                  # Django
    FASTAPI = "fastapi"                # FastAPI
    EXPRESS = "express"                # Express.js
    NESTJS = "nestjs"                  # NestJS

    # Data Processing
    SPARK = "spark"                    # Apache Spark
    FLINK = "flink"                    # Apache Flink
    BEAM = "beam"                      # Apache Beam
    AIRFLOW = "airflow"                # Apache Airflow
    DBT = "dbt"                        # dbt

    # Databases
    POSTGRESQL = "postgresql"          # PostgreSQL
    MONGODB = "mongodb"                # MongoDB
    REDIS = "redis"                    # Redis
    ELASTICSEARCH = "elasticsearch"    # Elasticsearch

    # Kubernetes & Containers
    KUBERNETES = "kubernetes"          # Kubernetes
    DOCKER = "docker"                  # Docker

    # Standalone
    STANDALONE = "standalone"          # Code standalone, pas de framework


@dataclass
class TechnologyContext:
    """
    Contexte technologique extrait du voiceover.

    Permet de situer l'utilisateur dans l'écosystème correct.
    Ex: Un "transformer" Kafka Connect ≠ un "transformer" MuleSoft
    """
    # Écosystème principal
    ecosystem: TechnologyEcosystem

    # Composant spécifique dans l'écosystème
    component: str                     # Ex: "Kafka Connect", "Kafka Streams", "Lambda"

    # Version si mentionnée
    version: Optional[str] = None      # Ex: "Kafka 3.x", "Spring Boot 3"

    # Pattern architectural
    architecture_pattern: Optional[str] = None  # Ex: "EIP", "CQRS", "Event Sourcing"

    # APIs/interfaces spécifiques à utiliser
    required_apis: List[str] = field(default_factory=list)  # Ex: ["Transformation", "SourceConnector"]

    # Imports/dépendances implicites
    implicit_imports: List[str] = field(default_factory=list)  # Ex: ["org.apache.kafka.connect.transforms"]

    # Conventions de nommage du contexte
    naming_conventions: Dict[str, str] = field(default_factory=dict)  # Ex: {"class": "PascalCase", "method": "camelCase"}

    # Description pour l'utilisateur
    context_description: str = ""      # Ex: "Dans le contexte de Kafka Connect, un transformer..."


@dataclass
class ExampleIO:
    """Exemple d'entrée/sortie pour le code"""
    input_value: str                   # Valeur d'entrée
    input_description: str             # Description de l'entrée
    expected_output: str               # Sortie attendue
    output_description: str            # Description de la sortie

    # Pour affichage console
    input_display: Optional[str] = None   # Comment afficher l'input
    output_display: Optional[str] = None  # Comment afficher l'output


@dataclass
class CodeSpec:
    """
    Spécification de code extraite du voiceover/concept.

    C'est le CONTRAT entre ce qui est expliqué et ce qui sera généré.
    Maestro garantit que ce contrat est respecté.
    """
    # Identification (required)
    spec_id: str
    concept_name: str                  # Ex: "Pattern Transformer"

    # Langage et type (required)
    language: CodeLanguage
    purpose: CodePurpose

    # Description fonctionnelle (required)
    description: str                   # Ce que le code fait
    input_type: str                    # Type d'entrée (ex: "XML string")
    output_type: str                   # Type de sortie (ex: "JSON string")

    # Opérations clés (required)
    key_operations: List[str]          # Ex: ["parse XML", "build JSON", "serialize"]

    # CONTEXTE TECHNOLOGIQUE (optional - CRITIQUE pour la cohérence)
    # Un transformer Kafka Connect ≠ un transformer MuleSoft ≠ un transformer standalone
    context: Optional[TechnologyContext] = None

    # Contraintes (optional with defaults)
    must_include: List[str] = field(default_factory=list)    # Éléments obligatoires
    must_not_include: List[str] = field(default_factory=list)  # Éléments interdits

    # Exemple I/O (optional)
    example_io: Optional[ExampleIO] = None

    # Contexte pédagogique (optional with defaults)
    voiceover_excerpt: str = ""        # Extrait du voiceover qui décrit ce code
    pedagogical_goal: str = ""         # Objectif pédagogique

    # Métadonnées (optional with defaults)
    complexity_level: str = "intermediate"  # beginner, intermediate, advanced
    estimated_lines: int = 20          # Nombre de lignes estimé

    # Validation (optional with defaults)
    is_validated: bool = False
    validation_notes: List[str] = field(default_factory=list)


@dataclass
class GeneratedCode:
    """Code généré à partir d'une spec"""
    spec_id: str
    language: CodeLanguage

    # Code
    code: str                          # Code complet
    highlighted_lines: List[int] = field(default_factory=list)  # Lignes importantes

    # Documentation inline
    comments: Dict[int, str] = field(default_factory=dict)  # line_number -> comment

    # Pour console
    runnable: bool = False             # Peut être exécuté
    main_function: Optional[str] = None  # Point d'entrée
    dependencies: List[str] = field(default_factory=list)  # Imports nécessaires

    # Validation
    matches_spec: bool = False
    spec_violations: List[str] = field(default_factory=list)


@dataclass
class ConsoleExecution:
    """Résultat d'exécution console"""
    spec_id: str

    # Exécution
    input_shown: str                   # Input affiché
    output_shown: str                  # Output affiché
    execution_time_ms: float = 0

    # Validation
    matches_expected: bool = False
    difference_notes: List[str] = field(default_factory=list)

    # Affichage
    formatted_console: str = ""        # Console formatée pour slide


@dataclass
class CodeSyntaxError:
    """Erreur de syntaxe détectée dans le code"""
    line: int                          # Numéro de ligne
    column: int                        # Numéro de colonne
    message: str                       # Message d'erreur
    severity: str = "error"            # "error" | "warning"


@dataclass
class SyntaxValidationResult:
    """Résultat de la validation syntaxique du code"""
    is_valid: bool                     # Code syntaxiquement valide
    language: str                      # Langage validé
    errors: List[CodeSyntaxError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrected_code: Optional[str] = None  # Code corrigé si auto-correction appliquée
    correction_applied: bool = False   # Auto-correction a été appliquée
    validation_method: str = "ast"     # "ast", "regex", "llm"


@dataclass
class SummarizedCode:
    """Code résumé pour l'affichage sur slide"""
    display_code: str                  # Code pour l'affichage slide (max_lines)
    full_code: str                     # Code complet original
    lines_removed: int = 0             # Nombre de lignes supprimées
    summary_strategy: str = ""         # "imports", "comments", "docstrings", "ellipsis"
    key_sections_preserved: List[str] = field(default_factory=list)  # Sections importantes gardées


@dataclass
class CodeSlidePackage:
    """Package complet pour créer les slides de code"""
    spec: CodeSpec
    generated_code: GeneratedCode
    console_execution: Optional[ConsoleExecution] = None

    # Slides à générer
    slides: List[Dict[str, Any]] = field(default_factory=list)

    # Voiceover adapté
    code_voiceover: str = ""           # Voiceover pour le slide de code
    console_voiceover: str = ""        # Voiceover pour le slide console

    # Validation globale
    is_coherent: bool = False
    coherence_score: float = 0.0
    coherence_issues: List[str] = field(default_factory=list)

    # NOUVEAU: Validation syntaxique (Phase 1 - SyntaxVerifier)
    syntax_validated: bool = False     # Syntaxe validée
    syntax_errors: List[str] = field(default_factory=list)  # Erreurs de syntaxe

    # NOUVEAU: Code résumé (Phase 1 - CodeSummarizer)
    display_code: Optional[str] = None  # Code pour slide (max 25-30 lignes)
    full_code: Optional[str] = None     # Code complet exécutable


# Pydantic models pour API

class CodeSpecRequest(BaseModel):
    """Requête pour extraire une spec de code"""
    voiceover_text: str
    concept_name: str
    preferred_language: Optional[str] = None
    audience_level: str = "intermediate"
    content_language: str = "fr"


class CodeSpecResponse(BaseModel):
    """Réponse avec la spec extraite"""
    success: bool
    spec_id: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GenerateCodeRequest(BaseModel):
    """Requête pour générer du code depuis une spec"""
    spec_id: str
    spec: Dict[str, Any]
    include_comments: bool = True
    optimize_for_display: bool = True  # Optimiser pour affichage slide


class GenerateCodeResponse(BaseModel):
    """Réponse avec le code généré"""
    success: bool
    code: Optional[str] = None
    highlighted_lines: List[int] = []
    runnable: bool = False
    error: Optional[str] = None


class ExecuteCodeRequest(BaseModel):
    """Requête pour exécuter du code"""
    code: str
    language: str
    input_value: Optional[str] = None
    timeout_seconds: int = 10


class ExecuteCodeResponse(BaseModel):
    """Réponse d'exécution"""
    success: bool
    output: Optional[str] = None
    execution_time_ms: float = 0
    error: Optional[str] = None
