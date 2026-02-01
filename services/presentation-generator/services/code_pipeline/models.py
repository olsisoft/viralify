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
    """
    # Identification
    spec_id: str
    concept_name: str                  # Ex: "Pattern Transformer"

    # Langage et type
    language: CodeLanguage
    purpose: CodePurpose

    # Description fonctionnelle
    description: str                   # Ce que le code fait
    input_type: str                    # Type d'entrée (ex: "XML string")
    output_type: str                   # Type de sortie (ex: "JSON string")

    # Opérations clés (visibles dans le code)
    key_operations: List[str]          # Ex: ["parse XML", "build JSON", "serialize"]

    # Contraintes
    must_include: List[str] = field(default_factory=list)    # Éléments obligatoires
    must_not_include: List[str] = field(default_factory=list)  # Éléments interdits

    # Exemple I/O
    example_io: Optional[ExampleIO] = None

    # Contexte pédagogique
    voiceover_excerpt: str = ""        # Extrait du voiceover qui décrit ce code
    pedagogical_goal: str = ""         # Objectif pédagogique

    # Métadonnées
    complexity_level: str = "intermediate"  # beginner, intermediate, advanced
    estimated_lines: int = 20          # Nombre de lignes estimé

    # Validation
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
