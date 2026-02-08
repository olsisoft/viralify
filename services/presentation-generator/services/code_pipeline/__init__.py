"""
Code Pipeline - Génération de code cohérente avec le voiceover

Ce module garantit que le code généré est cohérent avec ce qui est expliqué
dans le voiceover. Il utilise Maestro pour extraire une spécification (CodeSpec)
du voiceover, puis génère du code qui respecte strictement cette spec.

Flow (7 étapes):
    1. EXTRACT: Voiceover → CodeSpec
    2. GENERATE: CodeSpec → Code
    3. VERIFY SYNTAX: Validation syntaxique + auto-correction
    4. EXECUTE: Exécution console (optionnel)
    5. VALIDATE: Vérification cohérence
    6. SUMMARIZE: Résumé pour affichage slide
    7. PACKAGE: Construction slides

Usage:
    from services.code_pipeline import get_code_pipeline, CodePipelineResult

    pipeline = get_code_pipeline()

    result = await pipeline.process(
        voiceover_text="Maintenant nous allons développer en Java un transformateur qui prend du XML pour transformer en JSON",
        concept_name="Pattern Transformer",
        preferred_language="java",
        audience_level="intermediate",
        content_language="fr"
    )

    if result.success:
        for slide in result.package.slides:
            print(f"Slide: {slide['type']} - {slide['title']}")
            print(f"Voiceover: {slide['voiceover']}")
            print(f"Code (display): {slide['code'][:100]}...")
"""

# Models
from .models import (
    # Enums
    CodeLanguage,
    CodePurpose,
    TechnologyEcosystem,
    # Data classes
    TechnologyContext,
    ExampleIO,
    CodeSpec,
    GeneratedCode,
    ConsoleExecution,
    CodeSlidePackage,
    # NOUVEAU: Syntax verification
    CodeSyntaxError,
    SyntaxValidationResult,
    # NOUVEAU: Code summarization
    SummarizedCode,
    # Request/Response models
    CodeSpecRequest,
    CodeSpecResponse,
    GenerateCodeRequest,
    GenerateCodeResponse,
    ExecuteCodeRequest,
    ExecuteCodeResponse,
)

# Components
from .spec_extractor import (
    MaestroSpecExtractor,
    get_spec_extractor,
)

from .code_generator import (
    SpecConstrainedCodeGenerator,
    get_code_generator,
)

from .console_executor import (
    ConsoleExecutor,
    get_console_executor,
)

# NOUVEAU: Syntax verification
from .syntax_verifier import (
    SyntaxVerifier,
    get_syntax_verifier,
)

# NOUVEAU: Code summarization
from .code_summarizer import (
    CodeSummarizer,
    get_code_summarizer,
)

# Pipeline
from .pipeline import (
    CodePipeline,
    CodePipelineResult,
    get_code_pipeline,
)


__all__ = [
    # Enums
    "CodeLanguage",
    "CodePurpose",
    "TechnologyEcosystem",
    # Data classes
    "TechnologyContext",
    "ExampleIO",
    "CodeSpec",
    "GeneratedCode",
    "ConsoleExecution",
    "CodeSlidePackage",
    # NOUVEAU: Syntax verification
    "CodeSyntaxError",
    "SyntaxValidationResult",
    # NOUVEAU: Code summarization
    "SummarizedCode",
    # Request/Response
    "CodeSpecRequest",
    "CodeSpecResponse",
    "GenerateCodeRequest",
    "GenerateCodeResponse",
    "ExecuteCodeRequest",
    "ExecuteCodeResponse",
    # Components
    "MaestroSpecExtractor",
    "get_spec_extractor",
    "SpecConstrainedCodeGenerator",
    "get_code_generator",
    "ConsoleExecutor",
    "get_console_executor",
    # NOUVEAU: Syntax verification
    "SyntaxVerifier",
    "get_syntax_verifier",
    # NOUVEAU: Code summarization
    "CodeSummarizer",
    "get_code_summarizer",
    # Pipeline
    "CodePipeline",
    "CodePipelineResult",
    "get_code_pipeline",
]
