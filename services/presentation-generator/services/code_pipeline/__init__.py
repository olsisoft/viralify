"""
Code Pipeline - Génération de code cohérente avec le voiceover

Ce module garantit que le code généré est cohérent avec ce qui est expliqué
dans le voiceover. Il utilise Maestro pour extraire une spécification (CodeSpec)
du voiceover, puis génère du code qui respecte strictement cette spec.

Flow:
    Voiceover → Spec → Code → Console → Slides

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
"""

# Models
from .models import (
    # Enums
    CodeLanguage,
    CodePurpose,
    # Data classes
    ExampleIO,
    CodeSpec,
    GeneratedCode,
    ConsoleExecution,
    CodeSlidePackage,
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
    # Data classes
    "ExampleIO",
    "CodeSpec",
    "GeneratedCode",
    "ConsoleExecution",
    "CodeSlidePackage",
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
    # Pipeline
    "CodePipeline",
    "CodePipelineResult",
    "get_code_pipeline",
]
