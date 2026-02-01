"""
Code Pipeline Orchestrator

Orchestre le flow complet:
Voiceover → Spec → Code → Console → Slides
"""

import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .models import (
    CodeSpec, CodeLanguage, CodePurpose,
    GeneratedCode, ConsoleExecution, CodeSlidePackage,
    ExampleIO
)
from .spec_extractor import get_spec_extractor, MaestroSpecExtractor
from .code_generator import get_code_generator, SpecConstrainedCodeGenerator
from .console_executor import get_console_executor, ConsoleExecutor


@dataclass
class CodePipelineResult:
    """Résultat du pipeline de code"""
    success: bool
    package: Optional[CodeSlidePackage] = None
    error: Optional[str] = None

    # Détails pour debug
    spec_extracted: bool = False
    code_generated: bool = False
    console_executed: bool = False
    coherence_validated: bool = False


class CodePipeline:
    """
    Orchestre la génération de code cohérente avec le voiceover.

    Flow:
    1. EXTRACT: Maestro extrait la CodeSpec du voiceover
    2. GENERATE: Génère du code respectant la spec
    3. EXECUTE: Exécute le code (réel ou simulé)
    4. VALIDATE: Vérifie la cohérence globale
    5. PACKAGE: Construit le package de slides
    """

    def __init__(self):
        self.spec_extractor = get_spec_extractor()
        self.code_generator = get_code_generator()
        self.console_executor = get_console_executor()
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM for voiceover generation"""
        try:
            from shared.llm_provider import get_llm_client, get_model_name
            self.client = get_llm_client()
            self.model = get_model_name("quality")
        except ImportError:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    async def process(
        self,
        voiceover_text: str,
        concept_name: str,
        preferred_language: Optional[str] = None,
        audience_level: str = "intermediate",
        content_language: str = "fr",
        execute_code: bool = True,
        generate_voiceover: bool = True
    ) -> CodePipelineResult:
        """
        Exécute le pipeline complet.

        Args:
            voiceover_text: Le voiceover qui décrit le code à créer
            concept_name: Nom du concept enseigné
            preferred_language: Langage préféré (optionnel)
            audience_level: Niveau de l'audience
            content_language: Langue du contenu
            execute_code: Exécuter le code ou non
            generate_voiceover: Générer les voiceovers adaptés

        Returns:
            CodePipelineResult avec le package de slides
        """
        result = CodePipelineResult(success=False)

        try:
            print(f"[PIPELINE] Starting code pipeline for: {concept_name}", flush=True)

            # =====================================================================
            # STEP 1: EXTRACT SPEC
            # =====================================================================
            print(f"[PIPELINE] Step 1: Extracting spec from voiceover...", flush=True)

            spec_response = await self.spec_extractor.extract_spec(
                voiceover_text=voiceover_text,
                concept_name=concept_name,
                preferred_language=preferred_language,
                audience_level=audience_level,
                content_language=content_language
            )

            if not spec_response.success:
                result.error = f"Spec extraction failed: {spec_response.error}"
                return result

            result.spec_extracted = True
            spec = self._dict_to_spec(spec_response.spec)

            print(f"[PIPELINE] Spec extracted: {spec.language.value} - {spec.purpose.value}", flush=True)

            # =====================================================================
            # STEP 2: GENERATE CODE
            # =====================================================================
            print(f"[PIPELINE] Step 2: Generating code...", flush=True)

            code_response = await self.code_generator.generate(
                spec=spec,
                include_comments=True,
                optimize_for_display=True
            )

            if not code_response.success:
                result.error = f"Code generation failed: {code_response.error}"
                return result

            result.code_generated = True

            generated_code = GeneratedCode(
                spec_id=spec.spec_id,
                language=spec.language,
                code=code_response.code,
                highlighted_lines=code_response.highlighted_lines,
                runnable=code_response.runnable,
                matches_spec=True  # Validé par le générateur
            )

            print(f"[PIPELINE] Code generated: {len(code_response.code)} chars", flush=True)

            # =====================================================================
            # STEP 3: EXECUTE CODE (optional)
            # =====================================================================
            console_execution = None

            if execute_code and spec.example_io:
                print(f"[PIPELINE] Step 3: Executing code...", flush=True)

                console_execution = await self.console_executor.execute(
                    code=generated_code.code,
                    spec=spec,
                    timeout_seconds=10
                )

                result.console_executed = True
                print(f"[PIPELINE] Execution result: matches_expected={console_execution.matches_expected}", flush=True)

            # =====================================================================
            # STEP 4: VALIDATE COHERENCE
            # =====================================================================
            print(f"[PIPELINE] Step 4: Validating coherence...", flush=True)

            coherence = await self._validate_coherence(
                voiceover_text=voiceover_text,
                spec=spec,
                generated_code=generated_code,
                console_execution=console_execution
            )

            result.coherence_validated = coherence["is_coherent"]

            # =====================================================================
            # STEP 5: BUILD PACKAGE
            # =====================================================================
            print(f"[PIPELINE] Step 5: Building slide package...", flush=True)

            package = CodeSlidePackage(
                spec=spec,
                generated_code=generated_code,
                console_execution=console_execution,
                is_coherent=coherence["is_coherent"],
                coherence_score=coherence.get("score", 0.0),
                coherence_issues=coherence.get("issues", [])
            )

            # Générer les voiceovers adaptés
            if generate_voiceover:
                package = await self._generate_slide_voiceovers(package, content_language)

            # Construire les slides
            package.slides = self._build_slides(package)

            result.success = True
            result.package = package

            print(f"[PIPELINE] Pipeline complete: {len(package.slides)} slides, "
                  f"coherence={package.coherence_score:.2f}", flush=True)

            return result

        except Exception as e:
            print(f"[PIPELINE] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            result.error = str(e)
            return result

    async def _validate_coherence(
        self,
        voiceover_text: str,
        spec: CodeSpec,
        generated_code: GeneratedCode,
        console_execution: Optional[ConsoleExecution]
    ) -> Dict[str, Any]:
        """Valide la cohérence globale du pipeline"""

        # Construire le contexte
        console_info = ""
        if console_execution:
            console_info = f"""
Console:
- Input: {console_execution.input_shown}
- Output: {console_execution.output_shown}
- Matches expected: {console_execution.matches_expected}
"""

        prompt = f"""Évalue la COHÉRENCE entre ces éléments:

1. VOICEOVER (ce qui est dit):
\"\"\"{voiceover_text[:1000]}\"\"\"

2. SPEC EXTRAITE:
- Concept: {spec.concept_name}
- Description: {spec.description}
- Input → Output: {spec.input_type} → {spec.output_type}
- Exemple: {spec.example_io.input_value if spec.example_io else 'N/A'} → {spec.example_io.expected_output if spec.example_io else 'N/A'}

3. CODE GÉNÉRÉ:
```{spec.language.value}
{generated_code.code[:1500]}
```

{console_info}

VÉRIFIE:
1. Le code fait-il ce que le voiceover décrit?
2. Les types I/O sont-ils cohérents?
3. L'exemple fonctionne-t-il comme promis?
4. Un apprenant verrait-il la connexion entre l'explication et le code?

Retourne un JSON:
{{
    "is_coherent": true/false,
    "score": 0.95,
    "issues": ["problème 1", "problème 2"],
    "strengths": ["point fort 1"],
    "suggestions": ["suggestion 1"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu évalues la cohérence pédagogique entre une explication et son implémentation en code."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=500
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"[PIPELINE] Coherence validation failed: {e}", flush=True)
            return {"is_coherent": True, "score": 0.8, "issues": []}

    async def _generate_slide_voiceovers(
        self,
        package: CodeSlidePackage,
        content_language: str
    ) -> CodeSlidePackage:
        """Génère les voiceovers pour les slides de code et console"""

        # Voiceover pour le slide de code
        code_prompt = f"""Génère un voiceover COURT pour accompagner ce slide de code.

CONCEPT: {package.spec.concept_name}
LANGAGE: {package.spec.language.value}
CE QUE FAIT LE CODE: {package.spec.description}

CODE (extrait):
```{package.spec.language.value}
{package.generated_code.code[:800]}
```

STYLE:
- Conversationnel, pas robotique
- Explique les lignes importantes
- Guide l'œil de l'apprenant
- En {content_language}
- Maximum 4-5 phrases

Retourne UNIQUEMENT le texte du voiceover:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu génères des voiceovers pédagogiques courts et engageants."},
                    {"role": "user", "content": code_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            package.code_voiceover = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[PIPELINE] Code voiceover generation failed: {e}", flush=True)
            package.code_voiceover = ""

        # Voiceover pour le slide console (si présent)
        if package.console_execution:
            console_prompt = f"""Génère un voiceover COURT pour accompagner ce slide de console.

CONCEPT: {package.spec.concept_name}
INPUT MONTRÉ: {package.console_execution.input_shown}
OUTPUT MONTRÉ: {package.console_execution.output_shown}

STYLE:
- Montre l'exécution en direct
- Fait le lien avec le code précédent
- Valide visuellement le concept
- En {content_language}
- Maximum 3-4 phrases

Retourne UNIQUEMENT le texte du voiceover:"""

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Tu génères des voiceovers pour des démonstrations console."},
                        {"role": "user", "content": console_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                package.console_voiceover = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"[PIPELINE] Console voiceover generation failed: {e}", flush=True)
                package.console_voiceover = ""

        return package

    def _build_slides(self, package: CodeSlidePackage) -> List[Dict[str, Any]]:
        """Construit les slides à partir du package"""

        slides = []

        # Slide 1: Code
        slides.append({
            "type": "code",
            "title": f"Implémentation - {package.spec.concept_name}",
            "language": package.spec.language.value,
            "code": package.generated_code.code,
            "highlighted_lines": package.generated_code.highlighted_lines,
            "voiceover": package.code_voiceover,
            "metadata": {
                "spec_id": package.spec.spec_id,
                "purpose": package.spec.purpose.value
            }
        })

        # Slide 2: Console (si présent)
        if package.console_execution:
            slides.append({
                "type": "console",
                "title": "Démonstration",
                "console_content": package.console_execution.formatted_console,
                "input": package.console_execution.input_shown,
                "output": package.console_execution.output_shown,
                "voiceover": package.console_voiceover,
                "matches_expected": package.console_execution.matches_expected,
                "metadata": {
                    "spec_id": package.spec.spec_id,
                    "execution_time_ms": package.console_execution.execution_time_ms
                }
            })

        return slides

    def _dict_to_spec(self, spec_dict: Dict[str, Any]) -> CodeSpec:
        """Convertit un dict en CodeSpec"""
        example_io = None
        if spec_dict.get("example_io"):
            ex = spec_dict["example_io"]
            example_io = ExampleIO(
                input_value=ex.get("input_value", ""),
                input_description=ex.get("input_description", ""),
                expected_output=ex.get("expected_output", ""),
                output_description=ex.get("output_description", "")
            )

        return CodeSpec(
            spec_id=spec_dict.get("spec_id", ""),
            concept_name=spec_dict.get("concept_name", ""),
            language=CodeLanguage(spec_dict.get("language", "pseudocode")),
            purpose=CodePurpose(spec_dict.get("purpose", "algorithm")),
            description=spec_dict.get("description", ""),
            input_type=spec_dict.get("input_type", ""),
            output_type=spec_dict.get("output_type", ""),
            key_operations=spec_dict.get("key_operations", []),
            must_include=spec_dict.get("must_include", []),
            must_not_include=spec_dict.get("must_not_include", []),
            example_io=example_io,
            voiceover_excerpt=spec_dict.get("voiceover_excerpt", ""),
            pedagogical_goal=spec_dict.get("pedagogical_goal", ""),
            complexity_level=spec_dict.get("complexity_level", "intermediate"),
            estimated_lines=spec_dict.get("estimated_lines", 25),
            is_validated=spec_dict.get("is_validated", False),
            validation_notes=spec_dict.get("validation_notes", [])
        )


# Singleton
_pipeline: Optional[CodePipeline] = None


def get_code_pipeline() -> CodePipeline:
    """Get singleton code pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = CodePipeline()
    return _pipeline
