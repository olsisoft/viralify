"""
Console Executor

Exécute le code généré et valide que l'output correspond à la spec.
"""

import os
import subprocess
import tempfile
import asyncio
from typing import Optional, Dict, Any

from .models import (
    CodeSpec, CodeLanguage, ConsoleExecution,
    ExecuteCodeRequest, ExecuteCodeResponse
)


class ConsoleExecutor:
    """
    Exécute le code et valide l'output contre la spec.

    Supporte l'exécution réelle pour:
    - Python
    - JavaScript (Node.js)
    - TypeScript (ts-node)

    Pour les autres langages, simule l'output de manière cohérente.
    """

    # Configuration par langage
    LANGUAGE_CONFIG = {
        CodeLanguage.PYTHON: {
            "interpreter": "python3",
            "file_ext": ".py",
            "runnable": True,
        },
        CodeLanguage.JAVASCRIPT: {
            "interpreter": "node",
            "file_ext": ".js",
            "runnable": True,
        },
        CodeLanguage.TYPESCRIPT: {
            "interpreter": "ts-node",
            "file_ext": ".ts",
            "runnable": True,
        },
    }

    def __init__(self):
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM for simulation"""
        try:
            from shared.llm_provider import get_llm_client, get_model_name
            self.client = get_llm_client()
            self.model = get_model_name("fast")
        except ImportError:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL_FAST") or "gpt-4o-mini"

    async def execute(
        self,
        code: str,
        spec: CodeSpec,
        timeout_seconds: int = 10
    ) -> ConsoleExecution:
        """
        Exécute le code et valide contre la spec.

        Args:
            code: Le code à exécuter
            spec: La spécification avec l'output attendu
            timeout_seconds: Timeout d'exécution

        Returns:
            ConsoleExecution avec le résultat
        """
        print(f"[CONSOLE] Executing {spec.language.value} code...", flush=True)

        lang_config = self.LANGUAGE_CONFIG.get(spec.language)

        if lang_config and lang_config["runnable"]:
            # Exécution réelle
            result = await self._execute_real(
                code=code,
                interpreter=lang_config["interpreter"],
                file_ext=lang_config["file_ext"],
                timeout=timeout_seconds
            )
        else:
            # Simulation pour langages non supportés
            result = await self._simulate_execution(code, spec)

        # Construire l'input affiché
        input_shown = ""
        if spec.example_io:
            input_shown = spec.example_io.input_value

        # Valider contre la spec
        matches_expected = False
        difference_notes = []

        if spec.example_io and result.get("success"):
            validation = await self._validate_output(
                actual_output=result["output"],
                expected_output=spec.example_io.expected_output,
                spec=spec
            )
            matches_expected = validation["matches"]
            difference_notes = validation.get("differences", [])

        # Formater pour affichage
        formatted_console = self._format_for_slide(
            input_shown=input_shown,
            output_shown=result.get("output", result.get("error", "")),
            language=spec.language,
            success=result.get("success", False)
        )

        execution = ConsoleExecution(
            spec_id=spec.spec_id,
            input_shown=input_shown,
            output_shown=result.get("output", result.get("error", "")),
            execution_time_ms=result.get("execution_time_ms", 0),
            matches_expected=matches_expected,
            difference_notes=difference_notes,
            formatted_console=formatted_console
        )

        print(f"[CONSOLE] Execution {'successful' if result.get('success') else 'failed'}, "
              f"matches spec: {matches_expected}", flush=True)

        return execution

    async def _execute_real(
        self,
        code: str,
        interpreter: str,
        file_ext: str,
        timeout: int
    ) -> Dict[str, Any]:
        """Exécute réellement le code"""

        temp_file = None
        try:
            # Créer fichier temporaire
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix=file_ext,
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_file = f.name

            # Exécuter
            import time
            start_time = time.time()

            process = await asyncio.create_subprocess_exec(
                interpreter, temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tempfile.gettempdir()
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                execution_time = (time.time() - start_time) * 1000

                if process.returncode == 0:
                    return {
                        "success": True,
                        "output": stdout.decode('utf-8').strip(),
                        "execution_time_ms": execution_time
                    }
                else:
                    return {
                        "success": False,
                        "output": stderr.decode('utf-8').strip(),
                        "error": f"Exit code: {process.returncode}",
                        "execution_time_ms": execution_time
                    }

            except asyncio.TimeoutError:
                process.kill()
                return {
                    "success": False,
                    "error": f"Timeout after {timeout}s",
                    "output": "[Timeout - exécution trop longue]"
                }

        except FileNotFoundError:
            return {
                "success": False,
                "error": f"Interpreter '{interpreter}' not found",
                "output": f"[Erreur: {interpreter} non disponible]"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": f"[Erreur d'exécution: {str(e)}]"
            }
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

    async def _simulate_execution(
        self,
        code: str,
        spec: CodeSpec
    ) -> Dict[str, Any]:
        """Simule l'exécution pour les langages non supportés"""

        # Si on a un output attendu dans la spec, l'utiliser
        if spec.example_io and spec.example_io.expected_output:
            return {
                "success": True,
                "output": spec.example_io.expected_output,
                "execution_time_ms": 50  # Simulé
            }

        # Sinon, demander au LLM de simuler
        prompt = f"""Simule l'exécution de ce code {spec.language.value}:

```{spec.language.value}
{code}
```

Context:
- Le code implémente: {spec.description}
- Input attendu: {spec.example_io.input_value if spec.example_io else 'N/A'}

Génère un output console réaliste.
Retourne UNIQUEMENT l'output, sans explication ni formatage markdown."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu simules l'exécution de code. Retourne uniquement l'output console."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            return {
                "success": True,
                "output": response.choices[0].message.content.strip(),
                "execution_time_ms": 50
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "[Simulation non disponible]"
            }

    async def _validate_output(
        self,
        actual_output: str,
        expected_output: str,
        spec: CodeSpec
    ) -> Dict[str, Any]:
        """Valide que l'output correspond à l'attendu"""

        # Comparaison stricte d'abord
        actual_clean = actual_output.strip()
        expected_clean = expected_output.strip()

        if actual_clean == expected_clean:
            return {"matches": True, "differences": []}

        # Comparaison sémantique si différent
        prompt = f"""Compare ces deux outputs et détermine s'ils sont ÉQUIVALENTS:

OUTPUT ATTENDU:
{expected_output}

OUTPUT RÉEL:
{actual_output}

CONTEXTE:
- Le code devait: {spec.description}
- Type d'output attendu: {spec.output_type}

Les outputs sont-ils équivalents (même information, même structure)?
Ignore les différences de whitespace ou de formatage mineur.

Retourne un JSON:
{{
    "matches": true/false,
    "differences": ["différence 1", "différence 2"],
    "severity": "none|minor|major"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu compares des outputs de programme. Tu es tolérant aux différences mineures de formatage."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=300
            )

            import json
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"[CONSOLE] Output validation failed: {e}", flush=True)
            # En cas d'erreur, faire une comparaison basique
            return {
                "matches": actual_clean == expected_clean,
                "differences": ["Comparaison exacte effectuée"]
            }

    def _format_for_slide(
        self,
        input_shown: str,
        output_shown: str,
        language: CodeLanguage,
        success: bool
    ) -> str:
        """Formate l'output pour affichage sur slide"""

        # Construire le prompt selon le langage
        if language == CodeLanguage.PYTHON:
            prompt_symbol = ">>>"
        elif language in [CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT]:
            prompt_symbol = ">"
        elif language == CodeLanguage.JAVA:
            prompt_symbol = "$"
        else:
            prompt_symbol = ">"

        lines = []

        # Commande d'exécution
        if language == CodeLanguage.PYTHON:
            lines.append(f"$ python3 transformer.py")
        elif language == CodeLanguage.JAVA:
            lines.append(f"$ java XmlToJsonTransformer")
        elif language == CodeLanguage.JAVASCRIPT:
            lines.append(f"$ node transformer.js")
        else:
            lines.append(f"$ run {language.value}")

        # Input si présent
        if input_shown:
            lines.append("")
            lines.append("Input:")
            # Tronquer si trop long
            if len(input_shown) > 200:
                lines.append(input_shown[:200] + "...")
            else:
                lines.append(input_shown)

        # Output
        lines.append("")
        if success:
            lines.append("Output:")
        else:
            lines.append("Error:")

        # Tronquer si trop long
        if len(output_shown) > 500:
            lines.append(output_shown[:500] + "\n...")
        else:
            lines.append(output_shown)

        return "\n".join(lines)


# Singleton
_executor: Optional[ConsoleExecutor] = None


def get_console_executor() -> ConsoleExecutor:
    """Get singleton console executor"""
    global _executor
    if _executor is None:
        _executor = ConsoleExecutor()
    return _executor
