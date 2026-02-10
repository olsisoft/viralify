"""
Code Generator

Génère du code à partir d'une CodeSpec, en respectant strictement le contrat.
"""

import os
import json
from typing import Optional, Dict, Any, List

from .models import (
    CodeSpec, CodeLanguage, CodePurpose,
    GeneratedCode, GenerateCodeRequest, GenerateCodeResponse
)


class SpecConstrainedCodeGenerator:
    """
    Génère du code en respectant STRICTEMENT la CodeSpec.

    Le code généré:
    1. Utilise le langage spécifié
    2. Implémente les opérations clés listées
    3. Respecte les types I/O
    4. Produit l'output attendu avec l'input d'exemple
    5. Est optimisé pour affichage sur slide (lisible, commenté)
    """

    # Templates de structure par langage
    LANGUAGE_TEMPLATES = {
        CodeLanguage.PYTHON: {
            "file_ext": ".py",
            "comment_style": "#",
            "main_wrapper": 'if __name__ == "__main__":\n    {main_code}',
            "print_fn": "print",
        },
        CodeLanguage.JAVA: {
            "file_ext": ".java",
            "comment_style": "//",
            "main_wrapper": "public static void main(String[] args) {{\n    {main_code}\n}}",
            "print_fn": "System.out.println",
        },
        CodeLanguage.JAVASCRIPT: {
            "file_ext": ".js",
            "comment_style": "//",
            "main_wrapper": "{main_code}",
            "print_fn": "console.log",
        },
        CodeLanguage.TYPESCRIPT: {
            "file_ext": ".ts",
            "comment_style": "//",
            "main_wrapper": "{main_code}",
            "print_fn": "console.log",
        },
        CodeLanguage.GO: {
            "file_ext": ".go",
            "comment_style": "//",
            "main_wrapper": "func main() {{\n    {main_code}\n}}",
            "print_fn": "fmt.Println",
        },
    }

    def __init__(self):
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client"""
        try:
            from shared.llm_provider import get_llm_client, get_model_name
            self.client = get_llm_client()
            self.model = get_model_name("quality")
        except ImportError:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    async def generate(
        self,
        spec: CodeSpec,
        include_comments: bool = True,
        optimize_for_display: bool = True
    ) -> GenerateCodeResponse:
        """
        Génère du code à partir de la spec.

        Args:
            spec: La spécification de code
            include_comments: Inclure des commentaires pédagogiques
            optimize_for_display: Optimiser pour affichage slide

        Returns:
            GenerateCodeResponse avec le code généré
        """
        try:
            print(f"[CODE_GEN] Generating {spec.language.value} code for: {spec.concept_name}", flush=True)

            # Construire le prompt contraint par la spec
            code_result = await self._generate_constrained_code(
                spec, include_comments, optimize_for_display
            )

            if not code_result:
                return GenerateCodeResponse(
                    success=False,
                    error="Failed to generate code"
                )

            # Valider que le code respecte la spec
            validation = await self._validate_against_spec(code_result["code"], spec)

            if not validation["matches_spec"]:
                print(f"[CODE_GEN] Code doesn't match spec, regenerating...", flush=True)
                # Tenter une correction
                code_result = await self._correct_code(
                    code_result["code"],
                    spec,
                    validation["violations"]
                )

            return GenerateCodeResponse(
                success=True,
                code=code_result["code"],
                highlighted_lines=code_result.get("highlighted_lines", []),
                runnable=code_result.get("runnable", False)
            )

        except Exception as e:
            print(f"[CODE_GEN] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return GenerateCodeResponse(success=False, error=str(e))

    async def _generate_constrained_code(
        self,
        spec: CodeSpec,
        include_comments: bool,
        optimize_for_display: bool
    ) -> Optional[Dict[str, Any]]:
        """Génère le code en respectant les contraintes de la spec"""

        # Construire les contraintes
        constraints = self._build_constraints(spec)

        # Exemple I/O pour le test
        example_io_str = ""
        if spec.example_io:
            example_io_str = f"""
EXEMPLE I/O À RESPECTER:
- Input: {spec.example_io.input_value}
- Output attendu: {spec.example_io.expected_output}
Le code DOIT produire exactement cet output quand on lui donne cet input.
"""

        prompt = f"""Génère du code {spec.language.value} qui respecte STRICTEMENT cette spécification:

SPEC:
- Concept: {spec.concept_name}
- Purpose: {spec.purpose.value}
- Description: {spec.description}
- Input type: {spec.input_type}
- Output type: {spec.output_type}

OPÉRATIONS OBLIGATOIRES (doivent être VISIBLES dans le code):
{chr(10).join(f'- {op}' for op in spec.key_operations)}

ÉLÉMENTS OBLIGATOIRES:
{chr(10).join(f'- {elem}' for elem in spec.must_include) if spec.must_include else '- Aucun spécifique'}

ÉLÉMENTS INTERDITS:
{chr(10).join(f'- {elem}' for elem in spec.must_not_include) if spec.must_not_include else '- Aucun spécifique'}

{example_io_str}

CONTRAINTES DE GÉNÉRATION:
{constraints}

CONTRAINTES D'AFFICHAGE (slide):
- Maximum {spec.estimated_lines} lignes
- Code lisible et bien formaté
- {"Commentaires pédagogiques en français" if include_comments else "Pas de commentaires"}
- Noms de variables/fonctions explicites
- Une fonction principale clairement identifiable

IMPORTANT: Return valid JSON. For the "code" field:
- Use \\n for newlines (NOT actual newlines inside the string)
- Use \\" for quotes inside the code
- Do NOT use triple quotes

Example format:
{{
    "code": "def example():\\n    print(\\"Hello\\")\\n    return 42",
    "highlighted_lines": [1, 2],
    "main_function": "example",
    "runnable": true,
    "test_command": "python -c \\"from example import example; print(example())\\""
}}

Return ONLY valid JSON:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Tu es un expert en {spec.language.value}. Tu génères du code pédagogique qui respecte STRICTEMENT les spécifications. Le code doit être fonctionnel et produire exactement l'output attendu."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2000
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"[CODE_GEN] Generation failed: {e}", flush=True)
            return None

    def _build_constraints(self, spec: CodeSpec) -> str:
        """Construit les contraintes spécifiques au langage"""
        lang_config = self.LANGUAGE_TEMPLATES.get(spec.language, {})

        constraints = []

        # Contraintes de langage
        if spec.language == CodeLanguage.JAVA:
            constraints.append("- Classe avec méthode main() pour tester")
            constraints.append("- Imports nécessaires en haut du fichier")

        elif spec.language == CodeLanguage.PYTHON:
            constraints.append("- if __name__ == '__main__': pour le test")
            constraints.append("- Type hints recommandés")

        elif spec.language == CodeLanguage.JAVASCRIPT:
            constraints.append("- Utiliser des fonctions fléchées quand approprié")
            constraints.append("- const/let au lieu de var")

        elif spec.language == CodeLanguage.GO:
            constraints.append("- Package main avec func main()")
            constraints.append("- Gestion d'erreurs explicite")

        # Contraintes de purpose
        if spec.purpose == CodePurpose.TRANSFORMER:
            constraints.append("- Fonction de transformation clairement nommée (ex: transform, convert)")
            constraints.append("- Input et output clairement séparés")

        elif spec.purpose == CodePurpose.PARSER:
            constraints.append("- Fonction de parsing avec gestion d'erreur")

        elif spec.purpose == CodePurpose.ALGORITHM:
            constraints.append("- Algorithme dans une fonction dédiée")
            constraints.append("- Complexité visible dans le code")

        return "\n".join(constraints)

    async def _validate_against_spec(
        self,
        code: str,
        spec: CodeSpec
    ) -> Dict[str, Any]:
        """Valide que le code respecte la spec"""

        prompt = f"""Vérifie si ce code respecte STRICTEMENT la spécification.

CODE:
```{spec.language.value}
{code}
```

SPEC À RESPECTER:
- Langage: {spec.language.value}
- Input type: {spec.input_type}
- Output type: {spec.output_type}
- Opérations obligatoires: {spec.key_operations}
- Éléments obligatoires: {spec.must_include}

{f"Exemple I/O: {spec.example_io.input_value} → {spec.example_io.expected_output}" if spec.example_io else ""}

Vérifie:
1. Toutes les opérations clés sont-elles VISIBLES dans le code?
2. Les types I/O sont-ils respectés?
3. Le code produirait-il l'output attendu avec l'input d'exemple?
4. Les éléments obligatoires sont-ils présents?

Retourne un JSON:
{{
    "matches_spec": true/false,
    "violations": ["violation 1", "violation 2"],
    "operations_found": ["op1", "op2"],
    "operations_missing": ["op3"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un validateur de code strict. Tu vérifies la conformité du code par rapport à une spécification."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=500
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"[CODE_GEN] Validation failed: {e}", flush=True)
            return {"matches_spec": True, "violations": []}

    async def _correct_code(
        self,
        code: str,
        spec: CodeSpec,
        violations: List[str]
    ) -> Dict[str, Any]:
        """Corrige le code pour qu'il respecte la spec"""

        prompt = f"""Corrige ce code pour qu'il respecte la spécification.

CODE ACTUEL:
```{spec.language.value}
{code}
```

VIOLATIONS À CORRIGER:
{chr(10).join(f'- {v}' for v in violations)}

SPEC À RESPECTER:
- Opérations obligatoires: {spec.key_operations}
- Exemple I/O: {spec.example_io.input_value if spec.example_io else 'N/A'} → {spec.example_io.expected_output if spec.example_io else 'N/A'}

IMPORTANT: Return valid JSON. For the "code" field, use proper JSON string escaping:
- Use \\n for newlines (NOT actual newlines inside the string)
- Use \\" for quotes inside the code
- Do NOT use triple quotes

Example of correct JSON format:
{{
    "code": "public class Example {{\\n    public static void main(String[] args) {{\\n        System.out.println(\\"Hello\\");\\n    }}\\n}}",
    "highlighted_lines": [1, 3],
    "corrections_made": ["Added main method"],
    "runnable": true
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu corriges du code pour qu'il respecte une spécification précise."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=2000
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"[CODE_GEN] Correction failed: {e}", flush=True)
            return {"code": code, "highlighted_lines": [], "runnable": False}


# Singleton
_generator: Optional[SpecConstrainedCodeGenerator] = None


def get_code_generator() -> SpecConstrainedCodeGenerator:
    """Get singleton code generator"""
    global _generator
    if _generator is None:
        _generator = SpecConstrainedCodeGenerator()
    return _generator
