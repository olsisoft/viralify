"""
Maestro Code Spec Extractor

Extrait une spécification de code à partir du voiceover et du concept expliqué.
C'est le gardien de la cohérence entre ce qui est dit et ce qui sera codé.
"""

import os
import json
import uuid
from typing import Optional, Dict, Any, List, Tuple

from .models import (
    CodeSpec, CodeLanguage, CodePurpose, ExampleIO,
    CodeSpecRequest, CodeSpecResponse
)


class MaestroSpecExtractor:
    """
    Maestro: Extrait et valide les spécifications de code.

    Responsabilités:
    1. Analyser le voiceover pour identifier les intentions de code
    2. Extraire le langage, le type, les I/O attendus
    3. Créer un contrat (CodeSpec) que le générateur devra respecter
    4. Valider que la spec est cohérente avec le concept enseigné
    """

    # Patterns pour détecter le langage dans le voiceover
    LANGUAGE_PATTERNS = {
        "python": ["python", "en python", "avec python", "script python"],
        "java": ["java", "en java", "avec java", "classe java"],
        "javascript": ["javascript", "js", "node", "nodejs", "en javascript"],
        "typescript": ["typescript", "ts", "en typescript"],
        "go": ["golang", "en go", "avec go"],
        "rust": ["rust", "en rust", "avec rust"],
        "csharp": ["c#", "csharp", "c sharp", ".net"],
        "kotlin": ["kotlin", "en kotlin"],
        "scala": ["scala", "en scala"],
        "sql": ["sql", "requête sql", "query sql"],
        "bash": ["bash", "shell", "script shell", "terminal"],
    }

    # Patterns pour détecter le type de code
    PURPOSE_PATTERNS = {
        CodePurpose.TRANSFORMER: [
            "transformer", "convertir", "transformation", "conversion",
            "translate", "convert", "map", "mapper"
        ],
        CodePurpose.VALIDATOR: [
            "valider", "validation", "vérifier", "validate", "check"
        ],
        CodePurpose.PARSER: [
            "parser", "parsing", "analyser", "parse", "lire", "read"
        ],
        CodePurpose.PROCESSOR: [
            "traiter", "process", "traitement", "processing"
        ],
        CodePurpose.ALGORITHM: [
            "algorithme", "algorithm", "calculer", "compute", "trier", "sort"
        ],
        CodePurpose.PATTERN_DEMO: [
            "pattern", "design pattern", "modèle", "architecture"
        ],
        CodePurpose.API_CLIENT: [
            "api", "client", "appeler", "call", "requête", "request"
        ],
        CodePurpose.CONNECTOR: [
            "connecter", "connect", "connexion", "connection", "database", "db"
        ],
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

    async def extract_spec(
        self,
        voiceover_text: str,
        concept_name: str,
        preferred_language: Optional[str] = None,
        audience_level: str = "intermediate",
        content_language: str = "fr"
    ) -> CodeSpecResponse:
        """
        Extrait une spécification de code du voiceover.

        Args:
            voiceover_text: Texte du voiceover qui mentionne le code
            concept_name: Nom du concept enseigné
            preferred_language: Langage préféré (optionnel)
            audience_level: Niveau de l'audience
            content_language: Langue du contenu

        Returns:
            CodeSpecResponse avec la spec extraite
        """
        try:
            # 1. Pré-analyse: détecter langage et purpose
            detected_language = self._detect_language(voiceover_text, preferred_language)
            detected_purpose = self._detect_purpose(voiceover_text)

            print(f"[MAESTRO] Detected language: {detected_language}", flush=True)
            print(f"[MAESTRO] Detected purpose: {detected_purpose}", flush=True)

            # 2. Extraction complète via LLM
            spec = await self._extract_full_spec(
                voiceover_text=voiceover_text,
                concept_name=concept_name,
                detected_language=detected_language,
                detected_purpose=detected_purpose,
                audience_level=audience_level,
                content_language=content_language
            )

            if not spec:
                return CodeSpecResponse(
                    success=False,
                    error="Failed to extract code specification from voiceover"
                )

            # 3. Validation de la spec
            validation_result = await self._validate_spec(spec, voiceover_text, concept_name)
            spec.is_validated = validation_result["is_valid"]
            spec.validation_notes = validation_result.get("notes", [])

            print(f"[MAESTRO] Spec validated: {spec.is_validated}", flush=True)
            if spec.validation_notes:
                for note in spec.validation_notes:
                    print(f"[MAESTRO] Note: {note}", flush=True)

            return CodeSpecResponse(
                success=True,
                spec_id=spec.spec_id,
                spec=self._spec_to_dict(spec)
            )

        except Exception as e:
            print(f"[MAESTRO] Error extracting spec: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return CodeSpecResponse(success=False, error=str(e))

    def _detect_language(
        self,
        voiceover_text: str,
        preferred_language: Optional[str]
    ) -> CodeLanguage:
        """Détecte le langage mentionné dans le voiceover"""
        text_lower = voiceover_text.lower()

        # Si préférence explicite
        if preferred_language:
            try:
                return CodeLanguage(preferred_language.lower())
            except ValueError:
                pass

        # Chercher dans le texte
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return CodeLanguage(lang)

        # Par défaut: pseudo-code
        return CodeLanguage.PSEUDOCODE

    def _detect_purpose(self, voiceover_text: str) -> CodePurpose:
        """Détecte le type/purpose du code"""
        text_lower = voiceover_text.lower()

        for purpose, patterns in self.PURPOSE_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return purpose

        # Par défaut
        return CodePurpose.ALGORITHM

    async def _extract_full_spec(
        self,
        voiceover_text: str,
        concept_name: str,
        detected_language: CodeLanguage,
        detected_purpose: CodePurpose,
        audience_level: str,
        content_language: str
    ) -> Optional[CodeSpec]:
        """Extraction complète de la spec via LLM"""

        prompt = f"""Analyse ce voiceover de cours et extrais une spécification de code précise.

VOICEOVER:
\"\"\"
{voiceover_text}
\"\"\"

CONCEPT ENSEIGNÉ: {concept_name}
LANGAGE DÉTECTÉ: {detected_language.value}
TYPE DE CODE DÉTECTÉ: {detected_purpose.value}
NIVEAU AUDIENCE: {audience_level}

Ta tâche: Extraire une SPEC PRÉCISE que le générateur de code devra respecter.

Retourne un JSON avec:
{{
    "description": "Ce que le code doit faire (1-2 phrases)",
    "input_type": "Type d'entrée précis (ex: 'XML string', 'List of integers')",
    "output_type": "Type de sortie précis (ex: 'JSON string', 'Sorted list')",
    "key_operations": ["opération1", "opération2", "opération3"],
    "must_include": ["élément obligatoire 1", "élément obligatoire 2"],
    "must_not_include": ["à éviter 1"],
    "example_io": {{
        "input_value": "exemple d'entrée concret",
        "input_description": "Description de l'entrée",
        "expected_output": "sortie attendue exacte",
        "output_description": "Description de la sortie"
    }},
    "pedagogical_goal": "Qu'est-ce que l'apprenant doit comprendre en voyant ce code?",
    "estimated_lines": 25
}}

RÈGLES:
1. L'exemple I/O doit être CONCRET et EXÉCUTABLE
2. Les key_operations doivent être VISIBLES dans le code final
3. L'input/output doivent correspondre EXACTEMENT à ce que le voiceover décrit
4. Le code généré sera affiché sur un slide: garder simple (max 30 lignes)

Retourne UNIQUEMENT le JSON:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es Maestro, expert en analyse pédagogique. Tu extrais des spécifications précises pour garantir la cohérence entre l'explication et le code."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1500
            )

            result = json.loads(response.choices[0].message.content)

            # Construire le CodeSpec
            example_io = None
            if result.get("example_io"):
                ex = result["example_io"]
                example_io = ExampleIO(
                    input_value=ex.get("input_value", ""),
                    input_description=ex.get("input_description", ""),
                    expected_output=ex.get("expected_output", ""),
                    output_description=ex.get("output_description", "")
                )

            spec = CodeSpec(
                spec_id=f"spec_{uuid.uuid4().hex[:8]}",
                concept_name=concept_name,
                language=detected_language,
                purpose=detected_purpose,
                description=result.get("description", ""),
                input_type=result.get("input_type", ""),
                output_type=result.get("output_type", ""),
                key_operations=result.get("key_operations", []),
                must_include=result.get("must_include", []),
                must_not_include=result.get("must_not_include", []),
                example_io=example_io,
                voiceover_excerpt=voiceover_text[:500],
                pedagogical_goal=result.get("pedagogical_goal", ""),
                complexity_level=audience_level,
                estimated_lines=result.get("estimated_lines", 25)
            )

            return spec

        except Exception as e:
            print(f"[MAESTRO] LLM extraction failed: {e}", flush=True)
            return None

    async def _validate_spec(
        self,
        spec: CodeSpec,
        voiceover_text: str,
        concept_name: str
    ) -> Dict[str, Any]:
        """Valide que la spec est cohérente avec le voiceover"""

        prompt = f"""Valide cette spécification de code par rapport au voiceover original.

VOICEOVER ORIGINAL:
\"\"\"
{voiceover_text}
\"\"\"

CONCEPT: {concept_name}

SPEC EXTRAITE:
- Langage: {spec.language.value}
- Purpose: {spec.purpose.value}
- Description: {spec.description}
- Input: {spec.input_type}
- Output: {spec.output_type}
- Opérations: {spec.key_operations}
- Exemple I/O: {spec.example_io.input_value if spec.example_io else 'N/A'} → {spec.example_io.expected_output if spec.example_io else 'N/A'}

Vérifie:
1. Le langage correspond-il à ce qui est dit dans le voiceover?
2. Les types I/O correspondent-ils à ce qui est décrit?
3. L'exemple est-il cohérent avec l'explication?
4. Les opérations clés sont-elles mentionnées ou implicites dans le voiceover?

Retourne un JSON:
{{
    "is_valid": true/false,
    "confidence": 0.95,
    "notes": ["note 1", "note 2"],
    "corrections": {{
        "field_name": "valeur corrigée"
    }}
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un validateur de spécifications. Tu vérifies la cohérence entre les specs et le contenu source."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=500
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"[MAESTRO] Validation failed: {e}", flush=True)
            return {"is_valid": True, "notes": ["Validation skipped due to error"]}

    def _spec_to_dict(self, spec: CodeSpec) -> Dict[str, Any]:
        """Convertit une CodeSpec en dict"""
        return {
            "spec_id": spec.spec_id,
            "concept_name": spec.concept_name,
            "language": spec.language.value,
            "purpose": spec.purpose.value,
            "description": spec.description,
            "input_type": spec.input_type,
            "output_type": spec.output_type,
            "key_operations": spec.key_operations,
            "must_include": spec.must_include,
            "must_not_include": spec.must_not_include,
            "example_io": {
                "input_value": spec.example_io.input_value,
                "input_description": spec.example_io.input_description,
                "expected_output": spec.example_io.expected_output,
                "output_description": spec.example_io.output_description
            } if spec.example_io else None,
            "voiceover_excerpt": spec.voiceover_excerpt,
            "pedagogical_goal": spec.pedagogical_goal,
            "complexity_level": spec.complexity_level,
            "estimated_lines": spec.estimated_lines,
            "is_validated": spec.is_validated,
            "validation_notes": spec.validation_notes
        }


# Singleton
_extractor: Optional[MaestroSpecExtractor] = None


def get_spec_extractor() -> MaestroSpecExtractor:
    """Get singleton spec extractor"""
    global _extractor
    if _extractor is None:
        _extractor = MaestroSpecExtractor()
    return _extractor
