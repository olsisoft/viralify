"""
Maestro Code Spec Extractor

Extrait une spécification de code à partir du voiceover et du concept expliqué.
C'est le gardien de la cohérence entre ce qui est dit et ce qui sera codé.

MAESTRO = Le Chef d'Orchestre Pédagogique
- Comprend le CONTEXTE (Kafka vs ESB vs standalone)
- Extrait l'INTENTION (ce que l'instructeur veut enseigner)
- Crée le CONTRAT (CodeSpec) que le générateur doit respecter
- Valide la COHÉRENCE entre explication et implémentation
"""

import os
import json
import uuid
from typing import Optional, Dict, Any, List, Tuple

from .models import (
    CodeSpec, CodeLanguage, CodePurpose, ExampleIO,
    TechnologyEcosystem, TechnologyContext,
    CodeSpecRequest, CodeSpecResponse
)


class MaestroSpecExtractor:
    """
    Maestro: Extrait et valide les spécifications de code.

    Responsabilités:
    1. COMPRENDRE LE CONTEXTE - Dans quel écosystème? (Kafka, ESB, Spring...)
    2. EXTRAIRE L'INTENTION - Qu'est-ce que l'instructeur veut enseigner?
    3. CRÉER LE CONTRAT - CodeSpec précise et non-ambiguë
    4. VALIDER LA COHÉRENCE - Le code correspondra-t-il au voiceover?

    Un "transformer" Kafka Connect ≠ un "transformer" MuleSoft ≠ un "transformer" standalone
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
            "translate", "convert", "map", "mapper", "smt"
        ],
        CodePurpose.VALIDATOR: [
            "valider", "validation", "vérifier", "validate", "check"
        ],
        CodePurpose.PARSER: [
            "parser", "parsing", "analyser", "parse", "lire", "read"
        ],
        CodePurpose.PROCESSOR: [
            "traiter", "process", "traitement", "processing", "processor"
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
            "connecter", "connect", "connexion", "connection", "connector",
            "source connector", "sink connector"
        ],
    }

    # Patterns pour détecter le contexte technologique
    CONTEXT_PATTERNS = {
        # Messaging & Streaming
        TechnologyEcosystem.KAFKA: [
            "kafka", "kafka connect", "kafka streams", "ksql", "ksqldb",
            "confluent", "topic", "broker", "consumer", "producer",
            "smt", "single message transform", "connect api"
        ],
        TechnologyEcosystem.RABBITMQ: [
            "rabbitmq", "rabbit mq", "amqp", "exchange", "queue rabbitmq"
        ],
        TechnologyEcosystem.PULSAR: [
            "pulsar", "apache pulsar", "bookkeeper"
        ],

        # ESB & Integration
        TechnologyEcosystem.MULESOFT: [
            "mulesoft", "mule", "anypoint", "dataweave", "mule 4"
        ],
        TechnologyEcosystem.TALEND: [
            "talend", "talend esb", "talend studio"
        ],
        TechnologyEcosystem.APACHE_CAMEL: [
            "camel", "apache camel", "camel route", "enterprise integration patterns", "eip"
        ],
        TechnologyEcosystem.SPRING_INTEGRATION: [
            "spring integration", "spring messaging", "message channel"
        ],

        # Cloud
        TechnologyEcosystem.AWS: [
            "aws", "amazon", "lambda", "step functions", "sqs", "sns",
            "kinesis", "eventbridge", "s3"
        ],
        TechnologyEcosystem.GCP: [
            "gcp", "google cloud", "cloud functions", "pubsub", "pub/sub",
            "bigquery", "dataflow"
        ],
        TechnologyEcosystem.AZURE: [
            "azure", "microsoft azure", "azure functions", "event hub",
            "service bus", "cosmos"
        ],

        # Frameworks
        TechnologyEcosystem.SPRING: [
            "spring", "spring boot", "spring mvc", "spring data",
            "spring batch", "@autowired", "@service", "@component"
        ],
        TechnologyEcosystem.QUARKUS: [
            "quarkus", "graalvm native"
        ],
        TechnologyEcosystem.FASTAPI: [
            "fastapi", "pydantic", "uvicorn"
        ],
        TechnologyEcosystem.DJANGO: [
            "django", "django rest", "drf"
        ],

        # Data Processing
        TechnologyEcosystem.SPARK: [
            "spark", "apache spark", "pyspark", "spark sql", "dataframe spark"
        ],
        TechnologyEcosystem.FLINK: [
            "flink", "apache flink", "flink sql"
        ],
        TechnologyEcosystem.AIRFLOW: [
            "airflow", "apache airflow", "dag", "airflow dag"
        ],
        TechnologyEcosystem.DBT: [
            "dbt", "data build tool", "dbt model"
        ],

        # Kubernetes
        TechnologyEcosystem.KUBERNETES: [
            "kubernetes", "k8s", "kubectl", "pod", "deployment", "helm"
        ],
    }

    # Composants spécifiques par écosystème
    ECOSYSTEM_COMPONENTS = {
        TechnologyEcosystem.KAFKA: {
            "patterns": {
                "connect": ["kafka connect", "connect api", "connector", "smt"],
                "streams": ["kafka streams", "kstream", "ktable"],
                "producer": ["producer", "kafka producer"],
                "consumer": ["consumer", "kafka consumer"],
                "ksql": ["ksql", "ksqldb"]
            },
            "default_imports": {
                "connect": ["org.apache.kafka.connect.transforms.Transformation"],
                "streams": ["org.apache.kafka.streams"],
                "producer": ["org.apache.kafka.clients.producer"],
                "consumer": ["org.apache.kafka.clients.consumer"],
            }
        },
        TechnologyEcosystem.SPRING: {
            "patterns": {
                "boot": ["spring boot", "@springbootapplication"],
                "integration": ["spring integration", "@messaginggateway"],
                "batch": ["spring batch", "job", "step"],
                "data": ["spring data", "repository"]
            },
            "default_imports": {
                "boot": ["org.springframework.boot"],
                "integration": ["org.springframework.integration"],
            }
        },
        TechnologyEcosystem.MULESOFT: {
            "patterns": {
                "flow": ["mule flow", "flow ref"],
                "dataweave": ["dataweave", "%dw"],
                "connector": ["mule connector", "anypoint connector"]
            },
            "default_imports": {}
        }
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
            # 1. Pré-analyse: détecter langage, purpose et CONTEXTE
            detected_language = self._detect_language(voiceover_text, preferred_language)
            detected_purpose = self._detect_purpose(voiceover_text)
            detected_context = self._detect_context(voiceover_text)

            print(f"[MAESTRO] Detected language: {detected_language}", flush=True)
            print(f"[MAESTRO] Detected purpose: {detected_purpose}", flush=True)
            print(f"[MAESTRO] Detected context: {detected_context.ecosystem.value if detected_context else 'standalone'} "
                  f"({detected_context.component if detected_context else 'N/A'})", flush=True)

            # 2. Extraction complète via LLM (avec contexte)
            spec = await self._extract_full_spec(
                voiceover_text=voiceover_text,
                concept_name=concept_name,
                detected_language=detected_language,
                detected_purpose=detected_purpose,
                detected_context=detected_context,
                audience_level=audience_level,
                content_language=content_language
            )

            if not spec:
                return CodeSpecResponse(
                    success=False,
                    error="Failed to extract code specification from voiceover"
                )

            # 3. Validation de la spec (inclut validation du contexte)
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

    def _detect_context(self, voiceover_text: str) -> Optional[TechnologyContext]:
        """
        Détecte le contexte technologique du voiceover.

        CRITIQUE: Un transformer Kafka Connect ≠ un transformer MuleSoft
        """
        text_lower = voiceover_text.lower()

        # Trouver l'écosystème
        detected_ecosystem = None
        max_matches = 0

        for ecosystem, patterns in self.CONTEXT_PATTERNS.items():
            matches = sum(1 for p in patterns if p in text_lower)
            if matches > max_matches:
                max_matches = matches
                detected_ecosystem = ecosystem

        if not detected_ecosystem or max_matches == 0:
            return None  # Standalone

        # Trouver le composant spécifique
        component = detected_ecosystem.value  # Par défaut: le nom de l'écosystème
        required_apis = []
        implicit_imports = []

        ecosystem_config = self.ECOSYSTEM_COMPONENTS.get(detected_ecosystem, {})
        component_patterns = ecosystem_config.get("patterns", {})
        default_imports = ecosystem_config.get("default_imports", {})

        for comp_name, comp_patterns in component_patterns.items():
            if any(p in text_lower for p in comp_patterns):
                component = f"{detected_ecosystem.value} {comp_name}"
                implicit_imports = default_imports.get(comp_name, [])
                break

        # Construire la description du contexte
        context_description = self._build_context_description(
            detected_ecosystem, component, voiceover_text
        )

        return TechnologyContext(
            ecosystem=detected_ecosystem,
            component=component,
            version=None,  # Pourrait être extrait si mentionné
            architecture_pattern=self._detect_architecture_pattern(text_lower),
            required_apis=required_apis,
            implicit_imports=implicit_imports,
            context_description=context_description
        )

    def _detect_architecture_pattern(self, text_lower: str) -> Optional[str]:
        """Détecte le pattern architectural mentionné"""
        patterns = {
            "eip": ["enterprise integration pattern", "eip", "routing slip", "content-based router"],
            "cqrs": ["cqrs", "command query"],
            "event_sourcing": ["event sourcing", "event store"],
            "saga": ["saga pattern", "saga"],
            "microservices": ["microservice", "micro-service"],
            "hexagonal": ["hexagonal", "ports and adapters", "clean architecture"],
        }

        for pattern_name, keywords in patterns.items():
            if any(kw in text_lower for kw in keywords):
                return pattern_name

        return None

    def _build_context_description(
        self,
        ecosystem: TechnologyEcosystem,
        component: str,
        voiceover_text: str
    ) -> str:
        """Construit une description du contexte pour l'utilisateur"""

        descriptions = {
            TechnologyEcosystem.KAFKA: {
                "kafka connect": "Dans Kafka Connect, un transformer (SMT) est une classe qui implémente l'interface Transformation<R> pour modifier les messages en transit.",
                "kafka streams": "Dans Kafka Streams, on utilise des opérations sur KStream/KTable pour transformer les données en temps réel.",
                "default": "Dans l'écosystème Kafka, les transformations se font via des SMT (Connect) ou des opérations Streams."
            },
            TechnologyEcosystem.MULESOFT: {
                "dataweave": "Dans MuleSoft, DataWeave est le langage de transformation principal, utilisant une syntaxe %dw 2.0.",
                "default": "Dans MuleSoft Anypoint, les transformations utilisent DataWeave dans les flows."
            },
            TechnologyEcosystem.SPRING: {
                "spring integration": "Dans Spring Integration, les transformations utilisent des MessageHandler ou des @Transformer annotés.",
                "default": "Dans Spring, les transformations peuvent utiliser divers patterns selon le module."
            },
            TechnologyEcosystem.APACHE_CAMEL: {
                "default": "Dans Apache Camel, les transformations utilisent les EIP (Enterprise Integration Patterns) avec une DSL fluide."
            }
        }

        eco_descriptions = descriptions.get(ecosystem, {})
        for key, desc in eco_descriptions.items():
            if key in component.lower():
                return desc
        return eco_descriptions.get("default", f"Dans le contexte de {ecosystem.value}.")

    async def _extract_full_spec(
        self,
        voiceover_text: str,
        concept_name: str,
        detected_language: CodeLanguage,
        detected_purpose: CodePurpose,
        detected_context: Optional[TechnologyContext],
        audience_level: str,
        content_language: str
    ) -> Optional[CodeSpec]:
        """Extraction complète de la spec via LLM (avec contexte)"""

        # Section contexte pour le prompt
        context_section = ""
        if detected_context:
            context_section = f"""
CONTEXTE TECHNOLOGIQUE DÉTECTÉ:
- Écosystème: {detected_context.ecosystem.value}
- Composant: {detected_context.component}
- Pattern architectural: {detected_context.architecture_pattern or 'Non spécifié'}
- Imports implicites: {detected_context.implicit_imports}

IMPORTANT: Le code généré DOIT correspondre à ce contexte!
- Un transformer Kafka Connect implémente Transformation<R>
- Un transformer MuleSoft utilise DataWeave
- Un transformer standalone est une simple fonction
"""

        prompt = f"""Analyse ce voiceover de cours et extrais une spécification de code précise.

VOICEOVER:
\"\"\"
{voiceover_text}
\"\"\"

CONCEPT ENSEIGNÉ: {concept_name}
LANGAGE DÉTECTÉ: {detected_language.value}
TYPE DE CODE DÉTECTÉ: {detected_purpose.value}
NIVEAU AUDIENCE: {audience_level}
{context_section}

Ta tâche: Extraire une SPEC PRÉCISE que le générateur de code devra respecter.

Retourne un JSON avec:
{{
    "description": "Ce que le code doit faire (1-2 phrases)",
    "input_type": "Type d'entrée précis (ex: 'XML string', 'ConnectRecord')",
    "output_type": "Type de sortie précis (ex: 'JSON string', 'ConnectRecord modifié')",
    "key_operations": ["opération1", "opération2", "opération3"],
    "must_include": ["élément obligatoire spécifique au contexte"],
    "must_not_include": ["à éviter - incohérent avec le contexte"],
    "example_io": {{
        "input_value": "exemple d'entrée concret adapté au contexte",
        "input_description": "Description de l'entrée",
        "expected_output": "sortie attendue exacte",
        "output_description": "Description de la sortie"
    }},
    "context_specifics": {{
        "interfaces_to_implement": ["interface si applicable"],
        "annotations_required": ["@annotation si applicable"],
        "base_class": "classe parente si applicable"
    }},
    "pedagogical_goal": "Qu'est-ce que l'apprenant doit comprendre en voyant ce code?",
    "estimated_lines": 25
}}

RÈGLES:
1. L'exemple I/O doit être CONCRET et adapté au CONTEXTE
2. Les key_operations doivent être VISIBLES dans le code final
3. Si contexte Kafka Connect: types I/O = ConnectRecord, interface = Transformation<R>
4. Si contexte MuleSoft: utiliser la syntaxe DataWeave
5. Si standalone: simple fonction sans framework
6. Le code généré sera affiché sur un slide: garder simple (max 30 lignes)

Retourne UNIQUEMENT le JSON:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es Maestro, expert en analyse pédagogique. Tu extrais des spécifications précises DANS LE BON CONTEXTE TECHNOLOGIQUE. Un transformer Kafka ≠ un transformer ESB."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=2000
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

            # Ajouter les spécificités du contexte aux must_include
            must_include = result.get("must_include", [])
            context_specifics = result.get("context_specifics", {})
            if context_specifics.get("interfaces_to_implement"):
                must_include.extend([f"implements {i}" for i in context_specifics["interfaces_to_implement"]])
            if context_specifics.get("annotations_required"):
                must_include.extend(context_specifics["annotations_required"])

            spec = CodeSpec(
                spec_id=f"spec_{uuid.uuid4().hex[:8]}",
                concept_name=concept_name,
                language=detected_language,
                purpose=detected_purpose,
                description=result.get("description", ""),
                input_type=result.get("input_type", ""),
                output_type=result.get("output_type", ""),
                key_operations=result.get("key_operations", []),
                context=detected_context,  # Contexte technologique (optional)
                must_include=must_include,
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
        """Valide que la spec est cohérente avec le voiceover ET le contexte"""

        context_validation = ""
        if spec.context:
            context_validation = f"""
5. Le CONTEXTE est-il respecté?
   - Écosystème attendu: {spec.context.ecosystem.value}
   - Composant: {spec.context.component}
   - Les types I/O sont-ils cohérents avec ce contexte?
"""

        prompt = f"""Valide cette spécification de code par rapport au voiceover original.

VOICEOVER ORIGINAL:
\"\"\"
{voiceover_text}
\"\"\"

CONCEPT: {concept_name}

SPEC EXTRAITE:
- Langage: {spec.language.value}
- Purpose: {spec.purpose.value}
- Contexte: {spec.context.ecosystem.value if spec.context else 'standalone'} ({spec.context.component if spec.context else 'N/A'})
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
{context_validation}

Retourne un JSON:
{{
    "is_valid": true/false,
    "confidence": 0.95,
    "notes": ["note 1", "note 2"],
    "context_coherent": true/false,
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
                        "content": "Tu es un validateur de spécifications. Tu vérifies la cohérence entre les specs, le contenu source ET le contexte technologique."
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
        """Convertit une CodeSpec en dict (inclut le contexte)"""
        context_dict = None
        if spec.context:
            context_dict = {
                "ecosystem": spec.context.ecosystem.value,
                "component": spec.context.component,
                "version": spec.context.version,
                "architecture_pattern": spec.context.architecture_pattern,
                "required_apis": spec.context.required_apis,
                "implicit_imports": spec.context.implicit_imports,
                "naming_conventions": spec.context.naming_conventions,
                "context_description": spec.context.context_description
            }

        return {
            "spec_id": spec.spec_id,
            "concept_name": spec.concept_name,
            "language": spec.language.value,
            "purpose": spec.purpose.value,
            "context": context_dict,  # NOUVEAU
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
