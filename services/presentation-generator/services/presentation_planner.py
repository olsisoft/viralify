"""
Presentation Planner Service

Uses GPT-4 to generate a structured presentation script from a topic prompt.
Enhanced with TechPromptBuilder for domain-specific, professional-grade content.
"""
import json
import os
from typing import Optional, List
import httpx
from openai import AsyncOpenAI

from models.presentation_models import (
    PresentationScript,
    Slide,
    SlideType,
    CodeBlock,
    GeneratePresentationRequest,
)
from services.tech_prompt_builder import (
    TechPromptBuilder,
    AudienceLevel,
)
from models.tech_domains import (
    TechDomain,
    TechCareer,
    CodeLanguage,
)
from services.rag_verifier import verify_rag_usage, RAGVerificationResult
from services.rag_threshold_validator import (
    validate_rag_threshold,
    RAGMode,
    RAGThresholdResult,
)


PLANNING_SYSTEM_PROMPT = """You are an expert technical TRAINER and COURSE CREATOR for professional IT training programs. Your task is to create a structured TRAINING VIDEO script - NOT a conference talk, NOT a presentation for meetings.

CONTEXT: This is for an ONLINE TRAINING PLATFORM (like Udemy, Coursera, LinkedIn Learning).
- You are creating EDUCATIONAL CONTENT for learners who want to MASTER a skill
- The tone should be that of a TEACHER explaining to students, not a speaker at a conference
- Focus on LEARNING OBJECTIVES and SKILL ACQUISITION
- Include PRACTICAL EXERCISES and HANDS-ON examples
- Structure content for KNOWLEDGE RETENTION (not just information delivery)

You will receive:
- A topic/prompt describing what to teach
- The target programming language
- The CONTENT LANGUAGE (e.g., 'en', 'fr', 'es') - ALL text content MUST be in this language
- The target duration in seconds
- The target audience level
- OPTIONAL: Source documents (RAG context) to base the training on - USE THIS AS PRIMARY SOURCE

CRITICAL LANGUAGE COMPLIANCE:
- ALL voiceover_text, titles, subtitles, bullet_points, content, and notes MUST be written in the specified content language
- Code comments can be in the content language for educational purposes
- Variable names and syntax keywords stay in the programming language (usually English)
- If content language is 'fr', write in French. If 'es', write in Spanish. If 'de', write in German. etc.
- NEVER mix languages in the content - be consistent throughout

GRAMMAR AND STYLE (especially for non-English):
- Write with PERFECT grammar - no spelling or grammatical errors
- For FRENCH: Use proper accents (é, è, ê, à, ù, ç), correct agreements, proper verb conjugations
- For FRENCH: Write natural, professional French - not literal translations from English
- Use formal register appropriate for educational content

VOICEOVER STYLE - CRITICAL FOR NATURAL TRAINING NARRATION:
- Write as a TEACHER explaining to STUDENTS, NOT a speaker at a conference
- Use contractions for natural flow: "it's", "you'll", "let's", "we're", "that's"
- Start sentences with TRAINING transitions: "Dans cette leçon,", "Apprenons ensemble,", "Voyons maintenant,", "Pratiquons,"
- For ENGLISH: "In this lesson,", "Let's learn together,", "Now let's practice,", "Here's an exercise,"
- Vary sentence length - mix short punchy sentences with longer explanations
- Include PEDAGOGICAL questions: "Why is this important?", "How would you apply this?", "What do you think happens?"
- Add teaching enthusiasm: "This concept is fundamental because...", "Once you master this, you'll be able to..."
- Use "we" for inclusive learning: "Let's learn how...", "Together, we'll explore...", "We're going to practice..."
- AVOID conference phrases: "In this presentation", "As I mentioned earlier", "Thank you for attending"
- USE training phrases: "Dans cette formation", "Au cours de ce module", "À la fin de cette leçon, vous saurez..."
- Add natural pauses for comprehension
- Create smooth transitions focused on LEARNING PROGRESSION
- For FRENCH: Use "on va apprendre", "voyons ensemble", "pratiquons", "retenez bien que"
- NEVER include technical markers like slide numbers, timecodes, or formatting instructions in voiceover
- NEVER say "conference", "presentation", "meeting" - this is a TRAINING/FORMATION

Your output MUST be a valid JSON object with this structure:
{
  "title": "Presentation Title",
  "description": "Brief description of what viewers will learn",
  "target_audience": "Who this is for",
  "language": "python",
  "total_duration": 300,
  "slides": [
    {
      "type": "title|content|code|code_demo|diagram|conclusion",
      "title": "Slide Title",
      "subtitle": "Optional subtitle",
      "content": "Main text content (markdown supported)",
      "bullet_points": ["Point 1", "Point 2"],
      "code_blocks": [
        {
          "language": "python",
          "code": "actual working code here",
          "filename": "example.py",
          "highlight_lines": [1, 3],
          "expected_output": "what this code outputs"
        }
      ],
      "duration": 10.0,
      "voiceover_text": "What the narrator should say for this slide",
      "notes": "Speaker notes"
    }
  ],
  "code_context": {}
}

SYNC ANCHORS FOR TIMELINE PRECISION (internal use - will be stripped before TTS):
Include [SYNC:slide_XXX] markers at the START of each slide's voiceover_text.
Example: "[SYNC:slide_001] Welcome to this tutorial on Python basics."
These markers enable audio-video sync but are automatically removed before text-to-speech.
The actual narration starts AFTER the marker - write naturally from that point.

IMPORTANT RULES:
1. Start with a "title" slide introducing the topic
2. Use "content" slides for explanations with bullet points
3. Use "code" slides to show code snippets with syntax highlighting
4. Use "code_demo" slides when you want to show code being executed
5. End with a "conclusion" slide summarizing key points
6. The voiceover_text should be natural, conversational narration
7. All code MUST be syntactically correct and functional
8. Duration should be based on voiceover length (~150 words per minute)
9. Include 8-15 slides for a 5-minute presentation
10. Progress from simple to complex concepts
11. Each slide's voiceover_text MUST start with [SYNC:slide_XXX] where XXX is the slide index (001, 002, etc.)

SLIDE TYPE GUIDELINES:
- title: Main presentation title with subtitle (MUST have voiceover_text introducing the topic)
- content: Bullet points explaining concepts (3-5 points)
- code: Code snippet with explanation (include voiceover explaining line by line)
- code_demo: Code that will be executed with expected output
- diagram: Visual diagram to explain architecture, processes, or comparisons. MUST include:
  * "diagram_type": one of "flowchart", "architecture", "process", "comparison", "hierarchy"
  * "content": detailed description of what the diagram should show (nodes, connections, labels)
  * Use diagrams when explaining: system architecture, data flow, step-by-step processes, comparisons between options
- conclusion: Summary slide with bullet_points listing 3-5 key takeaways. MUST include bullet_points array AND voiceover_text that summarizes what was learned and thanks the viewer.

DIAGRAM USAGE (IMPORTANT):
- Include at least 1-2 diagram slides for topics involving architecture, workflows, or comparisons
- For "How X works" topics: use a flowchart or process diagram
- For "X vs Y" topics: use a comparison diagram
- For system/architecture topics: use an architecture diagram

DIAGRAM COMPLEXITY BY AUDIENCE LEVEL:
Adjust diagram complexity based on the target audience:

For BEGINNER audience:
- Maximum 5-7 nodes/components in the diagram
- Use high-level generic concepts (e.g., "Database" not "PostgreSQL Primary + Read Replica")
- NO networking details (no VPCs, subnets, load balancers)
- Focus on WHAT the system does, not HOW it's implemented
- Simple, readable labels with short names
- Show only the main data flow with minimal arrows

For SENIOR/EXPERT audience:
- 10-15 nodes is acceptable for detailed architectures
- Include VPCs, subnets, Kubernetes namespaces in clusters
- Show caching layers, load balancers, message queues
- Include data flow directions with descriptive Edge labels
- Show redundancy patterns (primary/replica, multi-AZ)
- Use specific service names (e.g., "ElastiCache Redis" not just "Cache")
- Include monitoring and logging components when relevant

For EXECUTIVE/BUSINESS audience:
- Maximum 6-8 nodes - keep it scannable in 5 seconds
- Focus on VALUE FLOW and system boundaries
- Show: Users → System → Business Value/Output
- Hide implementation details (no internal queues, caches, databases)
- Use BUSINESS terms, not technical jargon
- Emphasize external integrations, data sources, and outputs
- Show cost/billing boundaries if relevant to the topic

CRITICAL: EVERY slide MUST have a non-empty voiceover_text field. The conclusion slide voiceover should recap the key points and end with a natural closing like "Thanks for watching!" or "That's it for this tutorial!".

Output ONLY valid JSON, no markdown code blocks or additional text."""


# Enhanced prompt with visual-audio alignment validation and sync anchors
VALIDATED_PLANNING_PROMPT = """You are an expert technical TRAINER creating TRAINING VIDEOS for an online learning platform (like Udemy, Coursera). Your task is to create a structured TRAINING script where the voiceover PERFECTLY MATCHES the visuals.

TRAINING CONTEXT (CRITICAL):
- This is a FORMATION/TRAINING video, NOT a conference or meeting presentation
- You are a TEACHER explaining to STUDENTS who want to LEARN and MASTER skills
- Focus on PEDAGOGY: learning objectives, exercises, practical examples
- NEVER use conference vocabulary ("presentation", "attendees", "thank you for joining")
- USE training vocabulary: "formation", "leçon", "module", "apprendre", "maîtriser", "pratiquer"

CRITICAL LANGUAGE COMPLIANCE:
- ALL voiceover_text, titles, subtitles, bullet_points, content, and notes MUST be written in the specified CONTENT LANGUAGE
- Code comments can be in the content language for educational purposes
- Variable names and syntax keywords stay in the programming language (usually English)
- If content language is 'fr', write in French. If 'es', write in Spanish. If 'de', write in German. etc.
- NEVER mix languages in the content - be consistent throughout

GRAMMAR AND STYLE REQUIREMENTS (especially for non-English):
- Write with PERFECT grammar - no spelling or grammatical errors
- For FRENCH (fr): Use proper accents (é, è, ê, à, ù, ç, etc.), correct article agreements (le/la/les), proper verb conjugations
- For FRENCH: Use natural, professional French - not literal translations from English
- For FRENCH: Avoid anglicisms when French alternatives exist (e.g., "logiciel" not "software")
- Use formal/professional register appropriate for educational content
- Proofread all text for language-specific errors

SYNC ANCHORS FOR TIMELINE PRECISION:
To enable millisecond-precision audio-video synchronization, include [SYNC:slide_id] markers in the voiceover_text.
These markers indicate when a slide should appear based on the narration timing.

Example:
- "Welcome to this tutorial. [SYNC:slide_001] Let's start by understanding the basics."
- The slide_001 will appear exactly when the word after [SYNC:slide_001] is spoken.

Rules for sync anchors:
1. Place [SYNC:slide_id] at the BEGINNING of text content specific to that slide
2. The slide_id MUST match the "id" field of the corresponding slide
3. Every slide should have its sync anchor in the combined voiceover text
4. Do NOT include the sync anchor brackets in the actual spoken text (they are markers only)

CRITICAL VISUAL-AUDIO ALIGNMENT RULES:
1. NEVER say "as you can see", "look at this diagram", "notice here" unless the slide VISUALLY shows exactly what you're describing
2. For "code" slides: The voiceover must describe what the CODE LOOKS LIKE, not what happens when it runs
3. For "code_demo" slides: You CAN describe execution and output because it will be shown
4. For "diagram" slides: Provide detailed diagram_description so it can be generated
5. For "content" slides: Only reference bullet points that are actually visible

TIMING ALIGNMENT:
- Calculate duration based on: max(voiceover_time, animation_time, visual_time)
- Code typing animation: ~30 characters/second
- Voiceover: ~150 words/minute (2.5 words/second)
- Add 2 seconds buffer per slide for transitions

SLIDE STRUCTURE:
{
  "title": "Presentation Title",
  "description": "Brief description",
  "target_audience": "Who this is for",
  "language": "python",
  "total_duration": 300,
  "slides": [
    {
      "id": "slide_001",
      "type": "title|content|code|code_demo|diagram|conclusion",
      "title": "Slide Title",
      "subtitle": "Optional subtitle",
      "content": "Main text content",
      "bullet_points": ["Point 1", "Point 2"],
      "code_blocks": [
        {
          "language": "python",
          "code": "print('hello')",
          "filename": "example.py",
          "highlight_lines": [1],
          "expected_output": "hello",
          "show_execution": true
        }
      ],
      "diagram_description": "For diagram slides: detailed description of what to draw",
      "diagram_type": "flowchart|architecture|sequence|mindmap|comparison",
      "duration": 15.0,
      "voiceover_text": "[SYNC:slide_001] Narration that matches EXACTLY what's visible",
      "visual_cues": ["code appears", "output shows"],
      "timing_notes": "Wait for code typing to complete before mentioning output"
    }
  ]
}

FORBIDDEN PHRASES (unless slide type matches):
- "this diagram shows..." (only on diagram slides)
- "as you can see the output..." (only on code_demo slides)
- "watch as..." (only on animation slides)
- "here we have a chart..." (only if chart is actually shown)

REQUIRED FOR EACH SLIDE TYPE:
- title: voiceover_text introducing topic (with [SYNC:slide_id] at start)
- content: bullet_points array with items mentioned in voiceover
- code: code_blocks with actual code, voiceover describes the code structure
- code_demo: code_blocks with expected_output, voiceover can mention execution
- diagram: diagram_description (50+ words), diagram_type, voiceover describes the diagram
- conclusion: bullet_points summary, voiceover recap

DIAGRAM COMPLEXITY BY AUDIENCE LEVEL:
Adjust diagram complexity based on the target audience:

For BEGINNER audience:
- Maximum 5-7 nodes/components in the diagram
- Use high-level generic concepts (e.g., "Database" not "PostgreSQL Primary + Read Replica")
- NO networking details (no VPCs, subnets, load balancers)
- Focus on WHAT the system does, not HOW it's implemented
- Simple, readable labels with short names
- Show only the main data flow with minimal arrows

For SENIOR/EXPERT audience:
- 10-15 nodes is acceptable for detailed architectures
- Include VPCs, subnets, Kubernetes namespaces in clusters
- Show caching layers, load balancers, message queues
- Include data flow directions with descriptive Edge labels
- Show redundancy patterns (primary/replica, multi-AZ)
- Use specific service names (e.g., "ElastiCache Redis" not just "Cache")
- Include monitoring and logging components when relevant

For EXECUTIVE/BUSINESS audience:
- Maximum 6-8 nodes - keep it scannable in 5 seconds
- Focus on VALUE FLOW and system boundaries
- Show: Users → System → Business Value/Output
- Hide implementation details (no internal queues, caches, databases)
- Use BUSINESS terms, not technical jargon
- Emphasize external integrations, data sources, and outputs
- Show cost/billing boundaries if relevant to the topic

Output ONLY valid JSON."""


class PresentationPlannerService:
    """Service for planning presentation structure using GPT-4"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=120.0,  # 2 minutes timeout for GPT-4 calls
            max_retries=2
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        # Initialize the tech prompt builder for enhanced code/diagram generation
        self.prompt_builder = TechPromptBuilder()

    async def generate_script(
        self,
        request: GeneratePresentationRequest,
        on_progress: Optional[callable] = None
    ) -> PresentationScript:
        """
        Generate a complete presentation script from a topic prompt.

        Args:
            request: The presentation generation request
            on_progress: Optional callback for progress updates

        Returns:
            PresentationScript with all slides defined

        Raises:
            ValueError: If RAG context is insufficient (blocked mode)
        """
        if on_progress:
            await on_progress(0, "Analyzing topic...")

        # Check RAG threshold BEFORE generation
        rag_context = getattr(request, 'rag_context', None)
        has_documents = bool(getattr(request, 'document_ids', None))

        threshold_result = validate_rag_threshold(rag_context, has_documents)

        # Block if insufficient
        if threshold_result.should_block:
            print(f"[PLANNER] RAG BLOCKED: {threshold_result.error_message}", flush=True)
            raise ValueError(threshold_result.error_message)

        # Log warning if partial
        if threshold_result.has_warning:
            print(f"[PLANNER] RAG WARNING: {threshold_result.warning_message}", flush=True)

        # Store RAG mode for later use
        rag_mode = threshold_result.mode.value
        rag_token_count = threshold_result.token_count
        print(f"[PLANNER] RAG mode: {rag_mode} ({rag_token_count} tokens)", flush=True)

        # Build the user prompt
        user_prompt = self._build_prompt(request)

        if on_progress:
            await on_progress(10, "Generating presentation structure...")

        # Call GPT-4 to generate the script
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )

        if on_progress:
            await on_progress(80, "Parsing presentation script...")

        # Parse the response
        content = response.choices[0].message.content
        script_data = json.loads(content)

        # Ensure sync anchors are present for timeline composition
        script_data = self._ensure_sync_anchors(script_data)

        # Convert to PresentationScript
        script = self._parse_script(script_data, request)

        # RAG Verification: Check that generated content uses source documents
        if rag_context:
            rag_result = verify_rag_usage(script_data, rag_context, verbose=True)
            # Store verification result in script metadata
            script.rag_verification = {
                "mode": rag_mode,
                "token_count": rag_token_count,
                "coverage": rag_result.overall_coverage,
                "is_compliant": rag_result.is_compliant,
                "summary": rag_result.summary,
                "potential_hallucinations": len(rag_result.potential_hallucinations),
                "warning": threshold_result.warning_message,
            }
            print(f"[PLANNER] {rag_result.summary}", flush=True)
        else:
            # Store RAG mode even when no context (NONE mode)
            script.rag_verification = {
                "mode": rag_mode,
                "token_count": 0,
                "coverage": 0.0,
                "is_compliant": True,  # No RAG means no compliance check
                "summary": "No source documents - standard AI generation",
                "warning": threshold_result.warning_message,
            }

        if on_progress:
            await on_progress(100, "Script generation complete")

        return script

    # Language code to full name mapping
    LANGUAGE_NAMES = {
        "en": "English",
        "fr": "French (Français)",
        "es": "Spanish (Español)",
        "de": "German (Deutsch)",
        "pt": "Portuguese (Português)",
        "it": "Italian (Italiano)",
        "nl": "Dutch (Nederlands)",
        "pl": "Polish (Polski)",
        "ru": "Russian (Русский)",
        "zh": "Chinese (中文)",
        "ja": "Japanese (日本語)",
        "ko": "Korean (한국어)",
        "ar": "Arabic (العربية)",
    }

    def _get_language_name(self, code: str) -> str:
        """Get full language name from code"""
        return self.LANGUAGE_NAMES.get(code.lower(), code)

    def _map_audience_level(self, audience: str) -> AudienceLevel:
        """Map audience string to AudienceLevel enum."""
        mapping = {
            "beginner": AudienceLevel.BEGINNER,
            "absolute beginner": AudienceLevel.ABSOLUTE_BEGINNER,
            "débutant": AudienceLevel.BEGINNER,
            "débutant absolu": AudienceLevel.ABSOLUTE_BEGINNER,
            "intermediate": AudienceLevel.INTERMEDIATE,
            "intermédiaire": AudienceLevel.INTERMEDIATE,
            "advanced": AudienceLevel.ADVANCED,
            "avancé": AudienceLevel.ADVANCED,
            "expert": AudienceLevel.EXPERT,
        }
        return mapping.get(audience.lower(), AudienceLevel.INTERMEDIATE)

    def _detect_domain(self, topic: str, language: str) -> Optional[TechDomain]:
        """Detect tech domain from topic and programming language."""
        topic_lower = topic.lower()

        # Domain detection keywords
        domain_keywords = {
            TechDomain.DATA_ENGINEERING: ["data pipeline", "etl", "data lake", "spark", "airflow", "data warehouse"],
            TechDomain.MACHINE_LEARNING: ["machine learning", "ml", "neural", "deep learning", "tensorflow", "pytorch", "model training"],
            TechDomain.DEVOPS: ["devops", "ci/cd", "deployment", "infrastructure", "terraform", "ansible"],
            TechDomain.KUBERNETES: ["kubernetes", "k8s", "helm", "kubectl", "pod", "deployment"],
            TechDomain.CLOUD_AWS: ["aws", "amazon", "lambda", "s3", "ec2", "dynamodb"],
            TechDomain.CLOUD_AZURE: ["azure", "microsoft cloud", "azure function"],
            TechDomain.CLOUD_GCP: ["gcp", "google cloud", "bigquery", "cloud run"],
            TechDomain.CYBERSECURITY: ["security", "authentication", "encryption", "vulnerability", "pentest"],
            TechDomain.BLOCKCHAIN: ["blockchain", "smart contract", "solidity", "web3", "ethereum"],
            TechDomain.QUANTUM_COMPUTING: ["quantum", "qubit", "qiskit", "cirq"],
            TechDomain.WEB_FRONTEND: ["react", "vue", "angular", "frontend", "css", "html"],
            TechDomain.WEB_BACKEND: ["api", "rest", "graphql", "backend", "server"],
            TechDomain.MOBILE_DEVELOPMENT: ["mobile", "ios", "android", "flutter", "react native"],
            TechDomain.DATABASES: ["database", "sql", "mongodb", "postgres", "mysql", "redis"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in topic_lower for kw in keywords):
                return domain

        return None

    def _detect_career(self, topic: str) -> Optional[TechCareer]:
        """Detect tech career from topic keywords."""
        topic_lower = topic.lower()

        # Career detection keywords - maps keywords to specific careers
        career_keywords = {
            # Data Engineering careers
            TechCareer.DATA_ENGINEER: ["data engineer", "data engineering", "etl developer", "elt"],
            TechCareer.DATA_LINEAGE_ARCHITECT: ["data lineage", "lineage architect"],
            TechCareer.DATA_LINEAGE_DEVELOPER: ["lineage developer", "lineage tool"],
            TechCareer.DATA_QUALITY_ENGINEER: ["data quality", "dq engineer"],
            TechCareer.DATA_GOVERNANCE_ANALYST: ["data governance", "governance analyst"],
            TechCareer.DATA_STEWARD: ["data steward", "data custodian"],
            TechCareer.DATA_ARCHITECT: ["data architect", "data modeling"],
            TechCareer.ANALYTICS_ENGINEER: ["analytics engineer", "dbt", "data transformation"],
            TechCareer.DATA_CATALOG_ENGINEER: ["data catalog", "metadata management"],
            TechCareer.BIG_DATA_ENGINEER: ["big data", "hadoop", "spark engineer"],
            TechCareer.STREAMING_DATA_ENGINEER: ["streaming data", "kafka engineer", "flink"],
            TechCareer.DATA_ENABLER: ["data enabler", "data enablement"],

            # ML/AI careers
            TechCareer.ML_ENGINEER: ["ml engineer", "machine learning engineer"],
            TechCareer.MLOPS_ENGINEER: ["mlops", "ml ops", "machine learning operations"],
            TechCareer.DATA_SCIENTIST: ["data scientist", "data science"],
            TechCareer.DEEP_LEARNING_ENGINEER: ["deep learning", "neural network engineer"],
            TechCareer.NLP_ENGINEER: ["nlp engineer", "natural language processing"],
            TechCareer.LLM_ENGINEER: ["llm engineer", "large language model"],
            TechCareer.PROMPT_ENGINEER: ["prompt engineer", "prompt engineering"],
            TechCareer.COMPUTER_VISION_ENGINEER: ["computer vision", "cv engineer", "image processing"],
            TechCareer.RECOMMENDATION_ENGINEER: ["recommendation system", "recommender"],
            TechCareer.FEATURE_STORE_ENGINEER: ["feature store", "feature engineering"],

            # DevOps/Platform careers
            TechCareer.DEVOPS_ENGINEER: ["devops engineer", "devops"],
            TechCareer.PLATFORM_ENGINEER: ["platform engineer", "platform engineering"],
            TechCareer.SRE: ["sre", "site reliability", "reliability engineer"],
            TechCareer.KUBERNETES_ENGINEER: ["kubernetes engineer", "k8s engineer"],
            TechCareer.INFRASTRUCTURE_ENGINEER: ["infrastructure engineer", "infra engineer"],
            TechCareer.OBSERVABILITY_ENGINEER: ["observability", "monitoring engineer"],
            TechCareer.CICD_ENGINEER: ["ci/cd engineer", "cicd", "release engineer"],

            # Cloud careers
            TechCareer.CLOUD_ARCHITECT: ["cloud architect"],
            TechCareer.AWS_SOLUTIONS_ARCHITECT: ["aws architect", "aws solutions"],
            TechCareer.AZURE_SOLUTIONS_ARCHITECT: ["azure architect", "azure solutions"],
            TechCareer.GCP_CLOUD_ARCHITECT: ["gcp architect", "google cloud architect"],
            TechCareer.FINOPS_ENGINEER: ["finops", "cloud cost", "cost optimization"],
            TechCareer.SERVERLESS_ARCHITECT: ["serverless architect", "lambda architect"],

            # Security careers
            TechCareer.SECURITY_ENGINEER: ["security engineer", "appsec"],
            TechCareer.DEVSECOPS_ENGINEER: ["devsecops", "security automation"],
            TechCareer.PENETRATION_TESTER: ["penetration tester", "pentester", "ethical hacker"],
            TechCareer.CLOUD_SECURITY_ARCHITECT: ["cloud security", "security architect"],
            TechCareer.IAM_ENGINEER: ["iam engineer", "identity management"],
            TechCareer.THREAT_HUNTER: ["threat hunter", "threat intelligence"],

            # Database careers
            TechCareer.DBA: ["dba", "database administrator"],
            TechCareer.DATABASE_ARCHITECT: ["database architect", "db architect"],
            TechCareer.DBRE: ["dbre", "database reliability"],

            # Architecture careers
            TechCareer.SOFTWARE_ARCHITECT: ["software architect"],
            TechCareer.MICROSERVICES_ARCHITECT: ["microservices architect"],
            TechCareer.API_ARCHITECT: ["api architect", "api design"],
            TechCareer.ENTERPRISE_ARCHITECT: ["enterprise architect", "togaf"],

            # Emerging tech careers
            TechCareer.BLOCKCHAIN_DEVELOPER: ["blockchain developer", "smart contract developer", "solidity developer"],
            TechCareer.QUANTUM_SOFTWARE_ENGINEER: ["quantum engineer", "quantum developer", "qiskit"],
            TechCareer.IOT_ENGINEER: ["iot engineer", "embedded iot"],
            TechCareer.ROBOTICS_SOFTWARE_ENGINEER: ["robotics engineer", "ros developer"],

            # Development careers
            TechCareer.FRONTEND_DEVELOPER: ["frontend developer", "front-end", "ui developer"],
            TechCareer.FULLSTACK_DEVELOPER: ["fullstack", "full-stack", "full stack developer"],
            TechCareer.BACKEND_DEVELOPER: ["backend developer", "back-end", "api developer"],
        }

        for career, keywords in career_keywords.items():
            if any(kw in topic_lower for kw in keywords):
                return career

        return None

    def _detect_code_language(self, language: str) -> Optional[CodeLanguage]:
        """Map programming language string to CodeLanguage enum."""
        mapping = {
            "python": CodeLanguage.PYTHON,
            "javascript": CodeLanguage.JAVASCRIPT,
            "typescript": CodeLanguage.TYPESCRIPT,
            "java": CodeLanguage.JAVA,
            "go": CodeLanguage.GO,
            "golang": CodeLanguage.GO,
            "rust": CodeLanguage.RUST,
            "c++": CodeLanguage.CPP,
            "cpp": CodeLanguage.CPP,
            "c#": CodeLanguage.CSHARP,
            "csharp": CodeLanguage.CSHARP,
            "kotlin": CodeLanguage.KOTLIN,
            "swift": CodeLanguage.SWIFT,
            "ruby": CodeLanguage.RUBY,
            "php": CodeLanguage.PHP,
            "scala": CodeLanguage.SCALA,
            "r": CodeLanguage.R,
            "sql": CodeLanguage.SQL,
            "bash": CodeLanguage.BASH,
            "shell": CodeLanguage.BASH,
            "terraform": CodeLanguage.TERRAFORM,
            "yaml": CodeLanguage.YAML,
            "dockerfile": CodeLanguage.DOCKERFILE,
            "solidity": CodeLanguage.SOLIDITY,
        }
        return mapping.get(language.lower())

    def _build_prompt(self, request: GeneratePresentationRequest) -> str:
        """Build the prompt for GPT-4 with enhanced code quality standards."""
        minutes = request.duration // 60
        seconds = request.duration % 60

        duration_str = f"{minutes} minutes"
        if seconds > 0:
            duration_str += f" {seconds} seconds"

        # Get content language name
        content_lang = getattr(request, 'content_language', 'en')
        content_lang_name = self._get_language_name(content_lang)

        # Detect domain, career, and audience level for enhanced prompts
        domain = self._detect_domain(request.topic, request.language)
        career = self._detect_career(request.topic)
        audience_level = self._map_audience_level(request.target_audience)
        code_language = self._detect_code_language(request.language)

        # Log detected career for debugging
        if career:
            print(f"[PLANNER] Detected career: {career.value}", flush=True)

        # Build enhanced code quality prompt using TechPromptBuilder
        code_languages = [code_language] if code_language else None

        enhanced_code_prompt = self.prompt_builder.build_code_prompt(
            topic=request.topic,
            domain=domain,
            career=career,
            audience_level=audience_level,
            languages=code_languages,
            content_language=content_lang
        )

        # Build RAG context section if documents are available
        rag_section = self._build_rag_section(request)

        return f"""Create a TRAINING VIDEO script for the following:

TOPIC: {request.topic}

PARAMETERS:
- Programming Language: {request.language}
- CONTENT LANGUAGE: {content_lang_name} (code: {content_lang})
- Target Duration: {duration_str} ({request.duration} seconds total)
- Target Audience: {request.target_audience}
- Visual Style: {request.style.value}

IMPORTANT: ALL text content (titles, subtitles, voiceover_text, bullet_points, content) MUST be written in {content_lang_name}.
- Code syntax and keywords stay in the programming language
- Code comments SHOULD be in {content_lang_name} for educational clarity

{rag_section}

{enhanced_code_prompt}

Please create a well-structured, educational TRAINING VIDEO that:
1. Introduces the topic clearly in {content_lang_name} - this is a FORMATION, not a conference
2. Explains concepts progressively with the teaching style appropriate for {request.target_audience}
3. Includes practical code examples that are 100% functional and follow ALL quality standards above
4. Has natural, engaging narration text in {content_lang_name} using TRAINING vocabulary
5. Ends with a clear summary in {content_lang_name}
6. If source documents are provided, BASE YOUR CONTENT ON THEM - they are the PRIMARY source

The training video should feel like a high-quality lesson from platforms like Udemy or Coursera.
NEVER use conference vocabulary ("presentation", "attendees"). Use training vocabulary ("formation", "leçon", "apprendre")."""

    def _parse_script(
        self,
        data: dict,
        request: GeneratePresentationRequest
    ) -> PresentationScript:
        """Parse GPT-4 response into PresentationScript"""
        slides = []

        for slide_data in data.get("slides", []):
            # Parse code blocks
            code_blocks = []
            for cb_data in slide_data.get("code_blocks", []):
                code_blocks.append(CodeBlock(
                    language=cb_data.get("language", request.language),
                    code=cb_data.get("code", ""),
                    filename=cb_data.get("filename"),
                    highlight_lines=cb_data.get("highlight_lines", []),
                    execution_order=cb_data.get("execution_order", 0),
                    expected_output=cb_data.get("expected_output"),
                    show_line_numbers=cb_data.get("show_line_numbers", True)
                ))

            # Parse slide type
            slide_type_str = slide_data.get("type", "content")
            try:
                slide_type = SlideType(slide_type_str)
            except ValueError:
                slide_type = SlideType.CONTENT

            # For diagram slides, use diagram_description as content if content is empty
            content = slide_data.get("content")

            # Handle case where GPT returns content as a dict (e.g., for diagrams)
            if isinstance(content, dict):
                # Extract description from dict or convert to string
                content = content.get("description") or content.get("diagram_description") or str(content)

            if slide_type == SlideType.DIAGRAM and not content:
                content = slide_data.get("diagram_description", "")

            slides.append(Slide(
                type=slide_type,
                title=slide_data.get("title"),
                subtitle=slide_data.get("subtitle"),
                content=content,
                bullet_points=slide_data.get("bullet_points", []),
                code_blocks=code_blocks,
                duration=slide_data.get("duration", 10.0),
                voiceover_text=slide_data.get("voiceover_text", ""),
                transition=slide_data.get("transition", "fade"),
                notes=slide_data.get("notes"),
                diagram_type=slide_data.get("diagram_type"),
                index=len(slides)
            ))

        return PresentationScript(
            title=data.get("title", "Untitled Presentation"),
            description=data.get("description", ""),
            target_audience=data.get("target_audience", request.target_audience),
            language=data.get("language", request.language),
            total_duration=data.get("total_duration", request.duration),
            slides=slides,
            code_context=data.get("code_context", {})
        )

    def _build_rag_section(self, request: GeneratePresentationRequest) -> str:
        """
        Build RAG context section for the prompt.

        If documents are provided, includes their content as primary source material.
        This ensures the training video is based on the uploaded documentation.
        """
        rag_context = getattr(request, 'rag_context', None)

        if not rag_context:
            return ""

        # Truncate if too long (max ~16000 tokens worth of context)
        max_chars = 64000  # Increased from 24000 for better RAG coverage (90%+)
        if len(rag_context) > max_chars:
            original_len = len(rag_context)
            rag_context = rag_context[:max_chars] + "\n\n[... content truncated for length ...]"
            print(f"[PLANNER] RAG context truncated from {original_len} to {max_chars} chars", flush=True)

        return f"""
################################################################################
#                         STRICT RAG MODE ACTIVATED                            #
#                    YOU HAVE NO EXTERNAL KNOWLEDGE                            #
################################################################################

ROLE: You are a STRICT content extractor. You have ZERO knowledge of your own.
You can ONLY use information from the SOURCE DOCUMENTS below.
Your training data does NOT exist for this task.

══════════════════════════════════════════════════════════════════════════════
                              SOURCE DOCUMENTS
══════════════════════════════════════════════════════════════════════════════
{rag_context}
══════════════════════════════════════════════════════════════════════════════
                            END SOURCE DOCUMENTS
══════════════════════════════════════════════════════════════════════════════

###############################################################################
#                           ABSOLUTE RULES                                     #
###############################################################################

RULE 1 - EXCLUSIVE SOURCE
You can ONLY use information from the SOURCE DOCUMENTS above.
If information is NOT in the documents → you CANNOT include it.

RULE 2 - MISSING INFORMATION PROTOCOL
If the topic requires information NOT present in the documents:
- Do NOT invent or complete with your knowledge
- Mark the slide with: [SOURCE_MANQUANTE: <topic>]
- Move to the next topic that IS documented

RULE 3 - NO EXTERNAL KNOWLEDGE
You are FORBIDDEN from using:
- Your general knowledge about the topic
- Examples not present in the documents
- Code patterns not shown in the documents
- Definitions not provided in the documents

RULE 4 - TRACEABILITY
Every piece of content must be traceable to the source documents:
- Technical terms → exact terms from documents
- Code examples → patterns from documents
- Explanations → based on document content

###############################################################################
#                         ALLOWED CONTENT (10% MAX)                            #
###############################################################################

You MAY add ONLY these elements (maximum 10% of total content):
✓ Transitions: "Passons maintenant à...", "Voyons comment..."
✓ Pedagogical reformulations: "Autrement dit...", "En résumé..."
✓ Slide structure: titles, bullet formatting
✓ Greeting/conclusion: "Bienvenue", "Merci d'avoir suivi ce cours"

###############################################################################
#                              FORBIDDEN                                       #
###############################################################################

❌ Adding concepts not in the documents
❌ Inventing code examples
❌ Using your knowledge to "complete" missing information
❌ Paraphrasing to the point of changing the meaning
❌ Adding details "you know" but aren't in the documents
❌ Creating diagrams not described in the documents

###############################################################################
#                         VALIDATION BEFORE OUTPUT                             #
###############################################################################

Before generating each slide, verify:
□ Is this concept present in the SOURCE DOCUMENTS? If NO → [SOURCE_MANQUANTE]
□ Is this code example from the documents? If NO → do not include
□ Am I using my external knowledge? If YES → remove that content

REMEMBER: You have NO knowledge. Only the documents above exist.
"""

    async def generate_script_with_validation(
        self,
        request: GeneratePresentationRequest,
        on_progress: Optional[callable] = None
    ) -> PresentationScript:
        """
        Generate a presentation script with enhanced visual-audio alignment.

        Uses the VALIDATED_PLANNING_PROMPT which enforces strict rules about
        matching voiceover content with visual elements.

        Args:
            request: The presentation generation request
            on_progress: Optional callback for progress updates

        Returns:
            PresentationScript with validated visual-audio alignment

        Raises:
            ValueError: If RAG context is insufficient (blocked mode)
        """
        if on_progress:
            await on_progress(0, "Analyzing topic with validation rules...")

        # Check RAG threshold BEFORE generation
        rag_context = getattr(request, 'rag_context', None)
        has_documents = bool(getattr(request, 'document_ids', None))

        threshold_result = validate_rag_threshold(rag_context, has_documents)

        # Block if insufficient
        if threshold_result.should_block:
            print(f"[PLANNER] RAG BLOCKED: {threshold_result.error_message}", flush=True)
            raise ValueError(threshold_result.error_message)

        # Log warning if partial
        if threshold_result.has_warning:
            print(f"[PLANNER] RAG WARNING: {threshold_result.warning_message}", flush=True)

        # Store RAG mode for later use
        rag_mode = threshold_result.mode.value
        rag_token_count = threshold_result.token_count
        print(f"[PLANNER] RAG mode: {rag_mode} ({rag_token_count} tokens)", flush=True)

        user_prompt = self._build_validated_prompt(request)

        if on_progress:
            await on_progress(10, "Generating validated presentation structure...")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": VALIDATED_PLANNING_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,  # Lower temperature for more consistent output
            max_tokens=4000,  # Max tokens for detailed descriptions
            response_format={"type": "json_object"}
        )

        if on_progress:
            await on_progress(80, "Parsing validated script...")

        content = response.choices[0].message.content
        script_data = json.loads(content)

        # Post-process to ensure IDs and calculate timing
        script_data = self._post_process_validated_script(script_data)

        script = self._parse_script(script_data, request)

        # RAG Verification: Check that generated content uses source documents
        if rag_context:
            rag_result = verify_rag_usage(script_data, rag_context, verbose=True)
            # Store verification result in script metadata
            script.rag_verification = {
                "mode": rag_mode,
                "token_count": rag_token_count,
                "coverage": rag_result.overall_coverage,
                "is_compliant": rag_result.is_compliant,
                "summary": rag_result.summary,
                "potential_hallucinations": len(rag_result.potential_hallucinations),
                "warning": threshold_result.warning_message,
            }
            print(f"[PLANNER] {rag_result.summary}", flush=True)
        else:
            # Store RAG mode even when no context (NONE mode)
            script.rag_verification = {
                "mode": rag_mode,
                "token_count": 0,
                "coverage": 0.0,
                "is_compliant": True,  # No RAG means no compliance check
                "summary": "No source documents - standard AI generation",
                "warning": threshold_result.warning_message,
            }

        if on_progress:
            await on_progress(100, "Validated script generation complete")

        return script

    def _build_validated_prompt(self, request: GeneratePresentationRequest) -> str:
        """Build prompt for validated script generation with enhanced code quality."""
        minutes = request.duration // 60
        seconds = request.duration % 60
        duration_str = f"{minutes} minutes" + (f" {seconds} seconds" if seconds > 0 else "")

        # Get content language name
        content_lang = getattr(request, 'content_language', 'en')
        content_lang_name = self._get_language_name(content_lang)

        # Detect domain, career, and audience level for enhanced prompts
        domain = self._detect_domain(request.topic, request.language)
        career = self._detect_career(request.topic)
        audience_level = self._map_audience_level(request.target_audience)
        code_language = self._detect_code_language(request.language)

        # Log detected career for debugging
        if career:
            print(f"[PLANNER] Detected career: {career.value}", flush=True)

        # Build enhanced code quality prompt using TechPromptBuilder
        code_languages = [code_language] if code_language else None

        enhanced_code_prompt = self.prompt_builder.build_code_prompt(
            topic=request.topic,
            domain=domain,
            career=career,
            audience_level=audience_level,
            languages=code_languages,
            content_language=content_lang
        )

        # Build RAG context section - THIS IS CRITICAL FOR ACCURACY
        rag_section = self._build_rag_section(request)
        has_rag = bool(rag_section)

        if has_rag:
            print(f"[PLANNER] Using RAG context for validated script generation", flush=True)

        return f"""Create a TRAINING VIDEO script for:

TOPIC: {request.topic}

PARAMETERS:
- Programming Language: {request.language}
- CONTENT LANGUAGE: {content_lang_name} (code: {content_lang})
- Target Duration: {duration_str} ({request.duration} seconds total)
- Target Audience: {request.target_audience}
- Visual Style: {request.style.value}
- HAS SOURCE DOCUMENTS: {"YES - USE THEM AS PRIMARY SOURCE (90%)" if has_rag else "NO"}

IMPORTANT LANGUAGE REQUIREMENT:
ALL text content (titles, subtitles, voiceover_text, bullet_points, content, notes) MUST be written in {content_lang_name}.
Code syntax stays in the programming language, but code comments SHOULD be in {content_lang_name}.

{rag_section}

{enhanced_code_prompt}

REQUIREMENTS:
1. {"BASE 90% OF CONTENT ON THE SOURCE DOCUMENTS ABOVE - DO NOT INVENT" if has_rag else "Create educational content based on the topic"}
2. {"USE DIAGRAMS/SCHEMAS FROM DOCUMENTS - describe them exactly as shown in the source" if has_rag else "Create appropriate diagrams for the topic"}
3. Every visual reference in voiceover MUST have corresponding visual content
4. Code slides: describe the code structure, NOT the execution
5. Code demo slides: CAN describe execution since output will be shown
6. Diagram slides: {"COPY diagram descriptions from source documents" if has_rag else "provide detailed diagram_description for generation"}
7. Calculate duration = max(voiceover_time, code_length/30, bullet_count*2) + 2s buffer
8. ALL text content MUST be in {content_lang_name}
9. ALL code must follow the CODE QUALITY STANDARDS above
10. This is a FORMATION/TRAINING - use pedagogical vocabulary, NOT conference vocabulary

{"CRITICAL: The source documents are the TRUTH. Do not hallucinate or invent content not in the documents." if has_rag else ""}

Create a well-structured TRAINING VIDEO in {content_lang_name} where the viewer sees EXACTLY what the narrator describes."""

    def _ensure_sync_anchors(self, script_data: dict) -> dict:
        """
        Ensure all slides have sync anchors in their voiceover text.

        If a slide's voiceover doesn't have a sync anchor, add one at the beginning.
        This enables timeline-based composition for precise audio-video sync.
        """
        import re

        slides = script_data.get("slides", [])

        for i, slide in enumerate(slides):
            slide_id = slide.get("id", f"slide_{i:03d}")
            voiceover = slide.get("voiceover_text", "")

            # Check if sync anchor already exists
            sync_pattern = r'\[SYNC:[\w_]+\]'
            has_sync = bool(re.search(sync_pattern, voiceover))

            if not has_sync and voiceover:
                # Add sync anchor at the beginning
                slide["voiceover_text"] = f"[SYNC:{slide_id}] {voiceover}"

        script_data["slides"] = slides
        return script_data

    def _post_process_validated_script(self, script_data: dict) -> dict:
        """Post-process the script to ensure consistency"""
        slides = script_data.get("slides", [])

        for i, slide in enumerate(slides):
            # Ensure ID exists
            if "id" not in slide:
                slide["id"] = f"slide_{i:03d}"

            # Calculate proper duration based on content
            voiceover = slide.get("voiceover_text", "")
            # Remove sync anchors for word count
            import re
            clean_voiceover = re.sub(r'\[SYNC:[\w_]+\]', '', voiceover).strip()
            word_count = len(clean_voiceover.split())
            voiceover_duration = word_count / 2.5  # 150 words/minute

            # Code animation duration
            animation_duration = 0
            for block in slide.get("code_blocks", []):
                code_len = len(block.get("code", ""))
                animation_duration = max(animation_duration, code_len / 30.0)

            # Visual reading time
            bullets = slide.get("bullet_points", [])
            visual_duration = len(bullets) * 2.0

            # Set duration to max + buffer
            calculated = max(voiceover_duration, animation_duration, visual_duration) + 2.0

            # Only override if significantly different
            current = slide.get("duration", 10.0)
            if abs(current - calculated) > 5:
                slide["duration"] = round(calculated, 1)

        script_data["slides"] = slides

        # Ensure sync anchors are present
        script_data = self._ensure_sync_anchors(script_data)

        return script_data

    async def refine_slide(
        self,
        slide: Slide,
        feedback: str,
        language: str
    ) -> Slide:
        """
        Refine a single slide based on feedback.

        Args:
            slide: The slide to refine
            feedback: User feedback for improvement
            language: Programming language context

        Returns:
            Refined Slide object
        """
        prompt = f"""Refine this presentation slide based on the feedback:

CURRENT SLIDE:
{json.dumps(slide.model_dump(), indent=2)}

FEEDBACK:
{feedback}

LANGUAGE: {language}

Return the improved slide as JSON with the same structure.
Output ONLY valid JSON."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a presentation expert. Improve slides based on feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        slide_data = json.loads(response.choices[0].message.content)

        # Parse code blocks
        code_blocks = []
        for cb_data in slide_data.get("code_blocks", []):
            code_blocks.append(CodeBlock(
                language=cb_data.get("language", language),
                code=cb_data.get("code", ""),
                filename=cb_data.get("filename"),
                highlight_lines=cb_data.get("highlight_lines", []),
                expected_output=cb_data.get("expected_output")
            ))

        slide_type_str = slide_data.get("type", slide.type.value)
        try:
            slide_type = SlideType(slide_type_str)
        except ValueError:
            slide_type = slide.type

        # For diagram slides, use diagram_description as content if content is empty
        content = slide_data.get("content")
        if slide_type == SlideType.DIAGRAM and not content:
            content = slide_data.get("diagram_description", "")

        return Slide(
            id=slide.id,
            type=slide_type,
            title=slide_data.get("title", slide.title),
            subtitle=slide_data.get("subtitle"),
            content=content,
            bullet_points=slide_data.get("bullet_points", []),
            code_blocks=code_blocks,
            duration=slide_data.get("duration", slide.duration),
            voiceover_text=slide_data.get("voiceover_text", slide.voiceover_text),
            transition=slide_data.get("transition", slide.transition),
            notes=slide_data.get("notes"),
            diagram_type=slide_data.get("diagram_type"),
            index=getattr(slide, 'index', 0)
        )
