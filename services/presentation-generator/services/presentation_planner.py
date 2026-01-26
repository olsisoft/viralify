"""
Presentation Planner Service

Uses LLM to generate a structured presentation script from a topic prompt.
Enhanced with TechPromptBuilder for domain-specific, professional-grade content.

Supports multiple providers via shared.llm_provider:
- OpenAI (GPT-4o, GPT-4o-mini)
- DeepSeek (V3.2 - 90% cheaper)
- Groq (Llama 3.3 - Ultra-fast)
- Mistral, Together AI, xAI Grok
"""
import json
import os
import re
from typing import Optional, List
import httpx

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import (
        get_llm_client,
        get_model_name,
        get_provider_config,
        get_provider,
        LLMProvider,
        print_provider_info,
    )
    from shared.training_logger import (
        log_training_example,
        TaskType,
    )
    USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    USE_SHARED_LLM = False
    log_training_example = None  # Fallback: no logging
    LLMProvider = None
    get_provider = None
    print("[PLANNER] Warning: shared.llm_provider not found, using direct OpenAI", flush=True)

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
from services.title_style_system import (
    TitleStyleSystem,
    TitleStyle as TitleStyleEnum,
    get_title_style_prompt,
    validate_slide_titles,
)

# NEXUS Adapter for enhanced pedagogical code generation (Phase 8B)
try:
    from services.nexus_adapter import (
        NexusAdapterService,
        NexusGenerationResult,
        NexusCodeSegment,
        get_nexus_adapter,
    )
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    NexusAdapterService = None
    NexusGenerationResult = None
    NexusCodeSegment = None
    get_nexus_adapter = None

# Feature flag for NEXUS code enhancement
USE_NEXUS_CODE_ENHANCEMENT = os.getenv("USE_NEXUS_CODE_ENHANCEMENT", "true").lower() == "true"


# Practical focus configuration - determines slide type distribution
PRACTICAL_FOCUS_CONFIG = {
    "theoretical": {
        "name": "Théorique (concepts)",
        "aliases": ["théorique", "theoretical", "concepts", "théorique (concepts)"],
        "slide_ratio": {
            "content": 0.50,      # 50% explanation slides
            "diagram": 0.25,      # 25% diagrams
            "code": 0.15,         # 15% code examples
            "code_demo": 0.05,    # 5% live demos
            "conclusion": 0.05,   # 5% summary
        },
        "instructions": """
═══════════════════════════════════════════════════════════════════════════════
                    THEORETICAL FOCUS - CONCEPTUAL LEARNING
═══════════════════════════════════════════════════════════════════════════════

This course emphasizes UNDERSTANDING over DOING. Follow these guidelines:

📚 CONTENT STRUCTURE:
- Prioritize deep conceptual understanding over hands-on practice
- Each concept should be explained thoroughly with WHY and HOW it works
- Use diagrams to visualize abstract concepts
- Code examples should ILLUSTRATE concepts, not be the main focus

📊 SLIDE TYPE REQUIREMENTS:
- 50% content slides (conceptual explanations)
- 25% diagram slides (visualizations)
- 15% code slides (illustrative examples)
- 5% code_demo slides (brief demonstrations)
- 5% conclusion slides

⚠️ IMPORTANT:
- Every code example MUST be preceded by conceptual explanation
- Include 'Why this works' sections BEFORE 'How to do it'
- Focus on mental models and understanding patterns
- Code should illustrate concepts, not be learned for its own sake
""",
    },
    "balanced": {
        "name": "Équilibré (50/50)",
        "aliases": ["équilibré", "balanced", "50/50", "équilibré (50/50)", "mixed"],
        "slide_ratio": {
            "content": 0.35,      # 35% explanation slides
            "diagram": 0.20,      # 20% diagrams
            "code": 0.25,         # 25% code examples
            "code_demo": 0.15,    # 15% live demos
            "conclusion": 0.05,   # 5% summary
        },
        "instructions": """
═══════════════════════════════════════════════════════════════════════════════
                    BALANCED FOCUS - THEORY + PRACTICE (50/50)
═══════════════════════════════════════════════════════════════════════════════

This course balances UNDERSTANDING with DOING. Follow these guidelines:

📚 CONTENT STRUCTURE:
- Equal emphasis on understanding concepts AND applying them
- Each concept: first explain (content slide), then show (code slide)
- Alternate between theory and practice throughout each section
- Diagrams should bridge theory and implementation

📊 SLIDE TYPE REQUIREMENTS:
- 35% content slides (explanations)
- 20% diagram slides (visualizations)
- 25% code slides (examples)
- 15% code_demo slides (demonstrations with output)
- 5% conclusion slides

⚠️ IMPORTANT:
- For every concept: explain WHY, then show HOW
- Include both 'why it works' and 'how to use it'
- Code examples should reinforce theoretical concepts
""",
    },
    "practical": {
        "name": "Très pratique (projets)",
        "aliases": ["pratique", "practical", "hands-on", "projets", "très pratique", "très pratique (projets)"],
        "slide_ratio": {
            "content": 0.20,      # 20% brief explanations
            "diagram": 0.10,      # 10% architecture diagrams
            "code": 0.35,         # 35% code examples
            "code_demo": 0.30,    # 30% live demos with output
            "conclusion": 0.05,   # 5% summary
        },
        "instructions": """
═══════════════════════════════════════════════════════════════════════════════
                    PRACTICAL FOCUS - HANDS-ON PROJECTS
═══════════════════════════════════════════════════════════════════════════════

This course emphasizes DOING over theoretical explanations. Follow these guidelines:

📚 CONTENT STRUCTURE:
- Prioritize learning by DOING over theoretical explanations
- Start with a BRIEF concept intro, then IMMEDIATELY show code
- Every lecture should include EXECUTABLE code examples
- Use code_demo slides to show REAL output and results
- Build towards a mini-project in each section

📊 SLIDE TYPE REQUIREMENTS:
- 20% content slides (brief context only)
- 10% diagram slides (architecture/flow only)
- 35% code slides (hands-on examples)
- 30% code_demo slides (with real output)
- 5% conclusion slides

⚠️ IMPORTANT:
- Minimum 65% of slides should be code or code_demo
- Theory should be MINIMAL - just enough context to understand the code
- Focus on 'how to build' rather than 'why it works'
- Include common errors and debugging tips
- Show REAL-WORLD use cases and practical applications
- Every section should end with a working mini-example
""",
    },
}


def parse_practical_focus(value: str | None) -> str:
    """Parse practical focus value to normalized key."""
    if not value:
        return "balanced"

    value_lower = value.lower().strip()

    for level_key, level_config in PRACTICAL_FOCUS_CONFIG.items():
        if value_lower in [alias.lower() for alias in level_config["aliases"]]:
            return level_key

    return "balanced"


def get_practical_focus_instructions(practical_focus: str | None) -> str:
    """Get instructions for the practical focus level."""
    level = parse_practical_focus(practical_focus)
    config = PRACTICAL_FOCUS_CONFIG.get(level, PRACTICAL_FOCUS_CONFIG["balanced"])
    return config["instructions"]


def get_practical_focus_slide_ratio(practical_focus: str | None) -> dict:
    """Get slide type ratio for the practical focus level."""
    level = parse_practical_focus(practical_focus)
    config = PRACTICAL_FOCUS_CONFIG.get(level, PRACTICAL_FOCUS_CONFIG["balanced"])
    return config["slide_ratio"]


PLANNING_SYSTEM_PROMPT = """You are an expert technical TRAINER and COURSE CREATOR for professional IT training programs. Your task is to create a structured TRAINING VIDEO script - NOT a conference talk, NOT a presentation for meetings.

CONTEXT: This is for an ONLINE TRAINING PLATFORM (like Udemy, Coursera, LinkedIn Learning).
- You are creating EDUCATIONAL CONTENT for learners who want to MASTER a skill
- The tone should be that of a TEACHER explaining to students, not a speaker at a conference
- Focus on LEARNING OBJECTIVES and SKILL ACQUISITION
- Include PRACTICAL EXERCISES and HANDS-ON examples
- Structure content for KNOWLEDGE RETENTION (not just information delivery)

##############################################################################
#                    CRITICAL: SLIDE vs VOICEOVER SEPARATION                  #
##############################################################################

THE SLIDE (what viewers SEE) and the VOICEOVER (what viewers HEAR) are DIFFERENT!

###############################################################################
#                         CONTENT SLIDES RULES                                 #
###############################################################################

For slides of type "content":
1. DO NOT use the "content" field - leave it EMPTY or null
2. Use ONLY "bullet_points" array for visual content
3. Minimum 5 bullet points, maximum 7 bullet points
4. Each bullet point: 3-7 words, descriptive but concise
5. NO paragraphs, NO introductory sentences on the slide

BULLET POINT FORMAT:
- NOT just one word: "Définition" ❌
- Descriptive phrase: "Définition d'un patron d'intégration" ✓
- NOT a full sentence: "Un patron est une solution réutilisable" ❌
- Keyword phrase: "Solution réutilisable aux problèmes courants" ✓

VOICEOVER (voiceover_text) - AUDIO ONLY:
- Full conversational sentences explaining EACH bullet point
- Detailed explanations for every point
- Natural speech flow
- Must explain ALL bullet points in order
- This is what the narrator SAYS while the slide is displayed

EXAMPLE - CORRECT:
{
  "type": "content",
  "title": "Qu'est-ce qu'un Patron d'Intégration ?",
  "content": null,
  "bullet_points": [
    "Définition d'un patron d'intégration",
    "Problèmes résolus par les patrons",
    "Avantages de la réutilisation",
    "Exemples concrets: Message Channel, Router",
    "Quand utiliser un patron d'intégration"
  ],
  "voiceover_text": "Commençons par comprendre ce qu'est un patron d'intégration. Premièrement, un patron d'intégration est une solution éprouvée et réutilisable à un problème récurrent dans l'intégration de systèmes. Deuxièmement, ces patrons résolvent des problèmes comme la communication entre applications hétérogènes, la gestion des erreurs de transmission, et la transformation de données. Troisièmement, l'avantage principal est la réutilisation - vous n'avez pas à réinventer la roue à chaque projet. Quatrièmement, parmi les exemples concrets, on trouve le Message Channel pour transporter les messages, et le Router pour diriger les messages vers la bonne destination. Enfin, vous devriez utiliser un patron quand vous reconnaissez un problème d'intégration classique dans votre architecture."
}

EXAMPLE - WRONG (DO NOT DO THIS):
{
  "type": "content",
  "title": "Qu'est-ce qu'un Patron d'Intégration ?",
  "content": "Un patron d'intégration est une solution éprouvée à un problème courant.",
  "bullet_points": ["Définition", "Importance", "Exemples"],
  "voiceover_text": "Un patron d'intégration est une solution éprouvée."
}

PROBLEMS with WRONG example:
- Has "content" field with paragraph text ❌
- Only 3 bullet points (need 5+) ❌
- Bullet points are single vague words ❌
- Voiceover doesn't explain each point ❌

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
- ⚠️ CONCEPT DEPENDENCY: NEVER use a technical term before explaining it! Example: don't say "use a decorator" before explaining what a decorator is

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
8. Duration should be based on voiceover length (~150 words per minute = 2.5 words/second)
9. SLIDE COUNT FORMULA: Calculate slides based on target duration:
   - Minimum 2 slides per minute of content
   - For 5 min (300s): 10-15 slides, For 3 min (180s): 6-9 slides, For 10 min (600s): 20-30 slides
   - Use the "Target Duration" parameter from the request to calculate
10. VOICEOVER LENGTH FORMULA (CRITICAL FOR DURATION):
   - Total words needed = target_duration_seconds × 2.5
   - Words per slide = total_words / number_of_slides
   - Each slide's voiceover_text MUST have 60-100 words minimum
   - For a 5-minute video: ~750 total words across all slides
   - For a 10-minute video: ~1500 total words across all slides
   - NEVER create short voiceovers under 40 words per slide
11. Progress from simple to complex concepts
12. Each slide's voiceover_text MUST start with [SYNC:slide_XXX] where XXX is the slide index (001, 002, etc.)

##############################################################################
#            CRITICAL: CONCEPT DEPENDENCY RULE (NEVER VIOLATE)               #
##############################################################################

A concept CANNOT be used in voiceover until it has been EXPLAINED first!

❌ WRONG ORDER (violates concept dependency):
- Slide 1: "Let's use a decorator to cache our function results"
- Slide 2: "What is a decorator? A decorator is..."
→ ERROR: "decorator" used BEFORE being explained!

✅ CORRECT ORDER (respects concept dependency):
- Slide 1: "What is a decorator? A decorator is a function that wraps another function..."
- Slide 2: "Now that we understand decorators, let's use one to cache results"
→ CORRECT: "decorator" explained BEFORE being used!

RULES FOR VOICEOVER:
1. NEVER use technical terms before defining them
2. NEVER reference code patterns before explaining them
3. NEVER assume the learner knows ANY concept not yet covered in THIS video
4. If you need to use a term, FIRST explain it in a previous slide
5. Use phrases like "Now that we've learned X, let's see how to use it for Y"
6. Each new concept must have its own explanation slide BEFORE any usage

This applies to ALL technical vocabulary: functions, classes, patterns, libraries,
frameworks, protocols, data structures, algorithms, etc.

##############################################################################

PEDAGOGICAL STRUCTURE FOR EACH CONCEPT (CRITICAL):
When teaching a concept from RAG documents, follow this strict 4-step structure:

┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: EXPLAIN THE CONCEPT (content slide)                                │
│  - What is it? Define the concept clearly                                   │
│  - Why is it important? Explain the purpose and benefits                    │
│  - 3-5 bullet points with key aspects                                       │
│  - Voiceover explains each point in detail                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  STEP 2: VISUALIZE WITH DIAGRAM (diagram slide)                             │
│  - Show a diagram that illustrates the concept                              │
│  - Use flowchart for processes, architecture for systems                    │
│  - Voiceover MUST describe each element of the diagram                      │
│  - Include spatial references (top, left, right, arrows)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  STEP 3: IMPLEMENT WITH CODE (code slide)                                   │
│  - Show working code that implements the concept                            │
│  - Voiceover explains each line/section                                     │
│  - Highlight key parts of the code                                          │
│  - Use proper syntax and best practices                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  STEP 4: DEMONSTRATE (code_demo slide)                                      │
│  - Execute the code to show it working                                      │
│  - Show expected output                                                     │
│  - Voiceover explains what the output means                                 │
│  - Connect output back to the original concept                              │
└─────────────────────────────────────────────────────────────────────────────┘

EXAMPLE SEQUENCE for teaching "API Endpoints":
1. content: "What are API Endpoints?" - explain REST, HTTP methods, URLs
2. diagram: architecture showing Client → API → Database flow
3. code: Python/Flask code showing a GET endpoint implementation
4. code_demo: Running the API and showing curl/Postman response

This 4-step structure ensures learners UNDERSTAND (explain), VISUALIZE (diagram),
IMPLEMENT (code), and VERIFY (demo) each concept. Repeat for each major concept.

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

DIAGRAM NARRATION (CRITICAL FOR LEARNING):
The voiceover for diagram slides MUST describe EACH element in a logical order so learners can follow along:

1. START with an overview: "Ce diagramme montre l'architecture de..." / "This diagram shows the architecture of..."
2. DESCRIBE elements in logical order:
   - For flowcharts: follow the flow from start to end
   - For architectures: top-to-bottom or left-to-right, then connections
   - For hierarchies: parent to children
3. NAME each component explicitly: "En haut, nous avons le Load Balancer..." / "At the top, we have the Load Balancer..."
4. EXPLAIN connections: "...qui distribue le trafic vers les serveurs API" / "...which distributes traffic to the API servers"
5. USE spatial references: "à gauche", "à droite", "en haut", "en bas", "au centre" / "on the left", "on the right", "at the top", "at the bottom", "in the center"
6. END with the value/outcome: "Ce flux permet de..." / "This flow enables..."

DIAGRAM VOICEOVER TEMPLATE:
"[Overview] Ce diagramme illustre [what it shows].
[Element 1] En haut/À gauche, nous avons [component name] qui [what it does].
[Element 2] Ensuite, [next component] [its role].
[Connection] [Component A] envoie/transmet [what] à [Component B].
[Continue for each element...]
[Conclusion] Grâce à cette architecture, [benefit/outcome]."

EXAMPLE (French):
"Ce diagramme montre l'architecture de notre système de traitement de données.
En haut à gauche, nous avons les Sources de Données, qui collectent les informations.
Ces données sont envoyées vers le Message Queue au centre, qui gère le flux.
Le Processing Engine, situé à droite, traite les messages en parallèle.
Finalement, les résultats sont stockés dans la Base de Données en bas.
Cette architecture permet un traitement scalable et fiable."

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

##############################################################################
#                    CRITICAL: SLIDE vs VOICEOVER SEPARATION                  #
##############################################################################

THE SLIDE (what viewers SEE) and the VOICEOVER (what viewers HEAR) are DIFFERENT!

###############################################################################
#                         CONTENT SLIDES RULES                                 #
###############################################################################

For slides of type "content":
1. DO NOT use the "content" field - leave it EMPTY or null
2. Use ONLY "bullet_points" array for visual content
3. Minimum 5 bullet points, maximum 7 bullet points
4. Each bullet point: 3-7 words, descriptive but concise
5. NO paragraphs, NO introductory sentences on the slide

BULLET POINT FORMAT:
- NOT just one word: "Définition" ❌
- Descriptive phrase: "Définition d'un patron d'intégration" ✓
- NOT a full sentence: "Un patron est une solution réutilisable" ❌
- Keyword phrase: "Solution réutilisable aux problèmes courants" ✓

VOICEOVER (voiceover_text) - AUDIO ONLY:
- Full conversational sentences explaining EACH bullet point
- Detailed explanations for every point
- Natural speech flow
- Must explain ALL bullet points in order
- This is what the narrator SAYS while the slide is displayed

EXAMPLE - CORRECT:
{
  "type": "content",
  "title": "Qu'est-ce qu'un Patron d'Intégration ?",
  "content": null,
  "bullet_points": [
    "Définition d'un patron d'intégration",
    "Problèmes résolus par les patrons",
    "Avantages de la réutilisation",
    "Exemples concrets: Message Channel, Router",
    "Quand utiliser un patron d'intégration"
  ],
  "voiceover_text": "Commençons par comprendre ce qu'est un patron d'intégration. Premièrement, un patron d'intégration est une solution éprouvée et réutilisable à un problème récurrent dans l'intégration de systèmes. Deuxièmement, ces patrons résolvent des problèmes comme la communication entre applications hétérogènes. Troisièmement, l'avantage principal est la réutilisation. Quatrièmement, parmi les exemples concrets, on trouve le Message Channel et le Router. Enfin, utilisez un patron quand vous reconnaissez un problème d'intégration classique."
}

EXAMPLE - WRONG (DO NOT DO THIS):
{
  "type": "content",
  "title": "Qu'est-ce qu'un Patron d'Intégration ?",
  "content": "Un patron d'intégration est une solution éprouvée à un problème courant.",
  "bullet_points": ["Définition", "Importance", "Exemples"],
  "voiceover_text": "Un patron d'intégration est une solution éprouvée."
}

PROBLEMS with WRONG example:
- Has "content" field with paragraph text ❌
- Only 3 bullet points (need 5+) ❌
- Bullet points are single vague words ❌
- Voiceover doesn't explain each point ❌

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

DIAGRAM NARRATION (CRITICAL FOR LEARNING):
The voiceover for diagram slides MUST describe EACH element in a logical order:

1. START with overview: "Ce diagramme montre..." / "This diagram shows..."
2. DESCRIBE elements in order (follow the flow, or top-to-bottom, left-to-right)
3. NAME each component: "En haut, nous avons le..." / "At the top, we have the..."
4. EXPLAIN connections: "...qui envoie les données vers..." / "...which sends data to..."
5. USE spatial references: "à gauche", "à droite", "en haut", "en bas", "au centre"
6. END with benefit: "Cette architecture permet de..." / "This architecture enables..."

DIAGRAM VOICEOVER EXAMPLE:
"Ce diagramme illustre notre pipeline de données.
En haut à gauche, les Sources collectent les données brutes.
Ces données transitent par le Message Queue au centre.
Le Processing Engine à droite traite les messages.
Les résultats sont stockés dans la Database en bas.
Ce flux garantit un traitement fiable et scalable."

Output ONLY valid JSON."""


class PresentationPlannerService:
    """Service for planning presentation structure using LLM (multi-provider)"""

    # Approximate token estimation (4 chars per token average)
    CHARS_PER_TOKEN = 4

    def __init__(self):
        # Use shared LLM provider if available, fallback to direct OpenAI
        if USE_SHARED_LLM:
            self.client = get_llm_client()
            self.model = get_model_name("quality")
            config = get_provider_config()
            self.max_context = config.max_context
            self.provider_name = config.name
            print(f"[PLANNER] Using {config.name} provider with model {self.model}", flush=True)
            print(f"[PLANNER] Max context: {self.max_context} tokens", flush=True)
        else:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=120.0,
                max_retries=2
            )
            self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
            self.max_context = 128000  # Default for OpenAI
            self.provider_name = "OpenAI"
            print(f"[PLANNER] Using direct OpenAI with model {self.model}", flush=True)

        # Initialize the tech prompt builder for enhanced code/diagram generation
        self.prompt_builder = TechPromptBuilder()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (approximate)."""
        if not text:
            return 0
        return len(text) // self.CHARS_PER_TOKEN

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if not text:
            return ""
        max_chars = max_tokens * self.CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[... content truncated due to provider limits ...]"

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

        # LOG: Debug duration received
        print(f"[PLANNER] REQUEST RECEIVED: duration={request.duration}s, topic={request.topic[:50]}...", flush=True)

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

        # Decide whether to use Chain of Density approach
        # For presentations > 5 minutes, Chain of Density is more reliable
        use_chain_of_density = (
            request.duration >= 300 and  # >= 5 minutes
            os.getenv("USE_CHAIN_OF_DENSITY", "true").lower() == "true"
        )

        if use_chain_of_density:
            if on_progress:
                await on_progress(5, "Using Chain of Density generation...")

            # Try Chain of Density first
            script_data = await self._generate_script_chain_of_density(request, on_progress)

            if script_data and len(script_data.get("slides", [])) >= 5:
                print(f"[PLANNER] Chain of Density SUCCESS: {len(script_data.get('slides', []))} slides", flush=True)
                # Skip validation since CoD already ensures correct slide count
            else:
                print(f"[PLANNER] Chain of Density failed, falling back to single-prompt", flush=True)
                use_chain_of_density = False  # Fall through to single-prompt

        if not use_chain_of_density or script_data is None:
            if on_progress:
                await on_progress(10, "Generating presentation structure...")

            # Original single-prompt approach (for short presentations or fallback)
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,  # Reduced from 0.7 for better instruction adherence
                max_tokens=4000,
                response_format={"type": "json_object"}
            )

            if on_progress:
                await on_progress(80, "Parsing presentation script...")

            # Parse the response with robust JSON handling
            content = response.choices[0].message.content
            script_data = self._parse_json_robust(content)

            # VALIDATION: Check slide count and regenerate if insufficient
            script_data = await self._validate_and_regenerate_if_needed(
                script_data, request, user_prompt, on_progress
            )

        # Log successful LLM response for training data collection
        # Note: Only log when using single-prompt approach (content/response exist)
        # Chain of Density has its own logging in _generate_slides_batch
        if log_training_example and not use_chain_of_density:
            log_training_example(
                messages=[
                    {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response=content,
                task_type=TaskType.PRESENTATION_PLANNING,
                model=self.model,
                input_tokens=getattr(response.usage, 'prompt_tokens', None),
                output_tokens=getattr(response.usage, 'completion_tokens', None),
                metadata={
                    "topic": request.topic,
                    "language": request.language,
                    "content_language": getattr(request, 'content_language', 'en'),
                    "duration": request.duration,
                    "target_audience": request.target_audience,
                    "slide_count": len(script_data.get("slides", [])),
                }
            )

        # Ensure sync anchors are present for timeline composition
        script_data = self._ensure_sync_anchors(script_data)

        # Validate slide titles against anti-patterns
        title_style = getattr(request, 'title_style', None)
        if title_style:
            try:
                style_enum = TitleStyleEnum(title_style.value if hasattr(title_style, 'value') else title_style)
            except (ValueError, AttributeError):
                style_enum = TitleStyleEnum.ENGAGING
        else:
            style_enum = TitleStyleEnum.ENGAGING

        # Get content language from request
        content_lang = getattr(request, 'content_language', 'en') or 'en'

        slides_for_validation = script_data.get("slides", [])
        title_validations = validate_slide_titles(slides_for_validation, style_enum)

        # ENFORCEMENT: Fix bad titles, don't just log them
        issues_count = sum(1 for v in title_validations if not v.is_valid)
        if issues_count > 0:
            print(f"[PLANNER] Title quality check: {issues_count}/{len(title_validations)} slides have title issues - FIXING", flush=True)
            for i, validation in enumerate(title_validations):
                if not validation.is_valid:
                    old_title = slides_for_validation[i].get("title", "Untitled")
                    print(f"[PLANNER]   Slide {i+1} '{old_title}': {', '.join(validation.issues)}", flush=True)

                    # FIX: Generate a better title based on slide content
                    slide = slides_for_validation[i]
                    new_title = self._generate_better_title(slide, style_enum, content_lang)
                    if new_title and new_title != old_title:
                        slides_for_validation[i]["title"] = new_title
                        print(f"[PLANNER]   → Fixed to: '{new_title}'", flush=True)

            # Update script_data with fixed titles
            script_data["slides"] = slides_for_validation
        else:
            print(f"[PLANNER] Title quality check: All {len(title_validations)} slides passed", flush=True)

        # VALIDATION: Ensure diagram slides have proper narration
        self._validate_and_fix_diagram_narration(script_data, content_lang)

        # VALIDATION: Ensure bullet points are SHORT (not full sentences)
        self._validate_and_fix_bullet_points(script_data, content_lang)

        # Convert to PresentationScript
        script = self._parse_script(script_data, request)

        # RAG Verification v3: Comprehensive multi-method verification
        if rag_context:
            rag_result = verify_rag_usage(script_data, rag_context, verbose=True, comprehensive=True)

            # Store comprehensive verification result in script metadata
            script.rag_verification = {
                "mode": rag_mode,
                "token_count": rag_token_count,
                # Semantic coverage
                "coverage": rag_result.overall_coverage,
                "is_compliant": rag_result.is_compliant,
                "summary": rag_result.summary,
                # Keyword validation
                "keyword_coverage": rag_result.keyword_coverage,
                "source_keywords_found": rag_result.source_keywords_found,
                "source_keywords_missing": rag_result.source_keywords_missing[:10],
                # Topic validation
                "topic_match_score": rag_result.topic_match_score,
                "source_topics": rag_result.source_topics[:10],
                "generated_topics": rag_result.generated_topics[:10],
                # Hallucinations
                "potential_hallucinations": len(rag_result.potential_hallucinations),
                "hallucination_details": [
                    {"slide": h.get("slide_index"), "similarity": h.get("similarity")}
                    for h in rag_result.potential_hallucinations[:5]
                ],
                # Failure analysis
                "failure_reasons": rag_result.failure_reasons,
                "warning": threshold_result.warning_message,
            }

            print(f"[PLANNER] {rag_result.summary}", flush=True)

            # Log detailed analysis if not compliant
            if not rag_result.is_compliant:
                print(f"[PLANNER] ❌ RAG VERIFICATION FAILED:", flush=True)
                print(f"[PLANNER]   - Semantic: {rag_result.overall_coverage:.1%}", flush=True)
                print(f"[PLANNER]   - Keywords: {rag_result.keyword_coverage:.1%}", flush=True)
                print(f"[PLANNER]   - Topics: {rag_result.topic_match_score:.1%}", flush=True)
                if rag_result.source_keywords_missing:
                    print(f"[PLANNER]   - Missing keywords: {rag_result.source_keywords_missing[:5]}", flush=True)
                print(f"[PLANNER]   - Source topics: {rag_result.source_topics[:5]}", flush=True)
                print(f"[PLANNER]   - Generated topics: {rag_result.generated_topics[:5]}", flush=True)
                print(f"[PLANNER]   ⚠️ Content may not be based on source documents!", flush=True)
        else:
            # Store RAG mode even when no context (NONE mode)
            script.rag_verification = {
                "mode": rag_mode,
                "token_count": 0,
                "coverage": 0.0,
                "is_compliant": True,  # No RAG means no compliance check
                "summary": "No source documents - standard AI generation",
                "keyword_coverage": 1.0,
                "topic_match_score": 1.0,
                "failure_reasons": [],
                "warning": threshold_result.warning_message,
            }

        # NEXUS Enhancement: Improve code quality on code/code_demo slides
        if USE_NEXUS_CODE_ENHANCEMENT and NEXUS_AVAILABLE:
            if on_progress:
                await on_progress(92, "Enhancing code with NEXUS...")
            try:
                script = await self._enhance_code_slides_with_nexus(script, request)
                print(f"[PLANNER] NEXUS code enhancement completed", flush=True)
            except Exception as e:
                print(f"[PLANNER] NEXUS enhancement failed (using original code): {e}", flush=True)
                # Continue with original code if NEXUS fails

        if on_progress:
            await on_progress(100, "Script generation complete")

        return script

    async def _enhance_code_slides_with_nexus(
        self,
        script: PresentationScript,
        request: GeneratePresentationRequest
    ) -> PresentationScript:
        """
        Enhance code slides using NEXUS multi-agent code generation.

        NEXUS provides:
        - Multi-agent quality assurance (Architect → Coder → Reviewer → Executor)
        - Audience-adapted code style
        - Synchronized narration scripts
        - Common mistakes and key concepts

        Args:
            script: The generated presentation script
            request: Original generation request

        Returns:
            Enhanced script with improved code
        """
        # Get code/code_demo slides
        code_slides = [
            (i, slide) for i, slide in enumerate(script.slides)
            if slide.type in [SlideType.CODE, SlideType.CODE_DEMO]
        ]

        if not code_slides:
            print("[PLANNER] No code slides to enhance with NEXUS", flush=True)
            return script

        print(f"[PLANNER] Enhancing {len(code_slides)} code slides with NEXUS...", flush=True)

        # Get NEXUS adapter
        nexus = get_nexus_adapter()

        # Check availability
        is_available = await nexus.is_available()
        if not is_available:
            print("[PLANNER] NEXUS engine not available, skipping enhancement", flush=True)
            return script

        # Map audience level for NEXUS
        audience_mapping = {
            "beginner": "student",
            "débutant": "student",
            "intermediate": "student",
            "intermédiaire": "student",
            "advanced": "developer",
            "avancé": "developer",
            "expert": "architect",
        }
        nexus_audience = audience_mapping.get(
            request.target_audience.lower(),
            "student"
        )

        # Map verbosity based on practical focus
        practical_focus = getattr(request, 'practical_focus', None) or 'balanced'
        verbosity_mapping = {
            "theoretical": "verbose",
            "balanced": "standard",
            "practical": "standard",
        }
        nexus_verbosity = verbosity_mapping.get(practical_focus, "standard")

        # Process each code slide
        for slide_idx, slide in code_slides:
            try:
                # Build context from slide content
                context_parts = []
                if slide.title:
                    context_parts.append(f"Topic: {slide.title}")
                if slide.voiceover_text:
                    context_parts.append(f"Context: {slide.voiceover_text[:200]}")
                if slide.code_blocks:
                    # Use existing code as reference
                    existing_code = slide.code_blocks[0].code
                    context_parts.append(f"Reference code:\n{existing_code}")

                project_description = "\n".join(context_parts)

                # Calculate time budget for this slide
                time_budget = int(slide.duration * 1.5)  # Give 50% more time

                # Call NEXUS for code generation
                result = await nexus.generate_code(
                    project_description=project_description,
                    lesson_context=f"Slide {slide_idx + 1} of {request.topic}",
                    skill_level=request.target_audience.lower(),
                    language=request.language,
                    target_audience=nexus_audience,
                    verbosity=nexus_verbosity,
                    allocated_time_seconds=time_budget,
                    max_segments=1,  # One segment per slide
                    show_mistakes=True,
                    show_evolution=False,
                )

                if result.code_segments:
                    segment = result.code_segments[0]

                    # Update the slide with NEXUS-generated code
                    if slide.code_blocks:
                        # Update existing code block
                        slide.code_blocks[0].code = segment.code
                        if segment.key_concepts:
                            slide.code_blocks[0].highlight_lines = []  # Reset
                    else:
                        # Create new code block
                        slide.code_blocks = [CodeBlock(
                            language=segment.language,
                            code=segment.code,
                            filename=segment.filename,
                        )]

                    # Enhance voiceover with NEXUS narration if available
                    if segment.narration_script and len(segment.narration_script) > 50:
                        # Append NEXUS insights to existing voiceover
                        original_voiceover = slide.voiceover_text or ""
                        nexus_insights = []

                        if segment.key_concepts:
                            concepts = ", ".join(segment.key_concepts[:3])
                            nexus_insights.append(f"Key concepts: {concepts}.")

                        if segment.common_mistakes and slide.type == SlideType.CODE_DEMO:
                            mistakes = segment.common_mistakes[0] if segment.common_mistakes else ""
                            if mistakes:
                                nexus_insights.append(f"A common mistake to avoid: {mistakes}.")

                        if nexus_insights:
                            enhanced_voiceover = f"{original_voiceover} {' '.join(nexus_insights)}"
                            slide.voiceover_text = enhanced_voiceover.strip()

                    # Store key concepts for quiz generation
                    if segment.key_concepts:
                        slide.key_takeaways = segment.key_concepts[:5]

                    print(f"[PLANNER] Enhanced slide {slide_idx + 1}: {segment.filename} ({len(segment.code)} chars)", flush=True)

            except Exception as e:
                print(f"[PLANNER] NEXUS enhancement failed for slide {slide_idx + 1}: {e}", flush=True)
                # Keep original code if NEXUS fails for this slide
                continue

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
        """Build the prompt for LLM with enhanced code quality standards."""
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

        # Build title style enhancement
        title_style = getattr(request, 'title_style', None)
        if title_style:
            # Convert from model enum to service enum
            try:
                style_enum = TitleStyleEnum(title_style.value if hasattr(title_style, 'value') else title_style)
            except (ValueError, AttributeError):
                style_enum = TitleStyleEnum.ENGAGING
            title_style_prompt = get_title_style_prompt(style_enum, content_lang)
            print(f"[PLANNER] Using title style: {style_enum.value}", flush=True)
        else:
            title_style_prompt = get_title_style_prompt(TitleStyleEnum.ENGAGING, content_lang)

        # Calculate dynamic slide count and word requirements based on duration
        duration_minutes = request.duration / 60
        min_slides = max(6, int(duration_minutes * 2))  # At least 2 slides per minute
        max_slides = max(10, int(duration_minutes * 3))  # Up to 3 slides per minute
        total_words_needed = int(request.duration * 2.5)  # 150 words/min = 2.5 words/sec
        words_per_slide = total_words_needed // max(min_slides, 1)

        print(f"[PLANNER] 📐 Duration requirements: {request.duration}s ({duration_minutes:.1f}min) → "
              f"slides:{min_slides}-{max_slides}, words:{total_words_needed}, words/slide:{words_per_slide}", flush=True)

        # Get practical focus configuration
        practical_focus = getattr(request, 'practical_focus', None)
        practical_focus_instructions = get_practical_focus_instructions(practical_focus)
        practical_focus_level = parse_practical_focus(practical_focus)
        slide_ratio = get_practical_focus_slide_ratio(practical_focus)

        if practical_focus:
            print(f"[PLANNER] 🎯 Practical focus: {practical_focus} (level: {practical_focus_level})", flush=True)

        return f"""Create a TRAINING VIDEO script for the following:

TOPIC: {request.topic}

PARAMETERS:
- Programming Language: {request.language}
- CONTENT LANGUAGE: {content_lang_name} (code: {content_lang})
- Target Duration: {duration_str} ({request.duration} seconds total)
- Target Audience: {request.target_audience}
- Visual Style: {request.style.value}
- Practical Focus: {PRACTICAL_FOCUS_CONFIG.get(practical_focus_level, PRACTICAL_FOCUS_CONFIG["balanced"])["name"]}

═══════════════════════════════════════════════════════════════════════════════
                    CRITICAL DURATION REQUIREMENTS (READ CAREFULLY)
═══════════════════════════════════════════════════════════════════════════════

To achieve the target duration of {duration_str}, you MUST follow these requirements:

📊 SLIDE COUNT: Create between {min_slides} and {max_slides} slides
   - Calculated as: {duration_minutes:.1f} minutes × 2-3 slides/minute

📝 TOTAL WORDS NEEDED: ~{total_words_needed} words across all voiceovers
   - Calculated as: {request.duration} seconds × 2.5 words/second (150 words/min speaking rate)

📄 WORDS PER SLIDE: Each slide's voiceover_text should have ~{words_per_slide} words
   - MINIMUM 60 words per slide (shorter = video too short!)
   - IDEAL: 70-100 words per slide for proper pacing

⚠️ COMMON MISTAKE: Creating short voiceovers (20-40 words) results in videos under 2 minutes!
   Each voiceover should be a full paragraph explaining the slide content in detail.

{practical_focus_instructions}

REQUIRED SLIDE TYPE DISTRIBUTION (based on practical focus):
- content slides: ~{int(slide_ratio['content']*100)}% ({int(min_slides * slide_ratio['content'])}-{int(max_slides * slide_ratio['content'])} slides)
- diagram slides: ~{int(slide_ratio['diagram']*100)}% ({int(min_slides * slide_ratio['diagram'])}-{int(max_slides * slide_ratio['diagram'])} slides)
- code slides: ~{int(slide_ratio['code']*100)}% ({int(min_slides * slide_ratio['code'])}-{int(max_slides * slide_ratio['code'])} slides)
- code_demo slides: ~{int(slide_ratio['code_demo']*100)}% ({int(min_slides * slide_ratio['code_demo'])}-{int(max_slides * slide_ratio['code_demo'])} slides)
- conclusion slides: ~{int(slide_ratio['conclusion']*100)}%

IMPORTANT: ALL text content (titles, subtitles, voiceover_text, bullet_points, content) MUST be written in {content_lang_name}.
- Code syntax and keywords stay in the programming language
- Code comments SHOULD be in {content_lang_name} for educational clarity

{rag_section}

{enhanced_code_prompt}

{title_style_prompt}

Please create a well-structured, educational TRAINING VIDEO that:
1. Introduces the topic clearly in {content_lang_name} - this is a FORMATION, not a conference
2. Explains concepts progressively with the teaching style appropriate for {request.target_audience}
3. Includes practical code examples that are 100% functional and follow ALL quality standards above
4. Has natural, engaging narration text in {content_lang_name} using TRAINING vocabulary
5. Ends with a clear summary in {content_lang_name}
6. If source documents are provided, BASE YOUR CONTENT ON THEM - they are the PRIMARY source
7. CRITICAL: Ensure each voiceover_text has {words_per_slide}+ words to meet the {duration_str} target duration
8. FOLLOW THE SLIDE TYPE DISTRIBUTION above based on the practical focus level
9. ⚠️ CONCEPT DEPENDENCY RULE: NEVER use a technical term in voiceover BEFORE explaining it!
   - WRONG: "Let's use a decorator" → then later "What is a decorator?"
   - CORRECT: "What is a decorator? It's..." → then "Now let's use a decorator"
   - Each concept MUST be explained BEFORE being used in any context

The training video should feel like a high-quality lesson from platforms like Udemy or Coursera.
NEVER use conference vocabulary ("presentation", "attendees"). Use training vocabulary ("formation", "leçon", "apprendre")."""

    def _generate_better_title(
        self,
        slide: dict,
        style: 'TitleStyleEnum',
        language: str
    ) -> str:
        """
        Generate a better title for a slide that failed validation.

        Uses slide content to create a specific, engaging title
        that follows the title style guidelines.
        """
        # Extract key content from slide (handle None values properly)
        content = slide.get("content") or ""
        bullet_points = slide.get("bullet_points") or []
        voiceover = slide.get("voiceover_text") or ""
        slide_type = slide.get("type") or "content"

        # Filter out None values from bullet_points list
        bullet_points = [bp for bp in bullet_points if bp is not None]

        # Build context from slide content
        context_text = f"{content} {' '.join(bullet_points)} {voiceover}"[:500]

        # Extract key concepts (simple keyword extraction)
        import re
        words = re.findall(r'\b[A-Za-zÀ-ÿ]{4,}\b', context_text)
        # Filter common words, safety filter for "None" literal, and prompt leakage terms
        stopwords = {'dans', 'avec', 'pour', 'cette', 'sont', 'nous', 'vous', 'leur', 'plus',
                     'tout', 'comme', 'elle', 'fait', 'être', 'avoir', 'faire', 'peut',
                     'from', 'with', 'that', 'this', 'have', 'will', 'your', 'which',
                     'none', 'null', 'undefined',
                     # Prompt leakage terms - prevent LLM hallucinations from internal markers
                     'sync', 'anchor', 'marker', 'slide', 'placeholder', 'example',
                     'bienvenue', 'welcome', 'tutorial', 'module', 'section', 'content',
                     'lecon', 'lesson', 'title', 'introduction', 'conclusion', 'chapitre',
                     'chapter', 'partie', 'part', 'cours', 'course', 'formation',
                     'training', 'video', 'presentation', 'diapositive', 'voiceover',
                     'narration', 'script', 'texte', 'text'}
        key_words = [w for w in words if w.lower() not in stopwords][:5]

        if not key_words:
            return None  # Can't generate a better title without content

        # Generate title based on style and type
        main_concept = key_words[0].capitalize() if key_words else "Concept"

        if language.startswith('fr'):
            # French title patterns
            patterns = {
                'title': f"Maîtriser {main_concept}",
                'content': f"Les secrets de {main_concept}",
                'code': f"{main_concept} en pratique",
                'diagram': f"Comprendre l'architecture de {main_concept}",
                'conclusion': f"Ce que vous savez maintenant sur {main_concept}",
            }
        else:
            # English title patterns
            patterns = {
                'title': f"Mastering {main_concept}",
                'content': f"The Power of {main_concept}",
                'code': f"{main_concept} in Action",
                'diagram': f"Understanding {main_concept} Architecture",
                'conclusion': f"What You Now Know About {main_concept}",
            }

        new_title = patterns.get(slide_type, patterns['content'])

        # Add secondary concept if available
        if len(key_words) > 1:
            if language.startswith('fr'):
                new_title = f"{new_title} avec {key_words[1].capitalize()}"
            else:
                new_title = f"{new_title} with {key_words[1].capitalize()}"

        return new_title

    def _validate_and_fix_diagram_narration(
        self,
        script_data: dict,
        language: str
    ) -> None:
        """
        Validate and fix diagram slide narrations to ensure they properly explain
        the visual content. This is critical for learning - viewers need to
        understand what they're seeing.

        Checks:
        1. Voiceover is not empty
        2. Voiceover mentions diagram elements (from content/description)
        3. Voiceover uses spatial references (en haut, à gauche, etc.)
        4. Voiceover is long enough to properly explain the diagram
        """
        slides = script_data.get("slides", [])

        # Spatial reference keywords by language
        spatial_keywords = {
            'fr': ['en haut', 'en bas', 'à gauche', 'à droite', 'au centre', 'ensuite',
                   'puis', 'vers', 'depuis', 'entre', 'connecté', 'envoie', 'reçoit'],
            'en': ['at the top', 'at the bottom', 'on the left', 'on the right', 'in the center',
                   'then', 'next', 'to', 'from', 'between', 'connected', 'sends', 'receives'],
            'es': ['arriba', 'abajo', 'a la izquierda', 'a la derecha', 'en el centro',
                   'luego', 'después', 'hacia', 'desde', 'entre', 'conectado', 'envía', 'recibe'],
        }

        # Get appropriate keywords for language
        lang_prefix = language[:2] if language else 'en'
        spatial_refs = spatial_keywords.get(lang_prefix, spatial_keywords['en'])

        for i, slide in enumerate(slides):
            slide_type = slide.get("type", "")

            # Only check diagram slides
            if slide_type != "diagram":
                continue

            voiceover = (slide.get("voiceover_text") or "").lower()
            content = (slide.get("content") or "").lower()
            diagram_desc = (slide.get("diagram_description") or "").lower()

            # Combine content sources
            diagram_context = f"{content} {diagram_desc}".strip()

            issues = []

            # Check 1: Voiceover is not empty
            if len(voiceover.strip()) < 50:
                issues.append("voiceover_too_short")

            # Check 2: Voiceover should mention some diagram elements
            # Extract key concepts from diagram content
            import re
            content_words = set(re.findall(r'\b[A-Za-zÀ-ÿ]{4,}\b', diagram_context))
            common_stopwords = {'dans', 'avec', 'pour', 'cette', 'sont', 'nous', 'vous',
                               'from', 'with', 'that', 'this', 'have', 'will', 'which',
                               'diagram', 'diagramme', 'montre', 'shows', 'architecture'}
            key_concepts = [w for w in content_words if w.lower() not in common_stopwords][:8]

            # Check how many key concepts are mentioned in voiceover
            concepts_mentioned = sum(1 for c in key_concepts if c.lower() in voiceover)
            if key_concepts and concepts_mentioned < len(key_concepts) * 0.3:
                issues.append("missing_diagram_elements")

            # Check 3: Voiceover uses spatial references
            has_spatial = any(ref in voiceover for ref in spatial_refs)
            if not has_spatial and len(voiceover) > 100:
                issues.append("no_spatial_references")

            if issues:
                print(f"[PLANNER] Diagram slide {i+1} narration issues: {issues}", flush=True)

                # FIX: Enhance the voiceover with better narration
                enhanced_voiceover = self._enhance_diagram_voiceover(
                    slide, language, key_concepts, spatial_refs
                )
                if enhanced_voiceover and len(enhanced_voiceover) > len(slide.get("voiceover_text") or ""):
                    slides[i]["voiceover_text"] = enhanced_voiceover
                    print(f"[PLANNER]   → Enhanced diagram voiceover ({len(enhanced_voiceover)} chars)", flush=True)

        script_data["slides"] = slides

    async def _validate_and_regenerate_if_needed(
        self,
        script_data: dict,
        request: GeneratePresentationRequest,
        original_prompt: str,
        on_progress: Optional[callable] = None,
        max_attempts: int = 2
    ) -> dict:
        """
        Validate slide count and regenerate if insufficient.

        The LLM sometimes ignores slide count requirements. This method:
        1. Checks if generated slides are within expected range
        2. Regenerates with stricter prompt if too few slides
        3. Maximum 2 regeneration attempts to avoid infinite loops

        Returns:
            Updated script_data with sufficient slides
        """
        duration_minutes = request.duration / 60
        min_slides = max(6, int(duration_minutes * 2))  # At least 2 slides per minute
        max_slides = max(10, int(duration_minutes * 3))  # Up to 3 slides per minute

        current_slides = len(script_data.get("slides", []))

        # Check if slide count is acceptable
        if current_slides >= min_slides:
            print(f"[PLANNER] SLIDE COUNT OK: {current_slides} slides (required: {min_slides}-{max_slides})", flush=True)
            return script_data

        # Need to regenerate - slide count is too low
        print(f"[PLANNER] SLIDE COUNT LOW: {current_slides} slides < {min_slides} minimum - REGENERATING", flush=True)

        for attempt in range(1, max_attempts + 1):
            if on_progress:
                await on_progress(50 + attempt * 10, f"Regenerating (attempt {attempt}/{max_attempts})...")

            # Build a stricter prompt emphasizing slide count
            strict_prompt = self._build_strict_slide_count_prompt(
                request, min_slides, max_slides, current_slides, attempt
            )

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                        {"role": "user", "content": strict_prompt}
                    ],
                    temperature=0.3,  # Lower temperature for better instruction following
                    max_tokens=8000,  # More tokens for more slides
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                new_script_data = self._parse_json_robust(content)
                new_slides = len(new_script_data.get("slides", []))

                print(f"[PLANNER] REGENERATION attempt {attempt}: {new_slides} slides generated", flush=True)

                if new_slides >= min_slides:
                    print(f"[PLANNER] SLIDE COUNT FIXED: {new_slides} slides (required: {min_slides}-{max_slides})", flush=True)
                    # Ensure sync anchors
                    new_script_data = self._ensure_sync_anchors(new_script_data)
                    return new_script_data
                else:
                    current_slides = new_slides
                    script_data = new_script_data

            except Exception as e:
                print(f"[PLANNER] REGENERATION attempt {attempt} failed: {e}", flush=True)

        # After max attempts, use what we have
        print(f"[PLANNER] SLIDE COUNT WARNING: Only {current_slides} slides after {max_attempts} attempts (required: {min_slides})", flush=True)
        return script_data

    def _build_strict_slide_count_prompt(
        self,
        request: GeneratePresentationRequest,
        min_slides: int,
        max_slides: int,
        current_slides: int,
        attempt: int
    ) -> str:
        """Build a stricter prompt that emphasizes slide count requirements."""
        duration_minutes = request.duration / 60
        total_words = int(request.duration * 2.5)
        words_per_slide = total_words // min_slides

        # Get content language
        content_lang = getattr(request, 'content_language', 'en') or 'en'
        lang_names = {'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German'}
        content_lang_name = lang_names.get(content_lang, content_lang.upper())

        # Get RAG context if available
        rag_section = ""
        rag_context = getattr(request, 'rag_context', None)
        if rag_context:
            rag_section = f"""
=== SOURCE DOCUMENTS (BASE YOUR CONTENT ON THIS) ===
{rag_context[:15000]}
=== END SOURCE DOCUMENTS ===
"""

        strict_prompt = f"""
⚠️ CRITICAL: YOUR PREVIOUS RESPONSE HAD ONLY {current_slides} SLIDES - THIS IS INSUFFICIENT!

YOU MUST CREATE AT LEAST {min_slides} SLIDES. This is a HARD REQUIREMENT.

TOPIC: {request.topic}
DURATION: {request.duration} seconds ({duration_minutes:.1f} minutes)
LANGUAGE: {content_lang_name}

{rag_section}

📊 MANDATORY SLIDE COUNT: {min_slides} to {max_slides} slides
   - This is calculated from: {duration_minutes:.1f} minutes × 2-3 slides per minute
   - You generated only {current_slides} slides before - THIS IS NOT ACCEPTABLE
   - Each slide should have ~{words_per_slide} words in voiceover_text

📝 TOTAL WORDS NEEDED: {total_words} words across all voiceovers
   - At 150 words/minute speaking rate
   - Distribute evenly across {min_slides}+ slides

STRUCTURE YOUR PRESENTATION WITH THESE SECTIONS:
1. Introduction (2-3 slides): Present the topic, objectives, what will be covered
2. Core Concepts ({"8-12" if min_slides > 20 else "4-6"} slides): Explain main ideas with examples
3. Implementation/Practice ({"8-12" if min_slides > 20 else "4-6"} slides): Show code, diagrams, practical applications
4. Advanced Topics ({"4-6" if min_slides > 20 else "2-4"} slides): Deeper concepts, best practices
5. Summary & Conclusion (2-3 slides): Recap key points, next steps

SLIDE TYPES TO USE:
- title: Opening slide
- content: Explanatory slides with bullet points
- diagram: Visual explanations (architecture, flow, etc.)
- code: Code examples with explanations
- code_demo: Code with expected output
- conclusion: Summary slide

REMEMBER:
- MINIMUM {min_slides} SLIDES - DO NOT GENERATE FEWER
- Each slide needs substantial voiceover_text (~{words_per_slide} words)
- Cover the topic thoroughly - don't rush through concepts
- ALL content must be in {content_lang_name}

Generate the presentation now with AT LEAST {min_slides} slides:
"""
        return strict_prompt

    # =========================================================================
    # CHAIN OF DENSITY: Two-phase generation for reliable slide counts
    # =========================================================================

    async def _generate_script_chain_of_density(
        self,
        request: GeneratePresentationRequest,
        on_progress: Optional[callable] = None
    ) -> dict:
        """
        Generate presentation using Chain of Density approach.

        Instead of generating all slides in one prompt (which often fails for 20+ slides),
        we split into two phases:

        Phase 1: Generate outline (titles + descriptions)
        Phase 2: Generate content in batches of 5 slides

        This is more reliable and often cheaper (no regeneration needed).
        """
        duration_minutes = request.duration / 60
        target_slides = max(8, int(duration_minutes * 2.5))  # ~2.5 slides per minute

        print(f"[PLANNER] Using CHAIN OF DENSITY: {target_slides} slides for {duration_minutes:.1f} min", flush=True)

        # Get content language
        content_lang = getattr(request, 'content_language', 'en') or 'en'
        lang_names = {'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German'}
        content_lang_name = lang_names.get(content_lang, content_lang.upper())

        # PHASE 1: Generate outline
        if on_progress:
            await on_progress(10, "Phase 1: Generating outline...")

        outline = await self._generate_outline(request, target_slides, content_lang_name)

        if not outline or len(outline) < 5:
            print(f"[PLANNER] Outline generation failed, falling back to single-prompt", flush=True)
            return None  # Signal to use fallback

        print(f"[PLANNER] Outline generated: {len(outline)} slides planned", flush=True)

        # PHASE 2: Generate content in batches
        all_slides = []
        batch_size = 5
        total_batches = (len(outline) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(outline))
            batch_outline = outline[start_idx:end_idx]

            progress_pct = 20 + int((batch_idx / total_batches) * 60)
            if on_progress:
                await on_progress(progress_pct, f"Phase 2: Generating slides {start_idx + 1}-{end_idx}...")

            print(f"[PLANNER] Generating batch {batch_idx + 1}/{total_batches}: slides {start_idx + 1}-{end_idx}", flush=True)

            batch_slides = await self._generate_slides_batch(
                request, batch_outline, start_idx, content_lang_name
            )

            if batch_slides:
                all_slides.extend(batch_slides)
            else:
                # If batch fails, create placeholder slides from outline
                print(f"[PLANNER] Batch {batch_idx + 1} failed, using outline as fallback", flush=True)
                for i, item in enumerate(batch_outline):
                    all_slides.append({
                        "type": item.get("type", "content"),
                        "title": item.get("title", f"Slide {start_idx + i + 1}"),
                        "subtitle": item.get("subtitle"),
                        "bullet_points": item.get("key_points", []),
                        "voiceover_text": item.get("description", ""),
                        "duration": 30
                    })

        print(f"[PLANNER] Chain of Density complete: {len(all_slides)} slides generated", flush=True)

        # Build final script_data
        script_data = {
            "title": request.topic,
            "slides": all_slides
        }

        return script_data

    async def _generate_outline(
        self,
        request: GeneratePresentationRequest,
        target_slides: int,
        content_lang_name: str
    ) -> list:
        """
        Phase 1: Generate outline with slide titles and brief descriptions.

        This is a focused prompt that's easier for LLMs to follow.
        """
        # Get RAG context if available
        rag_context = getattr(request, 'rag_context', None) or ""
        rag_section = ""
        if rag_context and len(rag_context) > 100:
            rag_section = f"""
SOURCE DOCUMENTS (use these as the primary source):
{rag_context[:8000]}
"""

        outline_prompt = f"""You are creating an OUTLINE for a {request.duration // 60}-minute training video on: {request.topic}

Target audience: {request.target_audience}
Content language: {content_lang_name}

{rag_section}

Generate an outline with EXACTLY {target_slides} slides. Each slide needs:
- title: Clear, engaging title (4-8 words)
- type: One of "title", "content", "code", "diagram", "conclusion"
- description: One sentence about what this slide covers
- key_points: 3-5 bullet points to cover

IMPORTANT:
- First slide must be type "title" (introduction)
- Last slide must be type "conclusion"
- Include 1-2 "code" slides if the topic is technical
- Include 1 "diagram" slide for architecture/process topics
- ALL text must be in {content_lang_name}

Return JSON array:
[
  {{"title": "...", "type": "title", "description": "...", "key_points": ["...", "..."]}},
  {{"title": "...", "type": "content", "description": "...", "key_points": ["...", "..."]}},
  ...
]

Generate EXACTLY {target_slides} slides:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert course planner. Generate structured outlines in JSON format."},
                    {"role": "user", "content": outline_prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = self._parse_json_robust(content)

            # Handle both array and object responses
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return result.get("slides") or result.get("outline") or list(result.values())[0] if result else []

            return []

        except Exception as e:
            print(f"[PLANNER] Outline generation error: {e}", flush=True)
            return []

    async def _generate_slides_batch(
        self,
        request: GeneratePresentationRequest,
        batch_outline: list,
        start_index: int,
        content_lang_name: str
    ) -> list:
        """
        Phase 2: Generate full slide content for a batch of 5 slides.

        Given the outline, generate complete slides with voiceover text.
        """
        # Get RAG context if available
        rag_context = getattr(request, 'rag_context', None) or ""
        rag_section = ""
        if rag_context and len(rag_context) > 100:
            # Use a portion of RAG context relevant to this batch
            rag_section = f"""
SOURCE DOCUMENTS (use ONLY this content, do not invent):
{rag_context[:6000]}
"""

        # Build outline description for context
        outline_desc = "\n".join([
            f"{start_index + i + 1}. [{item.get('type', 'content')}] {item.get('title', '')}: {item.get('description', '')}"
            for i, item in enumerate(batch_outline)
        ])

        batch_prompt = f"""Generate FULL CONTENT for slides {start_index + 1} to {start_index + len(batch_outline)} of a training video.

Topic: {request.topic}
Target audience: {request.target_audience}
Content language: {content_lang_name}

{rag_section}

OUTLINE FOR THESE SLIDES:
{outline_desc}

For EACH slide, generate:
- type: Keep the type from outline
- title: Keep or improve the title
- subtitle: Optional subtitle
- bullet_points: 5-7 concise bullet points (3-7 words each)
- voiceover_text: Natural spoken narration (50-80 words per slide)
- duration: Estimated seconds (25-40s per slide)

For CODE slides, also include:
- code_blocks: Array with {{"language": "python", "code": "...", "description": "..."}}

For DIAGRAM slides, also include:
- diagram_type: "flowchart", "architecture", or "process"

IMPORTANT:
- voiceover_text should sound natural when read aloud
- bullet_points are for VISUAL display (short phrases, not sentences)
- ALL text must be in {content_lang_name}

Return JSON:
{{"slides": [
  {{"type": "...", "title": "...", "bullet_points": [...], "voiceover_text": "...", "duration": 30}},
  ...
]}}

Generate content for slides {start_index + 1}-{start_index + len(batch_outline)}:"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert course content creator. Generate detailed slide content in JSON format."},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.4,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            result = self._parse_json_robust(content)

            # Handle response
            if isinstance(result, dict):
                slides = result.get("slides", [])
                if slides:
                    return slides

            if isinstance(result, list):
                return result

            return []

        except Exception as e:
            print(f"[PLANNER] Batch generation error: {e}", flush=True)
            return []

    def _validate_and_fix_bullet_points(
        self,
        script_data: dict,
        language: str
    ) -> None:
        """
        Validate and fix bullet points for content slides.

        Rules:
        1. Content slides should NOT have 'content' field (only bullet_points)
        2. Minimum 5 bullet points per content slide
        3. Bullet points should be 3-7 words (not single words, not sentences)

        This post-processing step catches cases where the LLM still
        generates incorrect content despite instructions.
        """
        slides = script_data.get("slides", [])
        max_words_per_bullet = 8
        min_words_per_bullet = 2
        min_bullet_count = 5

        fixed_count = 0

        for i, slide in enumerate(slides):
            slide_type = slide.get("type", "content")

            # Skip non-content slides
            if slide_type not in ["content"]:
                continue

            changes_made = []

            # FIX 1: Remove 'content' field from content slides (should only have bullet_points)
            content = slide.get("content")
            if content and isinstance(content, str) and len(content.strip()) > 0:
                # Save content for potential use in expanding bullet points
                saved_content = content
                slide["content"] = None
                changes_made.append(f"Removed paragraph content ({len(saved_content)} chars)")

            # FIX 2: Ensure minimum bullet points
            bullet_points = slide.get("bullet_points") or []  # Handle None case

            if len(bullet_points) < min_bullet_count:
                # Try to expand bullet points based on slide title or voiceover
                voiceover = slide.get("voiceover_text") or ""
                title = slide.get("title") or ""

                # Extract additional topics from voiceover
                additional_bullets = self._extract_bullet_topics(voiceover, bullet_points, language)

                # Add additional bullets up to minimum
                while len(bullet_points) < min_bullet_count and additional_bullets:
                    bullet_points.append(additional_bullets.pop(0))

                if len(bullet_points) < min_bullet_count:
                    changes_made.append(f"WARNING: Only {len(bullet_points)} bullets (need {min_bullet_count})")
                else:
                    changes_made.append(f"Expanded to {len(bullet_points)} bullets")

            # FIX 3: Fix bullet points that are too short (single words) or too long
            fixed_bullets = []
            for bullet in bullet_points:
                words = bullet.split()

                if len(words) < min_words_per_bullet:
                    # Expand single-word bullet to be more descriptive
                    expanded = self._expand_short_bullet(bullet, slide, language)
                    fixed_bullets.append(expanded)
                    changes_made.append(f"Expanded '{bullet}' → '{expanded}'")
                elif len(words) > max_words_per_bullet:
                    # Truncate long bullet
                    truncated = self._extract_key_phrase(bullet, language)
                    fixed_bullets.append(truncated)
                    changes_made.append(f"Truncated bullet to {len(truncated.split())} words")
                else:
                    fixed_bullets.append(bullet)

            slide["bullet_points"] = fixed_bullets

            if changes_made:
                fixed_count += 1
                print(f"[PLANNER] Slide {i+1} '{slide.get('title', 'Untitled')[:30]}': {len(changes_made)} fixes", flush=True)
                for change in changes_made[:3]:
                    print(f"[PLANNER]   - {change}", flush=True)

        if fixed_count > 0:
            print(f"[PLANNER] Fixed content slides: {fixed_count} slides modified", flush=True)
        else:
            print(f"[PLANNER] Bullet point validation: All slides pass", flush=True)

        script_data["slides"] = slides

    def _extract_bullet_topics(self, voiceover: str, existing_bullets: list, language: str) -> list:
        """Extract additional bullet point topics from voiceover text."""
        import re

        additional = []
        existing_lower = [b.lower() for b in existing_bullets]

        # Look for numbered items or key phrases in voiceover
        # Patterns like "Premièrement", "Deuxièmement", etc.
        fr_ordinals = ['premièrement', 'deuxièmement', 'troisièmement', 'quatrièmement', 'cinquièmement']
        en_ordinals = ['first', 'second', 'third', 'fourth', 'fifth', 'firstly', 'secondly']

        ordinals = fr_ordinals if language.startswith('fr') else en_ordinals

        # Split voiceover into sentences
        sentences = re.split(r'[.!?]', voiceover)

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if not sentence_lower:
                continue

            # Check if sentence starts with ordinal
            for ordinal in ordinals:
                if ordinal in sentence_lower:
                    # Extract key phrase after ordinal
                    key_phrase = self._extract_key_phrase(sentence, language)
                    if key_phrase and key_phrase.lower() not in existing_lower:
                        additional.append(key_phrase)
                        existing_lower.append(key_phrase.lower())
                        break

        return additional[:5]  # Return max 5 additional bullets

    def _expand_short_bullet(self, bullet: str, slide: dict, language: str) -> str:
        """Expand a single-word bullet point to be more descriptive."""
        title = slide.get("title") or ""
        voiceover = (slide.get("voiceover_text") or "").lower()

        bullet_lower = bullet.lower()

        # Common expansions for French
        fr_expansions = {
            "définition": "Définition et concept clé",
            "importance": "Importance et bénéfices",
            "exemples": "Exemples concrets d'utilisation",
            "avantages": "Avantages et points forts",
            "inconvénients": "Inconvénients et limitations",
            "utilisation": "Cas d'utilisation pratiques",
            "fonctionnement": "Fonctionnement et mécanisme",
            "architecture": "Architecture et composants",
            "implémentation": "Implémentation et mise en œuvre",
            "configuration": "Configuration et paramétrage",
        }

        # Common expansions for English
        en_expansions = {
            "definition": "Definition and core concept",
            "importance": "Importance and benefits",
            "examples": "Practical usage examples",
            "advantages": "Key advantages and strengths",
            "disadvantages": "Limitations and trade-offs",
            "usage": "Practical use cases",
            "implementation": "Implementation approach",
            "architecture": "Architecture and components",
            "configuration": "Configuration and setup",
        }

        expansions = fr_expansions if language.startswith('fr') else en_expansions

        # Check if we have a predefined expansion
        if bullet_lower in expansions:
            return expansions[bullet_lower]

        # Try to find context from voiceover
        # Look for the bullet word in voiceover and extract surrounding context
        if bullet_lower in voiceover:
            # Find sentence containing the bullet word
            import re
            sentences = re.split(r'[.!?]', slide.get("voiceover_text") or "")
            for sentence in sentences:
                if bullet_lower in sentence.lower():
                    # Extract key phrase from this sentence
                    phrase = self._extract_key_phrase(sentence, language)
                    if phrase and len(phrase.split()) >= 3:
                        return phrase

        # Fallback: add generic context based on title
        if language.startswith('fr'):
            return f"{bullet} du sujet"
        else:
            return f"{bullet} overview"

    def _extract_key_phrase(self, text: str, language: str) -> str:
        """
        Extract the key phrase from a long bullet point.

        Strategies:
        1. Take text before first comma, colon, or dash
        2. Take first 5-7 meaningful words
        3. Remove common filler words
        """
        import re

        # Strategy 1: Split on punctuation that often separates key concept from explanation
        split_patterns = [
            r'^([^,:\-–—]+)',  # Before comma, colon, or dash
            r'^([^.!?]+)',     # Before sentence-ending punctuation
        ]

        for pattern in split_patterns:
            match = re.match(pattern, text.strip())
            if match:
                candidate = match.group(1).strip()
                words = candidate.split()
                if 2 <= len(words) <= 7:
                    return candidate

        # Strategy 2: Take first 5-7 words
        words = text.split()

        # Remove common filler words at the start
        filler_words = {
            'fr': {'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'ce', 'cela', 'ceci', 'qui', 'que', 'dont', 'où'},
            'en': {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'is', 'are', 'was', 'were', 'be'},
        }

        lang_prefix = language[:2] if language else 'en'
        fillers = filler_words.get(lang_prefix, filler_words['en'])

        # Remove leading filler words
        while words and words[0].lower() in fillers:
            words = words[1:]

        # Take 5-7 words
        key_words = words[:6]

        return ' '.join(key_words)

    def _enhance_diagram_voiceover(
        self,
        slide: dict,
        language: str,
        key_concepts: list,
        spatial_refs: list
    ) -> str:
        """
        Generate an enhanced voiceover for a diagram slide that properly
        explains all the visual elements.
        """
        original = slide.get("voiceover_text") or ""
        content = slide.get("content") or ""
        diagram_desc = slide.get("diagram_description") or ""
        title = slide.get("title") or ""

        # If original is already good, just return it
        if len(original) > 200 and any(ref in original.lower() for ref in spatial_refs):
            return original

        lang_prefix = language[:2] if language else 'en'

        # Build enhanced narration based on language
        if lang_prefix == 'fr':
            intro = f"Ce diagramme illustre {title.lower()}. "
            elements_intro = "Examinons les différents éléments. "
            conclusion = "Cette architecture permet de comprendre comment les composants interagissent."

            # Add component descriptions
            if key_concepts:
                components = ", ".join(key_concepts[:4])
                elements_intro += f"Nous avons {components} comme composants principaux. "

        else:
            intro = f"This diagram illustrates {title.lower()}. "
            elements_intro = "Let's examine the different elements. "
            conclusion = "This architecture helps us understand how the components interact."

            if key_concepts:
                components = ", ".join(key_concepts[:4])
                elements_intro += f"The main components are {components}. "

        # Combine with original if it has useful content
        if original and len(original) > 30:
            # Keep original but add enhancement
            enhanced = f"{intro}{original}"
            # If original doesn't have conclusion-like content, add it
            if "permet" not in original.lower() and "enables" not in original.lower():
                enhanced += f" {conclusion}"
        else:
            # Use template-based narration
            enhanced = f"{intro}{elements_intro}"
            if content:
                enhanced += f"{content} "
            enhanced += conclusion

        return enhanced

    def _parse_json_robust(self, content: str) -> dict:
        """
        Parse JSON with robust error handling and repair capabilities.

        Handles common LLM JSON issues:
        - Truncated JSON (missing closing brackets)
        - Trailing commas
        - Unescaped characters in strings
        - Unterminated strings

        Args:
            content: Raw JSON string from LLM

        Returns:
            Parsed dictionary

        Raises:
            json.JSONDecodeError: If JSON cannot be repaired
        """
        if not content:
            raise json.JSONDecodeError("Empty content", "", 0)

        # First, try direct parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"[PLANNER] JSON parse error: {e.msg} at position {e.pos}", flush=True)
            print(f"[PLANNER] Attempting JSON repair...", flush=True)

        # Attempt repairs
        repaired = content

        # 1. Remove any text before the first { or after the last }
        first_brace = repaired.find('{')
        last_brace = repaired.rfind('}')
        if first_brace != -1 and last_brace != -1:
            repaired = repaired[first_brace:last_brace + 1]
        elif first_brace != -1:
            repaired = repaired[first_brace:]

        # 2. Try to fix truncated JSON by counting brackets
        open_braces = repaired.count('{')
        close_braces = repaired.count('}')
        open_brackets = repaired.count('[')
        close_brackets = repaired.count(']')

        # Add missing closing brackets/braces
        if open_brackets > close_brackets:
            repaired += ']' * (open_brackets - close_brackets)
        if open_braces > close_braces:
            repaired += '}' * (open_braces - close_braces)

        # 3. Remove trailing commas before ] or }
        repaired = re.sub(r',\s*]', ']', repaired)
        repaired = re.sub(r',\s*}', '}', repaired)

        # 4. Fix unterminated strings (look for unclosed quotes before comma/bracket)
        # This is a simplified fix - replace unescaped newlines in strings
        repaired = re.sub(r'(?<!\\)\n(?=[^"]*"[,\]\}])', '\\n', repaired)

        # Try parsing repaired JSON
        try:
            result = json.loads(repaired)
            print(f"[PLANNER] JSON repair successful!", flush=True)
            return result
        except json.JSONDecodeError as e:
            print(f"[PLANNER] JSON repair failed: {e.msg}", flush=True)

        # 5. Last resort: try to extract slides array and rebuild
        try:
            # Find the slides array
            slides_match = re.search(r'"slides"\s*:\s*\[', repaired)
            if slides_match:
                # Find all complete slide objects
                slides = []
                slide_pattern = r'\{\s*"type"\s*:\s*"[^"]+"\s*,.*?"voiceover_text"\s*:\s*"[^"]*"\s*[,\}]'
                for match in re.finditer(slide_pattern, repaired, re.DOTALL):
                    slide_str = match.group()
                    if not slide_str.endswith('}'):
                        slide_str = slide_str.rstrip(',') + '}'
                    try:
                        slide = json.loads(slide_str)
                        slides.append(slide)
                    except:
                        pass

                if slides:
                    print(f"[PLANNER] Extracted {len(slides)} slides from malformed JSON", flush=True)
                    return {
                        "title": "Presentation",
                        "description": "Generated presentation",
                        "slides": slides,
                        "total_duration": len(slides) * 30
                    }
        except Exception as ex:
            print(f"[PLANNER] Slide extraction failed: {ex}", flush=True)

        # If all repairs fail, raise the original error with context
        raise json.JSONDecodeError(
            f"Could not parse or repair JSON. Content length: {len(content)}, "
            f"Brackets: {{{open_braces}/{close_braces}, [{open_brackets}/{close_brackets}",
            content[:200],
            0
        )

    def _parse_script(
        self,
        data: dict,
        request: GeneratePresentationRequest
    ) -> PresentationScript:
        """Parse LLM response into PresentationScript"""
        slides = []

        for slide_data in data.get("slides", []):
            # Parse code blocks (handle None values)
            code_blocks = []
            code_blocks_data = slide_data.get("code_blocks") or []
            for cb_data in code_blocks_data:
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

            # Handle case where LLM returns content as a dict (e.g., for diagrams)
            if isinstance(content, dict):
                # Extract description from dict or convert to string
                content = content.get("description") or content.get("diagram_description") or str(content)

            if slide_type == SlideType.DIAGRAM and not content:
                content = slide_data.get("diagram_description") or ""

            # Handle None values for lists
            bullet_points = slide_data.get("bullet_points") or []
            voiceover_text = slide_data.get("voiceover_text") or ""

            slides.append(Slide(
                type=slide_type,
                title=slide_data.get("title"),
                subtitle=slide_data.get("subtitle"),
                content=content,
                bullet_points=bullet_points,
                code_blocks=code_blocks,
                duration=slide_data.get("duration", 10.0),
                voiceover_text=voiceover_text,
                transition=slide_data.get("transition", "fade"),
                notes=slide_data.get("notes"),
                diagram_type=slide_data.get("diagram_type"),
                index=len(slides)
            ))

        return PresentationScript(
            title=data.get("title", "Untitled Presentation"),
            description=data.get("description", ""),
            target_audience=data.get("target_audience", request.target_audience),
            target_career=request.target_career,  # Propagate career for diagram focus
            language=data.get("language", request.language),
            total_duration=data.get("total_duration", request.duration),
            slides=slides,
            code_context=data.get("code_context", {})
        )

    def _extract_source_topics(self, text: str, top_n: int = 25) -> list:
        """Extract main topics from source documents for topic locking."""
        import re
        from collections import Counter

        # Common stopwords in English and French
        stopwords = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has',
            'will', 'can', 'are', 'was', 'were', 'been', 'being', 'would', 'could',
            'should', 'may', 'might', 'must', 'need', 'also', 'just', 'like',
            'make', 'made', 'use', 'used', 'using', 'then', 'than', 'more', 'most',
            'some', 'such', 'only', 'other', 'into', 'over', 'which', 'where',
            'when', 'what', 'while', 'there', 'here', 'they', 'their', 'them',
            'example', 'examples', 'section', 'chapter', 'part', 'page', 'figure',
            'vous', 'nous', 'elle', 'sont', 'avec', 'pour', 'dans', 'cette',
            'about', 'each', 'very', 'your', 'these', 'those', 'does', 'done',
        }

        # Extract words (4+ chars)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_counts = Counter(w for w in words if w not in stopwords)

        return [word for word, _ in word_counts.most_common(top_n)]

    def _build_rag_section(self, request: GeneratePresentationRequest) -> str:
        """
        Build RAG context section for the prompt.

        If documents are provided, includes their content as primary source material.
        This ensures the training video is based on the uploaded documentation.

        Includes TOPIC LOCK to prevent LLM from going off-topic.
        Automatically truncates based on provider's max_context limit.
        """
        rag_context = getattr(request, 'rag_context', None)

        if not rag_context:
            print("[PLANNER] [RAG] No RAG context provided - using standard AI generation", flush=True)
            return ""

        # Log RAG context detection
        print(f"[PLANNER] [RAG] ╔════════════════════════════════════════════════════════════════╗", flush=True)
        print(f"[PLANNER] [RAG] ║            RAG CONTEXT DETECTED - INJECTING INTO PROMPT         ║", flush=True)
        print(f"[PLANNER] [RAG] ╚════════════════════════════════════════════════════════════════╝", flush=True)
        print(f"[PLANNER] [RAG] Original RAG context length: {len(rag_context)} characters", flush=True)
        print(f"[PLANNER] [RAG] First 300 chars of RAG context:", flush=True)
        print(f"[PLANNER] [RAG] >>> {rag_context[:300]}...", flush=True)

        # Extract topics from source for TOPIC LOCK
        source_topics = self._extract_source_topics(rag_context, top_n=25)
        topics_str = ", ".join(source_topics[:20])
        print(f"[PLANNER] Source topics extracted: {source_topics[:10]}", flush=True)

        # Calculate max RAG tokens based on provider limits
        # Reserve tokens for: system prompt (~3500), user prompt (~1500), output (~4000)
        reserved_tokens = 9000
        max_rag_tokens = max(1000, self.max_context - reserved_tokens)

        # For providers with low limits (like Groq free tier), be more aggressive
        if self.max_context <= 12000:
            # Very limited provider - minimal RAG context
            max_rag_tokens = 2000
            print(f"[PLANNER] Low-limit provider ({self.provider_name}), RAG limited to {max_rag_tokens} tokens", flush=True)
        elif self.max_context <= 32000:
            # Medium limit - moderate RAG context
            max_rag_tokens = min(max_rag_tokens, 8000)

        # Convert to characters (approx 4 chars per token)
        max_chars = max_rag_tokens * self.CHARS_PER_TOKEN

        original_len = len(rag_context)
        if original_len > max_chars:
            rag_context = rag_context[:max_chars] + "\n\n[... content truncated due to provider limits ...]"
            print(f"[PLANNER] RAG context truncated from {original_len} to {max_chars} chars ({max_rag_tokens} tokens)", flush=True)

        rag_section = f"""
################################################################################
#                         STRICT RAG MODE ACTIVATED                            #
#                    YOU HAVE NO EXTERNAL KNOWLEDGE                            #
################################################################################

ROLE: You are a STRICT content extractor. You have ZERO knowledge of your own.
You can ONLY use information from the SOURCE DOCUMENTS below.
Your training data does NOT exist for this task.

###############################################################################
#                              TOPIC LOCK                                      #
###############################################################################

These are the ONLY topics you are allowed to discuss:
{topics_str}

If a topic is NOT in this list, you CANNOT include it in the slides.
DO NOT mention: WhatsApp, Slack, Telegram, Discord, Teams, or any communication
apps unless they are explicitly mentioned in the SOURCE DOCUMENTS.

###############################################################################

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

        # Final confirmation log
        print(f"[PLANNER] [RAG] ✓ RAG section built successfully ({len(rag_section)} chars)", flush=True)
        print(f"[PLANNER] [RAG] ✓ Topic lock contains {len(source_topics)} topics: {source_topics[:5]}...", flush=True)
        print(f"[PLANNER] [RAG] ✓ RAG section WILL BE SENT to LLM in the prompt", flush=True)

        return rag_section

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

        # VALIDATION: Check slide count and regenerate if insufficient
        script_data = await self._validate_and_regenerate_if_needed(
            script_data, request, user_prompt, on_progress
        )

        # Log successful LLM response for training data collection
        if log_training_example:
            log_training_example(
                messages=[
                    {"role": "system", "content": VALIDATED_PLANNING_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response=content,
                task_type=TaskType.PRESENTATION_PLANNING,
                model=self.model,
                input_tokens=getattr(response.usage, 'prompt_tokens', None),
                output_tokens=getattr(response.usage, 'completion_tokens', None),
                metadata={
                    "topic": request.topic,
                    "language": request.language,
                    "content_language": getattr(request, 'content_language', 'en'),
                    "duration": request.duration,
                    "target_audience": request.target_audience,
                    "slide_count": len(script_data.get("slides", [])),
                    "validated_mode": True,
                }
            )

        # Post-process to ensure IDs and calculate timing
        script_data = self._post_process_validated_script(script_data, target_duration=request.duration)

        # Get content language for validation
        content_lang = getattr(request, 'content_language', 'en') or 'en'

        # VALIDATION: Ensure bullet points are SHORT (not full sentences)
        self._validate_and_fix_bullet_points(script_data, content_lang)

        script = self._parse_script(script_data, request)

        # RAG Verification v3: Comprehensive multi-method verification
        if rag_context:
            rag_result = verify_rag_usage(script_data, rag_context, verbose=True, comprehensive=True)

            # Store comprehensive verification result in script metadata
            script.rag_verification = {
                "mode": rag_mode,
                "token_count": rag_token_count,
                # Semantic coverage
                "coverage": rag_result.overall_coverage,
                "is_compliant": rag_result.is_compliant,
                "summary": rag_result.summary,
                # Keyword validation
                "keyword_coverage": rag_result.keyword_coverage,
                "source_keywords_found": rag_result.source_keywords_found,
                "source_keywords_missing": rag_result.source_keywords_missing[:10],
                # Topic validation
                "topic_match_score": rag_result.topic_match_score,
                "source_topics": rag_result.source_topics[:10],
                "generated_topics": rag_result.generated_topics[:10],
                # Hallucinations
                "potential_hallucinations": len(rag_result.potential_hallucinations),
                "hallucination_details": [
                    {"slide": h.get("slide_index"), "similarity": h.get("similarity")}
                    for h in rag_result.potential_hallucinations[:5]
                ],
                # Failure analysis
                "failure_reasons": rag_result.failure_reasons,
                "warning": threshold_result.warning_message,
            }

            print(f"[PLANNER] {rag_result.summary}", flush=True)

            # Log detailed analysis if not compliant
            if not rag_result.is_compliant:
                print(f"[PLANNER] ❌ RAG VERIFICATION FAILED:", flush=True)
                print(f"[PLANNER]   - Semantic: {rag_result.overall_coverage:.1%}", flush=True)
                print(f"[PLANNER]   - Keywords: {rag_result.keyword_coverage:.1%}", flush=True)
                print(f"[PLANNER]   - Topics: {rag_result.topic_match_score:.1%}", flush=True)
                if rag_result.source_keywords_missing:
                    print(f"[PLANNER]   - Missing keywords: {rag_result.source_keywords_missing[:5]}", flush=True)
                print(f"[PLANNER]   - Source topics: {rag_result.source_topics[:5]}", flush=True)
                print(f"[PLANNER]   - Generated topics: {rag_result.generated_topics[:5]}", flush=True)
                print(f"[PLANNER]   ⚠️ Content may not be based on source documents!", flush=True)
        else:
            # Store RAG mode even when no context (NONE mode)
            script.rag_verification = {
                "mode": rag_mode,
                "token_count": 0,
                "coverage": 0.0,
                "is_compliant": True,  # No RAG means no compliance check
                "summary": "No source documents - standard AI generation",
                "keyword_coverage": 1.0,
                "topic_match_score": 1.0,
                "failure_reasons": [],
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

        # Calculate dynamic slide count and word requirements based on duration
        duration_minutes = request.duration / 60
        min_slides = max(6, int(duration_minutes * 2))  # At least 2 slides per minute
        max_slides = max(10, int(duration_minutes * 3))  # Up to 3 slides per minute
        total_words_needed = int(request.duration * 2.5)  # 150 words/min = 2.5 words/sec
        words_per_slide = total_words_needed // max(min_slides, 1)

        print(f"[PLANNER] DURATION CALC: {request.duration}s = {duration_minutes:.1f}min -> slides:{min_slides}-{max_slides}, words:{total_words_needed}", flush=True)

        # Get practical focus configuration
        practical_focus = getattr(request, 'practical_focus', None)
        practical_focus_instructions = get_practical_focus_instructions(practical_focus)
        practical_focus_level = parse_practical_focus(practical_focus)
        slide_ratio = get_practical_focus_slide_ratio(practical_focus)

        if practical_focus:
            print(f"[PLANNER] 🎯 Practical focus (validated): {practical_focus} (level: {practical_focus_level})", flush=True)

        return f"""Create a TRAINING VIDEO script for:

TOPIC: {request.topic}

PARAMETERS:
- Programming Language: {request.language}
- CONTENT LANGUAGE: {content_lang_name} (code: {content_lang})
- Target Duration: {duration_str} ({request.duration} seconds total)
- Target Audience: {request.target_audience}
- Visual Style: {request.style.value}
- HAS SOURCE DOCUMENTS: {"YES - USE THEM AS PRIMARY SOURCE (90%)" if has_rag else "NO"}
- Practical Focus: {PRACTICAL_FOCUS_CONFIG.get(practical_focus_level, PRACTICAL_FOCUS_CONFIG["balanced"])["name"]}

═══════════════════════════════════════════════════════════════════════════════
                    CRITICAL DURATION REQUIREMENTS (READ CAREFULLY)
═══════════════════════════════════════════════════════════════════════════════

To achieve the target duration of {duration_str}, you MUST follow these requirements:

📊 SLIDE COUNT: Create between {min_slides} and {max_slides} slides
   - Calculated as: {duration_minutes:.1f} minutes × 2-3 slides/minute

📝 TOTAL WORDS NEEDED: ~{total_words_needed} words across all voiceovers
   - Calculated as: {request.duration} seconds × 2.5 words/second (150 words/min speaking rate)

📄 WORDS PER SLIDE: Each slide's voiceover_text should have ~{words_per_slide} words
   - MINIMUM 60 words per slide (shorter = video too short!)
   - IDEAL: 70-100 words per slide for proper pacing

⚠️ COMMON MISTAKE: Creating short voiceovers (20-40 words) results in videos under 2 minutes!
   Each voiceover should be a full paragraph explaining the slide content in detail.

{practical_focus_instructions}

REQUIRED SLIDE TYPE DISTRIBUTION (based on practical focus):
- content slides: ~{int(slide_ratio['content']*100)}% ({int(min_slides * slide_ratio['content'])}-{int(max_slides * slide_ratio['content'])} slides)
- diagram slides: ~{int(slide_ratio['diagram']*100)}% ({int(min_slides * slide_ratio['diagram'])}-{int(max_slides * slide_ratio['diagram'])} slides)
- code slides: ~{int(slide_ratio['code']*100)}% ({int(min_slides * slide_ratio['code'])}-{int(max_slides * slide_ratio['code'])} slides)
- code_demo slides: ~{int(slide_ratio['code_demo']*100)}% ({int(min_slides * slide_ratio['code_demo'])}-{int(max_slides * slide_ratio['code_demo'])} slides)
- conclusion slides: ~{int(slide_ratio['conclusion']*100)}%

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
11. CRITICAL: Each voiceover_text MUST have at least {words_per_slide} words to achieve {duration_str} target
12. ⚠️ CONCEPT DEPENDENCY RULE: NEVER use a technical term BEFORE explaining it!
    - A concept CANNOT appear in voiceover until it has been DEFINED in a previous slide
    - WRONG ORDER: "Use a decorator" → then later "What is a decorator?"
    - CORRECT ORDER: "What is a decorator? It's a function..." → then "Now let's use a decorator"
    - This applies to ALL technical vocabulary: functions, classes, patterns, libraries, frameworks

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
            voiceover = slide.get("voiceover_text") or ""

            # Check if sync anchor already exists
            sync_pattern = r'\[SYNC:[\w_]+\]'
            has_sync = bool(re.search(sync_pattern, voiceover))

            if not has_sync and voiceover:
                # Add sync anchor at the beginning
                slide["voiceover_text"] = f"[SYNC:{slide_id}] {voiceover}"

        script_data["slides"] = slides
        return script_data

    def _post_process_validated_script(self, script_data: dict, target_duration: int = 300) -> dict:
        """Post-process the script to ensure consistency and adequate duration"""
        import re

        slides = script_data.get("slides", [])
        total_words = 0
        total_calculated_duration = 0
        short_voiceover_count = 0
        min_words_per_slide = 60  # Minimum for adequate duration

        for i, slide in enumerate(slides):
            # Ensure ID exists
            if "id" not in slide:
                slide["id"] = f"slide_{i:03d}"

            # Calculate proper duration based on content
            voiceover = slide.get("voiceover_text") or ""
            # Remove sync anchors for word count
            clean_voiceover = re.sub(r'\[SYNC:[\w_]+\]', '', voiceover).strip()
            word_count = len(clean_voiceover.split())
            total_words += word_count
            voiceover_duration = word_count / 2.5  # 150 words/minute

            # Track short voiceovers
            if word_count < min_words_per_slide:
                short_voiceover_count += 1
                print(f"[PLANNER] ⚠️ Slide {i} has short voiceover: {word_count} words (min: {min_words_per_slide})", flush=True)

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
            total_calculated_duration += calculated

            # Only override if significantly different
            current = slide.get("duration", 10.0)
            if abs(current - calculated) > 5:
                slide["duration"] = round(calculated, 1)

        script_data["slides"] = slides

        # Log duration analysis
        target_words = int(target_duration * 2.5)
        duration_ratio = total_calculated_duration / target_duration if target_duration > 0 else 0
        print(f"[PLANNER] 📊 Duration Analysis:", flush=True)
        print(f"[PLANNER]   - Slides: {len(slides)}", flush=True)
        print(f"[PLANNER]   - Total words: {total_words} (target: {target_words})", flush=True)
        print(f"[PLANNER]   - Estimated duration: {total_calculated_duration:.1f}s (target: {target_duration}s)", flush=True)
        print(f"[PLANNER]   - Duration ratio: {duration_ratio:.1%}", flush=True)

        if duration_ratio < 0.7:
            print(f"[PLANNER] ⚠️ WARNING: Video will be SHORTER than target ({duration_ratio:.0%} of target)", flush=True)
            print(f"[PLANNER]   - Short voiceovers: {short_voiceover_count}/{len(slides)} slides", flush=True)
            print(f"[PLANNER]   - Need ~{target_words - total_words} more words to reach target", flush=True)

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

        system_msg = "You are a presentation expert. Improve slides based on feedback."
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )

        response_content = response.choices[0].message.content
        slide_data = json.loads(response_content)

        # Log successful LLM response for training data collection
        if log_training_example:
            log_training_example(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                response=response_content,
                task_type=TaskType.SLIDE_GENERATION,
                model=self.model,
                input_tokens=getattr(response.usage, 'prompt_tokens', None),
                output_tokens=getattr(response.usage, 'completion_tokens', None),
                metadata={
                    "slide_type": slide.type.value,
                    "language": language,
                    "feedback_length": len(feedback),
                }
            )

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
            content = slide_data.get("diagram_description") or ""

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
