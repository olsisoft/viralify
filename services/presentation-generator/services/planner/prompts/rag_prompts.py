"""
RAG Prompts for Presentation Planner

Contains templates for RAG (Retrieval-Augmented Generation) context injection.
These templates ensure the LLM uses source documents as the primary content source.
"""

from typing import List, Optional


RAG_STRICT_MODE_TEMPLATE = """
################################################################################
#                         STRICT RAG MODE ACTIVATED                            #
#                    YOU HAVE NO EXTERNAL KNOWLEDGE                            #
################################################################################

ROLE: You are a STRICT content extractor. You have ZERO knowledge of your own.
You can ONLY use information from the SOURCE DOCUMENTS below.
Your training data does NOT exist for this task.
"""

RAG_TOPIC_LOCK_TEMPLATE = """
###############################################################################
#                              TOPIC LOCK                                      #
###############################################################################

These are the ONLY topics you are allowed to discuss:
{topics}

If a topic is NOT in this list, you CANNOT include it in the slides.
DO NOT mention: WhatsApp, Slack, Telegram, Discord, Teams, or any communication
apps unless they are explicitly mentioned in the SOURCE DOCUMENTS.
"""

RAG_RULES_TEMPLATE = """
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
"""

RAG_ALLOWED_CONTENT_TEMPLATE = """
###############################################################################
#                         ALLOWED CONTENT (10% MAX)                            #
###############################################################################

You MAY add ONLY these elements (maximum 10% of total content):
✓ Transitions: "Passons maintenant à...", "Voyons comment..."
✓ Pedagogical reformulations: "Autrement dit...", "En résumé..."
✓ Slide structure: titles, bullet formatting
✓ Greeting/conclusion: "Bienvenue", "Merci d'avoir suivi ce cours"
"""

RAG_FORBIDDEN_TEMPLATE = """
###############################################################################
#                              FORBIDDEN                                       #
###############################################################################

❌ Adding concepts not in the documents
❌ Inventing code examples
❌ Using your knowledge to "complete" missing information
❌ Paraphrasing to the point of changing the meaning
❌ Adding details "you know" but aren't in the documents
❌ Creating diagrams not described in the documents
"""

RAG_VALIDATION_TEMPLATE = """
###############################################################################
#                         VALIDATION BEFORE OUTPUT                             #
###############################################################################

Before generating each slide, verify:
□ Is this concept present in the SOURCE DOCUMENTS? If NO → [SOURCE_MANQUANTE]
□ Is this code example from the documents? If NO → do not include
□ Am I using my external knowledge? If YES → remove that content

REMEMBER: You have NO knowledge. Only the documents above exist.
"""


def build_rag_section(
    rag_context: str,
    source_topics: List[str],
    max_chars: Optional[int] = None
) -> str:
    """
    Build the complete RAG section for injection into the prompt.

    Args:
        rag_context: The raw RAG context from documents
        source_topics: List of topics extracted from source documents
        max_chars: Maximum characters for RAG context (for truncation)

    Returns:
        Complete RAG section string ready for prompt injection
    """
    # Truncate if needed
    if max_chars and len(rag_context) > max_chars:
        rag_context = rag_context[:max_chars] + "\n\n[... content truncated due to provider limits ...]"

    # Format topics for topic lock
    topics_str = ", ".join(source_topics[:20])

    # Build the complete RAG section with XML-style tags for clear delimitation
    rag_section = f"""
{RAG_TOPIC_LOCK_TEMPLATE.format(topics=topics_str)}

<source_documents>
{rag_context}
</source_documents>

{RAG_RULES_TEMPLATE}

{RAG_ALLOWED_CONTENT_TEMPLATE}

{RAG_FORBIDDEN_TEMPLATE}

{RAG_VALIDATION_TEMPLATE}
"""

    return rag_section


def build_rag_section_minimal(
    rag_context: str,
    source_topics: List[str],
    max_chars: Optional[int] = None
) -> str:
    """
    Build a minimal RAG section for providers with limited context.

    Args:
        rag_context: The raw RAG context from documents
        source_topics: List of topics extracted from source documents
        max_chars: Maximum characters for RAG context

    Returns:
        Minimal RAG section string
    """
    if max_chars and len(rag_context) > max_chars:
        rag_context = rag_context[:max_chars] + "\n[...truncated...]"

    topics_str = ", ".join(source_topics[:10])

    return f"""
## STRICT RAG MODE - USE ONLY SOURCE DOCUMENTS

Allowed topics: {topics_str}

=== SOURCE DOCUMENTS ===
{rag_context}
=== END DOCUMENTS ===

RULES: Only use content from documents above. Mark missing info with [SOURCE_MANQUANTE].
"""
