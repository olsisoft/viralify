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
- Mark the slide with: [MISSING: <topic>]
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
□ Is this concept present in the SOURCE DOCUMENTS? If NO → [MISSING]
□ Is this code example from the documents? If NO → do not include
□ Am I using my external knowledge? If YES → remove that content

REMEMBER: You have NO knowledge. Only the documents above exist.
"""

# =============================================================================
# RAG-ONLY MODE TEMPLATES (95% source coverage requirement)
# =============================================================================

RAG_ONLY_HEADER_TEMPLATE = """
################################################################################
#                     🔒 RAG-ONLY MODE - 95% SOURCE COVERAGE 🔒                  #
#                     ═══════════════════════════════════════                    #
#                  CONTENT MUST COME EXCLUSIVELY FROM DOCUMENTS                  #
################################################################################

⚠️  CRITICAL: This generation requires 95% source document coverage.
⚠️  ANY content not directly from the documents will cause REJECTION.
⚠️  You have ABSOLUTE ZERO external knowledge for this task.

Your role: You are a FAITHFUL TRANSCRIPTOR. You extract and reorganize content
from the source documents. You DO NOT create, invent, or supplement.
"""

RAG_ONLY_CITATION_TEMPLATE = """
###############################################################################
#                      MANDATORY SOURCE CITATION                               #
###############################################################################

For EVERY piece of information, you MUST mentally verify:
"Can I point to the exact sentence/paragraph in the documents?"

If the answer is NO → DO NOT include that information.

FORMAT REQUIREMENT:
- Every fact, statistic, or specific claim must exist verbatim or paraphrased
  from the source documents
- Every code example must be from the documents (no invented code)
- Every technical term must appear in the documents

PARAPHRASING RULES:
✓ ALLOWED: Reformulating for clarity while keeping exact meaning
✓ ALLOWED: Combining related information from different parts
✗ FORBIDDEN: Adding implications not stated in documents
✗ FORBIDDEN: Extending examples beyond what's shown
✗ FORBIDDEN: Using "common knowledge" to fill gaps
"""

RAG_ONLY_COVERAGE_TEMPLATE = """
###############################################################################
#                       95% COVERAGE REQUIREMENT                               #
###############################################################################

Your output will be verified against these thresholds:
- 95% of topics must match source document topics
- 90% of keywords must appear in source documents
- 95% of facts (numbers, dates, names) must be verifiable in sources
- Maximum 5% of content can be transitions/structure (non-source)

WHAT COUNTS AS THE 5% ALLOWED:
✓ Slide titles that summarize source content
✓ "Let's look at..." / "Next, we'll cover..."
✓ "In summary..." / "The key points are..."
✓ Bullet point formatting
✓ "Welcome" / "Thank you" slides

WHAT MUST BE 100% FROM SOURCE (the 95%):
- All technical explanations
- All examples and code
- All definitions
- All statistics and numbers
- All named entities (products, companies, people)
- All processes and procedures described
"""

RAG_ONLY_REJECTION_WARNING = """
###############################################################################
#                        ⛔ AUTOMATIC REJECTION TRIGGERS ⛔                      #
###############################################################################

Your output will be AUTOMATICALLY REJECTED if:

1. HALLUCINATED FACTS: Any statistic, date, or number not in documents
   Example: "This technology is used by 80% of companies" (if not in docs)

2. INVENTED EXAMPLES: Code or scenarios not present in documents
   Example: Adding a "practical example" that wasn't in the source

3. EXTERNAL KNOWLEDGE: Using your training data to "help"
   Example: "As is commonly known..." or "Typically, this means..."

4. TOPIC DRIFT: Discussing related topics not covered in documents
   Example: Documents about REST API → you mention GraphQL (not in docs)

5. ASSUMPTION MAKING: Filling gaps with logical assumptions
   Example: "This probably means..." or "It's likely that..."

When information is missing, use: [MISSING: description of missing content]
This is REQUIRED - do not try to work around missing information.
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

RULES: Only use content from documents above. Mark missing info with [MISSING].
"""


def build_rag_only_section(
    rag_context: str,
    source_topics: List[str],
    max_chars: Optional[int] = None
) -> str:
    """
    Build the RAG-ONLY section requiring 95% source coverage.

    This is the strictest RAG mode - content MUST come exclusively
    from the source documents with minimal (5%) allowed additions.

    Args:
        rag_context: The raw RAG context from documents
        source_topics: List of topics extracted from source documents
        max_chars: Maximum characters for RAG context

    Returns:
        Complete RAG-ONLY section string
    """
    # Truncate if needed
    if max_chars and len(rag_context) > max_chars:
        rag_context = rag_context[:max_chars] + "\n\n[... content truncated due to provider limits ...]"

    # Format topics for topic lock
    topics_str = ", ".join(source_topics[:20])

    # Build the complete RAG-ONLY section
    rag_section = f"""
{RAG_ONLY_HEADER_TEMPLATE}

{RAG_TOPIC_LOCK_TEMPLATE.format(topics=topics_str)}

<source_documents>
################################################################################
#                    📄 SOURCE DOCUMENTS - YOUR ONLY TRUTH 📄                    #
################################################################################
{rag_context}
################################################################################
#                         END OF SOURCE DOCUMENTS                               #
################################################################################
</source_documents>

{RAG_ONLY_CITATION_TEMPLATE}

{RAG_ONLY_COVERAGE_TEMPLATE}

{RAG_ONLY_REJECTION_WARNING}

{RAG_RULES_TEMPLATE}

{RAG_FORBIDDEN_TEMPLATE}

###############################################################################
#                      FINAL REMINDER - 95% COVERAGE                           #
###############################################################################

Before you generate ANY content, ask yourself:
"Is this EXACTLY from the source documents, or am I inventing?"

If inventing → STOP and use [MISSING]

Your generation will be VERIFIED against the source documents.
Non-compliant content will be REJECTED.
"""

    return rag_section


def get_rag_section_for_mode(
    rag_context: str,
    source_topics: List[str],
    mode: str = "standard",
    max_chars: Optional[int] = None
) -> str:
    """
    Get the appropriate RAG section based on the verification mode.

    Args:
        rag_context: The raw RAG context from documents
        source_topics: List of topics extracted from source documents
        mode: RAG mode - "standard", "strict", "rag_only", or "minimal"
        max_chars: Maximum characters for RAG context

    Returns:
        RAG section string appropriate for the mode
    """
    if mode == "rag_only":
        return build_rag_only_section(rag_context, source_topics, max_chars)
    elif mode == "minimal":
        return build_rag_section_minimal(rag_context, source_topics, max_chars)
    else:
        # Standard and strict use the same section (verification differs)
        return build_rag_section(rag_context, source_topics, max_chars)
