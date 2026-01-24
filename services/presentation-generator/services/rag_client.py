"""
RAG Client Service

Calls the course-generator service to fetch RAG context from uploaded documents.
This allows presentation-generator to use documents uploaded via course-generator.
"""
import os
from typing import List, Optional, Dict, Any
import httpx


class RAGClient:
    """
    Client for fetching RAG context from course-generator service.

    Course-generator manages document uploads, parsing, and vector storage.
    This client queries that service to get relevant context for presentations.
    """

    def __init__(self):
        # Course-generator service URL (Docker internal)
        self.course_generator_url = os.getenv(
            "COURSE_GENERATOR_URL",
            "http://course-generator:8007"
        )
        self.timeout = 60.0  # RAG queries can take time

    async def get_context_for_presentation(
        self,
        document_ids: List[str],
        topic: str,
        max_chunks: int = 10,
        include_diagrams: bool = True
    ) -> Optional[str]:
        """
        Fetch RAG context from course-generator for presentation generation.

        Args:
            document_ids: List of document IDs to query
            topic: The presentation topic to find relevant content for
            max_chunks: Maximum number of chunks to retrieve
            include_diagrams: Whether to specifically look for diagram descriptions

        Returns:
            Combined context string or None if failed
        """
        if not document_ids:
            return None

        print(f"[RAG_CLIENT] Fetching context for {len(document_ids)} documents, topic: {topic[:50]}...", flush=True)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Query for relevant content
                response = await client.post(
                    f"{self.course_generator_url}/api/v1/documents/query",
                    json={
                        "query": topic,
                        "document_ids": document_ids,
                        "top_k": max_chunks,
                        "include_metadata": True
                    }
                )

                if response.status_code != 200:
                    print(f"[RAG_CLIENT] Query failed: {response.status_code} - {response.text[:200]}", flush=True)
                    return None

                result = response.json()
                chunks = result.get("chunks", [])

                if not chunks:
                    print(f"[RAG_CLIENT] No relevant chunks found", flush=True)
                    return None

                # Build context string
                context_parts = []
                diagram_descriptions = []

                for chunk in chunks:
                    content = chunk.get("content", "")
                    metadata = chunk.get("metadata", {})
                    source = metadata.get("source", "Unknown")
                    score = chunk.get("score", 0)

                    # Check if this chunk contains diagram/schema descriptions
                    if include_diagrams and self._is_diagram_content(content):
                        diagram_descriptions.append(f"[DIAGRAM FROM {source}]\n{content}")
                    else:
                        context_parts.append(f"[SOURCE: {source} (relevance: {score:.2f})]\n{content}")

                # Combine context with diagrams prioritized
                full_context = ""

                if diagram_descriptions:
                    full_context += "=== DIAGRAMS AND SCHEMAS FROM SOURCE DOCUMENTS ===\n"
                    full_context += "Use these diagrams as reference for generating visuals:\n\n"
                    full_context += "\n\n".join(diagram_descriptions)
                    full_context += "\n\n"

                if context_parts:
                    full_context += "=== RELEVANT CONTENT FROM SOURCE DOCUMENTS ===\n"
                    full_context += "Base your training content on this material:\n\n"
                    full_context += "\n\n".join(context_parts)

                print(f"[RAG_CLIENT] Retrieved {len(chunks)} chunks, {len(diagram_descriptions)} diagrams, total {len(full_context)} chars", flush=True)

                return full_context if full_context else None

        except httpx.ConnectError as e:
            print(f"[RAG_CLIENT] Cannot connect to course-generator: {e}", flush=True)
            return None
        except Exception as e:
            print(f"[RAG_CLIENT] Error fetching RAG context: {e}", flush=True)
            return None

    def _is_diagram_content(self, content: str) -> bool:
        """
        Check if content likely describes a diagram or schema.
        """
        diagram_keywords = [
            "diagram", "schema", "schéma", "architecture", "flowchart",
            "organigramme", "diagramme", "workflow", "pipeline",
            "data flow", "flux de données", "système", "system design",
            "infrastructure", "topology", "topologie", "network",
            "composants", "components", "→", "->", "──", "│", "├", "└"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in diagram_keywords)

    async def get_document_diagrams(
        self,
        document_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Specifically extract diagram descriptions from documents.

        Returns:
            List of diagram descriptions with metadata
        """
        if not document_ids:
            return []

        print(f"[RAG_CLIENT] Extracting diagrams from {len(document_ids)} documents", flush=True)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Query for diagram-related content
                diagram_queries = [
                    "diagram architecture system",
                    "schema data flow pipeline",
                    "workflow process steps",
                    "infrastructure components topology"
                ]

                all_diagrams = []
                seen_content = set()

                for query in diagram_queries:
                    response = await client.post(
                        f"{self.course_generator_url}/api/v1/documents/query",
                        json={
                            "query": query,
                            "document_ids": document_ids,
                            "top_k": 5,
                            "include_metadata": True
                        }
                    )

                    if response.status_code == 200:
                        result = response.json()
                        for chunk in result.get("chunks", []):
                            content = chunk.get("content", "")
                            if content not in seen_content and self._is_diagram_content(content):
                                seen_content.add(content)
                                all_diagrams.append({
                                    "description": content,
                                    "source": chunk.get("metadata", {}).get("source", "Unknown"),
                                    "type": self._detect_diagram_type(content)
                                })

                print(f"[RAG_CLIENT] Found {len(all_diagrams)} diagram descriptions", flush=True)
                return all_diagrams

        except Exception as e:
            print(f"[RAG_CLIENT] Error extracting diagrams: {e}", flush=True)
            return []

    def _detect_diagram_type(self, content: str) -> str:
        """
        Detect the type of diagram from content description.
        """
        content_lower = content.lower()

        if any(kw in content_lower for kw in ["architecture", "infrastructure", "composant", "component", "service"]):
            return "architecture"
        elif any(kw in content_lower for kw in ["flow", "flux", "process", "étape", "step", "workflow"]):
            return "flowchart"
        elif any(kw in content_lower for kw in ["sequence", "séquence", "interaction", "message"]):
            return "sequence"
        elif any(kw in content_lower for kw in ["hierarchy", "hiérarchie", "tree", "arbre", "organization"]):
            return "hierarchy"
        elif any(kw in content_lower for kw in ["comparison", "comparaison", "vs", "versus", "différence"]):
            return "comparison"
        else:
            return "architecture"  # Default


# Singleton instance
_rag_client = None

def get_rag_client() -> RAGClient:
    """Get singleton RAG client instance."""
    global _rag_client
    if _rag_client is None:
        _rag_client = RAGClient()
    return _rag_client
