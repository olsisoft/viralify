"""
Viralify API - RAG Course Generation Example (Python)

This example demonstrates how to generate a course from uploaded documents
using RAG (Retrieval Augmented Generation).

Requirements:
    pip install requests python-dotenv

Usage:
    export VIRALIFY_API_KEY="your_api_key"
    python rag_course.py document1.pdf document2.docx
"""

import os
import sys
import time
import requests
from pathlib import Path
from typing import List, Dict, Any


API_KEY = os.getenv("VIRALIFY_API_KEY")
BASE_URL = os.getenv("VIRALIFY_BASE_URL", "https://api.viralify.io")
USER_ID = os.getenv("VIRALIFY_USER_ID", "user_example")


def get_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }


def upload_document(file_path: str, role: str = "theory") -> Dict[str, Any]:
    """Upload a document for RAG processing."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"  Uploading: {path.name}...")

    with open(path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/api/v1/documents/upload",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": (path.name, f)},
            data={
                "user_id": USER_ID,
                "pedagogical_role": role
            }
        )
        response.raise_for_status()
        doc = response.json()
        print(f"    -> ID: {doc['id']}, Status: {doc['status']}")
        return doc


def wait_for_document_ready(doc_id: str, timeout: int = 300) -> bool:
    """Wait for document to be processed and indexed."""
    start = time.time()

    while time.time() - start < timeout:
        response = requests.get(
            f"{BASE_URL}/api/v1/documents/{doc_id}",
            headers=get_headers()
        )
        response.raise_for_status()
        doc = response.json()

        if doc["status"] == "ready":
            print(f"    Document {doc_id} ready ({doc.get('chunk_count', 0)} chunks)")
            return True
        elif doc["status"] == "failed":
            print(f"    Document {doc_id} failed to process")
            return False

        time.sleep(2)

    print(f"    Document {doc_id} processing timeout")
    return False


def query_documents(document_ids: List[str], query: str) -> Dict[str, Any]:
    """Test RAG query on documents."""
    response = requests.post(
        f"{BASE_URL}/api/v1/documents/query",
        headers=get_headers(),
        json={
            "query": query,
            "document_ids": document_ids,
            "max_results": 5
        }
    )
    response.raise_for_status()
    return response.json()


def generate_course_with_rag(
    topic: str,
    document_ids: List[str],
    language: str = "en"
) -> str:
    """Start course generation with RAG context."""
    response = requests.post(
        f"{BASE_URL}/api/v1/courses/generate",
        headers=get_headers(),
        json={
            "topic": topic,
            "document_ids": document_ids,
            "difficulty_start": "intermediate",
            "difficulty_end": "advanced",
            "structure": {
                "number_of_sections": 4,
                "lectures_per_section": 3
            },
            "context": {
                "category": "tech"
            },
            "language": language,
            "quiz_config": {
                "enabled": True,
                "frequency": "per_section"
            }
        }
    )
    response.raise_for_status()
    return response.json()["job_id"]


def wait_for_course(job_id: str, timeout: int = 3600) -> Dict[str, Any]:
    """Wait for course generation to complete."""
    start = time.time()

    while time.time() - start < timeout:
        response = requests.get(
            f"{BASE_URL}/api/v1/courses/jobs/{job_id}",
            headers=get_headers()
        )
        response.raise_for_status()
        status = response.json()

        progress = status.get("progress", 0)
        stage = status.get("status", "unknown")
        print(f"  [{stage.upper()}] {progress:.1f}%")

        if stage == "completed":
            return status
        elif stage == "failed":
            raise Exception(f"Course generation failed: {status.get('error')}")

        time.sleep(15)

    raise TimeoutError("Course generation timeout")


def main():
    if not API_KEY:
        print("Error: VIRALIFY_API_KEY not set")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python rag_course.py <file1> [file2] [file3] ...")
        print("\nExample:")
        print("  python rag_course.py chapter1.pdf chapter2.pdf references.docx")
        sys.exit(1)

    files = sys.argv[1:]
    print(f"\n=== RAG Course Generation ===")
    print(f"Documents: {len(files)} files\n")

    # Step 1: Upload documents
    print("Step 1: Uploading documents...")
    document_ids = []

    for file_path in files:
        try:
            # Assign pedagogical roles based on filename
            role = "theory"
            if "example" in file_path.lower():
                role = "example"
            elif "reference" in file_path.lower():
                role = "reference"

            doc = upload_document(file_path, role)
            document_ids.append(doc["id"])
        except Exception as e:
            print(f"  Error uploading {file_path}: {e}")

    if not document_ids:
        print("No documents uploaded successfully")
        sys.exit(1)

    # Step 2: Wait for processing
    print("\nStep 2: Waiting for document processing...")
    ready_ids = []

    for doc_id in document_ids:
        if wait_for_document_ready(doc_id):
            ready_ids.append(doc_id)

    if not ready_ids:
        print("No documents processed successfully")
        sys.exit(1)

    # Step 3: Test RAG query
    print("\nStep 3: Testing RAG retrieval...")
    topic = input("Enter course topic (or press Enter for auto-detect): ").strip()

    if not topic:
        # Auto-detect topic from first document
        result = query_documents(ready_ids, "What is the main topic of this document?")
        if result.get("results"):
            print(f"  Found context: {result['results'][0]['content'][:200]}...")
            topic = input("Enter course topic based on content: ").strip()

    if not topic:
        topic = "Course from Documents"

    # Step 4: Generate course
    print(f"\nStep 4: Generating course: '{topic}'...")
    job_id = generate_course_with_rag(topic, ready_ids)
    print(f"  Job ID: {job_id}")

    # Step 5: Wait for completion
    print("\nStep 5: Waiting for generation...")
    try:
        result = wait_for_course(job_id)

        print("\n=== Course Complete! ===")
        print(f"Videos: {len(result['output_urls']['videos'])}")
        print(f"ZIP: {result['output_urls']['zip']}")

        # Check RAG verification if available
        if "rag_verification" in result:
            rag = result["rag_verification"]
            print(f"\nRAG Coverage: {rag.get('coverage', 0)*100:.1f}%")
            print(f"Compliant: {rag.get('is_compliant', False)}")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
