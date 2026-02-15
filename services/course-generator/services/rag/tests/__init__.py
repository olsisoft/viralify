"""
RAG Module Unit Tests

Comprehensive test suite for the RAG (Retrieval-Augmented Generation) service.

Test Modules:
- test_keyword_extractor.py - Keyword extraction and coverage computation
- test_token_allocator.py - Token budget allocation algorithms
- test_scoring_models.py - Document relevance scoring models
- test_prompts.py - Prompt builders for LLM interactions
- test_structure_extractor.py - Document structure extraction
- test_chunk_prioritizer.py - Chunk prioritization with boost markers
- test_context_builder.py - Context building with token truncation
- test_image_retriever.py - Image relevance scoring for diagrams
- test_repository.py - Document repository (PostgreSQL + in-memory)
- test_file_storage.py - File storage (S3 + local)
- test_rag_service.py - Main RAG service orchestrator

Running Tests:
    # From course-generator directory
    pytest services/rag/tests/ -v

    # With coverage
    pytest services/rag/tests/ --cov=services.rag --cov-report=html

    # Specific test file
    pytest services/rag/tests/test_keyword_extractor.py -v
"""
