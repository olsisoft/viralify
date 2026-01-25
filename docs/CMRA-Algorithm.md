# CMRA — Conceptual Multi-lingual Resonance Algorithm

## Overview

CMRA is the core algorithm powering RAG Verifier v6 in Viralify. It ensures that AI-generated content faithfully reflects source documents, even across different languages.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CMRA Pipeline                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Document Upload    Concept Extraction    Query Expansion           │
│       │                    │                    │                   │
│       ▼                    ▼                    ▼                   │
│  ┌─────────┐         ┌──────────┐         ┌──────────┐             │
│  │ Source  │───────▶ │ WeaveGraph│───────▶│ Expanded │             │
│  │ Content │         │ Builder  │         │  Terms   │             │
│  └─────────┘         └──────────┘         └──────────┘             │
│                            │                    │                   │
│                            ▼                    ▼                   │
│                      ┌──────────┐         ┌──────────┐             │
│                      │ pgvector │         │Resonance │             │
│                      │  Store   │◀───────▶│ Matcher  │             │
│                      └──────────┘         └──────────┘             │
│                            │                    │                   │
│                            ▼                    ▼                   │
│                      ┌──────────────────────────┐                  │
│                      │   RAG Coverage Score     │                  │
│                      │   (semantic + boosts)    │                  │
│                      └──────────────────────────┘                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Three Phases

### Phase 1: E5-Large Multilingual Embeddings

**Purpose:** Enable cross-language semantic matching (e.g., English source → French output)

**Model:** `intfloat/multilingual-e5-large`
- 1024-dimensional embeddings
- Supports 100+ languages
- Trained on multilingual pairs

**Why E5-Large?**
- MiniLM only works well for same-language
- E5-Large maps "Apache Kafka" (EN) close to "Apache Kafka" (FR) in embedding space
- Essential for Viralify's multilingual content generation

```python
# E5-Large embedding
embedding = engine.embed("Apache Kafka messaging system")  # 1024 dims
similarity = cosine_similarity(source_emb, generated_emb)  # 0.0 - 1.0
```

### Phase 2: WeaveGraph Concept Graph

**Purpose:** Build a knowledge graph of concepts from documents for query expansion

**Components:**
- `ConceptExtractor`: NLP-based extraction of technical terms, entities, keywords
- `WeaveGraphBuilder`: Orchestrates extraction, embedding, and edge creation
- `WeaveGraphPgVectorStore`: PostgreSQL + pgvector storage

**Tables:**
```sql
weave_concepts (
    id UUID,
    canonical_name VARCHAR(255),  -- "apache_kafka"
    name VARCHAR(500),            -- "Apache Kafka"
    language VARCHAR(10),         -- "en", "fr"
    embedding vector(1024),       -- E5-Large embedding
    frequency INT,                -- occurrence count
    user_id VARCHAR(255)
)

weave_edges (
    source_concept_id UUID,
    target_concept_id UUID,
    relation_type VARCHAR(50),    -- similar, translation, synonym
    weight FLOAT                  -- 0.0 - 1.0
)
```

**Query Expansion:**
```
Original: ["kafka", "messaging"]
Expanded: ["kafka", "messaging", "apache kafka", "event streaming", "pub/sub"]
Boost: +2-5% coverage
```

### Phase 3: Resonance Propagation

**Purpose:** Propagate matching scores through the concept graph to find indirect matches

**Algorithm:**
```
resonance(neighbor) = parent_score × edge_weight × decay^depth
```

**Configuration:**
```python
decay_factor = 0.7      # Score reduction per hop
max_depth = 3           # Maximum propagation hops
min_resonance = 0.10    # Minimum score to consider
```

**Boost Multipliers:**
| Relation Type | Multiplier |
|--------------|------------|
| translation  | ×1.2       |
| synonym      | ×1.1       |
| similar      | ×1.0       |

**Example Propagation:**
```
"kafka" (direct match, score=1.0)
    ├── "event streaming" (similar, weight=0.85)
    │       resonance = 1.0 × 0.85 × 0.7^1 = 0.595
    │       ├── "pub/sub" (similar, weight=0.75)
    │       │       resonance = 0.595 × 0.75 × 0.7^2 = 0.219
    │       └── "message queue" (similar, weight=0.80)
    │               resonance = 0.595 × 0.80 × 0.7^2 = 0.233
    └── "Apache Kafka" (translation, weight=0.95)
            resonance = 1.0 × 0.95 × 0.7^1 × 1.2 = 0.798
```

## Coverage Calculation

### Final Score
```python
base_coverage = semantic_similarity(generated, source)  # 0.0 - 1.0
expansion_boost = weave_graph_expansion_boost           # 0.0 - 0.05
resonance_boost = resonance_propagation_boost           # 0.0 - 0.15

final_coverage = min(1.0, base_coverage + expansion_boost + resonance_boost)
```

### Compliance Thresholds
```python
MIN_SEMANTIC_THRESHOLD = 0.35        # Cross-language
MIN_SEMANTIC_THRESHOLD_SAME_LANG = 0.50  # Same language
MIN_KEYWORD_THRESHOLD = 0.30
MIN_TOPIC_THRESHOLD = 0.40
MAX_HALLUCINATION_RATIO = 0.30
```

## Verification Modes

### Auto Mode (Default)
- Detects source/generated language
- Uses semantic-only for cross-language
- Uses full validation for same-language

### Semantic-Only Mode
- Only checks embedding similarity
- Best for cross-language content
- Skips keyword/topic validation

### Comprehensive Mode
- Semantic similarity
- Keyword coverage
- Topic matching
- Hallucination detection

## Data Flow

```
1. Document Upload
   └── SourceLibrary sends sourceIds to backend

2. RAG Context Fetch
   └── course-generator retrieves document content

3. Concept Extraction (Background Task)
   ├── ConceptExtractor parses content
   ├── E5-Large generates embeddings
   └── pgvector stores concepts + edges

4. Presentation Generation
   ├── WeaveGraph expands query terms
   ├── ResonanceMatcher propagates scores
   └── RAGVerifier computes final coverage

5. Compliance Check
   ├── Coverage >= threshold → ✅ COMPLIANT
   └── Coverage < threshold → ❌ NON-COMPLIANT
```

## Environment Variables

```bash
# RAG Verifier Mode
RAG_VERIFIER_MODE=auto              # auto, semantic_only, comprehensive

# Embedding Backend
RAG_EMBEDDING_BACKEND=e5-large      # e5-large, minilm, bge-m3

# WeaveGraph
WEAVE_GRAPH_ENABLED=true

# Resonance
RESONANCE_ENABLED=true
RESONANCE_DECAY=0.7
RESONANCE_MAX_DEPTH=3
```

## Performance Characteristics

| Component | Latency | Memory |
|-----------|---------|--------|
| E5-Large embedding (per text) | ~50ms | ~2GB model |
| Concept extraction (500 terms) | ~2s | ~100MB |
| pgvector similarity search | ~10ms | Index-dependent |
| Resonance propagation | ~5ms | Graph size |

## Files

```
services/presentation-generator/
├── services/
│   ├── rag_verifier.py              # Main RAGCoverageVerifier
│   ├── weave_graph/
│   │   ├── __init__.py              # Exports
│   │   ├── models.py                # ConceptNode, ConceptEdge, etc.
│   │   ├── concept_extractor.py     # NLP extraction
│   │   ├── graph_builder.py         # WeaveGraphBuilder
│   │   ├── pgvector_store.py        # PostgreSQL storage
│   │   └── resonance_matcher.py     # ResonanceMatcher
│   └── sync/
│       └── embedding_engine.py      # E5-Large, MiniLM engines
└── main.py                          # Background concept extraction
```

## Version History

| Version | Features |
|---------|----------|
| v1-v4   | Basic keyword/topic matching |
| v5      | + WeaveGraph query expansion |
| v6      | + E5-Large + Resonance propagation |

---

*CMRA is designed for educational content generation where source fidelity is critical, especially in multilingual contexts.*
