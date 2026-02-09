# Architecture Overview

Vue d'ensemble de l'architecture Viralify et des interactions entre services.

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              VIRALIFY PLATFORM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   Frontend  │────▶│ API Gateway │────▶│   Course    │                   │
│  │  (Next.js)  │     │ (Spring)    │     │  Generator  │                   │
│  │   :3000     │     │   :8080     │     │   :8007     │                   │
│  └─────────────┘     └─────────────┘     └──────┬──────┘                   │
│                                                  │                          │
│                              ┌───────────────────┼───────────────────┐      │
│                              │                   │                   │      │
│                              ▼                   ▼                   ▼      │
│                    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐ │
│                    │Presentation │     │    Media    │     │   Visual    │ │
│                    │  Generator  │     │  Generator  │     │  Generator  │ │
│                    │   :8006     │     │   :8004     │     │   :8003     │ │
│                    └──────┬──────┘     └─────────────┘     └─────────────┘ │
│                           │                                                 │
│                    ┌──────┴──────┐                                         │
│                    │             │                                          │
│                    ▼             ▼                                          │
│           ┌─────────────┐ ┌─────────────┐                                  │
│           │   MAESTRO   │ │   NEXUS     │                                  │
│           │   Engine    │ │   Engine    │                                  │
│           │   :8010     │ │   :8011     │                                  │
│           └─────────────┘ └─────────────┘                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              INFRASTRUCTURE                                 │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ PostgreSQL  │  │    Redis    │  │  RabbitMQ   │  │Elasticsearch│       │
│  │ + pgvector  │  │   Cache     │  │   Queue     │  │   Search    │       │
│  │   :5432     │  │   :6379     │  │   :5672     │  │   :9200     │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Services Principaux

### Course Generator (Port 8007)

Orchestrateur principal pour la génération de cours.

**Responsabilités:**
- Génération de curriculum avec LLM
- Gestion des documents RAG
- Orchestration du pipeline de génération
- Job management

**Technologies:**
- FastAPI (Python)
- LangGraph pour les workflows
- pgvector pour embeddings
- RabbitMQ pour jobs asynchrones

### Presentation Generator (Port 8006)

Génération des présentations vidéo.

**Responsabilités:**
- Génération de scripts
- Rendu de slides
- Composition vidéo FFmpeg
- Synchronisation audio-vidéo (SSVS)

**Technologies:**
- FastAPI (Python)
- FFmpeg pour composition
- PIL/Pillow pour images
- Whisper pour transcription

### Media Generator (Port 8004)

Gestion des médias audio/vidéo.

**Responsabilités:**
- TTS multi-provider
- Voice cloning (ElevenLabs)
- Éditeur vidéo
- Génération d'images

**Technologies:**
- FastAPI (Python)
- ElevenLabs API
- OpenAI TTS
- FFmpeg

## Services Spécialisés

### Visual Generator (Port 8003)

Génération de diagrammes professionnels.

**Fonctionnalités:**
- Python Diagrams library
- Mermaid via Kroki
- Graphviz pour layouts complexes
- Support AWS/Azure/GCP icons

### MAESTRO Engine (Port 8010)

Pipeline 5 couches pour génération sans documents.

**Couches:**
1. Domain Discovery
2. Knowledge Graph
3. Difficulty Calibration (4D)
4. Curriculum Sequencing
5. Content Generation

### NEXUS Engine (Port 8011)

Génération de code pédagogique.

**Fonctionnalités:**
- Code segmenté avec explications
- Multi-langage (Python, JS, Go, Rust...)
- Évolution progressive (v1 → v2 → v3)
- Erreurs courantes à éviter

### VQV-HALLU (Port 8009)

Validation qualité des voiceovers.

**Pipeline:**
1. Acoustic Analysis (spectral anomalies)
2. Linguistic Coherence (ASR reverse)
3. Semantic Alignment (embeddings)
4. Score Fusion

## Pipeline de Génération de Cours

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COURSE GENERATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PLANNING PHASE (Course Generator)                                       │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Context Analysis (detect category, audience)                      │    │
│  │ • Profile Adaptation (content preferences)                          │    │
│  │ • Element Suggestion (AI-suggested elements)                        │    │
│  │ • Quiz Planning (placement strategy)                                │    │
│  │ • Language Validation (target language)                             │    │
│  │ • Structure Validation (pedagogical quality)                        │    │
│  │ • Outline Refinement (iterative improvement)                        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   ↓                                         │
│  2. PRODUCTION PHASE (Presentation Generator)                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ For each lecture:                                                   │    │
│  │   • Script Generation (with RAG context)                           │    │
│  │   • Slide Rendering (code, diagrams, content)                      │    │
│  │   • TTS Generation (Direct Sync / SSVS)                            │    │
│  │   • VQV-HALLU Validation                                           │    │
│  │   • Video Composition (FFmpeg)                                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   ↓                                         │
│  3. FINALIZATION PHASE                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Quiz Generation (per section/lecture)                            │    │
│  │ • Video Concatenation                                              │    │
│  │ • ZIP Package Creation                                             │    │
│  │ • Metadata Generation                                              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. INGESTION                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Document Upload → Security Scan → Parsing → Chunking               │    │
│  │                                                                     │    │
│  │ Formats: PDF, DOCX, PPTX, XLSX, TXT, MD, URL, YouTube              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   ↓                                         │
│  2. EMBEDDING                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Chunks → E5-Large Multilingual Embeddings → pgvector Storage       │    │
│  │                                                                     │    │
│  │ Support: 100+ langues, cross-langue (FR/EN)                        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   ↓                                         │
│  3. RETRIEVAL                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Query → Vector Search → Cross-Encoder Re-ranking → Top-K Results   │    │
│  │                                                                     │    │
│  │ WeaveGraph: Query expansion via concept graph (+5% boost)          │    │
│  │ Resonance: Multi-hop propagation (decay=0.7, depth=3)             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   ↓                                         │
│  4. VERIFICATION                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ RAG Verifier v6: 90% minimum coverage garantie                     │    │
│  │                                                                     │    │
│  │ Méthodes: N-gram (40%) + Terms (30%) + Semantic (30%)             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Infrastructure

### PostgreSQL + pgvector

- Base de données principale
- Extension pgvector pour embeddings
- Index HNSW pour recherche rapide

### Redis

- Cache des jobs
- Session storage
- Rate limiting

### RabbitMQ

- Queue pour jobs asynchrones
- Dead Letter Queue (DLQ) pour erreurs
- Retry automatique

### Elasticsearch (Optionnel)

- Analytics et métriques
- Full-text search
- Logs centralisés

## Communication Inter-Services

### HTTP REST

Communication synchrone entre services:

```
Course Generator ──HTTP──▶ Presentation Generator
                 ──HTTP──▶ Media Generator
                 ──HTTP──▶ Visual Generator
```

### Message Queue

Communication asynchrone pour jobs longs:

```
API Gateway ──RabbitMQ──▶ Course Worker
```

## Déploiement

### Docker Compose (Development)

```yaml
services:
  course-generator:
    build: ./services/course-generator
    ports: ["8007:8000"]

  presentation-generator:
    build: ./services/presentation-generator
    ports: ["8006:8000"]

  # ... autres services
```

### Kubernetes (Production)

Manifests disponibles dans `/k8s`:

```
k8s/
├── namespace.yaml
├── configmap.yaml
├── secrets.yaml
├── deployments/
└── services/
```

## Monitoring

### Health Checks

Chaque service expose `/health`:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "database": true,
    "redis": true,
    "nexus_engine": true
  }
}
```

### Métriques Prometheus

```
/metrics  # Métriques Prometheus
```

### Logs

Format JSON structuré:

```json
{
  "timestamp": "2026-02-08T10:30:00Z",
  "level": "INFO",
  "service": "course-generator",
  "message": "Job completed",
  "job_id": "abc123"
}
```
