# Viralify - Claude Code Context

---

## Règles de travail Claude

### Processus obligatoire avant tout changement de code

1. **Comprendre** - Lire les fichiers concernés, ne jamais supposer
2. **Expliquer** - Décrire le problème tel que compris
3. **Proposer** - Présenter 2-3 approches avec pros/cons
4. **Valider** - Attendre l'approbation explicite de l'utilisateur
5. **Implémenter** - Écrire le code seulement après validation
6. **Montrer** - Afficher le diff avant de commit
7. **Confirmer** - Attendre l'approbation pour commit/push

### Ne jamais faire sans validation

- Changements architecturaux
- Ajout de dépendances
- Modification de fichiers de config (docker-compose, etc.)
- Suppression de code existant
- Refactoring non demandé

### Session tracking

**Dernier commit:** `pending` - feat: implement MAESTRO Engine integration (Phase 8)
**Date:** 2026-01-26
**Travail en cours:** Phase 8 MAESTRO - Microservice de génération avancée avec calibration 4D

### RAG Verifier v6 - Phases Complétées

| Phase | Fonctionnalité | Status |
|-------|---------------|--------|
| Phase 1 | E5-Large Multilingual embeddings | ✅ |
| Phase 2 | WeaveGraph concept graph + pgvector | ✅ |
| Phase 3 | Resonate Match + auto-extraction concepts | ✅ |

**Flux complet:**
1. User uploade document → SourceLibrary envoie sourceIds
2. RAG context reçu → concepts extraits en background (WeaveGraphBuilder)
3. Concepts stockés dans pgvector avec embeddings E5-large
4. Query expansion via graph de concepts (+2-5% boost)
5. Resonance propagation multi-hop (decay=0.7, depth=3)
6. Boost combiné appliqué au coverage score

### Travaux futurs planifiés

#### Phase 8: Viralify Diagrams (Fork de mingrammer/diagrams)
**Objectif:** Fork personnalisé de la librairie Python Diagrams pour améliorer la lisibilité

**Problèmes actuels:**
- Layout Graphviz automatique difficile à contrôler
- Diagrammes peu lisibles avec beaucoup de nodes
- Pas de contrôle sur le positionnement
- Style/icônes trop détaillés pour vidéo

**Améliorations prévues:**
- Layouts forcés (grille, horizontal, vertical, radial)
- Zones visuelles avec bordures claires
- Limite de nodes (max 8-10, auto-simplification)
- Icônes simplifiées pour lisibilité vidéo
- Labels courts avec truncation automatique
- Thèmes Viralify (dark/light cohérents)
- Export SVG optimisé pour animation

**Licence:** MIT (permet modification et usage commercial)

---

## Project Overview

Viralify est une plateforme de création de contenu viral pour les réseaux sociaux, avec un focus particulier sur la génération automatisée de cours vidéo éducatifs.

### Niche Tech - Plateforme pour toute la Tech IT

Cette plateforme est conçue pour couvrir **l'ensemble de l'écosystème IT**, incluant:
- **545+ métiers** tech (Data Engineer, MLOps Engineer, Cloud Architect, etc.)
- **80+ domaines** (Data Engineering, DevOps, Cybersecurity, Quantum Computing, etc.)
- **120+ langages** de programmation (Python, Go, Rust, Solidity, Qiskit, etc.)

Les agents génèrent du contenu adapté à chaque profil, niveau et domaine technique.

## Architecture

### Services principaux
- **frontend** (Next.js) - Port 3000
- **api-gateway** (Spring Boot) - Port 8080
- **course-generator** (FastAPI) - Port 8007
- **presentation-generator** (FastAPI) - Port 8006
- **media-generator** (FastAPI) - Port 8004
- **visual-generator** (FastAPI) - Port 8003 (microservice diagrammes)
- **vqv-hallu** (FastAPI) - Port 8009 (validation voiceover TTS)
- **maestro-engine** (FastAPI) - Port 8010 (génération avancée avec calibration 4D)

### Infrastructure
- PostgreSQL, Redis, RabbitMQ, Elasticsearch
- **Kroki** - Self-hosted diagram rendering (Mermaid, PlantUML, D2, GraphViz)
- Docker Compose pour l'orchestration

---

## Course Generator - Roadmap

### Phase 1: Éléments de leçon adaptatifs + Quiz (ACTUELLE)

#### Objectifs
1. **Option A** - Mapping des éléments par catégorie de profil
2. **Option C** - Suggestion IA des éléments selon le sujet
3. **Quiz obligatoires** - Tous les cours incluent des évaluations

#### Éléments par catégorie

| Catégorie | Éléments spécifiques |
|-----------|---------------------|
| **Tech** | `code_demo`, `terminal_output`, `architecture_diagram`, `debug_tips` |
| **Business** | `case_study`, `framework_template`, `roi_metrics`, `action_checklist`, `market_analysis` |
| **Health** | `exercise_demo`, `safety_warning`, `body_diagram`, `progression_plan`, `rest_guidance` |
| **Creative** | `before_after`, `technique_demo`, `tool_tutorial`, `creative_exercise`, `critique_section` |
| **Education** | `memory_aid`, `practice_problem`, `multiple_explanations`, `summary_card` |
| **Lifestyle** | `daily_routine`, `reflection_exercise`, `goal_setting`, `habit_tracker`, `milestone` |

#### Éléments communs (tous les cours)

| Élément | Description | Obligatoire |
|---------|-------------|-------------|
| `concept_intro` | Introduction du concept principal | Oui |
| `voiceover` | Narration explicative | Oui |
| `curriculum_slide` | Position dans le cours | Oui |
| `conclusion` | Récapitulatif des points clés | Oui |
| `quiz_evaluation` | Quiz interactif | **Oui** |

#### Configuration des Quiz

- **Format**: Style Udemy (QCM, Vrai/Faux, association, réponses courtes)
- **Fréquence**: Configurable par l'utilisateur au frontend
  - Par lecture
  - Par section
  - À la fin du cours uniquement
  - Personnalisé (toutes les N lectures)
- **Génération**: L'IA génère les questions basées sur le contenu de la leçon

#### Suggestion IA (Option C)

L'IA analyse le sujet, la description et le contexte pour:
1. Détecter automatiquement la catégorie si non spécifiée
2. Suggérer les éléments les plus pertinents avec score de pertinence
3. Proposer des éléments additionnels basés sur le sujet spécifique

---

### Phase 2: RAG - Documentation Source (COMPLÉTÉE + INTÉGRÉE + 90% GARANTI)

#### Objectifs
L'utilisateur peut uploader des documents comme source de contenu pour la génération de cours.
Le système garantit que **90% minimum** du contenu généré provient des documents source.

#### Formats supportés
- PDF (via PyMuPDF)
- Word (DOCX, DOC)
- PowerPoint (PPTX, PPT)
- Texte (TXT, MD)
- Excel (XLSX, XLS, CSV)
- URLs/pages web
- Vidéos YouTube (transcription via youtube-transcript-api)

#### Sécurité des documents (Implémentée)
- **Validation du type MIME** avec python-magic
- **Vérification extension vs contenu** réel
- **Détection de macros** (VBA dans Office)
- **Détection d'objets embarqués** dangereux
- **Limite de taille** par type de fichier (50 MB max)
- **Détection de patterns malicieux** (injection, scripts)
- **Protection zip bomb** (ratio de compression)
- **Sanitization des noms de fichiers**

#### Architecture RAG (Implémentée)
```
services/course-generator/
├── models/
│   └── document_models.py     # Document, DocumentChunk, RAGQuery models
└── services/
    ├── security_scanner.py    # Validation sécurité complète
    ├── document_parser.py     # Extraction multi-format
    ├── vector_store.py        # Embeddings OpenAI + ChromaDB/Memory
    └── retrieval_service.py   # Orchestration RAG complète
```

#### Endpoints API Documents
- `POST /api/v1/documents/upload` - Upload fichier
- `POST /api/v1/documents/upload-url` - Import URL/YouTube
- `GET /api/v1/documents` - Liste documents
- `GET /api/v1/documents/{id}` - Détails document
- `DELETE /api/v1/documents/{id}` - Supprimer
- `POST /api/v1/documents/query` - Recherche RAG
- `GET /api/v1/documents/context/{course_id}` - Contexte pour génération

#### Frontend Components
- `lib/document-types.ts` - Types TypeScript
- `components/DocumentUpload.tsx` - Upload drag & drop + URL

---

### Phase 3: Édition Vidéo Utilisateur (COMPLÉTÉE)

#### Objectifs
L'utilisateur peut modifier la vidéo de cours générée et ajouter ses propres enregistrements.

#### Fonctionnalités
- Visualisation de la timeline du cours
- Remplacement de segments vidéo
- Ajout d'enregistrements personnels (webcam, screen recording)
- Ajustement de la synchronisation audio/vidéo
- Insertion de slides personnalisées
- Trim/cut des sections
- Transitions entre segments
- Overlays texte et image
- Export vidéo final

#### Architecture Backend (media-generator)
```
services/media-generator/
├── models/
│   └── video_editor_models.py    # Modèles Pydantic (VideoProject, VideoSegment, etc.)
└── services/
    ├── timeline_service.py       # Gestion timeline + ProjectRepository
    ├── segment_manager.py        # Upload et traitement des médias
    └── video_merge_service.py    # Rendu FFmpeg final
```

#### Endpoints API Video Editor
- `POST /api/v1/editor/projects` - Créer un projet
- `GET /api/v1/editor/projects` - Lister les projets
- `GET /api/v1/editor/projects/{id}` - Détails projet
- `DELETE /api/v1/editor/projects/{id}` - Supprimer projet
- `PATCH /api/v1/editor/projects/{id}/settings` - Paramètres projet
- `POST /api/v1/editor/projects/{id}/segments` - Ajouter segment
- `POST /api/v1/editor/projects/{id}/segments/upload` - Upload média
- `PATCH /api/v1/editor/projects/{id}/segments/{segId}` - Modifier segment
- `DELETE /api/v1/editor/projects/{id}/segments/{segId}` - Supprimer segment
- `POST /api/v1/editor/projects/{id}/segments/reorder` - Réordonner
- `POST /api/v1/editor/projects/{id}/segments/{segId}/split` - Diviser segment
- `POST /api/v1/editor/projects/{id}/overlays/text` - Overlay texte
- `POST /api/v1/editor/projects/{id}/overlays/image` - Overlay image
- `POST /api/v1/editor/projects/{id}/render` - Lancer le rendu
- `GET /api/v1/editor/render-jobs/{jobId}` - Statut rendu
- `POST /api/v1/editor/projects/{id}/preview` - Preview rapide
- `GET /api/v1/editor/supported-formats` - Formats supportés

#### Frontend Components (editor)
```
frontend/src/app/dashboard/studio/editor/
├── lib/
│   └── editor-types.ts           # Types TypeScript
├── hooks/
│   └── useVideoEditor.ts         # Hook React pour l'API
├── components/
│   ├── Timeline.tsx              # Timeline principale
│   ├── SegmentItem.tsx           # Item de segment
│   └── SegmentProperties.tsx     # Panel propriétés
└── page.tsx                      # Page éditeur
```

#### Formats supportés
- **Vidéo**: mp4, mov, avi, mkv, webm, m4v (max 500 MB)
- **Audio**: mp3, wav, aac, m4a, ogg (max 50 MB)
- **Image**: jpg, jpeg, png, gif, webp (max 10 MB)

---

### Phase 4: Voice Cloning (COMPLÉTÉE)

#### Objectifs
L'utilisateur peut cloner sa voix pour personnaliser la narration des cours.

#### Fonctionnalités
- Upload d'échantillons vocaux (minimum 30 secondes)
- Analyse qualité audio (bruit, clarté)
- Entraînement du modèle via ElevenLabs API
- Génération TTS avec la voix clonée
- Ajustement stabilité, similarité, style
- Preview avant génération complète
- Consentement explicite obligatoire

#### Provider
- **ElevenLabs** - Instant Voice Cloning API
- Modèle: eleven_multilingual_v2
- Output: MP3 44.1kHz

#### Architecture Backend (media-generator)
```
services/media-generator/
├── models/
│   └── voice_cloning_models.py     # VoiceProfile, VoiceSample, etc.
└── services/
    ├── voice_sample_service.py     # Upload + validation audio
    ├── voice_cloning_service.py    # Intégration ElevenLabs
    └── voice_profile_manager.py    # Orchestration workflow
```

#### Endpoints API Voice Cloning
- `POST /api/v1/voice/profiles` - Créer profil vocal
- `GET /api/v1/voice/profiles` - Lister profils
- `GET /api/v1/voice/profiles/{id}` - Détails profil
- `DELETE /api/v1/voice/profiles/{id}` - Supprimer profil
- `PATCH /api/v1/voice/profiles/{id}` - Modifier profil
- `POST /api/v1/voice/profiles/{id}/samples` - Upload échantillon
- `DELETE /api/v1/voice/profiles/{id}/samples/{sampleId}` - Supprimer échantillon
- `POST /api/v1/voice/profiles/{id}/train` - Démarrer entraînement
- `GET /api/v1/voice/profiles/{id}/training-status` - Statut entraînement
- `POST /api/v1/voice/profiles/{id}/generate` - Générer audio
- `POST /api/v1/voice/profiles/{id}/preview` - Preview rapide
- `GET /api/v1/voice/requirements` - Requis pour échantillons
- `GET /api/v1/voice/usage` - Stats utilisation API

#### Frontend Components (voice-clone)
```
frontend/src/app/dashboard/studio/voice-clone/
├── lib/
│   └── voice-types.ts              # Types TypeScript
├── hooks/
│   └── useVoiceClone.ts            # Hook React pour l'API
├── components/
│   └── VoiceSampleUpload.tsx       # Upload avec drag & drop
└── page.tsx                        # Page gestion des voix
```

#### Requirements échantillons
- **Formats**: mp3, wav, m4a, ogg, webm, flac, aac
- **Durée par sample**: 5-300 secondes
- **Durée totale min**: 30 secondes
- **Durée idéale**: 60 secondes
- **Taille max**: 50 MB par fichier

#### Sécurité & Consentement
- Consentement explicite obligatoire avant entraînement
- Enregistrement IP et timestamp du consentement
- Profils suspendables en cas d'abus

---

## Décisions techniques

### Choisies
- [x] Éléments adaptatifs par catégorie (Option A)
- [x] Suggestion IA des éléments (Option C)
- [x] Quiz obligatoires format Udemy
- [x] Fréquence quiz configurable par utilisateur
- [x] Support multi-format documents (Phase 2)
- [x] Validation sécurité documents obligatoire
- [x] Vector store: ChromaDB par défaut, InMemory pour dev, architecture extensible
- [x] Embeddings: OpenAI text-embedding-3-small
- [x] Code generation: GPT-4o avec TechPromptBuilder contextuel (Phase 6)
- [x] Diagrammes: Librairie Diagrams (Python) avec icônes AWS/Azure/GCP (Phase 6)
- [x] Couverture tech: 545+ métiers, 80+ domaines, 120+ langages (Phase 6)
- [x] Multi-provider LLM: 8 providers (OpenAI, Groq, DeepSeek, Mistral, Together, xAI, Ollama, RunPod)
- [x] RAG Verifier v4: E5-large multilingue pour cross-langue (FR/EN)
- [x] Training Logger: Collecte JSONL pour fine-tuning futur
- [x] WeaveGraph: Graphe de concepts pour expansion de requêtes RAG (Phase 2)
- [x] Resonate Match: Propagation multi-hop avec decay pour RAG (Phase 3)

### En attente
- [x] Provider de voice cloning: ElevenLabs (Phase 4 complétée)
- [x] Éditeur vidéo: web-based (Phase 3 complétée)
- [x] Migration vector store vers pgvector pour production (Phase 5)
- [x] Analytics & Metrics (Phase 5A)
- [x] Multi-langue 10 langues (Phase 5B)
- [x] Monétisation Stripe + PayPal (Phase 5C)
- [x] Collaboration avec équipes (Phase 5D)
- [x] Enhanced code/diagram generation (Phase 6 complétée)

---

## Phase 1 - Implémentation (COMPLÉTÉE)

### Backend (course-generator)

**Nouveaux fichiers créés:**
- `models/lesson_elements.py` - Modèles pour éléments par catégorie + quiz
- `services/element_suggester.py` - Service IA de suggestion d'éléments (Option C)
- `services/quiz_generator.py` - Générateur de quiz style Udemy

**Fichiers modifiés:**
- `models/course_models.py` - Ajout QuizConfigRequest, AdaptiveElementsRequest
- `main.py` - Nouveaux endpoints API

**Nouveaux endpoints API:**
- `GET /api/v1/courses/config/categories` - Liste des catégories
- `GET /api/v1/courses/config/elements/{category}` - Éléments par catégorie
- `POST /api/v1/courses/config/suggest-elements` - Suggestion IA
- `GET /api/v1/courses/config/detect-category` - Détection auto catégorie
- `GET /api/v1/courses/config/quiz-options` - Options de quiz

### Frontend (courses)

**Nouveaux fichiers créés:**
- `lib/lesson-elements.ts` - Types TypeScript pour éléments et quiz
- `components/AdaptiveLessonElements.tsx` - Composant éléments adaptatifs
- `components/QuizConfigPanel.tsx` - Configuration des quiz

**Fichiers modifiés:**
- `lib/course-types.ts` - Ajout QuizConfig, AdaptiveElementsConfig
- `page.tsx` - État initial avec quiz et éléments adaptatifs

---

## Phase 2 - Implémentation (COMPLÉTÉE)

### Backend (course-generator)

**Nouveaux fichiers créés:**
- `models/document_models.py` - Modèles Document, DocumentChunk, RAG
  - Document: métadonnées fichier, statut, chunks
  - DocumentChunk: contenu + embedding pour RAG
  - SecurityScanResult: résultats validation sécurité
  - RAGQueryRequest/Response: requêtes de recherche
- `services/security_scanner.py` - Validation sécurité complète
  - Validation MIME type avec libmagic
  - Détection macros VBA
  - Détection objets embarqués dangereux
  - Protection path traversal et zip bombs
  - Patterns malicieux (scripts, injection)
- `services/document_parser.py` - Extraction multi-format
  - PDF via PyMuPDF
  - DOCX/DOC via python-docx
  - PPTX/PPT via python-pptx
  - XLSX/XLS via openpyxl/xlrd
  - CSV, TXT, Markdown
  - URLs via httpx + BeautifulSoup
  - YouTube via youtube-transcript-api
- `services/vector_store.py` - Embeddings et stockage
  - EmbeddingService: OpenAI text-embedding-3-small
  - ChromaVectorStore: backend ChromaDB
  - InMemoryVectorStore: backend développement
  - VectorStoreFactory: abstraction backends
- `services/retrieval_service.py` - Orchestration RAG
  - Upload et processing documents
  - Recherche sémantique
  - Construction de contexte pour génération

**Fichiers modifiés:**
- `main.py` - Endpoints API documents + initialisation RAGService
- `requirements.txt` - Dépendances RAG (PyMuPDF, python-docx, chromadb, etc.)

### Frontend (courses)

**Nouveaux fichiers créés:**
- `lib/document-types.ts` - Types TypeScript pour documents RAG
- `components/DocumentUpload.tsx` - Composant upload drag & drop + URL

### Variables d'environnement RAG
```
VECTOR_BACKEND=memory|chroma|pinecone|pgvector
DOCUMENT_STORAGE_PATH=/tmp/viralify/documents
```

### Intégration RAG avec Génération de Cours

**Flux complet:**
1. L'utilisateur uploade des documents via le composant `DocumentUpload`
2. Les documents sont parsés, sécurisés et vectorisés
3. Lors du preview/generate, les `document_ids` sont envoyés à l'API
4. Le backend récupère le contexte RAG via `rag_service.get_context_for_course_generation()`
5. Le `course_planner` inclut ce contexte dans les prompts GPT-4
6. Les cours générés sont basés sur le contenu des documents source

**Fichiers modifiés pour l'intégration:**
- `models/course_models.py` - Ajout `document_ids` et `rag_context` aux requests
- `services/course_planner.py` - Nouveau `_build_rag_section()`, prompts adaptés
- `main.py` - Fetch RAG context avant génération outline/cours
- `lib/course-types.ts` - Types TypeScript avec `document_ids`
- `hooks/useCourseGeneration.ts` - Envoi des document_ids dans les appels API

#### RAG 90% Coverage (Amélioration Janvier 2026)

**Objectif:** Garantir que 90%+ du contenu généré provient des documents source.

**Paramètres optimisés:**

| Paramètre | Avant | Après | Impact |
|-----------|-------|-------|--------|
| `max_chunks` | 10-15 | **40** | +166% de contexte récupéré |
| `max_chars` | 24000 | **64000** | Documents longs supportés |
| Instructions prompt | Soft | **Mandatory 90%** | Moins d'hallucinations |

**RAGVerifier - Service de vérification (nouveau):**

```
services/presentation-generator/services/rag_verifier.py
```

Analyse le contenu généré vs les documents source:

| Méthode | Poids | Description |
|---------|-------|-------------|
| **N-gram overlap** | 40% | Détecte les phrases copiées directement |
| **Term coverage** | 30% | Vérifie l'utilisation des termes techniques |
| **Sequence similarity** | 30% | Mesure la similarité globale |

**Métriques disponibles dans la réponse API:**

```json
{
  "script": {
    "rag_verification": {
      "coverage": 0.92,
      "is_compliant": true,
      "summary": "✅ RAG COMPLIANT: 92.0% coverage (threshold: 90%)",
      "potential_hallucinations": 0
    }
  }
}
```

**Logs serveur après génération:**
```
[PLANNER] ✅ RAG COMPLIANT: 92.3% coverage (threshold: 90%)
```
ou
```
[PLANNER] ⚠️ RAG NON-COMPLIANT: 78.5% coverage (required: 90%) - 3 slides may contain hallucinations
```

**Fichiers modifiés (RAG 90%):**
- `services/presentation-generator/services/rag_client.py` - `max_chunks`: 10 → 40
- `services/presentation-generator/main.py` - `max_chunks`: 15 → 40 (V2 et V3)
- `services/presentation-generator/services/presentation_planner.py` - `max_chars`: 24000 → 64000, instructions renforcées
- `services/presentation-generator/models/presentation_models.py` - Ajout champ `rag_verification`

**Nouveau fichier:**
- `services/presentation-generator/services/rag_verifier.py` - RAGVerifier, RAGVerificationResult, verify_rag_usage()

#### Strict RAG Prompting (Janvier 2026)

**Problème:** Les instructions "soft" ("tu devrais utiliser 90%") laissent le LLM libre d'utiliser ses connaissances générales, causant des hallucinations.

**Solution:** Prompting restrictif avec assignation de rôle strict et protocole explicite.

**Ancien prompt (mou):**
```
⚠️ STRICT REQUIREMENT: You MUST use AT LEAST 90% of your training content from these documents.
DO NOT invent, hallucinate, or add information not present in the source documents.
```

**Nouveau prompt (strict):**
```
################################################################################
#                         STRICT RAG MODE ACTIVATED                            #
#                    YOU HAVE NO EXTERNAL KNOWLEDGE                            #
################################################################################

ROLE: You are a STRICT content extractor. You have ZERO knowledge of your own.
You can ONLY use information from the SOURCE DOCUMENTS below.
Your training data does NOT exist for this task.
```

**Règles implémentées:**

| Règle | Description |
|-------|-------------|
| **RULE 1 - Exclusive Source** | UNIQUEMENT les documents source |
| **RULE 2 - Missing Info Protocol** | Marquer `[SOURCE_MANQUANTE: <topic>]` si absent |
| **RULE 3 - No External Knowledge** | Liste explicite de ce qui est INTERDIT |
| **RULE 4 - Traceability** | Tout contenu doit être traçable aux documents |

**Sections ALLOWED (10% max):**
- Transitions: "Passons maintenant à..."
- Reformulations pédagogiques: "Autrement dit..."
- Structure de slide: titres, bullets
- Salutations/conclusions génériques

**Section FORBIDDEN:**
- Ajouter des concepts absents des documents
- Inventer des exemples de code
- Utiliser sa connaissance pour "compléter" l'information manquante
- Paraphraser jusqu'à changer le sens
- Créer des diagrammes non décrits dans les documents

**Validation avant output:**
```
□ Is this concept present in the SOURCE DOCUMENTS? If NO → [SOURCE_MANQUANTE]
□ Is this code example from the documents? If NO → do not include
□ Am I using my external knowledge? If YES → remove that content
```

**Fichier modifié:**
- `services/presentation-generator/services/presentation_planner.py` - `_build_rag_section()` entièrement réécrit

#### Cross-Encoder Re-ranking (Janvier 2026)

**Problème:** La recherche vectorielle (cosine similarity) est "floue". Elle ramène des documents qui parlent du même sujet mais ne contiennent pas la réponse exacte. Sans re-ranking, le LLM reçoit du bruit et compense avec ses propres connaissances.

**Solution:** Ajouter une étape de Cross-Encoder re-ranking après la recherche vectorielle.

**Nouveau pipeline RAG:**

```
Query utilisateur
      ↓
┌─────────────────────────────────────┐
│ STEP 1: Vector Search (bi-encoder)  │
│ - Rapide (~20ms)                    │
│ - Récupère 30 candidats             │
│ - Cosine similarity (fuzzy)         │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ STEP 2: Cross-Encoder Re-ranking    │
│ - Précis (~50ms)                    │
│ - Query + Document ensemble         │
│ - Filtre le bruit sémantiquement    │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ STEP 3: Return Top-K                │
│ - Chunks les plus pertinents        │
│ - Moins de bruit pour le LLM        │
└─────────────────────────────────────┘
```

**Pourquoi Cross-Encoder > Bi-Encoder:**

| Aspect | Bi-Encoder (Vector) | Cross-Encoder (Re-rank) |
|--------|---------------------|-------------------------|
| **Input** | Query et Doc séparés | Query + Doc ensemble |
| **Vitesse** | ~20ms pour 1000 docs | ~50ms pour 30 docs |
| **Précision** | Similarité topique | Pertinence exacte |
| **Usage** | Retrieval (recall) | Re-ranking (precision) |

**Exemple concret:**

```
Query: "Quelles sont les mesures de sécurité de Kafka?"

AVANT (sans re-ranking):
1. "Kafka est un système de messaging..." (0.82) ← Hors sujet
2. "L'architecture de Kafka comprend..." (0.80) ← Hors sujet
3. "L'authentification SASL dans Kafka..." (0.78) ← Répond!

APRÈS (avec re-ranking):
1. "L'authentification SASL dans Kafka..." (0.91) ← Répond!
2. "Le chiffrement TLS pour Kafka..." (0.87) ← Répond!
3. "Kafka est un système de messaging..." (0.32) ← Filtré
```

**Configuration:** `RERANKER_BACKEND=auto|cross-encoder|cross-encoder-accurate|tfidf`

| Backend | Modèle | Latence | Qualité |
|---------|--------|---------|---------|
| `auto` | MiniLM → TF-IDF fallback | ~50ms | ⭐⭐⭐ |
| `cross-encoder` | ms-marco-MiniLM-L-6-v2 | ~50ms | ⭐⭐⭐ |
| `cross-encoder-accurate` | ms-marco-MiniLM-L-12-v2 | ~100ms | ⭐⭐⭐⭐ |
| `tfidf` | TF-IDF keywords | ~5ms | ⭐⭐ |

**Fichiers créés/modifiés:**

- `services/course-generator/services/reranker.py` - **NOUVEAU** - CrossEncoderReranker, TFIDFReranker, RerankerFactory
- `services/course-generator/services/retrieval_service.py` - Intégration reranker dans `query()` et `_rerank_results()`
- `services/course-generator/models/document_models.py` - Ajout `rerank_score` à RAGChunkResult

**Logs serveur:**
```
[RAG] Vector search returned 30 candidates
[RAG] Re-ranking 30 results with CrossEncoder...
[RAG] Re-ranking complete. Top score: 0.912 -> 0.234
[RAG] Returning 15 re-ranked chunks
```

#### RAG Threshold Validation (Janvier 2026)

**Problème:** Quand le RAG ne retourne rien ou peu de contenu, le graphe continue la génération standard. Le LLM compense avec ses propres connaissances, causant des hallucinations.

**Solution:** Ajouter une branche conditionnelle basée sur le nombre de tokens RAG.

**Nouveau flux conditionnel:**

```
                    pedagogical_analysis
                           ↓
                  check_rag_threshold
                    /            \
          (blocked)/              \(ok)
                  ↓                ↓
         insufficient_rag     plan_course
                  ↓                ↓
                END           génération...
```

**Seuils configurables:**

| Tokens RAG | Mode | Action |
|------------|------|--------|
| < 500 | `BLOCKED` | **Arrêt** - Erreur retournée, demande plus de documents |
| 500-2000 | `PARTIAL` | **Warning** - Génération continue avec avertissement |
| > 2000 | `FULL` | **OK** - Mode RAG optimal, 90%+ couverture attendue |
| 0 (pas de docs) | `NONE` | **Standard** - Génération IA pure avec warning |

**Configuration:** Variables d'environnement

```bash
RAG_MINIMUM_TOKENS=500   # Seuil de blocage (hard)
RAG_QUALITY_TOKENS=2000  # Seuil de qualité (warning)
```

**Réponse API enrichie:**

```json
{
  "script": {
    "rag_verification": {
      "mode": "partial",
      "token_count": 1200,
      "coverage": 0.85,
      "is_compliant": false,
      "warning": "Limited source content: 1200 tokens (recommended: 2000+)"
    }
  }
}
```

**Messages d'erreur (mode BLOCKED):**

```
Cannot generate content for 'Apache Kafka Security': Insufficient source
content: 320 tokens retrieved (minimum required: 500). Please provide more
comprehensive documents or check that the documents cover the requested topic.
```

**Suggestions retournées à l'utilisateur:**

- Upload more documents covering the topic
- Ensure documents contain text (not just images)
- Check that documents are relevant to the requested topic
- Consider using PDFs or Word documents with more content

**Fichiers créés:**

- `services/course-generator/services/rag_threshold_validator.py`
- `services/presentation-generator/services/rag_threshold_validator.py`

**Fichiers modifiés:**

- `agents/course_graph.py` - Nouveau node `check_rag_threshold` + `handle_insufficient_rag` + routing conditionnel
- `services/presentation_planner.py` - Vérification threshold avant `generate_script()`

**Logs serveur:**

```
[RAG_CHECK] FULL mode: 4500 tokens (sufficient)
```
ou
```
[RAG_CHECK] BLOCKED: 320 tokens (insufficient)
[INSUFFICIENT_RAG] Generation blocked. Tokens: 320
[INSUFFICIENT_RAG] User should provide more comprehensive documents.
```

---

## Phase 3 - Implémentation (COMPLÉTÉE)

### Backend (media-generator)

**Nouveaux fichiers créés:**
- `models/video_editor_models.py` - Modèles Pydantic complets
  - SegmentType, TransitionType, SegmentStatus, ProjectStatus (Enums)
  - AudioTrack: piste audio avec volume, fade in/out
  - VideoSegment: segment timeline avec trim, transitions, opacité
  - TextOverlay, ImageOverlay: overlays configurables
  - VideoProject: projet complet avec segments et settings
  - Request/Response models pour l'API
- `services/timeline_service.py` - Gestion timeline
  - ProjectRepository: stockage in-memory (PostgreSQL en prod)
  - TimelineService: CRUD projets, segments, overlays
  - _recalculate_timeline: mise à jour automatique des temps
- `services/segment_manager.py` - Gestion des médias
  - process_upload: traitement vidéo/audio/image
  - _get_video_duration, _generate_thumbnail via ffprobe/ffmpeg
  - trim_video, extract_audio: utilitaires FFmpeg
  - Validation formats et tailles
- `services/video_merge_service.py` - Rendu final FFmpeg
  - render_project: orchestration du rendu complet
  - _prepare_segment: normalisation résolution/fps
  - _image_to_video: conversion slides en vidéo
  - _concatenate_segments: fusion avec transitions
  - _finalize_video: overlays, musique, encodage final
  - QUALITY_PRESETS: low/medium/high

**Fichiers modifiés:**
- `main.py` - +400 lignes pour les endpoints Video Editor API

### Frontend (editor)

**Nouveaux fichiers créés:**
- `lib/editor-types.ts` - Types TypeScript
  - Enums: SegmentType, TransitionType, SegmentStatus, ProjectStatus
  - Interfaces: VideoSegment, VideoProject, TextOverlay, ImageOverlay
  - Request/Response types pour l'API
  - Helper functions: formatDuration, getSegmentTypeLabel, etc.
- `hooks/useVideoEditor.ts` - Hook React complet
  - State: project, isLoading, isSaving, isRendering, renderJob
  - Project actions: create, load, update, delete
  - Segment actions: add, upload, update, remove, reorder, split
  - Overlay actions: addTextOverlay, addImageOverlay
  - Render actions: startRender, checkRenderStatus, createPreview
- `components/SegmentItem.tsx` - Composant segment
  - Affichage thumbnail, durée, type
  - Toggle mute audio
  - Drag & drop pour réordonnancement
  - Menu contextuel (edit, mute, remove)
  - Handles de trim visuels
- `components/Timeline.tsx` - Timeline principale
  - Règle temporelle avec markers
  - Playhead interactif
  - Drop zones pour réordonnancement
  - Zoom in/out
  - Drag & drop upload
- `components/SegmentProperties.tsx` - Panel propriétés
  - Trim start/end
  - Volume audio + mute toggle
  - Opacité
  - Transitions in/out avec durée
  - Split segment
- `page.tsx` - Page éditeur complète
  - Modal création projet
  - Preview vidéo
  - Contrôles playback (play/pause/stop)
  - Upload média
  - Export/render avec progression
  - Gestion erreurs

### Accès à l'éditeur
- URL: `/dashboard/studio/editor`
- Paramètres: `?projectId=xxx` ou `?courseJobId=xxx` pour import

---

## Phase 4 - Implémentation (COMPLÉTÉE)

### Backend (media-generator)

**Nouveaux fichiers créés:**
- `models/voice_cloning_models.py` - Modèles Pydantic complets
  - VoiceProvider, SampleStatus, VoiceProfileStatus (Enums)
  - VoiceSample: échantillon audio avec quality metrics
  - VoiceProfile: profil vocal complet avec consent tracking
  - VoiceGenerationSettings: paramètres de génération
  - Request/Response models pour l'API
- `services/voice_sample_service.py` - Gestion échantillons
  - process_sample: upload et validation audio
  - _get_audio_duration via ffprobe
  - _analyze_quality: score qualité, bruit, clarté
  - convert_to_standard_format: normalisation audio
  - VoiceSampleRequirements: specs pour uploads
- `services/voice_cloning_service.py` - Intégration ElevenLabs
  - create_cloned_voice: création voix via API
  - generate_speech: génération TTS
  - delete_voice: suppression voix
  - get_usage_stats: stats utilisation API
- `services/voice_profile_manager.py` - Orchestration workflow
  - VoiceProfileRepository: stockage in-memory
  - create_profile, get_profile, list_profiles, delete_profile
  - add_sample, remove_sample
  - start_training avec vérification consentement
  - generate_speech, preview_voice
  - get_training_requirements

**Fichiers modifiés:**
- `main.py` - +250 lignes pour les endpoints Voice Cloning API

### Frontend (voice-clone)

**Nouveaux fichiers créés:**
- `lib/voice-types.ts` - Types TypeScript
  - Enums: VoiceProvider, SampleStatus, VoiceProfileStatus
  - Interfaces: VoiceSample, VoiceProfile, VoiceGenerationSettings
  - Request/Response types
  - Helper functions: formatDuration, getStatusLabel, etc.
- `hooks/useVoiceClone.ts` - Hook React complet
  - State: profiles, selectedProfile, requirements
  - Profile actions: create, select, update, delete
  - Sample actions: upload, delete
  - Training actions: startTraining, checkTrainingStatus
  - Generation actions: generateSpeech, previewVoice
- `components/VoiceSampleUpload.tsx` - Upload échantillons
  - Drag & drop avec validation format
  - Progress bar durée totale
  - Liste des samples avec quality score
  - Tips d'enregistrement
- `page.tsx` - Page gestion des voix
  - Liste des profils vocaux
  - Création profil (nom, genre, accent)
  - Upload samples avec progress
  - Modal consentement pour training
  - Test voice avec audio player
  - Stats utilisation

### Accès au Voice Cloning
- URL: `/dashboard/studio/voice-clone`

### Variables d'environnement
```
ELEVENLABS_API_KEY=your_api_key_here
```

---

## Conventions de code

### Backend (Python/FastAPI)
- Async/await pour toutes les opérations I/O
- Pydantic pour la validation des données
- Timeout de 120s pour les appels OpenAI
- Logging avec `print(..., flush=True)` pour Docker

### Frontend (Next.js/React)
- `useCallback` pour les handlers passés en props
- `useRef` pour les callbacks dans useEffect
- Composants client marqués `'use client'`

---

## Fichiers clés

### Course Generator
- `services/course-generator/services/course_planner.py` - Génération du curriculum
- `services/course-generator/services/context_questions.py` - Questions contextuelles
- `services/course-generator/models/course_models.py` - Modèles de données

### Frontend Courses
- `frontend/src/app/dashboard/studio/courses/page.tsx` - Page principale
- `frontend/src/app/dashboard/studio/courses/lib/course-types.ts` - Types TypeScript
- `frontend/src/app/dashboard/studio/courses/components/` - Composants UI

---

## Phase 5 - Implémentation (COMPLÉTÉE)

### Migration pgvector

**Fichiers modifiés:**
- `docker-compose.yml` - Image `pgvector/pgvector:pg16`, variable `VECTOR_BACKEND=pgvector`
- `infrastructure/docker/init-pgvector.sql` - Script création table + index HNSW
- `services/course-generator/services/vector_store.py` - Classe `PgVectorStore` avec asyncpg
- `services/course-generator/requirements.txt` - Dépendances asyncpg, pgvector

### Phase 5A: Analytics & Metrics

**Nouveaux fichiers créés:**
- `models/analytics_models.py` - Modèles Pydantic (CourseMetrics, APIUsageMetrics, etc.)
- `services/analytics_service.py` - Service tracking et agrégation

**Endpoints API Analytics:**
- `GET /api/v1/analytics/dashboard` - Dashboard complet
- `GET /api/v1/analytics/user/{user_id}` - Résumé utilisateur
- `GET /api/v1/analytics/api-usage` - Rapport usage API
- `GET /api/v1/analytics/quota/{user_id}` - Quotas utilisateur
- `POST /api/v1/analytics/track` - Track événement

**Frontend:**
- `frontend/src/app/dashboard/studio/analytics/` - Dashboard analytics cours

### Phase 5B: Multi-langue (10 langues)

**Langues supportées:** EN, FR, ES, DE, PT, IT, NL, PL, RU, ZH

**Nouveaux fichiers créés:**
- `models/translation_models.py` - Modèles (SupportedLanguage, CourseTranslation)
- `services/translation_service.py` - Service traduction via GPT-4o-mini

**Endpoints API Translation:**
- `GET /api/v1/translation/languages` - Langues supportées
- `POST /api/v1/translation/translate` - Traduire un texte
- `POST /api/v1/translation/translate-batch` - Traduction batch
- `POST /api/v1/translation/detect` - Détection de langue
- `POST /api/v1/translation/course/{course_id}` - Traduire un cours complet

### Phase 5C: Monétisation (Stripe + PayPal)

**Plans disponibles:**
| Plan | Prix/mois | Cours/mois | Storage | API Budget |
|------|-----------|------------|---------|------------|
| Free | $0 | 3 | 1 GB | $5 |
| Starter | $19 | 10 | 10 GB | $25 |
| Pro | $49 | 50 | 50 GB | $100 |
| Enterprise | $199 | Illimité | 500 GB | $500 |

**Nouveaux fichiers créés:**
- `models/billing_models.py` - Modèles (Subscription, Payment, Invoice)
- `services/billing_service.py` - Intégration Stripe + PayPal

**Endpoints API Billing:**
- `GET /api/v1/billing/plans` - Liste des plans
- `POST /api/v1/billing/checkout` - Créer session checkout
- `GET /api/v1/billing/subscription/{user_id}` - Abonnement actuel
- `POST /api/v1/billing/cancel` - Annuler abonnement
- `POST /api/v1/billing/portal` - Portail de facturation
- `POST /api/v1/billing/webhooks/stripe` - Webhook Stripe
- `POST /api/v1/billing/webhooks/paypal` - Webhook PayPal

**Variables d'environnement:**
```
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...
PAYPAL_CLIENT_ID=...
PAYPAL_CLIENT_SECRET=...
```

### Phase 5D: Collaboration (Équipes)

**Rôles disponibles:**
| Rôle | Permissions |
|------|-------------|
| Owner | Tout (gérer équipe, billing, supprimer workspace) |
| Admin | Gérer membres, créer/éditer/supprimer cours |
| Editor | Créer cours, éditer ses propres cours |
| Viewer | Voir cours seulement |

**Nouveaux fichiers créés:**
- `models/collaboration_models.py` - Modèles (Workspace, TeamMember, CourseShare)
- `services/collaboration_service.py` - Service gestion équipes

**Endpoints API Collaboration:**
- `POST /api/v1/workspaces` - Créer workspace
- `GET /api/v1/workspaces` - Lister workspaces
- `GET /api/v1/workspaces/{id}` - Détails workspace
- `PATCH /api/v1/workspaces/{id}` - Modifier workspace
- `POST /api/v1/workspaces/{id}/invite` - Inviter membre
- `POST /api/v1/workspaces/accept-invitation` - Accepter invitation
- `DELETE /api/v1/workspaces/{id}/members/{member_id}` - Retirer membre
- `PATCH /api/v1/workspaces/{id}/members/{id}/role` - Changer rôle
- `POST /api/v1/workspaces/{id}/leave` - Quitter workspace
- `POST /api/v1/courses/{id}/share` - Partager cours
- `GET /api/v1/courses/{id}/shares` - Liste partages
- `GET /api/v1/workspaces/{id}/activity` - Journal d'activité

---

## Phase 6 - Enhanced Code & Diagram Generation (IMPLÉMENTÉE)

### Objectifs
Améliorer la qualité du code et des diagrammes générés pour atteindre un niveau professionnel/enterprise-grade.

### Améliorations apportées

#### 1. TechPromptBuilder - Prompts contextuels dynamiques
Le nouveau `TechPromptBuilder` génère des prompts adaptés selon:
- **Niveau de l'audience**: Débutant absolu → Expert
- **Domaine tech**: Data Engineering, DevOps, ML, Cybersecurity, etc.
- **Carrière cible**: 545+ métiers IT avec contexte spécifique
- **Langages**: 120+ langages avec exemples de style

#### 2. Standards de qualité code obligatoires
Tous les codes générés respectent:
- Conventions de nommage (descriptif, pas de single letters)
- Structure (max 20 lignes/fonction, max 3 niveaux de nesting)
- Testabilité (fonctions pures, DI, pas d'état global)
- Documentation (docstrings, type hints)
- Gestion d'erreurs (exceptions spécifiques, messages significatifs)
- Design patterns appropriés

#### 3. DiagramsRenderer - Diagrammes professionnels
Intégré dans `diagram_generator.py` comme méthode **PRIMARY**:
- **Icônes officielles** AWS, Azure, GCP, Kubernetes, On-Premise
- **Rendu PNG** haute qualité avec post-traitement
- **Clustering** logique des composants
- **Génération via GPT-4o** pour meilleure précision
- **Détection auto** du cloud provider depuis la description

**Ordre de priorité des renderers:**
1. **PRIMARY** - Python Diagrams (architecture, hierarchy, process)
2. **SECONDARY** - Mermaid via Kroki (flowcharts, sequences, mindmaps)
3. **TERTIARY** - PIL fallback

#### 4. Modèles de données exhaustifs

**tech_domains.py** contient:
```python
class CodeLanguage(str, Enum):
    # 120+ langages incluant:
    PYTHON, JAVASCRIPT, GO, RUST, SOLIDITY, QISKIT, CIRQ...

class TechDomain(str, Enum):
    # 80+ domaines incluant:
    DATA_ENGINEERING, MLOPS, DEVOPS, KUBERNETES, QUANTUM_COMPUTING...

class TechCareer(str, Enum):
    # 545+ métiers IT 360° incluant:
    DATA_LINEAGE_DEVELOPER, MLOPS_ENGINEER, PLATFORM_ENGINEER...
```

#### 5. Visual-Generator Microservice (Architecture)

Le service `visual-generator` est un microservice isolé qui gère la génération de diagrammes:

**Communication:**
```
presentation-generator → HTTP POST → visual-generator:8003/api/v1/diagrams/generate
```

**Endpoints:**
- `POST /api/v1/diagrams/generate` - Génération diagramme depuis description
- `POST /api/v1/diagrams/mermaid` - Rendu Mermaid
- `POST /api/v1/diagrams/detect` - Détection besoin diagramme
- `GET /api/v1/diagrams/{filename}` - Récupération image
- `DELETE /api/v1/diagrams/{filename}` - Suppression
- `POST /api/v1/diagrams/cleanup` - Nettoyage fichiers anciens

#### 6. Complexité des diagrammes par audience

Le système adapte automatiquement la complexité des diagrammes selon l'audience cible:

| Audience | Nodes Max | Caractéristiques |
|----------|-----------|------------------|
| **BEGINNER** | 5-7 | Concepts haut-niveau, pas de détails réseau, labels courts |
| **SENIOR** | 10-15 | VPCs, caching, load balancers, redundancy, Edge labels |
| **EXECUTIVE** | 6-8 | Flux de valeur, termes business, pas de jargon technique |

**Flux complet de propagation de l'audience:**
```
PresentationPlanner (GPT-4 avec instructions DIAGRAM COMPLEXITY BY AUDIENCE)
    ↓ target_audience dans PresentationScript
PresentationCompositor / LangGraphOrchestrator
    ↓ target_audience passé à SlideGenerator
SlideGenerator.generate_slide_image(slide, style, target_audience)
    ↓ target_audience passé à _render_diagram_slide
DiagramGeneratorService.generate_diagram(..., target_audience)
    ↓ Mapping string → TargetAudience enum
DiagramsRenderer.generate_and_render(audience=TargetAudience.SENIOR)
    ↓ HTTP POST avec audience + cheat_sheet
visual-generator:8003/api/v1/diagrams/generate
    ↓ Prompt GPT-4o avec AUDIENCE_INSTRUCTIONS + DIAGRAMS_CHEAT_SHEET
Génération code Python Diagrams → Exécution Graphviz → PNG
```

**DIAGRAMS_CHEAT_SHEET:**
Liste exhaustive des imports valides pour empêcher les hallucinations du LLM:
- Évite les imports inexistants (ex: `from diagrams.aws.compute import NonExistentService`)
- Couvre: AWS, Azure, GCP, Kubernetes, On-Premise, Programming, SaaS, Generic

### Architecture

```
services/presentation-generator/
├── models/
│   └── tech_domains.py           # Enums: CodeLanguage, TechDomain, TechCareer
└── services/
    ├── tech_prompt_builder.py    # Construction prompts contextuels
    ├── presentation_planner.py   # Intégration du prompt builder + DIAGRAM COMPLEXITY
    ├── slide_generator.py        # Paramètre target_audience ajouté
    ├── diagram_generator.py      # Client HTTP vers visual-generator + audience mapping
    ├── presentation_compositor.py # Passage target_audience depuis script
    ├── langgraph_orchestrator.py # Passage target_audience depuis script
    └── agents/
        └── visual_sync_agent.py  # Propagation target_audience

services/visual-generator/         # MICROSERVICE ISOLÉ (Port 8003)
├── main.py                       # FastAPI avec endpoints diagrammes
├── Dockerfile                    # Graphviz + dépendances lourdes
├── models/
│   └── visual_models.py          # DiagramType, DiagramStyle, RenderFormat
└── renderers/
    ├── diagrams_renderer.py      # Python Diagrams + GPT-4o + audience/cheat_sheet
    ├── mermaid_renderer.py       # Mermaid via Kroki (PRIMARY) + mermaid.ink (FALLBACK)
    └── matplotlib_renderer.py    # Charts de données

#### 7. Kroki Self-Hosted (Diagram Rendering)

Le rendu Mermaid utilise **Kroki self-hosted** comme renderer PRIMARY avec fallback vers mermaid.ink:

**Avantages Kroki:**
- Self-hosted = pas de dépendance externe en production
- Privacy: les diagrammes restent dans l'infrastructure
- Supporte 20+ types de diagrammes (Mermaid, PlantUML, D2, GraphViz, etc.)
- Fiabilité: pas de rate limiting externe

**Ordre de priorité:**
1. **PRIMARY**: Kroki self-hosted (`http://kroki:8000`)
2. **FALLBACK**: mermaid.ink public API (si Kroki indisponible)

**Configuration:**
```python
# mermaid_renderer.py
KROKI_URL = os.getenv("KROKI_URL", "http://kroki:8000")
USE_KROKI = os.getenv("USE_KROKI", "true").lower() == "true"
```

**Docker Compose:**
```yaml
kroki:
  image: yuzutech/kroki
  container_name: viralify-kroki
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
```

#### 8. Sécurité - Validation AST du code généré

Le `DiagramsRenderer` exécute du code Python généré par GPT-4o. Pour prévenir les attaques par injection de code, un validateur AST (`CodeSecurityValidator`) analyse le code AVANT exécution:

**Protections implémentées:**

| Type | Éléments bloqués |
|------|------------------|
| **Imports dangereux** | `os`, `subprocess`, `sys`, `socket`, `requests`, `pickle`, etc. |
| **Fonctions dangereuses** | `exec()`, `eval()`, `open()`, `__import__()`, `getattr()` |
| **Attributs dunder** | `__class__`, `__bases__`, `__subclasses__`, `__globals__` |
| **Imports autorisés** | UNIQUEMENT `diagrams.*` |

**Flux de validation:**
```
Code généré par GPT-4o
    ↓
CodeSecurityValidator.validate(code)
    ↓ AST parsing
    ↓ Import whitelist check
    ↓ Blocked functions check
    ↓ Dangerous attributes check
    ↓
is_safe = True → Exécution autorisée
is_safe = False → REJET + log sécurité
```

**Exemple d'attaque bloquée:**
```python
# GPT pourrait générer (accidentellement ou par prompt injection):
import os
os.system("rm -rf /")  # ❌ BLOQUÉ: "SECURITY: Blocked import 'os'"

# ou
exec("malicious_code")  # ❌ BLOQUÉ: "SECURITY: Blocked function 'exec()'"
```

### Fichiers créés/modifiés

**Nouveaux fichiers:**
- `models/tech_domains.py` - 545+ métiers, 80+ domaines, 120+ langages
- `services/tech_prompt_builder.py` - Construction de prompts dynamiques
- `services/visual-generator/main.py` - Microservice FastAPI
- `services/visual-generator/Dockerfile` - Image avec Graphviz

**Fichiers modifiés (Phase 6 + audience + career):**
- `services/diagram_generator.py` - Client HTTP + DIAGRAMS_CHEAT_SHEET + audience mapping + **career focus**
- `services/presentation_planner.py` - DIAGRAM COMPLEXITY BY AUDIENCE instructions
- `services/slide_generator.py` - Paramètres `target_audience` + `target_career`
- `services/presentation_compositor.py` - Passage `target_audience` + `target_career`
- `services/langgraph_orchestrator.py` - Passage `target_audience` + `target_career`
- `services/agents/visual_sync_agent.py` - Propagation `target_audience` + `target_career`
- `renderers/diagrams_renderer.py` - Audience instructions + cheat_sheet + **CodeSecurityValidator** (AST validation)
- `models/presentation_models.py` - Champ `target_career` ajouté à `PresentationScript` et `GeneratePresentationRequest`
- `models/tech_domains.py` - `DiagramFocus` enum, `CAREER_DIAGRAM_FOCUS_MAP`, `DIAGRAM_FOCUS_INSTRUCTIONS`
- `renderers/mermaid_renderer.py` - Kroki self-hosted (PRIMARY) + mermaid.ink (FALLBACK)
- `docker-compose.prod.yml` - Services visual-generator + kroki ajoutés
- `docker-compose.yml` - Service kroki ajouté

#### 9. DiagramFocus - Différenciation par carrière

Le système adapte la **perspective** des diagrammes selon la carrière cible (545+ métiers):

**Principe:**
- Un Data Engineer et un Cloud Architect regardent le même système différemment
- Le Data Engineer veut voir pipelines, ETL, data lakes
- L'Architect veut voir VPCs, load balancers, zones de disponibilité

**DiagramFocus Enum:**

| Focus | Exemple de carrières | Ce qui est montré |
|-------|---------------------|-------------------|
| **CODE** | Software Developer, Backend Engineer | APIs, services, microservices, queues |
| **INFRASTRUCTURE** | Cloud Architect, Platform Engineer | VPCs, load balancers, scaling, HA |
| **DATA** | Data Engineer, Analytics Engineer | Pipelines, ETL/ELT, warehouses, lakes |
| **ML_PIPELINE** | ML Engineer, MLOps Engineer | Feature stores, model serving, training |
| **SECURITY** | Security Engineer, CISO | Zones, firewalls, IAM, encryption |
| **NETWORK** | Network Engineer, SRE | Topology, routing, DNS, CDN |
| **DATABASE** | DBA, Database Architect | Replication, sharding, HA, backups |
| **BUSINESS** | CTO, VP Engineering | Value flow, ROI, high-level view |
| **QA_TESTING** | QA Engineer, Test Lead | Test pyramid, CI/CD, environments |
| **EMBEDDED** | Firmware Engineer, IoT Developer | Hardware, sensors, protocols |

**API Usage:**
```python
# GeneratePresentationRequest
{
    "topic": "Building a data pipeline with Apache Kafka",
    "target_audience": "senior developers",
    "target_career": "data_engineer"  # NEW: différencie les diagrammes
}
```

**Flux de propagation:**
```
GeneratePresentationRequest.target_career
    ↓ Stocké dans PresentationScript.target_career
PresentationCompositor / LangGraphOrchestrator
    ↓ target_career passé à SlideGenerator
SlideGenerator.generate_slide_image(slide, style, target_audience, target_career)
    ↓ target_career passé à _render_diagram_slide
DiagramGeneratorService.generate_diagram(..., target_career)
    ↓ Parsing TechCareer enum + get_diagram_instructions_for_career()
DiagramsRenderer._build_enhanced_description(career=TechCareer.DATA_ENGINEER)
    ↓ Instructions spécifiques ajoutées au prompt GPT-4o
```

### Dépendances

```
# visual-generator/requirements.txt
diagrams>=0.23.4
fastapi
uvicorn
openai
pillow

# System requirement (dans Dockerfile visual-generator)
# Graphviz doit être installé:
# Ubuntu: apt-get install graphviz graphviz-dev
# macOS: brew install graphviz
# Windows: choco install graphviz
```

### Variables d'environnement

```
# presentation-generator
VISUAL_GENERATOR_URL=http://visual-generator:8003

# visual-generator
OPENAI_API_KEY=sk-...
OUTPUT_DIR=/tmp/viralify/diagrams
KROKI_URL=http://kroki:8000
USE_KROKI=true
```

---

## Phase 7 - Source Traceability System (IMPLÉMENTÉE)

### Objectifs
Système complet de traçabilité des sources utilisées pour la génération de cours, permettant de savoir exactement d'où vient chaque information.

### Phase 7.1: PedagogicalRole & Traceability

#### PedagogicalRole Enum
Chaque source uploadée peut avoir un rôle pédagogique:

| Rôle | Icon | Description |
|------|------|-------------|
| **THEORY** | 📚 | Définitions, concepts, explications théoriques |
| **EXAMPLE** | 💡 | Exemples pratiques, démos, tutoriels |
| **REFERENCE** | 📖 | Documentation officielle, spécifications |
| **OPINION** | 💭 | Notes personnelles, perspectives |
| **DATA** | 📊 | Statistiques, études, recherche |
| **CONTEXT** | 🔍 | Informations de fond, historique, prérequis |
| **AUTO** | 🤖 | L'IA détermine automatiquement le rôle |

#### Citation Configuration
```python
class SourceCitationConfig:
    enable_vocal_citations: bool = False  # Citations vocales dans le voiceover
    citation_style: CitationStyle = NATURAL  # NATURAL, ACADEMIC, MINIMAL, NONE
    show_traceability_panel: bool = True   # Panel de traçabilité visible
    include_page_numbers: bool = True
    include_timestamps: bool = True
    include_quote_excerpts: bool = True
```

#### Endpoints API Traceability
- `GET /api/v1/traceability/citation-styles` - Styles de citation disponibles
- `GET /api/v1/traceability/pedagogical-roles` - Rôles pédagogiques
- `GET /api/v1/traceability/default-config` - Configuration par défaut
- `GET /api/v1/courses/{job_id}/traceability` - Traçabilité complète d'un cours
- `GET /api/v1/courses/{job_id}/lectures/{lecture_id}/traceability` - Traçabilité d'une lecture
- `PATCH /api/v1/sources/{source_id}/pedagogical-role` - Modifier le rôle d'une source

### Phase 7.2: Coherence Check

Validation de la cohérence pédagogique entre les lectures.

#### Fonctionnalités
- **Détection des prérequis manquants**: Si une lecture utilise un concept introduit plus tard
- **Détection des gaps conceptuels**: Trop de nouveaux prérequis d'un coup
- **Score de cohérence**: 0-100, avec seuil de 50 pour warning
- **Enrichissement**: Ajoute `key_concepts`, `prerequisites`, `introduces`, `prepares_for` à chaque lecture

#### Intégration Pipeline
```
run_planning → check_coherence → build_knowledge_graph → iterate_lectures
```

#### Fichiers
- `services/coherence_service.py` - CoherenceCheckService
- `models/course_models.py` - Champs Lecture enrichis

### Phase 7.3: Knowledge Graph & Cross-Reference

Construction d'un graphe de connaissances et analyse des références croisées.

#### Knowledge Graph
Extraction de concepts depuis les sources avec relations:

```python
@dataclass
class Concept:
    name: str
    canonical_name: str
    aliases: List[str]
    definitions: List[ConceptDefinition]  # Une définition par source
    consolidated_definition: str  # Synthèse de toutes les sources
    prerequisites: List[str]      # Concepts prérequis
    related_concepts: List[str]
    parent_concepts: List[str]    # Concepts plus larges
    child_concepts: List[str]     # Concepts plus spécifiques
    complexity_level: int         # 1-5
    frequency: int                # Nombre de mentions
```

#### Cross-Reference Analysis
Analyse comment les sources se complètent:

```python
@dataclass
class TopicCrossReference:
    topic: str
    source_contributions: List[SourceContribution]
    consolidated_definition: str
    consolidated_examples: List[str]
    points_of_agreement: List[str]
    points_of_disagreement: List[str]
    coverage_score: float  # 0-1
    missing_aspects: List[str]  # theory, examples, reference, data
```

#### Endpoints API Knowledge Graph
- `GET /api/v1/courses/{job_id}/knowledge-graph` - Graphe de connaissances complet
- `GET /api/v1/courses/{job_id}/knowledge-graph/concepts` - Liste des concepts (paginé)
- `GET /api/v1/courses/{job_id}/knowledge-graph/concept/{concept_id}` - Détails d'un concept
- `GET /api/v1/courses/{job_id}/cross-references` - Analyse des références croisées
- `GET /api/v1/courses/{job_id}/cross-references/topic/{topic_name}` - Cross-ref pour un topic
- `POST /api/v1/sources/analyze-cross-references` - Analyser sources indépendamment d'un cours

### Architecture

```
services/course-generator/
├── models/
│   ├── traceability_models.py    # SourceCitationConfig, ContentReference, etc.
│   └── course_models.py          # Champs traceability, knowledge_graph ajoutés
├── services/
│   ├── traceability_service.py   # Génération de citations, références
│   ├── coherence_service.py      # Validation cohérence pédagogique
│   ├── knowledge_graph.py        # KnowledgeGraphBuilder
│   ├── cross_reference_service.py # CrossReferenceService
│   └── source_library.py         # Organisation par PedagogicalRole
└── agents/
    ├── orchestrator_graph.py     # Nodes: check_coherence, build_knowledge_graph
    └── state.py                  # Champs coherence + knowledge_graph
```

### Pipeline Complet

```
validate_input
    ↓
run_planning (curriculum)
    ↓
check_coherence (Phase 7.2)
    ↓ Validate prerequisites, detect concept gaps, enrich lectures
build_knowledge_graph (Phase 7.3)
    ↓ Extract concepts, build relationships, analyze cross-references
iterate_lectures (production)
    ↓
package_output
    ↓
finalize
```

### Commits

- `84039db` - feat(traceability): implement Phase 1 - Source Traceability System
- `769cdf1` - feat(coherence): implement Phase 2 - Pedagogical Coherence Check
- `87301fe` - feat(knowledge-graph): implement Phase 3 - Knowledge Graph & Cross-Reference
- `03b9385` - feat: add frontend traceability UI for source citations

---

## Title Style System (IMPLÉMENTÉ)

### Objectif
Éviter les titres de slides robotiques ("Introduction à X", "Conclusion") et générer des titres qui sonnent humains, tout en supportant différents styles selon le contexte d'utilisation.

### Styles disponibles

| Style | Description | Use Case |
|-------|-------------|----------|
| `corporate` | Professionnel et formel | Formation entreprise |
| `engaging` | Dynamique, accrocheur | Créateurs de contenu |
| `expert` | Précision technique | Audiences avancées |
| `mentor` | Chaleureux, pédagogique | Plateformes éducatives |
| `storyteller` | Narratif | Tutoriels, études de cas |
| `direct` | Clair, concis | Documentation |

### Anti-patterns détectés

Le système détecte et signale automatiquement les patterns robotiques:
- `introduction`: "Introduction à X", "Présentation de X"
- `conclusion`: "Conclusion", "Résumé", "Recap"
- `numbered`: "Partie 1", "Step 2:", "Section 3"
- `placeholder`: "Slide 1", "Title", "Untitled"
- `generic`: "What is X?", "Overview of X", "Basics of X"

### Architecture

```
services/presentation-generator/
├── models/
│   └── presentation_models.py      # TitleStyle enum ajouté
└── services/
    ├── title_style_system.py       # TitleStyleSystem, validation, prompts
    └── presentation_planner.py     # Intégration du système
```

### Frontend

```
frontend/src/app/dashboard/studio/courses/
├── lib/
│   └── course-types.ts             # TitleStyle type, TITLE_STYLE_INFO
├── components/
│   └── CourseForm.tsx              # Sélecteur de style dans Advanced
└── page.tsx                        # Default: 'engaging'
```

### Utilisation

1. Le frontend envoie `title_style` dans la requête
2. Le planner ajoute les guidelines de style au prompt GPT
3. Après génération, validation des titres avec logging des issues
4. Les anti-patterns sont signalés mais ne bloquent pas (monitoring)

### Commit

- `e7a71f1` - feat: implement Title Style System for human-quality slide titles

---

## SSVS - Synchronisation Audio-Vidéo Sémantique

### SSVS Algorithms (Implémenté)

Le système SSVS (Semantic Slide-Voiceover Synchronization) aligne précisément l'audio et la vidéo en utilisant l'analyse sémantique plutôt qu'une distribution proportionnelle simple.

**Architecture:**
```
services/presentation-generator/services/sync/
├── __init__.py                    # Exports publics
├── ssvs_algorithm.py              # SSVSSynchronizer principal
├── ssvs_calibrator.py             # Calibration multi-niveau
└── diagram_synchronizer.py        # Extension pour diagrammes
```

**Composants principaux:**

| Classe | Rôle |
|--------|------|
| `SSVSSynchronizer` | Alignement sémantique slides ↔ voiceover |
| `SSVSCalibrator` | Correction des 5 sources de désynchronisation |
| `DiagramAwareSynchronizer` | Focus sur éléments de diagramme |
| `SemanticEmbeddingEngine` | Embeddings TF-IDF ou Sentence-BERT |
| `FocusAnimationGenerator` | Génération keyframes d'animation |

### SSVS Calibrator (Janvier 2026)

Le calibrateur corrige **5 sources de désynchronisation audio-vidéo:**

| Source | Offset par défaut | Description |
|--------|-------------------|-------------|
| **Global offset** | -300ms | Décalage général voix/image |
| **STT latency** | -50ms | Latence de transcription |
| **Semantic anticipation** | -150ms | Anticiper le slide avant le mot |
| **Transition duration** | 200ms | Temps de transition visuelle |
| **Visual inertia** | Variable | L'œil a besoin de temps pour se fixer |

**Presets disponibles:**

| Preset | Usage | Caractéristiques |
|--------|-------|------------------|
| `default` | Standard | Offsets moyens |
| `fast_speech` | Speakers rapides | Anticipation réduite |
| `slow_speech` | Speakers lents | Plus de temps par slide |
| `technical_content` | Code, diagrammes | Slides plus longues |
| `simple_slides` | Texte simple | Transitions rapides |
| `live_presentation` | Style conférence | Dynamique |
| `training_course` | **Formation Viralify** | Offset -400ms, anticipation -200ms |

**Fichiers:**
- `services/sync/ssvs_calibrator.py` - CalibrationConfig, SSVSCalibrator, CalibrationPresets
- `services/timeline_builder.py` - Intégration avec `calibration_preset` parameter

**Usage:**
```python
builder = TimelineBuilder(
    sync_method=SyncMethod.SSVS,
    calibration_preset="training_course"  # Preset optimisé pour Viralify
)
```

### SSVS Embedding Engines (Janvier 2026)

Backends d'embedding configurables pour la synchronisation sémantique.

**Configuration:** `SSVS_EMBEDDING_BACKEND=auto|minilm|bge-m3|tfidf`

| Backend | Modèle | Dimensions | Taille | Qualité | Multilangue |
|---------|--------|------------|--------|---------|-------------|
| **minilm** | all-MiniLM-L6-v2 | 384 | ~80MB | ⭐⭐⭐ | 🟡 |
| **bge-m3** | BAAI/bge-m3 | 1024 | ~2GB | ⭐⭐⭐⭐ | ✅ 100+ |
| **tfidf** | TF-IDF (local) | Variable | 0 | ⭐⭐ | 🟡 |
| **auto** | MiniLM → TF-IDF | - | - | Adaptatif | - |

**Ordre de priorité (mode auto):**
1. MiniLM (si sentence-transformers installé)
2. TF-IDF (fallback sans dépendances)

**Architecture:**
```
services/sync/
├── embedding_engine.py        # NOUVEAU - Factory + backends
│   ├── EmbeddingEngineBase (ABC)
│   ├── TFIDFEmbeddingEngine
│   ├── SentenceTransformerEngine (MiniLM/BGE-M3)
│   └── EmbeddingEngineFactory
├── ssvs_algorithm.py          # SSVSSynchronizer (modifié)
└── __init__.py                # Exports mis à jour
```

**Dépendances (optionnelles):**
```
# requirements.txt
sentence-transformers>=2.2.0  # Pour MiniLM et BGE-M3
torch>=2.0.0                  # CPU version
```

**Usage:**
```python
# Via SSVSSynchronizer
sync = SSVSSynchronizer(embedding_backend="minilm")

# Via factory directe
from services.sync import EmbeddingEngineFactory
engine = EmbeddingEngineFactory.create("auto")
```

### SSVS Sync Anchors (Janvier 2026)

Les **sync anchors** sont des contraintes dures qui forcent l'alignement à des points précis.

**Format dans le voiceover:**
```
[SYNC:SLIDE_2] Passons maintenant au deuxième sujet...
```

**Comportement:**
- L'anchor `[SYNC:SLIDE_2]` FORCE le slide 2 à commencer exactement à ce mot
- L'algorithme partitionne le problème aux points d'anchor
- Chaque partition est résolue indépendamment par DP

**Flux d'implémentation:**
```
[SYNC:SLIDE_2] dans voiceover
       ↓
_find_sync_anchors() extrait les anchors (timeline_builder.py)
       ↓
_convert_to_ssvs_anchors() convertit en SSVSSyncAnchor
       ↓
synchronize_with_anchors() utilise comme CONTRAINTE DURE
       ↓
_create_anchor_partitions() divise le problème DP
       ↓
Résultat: slide aligné exactement à l'anchor
```

**Dataclass SyncAnchor:**
```python
@dataclass
class SyncAnchor:
    slide_index: int      # Slide concerné
    timestamp: float      # Timestamp cible (secondes)
    segment_index: int    # Segment vocal correspondant
    anchor_type: str      # SLIDE, CODE, DIAGRAM
    anchor_id: str        # "SLIDE_2"
    tolerance_ms: float   # Tolérance (défaut: 500ms)
```

**Résultat avec anchor_used:**
```python
@dataclass
class SynchronizationResult:
    # ... existing fields ...
    anchor_used: Optional[SyncAnchor] = None  # Si ancré
```

**Test de validation:**
```
Sans anchors:  Slide 2: 5.5s - 9.0s
Avec anchor:   Slide 2: 5.5s - 9.0s [ANCHORED] ← Contrainte respectée
```

**Fichiers modifiés:**
- `services/sync/ssvs_algorithm.py` - SyncAnchor, synchronize_with_anchors()
- `services/timeline_builder.py` - _convert_to_ssvs_anchors(), _find_segment_for_timestamp()
- `services/sync/__init__.py` - Export SyncAnchor

### SSVS Partition Fix (Janvier 2026)

**Problème:** Quand plusieurs anchors mappaient vers le même segment audio, des slides étaient ignorés dans la timeline, causant une désynchronisation audio-vidéo.

**Symptôme observé dans les logs:**
```
[TIMELINE] Anchor slide_001: slide 0 -> segment 0 @ 0.00s
[TIMELINE] Anchor slide_002: slide 1 -> segment 0 @ 24.18s   <-- MÊME segment!
[SSVS] Created 6 partitions from anchors  <-- Devrait être 8!
[TIMELINE] SSVS Slide 1: 0.000s - 23.368s
[TIMELINE] SSVS Slide 2: 23.378s - 59.930s
[TIMELINE] SSVS Slide 4: 59.940s - 73.288s   <-- Slide 3 MANQUANT!
```

**Cause racine:** Dans `_create_anchor_partitions()`, la condition `end_seg > start_seg` échouait quand `end_seg == start_seg` (plusieurs slides mappés au même segment), causant le skip de la partition.

**Fix appliqué dans `ssvs_algorithm.py`:**
1. **Tri des boundaries** par slide_index pour ordre correct
2. **Suppression des duplicates** de slide_index (garder le premier)
3. **Gestion edge case** où `end_seg <= start_seg` en ajustant les bornes de partition
4. **Vérification de couverture** avec fallback pour slides manquants
5. **Logging amélioré** pour détecter les problèmes

**Code ajouté:**
```python
# Sort boundaries by slide_index to ensure proper ordering
boundaries.sort(key=lambda b: (b[0], b[1]))

# Remove duplicate slide indices (keep the first one)
seen_slides = set()
unique_boundaries = []
for slide_idx, seg_idx, anchor in boundaries:
    if slide_idx not in seen_slides:
        unique_boundaries.append((slide_idx, seg_idx, anchor))
        seen_slides.add(slide_idx)

# Handle case where multiple slides map to same segment
if n_segments_in_partition <= 0:
    n_segments_in_partition = min(n_slides_in_partition, n_segments - start_seg)
    # ... adjusted partition creation
```

**Fichier modifié:**
- `services/presentation-generator/services/sync/ssvs_algorithm.py` - `_create_anchor_partitions()` réécrite

---

## Direct Sync - Option B+ (Janvier 2026)

### Problème résolu

SSVS (post-hoc matching) avait des limitations:
- Dépendance Whisper pour les timestamps (±200ms d'erreur)
- Drift cumulatif malgré la calibration
- Matching sémantique parfois inexact
- Transitions de slides ne respectant pas les phrases de transition

### Solution: TTS par slide + Crossfade

**Principe:** Synchronisation PARFAITE par construction.

```
AVANT (complexe, fragile):
─────────────────────────────
Planner → Script complet → TTS (1 appel) → Whisper → SSVS → Calibration → Timeline
                                              ↓
                                    Source d'erreurs multiples

APRÈS (simple, robuste):
────────────────────────────
Planner → Scripts par slide → TTS (N appels //) → Concat + Crossfade → Timeline directe
                                                          ↓
                                              Synchronisation PARFAITE
```

### Nouveaux composants

**1. SlideAudioGenerator** (`services/slide_audio_generator.py`)
- Génère TTS pour chaque slide en parallèle (async)
- Rate limiting avec semaphore (max 5 concurrent)
- Cache des audios générés
- Retourne `SlideAudioBatch` avec durées exactes

**2. AudioConcatenator** (`services/audio_concatenator.py`)
- Concatène les audios avec crossfade FFmpeg (100ms)
- Élimine les micro-pauses entre slides
- Normalise le volume
- Timeline ajustée pour le crossfade

**3. DirectTimelineBuilder** (`services/direct_timeline_builder.py`)
- Construit la timeline à partir des durées audio
- Pas de SSVS, pas de matching sémantique
- Sync quality: PARFAITE (by construction)

### Configuration

```env
# Activer Direct Sync (recommandé, défaut: true)
USE_DIRECT_SYNC=true

# Fallback vers SSVS si désactivé
USE_DIRECT_SYNC=false
```

### Avantages

| Critère | SSVS (avant) | Direct Sync (après) |
|---------|--------------|---------------------|
| Synchronisation | ±200ms | **PARFAITE** |
| Dépendance Whisper | Oui | **Non** |
| Calibration nécessaire | Oui | **Non** |
| Debug facile | Non | **Oui** |
| Transitions naturelles | Variable | **Avec crossfade** |

### Fichiers créés/modifiés

**Nouveaux fichiers:**
- `services/presentation-generator/services/slide_audio_generator.py`
- `services/presentation-generator/services/audio_concatenator.py`
- `services/presentation-generator/services/direct_timeline_builder.py`

**Fichiers modifiés:**
- `services/presentation-generator/services/presentation_compositor.py`
  - Ajout de `_generate_voiceover_direct()`
  - Ajout de `_compose_video_with_direct_timeline()`
  - Flag `use_direct_sync` pour basculer entre modes

---

## Phase 7B - Multi-Provider LLM & Training Data (Janvier 2026)

### Multi-Provider LLM Support

Support de 8 providers LLM avec configuration unifiée.

**Providers disponibles:**

| Provider | Modèle par défaut | Max Context | Usage recommandé |
|----------|-------------------|-------------|------------------|
| **OpenAI** | gpt-4o | 128K | Qualité maximale |
| **DeepSeek** | deepseek-chat | 64K | 90% moins cher qu'OpenAI |
| **Groq** | llama-3.3-70b-versatile | 128K | Ultra-rapide (inference) |
| **Mistral** | mistral-large | 32K | Bon pour contenu français |
| **Together AI** | Llama-3.1-405B | 128K | Grands modèles |
| **xAI (Grok)** | grok-2 | 128K | Context window 2M |
| **Ollama** | llama3.1:70b | 32K | Self-hosted (gratuit) |
| **RunPod** | llama-3.1-70b | 32K | Self-hosted GPU |

**Architecture:**
```
services/shared/
├── __init__.py              # Exports publics
└── llm_provider.py          # LLMProvider enum, ProviderConfig, LLMClientFactory
```

**Configuration:**
```env
# Choisir le provider
LLM_PROVIDER=groq

# Clés API par provider
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
MISTRAL_API_KEY=...
TOGETHER_API_KEY=...
XAI_API_KEY=xai-...

# Pour self-hosted (Ollama/RunPod)
OLLAMA_HOST=http://localhost:11434
RUNPOD_ENDPOINT_ID=your_endpoint_id
RUNPOD_API_KEY=...
LLM_TIMEOUT=120  # Configurable
```

**Usage dans le code:**
```python
from services.shared import LLMClientFactory, LLMProvider

# Création automatique selon LLM_PROVIDER
client, model = LLMClientFactory.create()

# Ou forcer un provider spécifique
client, model = LLMClientFactory.create(LLMProvider.GROQ)
```

---

### Training Logger (Fine-tuning Data Collection)

Système de collecte de données d'entraînement pour le fine-tuning de modèles.

**Objectif:** Capturer automatiquement toutes les interactions LLM validées pour créer un dataset de fine-tuning.

**Architecture:**
```
services/shared/
├── __init__.py              # Exports: TrainingLogger, TaskType, log_training_example
└── training_logger.py       # TrainingLogger class
```

**TaskType Enum:**
```python
class TaskType(str, Enum):
    COURSE_PLANNING = "course_planning"
    PRESENTATION_SCRIPT = "presentation_script"
    SLIDE_GENERATION = "slide_generation"
    DIAGRAM_GENERATION = "diagram_generation"
    CODE_GENERATION = "code_generation"
    QUIZ_GENERATION = "quiz_generation"
    TRANSLATION = "translation"
    RAG_RETRIEVAL = "rag_retrieval"
```

**Format JSONL:**
```json
{
  "id": "uuid",
  "timestamp": "2026-01-25T10:30:00Z",
  "task_type": "presentation_script",
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "response": "...",
  "input_tokens": 1500,
  "output_tokens": 800,
  "metadata": {"topic": "Apache Kafka", "audience": "senior"}
}
```

**Configuration:**
```env
TRAINING_LOGGER_ENABLED=true
TRAINING_DATA_PATH=/app/data/training_dataset.jsonl
```

**Volume Docker:**
```yaml
volumes:
  training_data:

services:
  presentation-generator:
    volumes:
      - training_data:/app/data
```

**Usage:**
```python
from services.shared import log_training_example, TaskType

# Logger une interaction validée
log_training_example(
    messages=[...],
    response="...",
    task_type=TaskType.PRESENTATION_SCRIPT,
    model="llama-3.3-70b-versatile",
    provider="groq",
    metadata={"topic": "Kafka"}
)
```

**Export pour OpenAI fine-tuning:**
```python
from services.shared import TrainingLogger

logger = TrainingLogger()
logger.export_for_openai("/app/data/openai_finetune.jsonl")
```

---

### RAG Verifier v4 - Support Multilingue (Janvier 2026)

**Problème résolu:** Le RAG Verifier v3 échouait sur le contenu cross-langue (documents source en anglais, génération en français) car les topics "integration" ≠ "intégration" ne matchaient pas.

**Solution:** Embeddings multilingues E5-large + détection automatique de langue + mode semantic-only pour cross-langue.

**Nouveau backend E5-Large:**

| Backend | Modèle | Dimensions | Multilangue | Qualité |
|---------|--------|------------|-------------|---------|
| **e5-large** | intfloat/multilingual-e5-large | 1024 | ✅ 100+ langues | ⭐⭐⭐⭐⭐ |
| minilm | all-MiniLM-L6-v2 | 384 | 🟡 anglais | ⭐⭐⭐ |
| bge-m3 | BAAI/bge-m3 | 1024 | ✅ 100+ | ⭐⭐⭐⭐ |
| tfidf | TF-IDF local | Variable | 🟡 | ⭐⭐ |

**Caractéristiques E5-Large:**
- Entraîné sur 100+ langues avec alignement cross-lingue
- "query: " prefix pour meilleure performance
- Embeddings français et anglais dans le MÊME espace vectoriel
- "integration" et "intégration" ont une similarité ~0.85

**Configuration:**
```env
# Mode: auto (détecte cross-language), semantic_only, comprehensive
RAG_VERIFIER_MODE=auto

# Backend: e5-large (recommandé), minilm, bge-m3, tfidf
RAG_EMBEDDING_BACKEND=e5-large
```

**Architecture:**
```
services/presentation-generator/services/
├── rag_verifier.py              # RAGVerifier v4 avec détection de langue
└── sync/
    └── embedding_engine.py      # E5-Large backend ajouté
```

**Détection de langue:**
```python
def _detect_language(self, text: str) -> str:
    """Simple language detection based on common words"""
    french_words = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'en', ...}
    english_words = {'the', 'a', 'an', 'is', 'are', 'of', 'and', 'to', ...}
    # Count occurrences and return 'fr' or 'en'
```

**Mode cross-langue automatique:**
```python
def verify_comprehensive(self, generated_content, source_documents, verbose=False):
    # Detect if source and generated are different languages
    if self._is_cross_language(source_text, generated_text):
        # Use semantic-only mode (skip keyword/topic matching)
        return self._verify_semantic_only(generated_content, source_documents)
    else:
        # Full verification with keywords + topics + semantic
        return self._verify_full(generated_content, source_documents)
```

**Seuils adaptés:**

| Mode | Seuil sémantique | Keywords | Topics |
|------|------------------|----------|--------|
| Same language | 45% | ✅ Oui | ✅ Oui |
| Cross-language | 35% | ❌ Skip | ❌ Skip |

**Logs serveur:**
```
[RAG_VERIFY] Detected cross-language: source=en, generated=fr
[RAG_VERIFY] Using semantic-only mode for cross-language verification
[RAG_VERIFY] E5-Large similarity: 0.78 (threshold: 0.35)
[RAG_VERIFY] ✅ RAG COMPLIANT: 78% semantic coverage
```

**Fichiers modifiés:**
- `services/sync/embedding_engine.py` - Ajout E5-Large backend
- `services/rag_verifier.py` - Détection langue + mode semantic-only
- `docker-compose.yml` - Variables RAG_VERIFIER_MODE, RAG_EMBEDDING_BACKEND
- `.env.example` - Documentation des nouvelles variables

**Commit:** `db3b2c8` - feat: RAG Verifier v4 with multilingual E5-large support

---

### WeaveGraph - Graphe de Concepts (Phase 2)

Système de graphe sémantique qui découvre automatiquement les relations entre concepts extraits des documents, permettant une expansion de requêtes pour une meilleure vérification RAG.

**Objectif:** Améliorer la qualité du matching RAG en comprenant que "Kafka" ↔ "message broker" ↔ "event streaming" sont des concepts reliés.

**Architecture:**
```
services/presentation-generator/services/weave_graph/
├── __init__.py              # Exports publics
├── models.py                # ConceptNode, ConceptEdge, WeaveGraph, QueryExpansion
├── concept_extractor.py     # Extraction NLP + regex patterns
├── graph_builder.py         # Construction du graphe avec E5-large embeddings
└── pgvector_store.py        # Stockage PostgreSQL + pgvector
```

**Modèles de données:**

```python
class ConceptNode:
    id: str
    name: str                    # "Apache Kafka"
    canonical_name: str          # "apache_kafka"
    language: str                # "en" ou "fr"
    embedding: List[float]       # E5-large 1024-dim
    source_document_ids: List[str]
    frequency: int               # Nombre d'occurrences
    aliases: List[str]

class ConceptEdge:
    source_id: str
    target_id: str
    relation_type: RelationType  # similar, translation, part_of, prerequisite
    weight: float                # Force de la relation (0-1)
    bidirectional: bool

class QueryExpansion:
    original_query: str
    expanded_terms: List[str]    # Termes enrichis via le graphe
    expansion_paths: Dict        # Chemins de l'expansion
    languages_covered: Set[str]  # Langues couvertes
```

**Tables pgvector:**

```sql
-- Concepts avec embeddings E5-large
CREATE TABLE weave_concepts (
    id UUID PRIMARY KEY,
    canonical_name VARCHAR(255),
    name VARCHAR(500),
    language VARCHAR(10),
    embedding vector(1024),      -- E5-large multilingual
    source_document_ids TEXT[],
    frequency INT,
    user_id VARCHAR(255),
    UNIQUE(canonical_name, user_id)
);

-- Relations entre concepts
CREATE TABLE weave_edges (
    source_concept_id UUID REFERENCES weave_concepts(id),
    target_concept_id UUID REFERENCES weave_concepts(id),
    relation_type VARCHAR(50),   -- similar, translation, part_of
    weight FLOAT,                -- 0-1
    UNIQUE(source_concept_id, target_concept_id, relation_type)
);
```

**Flux de construction:**

```
1. Upload document → ConceptExtractor.extract_concepts()
   - Patterns regex (CamelCase, snake_case, acronymes)
   - TF-IDF pour keywords importants
   - Termes de domaine tech connus

2. Pour chaque concept → E5-large embedding → pgvector INSERT

3. GraphBuilder.build_edges()
   - Similarité cosine entre embeddings
   - Seuil: 0.70 (same language), 0.80 (cross-language)
   - Max 10 edges par concept

4. RAG Verifier → WeaveGraphBuilder.expand_query("Kafka")
   → ["Kafka", "message broker", "consumer", "producer", "file d'attente"]
```

**Intégration RAG Verifier v5:**

```python
# Dans verify_comprehensive()
if self._weave_graph_enabled:
    # Expand query terms using the graph
    expanded_terms, boost = self._expand_with_weave_graph_sync(terms, user_id)

    # Apply boost to coverage (max +15%)
    boosted_coverage = min(1.0, coverage + boost)
```

**Configuration:**

```env
# Activer WeaveGraph
WEAVE_GRAPH_ENABLED=true

# Variables database (pour pgvector)
DATABASE_HOST=postgres
DATABASE_PORT=5432
DATABASE_USER=tiktok_user
DATABASE_PASSWORD=...
DATABASE_NAME=tiktok_platform
```

**Dépendances ajoutées:**
- `asyncpg>=0.29.0` dans requirements.txt

**RelationType enum:**

| Type | Description |
|------|-------------|
| `similar` | Similarité sémantique (embedding >0.7) |
| `translation` | Équivalent cross-langue (EN↔FR) |
| `part_of` | Concept fait partie d'un autre |
| `prerequisite` | Concept prérequis |
| `synonym` | Même signification |
| `hypernym` | Concept plus général |
| `hyponym` | Concept plus spécifique |

**Logs serveur:**

```
[WEAVE_GRAPH] Building graph from 5 documents
[WEAVE_GRAPH] Extracted 127 unique concepts
[WEAVE_GRAPH] Generated 127/127 embeddings
[WEAVE_GRAPH] Stored 127 concepts with embeddings
[WEAVE_GRAPH] Built 342 similarity edges
[RAG_VERIFIER] WeaveGraph expanded 12 -> 28 terms (+6% boost)
```

---

### Resonate Match - Propagation de Résonance (Phase 3)

Système de propagation de scores à travers le graphe de concepts. Quand un concept matche directement, ses voisins reçoivent un score de "résonance" proportionnel à la force de leur connexion.

**Principe:**
```
Query: "Kafka consumer"
         ↓ match direct (1.0)
    [consumer] ─────────────────────────────
         │                                  │
         │ edge (0.85)                     │ edge (0.75)
         ↓                                  ↓
    [Kafka] ←── résonance 0.60        [producer] ←── résonance 0.52
         │
         │ edge (0.70)
         ↓
    [message broker] ←── résonance 0.42
```

**Algorithme multi-hop avec decay:**
```python
resonance(neighbor) = parent_score × edge_weight × decay^depth

# Configuration par défaut
decay = 0.7       # Score decay par hop
max_depth = 3     # Profondeur maximum
min_resonance = 0.10  # Seuil minimum pour continuer
```

**Architecture:**
```
services/presentation-generator/services/weave_graph/
└── resonance_matcher.py    # ResonanceMatcher, ResonanceConfig, ResonanceResult
```

**Classes principales:**

```python
@dataclass
class ResonanceConfig:
    decay_factor: float = 0.7           # Score decay per hop
    max_depth: int = 3                  # Maximum propagation depth
    min_resonance: float = 0.10         # Minimum score to propagate
    boost_translation: float = 1.2      # Boost for cross-language edges
    boost_synonym: float = 1.1          # Boost for synonym edges
    max_resonating_concepts: int = 50   # Limit total concepts

@dataclass
class ResonanceResult:
    scores: Dict[str, float]            # concept_id -> resonance score
    depths: Dict[str, int]              # concept_id -> depth reached
    paths: Dict[str, List[str]]         # concept_id -> path from match
    direct_matches: int
    propagated_matches: int
    total_resonance: float
    max_depth_reached: int
```

**Intégration RAG Verifier v6:**

```python
# Dans verify_comprehensive()
if self._resonance_enabled and self._resonance_matcher:
    resonance_result = self._compute_resonance_sync(
        generated_terms, source_terms, user_id
    )
    resonance_boost = resonance_result['boost']  # Max +15%

# Boost combiné
total_boost = expansion_boost + resonance_boost
boosted_coverage = min(1.0, coverage + total_boost)
```

**Configuration:**

```env
# Activer la résonance
RESONANCE_ENABLED=true

# Paramètres de propagation
RESONANCE_DECAY=0.7      # Decay par hop (0-1)
RESONANCE_MAX_DEPTH=3    # Profondeur max (1-5)
```

**Boosts par type de relation:**

| RelationType | Boost |
|--------------|-------|
| `translation` | ×1.2 |
| `synonym` | ×1.1 |
| `similar` | ×1.0 |
| `part_of` | ×1.0 |

**Logs serveur:**

```
[RAG_VERIFIER] Resonance: 5 direct, 12 propagated, +8% boost
[RAG_VERIFIER] ✅ RAG COMPLIANT (SEMANTIC+WeaveGraph+Resonance): 72% semantic similarity
```

**Fichiers modifiés:**
- `services/weave_graph/resonance_matcher.py` (CRÉÉ)
- `services/weave_graph/__init__.py` - Exports ResonanceMatcher
- `services/rag_verifier.py` - RAG Verifier v6 avec resonance
- `docker-compose.yml` - Variables RESONANCE_*
- `.env.example` - Documentation variables

---

## Phase 7 - VQV-HALLU: Voice Quality Verification (IMPLÉMENTÉ)

### Objectif
Détecter les hallucinations audio dans les voiceovers générés par TTS (ElevenLabs) avant composition vidéo.

### Architecture (4 Layers)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VQV-HALLU PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌─────────────────────────────────────────────────────┐   │
│  │  Audio   │───▶│              LAYER 1: ACOUSTIC ANALYZER              │   │
│  │  Input   │    │  • Spectral Anomaly Detection (distortion, noise)   │   │
│  └──────────┘    │  • Click/Pop Detection, Silence Analysis            │   │
│                  └─────────────────────────┬───────────────────────────┘   │
│                                            ▼                                │
│  ┌──────────┐    ┌─────────────────────────────────────────────────────┐   │
│  │  Text    │───▶│              LAYER 2: LINGUISTIC COHERENCE          │   │
│  │  Source  │    │  • ASR Reverse Transcription (Whisper large-v3)     │   │
│  └──────────┘    │  • Gibberish Detection, Language Consistency        │   │
│                  └─────────────────────────┬───────────────────────────┘   │
│                                            ▼                                │
│                  ┌─────────────────────────────────────────────────────┐   │
│                  │              LAYER 3: SEMANTIC ALIGNMENT            │   │
│                  │  • Embedding Similarity (sentence-transformers)     │   │
│                  │  • Hallucination Boundary Detection                 │   │
│                  └─────────────────────────┬───────────────────────────┘   │
│                                            ▼                                │
│                  ┌─────────────────────────────────────────────────────┐   │
│                  │              LAYER 4: SCORE FUSION ENGINE           │   │
│                  │  • Weighted Ensemble + Cross-Layer Patterns         │   │
│                  └─────────────────────────┬───────────────────────────┘   │
│                                            ▼                                │
│                            ┌──────────────────────────┐                    │
│                            │   QUALITY SCORE (0-100)  │                    │
│                            │   + Action Recommendation│                    │
│                            └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Feature Flags (désactivable)

```bash
# Variables d'environnement
VQV_HALLU_ENABLED=true/false  # Activer/désactiver le service
VQV_STRICT_MODE=false         # Si true, bloque sur erreur; si false, accepte par défaut
VQV_MIN_SCORE=70              # Score minimum acceptable
VQV_MAX_REGEN=3               # Tentatives de régénération max
```

### Graceful Degradation

Le service est conçu pour ne jamais bloquer la génération:
- Si `VQV_HALLU_ENABLED=false` → audio accepté sans validation
- Si service indisponible → audio accepté avec warning
- Si erreur d'analyse → audio accepté en mode non-strict
- Circuit breaker après 5 échecs consécutifs

### Endpoints API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check avec status modèles |
| `GET /api/v1/status` | Status détaillé + statistiques |
| `POST /api/v1/analyze` | Analyse un voiceover |
| `POST /api/v1/analyze/batch` | Analyse plusieurs voiceovers |
| `POST /api/v1/analyze/upload` | Analyse fichier uploadé |
| `GET /api/v1/config/content-types` | Types de contenu et seuils |

### Intégration dans le Pipeline Voiceover (slide_audio_generator.py)

Le service VQV-HALLU est intégré directement dans `SlideAudioGenerator`:

```python
# Initialisation avec paramètres VQV
generator = SlideAudioGenerator(
    voice_id="alloy",
    language="fr",
    vqv_enabled=True,        # Activer validation VQV
    vqv_max_attempts=3,      # Max régénérations si échec
    vqv_min_score=70.0,      # Score minimum acceptable
)

# Génération batch avec validation automatique
batch = await generator.generate_batch(slides, job_id="course_123")

# Chaque SlideAudio contient les résultats VQV
for audio in batch.slide_audios:
    print(f"Slide {audio.slide_index}: validated={audio.vqv_validated}, score={audio.vqv_score}")
    if audio.vqv_issues:
        print(f"  Issues: {audio.vqv_issues}")

# Log automatique en fin de batch:
# [SLIDE_AUDIO] VQV Summary: 8/10 validated, avg_score=82.3, attempts=14, issues=2
```

**Flux de validation:**
1. Audio généré par TTS
2. Si `VQV_HALLU_ENABLED=true` → validation via VQVHalluClient
3. Si score < 70 → suppression audio et régénération (max 3 tentatives)
4. Si 3 échecs → audio accepté avec warning (graceful degradation)
5. Résultats stockés dans `SlideAudio.vqv_*` fields

**Champs SlideAudio ajoutés:**
| Champ | Type | Description |
|-------|------|-------------|
| `vqv_validated` | bool | Audio validé par VQV |
| `vqv_score` | float | Score final (0-100) |
| `vqv_attempts` | int | Nombre de tentatives |
| `vqv_issues` | List[str] | Problèmes détectés |

### Utilisation directe du client

```python
from services.vqv_hallu_client import validate_voiceover

result = await validate_voiceover(
    source_text="Le texte source du voiceover",
    audio_url="https://storage.example.com/audio.mp3",
    audio_id="slide_001",
    content_type="technical_course",
    language="fr",
)

if result.should_regenerate:
    # Score < 70, régénérer le voiceover
    print(f"Issues: {result.primary_issues}")
else:
    # Audio acceptable
    pass
```

### Fichiers créés

```
services/vqv-hallu/
├── main.py                    # FastAPI service
├── client.py                  # Client library
├── requirements.txt
├── Dockerfile
├── analyzers/
│   ├── acoustic_analyzer.py   # Layer 1
│   ├── linguistic_analyzer.py # Layer 2
│   └── semantic_analyzer.py   # Layer 3
├── core/
│   ├── pipeline.py            # Orchestration
│   └── score_fusion.py        # Layer 4
├── models/
│   └── data_models.py         # Structures de données
└── config/
    └── settings.py            # Configuration et seuils
```

---

## Phase 8 - MAESTRO Engine Integration (IMPLÉMENTÉ)

### Objectif

Intégration du système MAESTRO (Multi-level Adaptive Educational Structuring & Teaching Resource Orchestrator) pour la génération de cours avancée avec:
- **Calibration 4D de difficulté** (conceptual_complexity, prerequisites_depth, information_density, cognitive_load)
- **Taxonomie de Bloom** alignée sur les quiz et exercices
- **Progression fluide** (max 15% de saut de difficulté entre concepts)
- **Script segmenté** (intro, explanation, example, summary)

### Architecture (5 Layers Pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAESTRO 5-LAYER PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 1: DOMAIN DISCOVERY                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Analyse du sujet et identification des thèmes                    │    │
│  │ • Extraction des objectifs d'apprentissage                         │    │
│  │ • Détection des prérequis                                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   ↓                                         │
│  Layer 2: KNOWLEDGE GRAPH                                                   │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Construction du graphe de prérequis                              │    │
│  │ • Tri topologique (Kahn's algorithm)                               │    │
│  │ • Détection et résolution des cycles                               │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   ↓                                         │
│  Layer 3: DIFFICULTY CALIBRATION                                            │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Vecteur 4D de difficulté par concept                             │    │
│  │ • Score composite pondéré                                          │    │
│  │ • Mapping vers SkillLevel et BloomLevel                            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   ↓                                         │
│  Layer 4: CURRICULUM SEQUENCING                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Ordre d'apprentissage optimal                                    │    │
│  │ • Progression fluide (max 15% jump)                                │    │
│  │ • Groupement en modules                                            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   ↓                                         │
│  Layer 5: CONTENT GENERATION                                                │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ • Scripts segmentés (intro, explanation, example, summary)         │    │
│  │ • Quiz alignés sur Bloom's Taxonomy                                │    │
│  │ • Exercices pratiques avec solutions                               │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Modes de Génération

| Mode | Description | Utilisation |
|------|-------------|-------------|
| **RAG** | Utilise les documents uploadés | Cours basé sur contenu existant |
| **MAESTRO** | Pipeline 5 couches sans documents | Cours générés à partir de zéro |
| **HYBRID** | (Futur) Combine RAG + MAESTRO | Meilleur des deux mondes |

### Vecteur de Difficulté 4D

```python
@dataclass
class DifficultyVector:
    conceptual_complexity: float  # 0.0-1.0 - Niveau d'abstraction
    prerequisites_depth: float    # 0.0-1.0 - Profondeur des prérequis
    information_density: float    # 0.0-1.0 - Quantité d'information
    cognitive_load: float         # 0.0-1.0 - Effort mental requis

    # Pondération par défaut
    WEIGHTS = {
        "conceptual_complexity": 0.25,
        "prerequisites_depth": 0.20,
        "information_density": 0.25,
        "cognitive_load": 0.30,
    }

    @property
    def composite_score(self) -> float:
        return sum(self.WEIGHTS[k] * getattr(self, k) for k in self.WEIGHTS)
```

### Skill Levels & Bloom's Taxonomy

**Skill Levels:**

| Level | Score Range | Description |
|-------|-------------|-------------|
| BEGINNER | 0.00-0.20 | Concepts fondamentaux |
| INTERMEDIATE | 0.20-0.40 | Application pratique |
| ADVANCED | 0.40-0.60 | Analyse complexe |
| VERY_ADVANCED | 0.60-0.80 | Évaluation critique |
| EXPERT | 0.80-1.00 | Création avancée |

**Bloom's Taxonomy:**

| Level | Cognitive Load | Verbs |
|-------|----------------|-------|
| REMEMBER | < 0.15 | définir, lister, identifier |
| UNDERSTAND | 0.15-0.35 | expliquer, décrire, interpréter |
| APPLY | 0.35-0.50 | utiliser, implémenter, résoudre |
| ANALYZE | 0.50-0.70 | comparer, différencier, examiner |
| EVALUATE | 0.70-0.85 | critiquer, justifier, recommander |
| CREATE | > 0.85 | concevoir, construire, développer |

### Endpoints API MAESTRO (Port 8010)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `POST /api/v1/courses/generate` | Démarrer génération course |
| `GET /api/v1/courses/jobs/{job_id}` | Status du job |
| `GET /api/v1/courses/{course_id}` | Récupérer le cours généré |
| `POST /api/v1/domain/analyze` | Analyser un domaine (preview) |
| `POST /api/v1/concepts/extract` | Extraire concepts d'un thème |
| `GET /api/v1/config/progression-paths` | Paths de progression disponibles |
| `GET /api/v1/config/skill-levels` | Niveaux de compétence |
| `GET /api/v1/config/bloom-levels` | Niveaux Bloom |

### Fichiers Créés

**Nouveau microservice:**
```
services/maestro-engine/
├── main.py                           # FastAPI service
├── Dockerfile
├── requirements.txt
├── models/
│   ├── __init__.py
│   └── data_models.py                # Concept, Lesson, Module, CoursePackage
├── engines/
│   ├── __init__.py
│   ├── domain_discovery.py           # Layer 1
│   ├── knowledge_graph.py            # Layer 2
│   ├── difficulty_calibrator.py      # Layer 3
│   └── curriculum_sequencer.py       # Layer 4
└── generators/
    ├── __init__.py
    └── content_generator.py          # Layer 5
```

**Enrichissement course-generator (Phase A):**
```
services/course-generator/
├── models/
│   └── difficulty_models.py          # DifficultyVector, CalibratedConcept, BloomLevel
├── services/
│   ├── difficulty_calibrator.py      # DifficultyCalibratorService
│   ├── curriculum_sequencer.py       # CurriculumSequencer, LearningPath
│   ├── exercise_generator.py         # PracticalExercise, ExerciseGenerator
│   ├── maestro_adapter.py            # Adapter pour communication HTTP
│   ├── quiz_generator.py             # + BLOOM_QUESTION_MAPPING
│   └── knowledge_graph.py            # + topological_sort(), get_learning_order()
```

**Enrichissement presentation-generator:**
```
services/presentation-generator/
└── models/
    └── presentation_models.py        # + ScriptSegmentType, ScriptSegment, script_segments
```

### Configuration Docker

```yaml
# docker-compose.yml
maestro-engine:
  build:
    context: ./services/maestro-engine
    dockerfile: Dockerfile
  container_name: viralify-maestro-engine
  ports:
    - "8010:8008"
  environment:
    - OPENAI_API_KEY=${OPENAI_API_KEY}
    - OPENAI_MODEL=gpt-4o-mini
  networks:
    - tiktok-network
```

**Variable d'environnement course-generator:**
```env
MAESTRO_ENGINE_URL=http://maestro-engine:8008
```

### Utilisation de l'Adapter

```python
from services.maestro_adapter import get_maestro_adapter, GenerationMode

adapter = get_maestro_adapter()

# Vérifier disponibilité
if await adapter.is_available():
    # Générer un cours avec MAESTRO
    job = await adapter.generate_course(
        subject="Python Programming",
        progression_path="beginner_to_intermediate",
        num_modules=5,
        language="fr",
    )

    # Polling du status
    while job.status != "completed":
        await asyncio.sleep(5)
        job = await adapter.get_job_status(job.job_id)

    # Récupérer le cours
    course = await adapter.get_course(job.job_id)

    # Convertir au format Viralify
    viralify_course = adapter.convert_to_viralify_format(course)
```

---

## Notes importantes

- Les volumes Docker persistants sont configurés pour `/tmp/viralify/videos`, `/tmp/presentations`, `/app/output`, `/tmp/viralify/diagrams`
- Les timeouts OpenAI sont fixés à 120s avec 2 retries
- Le frontend nécessite un rebuild Docker après modification des fichiers
- pgvector utilise l'index HNSW pour la recherche vectorielle rapide
- Les webhooks Stripe/PayPal doivent être configurés dans les dashboards respectifs
- **Graphviz** est requis pour la génération de diagrammes avec la librairie Diagrams
- **visual-generator** est un microservice isolé (port 8003) - les dépendances lourdes (Graphviz, Diagrams) y sont centralisées
- La complexité des diagrammes s'adapte automatiquement selon `target_audience` de la présentation
- **Kroki** est self-hosted pour le rendu Mermaid (privacy + reliability), avec fallback vers mermaid.ink si indisponible
- **maestro-engine** est un microservice isolé (port 8010) pour la génération de cours avec calibration 4D de difficulté
