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

**Dernier commit:** `f140b1c` - feat: implement sync anchors as hard constraints in SSVS algorithm
**Date:** 2026-01-23
**Travail en cours:** SSVS complet avec anchors, embeddings configurables, calibration

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
