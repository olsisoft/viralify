# VIRALIFY - Documentation d'Architecture

> Version: 2.0
> Dernière mise à jour: Janvier 2026
> Auteurs: Équipe Viralify

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture Microservices](#2-architecture-microservices)
3. [Système de Queue & Workers](#3-système-de-queue--workers)
4. [Pipeline de Génération de Cours](#4-pipeline-de-génération-de-cours)
5. [Algorithmes](#5-algorithmes)
   - [5.1 SSVS - Semantic Slide-Voiceover Synchronization](#51-ssvs---semantic-slide-voiceover-synchronization)
   - [5.2 SSVS-Calibrator - Correction d'Offset Audio-Vidéo](#52-ssvs-calibrator---correction-doffset-audio-vidéo)
   - [5.3 SSVS-D - Synchronisation Orientée Diagrammes](#53-ssvs-d---synchronisation-orientée-diagrammes)
   - [5.4 SSVS-C - Synchronisation Orientée Code](#54-ssvs-c---synchronisation-orientée-code)
   - [5.5 VQV-HALLU - Vérification Qualité Vocale & Détection d'Hallucinations](#55-vqv-hallu---vérification-qualité-vocale--détection-dhallucinations)
   - [5.6 FFmpeg Timeline Compositor](#56-ffmpeg-timeline-compositor)
   - [5.7 Timeline Builder](#57-timeline-builder)
   - [5.8 CompoundTermDetector - Détection ML de Termes Composés](#58-compoundtermdetector---détection-ml-de-termes-composés)
6. [Modèles de Données](#6-modèles-de-données)
7. [Déploiement Multi-Serveurs](#7-déploiement-multi-serveurs)
8. [Configuration & Tuning](#8-configuration--tuning)

---

## 1. Vue d'ensemble

### 1.1 Qu'est-ce que Viralify ?

Viralify est une plateforme de génération automatique de cours vidéo utilisant l'IA. Elle transforme un simple sujet en un cours complet avec :

- **Outline structuré** (sections, leçons, objectifs)
- **Scripts pédagogiques** générés par LLM
- **Voiceover TTS** (Text-to-Speech)
- **Slides animés** avec code, diagrammes, illustrations
- **Vidéos MP4** synchronisées avec précision milliseconde

### 1.2 Caractéristiques principales

| Fonctionnalité | Description |
|----------------|-------------|
| **Multi-LLM** | Support OpenAI, Groq, DeepSeek, Mistral |
| **RAG** | Génération basée sur documents sources |
| **Multi-langue** | Contenu en EN, FR, ES, etc. |
| **Scalabilité** | Architecture workers distribuée |
| **Qualité** | Vérification VQV-HALLU du TTS |

### 1.3 Stack Technologique

```
┌─────────────────────────────────────────────────────────────────────┐
│                           FRONTEND                                  │
│                     Next.js 14 + TypeScript                         │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY                                 │
│                    Spring Cloud Gateway (8080)                      │
└─────────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│    Python     │      │     Java      │      │   Node.js     │
│  Microservices│      │  Microservices│      │  Microservices│
│               │      │               │      │               │
│ FastAPI/Uvicorn│     │ Spring Boot   │      │    Express    │
└───────────────┘      └───────────────┘      └───────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│  PostgreSQL + pgvector │ Redis │ RabbitMQ │ Elasticsearch          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Microservices

### 2.1 Vue d'ensemble des services

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              VIRALIFY PLATFORM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         CORE SERVICES                                    │   │
│  │                                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│   │
│  │  │   course-    │  │presentation- │  │    media-    │  │   visual-    ││   │
│  │  │  generator   │  │  generator   │  │  generator   │  │  generator   ││   │
│  │  │    :8007     │  │    :8006     │  │    :8004     │  │    :8003     ││   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘│   │
│  │                                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│   │
│  │  │   diagrams-  │  │   vqv-hallu  │  │   maestro-   │  │   nexus-     ││   │
│  │  │  generator   │  │    :8008     │  │   engine     │  │   engine     ││   │
│  │  │    :8009     │  │              │  │    :8008     │  │    :8009     ││   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       SUPPORT SERVICES                                   │   │
│  │                                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│   │
│  │  │    auth-     │  │  analytics-  │  │ notification-│  │   trend-     ││   │
│  │  │   service    │  │   service    │  │   service    │  │  analyzer    ││   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Responsabilités des services

| Service | Port | Responsabilité |
|---------|------|----------------|
| `api-gateway` | 8080 | Routage, authentification, rate limiting |
| `course-generator` | 8007 | Orchestration génération de cours, RAG |
| `presentation-generator` | 8006 | Génération de leçons individuelles |
| `media-generator` | 8004 | TTS, composition vidéo finale |
| `visual-generator` | 8003 | Génération d'images IA |
| `diagrams-generator` | 8009 | Diagrammes Mermaid, graphiques |
| `vqv-hallu` | 8008 | Vérification qualité TTS |
| `maestro-engine` | 8008 | Génération avancée sans RAG |
| `nexus-engine` | 8009 | Génération de code pédagogique |

### 2.3 Communication inter-services

```
┌──────────────────┐         HTTP/REST          ┌──────────────────┐
│ course-generator │ ────────────────────────▶  │presentation-gen  │
└──────────────────┘                            └──────────────────┘
         │                                               │
         │                                               │
         ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│    RabbitMQ      │                            │  media-generator │
│  (async jobs)    │                            └──────────────────┘
└──────────────────┘                                     │
         │                                               │
         ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│  course-worker   │                            │    vqv-hallu     │
│  (background)    │                            │  (verification)  │
└──────────────────┘                            └──────────────────┘
```

---

## 3. Système de Queue & Workers

### 3.1 Architecture Queue

Le système utilise **RabbitMQ** pour la distribution des jobs et **Redis** pour la synchronisation des statuts.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODE QUEUE (USE_QUEUE=true)                           │
└─────────────────────────────────────────────────────────────────────────────────┘

     CLIENT                    API                      QUEUE                 WORKER
        │                       │                         │                      │
        │  POST /generate       │                         │                      │
        │──────────────────────▶│                         │                      │
        │                       │                         │                      │
        │                       │  publish(job)           │                      │
        │                       │────────────────────────▶│                      │
        │                       │                         │                      │
        │   { job_id: "abc" }   │                         │   consume(job)       │
        │◀──────────────────────│                         │─────────────────────▶│
        │                       │                         │                      │
        │                       │                         │                      │
        │  GET /jobs/{id}       │                         │    traite le job     │
        │──────────────────────▶│                         │                      │
        │                       │  read status            │                      │
        │                       │◀────────────────────────┼──────────────────────│
        │                       │        REDIS            │   update status      │
        │   { progress: 45% }   │                         │                      │
        │◀──────────────────────│                         │                      │
```

### 3.2 Composants du système de queue

#### RabbitMQ - Distribution des jobs

```python
# Configuration des queues
QUEUE_NAME = "course_generation_queue"
DLQ_NAME = "course_generation_dlq"  # Dead Letter Queue

# Caractéristiques
- Durabilité: Messages persistent aux redémarrages
- Priorité: Support 1-10 (1 = plus haute priorité)
- Prefetch: 1 job par worker à la fois
- Acknowledgment: ACK manuel après traitement complet
```

#### Redis - Synchronisation des statuts

```python
# Structure de données
course_job:{job_id} = {
    "status": "processing",
    "progress": 45.0,
    "current_lecture": "Les boucles for",
    "lectures_completed": 4,
    "lectures_total": 10,
    "outline": "{...JSON...}",
    "updated_at": "2026-01-30T10:30:00Z"
}
```

### 3.3 Cycle de vie d'un Job

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   QUEUED    │────▶│  PLANNING   │────▶│ GENERATING  │────▶│  COMPLETED  │
└─────────────┘     └─────────────┘     │  LECTURES   │     └─────────────┘
                                        └─────────────┘
                                              │
                                              ▼
                                        ┌─────────────┐
                                        │   FAILED    │──▶ DLQ
                                        └─────────────┘
```

### 3.4 Worker - Fonctionnement

```python
class CourseWorker:
    """
    Worker qui consomme les jobs de la queue RabbitMQ.

    Caractéristiques:
    - Traite 1 job à la fois (max_concurrent_jobs=1)
    - Met à jour le statut dans Redis
    - Gère les échecs avec retry automatique
    - Peut être scalé horizontalement
    """

    async def process_job(self, queued_job):
        # 1. Planification du cours
        outline = await self.planner.plan_course(request)
        await self._update_status("generating_lectures", progress=15)

        # 2. Génération parallèle des lectures (max 3)
        result = await self.compositor.compose_course(outline)

        # 3. Packaging et finalisation
        await self._update_status("completed", progress=100)
```

### 3.5 Coordination entre Workers

| Mécanisme | Rôle | Comportement |
|-----------|------|--------------|
| `prefetch_count=1` | Flow control | 1 job par worker |
| `ACK` | Confirmation | Job terminé, envoie suivant |
| `NACK` | Échec | Job → DLQ ou retry |
| Redis | Statut | API lit, Worker écrit |

### 3.6 Gestion des pannes

```
Scénario                          │ Comportement
──────────────────────────────────┼─────────────────────────────────────
Worker crash pendant un job       │ RabbitMQ timeout → requeue → autre worker
Worker lent                       │ Autres workers prennent les jobs libres
Tous les workers occupés          │ Jobs s'accumulent dans la queue
Job échoue 3 fois                 │ Envoyé en Dead Letter Queue (DLQ)
```

---

## 4. Pipeline de Génération de Cours

### 4.1 Vue d'ensemble du pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE DE GÉNÉRATION                             │
└─────────────────────────────────────────────────────────────────────────────────┘

  INPUT                                                                OUTPUT
    │                                                                     │
    ▼                                                                     ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐
│ "Cours sur  │    │   Course    │    │ Presentation│    │  10 vidéos MP4      │
│  Python,    │───▶│  Planner    │───▶│  Generator  │───▶│  + ZIP              │
│  10 leçons" │    │  (outline)  │    │  (vidéos)   │    │  téléchargeable     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────────────┘
                         │                   │
                         ▼                   ▼
                   ┌───────────┐       ┌───────────┐
                   │   LLM     │       │  FFmpeg   │
                   │ GPT-4o    │       │  Encoding │
                   └───────────┘       └───────────┘
```

### 4.2 Étape 1: Course Planning

```python
# Entrée
GenerateCourseRequest:
    topic: "Python pour débutants"
    num_sections: 2
    lectures_per_section: 5
    difficulty_start: "beginner"
    difficulty_end: "intermediate"
    language: "fr"

# Sortie
CourseOutline:
    title: "Maîtriser Python: Du Débutant à l'Intermédiaire"
    sections: [
        Section(
            title: "Fondamentaux",
            lectures: [
                Lecture(title: "Introduction aux variables", duration: 300s),
                Lecture(title: "Types de données", duration: 300s),
                ...
            ]
        ),
        ...
    ]
```

### 4.3 Étape 2: Génération des lectures (parallèle)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GÉNÉRATION PARALLÈLE (MAX 3 CONCURRENT)                      │
└─────────────────────────────────────────────────────────────────────────────────┘

Temps ──────────────────────────────────────────────────────────────────────────▶

     ┌─────────────────┐
     │   Lecture 1     │ ✓
     └─────────────────┘
     ┌─────────────────┐
     │   Lecture 2     │ ✓
     └─────────────────┘
     ┌─────────────────┐
     │   Lecture 3     │ ✓
     └─────────────────┘
                       ┌─────────────────┐
                       │   Lecture 4     │ ✓
                       └─────────────────┘
                       ┌─────────────────┐
                       │   Lecture 5     │ ✓
                       └─────────────────┘
                       ┌─────────────────┐
                       │   Lecture 6     │ ✓
                       └─────────────────┘
                                         ...
```

### 4.4 Pipeline d'une lecture individuelle

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         GÉNÉRATION D'UNE LECTURE (~12 min)                      │
└─────────────────────────────────────────────────────────────────────────────────┘

1. SCRIPT GENERATION (LLM)                                              ~30s
   │  Prompt: "Génère un script pour: Introduction aux variables Python"
   │  Output: Script structuré avec voiceover + contenu slides
   │
   ▼
2. SLIDE PLANNING                                                       ~10s
   │  Découpage en 5-10 slides
   │  Chaque slide: titre, contenu, code, type de visuel
   │
   ▼
3. VOICEOVER GENERATION (TTS)                                           ~60s
   │  Pour chaque slide: texte → audio MP3
   │  Service: ElevenLabs ou OpenAI TTS
   │
   ▼
4. VQV-HALLU VERIFICATION                                               ~30s
   │  Vérification qualité audio
   │  Détection hallucinations TTS
   │  Score minimum: 70%
   │
   ▼
5. VISUAL GENERATION                                                    ~120s
   │  • Code syntax highlighting
   │  • Diagrammes Mermaid
   │  • Images IA (si nécessaire)
   │
   ▼
6. ANIMATION GENERATION                                                 ~120s
   │  • Typing animations pour le code
   │  • Transitions entre slides
   │  • Effets visuels
   │
   ▼
7. SSVS SYNCHRONIZATION                                                 ~10s
   │  Alignement sémantique slides ↔ voiceover
   │  Calibration des offsets temporels
   │
   ▼
8. VIDEO COMPOSITION (FFmpeg)                                           ~300s
   │  Assembly: visuels + audio + animations
   │  Encodage: libx264, CRF 23, preset veryfast
   │
   ▼
9. OUTPUT
   └──▶ lecture_01_introduction_variables.mp4
```

### 4.5 Timeouts et limites

| Opération | Timeout | Raison |
|-----------|---------|--------|
| Génération script LLM | 120s | Appels API OpenAI |
| TTS par slide | 120s | ElevenLabs/OpenAI |
| Encodage segment FFmpeg | max(120s, 3×durée) | Proportionnel |
| Concat vidéo | 300s | Opération I/O |
| Audio mux | 300s | Synchronisation |
| **Max par lecture** | **3600s (60 min)** | Sécurité |

---

## 5. Algorithmes

### 5.1 SSVS - Semantic Slide-Voiceover Synchronization

#### 5.1.1 Problématique

**Le problème fondamental**: Comment aligner précisément les slides d'une présentation avec la narration audio ?

Les approches traditionnelles utilisent des proportions basées sur le nombre de mots ou caractères:
```
Slide 1: 100 mots → 20% du temps
Slide 2: 150 mots → 30% du temps
...
```

**Limites de l'approche proportionnelle**:
- Ne tient pas compte du contenu sémantique
- Ignore les pauses naturelles du narrateur
- Ne détecte pas les transitions logiques ("Maintenant, passons à...")
- Peut couper au milieu d'une phrase

#### 5.1.2 Solution implémentée

SSVS utilise une approche **sémantique** basée sur les embeddings de texte pour trouver l'alignement optimal.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            ALGORITHME SSVS                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

ENTRÉES:
  - slides[]: Liste des slides avec leur contenu textuel
  - segments[]: Segments audio avec transcription + timestamps

PROCESSUS:

1. PHASE D'EMBEDDING
   ┌─────────────┐                    ┌─────────────┐
   │   Slide 1   │──▶ Encoder ──▶    │  Vec 384D   │
   │   Slide 2   │──▶ Encoder ──▶    │  Vec 384D   │
   │   Slide 3   │──▶ Encoder ──▶    │  Vec 384D   │
   └─────────────┘                    └─────────────┘

   ┌─────────────┐                    ┌─────────────┐
   │  Segment 1  │──▶ Encoder ──▶    │  Vec 384D   │
   │  Segment 2  │──▶ Encoder ──▶    │  Vec 384D   │
   │     ...     │──▶ Encoder ──▶    │  Vec 384D   │
   │  Segment M  │──▶ Encoder ──▶    │  Vec 384D   │
   └─────────────┘                    └─────────────┘

2. MATRICE DE SIMILARITÉ
                     Segment 1   Segment 2   Segment 3   ...   Segment M
                   ┌───────────┬───────────┬───────────┬─────┬───────────┐
        Slide 1    │   0.85    │   0.72    │   0.31    │ ... │   0.15    │
        Slide 2    │   0.22    │   0.68    │   0.91    │ ... │   0.33    │
        Slide 3    │   0.11    │   0.25    │   0.44    │ ... │   0.88    │
                   └───────────┴───────────┴───────────┴─────┴───────────┘

3. PROGRAMMATION DYNAMIQUE
   Trouver la partition optimale qui:
   - Respecte l'ordre temporel (segments progressent)
   - Assure la couverture complète (tous segments utilisés)
   - Maximise le score de similarité total
   - Assigne des segments contigus à chaque slide

4. DÉTECTION DES TRANSITIONS
   Bonus pour les marqueurs de transition:
   - "maintenant", "passons à", "ensuite", "next", "now let's"
   - Indique un changement de slide naturel

SORTIE:
  - slide_timings[]: {slide_id, start_time, end_time, confidence_score}
```

#### 5.1.3 Formule de score combiné

```python
score_total = α × score_sémantique + β × score_temporel + γ × score_transition

où:
  α = 0.6  # Poids similarité sémantique (dominant)
  β = 0.3  # Poids cohérence temporelle
  γ = 0.1  # Poids marqueurs de transition

score_sémantique = cosine_similarity(embedding_slide, embedding_segment)

score_temporel = 1.0 - |durée_réelle - durée_attendue| / durée_attendue
  # Pénalise si la durée dévie trop de l'attendu

score_transition = 1.0 si marqueur détecté, 0.0 sinon
```

#### 5.1.4 Pourquoi ça marche

1. **Compréhension du contenu**: Les embeddings capturent le sens sémantique, pas juste les mots
2. **Robustesse aux reformulations**: "variable" et "stocker une valeur" ont des embeddings proches
3. **Contraintes temporelles**: La programmation dynamique garantit un alignement valide
4. **Détection des transitions**: Les marqueurs linguistiques confirment les changements

#### 5.1.5 Avantages

| Avantage | Description |
|----------|-------------|
| Précision sémantique | Aligne basé sur le sens, pas les mots |
| Multi-langue | Embeddings multilingues (MiniLM, BGE-M3) |
| Robuste | Fonctionne même avec reformulations |
| Configurable | Poids α, β, γ ajustables |
| Fallback | Mode proportionnel si échec |

#### 5.1.6 Inconvénients et limites

| Limite | Impact | Mitigation |
|--------|--------|------------|
| Complexité O(n×m²) | Lent pour longs contenus | Segmentation préalable |
| Dépendance embeddings | Qualité variable selon modèle | Auto-sélection modèle |
| Sujets techniques | Embeddings moins précis sur jargon | Fine-tuning possible |
| Silence/musique | Pas de contenu à matcher | Détection silence |

#### 5.1.7 Améliorations possibles

1. **Fine-tuning des embeddings** sur corpus pédagogique
2. **Modèle de transition appris** au lieu de règles
3. **Beam search** au lieu de DP pour explorer plus de solutions
4. **Feedback loop** avec métriques utilisateur

#### 5.1.8 Fichiers clés

```
services/presentation-generator/services/sync/
├── ssvs_algorithm.py      # Algorithme principal (826 lignes)
├── embedding_engine.py    # Encodeurs d'embeddings
└── ssvs_calibrator.py     # Calibration post-sync
```

---

### 5.2 SSVS-Calibrator - Correction d'Offset Audio-Vidéo

#### 5.2.1 Problématique

SSVS produit un alignement **sémantiquement optimal**, mais il existe des décalages pratiques:

1. **Latence STT**: Whisper a un délai de traitement
2. **Anticipation visuelle**: Le slide doit apparaître AVANT que le narrateur en parle
3. **Durée des transitions**: Les animations prennent du temps
4. **Vitesse de parole**: Un narrateur rapide nécessite plus d'anticipation

```
SANS CALIBRATION:
Audio:   "Voici une variable..."
Vidéo:         [Slide variable]  ← Trop tard!

AVEC CALIBRATION:
Audio:   "Voici une variable..."
Vidéo:   [Slide variable]  ← Parfait timing
              ↑
        anticipation de 300ms
```

#### 5.2.2 Solution: Système de correction 5 couches

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        CALIBRATION 5 COUCHES                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

Layer 1: GLOBAL OFFSET
         └─▶ Décalage constant appliqué à tout (-300ms default)

Layer 2: STT LATENCY COMPENSATION
         └─▶ Corrige le délai de transcription Whisper

Layer 3: SEMANTIC ANTICIPATION
         └─▶ Affiche le slide AVANT le mot-clé (-150ms)

Layer 4: TRANSITION DURATION
         └─▶ Compte le temps d'animation (+200ms)

Layer 5: DIAGRAM-TO-CODE SPECIAL
         └─▶ Extra anticipation pour diagramme→code (-1500ms)

RÉSULTAT: Timing perceptuellement parfait
```

#### 5.2.3 Analyse adaptative

```python
class SSVSCalibrator:
    def calibrate(self, timeline, word_timestamps):
        # 1. Analyse du speech rate
        wpm = self._analyze_speech_rate(word_timestamps)

        # 2. Détection des pauses naturelles
        pauses = self._detect_pauses(word_timestamps)

        # 3. Alignement sur frontières de phrases
        sentence_boundaries = self._find_sentence_boundaries(word_timestamps)

        # 4. Application des corrections
        for event in timeline:
            offset = self._compute_offset(event, wpm, pauses)
            event.start_time += offset
            event.end_time += offset

        # 5. Validation des contraintes
        self._enforce_constraints(timeline)  # min 2s, max 120s par slide
```

#### 5.2.4 Presets de calibration

| Preset | Cas d'usage | Anticipation | Global offset |
|--------|-------------|--------------|---------------|
| `default` | Usage général | -150ms | -300ms |
| `fast_speech` | Narrateur rapide | -250ms | -400ms |
| `slow_speech` | Narrateur lent | -100ms | -200ms |
| `technical_content` | Code/diagrammes | -200ms | -350ms |
| `simple_slides` | Slides textuels | -100ms | -250ms |

#### 5.2.5 Avantages

- **Timing perceptuel parfait**: Le slide apparaît au bon moment
- **Adaptatif**: S'ajuste à la vitesse du narrateur
- **Configurable**: Presets + paramètres personnalisables

#### 5.2.6 Limites

- Nécessite des word-level timestamps (Whisper)
- Paramètres empiriques (peuvent nécessiter tuning)

---

### 5.3 SSVS-D - Synchronisation Orientée Diagrammes

#### 5.3.1 Problématique

Quand un slide contient un **diagramme complexe**, il ne suffit pas d'afficher le diagramme entier. Il faut:

1. **Identifier** quel élément du diagramme est mentionné
2. **Mettre en évidence** cet élément au bon moment
3. **Suivre** le parcours visuel naturel

```
Narration: "Le client envoie une requête au serveur..."

Diagramme:
  ┌────────┐         ┌────────┐
  │ Client │───▶─────│ Server │
  └────────┘         └────────┘
     ↑                   ↑
  Focus ici           Puis ici
```

#### 5.3.2 Solution: SSVS-D (Diagram-Aware Synchronization)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            ALGORITHME SSVS-D                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

ENTRÉES:
  - diagram: Structure du diagramme (éléments, positions, connexions)
  - word_timestamps: Transcription avec timing mot par mot

PROCESSUS:

1. PARSING DU DIAGRAMME
   ┌─────────────────────────────────────────────┐
   │ elements: [                                 │
   │   {id: "client", label: "Client", x, y, w, h},
   │   {id: "server", label: "Server", x, y, w, h},
   │   {id: "arrow1", type: "connector", from, to}
   │ ]                                           │
   │ spatial_index: QuadTree pour lookup rapide  │
   │ connection_graph: Graphe des liens          │
   └─────────────────────────────────────────────┘

2. DÉTECTION DES MENTIONS (NLP)
   Narration: "Le client envoie une requête..."
                   ↓
   Mention détectée: "client" → element_id: "client"
   Timestamp: 2.3s - 2.5s

3. RÉSOLUTION SPATIALE
   "en haut à gauche" → quadrant (0, 0)
   "à droite" → x > 0.5
   "en dessous" → y > element.y

4. INFÉRENCE ORDRE DE LECTURE
   - Positions (haut→bas, gauche→droite)
   - Topologie des connexions
   - Poids d'importance

5. GÉNÉRATION DES ANIMATIONS
   ┌─────────────────────────────────────────────┐
   │ focus_points: [                             │
   │   {time: 2.3s, element: "client",           │
   │    type: "highlight", duration: 1.5s},      │
   │   {time: 3.8s, element: "arrow1",           │
   │    type: "animate", duration: 0.5s},        │
   │   {time: 4.3s, element: "server",           │
   │    type: "highlight", duration: 2.0s}       │
   │ ]                                           │
   └─────────────────────────────────────────────┘

SORTIE:
  - DiagramFocusTimeline: Séquence d'animations de focus
```

#### 5.3.3 Types d'animations de focus

| Type | Effet visuel | Cas d'usage |
|------|--------------|-------------|
| `highlight` | Surbrillance colorée | Élément mentionné |
| `zoom` | Zoom progressif | Détail important |
| `pointer` | Flèche indicatrice | Navigation |
| `outline` | Contour animé | Emphase légère |
| `fade_others` | Atténue le reste | Focus exclusif |

#### 5.3.4 Avantages

- **Engagement visuel**: L'œil suit naturellement les animations
- **Compréhension améliorée**: Lien explicite narration↔visuel
- **Automatique**: Pas d'annotation manuelle requise

#### 5.3.5 Limites

- Nécessite des diagrammes structurés (Mermaid, SVG avec IDs)
- Détection NLP imparfaite pour termes techniques
- Pas de support pour images bitmap

---

### 5.4 SSVS-C - Synchronisation Orientée Code

#### 5.5.1 Problématique

Quand une slide contient du **code source**, le voiceover explique généralement le code élément par élément:

```
Narration: "D'abord, nous définissons la fonction calculate_total qui prend une liste..."

Code:
  def calculate_total(items):     ← Révéler ici quand "fonction" est mentionnée
      total = 0
      for item in items:
          total += item.price
      return total
```

**Le défi**: Révéler progressivement le code en synchronisation avec la narration.

#### 5.5.2 Solution: SSVS-C (Code-Aware Synchronization)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            ALGORITHME SSVS-C                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

ENTRÉES:
  - code: Code source à afficher
  - language: Langage de programmation
  - word_timestamps: Transcription avec timing mot par mot

PROCESSUS:

1. PARSING DU CODE (AST ou Regex)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Python: AST parsing                                                      │
   │ Autres: Regex patterns                                                   │
   │                                                                          │
   │ Résultat:                                                                │
   │   elements: [                                                            │
   │     {type: "function", name: "calculate_total", lines: 1-5},            │
   │     {type: "variable", name: "total", lines: 2},                        │
   │     {type: "block", name: "for_loop", lines: 3-4},                      │
   │   ]                                                                      │
   └─────────────────────────────────────────────────────────────────────────┘

2. DÉTECTION DES MENTIONS (NLP)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Patterns de mention:                                                     │
   │   - "la fonction X", "the function X"                                   │
   │   - "cette variable", "this variable"                                   │
   │   - "la boucle", "the loop"                                             │
   │                                                                          │
   │ Flow markers:                                                            │
   │   - "d'abord", "first" → premier élément                                │
   │   - "ensuite", "next" → élément suivant                                 │
   │   - "enfin", "finally" → dernier élément                                │
   │                                                                          │
   │ Résultat:                                                                │
   │   mentions: [                                                            │
   │     {element: "calculate_total", timestamp: 2.3s},                      │
   │     {element: "for_loop", timestamp: 5.8s},                             │
   │   ]                                                                      │
   └─────────────────────────────────────────────────────────────────────────┘

3. GÉNÉRATION SÉQUENCE DE RÉVÉLATION
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ reveal_points: [                                                         │
   │   {element: "function_def", start: 2.1s, end: 5.5s, lines: 1},         │
   │   {element: "variable", start: 5.5s, end: 5.8s, lines: 2},             │
   │   {element: "for_loop", start: 5.8s, end: 8.0s, lines: 3-4},           │
   │   {element: "return", start: 8.0s, end: 9.5s, lines: 5},               │
   │ ]                                                                        │
   └─────────────────────────────────────────────────────────────────────────┘

4. GÉNÉRATION ANIMATIONS FFMPEG
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ FFmpeg drawbox filters pour masquer/révéler:                            │
   │                                                                          │
   │ drawbox=x=0:y=0:w=iw:h=ih:color=black@1:t=fill:enable='lt(t,2.1)',     │
   │ drawbox=x=0:y=50:w=iw:h=ih:color=black@1:t=fill:enable='between(t,2.1,5.5)',
   │ drawbox=x=0:y=100:w=iw:h=ih:color=black@1:t=fill:enable='between(t,5.5,5.8)',
   │ ...                                                                      │
   └─────────────────────────────────────────────────────────────────────────┘

SORTIE:
  - CodeRevealTimeline: Séquence de révélation ligne par ligne
  - FFmpegFilters: Commandes pour le rendu vidéo
```

#### 5.5.3 Types d'éléments de code détectés

| Type | Description | Exemple |
|------|-------------|---------|
| `FUNCTION` | Définition de fonction/méthode | `def calculate():` |
| `CLASS` | Définition de classe | `class Payment:` |
| `BLOCK` | Bloc de code (for, if, with, try) | `for item in items:` |
| `VARIABLE` | Déclaration de variable | `total = 0` |
| `IMPORT` | Import de module | `import pandas as pd` |
| `LINE` | Ligne individuelle | Toute ligne de code |

#### 5.5.4 Patterns de détection NLP

```python
# Patterns de mention directe
MENTION_PATTERNS = [
    r"(?:the |la |le )?(?:function|fonction|method|méthode)\s+[`'\"]?(\w+)",
    r"(?:the |la |le )?(?:class|classe)\s+[`'\"]?(\w+)",
    r"(?:this |cette |ce )?(?:loop|boucle|block|bloc)",
    r"(?:variable|var)\s+[`'\"]?(\w+)",
]

# Flow markers pour navigation séquentielle
FLOW_MARKERS = {
    "next": ["next", "suivant", "puis", "ensuite", "then"],
    "first": ["first", "d'abord", "premièrement", "commençons", "let's start"],
    "now": ["now", "maintenant", "ici", "here"],
    "finally": ["finally", "enfin", "pour finir", "lastly"],
}
```

#### 5.5.5 Architecture des classes

```python
@dataclass
class CodeElement:
    """Un élément de code parsé."""
    element_type: CodeElementType     # FUNCTION, CLASS, BLOCK, etc.
    name: str                         # "calculate_total"
    start_line: int                   # 1
    end_line: int                     # 5
    code_text: str                    # Le code source
    parent: Optional[str]             # Classe parente si méthode
    children: List[str]               # Méthodes si classe
    docstring: Optional[str]          # Docstring extraite

@dataclass
class CodeRevealPoint:
    """Un point de révélation dans la timeline."""
    element: CodeElement
    reveal_start: float               # Timestamp début (secondes)
    reveal_end: float                 # Timestamp fin
    reveal_type: str                  # "progressive" ou "instant"
    line_timings: List[Tuple[int, float]]  # [(line_num, timestamp), ...]

class CodeAwareSynchronizer:
    """Synchroniseur principal pour le code."""

    def __init__(self, code: str, language: str):
        self.elements = self._parse_code(code, language)
        self.mention_detector = CodeMentionDetector()

    def synchronize(
        self,
        voiceover_segments: List[VoiceoverSegment],
        slide_start_time: float
    ) -> List[CodeRevealPoint]:
        """Génère les points de révélation alignés sur le voiceover."""
        ...

    def generate_ffmpeg_filters(
        self,
        reveal_points: List[CodeRevealPoint],
        video_dimensions: Tuple[int, int]
    ) -> str:
        """Génère les filtres FFmpeg pour le reveal progressif."""
        ...
```

#### 5.5.6 Modes de révélation

| Mode | Description | Cas d'usage |
|------|-------------|-------------|
| `progressive` | Ligne par ligne | Explication détaillée |
| `block` | Bloc entier | Présentation rapide |
| `highlight` | Tout visible, highlight actif | Revue de code |

#### 5.5.7 Configuration

```bash
# Variables d'environnement
SSVS_CODE_SYNC_ENABLED=true          # Activer SSVS-C
CODE_REVEAL_MODE=progressive         # progressive | block | highlight
CODE_LINE_REVEAL_DURATION=0.3        # Secondes par ligne
CODE_ANTICIPATION_MS=-200            # Anticiper le reveal
```

#### 5.5.8 Avantages

| Avantage | Description |
|----------|-------------|
| **Engagement** | L'attention suit la révélation progressive |
| **Compréhension** | Lien explicite voix↔code |
| **Automatique** | Pas d'annotation manuelle |
| **Multi-langage** | Python (AST) + autres (regex) |

#### 5.5.9 Limites

| Limite | Impact | Mitigation |
|--------|--------|------------|
| Parsing imparfait | Éléments non détectés | Fallback sur lignes |
| NLP mention detection | Faux positifs/négatifs | Patterns étendus |
| Code très long | Performance | Segmentation |
| Langages exotiques | Regex moins précis | Contribution patterns |

#### 5.5.10 Fichiers clés

```
services/presentation-generator/services/sync/
└── code_synchronizer.py      # Implémentation complète
    ├── CodeElement           # Dataclass élément
    ├── CodeRevealPoint       # Dataclass reveal
    ├── CodeMentionDetector   # Détection NLP
    └── CodeAwareSynchronizer # Orchestrateur
```

---

### 5.5 VQV-HALLU - Vérification Qualité Vocale & Détection d'Hallucinations

#### 5.5.1 Problématique

Les systèmes TTS (Text-to-Speech) peuvent produire des erreurs:

| Type d'erreur | Exemple |
|---------------|---------|
| **Hallucination** | TTS ajoute "et voilà" qui n'est pas dans le texte |
| **Omission** | TTS saute un mot ou une phrase |
| **Distorsion** | Audio avec artefacts, clics, bruit |
| **Changement de langue** | Switch inattendu FR→EN |
| **Répétition** | "la la la variable variable" |

**Impact**: Un cours avec des erreurs TTS perd en crédibilité et compréhension.

#### 5.5.2 Solution: Pipeline 4 couches

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           VQV-HALLU PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │      AUDIO INPUT        │
                    │   (TTS généré + texte   │
                    │       source)           │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│  LAYER 1:         │ │  LAYER 2:         │ │  LAYER 3:         │
│  ACOUSTIC         │ │  LINGUISTIC       │ │  SEMANTIC         │
│                   │ │                   │ │                   │
│ • Spectral noise  │ │ • Whisper ASR     │ │ • Embeddings      │
│ • Distortion THD  │ │ • Phoneme check   │ │ • Word alignment  │
│ • Clicks/pops     │ │ • Language detect │ │ • Hallucination   │
│ • Silence ratio   │ │ • Repetition      │ │   boundaries      │
│ • Speech pace     │ │ • Gibberish       │ │ • Coverage %      │
│                   │ │                   │ │                   │
│ Score: 0-100      │ │ Score: 0-100      │ │ Score: 0-100      │
└─────────┬─────────┘ └─────────┬─────────┘ └─────────┬─────────┘
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      LAYER 4:         │
                    │   SCORE FUSION        │
                    │                       │
                    │ • Weighted average    │
                    │ • Cross-layer patterns│
                    │ • Anomaly detection   │
                    │                       │
                    │ Final Score: 0-100    │
                    │ Verdict: ACCEPT |     │
                    │   REGENERATE | REVIEW │
                    └───────────────────────┘
```

#### 5.5.3 Layer 1: Analyse Acoustique

```python
class AcousticAnalyzer:
    """Détecte les problèmes de qualité audio."""

    def analyze(self, audio_path: str) -> AcousticResult:
        # 1. Analyse spectrale
        spectral_flatness = self._compute_spectral_flatness(audio)
        # Proche de 1.0 = bruit blanc (mauvais)
        # Proche de 0.0 = tons purs (bon)

        # 2. Distorsion harmonique totale (THD)
        thd = self._compute_thd(audio)
        # > 5% = distorsion audible

        # 3. Détection clics/pops
        clicks = self._detect_impulses(audio)
        # Pics > 6 std dev = artefact

        # 4. Ratio silence/parole
        silence_ratio = self._analyze_silence(audio)
        # > 50% silence = problème

        # 5. Vitesse de parole
        wpm = self._estimate_speech_pace(audio)
        # Référence: 150 wpm

        return AcousticResult(
            score=weighted_average(...),
            issues=detected_issues
        )
```

#### 5.5.4 Layer 2: Analyse Linguistique

```python
class LinguisticAnalyzer:
    """Vérifie la validité linguistique de l'audio."""

    def analyze(self, audio_path: str, expected_lang: str) -> LinguisticResult:
        # 1. Transcription ASR
        transcript, confidence = self._transcribe_whisper(audio)

        # 2. Détection de langue
        detected_lang = self._detect_language(transcript)
        language_match = detected_lang == expected_lang

        # 3. Validation phonèmes
        gibberish_score = self._detect_gibberish(transcript)
        # Phonèmes invalides pour la langue

        # 4. Détection répétitions
        repetition_score = self._analyze_repetition(transcript)
        # "le le le" = problème

        return LinguisticResult(
            transcript=transcript,
            confidence=confidence,
            language_match=language_match,
            score=weighted_average(...)
        )
```

#### 5.5.5 Layer 3: Analyse Sémantique

```python
class SemanticAnalyzer:
    """Compare le texte source avec la transcription."""

    def analyze(self, source_text: str, transcript: str) -> SemanticResult:
        # 1. Similarité par embeddings
        source_emb = self.encoder.encode(source_text)
        trans_emb = self.encoder.encode(transcript)
        similarity = cosine_similarity(source_emb, trans_emb)

        # 2. Alignement mot à mot
        alignment = self._align_words(source_text, transcript)
        # Levenshtein distance

        # 3. Détection hallucinations
        hallucinations = self._find_extra_content(source_text, transcript)
        # Mots dans transcript mais pas dans source

        # 4. Couverture du contenu
        coverage = self._compute_coverage(source_text, transcript)
        # % du texte source présent dans transcript

        return SemanticResult(
            similarity=similarity,
            coverage=coverage,
            hallucinations=hallucinations,
            score=weighted_average(...)
        )
```

#### 5.5.6 Layer 4: Fusion des scores

```python
class ScoreFusionEngine:
    """Combine les scores et détecte les patterns cross-layer."""

    def fuse(self, acoustic: AcousticResult,
             linguistic: LinguisticResult,
             semantic: SemanticResult,
             content_type: str) -> FinalVerdict:

        # 1. Poids selon type de contenu
        weights = self.CONTENT_WEIGHTS[content_type]
        # technical_course: {acoustic: 0.3, linguistic: 0.3, semantic: 0.4}

        # 2. Score pondéré
        base_score = (
            weights['acoustic'] * acoustic.score +
            weights['linguistic'] * linguistic.score +
            weights['semantic'] * semantic.score
        )

        # 3. Détection patterns cross-layer
        if acoustic.has_distortion and linguistic.has_gibberish:
            # Corrélation = problème sévère
            base_score -= 20

        if linguistic.language_mismatch and semantic.has_hallucination:
            # Combinaison critique
            base_score -= 30

        # 4. Verdict final
        if base_score >= 70:
            verdict = "ACCEPT"
        elif base_score >= 50:
            verdict = "MANUAL_REVIEW"
        else:
            verdict = "REGENERATE"

        return FinalVerdict(score=base_score, verdict=verdict)
```

#### 5.5.7 Avantages

| Avantage | Description |
|----------|-------------|
| **Multi-couche** | Détecte différents types de problèmes |
| **Configurable** | Poids ajustables par type de contenu |
| **Automatique** | Pas d'intervention humaine |
| **Actionnable** | Verdict clair: Accept/Regenerate/Review |

#### 5.5.8 Limites

| Limite | Impact | Mitigation |
|--------|--------|------------|
| Latence Whisper | +30s par analyse | Cache, parallélisation |
| Faux positifs | Rejette du bon audio | Seuils ajustables |
| Langues rares | Moins précis | Modèles multilingues |
| GPU requis | Whisper est gourmand | CPU fallback possible |

#### 5.5.9 Fichiers clés

```
services/vqv-hallu/
├── main.py                    # FastAPI endpoints
├── core/
│   ├── pipeline.py           # Orchestration 4 layers
│   └── score_fusion.py       # Layer 4
├── analyzers/
│   ├── acoustic_analyzer.py  # Layer 1
│   ├── linguistic_analyzer.py # Layer 2
│   └── semantic_analyzer.py  # Layer 3
└── config/
    └── settings.py           # Configuration
```

---

### 5.6 FFmpeg Timeline Compositor

#### 5.7.1 Problématique

Assembler une vidéo à partir de multiples assets avec:
- Précision **milliseconde** pour la synchronisation audio
- Gestion de **différents formats** (PNG, MP4, GIF, MP3)
- **Performance** sur machines limitées (4 vCPU)
- **Robustesse** aux échecs réseau et I/O

#### 5.7.2 Solution: Stratégie Segment-Concat

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     STRATÉGIE SEGMENT-CONCAT                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

PHASE 1: CRÉATION DES SEGMENTS (parallèle, max 2)

    Event 1          Event 2          Event 3          Event 4
    (image)          (video)          (image)          (animation)
       │                │                │                │
       ▼                ▼                ▼                ▼
    ┌──────┐        ┌──────┐        ┌──────┐        ┌──────┐
    │FFmpeg│        │FFmpeg│        │FFmpeg│        │FFmpeg│
    │loop  │        │trim  │        │loop  │        │      │
    └──┬───┘        └──┬───┘        └──┬───┘        └──┬───┘
       │                │                │                │
       ▼                ▼                ▼                ▼
    seg_001.mp4     seg_002.mp4     seg_003.mp4     seg_004.mp4


PHASE 2: CONCATÉNATION (rapide, no re-encode)

    ┌─────────────────────────────────────────────────────────┐
    │  ffmpeg -f concat -safe 0 -i segments.txt -c copy       │
    │                                                         │
    │  segments.txt:                                          │
    │    file 'seg_001.mp4'                                   │
    │    file 'seg_002.mp4'                                   │
    │    file 'seg_003.mp4'                                   │
    │    file 'seg_004.mp4'                                   │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
                    video_no_audio.mp4


PHASE 3: AUDIO MUXING (synchronisation)

    ┌─────────────────────────────────────────────────────────┐
    │  ffmpeg -i video_no_audio.mp4 -i voiceover.mp3          │
    │         -c:v copy -c:a aac -map 0:v -map 1:a            │
    │         -shortest -vsync cfr                            │
    │         final_output.mp4                                │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
                    VIDÉO FINALE ✓
```

#### 5.7.3 Gestion des ressources

```python
class FFmpegResourceManager:
    """Limite les processus FFmpeg concurrents pour éviter OOM."""

    MAX_CONCURRENT = 2  # Configurable via env

    def __init__(self):
        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
        self.active_processes = []

    async def execute(self, command: List[str], timeout: int) -> bool:
        async with self.semaphore:  # Attend si 2 processus actifs
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.active_processes.append(process)

            try:
                await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            finally:
                self.active_processes.remove(process)
                gc.collect()  # Force garbage collection
```

#### 5.6.4 Stratégie de retry

```python
RETRY_CONFIG = {
    'max_retries': 3,
    'base_delay': 1.0,      # secondes
    'max_delay': 10.0,
    'backoff': 2.0,         # multiplicateur
    'jitter': True          # +0-1s aléatoire
}

async def download_with_retry(url: str) -> bytes:
    for attempt in range(RETRY_CONFIG['max_retries']):
        try:
            return await httpx.get(url, timeout=60)
        except Exception:
            delay = min(
                RETRY_CONFIG['base_delay'] * (RETRY_CONFIG['backoff'] ** attempt),
                RETRY_CONFIG['max_delay']
            )
            if RETRY_CONFIG['jitter']:
                delay += random.random()
            await asyncio.sleep(delay)
    raise MaxRetriesExceeded()
```

#### 5.6.5 Presets d'encodage

| Preset | CRF | FFmpeg Preset | Vitesse | Qualité | Usage |
|--------|-----|---------------|---------|---------|-------|
| `low` | 28 | ultrafast | 3x | Acceptable | Dev/Test |
| `medium` | 23 | veryfast | 2x | Bonne | Production |
| `high` | 20 | fast | 1x | Excellente | Premium |

#### 5.6.6 Avantages

- **Parallélisation contrôlée**: Max 2 FFmpeg = pas d'OOM
- **Concat rapide**: Pas de ré-encodage
- **Retry intelligent**: Récupération automatique
- **Timeout dynamique**: Proportionnel à la durée

#### 5.6.7 Limites

- Nécessite formats compatibles pour concat
- Ré-encodage si résolutions différentes
- Latence I/O sur stockage réseau

---

### 5.7 Timeline Builder

#### 5.7.1 Rôle

Le **Timeline Builder** orchestre tous les algorithmes précédents pour construire une timeline complète de la présentation.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         TIMELINE BUILDER                                        │
│                                                                                 │
│   ENTRÉES                           ORCHESTRATION                   SORTIE      │
│                                                                                 │
│   ┌──────────────┐                                                             │
│   │ Slides       │───┐                                                          │
│   └──────────────┘   │         ┌─────────────────────┐                         │
│   ┌──────────────┐   │         │                     │      ┌──────────────┐   │
│   │ Word         │───┼────────▶│   TimelineBuilder   │─────▶│  Timeline    │   │
│   │ Timestamps   │   │         │                     │      │  (events)    │   │
│   └──────────────┘   │         │  1. Parse anchors   │      └──────────────┘   │
│   ┌──────────────┐   │         │  2. SSVS sync       │                         │
│   │ Audio        │───┤         │  3. Calibration     │                         │
│   │ Duration     │   │         │  4. SSVS-D diagrams │                         │
│   └──────────────┘   │         │  5. Build events    │                         │
│   ┌──────────────┐   │         │  6. Add transitions │                         │
│   │ Diagrams     │───┘         │                     │                         │
│   └──────────────┘             └─────────────────────┘                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### 5.7.2 Flux d'exécution

```python
class TimelineBuilder:
    async def build(self, slides, word_timestamps, audio_duration, diagrams=None):
        # 1. Détection des ancres explicites [SYNC:SLIDE_2]
        anchors = self._find_sync_anchors(slides)

        # 2. Création des segments vocaux
        voice_segments = self._create_voice_segments(word_timestamps)

        # 3. Synchronisation SSVS (avec ancres si présentes)
        if anchors:
            timings = await self.ssvs.synchronize_with_anchors(
                slides, voice_segments, anchors
            )
        else:
            timings = await self.ssvs.synchronize(slides, voice_segments)

        # 4. Calibration des offsets
        calibrated = self.calibrator.calibrate(timings, word_timestamps)

        # 5. Synchronisation diagrammes (si présents)
        if diagrams:
            focus_events = await self.ssvs_d.synchronize(
                diagrams, word_timestamps
            )
            calibrated = self._merge_focus_events(calibrated, focus_events)

        # 6. Construction des événements visuels
        visual_events = self._build_visual_events(calibrated, slides)

        # 7. Ajout des transitions
        final_events = self._add_transitions(visual_events)

        return Timeline(
            total_duration=audio_duration,
            visual_events=final_events,
            sync_anchors=anchors
        )
```

#### 5.7.3 Fallback: Distribution proportionnelle

Si SSVS échoue, le système utilise une distribution proportionnelle:

```python
def _calculate_proportional_timings(self, slides, audio_duration):
    """Fallback quand SSVS échoue."""
    total_chars = sum(len(s.content) for s in slides)

    current_time = 0
    timings = []

    for slide in slides:
        # Proportion basée sur les caractères (plus précis que mots)
        proportion = len(slide.content) / total_chars
        duration = max(0.5, proportion * audio_duration)  # Min 0.5s

        timings.append(SlideTiming(
            slide_id=slide.id,
            start_time=current_time,
            end_time=current_time + duration
        ))
        current_time += duration

    return timings
```

---

### 5.8 CompoundTermDetector - Détection ML de Termes Composés

#### 5.8.1 Problématique

L'extraction de concepts à partir de textes techniques nécessite d'identifier les **termes composés** significatifs:

| Exemple | Problème sans détection |
|---------|-------------------------|
| "Machine Learning" | Extrait "machine" et "learning" séparément |
| "Data Pipeline" | Perd le sens composite |
| "API Gateway" | Concepts fragmentés |

**Le défi**: Comment identifier automatiquement ces termes sans liste hardcodée ?

#### 5.8.2 Solution: Approche Hybride PMI + Sémantique

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     COMPOUND TERM DETECTION PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────────┘

ENTRÉES:
  - corpus[]: Liste de documents textuels

PROCESSUS:

1. CALCUL PMI (Pointwise Mutual Information)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                          │
   │   PMI(x,y) = log₂( P(x,y) / (P(x) × P(y)) )                             │
   │                                                                          │
   │   • PMI élevé → mots qui co-apparaissent plus que par hasard            │
   │   • "Machine Learning" : PMI ≈ 5.0 (forte collocation)                  │
   │   • "The Data" : PMI ≈ 0.2 (mots communs, pas significatif)             │
   │                                                                          │
   │   Configuration:                                                         │
   │     - min_frequency: 2 (éviter les hapax)                               │
   │     - min_pmi: 1.5 (seuil de significativité)                           │
   │     - max_ngram_size: 3 (bigrams et trigrams)                           │
   │                                                                          │
   └─────────────────────────────────────────────────────────────────────────┘

2. FILTRAGE SÉMANTIQUE (optionnel)
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                          │
   │   Score sémantique via:                                                  │
   │     • TF-IDF contre corpus technique (fallback, rapide)                 │
   │     • Embeddings E5-large (si disponible, plus précis)                  │
   │                                                                          │
   │   Détection patterns techniques:                                         │
   │     • CamelCase: "DataFrame" → technique                                │
   │     • snake_case: "data_pipeline" → technique                           │
   │     • Acronymes: "API", "SQL" → technique                               │
   │                                                                          │
   └─────────────────────────────────────────────────────────────────────────┘

3. SCORE COMBINÉ
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                                                                          │
   │   combined_score = α × normalized_pmi + β × semantic_score              │
   │                                                                          │
   │   où:                                                                    │
   │     α = 0.6 (poids PMI)                                                 │
   │     β = 0.4 (poids sémantique)                                          │
   │                                                                          │
   │   Seuil: combined_score ≥ 0.3 → terme composé validé                    │
   │                                                                          │
   └─────────────────────────────────────────────────────────────────────────┘

SORTIE:
  - CompoundTermResult[]: Liste de termes avec scores
    {term: "machine learning", pmi: 5.2, frequency: 15, combined_score: 0.85}
```

#### 5.8.3 Architecture des classes

```python
@dataclass
class PMIConfig:
    min_frequency: int = 2          # Fréquence minimum pour considérer
    min_pmi: float = 1.5            # PMI minimum (log2)
    max_ngram_size: int = 3         # Jusqu'aux trigrams
    smoothing: float = 1.0          # Lissage Laplace

@dataclass
class CompoundDetectorConfig:
    pmi_config: PMIConfig = field(default_factory=PMIConfig)
    use_embeddings: bool = False    # Activer filtrage sémantique
    min_combined_score: float = 0.3 # Seuil score combiné
    pmi_weight: float = 0.6         # Poids PMI dans score
    semantic_weight: float = 0.4    # Poids sémantique

class PMICalculator:
    """Calcule le PMI pour les n-grams."""

    def train(self, texts: List[str]) -> None:
        # Compte unigrammes et n-grammes
        # Calcule les probabilités

    def calculate_pmi(self, ngram: str) -> float:
        # PMI = log2(P(ngram) / P(w1) × P(w2) × ...)

    def get_top_collocations(self, n: int, top_k: int) -> List[Tuple]:
        # Retourne les meilleurs n-grammes par PMI

class SemanticFilter:
    """Filtre les termes par pertinence technique."""

    def is_technical(self, term: str) -> Tuple[bool, float]:
        # Détecte CamelCase, snake_case, acronymes
        # Score TF-IDF ou embedding

    def filter_terms(self, terms: List) -> List:
        # Garde les termes techniques

class CompoundTermDetector:
    """Détecteur principal combinant PMI + sémantique."""

    def train(self, texts: List[str]) -> None:
        # Entraîne le PMI calculator sur le corpus

    def detect(self, top_k: int = 100) -> List[CompoundTermResult]:
        # Retourne les termes composés détectés

    def is_compound_term(self, ngram: str) -> Tuple[bool, float]:
        # Vérifie si un n-gram spécifique est un terme composé
```

#### 5.8.4 Intégration avec ConceptExtractor

```python
@dataclass
class ExtractionConfig:
    # ... existing config ...

    # ML-based compound detection
    use_ml_compound_detection: bool = True
    ml_min_pmi: float = 1.5
    ml_min_frequency: int = 2
    ml_min_combined_score: float = 0.3
    use_semantic_filter: bool = False

class ConceptExtractor:
    def __init__(self, config: ExtractionConfig):
        self._compound_detector: Optional[CompoundTermDetector] = None
        self._learned_compound_terms: Set[str] = set()
        self._is_trained: bool = False

        if config.use_ml_compound_detection:
            self._init_compound_detector()

    def train_on_corpus(self, texts: List[str]) -> int:
        """
        Entraîne le détecteur sur un corpus.

        Retourne le nombre de termes composés appris.
        """
        self._compound_detector.train(texts)
        results = self._compound_detector.detect(top_k=200)
        self._learned_compound_terms = {r.term for r in results}
        self._is_trained = True
        return len(self._learned_compound_terms)

    def _get_effective_compound_terms(self) -> Set[str]:
        """
        Retourne l'union des termes appris et hardcodés.
        """
        if self._learned_compound_terms:
            return self._learned_compound_terms | self.KNOWN_COMPOUND_TERMS
        return self.KNOWN_COMPOUND_TERMS
```

#### 5.8.5 Flux d'utilisation

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           WORKFLOW INTÉGRÉ                                       │
└─────────────────────────────────────────────────────────────────────────────────┘

1. PHASE D'ENTRAÎNEMENT (une fois par cours/corpus)

   Upload documents → RAG parsing → train_on_corpus(texts)
                                           │
                                           ▼
                                    PMI calculation
                                           │
                                           ▼
                                    Learned terms: {"machine learning",
                                                    "data pipeline",
                                                    "api gateway", ...}

2. PHASE D'EXTRACTION (pour chaque nouveau texte)

   extract_concepts(text)
         │
         ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ Pour chaque n-gram potentiel:                                   │
   │                                                                  │
   │   if ngram in learned_terms:                                    │
   │       score = ML_score (élevé)                                  │
   │   elif ngram in KNOWN_COMPOUND_TERMS:                           │
   │       score = hardcoded_score (moyen)                           │
   │   else:                                                          │
   │       score = title_case_detection (faible)                     │
   │                                                                  │
   │   Garde les concepts avec meilleur score                        │
   └─────────────────────────────────────────────────────────────────┘
         │
         ▼
   ConceptNode[]: Liste de concepts extraits
```

#### 5.8.6 Avantages

| Avantage | Description |
|----------|-------------|
| **Adaptabilité** | Apprend les termes spécifiques au domaine |
| **Précision** | PMI détecte les vraies collocations |
| **Extensibilité** | Termes manuels + appris + hardcodés |
| **Performance** | PMI = O(n), rapide sur grands corpus |
| **Multi-langue** | Fonctionne avec FR, EN, etc. |

#### 5.8.7 Limites

| Limite | Impact | Mitigation |
|--------|--------|------------|
| Besoin de corpus | Pas d'apprentissage sans textes | Fallback sur hardcoded |
| Termes rares | PMI instable si freq < 2 | Seuil min_frequency |
| Faux positifs | "The Data" pourrait passer | Filtrage sémantique |

#### 5.8.8 Configuration

```bash
# Variables d'environnement
COMPOUND_DETECTION_ENABLED=true
COMPOUND_MIN_PMI=1.5
COMPOUND_MIN_FREQUENCY=2
COMPOUND_MIN_SCORE=0.3
COMPOUND_USE_SEMANTIC_FILTER=false  # true pour plus de précision
```

#### 5.8.9 Fichiers clés

```
services/presentation-generator/services/weave_graph/
├── compound_detector.py           # PMICalculator, SemanticFilter, CompoundTermDetector
├── concept_extractor.py           # ConceptExtractor avec intégration ML
└── __init__.py                    # Exports publics

services/presentation-generator/tests/
├── test_compound_detector.py      # 49 tests unitaires
└── test_concept_extractor_integration.py  # 16 tests d'intégration
```

---

## 6. Modèles de Données

### 6.1 CourseJob

```python
class CourseJob(BaseModel):
    """Représente un job de génération de cours."""

    # Identifiant
    job_id: str                    # UUID unique

    # Statut
    status: str                    # queued | processing | completed | failed
    current_stage: CourseStage     # PLANNING | GENERATING | COMPILING
    progress: float                # 0.0 - 100.0
    message: str                   # Message de statut actuel

    # Outline généré
    outline: Optional[CourseOutline]

    # Tracking des lectures
    lectures_total: int
    lectures_completed: int
    lectures_in_progress: int
    lectures_failed: int
    current_lecture_title: str

    # Résultats
    output_urls: List[str]         # URLs des vidéos
    zip_url: Optional[str]         # URL du ZIP

    # Timestamps
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]

    # Erreurs
    error: Optional[str]
    failed_lecture_ids: List[str]
    failed_lecture_errors: Dict[str, str]
```

### 6.2 CourseOutline

```python
class CourseOutline(BaseModel):
    """Structure d'un cours généré."""

    title: str
    description: str
    target_audience: str
    language: str                  # 'en', 'fr', 'es', etc.
    difficulty_start: DifficultyLevel
    difficulty_end: DifficultyLevel
    total_duration_minutes: int

    sections: List[Section]

class Section(BaseModel):
    id: str
    title: str
    description: str
    order: int
    lectures: List[Lecture]

class Lecture(BaseModel):
    id: str
    title: str
    description: str
    objectives: List[str]
    difficulty: DifficultyLevel
    duration_seconds: int
    order: int
    status: str                    # pending | generating | completed | failed
    video_url: Optional[str]
```

### 6.3 Timeline

```python
class Timeline(BaseModel):
    """Timeline complète d'une présentation."""

    total_duration: float          # Durée totale en secondes
    visual_events: List[VisualEvent]
    sync_anchors: List[SyncAnchor]
    semantic_scores: Dict[str, float]  # Confiance par slide

class VisualEvent(BaseModel):
    event_id: str
    slide_id: str
    start_time: float
    end_time: float
    event_type: str                # slide | transition | focus
    asset_url: str
    animation_config: Optional[dict]

class SyncAnchor(BaseModel):
    marker: str                    # [SYNC:SLIDE_2]
    slide_id: str
    timestamp: float
```

---

## 7. Déploiement Multi-Serveurs

### 7.1 Architecture distribuée

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SERVEUR PRINCIPAL                                     │
│                                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   API GW    │  │  RabbitMQ   │  │    Redis    │  │  PostgreSQL │           │
│  │   :8080     │  │   :5672     │  │   :6379     │  │   :5432     │           │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘           │
│                          │                │                                     │
└──────────────────────────┼────────────────┼─────────────────────────────────────┘
                           │                │
          ┌────────────────┼────────────────┤
          │                │                │
┌─────────┼────────────────┼────────────────┼─────────────────────────────────────┐
│         ▼                ▼                ▼           SERVEUR WORKER 1          │
│  ┌─────────────────────────────────────────────┐                                │
│  │     course-worker (×3)                      │                                │
│  │     presentation-generator                  │                                │
│  │     media-generator                         │                                │
│  └─────────────────────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────────────────────────┐
│         ▼                                         SERVEUR WORKER 2              │
│  ┌─────────────────────────────────────────────┐                                │
│  │     course-worker (×3)                      │                                │
│  │     presentation-generator                  │                                │
│  │     media-generator                         │                                │
│  └─────────────────────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Déploiement des workers

```bash
# Sur le serveur worker
git clone https://github.com/olsisoft/viralify.git
cd viralify

# Configuration
cp .env.workers.example .env.workers
# Éditer .env.workers avec:
#   MAIN_SERVER_HOST=<IP serveur principal>
#   RABBITMQ_USER, RABBITMQ_PASSWORD
#   REDIS_PASSWORD
#   OPENAI_API_KEY

# Lancement avec 3 workers
docker-compose -f docker-compose.workers.yml --env-file .env.workers up -d --scale course-worker=3
```

### 7.3 Scaling recommendations

| RAM serveur | CPU | Workers recommandés |
|-------------|-----|---------------------|
| 8 GB | 4 cores | 2 workers |
| 16 GB | 8 cores | 4 workers |
| 32 GB | 16 cores | 8 workers |

**Formule**: `workers = min(RAM_GB / 2, CPU_cores / 2)`

---

## 8. Configuration & Tuning

### 8.1 Variables d'environnement principales

```bash
# Mode Queue
USE_QUEUE=true
RABBITMQ_URL=amqp://user:pass@rabbitmq:5672/
REDIS_URL=redis://:password@redis:6379/7

# LLM Provider
LLM_PROVIDER=openai           # openai | groq | deepseek
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...          # Alternative rapide et économique

# Performance FFmpeg
FFMPEG_MAX_CONCURRENT=2       # Max processus parallèles
FFMPEG_PRESET=veryfast        # ultrafast | veryfast | fast
UVICORN_WORKERS=1             # Workers HTTP par service

# VQV-HALLU
VQV_HALLU_ENABLED=true
VQV_MIN_SCORE=70              # Score minimum acceptable
VQV_MAX_REGEN=3               # Tentatives de régénération
```

### 8.2 Tuning SSVS

```python
# Dans ssvs_algorithm.py
SSVS_CONFIG = {
    'alpha': 0.6,              # Poids sémantique (augmenter pour plus de précision)
    'beta': 0.3,               # Poids temporel
    'gamma': 0.1,              # Poids transitions
    'embedding_backend': 'auto',  # minilm | bge-m3 | tfidf
    'min_segment_duration': 0.5,
    'max_segment_duration': 120.0
}
```

### 8.3 Tuning Calibration

```python
# Presets disponibles
CALIBRATION_PRESETS = {
    'default': {
        'global_offset_ms': -300,
        'semantic_anticipation_ms': -150,
        'transition_duration_ms': 200
    },
    'fast_speech': {
        'global_offset_ms': -400,
        'semantic_anticipation_ms': -250,
        'transition_duration_ms': 150
    },
    'technical_content': {
        'global_offset_ms': -350,
        'semantic_anticipation_ms': -200,
        'diagram_to_code_anticipation_ms': -1500
    }
}
```

### 8.4 Monitoring recommandé

```yaml
# Métriques clés à surveiller
metrics:
  - job_queue_depth         # Jobs en attente dans RabbitMQ
  - job_processing_time     # Temps moyen par job
  - lecture_success_rate    # % de lectures réussies
  - vqv_rejection_rate      # % d'audio rejeté par VQV
  - ffmpeg_memory_usage     # RAM utilisée par FFmpeg
  - worker_utilization      # % d'activité des workers
```

---

## Annexes

### A. Glossaire

| Terme | Définition |
|-------|------------|
| **SSVS** | Semantic Slide-Voiceover Synchronization |
| **VQV-HALLU** | Voice Quality Verification & Hallucination Detection |
| **TTS** | Text-to-Speech |
| **ASR** | Automatic Speech Recognition |
| **RAG** | Retrieval-Augmented Generation |
| **DLQ** | Dead Letter Queue |
| **CRF** | Constant Rate Factor (qualité FFmpeg) |

### B. Références

- [RabbitMQ Documentation](https://www.rabbitmq.com/documentation.html)
- [FFmpeg Filters](https://ffmpeg.org/ffmpeg-filters.html)
- [Whisper ASR](https://github.com/openai/whisper)
- [Sentence Transformers](https://www.sbert.net/)

---

*Document généré automatiquement - Viralify Platform v2.0*
