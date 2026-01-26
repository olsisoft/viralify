# NEXUS 🧠⚡

**Neural Execution & Understanding Synthesis**

Algorithme innovant et unique de génération de code pédagogique contextuel avec décomposition cognitive, orchestration multi-agents, et adaptation temporelle.

---

## 🎯 Innovation

NEXUS est **volontairement difficile à reproduire** car il combine plusieurs innovations uniques :

### 1. Cognitive Decomposition Algorithm (CDA)
```
"une plateforme e-commerce"
           │
           ▼
┌─────────────────────────────────────┐
│  PHASE 1: PERCEPTION                │
│  • Analyse heuristique propriétaire │
│  • Extraction d'entités (LLM)       │
│  • Matrice d'affinité de patterns   │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  PHASE 2: ANALYSIS                  │
│  • Identification des flux métier   │
│  • Mapping des relations            │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  PHASE 3: SYNTHESIS                 │
│  • Sélection de framework (matrice) │
│  • Choix des patterns               │
│  • Définition des layers            │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  PHASE 4: PLANNING                  │
│  • Blueprint cognitif               │
│  • Séquençage pédagogique           │
└─────────────────────────────────────┘
```

### 2. Architecture DNA
Représentation unique d'un projet encodant :
- **Entités** avec vecteurs de complexité
- **Relations** et graphe de dépendances
- **Flux métier** avec scénarios d'erreur
- **Patterns** et décisions architecturales

### 3. Multi-Agent Orchestration
5 agents spécialisés qui collaborent avec feedback loop :

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT ORCHESTRATION                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐                                                   │
│  │  ARCHITECT  │──────▶ Décisions de structure                    │
│  │   Agent     │        Patterns, interfaces                       │
│  └──────┬──────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                   │
│  │   CODER     │──────▶ Génération du code                        │
│  │   Agent     │        Adapté au public cible                     │
│  └──────┬──────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐      ┌─────────────────────────────────────────┐ │
│  │  REVIEWER   │◀────▶│         FEEDBACK LOOP                    │ │
│  │   Agent     │      │  Si score < seuil → régénération        │ │
│  └──────┬──────┘      │  Max 3 itérations par segment           │ │
│         │              └─────────────────────────────────────────┘ │
│         ▼                                                           │
│  ┌─────────────┐                                                   │
│  │  EXECUTOR   │──────▶ Exécution sandbox                         │
│  │   Agent     │        Validation runtime                         │
│  └──────┬──────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                   │
│  │  NARRATOR   │──────▶ Script de narration                       │
│  │   Agent     │        Synchronisation vidéo                      │
│  └─────────────┘                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. Temporal Code Synthesis
Adaptation dynamique au temps alloué :
- **Compression** de verbosité pour les segments non-critiques
- **Priorisation** des concepts essentiels
- **Versions progressives** (v1 naïve → v2 améliorée → v3 production)

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## 📖 Utilisation

### Simple (une ligne)

```python
from nexus import generate_code

result = generate_code(
    project_description="une plateforme e-commerce",
    skill_level="intermediate",
    language="python",
    audience="developer",
    allocated_time=300,  # 5 minutes
    provider="groq",
    api_key="your-api-key"
)

# Accéder aux segments générés
for segment in result.get_segments_ordered():
    print(f"📄 {segment.filename}")
    print(f"   {segment.explanation}")
    print(segment.code)
```

### Avancé

```python
from nexus import (
    NEXUSPipeline, 
    NexusRequest,
    NexusConfig,
    TargetAudience,
    CodeVerbosity,
    create_nexus_pipeline
)

# Configuration personnalisée
config = NexusConfig(
    enable_executor=True,      # Exécuter le code pour validation
    sandbox_enabled=True,      # Isolation sandbox
    max_feedback_iterations=3, # Boucles de feedback max
    verbose=True,
)

# Créer le pipeline
pipeline = create_nexus_pipeline(
    provider="groq",
    api_key="your-api-key",
    config=config
)

# Callback de progression
def on_progress(p):
    print(f"[{p.stage}] {p.percent:.0f}% - {p.message}")

pipeline.set_progress_callback(on_progress)

# Requête détaillée
request = NexusRequest(
    project_description="une plateforme e-commerce avec panier et paiement",
    lesson_context="Module 5: Architecture backend avancée",
    skill_level="advanced",
    language="python",
    target_audience=TargetAudience.DEVELOPER,
    verbosity=CodeVerbosity.PRODUCTION,
    allocated_time_seconds=600,  # 10 minutes
    show_mistakes=True,          # Montrer les erreurs courantes
    show_evolution=True,         # Montrer v1→v2→v3
    include_tests=False,
)

# Générer
response = pipeline.generate(request)

# Résultats
print(f"✅ {len(response.code_segments)} segments générés")
print(f"📝 {response.total_lines_of_code} lignes de code")
print(f"⏱️ {response.total_duration_seconds}s de contenu")
```

---

## 📦 Structure de sortie (JSON)

```json
{
  "request_id": "abc123",
  "architecture_dna": {
    "project_name": "E-commerce Platform",
    "entities": [...],
    "relations": [...],
    "flows": [...],
    "patterns": ["repository", "service_layer"],
    "framework": "fastapi"
  },
  "cognitive_blueprint": {
    "analysis_phase": [...],
    "design_phase": [...],
    "implementation_phase": [...],
    "validation_phase": [...]
  },
  "code_segments": [
    {
      "id": "seg_001",
      "filename": "models/product.py",
      "code": "class Product:\n    ...",
      "explanation": "Le modèle Product représente...",
      "narration_script": "Commençons par créer notre modèle Product...",
      "duration_seconds": 30,
      "key_concepts": ["model", "dataclass"],
      "common_mistakes": ["Oublier la validation"]
    }
  ],
  "sync_metadata": {
    "timeline": [...],
    "total_duration_seconds": 300
  }
}
```

---

## 🎬 Intégration avec pipeline vidéo

```python
from nexus import generate_code

# Générer le code
result = generate_code(
    project_description="une API REST",
    allocated_time=300,
    provider="groq",
    api_key="..."
)

# Pour l'assembleur vidéo
for entry in result.sync_metadata["timeline"]:
    segment_id = entry["segment_id"]
    start_time = entry["start_time_seconds"]
    duration = entry["duration_seconds"]
    narration = entry["narration_script"]
    
    # Trouver le segment correspondant
    segment = next(s for s in result.code_segments if s.id == segment_id)
    
    # Actions pour l'assembleur:
    # 1. Afficher segment.code dans l'IDE simulé à start_time
    # 2. Jouer narration via TTS
    # 3. Highlight segment.key_concepts
    # 4. Durée: duration secondes
```

---

## 🔧 Configuration des audiences

| Audience | Style de code |
|----------|---------------|
| `developer` | Production-ready, error handling, pragmatique |
| `architect` | Focus patterns, structure schématique |
| `student` | Pédagogique, très commenté, progressif |
| `lead` | Balance entre pratique et architecture |

---

## 🎚️ Niveaux de verbosité

| Niveau | Description |
|--------|-------------|
| `MINIMAL` | Squelette, code essentiel uniquement |
| `STANDARD` | Code propre avec commentaires clés |
| `VERBOSE` | Très commenté, chaque ligne expliquée |
| `PRODUCTION` | Production-ready avec logs et error handling |

---

## 🔐 Pourquoi NEXUS est unique

1. **Matrices propriétaires** : Affinité patterns/domaines, sélection de framework
2. **Décomposition cognitive** : 4 phases qui simulent un architecte senior
3. **Multi-agents** : 5 agents spécialisés avec feedback loop
4. **Architecture DNA** : Représentation unique encodant l'essence du projet
5. **Adaptation temporelle** : Compression/expansion selon le temps alloué

Même avec accès au code source, la reproduction nécessite de comprendre et recréer :
- Les matrices de décision
- La logique de décomposition cognitive
- Le protocole inter-agents
- L'algorithme d'adaptation temporelle

---

## 📁 Structure du projet

```
nexus/
├── __init__.py                 # Exports publics
├── core/
│   └── pipeline.py             # Pipeline principal
├── engines/
│   ├── cognitive_decomposition.py  # CDA - Algorithme de décomposition
│   ├── multi_agent_orchestrator.py # Orchestration des 5 agents
│   └── temporal_synthesizer.py     # Adaptation temporelle
├── models/
│   └── data_models.py          # Structures de données
├── providers/
│   └── llm_provider.py         # Interface LLM agnostique
├── examples/
│   └── usage_examples.py       # Exemples
├── requirements.txt
└── README.md
```

---

## 📄 Licence

Propriétaire - Tous droits réservés

---

## 🤝 Intégration avec MAESTRO

NEXUS est conçu pour s'intégrer dans la chaîne MAESTRO :

```
MAESTRO (génération cours) 
    → NEXUS (génération code)
    → VQV-HALLU (validation audio)
    → Assembleur vidéo
```
