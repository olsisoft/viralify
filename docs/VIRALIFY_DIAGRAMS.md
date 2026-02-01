# VIRALIFY DIAGRAMS - Documentation Technique

> Version: 1.1.0
> Repository: [github.com/olsisoft/viralify-diagrams](https://github.com/olsisoft/viralify-diagrams)
> Licence: MIT
> Dernière mise à jour: Février 2026

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Installation](#2-installation)
3. [Architecture](#3-architecture)
4. [Core Components](#4-core-components)
5. [Layouts](#5-layouts)
6. [Exporters](#6-exporters)
7. [Narration System](#7-narration-system)
8. [Thèmes](#8-thèmes)
9. [Intégration Viralify](#9-intégration-viralify)
10. [API Reference](#10-api-reference)

---

## 1. Vue d'ensemble

### 1.1 Qu'est-ce que Viralify Diagrams?

**Viralify Diagrams** est une librairie Python de génération de diagrammes professionnels optimisée pour le contenu vidéo éducatif.

Inspirée de [mingrammer/diagrams](https://github.com/mingrammer/diagrams), elle ajoute:

- **Optimisation vidéo**: Lisibilité HD, contraste élevé, polices larges
- **Animation intégrée**: SVG animés avec CSS, timeline exportable
- **Narration automatique**: Scripts voiceover synchronisés avec les diagrammes
- **Thèmes personnalisables**: 6 thèmes intégrés + upload JSON utilisateur
- **Auto-simplification**: Limite à 8-10 nodes pour clarté vidéo

### 1.2 Cas d'utilisation

| Contexte | Usage |
|----------|-------|
| Cours vidéo | Diagrammes d'architecture avec animation progressive |
| Tutoriels | Visualisation de flows et pipelines |
| Documentation | Export SVG/PNG haute qualité |
| Présentations | Slides animés avec voiceover synchronisé |

### 1.3 Caractéristiques principales

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         VIRALIFY DIAGRAMS FEATURES                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                 │
│  │   CORE          │  │   LAYOUTS       │  │   EXPORTERS     │                 │
│  │                 │  │                 │  │                 │                 │
│  │ • Diagram       │  │ • Grid          │  │ • SVG           │                 │
│  │ • Node          │  │ • Horizontal    │  │ • AnimatedSVG   │                 │
│  │ • Edge          │  │ • Vertical      │  │ • PNG Frames    │                 │
│  │ • Cluster       │  │ • Radial        │  │                 │                 │
│  │ • Theme         │  │ • Graphviz      │  │                 │                 │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                 │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐                                      │
│  │   NARRATION     │  │   THEMES        │                                      │
│  │                 │  │                 │                                      │
│  │ • DiagramNarrat.│  │ • dark          │                                      │
│  │ • SRT export    │  │ • light         │                                      │
│  │ • SSML export   │  │ • corporate     │                                      │
│  │ • JSON timeline │  │ • neon          │                                      │
│  │                 │  │ • ocean         │                                      │
│  │                 │  │ • gradient      │                                      │
│  └─────────────────┘  └─────────────────┘                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Installation

### 2.1 Installation standard

```bash
# Installation de base
pip install viralify-diagrams

# Avec support PNG export (requiert cairosvg)
pip install viralify-diagrams[png]

# Avec toutes les dépendances optionnelles
pip install viralify-diagrams[all]
```

### 2.2 Installation développeur

```bash
git clone https://github.com/olsisoft/viralify-diagrams.git
cd viralify-diagrams
pip install -e ".[dev]"
```

### 2.3 Dépendances

| Dépendance | Type | Usage |
|------------|------|-------|
| Pillow | Core | Traitement d'images |
| cairosvg | Optionnel | Export PNG depuis SVG |
| pygraphviz | Optionnel | Layout Graphviz (50+ nodes) |
| FFmpeg | Système | Création vidéo depuis frames |

---

## 3. Architecture

### 3.1 Structure du projet

```
viralify-diagrams/
├── viralify_diagrams/
│   ├── __init__.py              # Exports publics, version
│   ├── core/
│   │   ├── __init__.py
│   │   ├── diagram.py           # Diagram, Node, Edge, Cluster
│   │   └── theme.py             # Theme, ThemeManager, ThemeColors
│   ├── layouts/
│   │   ├── __init__.py          # get_layout(), auto_layout()
│   │   ├── base.py              # BaseLayout ABC
│   │   ├── grid.py              # GridLayout
│   │   ├── horizontal.py        # HorizontalLayout
│   │   ├── vertical.py          # VerticalLayout
│   │   ├── radial.py            # RadialLayout
│   │   └── graphviz_layout.py   # GraphvizLayout (dot, neato, fdp...)
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── svg_exporter.py      # SVGExporter
│   │   ├── animated_svg_exporter.py  # AnimatedSVGExporter
│   │   └── png_frame_exporter.py     # PNGFrameExporter
│   └── narration/
│       ├── __init__.py
│       └── diagram_narrator.py  # DiagramNarrator, NarrationScript
├── examples/
│   ├── basic_diagram.py
│   ├── custom_theme.py
│   └── animated_export.py
├── tests/
├── README.md
├── pyproject.toml
└── requirements.txt
```

### 3.2 Pipeline de génération

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DIAGRAM GENERATION PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────────┘

1. CREATION
   ┌─────────────┐
   │  Diagram    │  diagram = Diagram(title="API", theme="dark")
   │  + Nodes    │  diagram.add_node(Node("api", "API Gateway"))
   │  + Edges    │  diagram.connect("api", "db")
   │  + Clusters │  diagram.add_cluster(Cluster("Backend", ["api", "db"]))
   └──────┬──────┘
          │
          ▼
2. LAYOUT
   ┌─────────────┐
   │  Layout     │  layout = HorizontalLayout()
   │  Engine     │  diagram = layout.layout(diagram)
   │             │  # Calcule positions x, y de chaque élément
   └──────┬──────┘
          │
          ▼
3. ANIMATION ORDER
   ┌─────────────┐
   │  Assign     │  diagram.assign_animation_order()
   │  Order      │  # BFS depuis sources: node.order = 0, 1, 2...
   └──────┬──────┘
          │
          ▼
4. EXPORT
   ┌─────────────┐
   │  Exporter   │  exporter = AnimatedSVGExporter()
   │             │  svg = exporter.export(diagram, "output.svg")
   └──────┬──────┘
          │
          ▼
5. NARRATION (optionnel)
   ┌─────────────┐
   │  Narrator   │  narrator = DiagramNarrator(style="educational")
   │             │  script = narrator.generate_script(diagram)
   │             │  print(script.to_srt())  # Sous-titres SRT
   └─────────────┘
```

---

## 4. Core Components

### 4.1 Diagram

Conteneur principal pour tous les éléments du diagramme.

```python
from viralify_diagrams import Diagram, Node, Edge

diagram = Diagram(
    title="My Architecture",
    description="System overview",
    theme="dark",           # Nom du thème
    layout="horizontal",    # Layout par défaut
    width=1920,             # Largeur canvas
    height=1080,            # Hauteur canvas
    padding=50,             # Padding intérieur
    max_nodes=10            # Limite pour auto-simplification
)

# Ajouter des éléments
node = diagram.add_node(Node(label="API Gateway"))
diagram.connect("api", "database", label="SQL")
```

**Méthodes principales:**

| Méthode | Description |
|---------|-------------|
| `add_node(node)` | Ajoute un node |
| `add_edge(edge)` | Ajoute un edge |
| `add_cluster(cluster)` | Ajoute un cluster |
| `connect(src, dst, label)` | Crée et ajoute un edge |
| `get_node(id)` | Récupère un node par ID |
| `simplify()` | Réduit à max_nodes |
| `assign_animation_order()` | Calcule l'ordre d'apparition |
| `get_render_order()` | Liste ordonnée pour rendu |
| `to_dict()` | Export JSON-serializable |

### 4.2 Node

Représente un composant du diagramme.

```python
from viralify_diagrams import Node
from viralify_diagrams.core.diagram import NodeShape

node = Node(
    label="PostgreSQL",         # Affiché (max 20 chars, auto-truncated)
    id="db_main",               # Identifiant unique (auto-généré si omis)
    icon="aws/database/rds",    # Icône (optionnel)
    shape=NodeShape.CYLINDER,   # Forme
    description="Main database", # Pour narration
    order=0,                    # Ordre d'animation (auto-assigné)

    # Style overrides (None = use theme)
    fill_color="#1a3a5c",
    stroke_color="#00b4d8",
    text_color="#ffffff"
)
```

**NodeShape Enum:**

| Shape | Usage |
|-------|-------|
| `RECTANGLE` | Composant générique |
| `ROUNDED` | Défaut, coins arrondis |
| `CIRCLE` | Point central, hub |
| `DIAMOND` | Décision, condition |
| `HEXAGON` | Process, worker |
| `CYLINDER` | Database, storage |
| `PARALLELOGRAM` | I/O, file |
| `CLOUD` | Cloud service externe |

### 4.3 Edge

Connexion entre deux nodes.

```python
from viralify_diagrams import Edge
from viralify_diagrams.core.diagram import EdgeStyle, EdgeDirection

edge = Edge(
    source="api",               # ID du node source
    target="database",          # ID du node cible
    label="REST API",           # Label sur l'edge
    style=EdgeStyle.DASHED,     # Style de ligne
    direction=EdgeDirection.FORWARD,  # Direction de la flèche
    description="API calls",    # Pour narration

    # Style overrides
    color="#e94560",
    width=2
)
```

**EdgeStyle Enum:**

| Style | Rendu |
|-------|-------|
| `SOLID` | Ligne continue `────` |
| `DASHED` | Tirets `- - -` |
| `DOTTED` | Points `• • •` |

**EdgeDirection Enum:**

| Direction | Rendu |
|-----------|-------|
| `FORWARD` | `──▶` |
| `BACKWARD` | `◀──` |
| `BOTH` | `◀─▶` |
| `NONE` | `───` |

### 4.4 Cluster

Groupe de nodes avec bordure et label.

```python
from viralify_diagrams import Cluster

cluster = Cluster(
    label="Backend Services",
    id="backend",
    nodes=["api", "auth", "users"],  # IDs des nodes contenus
    description="Core backend infrastructure",

    # Style overrides
    fill_color="#16213e",
    stroke_color="#0f3460",
    label_color="#ffffff"
)

# Ajouter des nodes après création
cluster.add_node("cache")

# Clusters imbriqués
inner = Cluster(label="Database Layer", nodes=["db1", "db2"])
cluster.add_cluster(inner)
```

---

## 5. Layouts

### 5.1 Vue d'ensemble

| Layout | Description | Cas d'usage | Complexité |
|--------|-------------|-------------|------------|
| `GridLayout` | Grille uniforme | Comparaisons | ≤20 nodes |
| `HorizontalLayout` | Gauche → Droite | Pipelines, flows | ≤15 nodes |
| `VerticalLayout` | Haut → Bas | Hiérarchies | ≤15 nodes |
| `RadialLayout` | Hub central | API, étoile | ≤12 nodes |
| `GraphvizLayout` | Algorithme Graphviz | Complexe | 10-100+ nodes |

### 5.2 Utilisation

```python
from viralify_diagrams import HorizontalLayout, get_layout

# Instanciation directe
layout = HorizontalLayout()
diagram = layout.layout(diagram)

# Via factory
layout = get_layout("vertical")
diagram = layout.layout(diagram)

# Graphviz avec algorithme spécifique
layout = get_layout("graphviz", algorithm="dot")
diagram = layout.layout(diagram)
```

### 5.3 GridLayout

```
┌─────────────────────────────────────────────┐
│                                             │
│   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   │
│   │  1  │   │  2  │   │  3  │   │  4  │   │
│   └─────┘   └─────┘   └─────┘   └─────┘   │
│                                             │
│   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   │
│   │  5  │   │  6  │   │  7  │   │  8  │   │
│   └─────┘   └─────┘   └─────┘   └─────┘   │
│                                             │
└─────────────────────────────────────────────┘
```

### 5.4 HorizontalLayout

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     ┌─────┐     │
│   │ API │────▶│ Auth│────▶│Users│────▶│ DB  │────▶│Cache│     │
│   └─────┘     └─────┘     └─────┘     └─────┘     └─────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 VerticalLayout

```
┌─────────────────────────┐
│                         │
│        ┌─────────┐      │
│        │  Client │      │
│        └────┬────┘      │
│             │           │
│             ▼           │
│        ┌─────────┐      │
│        │   API   │      │
│        └────┬────┘      │
│             │           │
│             ▼           │
│        ┌─────────┐      │
│        │Database │      │
│        └─────────┘      │
│                         │
└─────────────────────────┘
```

### 5.6 RadialLayout

```
┌─────────────────────────────────────────────┐
│                                             │
│              ┌─────┐                        │
│              │  A  │                        │
│              └──┬──┘                        │
│         ┌──────┼──────┐                     │
│         │      │      │                     │
│      ┌──┴──┐┌──┴──┐┌──┴──┐                 │
│      │  B  ││ HUB ││  C  │                 │
│      └─────┘└──┬──┘└─────┘                 │
│                │                            │
│             ┌──┴──┐                         │
│             │  D  │                         │
│             └─────┘                         │
│                                             │
└─────────────────────────────────────────────┘
```

### 5.7 GraphvizLayout

Pour les diagrammes complexes (10+ nodes), utilise PyGraphviz.

```python
from viralify_diagrams import GraphvizLayout, GraphvizAlgorithm

# Algorithmes disponibles
layout = GraphvizLayout(algorithm=GraphvizAlgorithm.DOT)    # Hiérarchique (défaut)
layout = GraphvizLayout(algorithm=GraphvizAlgorithm.NEATO)  # Force-directed
layout = GraphvizLayout(algorithm=GraphvizAlgorithm.FDP)    # Force-directed optimisé
layout = GraphvizLayout(algorithm=GraphvizAlgorithm.SFDP)   # Scalable FDP (très gros graphes)
layout = GraphvizLayout(algorithm=GraphvizAlgorithm.CIRCO)  # Circulaire
layout = GraphvizLayout(algorithm=GraphvizAlgorithm.TWOPI)  # Radial Graphviz
```

**Avantages Graphviz:**
- Minimise les croisements d'edges
- Gère automatiquement les clusters
- Scale à 100+ nodes
- Algorithmes éprouvés

---

## 6. Exporters

### 6.1 SVGExporter

Export SVG statique avec groupes nommés pour animation externe.

```python
from viralify_diagrams import SVGExporter

exporter = SVGExporter(theme=None)  # None = utilise le thème du diagram

# Export vers fichier
svg_content = exporter.export(diagram, output_path="diagram.svg")

# Export en mémoire
svg_string = exporter.export(diagram)

# Récupérer les éléments pour traitement
elements = exporter.get_elements()  # List[SVGElement]
```

**Structure SVG générée:**
```xml
<svg viewBox="0 0 1920 1080">
  <g id="background">...</g>
  <g id="clusters">
    <g id="cluster_backend">...</g>
  </g>
  <g id="nodes">
    <g id="node_api" data-order="0">...</g>
    <g id="node_db" data-order="1">...</g>
  </g>
  <g id="edges">
    <g id="edge_api_db" data-order="2">...</g>
  </g>
</svg>
```

### 6.2 AnimatedSVGExporter

Export SVG avec animations CSS intégrées.

```python
from viralify_diagrams import AnimatedSVGExporter
from viralify_diagrams.exporters import AnimationConfig, AnimationType

# Configuration des animations
config = AnimationConfig(
    duration=0.5,           # Durée par élément (secondes)
    delay_between=0.3,      # Délai entre éléments
    easing="ease-out",      # Fonction d'easing CSS
    stagger=True            # Décalage progressif
)

exporter = AnimatedSVGExporter(config=config)

# Export avec types d'animation
svg = exporter.export(
    diagram,
    output_path="animated.svg",
    node_animation=AnimationType.SCALE_IN,
    edge_animation=AnimationType.DRAW,
    cluster_animation=AnimationType.FADE_IN
)

# Récupérer la timeline
timeline = exporter.get_timeline()
# [
#   {"id": "cluster_backend", "type": "cluster", "start": 0.0, "duration": 0.5},
#   {"id": "node_api", "type": "node", "start": 0.5, "duration": 0.5},
#   ...
# ]

# Durée totale
total = exporter.get_total_duration()  # 4.5 secondes

# Script de timing pour synchronisation externe
timing_script = exporter.export_timing_script()
```

**AnimationType Enum:**

| Type | Effet | Usage |
|------|-------|-------|
| `FADE_IN` | Opacité 0 → 1 | Défaut, subtil |
| `SCALE_IN` | Scale 0 → 1 depuis centre | Nodes |
| `SLIDE_LEFT` | Glisse depuis la gauche | Entrée |
| `SLIDE_RIGHT` | Glisse depuis la droite | Entrée |
| `SLIDE_UP` | Glisse depuis le haut | Entrée |
| `SLIDE_DOWN` | Glisse depuis le bas | Entrée |
| `DRAW` | Dessine le path | Edges |
| `PULSE` | Pulsation (loop) | Emphase |
| `GLOW` | Lueur (loop) | Highlight |

### 6.3 PNGFrameExporter

Export de frames PNG pour composition vidéo.

```python
from viralify_diagrams import PNGFrameExporter
from viralify_diagrams.exporters import FrameConfig

# Configuration
config = FrameConfig(
    fps=30,                 # Frames par seconde
    element_duration=0.5,   # Durée par élément
    width=1920,             # Résolution
    height=1080
)

exporter = PNGFrameExporter(config=config)

# Export des frames
frames = exporter.export(diagram, output_dir="./frames")
# Crée: frames/frame_0001.png, frame_0002.png, ...

# Manifest pour post-traitement
manifest = exporter.export_frame_manifest()
# {
#   "total_frames": 150,
#   "fps": 30,
#   "duration": 5.0,
#   "frames": [
#     {"index": 1, "time": 0.0, "visible_elements": ["cluster_backend"]},
#     ...
#   ]
# }

# Création automatique de vidéo (requiert FFmpeg)
exporter.create_video(
    output_path="diagram.mp4",
    audio_path="narration.mp3"  # Optionnel
)
```

---

## 7. Narration System

### 7.1 DiagramNarrator

Génère des scripts de narration synchronisés avec les animations.

```python
from viralify_diagrams import DiagramNarrator
from viralify_diagrams.narration import NarrationStyle

narrator = DiagramNarrator(
    style=NarrationStyle.EDUCATIONAL,
    language="en"
)

script = narrator.generate_script(
    diagram,
    element_duration=2.0,    # Durée par élément
    include_intro=True,
    include_conclusion=True
)
```

### 7.2 NarrationStyle

| Style | WPM | Description |
|-------|-----|-------------|
| `EDUCATIONAL` | 130 | Détaillé, pédagogique |
| `PROFESSIONAL` | 150 | Concis, business |
| `CASUAL` | 140 | Amical, conversationnel |
| `TECHNICAL` | 120 | Technique, précis |

### 7.3 NarrationScript

```python
# Export SRT (sous-titres)
srt_content = script.to_srt()
"""
1
00:00:00,000 --> 00:00:03,500
Let's look at Microservices Architecture.

2
00:00:04,000 --> 00:00:06,500
First, we have API Gateway.
...
"""

# Export SSML (Text-to-Speech)
ssml_content = script.to_ssml()
"""
<speak>
  <p>Let's look at <emphasis level="strong">Microservices Architecture</emphasis>.</p>
  <break time="500ms"/>
  <p>First, we have <emphasis level="strong">API Gateway</emphasis>.</p>
  ...
</speak>
"""

# Export JSON
json_content = script.to_json()
```

### 7.4 Synchronisation avec animations

```python
# 1. Générer le script
script = narrator.generate_script(diagram)

# 2. Exporter avec animations
exporter = AnimatedSVGExporter()
svg = exporter.export(diagram, "animated.svg")

# 3. Récupérer la timeline
timeline = exporter.export_timing_script()

# 4. Synchroniser le script avec la timeline
synced_script = narrator.synchronize_with_animation(script, timeline['elements'])

# Le script ajusté a les mêmes timings que les animations
```

---

## 8. Thèmes

### 8.1 Thèmes intégrés

| Thème | Background | Accent | Usage |
|-------|------------|--------|-------|
| `dark` | #1a1a2e | #e94560 | Défaut, vidéo |
| `light` | #ffffff | #3498db | Documentation |
| `gradient` | #0f0c29 | #ff6b6b | Moderne |
| `ocean` | #0a1628 | #00b4d8 | Tech, cloud |
| `corporate` | #ffffff | #0066cc | Business |
| `neon` | #0a0a0a | #ff00ff | Gaming, créatif |

### 8.2 Utiliser un thème

```python
from viralify_diagrams import Diagram, ThemeManager

# Via nom
diagram = Diagram(title="API", theme="ocean")

# Via ThemeManager
tm = ThemeManager()
theme = tm.get("corporate")
```

### 8.3 Créer un thème personnalisé

```python
from viralify_diagrams import Theme, ThemeManager
from viralify_diagrams.core.theme import ThemeColors, ThemeTypography, ThemeSpacing

# Option 1: Via JSON
my_theme = Theme.from_json('''{
    "name": "my-brand",
    "colors": {
        "background": "#0a0a1a",
        "node_fill": "#1a1a3e",
        "node_stroke": "#4a4aff",
        "edge_color": "#6a6aff",
        "text_primary": "#ffffff"
    }
}''')

# Option 2: Via Python
my_theme = Theme(
    name="my-brand",
    description="My custom theme",
    colors=ThemeColors(
        background="#0a0a1a",
        node_fill="#1a1a3e",
        node_stroke="#4a4aff"
    ),
    typography=ThemeTypography(
        font_family="Roboto, sans-serif",
        font_size_label=16
    )
)

# Enregistrer et utiliser
ThemeManager().register(my_theme)
diagram = Diagram(title="Test", theme="my-brand")

# Sauvegarder pour réutilisation
my_theme.save("my-brand.json")
```

### 8.4 Structure ThemeColors

```python
@dataclass
class ThemeColors:
    # Background
    background: str              # Fond principal
    background_secondary: str    # Fond secondaire (clusters)

    # Text
    text_primary: str            # Texte principal
    text_secondary: str          # Texte secondaire
    text_label: str              # Labels

    # Nodes
    node_fill: str               # Remplissage node
    node_stroke: str             # Bordure node
    node_stroke_width: int       # Épaisseur bordure

    # Edges
    edge_color: str              # Couleur edges
    edge_width: int              # Épaisseur edges
    edge_arrow_color: str        # Couleur flèches

    # Clusters
    cluster_fill: str            # Remplissage cluster
    cluster_stroke: str          # Bordure cluster
    cluster_stroke_width: int
    cluster_label_color: str

    # Highlights
    highlight_primary: str       # Couleur principale d'emphase
    highlight_secondary: str     # Couleur secondaire
    highlight_success: str       # Succès (vert)
    highlight_warning: str       # Warning (orange)
    highlight_error: str         # Erreur (rouge)

    # Effects
    shadow_color: str            # Ombre
    shadow_blur: int
    glow_color: str              # Lueur
    glow_blur: int
```

---

## 9. Intégration Viralify

### 9.1 Architecture d'intégration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         VIRALIFY PLATFORM                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  presentation-generator                     visual-generator                   │
│  ┌─────────────────────────┐               ┌─────────────────────────┐        │
│  │  slide_generator.py     │               │  main.py (FastAPI)      │        │
│  │                         │   HTTP POST   │                         │        │
│  │  generate_slide_image() │──────────────▶│  /api/v1/diagrams/gen   │        │
│  │                         │               │                         │        │
│  └─────────────────────────┘               │  DiagramsRenderer       │        │
│                                            │    ↓                    │        │
│                                            │  viralify-diagrams      │        │
│                                            │    ↓                    │        │
│                                            │  PNG/SVG output         │        │
│                                            └─────────────────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Usage dans visual-generator

```python
# Dans visual-generator/renderers/diagrams_renderer.py

from viralify_diagrams import (
    Diagram, Node, Edge, Cluster,
    HorizontalLayout, GraphvizLayout,
    SVGExporter, AnimatedSVGExporter,
    Theme, ThemeManager
)

class DiagramsRenderer:
    def generate_and_render(
        self,
        description: str,
        diagram_type: str = "architecture",
        theme: str = "dark",
        audience: str = "senior"
    ) -> bytes:
        # 1. Générer code Python via GPT-4o
        code = self._generate_diagram_code(description, audience)

        # 2. Exécuter le code (sandbox avec validation AST)
        diagram = self._execute_safe(code)

        # 3. Appliquer le layout
        layout = self._get_layout(diagram)
        diagram = layout.layout(diagram)

        # 4. Exporter en PNG
        exporter = SVGExporter()
        svg = exporter.export(diagram)
        png = self._svg_to_png(svg)

        return png
```

### 9.3 Thèmes utilisateur

Les utilisateurs peuvent uploader leurs propres thèmes JSON:

```python
# Frontend: upload theme.json
# Backend: stockage dans PostgreSQL ou S3

# Au moment de la génération:
custom_theme_json = user.get_custom_theme()  # Récupérer depuis DB
if custom_theme_json:
    theme = Theme.from_json(custom_theme_json)
    ThemeManager().register(theme)
    diagram = Diagram(title="...", theme=theme.name)
```

---

## 10. API Reference

### 10.1 Exports publics

```python
from viralify_diagrams import (
    # Core
    Diagram,
    Cluster,
    Node,
    Edge,

    # Themes
    Theme,
    ThemeManager,
    get_theme_manager,

    # Layouts
    GridLayout,
    HorizontalLayout,
    VerticalLayout,
    RadialLayout,
    GraphvizLayout,
    GraphvizAlgorithm,
    auto_layout,
    get_layout,

    # Exporters
    SVGExporter,
    PNGFrameExporter,
    AnimatedSVGExporter,

    # Narration
    NarrationScript,
    DiagramNarrator,
)
```

### 10.2 Enums

```python
from viralify_diagrams.core.diagram import (
    NodeShape,      # RECTANGLE, ROUNDED, CIRCLE, DIAMOND, HEXAGON, CYLINDER, CLOUD
    EdgeStyle,      # SOLID, DASHED, DOTTED
    EdgeDirection,  # FORWARD, BACKWARD, BOTH, NONE
)

from viralify_diagrams.narration import (
    NarrationStyle,  # EDUCATIONAL, PROFESSIONAL, CASUAL, TECHNICAL
)

from viralify_diagrams.exporters import (
    AnimationType,   # FADE_IN, SCALE_IN, SLIDE_*, DRAW, PULSE, GLOW
)
```

### 10.3 Exemple complet

```python
from viralify_diagrams import (
    Diagram, Node, Cluster,
    HorizontalLayout,
    AnimatedSVGExporter,
    DiagramNarrator,
    Theme, ThemeManager
)
from viralify_diagrams.core.diagram import NodeShape
from viralify_diagrams.exporters import AnimationConfig, AnimationType
from viralify_diagrams.narration import NarrationStyle

# 1. Créer le diagramme
diagram = Diagram(
    title="Microservices Architecture",
    description="A modern microservices setup",
    theme="ocean"
)

# 2. Ajouter les nodes
api = Node(label="API Gateway", shape=NodeShape.ROUNDED, description="Entry point")
auth = Node(label="Auth Service", shape=NodeShape.ROUNDED, description="JWT authentication")
users = Node(label="User Service", shape=NodeShape.ROUNDED, description="User management")
db = Node(label="PostgreSQL", shape=NodeShape.CYLINDER, description="Main database")
cache = Node(label="Redis", shape=NodeShape.HEXAGON, description="Session cache")

for node in [api, auth, users, db, cache]:
    diagram.add_node(node)

# 3. Connecter les nodes
diagram.connect(api.id, auth.id, label="JWT")
diagram.connect(api.id, users.id, label="gRPC")
diagram.connect(auth.id, db.id, label="SQL")
diagram.connect(users.id, db.id, label="SQL")
diagram.connect(auth.id, cache.id, label="Sessions")

# 4. Grouper en cluster
backend = Cluster(label="Backend", nodes=[auth.id, users.id])
data = Cluster(label="Data Layer", nodes=[db.id, cache.id])
diagram.add_cluster(backend)
diagram.add_cluster(data)

# 5. Appliquer le layout
layout = HorizontalLayout()
diagram = layout.layout(diagram)

# 6. Assigner l'ordre d'animation
diagram.assign_animation_order()

# 7. Exporter avec animations
config = AnimationConfig(duration=0.5, delay_between=0.3)
exporter = AnimatedSVGExporter(config=config)
svg = exporter.export(
    diagram,
    output_path="architecture.svg",
    node_animation=AnimationType.SCALE_IN,
    edge_animation=AnimationType.DRAW
)

# 8. Générer la narration
narrator = DiagramNarrator(style=NarrationStyle.EDUCATIONAL)
script = narrator.generate_script(diagram)

# 9. Synchroniser avec l'animation
timeline = exporter.export_timing_script()
synced = narrator.synchronize_with_animation(script, timeline['elements'])

# 10. Exporter les formats
print(synced.to_srt())   # Sous-titres
print(synced.to_ssml())  # Pour TTS
```

---

## Annexes

### A. Dépendances système

| Outil | Usage | Installation |
|-------|-------|--------------|
| Graphviz | Layout avancé | `apt install graphviz` |
| FFmpeg | Création vidéo | `apt install ffmpeg` |
| Cairo | Rendu PNG | `apt install libcairo2-dev` |

### B. Variables d'environnement

```bash
# Chemin vers les thèmes personnalisés
VIRALIFY_DIAGRAMS_THEMES_DIR=/app/themes

# Backend de layout par défaut
VIRALIFY_DIAGRAMS_DEFAULT_LAYOUT=horizontal
```

### C. Limitations

| Limitation | Valeur | Raison |
|------------|--------|--------|
| Max nodes (sans simplification) | 10 | Lisibilité vidéo |
| Max nodes (Graphviz) | 100+ | Performance |
| Max label length | 20 chars | Auto-truncated |
| Max cluster depth | 3 | Complexité visuelle |

---

*Document généré pour Viralify Platform - Février 2026*
