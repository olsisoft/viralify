# Getting Started with Viralify API

Ce guide vous aidera à démarrer avec l'API Viralify pour générer des cours vidéo éducatifs.

## Prérequis

- Compte Viralify avec API key
- Python 3.9+ ou Node.js 18+
- Docker (pour développement local)

## Installation

### Option 1: Utilisation directe de l'API

```bash
# Définir votre API key
export VIRALIFY_API_KEY="your_api_key_here"
export VIRALIFY_BASE_URL="https://api.viralify.io"
```

### Option 2: SDK Python

```bash
pip install viralify-sdk
```

```python
from viralify import ViralifyClient

client = ViralifyClient(api_key="your_api_key")
```

### Option 3: SDK JavaScript

```bash
npm install @viralify/sdk
```

```javascript
import { ViralifyClient } from '@viralify/sdk';

const client = new ViralifyClient({ apiKey: 'your_api_key' });
```

## Premier Cours en 5 Minutes

### Étape 1: Créer un outline

```python
import requests

# Preview du curriculum
response = requests.post(
    "https://api.viralify.io/api/v1/courses/preview-outline",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "topic": "Introduction to Python Programming",
        "difficulty_start": "beginner",
        "difficulty_end": "intermediate",
        "structure": {
            "number_of_sections": 3,
            "lectures_per_section": 3
        },
        "context": {
            "category": "tech",
            "profile_audience_level": "beginners"
        }
    }
)

outline = response.json()
print(f"Course: {outline['title']}")
print(f"Sections: {outline['section_count']}")
print(f"Total Lectures: {outline['total_lectures']}")
```

### Étape 2: Lancer la génération

```python
# Démarrer la génération
response = requests.post(
    "https://api.viralify.io/api/v1/courses/generate",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "topic": "Introduction to Python Programming",
        "difficulty_start": "beginner",
        "difficulty_end": "intermediate",
        "structure": {
            "number_of_sections": 3,
            "lectures_per_section": 3
        },
        "context": {
            "category": "tech"
        },
        "language": "en",
        "quiz_config": {
            "enabled": True,
            "frequency": "per_section",
            "questions_per_quiz": 5
        }
    }
)

job = response.json()
job_id = job["job_id"]
print(f"Job started: {job_id}")
```

### Étape 3: Suivre la progression

```python
import time

while True:
    response = requests.get(
        f"https://api.viralify.io/api/v1/courses/jobs/{job_id}",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    status = response.json()

    print(f"Status: {status['status']} - Progress: {status['progress']}%")

    if status["status"] == "completed":
        print(f"Course ready!")
        print(f"Videos: {status['output_urls']['videos']}")
        print(f"ZIP: {status['output_urls']['zip']}")
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break

    time.sleep(10)
```

## Utilisation avec Documents RAG

### Upload de documents source

```python
# Upload un PDF
with open("my_document.pdf", "rb") as f:
    response = requests.post(
        "https://api.viralify.io/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {API_KEY}"},
        files={"file": f},
        data={
            "user_id": "user_123",
            "pedagogical_role": "theory"
        }
    )

document = response.json()
document_id = document["id"]
print(f"Document uploaded: {document_id}")
```

### Génération avec RAG

```python
# Générer un cours basé sur les documents
response = requests.post(
    "https://api.viralify.io/api/v1/courses/generate",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "topic": "Machine Learning Fundamentals",
        "document_ids": [document_id],  # Documents source
        "difficulty_start": "intermediate",
        "difficulty_end": "advanced",
        "structure": {
            "number_of_sections": 4,
            "lectures_per_section": 3
        }
    }
)
```

## Modes de Génération

### Mode RAG (Recommandé avec documents)

Utilise vos documents comme source principale (90%+ du contenu provient des documents).

```python
{
    "topic": "Your Topic",
    "document_ids": ["doc_1", "doc_2"],  # Requis
    "generation_mode": "rag"
}
```

### Mode MAESTRO (Sans documents)

Génération 100% IA avec calibration 4D de difficulté.

```python
{
    "topic": "Your Topic",
    "generation_mode": "maestro"
}
```

## Configuration des Quiz

```python
{
    "quiz_config": {
        "enabled": True,
        "frequency": "per_section",  # per_lecture, per_section, end_only, custom
        "custom_frequency": 3,        # Toutes les 3 lectures (si frequency=custom)
        "questions_per_quiz": 5,
        "passing_score": 70,
        "show_explanations": True
    }
}
```

## Langues Supportées

| Code | Langue |
|------|--------|
| en | English |
| fr | Français |
| es | Español |
| de | Deutsch |
| pt | Português |
| it | Italiano |
| nl | Nederlands |
| pl | Polski |
| ru | Русский |
| zh | 中文 |

## Styles de Titres

| Style | Description | Exemple |
|-------|-------------|---------|
| `engaging` | Dynamique, accrocheur | "Unlock the Power of Functions" |
| `corporate` | Professionnel, formel | "Function Implementation Best Practices" |
| `expert` | Technique, précis | "Advanced Function Patterns in Python 3.11" |
| `mentor` | Pédagogique, chaleureux | "Let's Master Functions Together" |
| `storyteller` | Narratif | "The Journey from Simple to Complex Functions" |
| `direct` | Clair, concis | "Python Functions: Core Concepts" |

## Prochaines Étapes

- [Authentication](./authentication.md) - Sécuriser vos appels API
- [Workflows](./workflows.md) - Workflows avancés
- [API Reference](../openapi/) - Documentation complète des endpoints
- [Examples](../examples/) - Exemples de code
