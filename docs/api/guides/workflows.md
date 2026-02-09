# Workflows

Guide des workflows de génération Viralify.

## Workflow 1: Génération de Cours Standard

### Étapes

```
1. Preview Outline     →  Valider la structure
2. Start Generation    →  Lancer le job
3. Poll Status         →  Suivre la progression
4. Download Results    →  Récupérer les vidéos
```

### Exemple Complet

```python
import requests
import time

API_KEY = "your_api_key"
BASE_URL = "https://api.viralify.io"
headers = {"Authorization": f"Bearer {API_KEY}"}

# 1. Preview de l'outline
preview_response = requests.post(
    f"{BASE_URL}/api/v1/courses/preview-outline",
    headers=headers,
    json={
        "topic": "Docker Fundamentals",
        "difficulty_start": "beginner",
        "difficulty_end": "intermediate",
        "structure": {
            "number_of_sections": 4,
            "lectures_per_section": 3
        },
        "context": {
            "category": "tech",
            "profile_niche": "DevOps",
            "profile_audience_level": "junior developers"
        }
    }
)
outline = preview_response.json()
print(f"Preview: {outline['title']} - {outline['total_lectures']} lectures")

# 2. Lancer la génération
generate_response = requests.post(
    f"{BASE_URL}/api/v1/courses/generate",
    headers=headers,
    json={
        "topic": "Docker Fundamentals",
        "difficulty_start": "beginner",
        "difficulty_end": "intermediate",
        "structure": {
            "number_of_sections": 4,
            "lectures_per_section": 3
        },
        "context": {
            "category": "tech"
        },
        "language": "en",
        "quiz_config": {
            "enabled": True,
            "frequency": "per_section"
        },
        "title_style": "engaging"
    }
)
job = generate_response.json()
job_id = job["job_id"]

# 3. Polling du statut
while True:
    status_response = requests.get(
        f"{BASE_URL}/api/v1/courses/jobs/{job_id}",
        headers=headers
    )
    status = status_response.json()

    print(f"[{status['status']}] Progress: {status['progress']:.1f}%")

    if status["status"] == "completed":
        videos = status["output_urls"]["videos"]
        zip_url = status["output_urls"]["zip"]
        print(f"\nCompleted! {len(videos)} videos generated")
        print(f"ZIP: {zip_url}")
        break
    elif status["status"] == "failed":
        print(f"Failed: {status['error']}")
        break

    time.sleep(15)

# 4. Télécharger le ZIP
zip_response = requests.get(zip_url, headers=headers)
with open("course.zip", "wb") as f:
    f.write(zip_response.content)
```

## Workflow 2: Génération avec Documents RAG

### Étapes

```
1. Upload Documents    →  Uploader les sources
2. Wait Processing     →  Attendre l'indexation
3. Start Generation    →  Lancer avec document_ids
4. Poll & Download     →  Récupérer les vidéos
```

### Exemple

```python
# 1. Upload des documents
documents = []

for file_path in ["chapter1.pdf", "chapter2.pdf", "references.docx"]:
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/api/v1/documents/upload",
            headers=headers,
            files={"file": f},
            data={
                "user_id": "user_123",
                "pedagogical_role": "theory"
            }
        )
        doc = response.json()
        documents.append(doc["id"])
        print(f"Uploaded: {doc['filename']} -> {doc['id']}")

# 2. Attendre le processing
for doc_id in documents:
    while True:
        response = requests.get(
            f"{BASE_URL}/api/v1/documents/{doc_id}",
            headers=headers
        )
        doc = response.json()
        if doc["status"] == "ready":
            print(f"Document {doc_id} ready")
            break
        time.sleep(2)

# 3. Générer le cours avec RAG
response = requests.post(
    f"{BASE_URL}/api/v1/courses/generate",
    headers=headers,
    json={
        "topic": "Advanced Database Design",
        "document_ids": documents,  # Documents RAG
        "difficulty_start": "intermediate",
        "difficulty_end": "expert",
        "structure": {
            "number_of_sections": 5,
            "lectures_per_section": 4
        }
    }
)
job_id = response.json()["job_id"]

# 4. Polling (même que workflow 1)
# ...
```

## Workflow 3: Voice Cloning

### Étapes

```
1. Create Profile      →  Créer le profil vocal
2. Upload Samples      →  Uploader les échantillons
3. Start Training      →  Lancer l'entraînement
4. Generate Speech     →  Utiliser la voix clonée
```

### Exemple

```python
# 1. Créer le profil vocal
response = requests.post(
    f"{BASE_URL}/api/v1/voice/profiles",
    headers=headers,
    json={
        "user_id": "user_123",
        "name": "My Professional Voice",
        "description": "Voice for tech courses",
        "gender": "male",
        "accent": "american"
    }
)
profile = response.json()
profile_id = profile["id"]

# 2. Upload des échantillons (minimum 30 secondes total)
samples = ["sample1.mp3", "sample2.mp3", "sample3.mp3"]

for sample_path in samples:
    with open(sample_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/api/v1/voice/profiles/{profile_id}/samples",
            headers=headers,
            files={"file": f}
        )
        sample = response.json()
        print(f"Uploaded: {sample['filename']} ({sample['duration']:.1f}s)")

# 3. Démarrer l'entraînement (consentement requis)
response = requests.post(
    f"{BASE_URL}/api/v1/voice/profiles/{profile_id}/train",
    headers=headers,
    json={"consent": True}  # Consentement explicite
)
print("Training started...")

# Attendre la fin de l'entraînement
while True:
    response = requests.get(
        f"{BASE_URL}/api/v1/voice/profiles/{profile_id}/training-status",
        headers=headers
    )
    status = response.json()
    print(f"Training: {status['status']} - {status.get('progress', 0)}%")

    if status["status"] == "ready":
        print("Voice ready!")
        break
    elif status["status"] == "failed":
        print(f"Failed: {status.get('error')}")
        break

    time.sleep(10)

# 4. Générer du speech avec la voix clonée
response = requests.post(
    f"{BASE_URL}/api/v1/voice/profiles/{profile_id}/generate",
    headers=headers,
    json={
        "text": "Welcome to this course on Docker fundamentals.",
        "stability": 0.5,
        "similarity_boost": 0.75
    }
)
audio = response.json()
print(f"Audio generated: {audio['audio_url']}")

# Utiliser dans la génération de cours
response = requests.post(
    f"{BASE_URL}/api/v1/courses/generate",
    headers=headers,
    json={
        "topic": "Docker Fundamentals",
        "voice_profile_id": profile_id,  # Utiliser la voix clonée
        # ... autres paramètres
    }
)
```

## Workflow 4: Édition Vidéo Post-Génération

### Étapes

```
1. Create Project      →  Créer un projet éditeur
2. Import Segments     →  Importer les vidéos générées
3. Edit Timeline       →  Modifier la timeline
4. Add Overlays        →  Ajouter texte/images
5. Render              →  Exporter la vidéo finale
```

### Exemple

```python
# 1. Créer un projet d'édition depuis un cours
response = requests.post(
    f"{BASE_URL}/api/v1/editor/projects",
    headers=headers,
    json={
        "user_id": "user_123",
        "name": "Docker Course - Edited",
        "course_job_id": job_id,  # Importer depuis le cours
        "resolution": "1080p",
        "fps": 30
    }
)
project = response.json()
project_id = project["id"]

# 2. Voir les segments importés
response = requests.get(
    f"{BASE_URL}/api/v1/editor/projects/{project_id}",
    headers=headers
)
project = response.json()
print(f"Segments: {len(project['segments'])}")

# 3. Modifier un segment (trim)
segment_id = project["segments"][0]["id"]
response = requests.patch(
    f"{BASE_URL}/api/v1/editor/projects/{project_id}/segments/{segment_id}",
    headers=headers,
    json={
        "trim_start": 2.0,    # Couper les 2 premières secondes
        "trim_end": 1.5,      # Couper les 1.5 dernières secondes
        "volume": 0.9,        # 90% du volume
        "transition_in": "fade",
        "transition_out": "fade"
    }
)

# 4. Ajouter un overlay texte
response = requests.post(
    f"{BASE_URL}/api/v1/editor/projects/{project_id}/overlays/text",
    headers=headers,
    json={
        "text": "Introduction",
        "x": 50,
        "y": 50,
        "font_size": 48,
        "font_color": "#FFFFFF",
        "start_time": 0,
        "end_time": 5
    }
)

# 5. Réordonner les segments
segment_ids = [s["id"] for s in project["segments"]]
segment_ids[0], segment_ids[1] = segment_ids[1], segment_ids[0]  # Swap

response = requests.post(
    f"{BASE_URL}/api/v1/editor/projects/{project_id}/segments/reorder",
    headers=headers,
    json={"segment_ids": segment_ids}
)

# 6. Lancer le rendu
response = requests.post(
    f"{BASE_URL}/api/v1/editor/projects/{project_id}/render",
    headers=headers,
    json={
        "quality": "high",
        "format": "mp4"
    }
)
render_job_id = response.json()["job_id"]

# Attendre le rendu
while True:
    response = requests.get(
        f"{BASE_URL}/api/v1/editor/render-jobs/{render_job_id}",
        headers=headers
    )
    status = response.json()

    if status["status"] == "completed":
        print(f"Video ready: {status['output_url']}")
        break

    time.sleep(5)
```

## Workflow 5: Progressive Download (V3)

### Avantages

- Télécharger les leçons dès qu'elles sont prêtes
- Pas besoin d'attendre la vidéo finale
- Retry individuel des leçons échouées

### Exemple

```python
# Lancer génération V3
response = requests.post(
    f"{BASE_URL}/api/v1/presentations/generate/v3",
    headers=headers,
    json={
        "topic": "Kubernetes Architecture",
        "num_slides": 15,
        "duration": 600,
        "language": "en"
    }
)
job_id = response.json()["job_id"]

# Récupérer les leçons au fur et à mesure
downloaded = set()

while True:
    # Vérifier les leçons disponibles
    response = requests.get(
        f"{BASE_URL}/api/v1/presentations/jobs/v3/{job_id}/lessons",
        headers=headers
    )
    lessons = response.json()

    # Télécharger les nouvelles leçons
    for lesson in lessons["lessons"]:
        if lesson["status"] == "ready" and lesson["scene_index"] not in downloaded:
            print(f"Downloading lesson {lesson['scene_index']}: {lesson['title']}")
            # Download video...
            downloaded.add(lesson["scene_index"])

    # Vérifier si terminé
    if lessons["status"] == "completed":
        print(f"All lessons complete! Final: {lessons['final_video_url']}")
        break

    time.sleep(10)

# Gérer les erreurs
response = requests.get(
    f"{BASE_URL}/api/v1/presentations/jobs/v3/{job_id}/errors",
    headers=headers
)
errors = response.json()["errors"]

for error in errors:
    print(f"Error in scene {error['scene_index']}: {error['error']}")

    # Modifier le contenu si nécessaire
    requests.patch(
        f"{BASE_URL}/api/v1/presentations/jobs/v3/{job_id}/lessons/{error['scene_index']}",
        headers=headers,
        json={
            "voiceover": error["editable_content"]["voiceover"] + " (simplified)"
        }
    )

    # Retry
    requests.post(
        f"{BASE_URL}/api/v1/presentations/jobs/v3/{job_id}/lessons/{error['scene_index']}/retry",
        headers=headers
    )
```

## Bonnes Pratiques

### 1. Gestion des erreurs

```python
def safe_request(method, url, **kwargs):
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            # Rate limited - wait and retry
            time.sleep(60)
            return safe_request(method, url, **kwargs)
        raise
```

### 2. Timeouts appropriés

```python
# Jobs longs: timeout élevé
response = requests.get(
    f"{BASE_URL}/api/v1/courses/jobs/{job_id}",
    headers=headers,
    timeout=30
)
```

### 3. Polling intelligent

```python
def poll_with_backoff(job_id, max_wait=3600):
    start = time.time()
    interval = 5

    while time.time() - start < max_wait:
        status = get_job_status(job_id)

        if status["status"] in ["completed", "failed"]:
            return status

        time.sleep(interval)
        interval = min(interval * 1.5, 60)  # Backoff jusqu'à 60s

    raise TimeoutError("Job timeout")
```

### 4. Webhooks pour notifications

```python
# Configurer un webhook au lieu de polling
requests.post(
    f"{BASE_URL}/api/v1/webhooks",
    headers=headers,
    json={
        "url": "https://your-server.com/webhook",
        "events": ["course.completed", "course.failed"]
    }
)
```
