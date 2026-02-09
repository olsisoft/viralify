# cURL Examples

Exemples d'utilisation de l'API Viralify avec cURL.

## Configuration

```bash
# Définir les variables d'environnement
export VIRALIFY_API_KEY="your_api_key_here"
export VIRALIFY_BASE_URL="https://api.viralify.io"
```

## Health Check

```bash
curl -X GET "$VIRALIFY_BASE_URL/health"
```

## Course Generation

### 1. Preview Outline

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/courses/preview-outline" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Introduction to Docker",
    "difficulty_start": "beginner",
    "difficulty_end": "intermediate",
    "structure": {
      "number_of_sections": 4,
      "lectures_per_section": 3
    },
    "context": {
      "category": "tech"
    }
  }'
```

### 2. Start Course Generation

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/courses/generate" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Introduction to Docker",
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
      "enabled": true,
      "frequency": "per_section"
    },
    "title_style": "engaging"
  }'
```

Response:
```json
{
  "job_id": "abc123-def456",
  "status": "queued",
  "message": "Course generation started"
}
```

### 3. Check Job Status

```bash
JOB_ID="abc123-def456"

curl -X GET "$VIRALIFY_BASE_URL/api/v1/courses/jobs/$JOB_ID" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

### 4. List All Jobs

```bash
curl -X GET "$VIRALIFY_BASE_URL/api/v1/courses/jobs?status=completed&limit=10" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

### 5. Cancel Job

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/courses/jobs/$JOB_ID/cancel" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

## Document Upload (RAG)

### 1. Upload PDF

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/documents/upload" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -F "file=@/path/to/document.pdf" \
  -F "user_id=user_123" \
  -F "pedagogical_role=theory"
```

### 2. Upload from URL

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/documents/upload-url" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/document.pdf",
    "user_id": "user_123",
    "pedagogical_role": "reference"
  }'
```

### 3. List Documents

```bash
curl -X GET "$VIRALIFY_BASE_URL/api/v1/documents?user_id=user_123" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

### 4. RAG Query

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/documents/query" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main concepts?",
    "document_ids": ["doc_123", "doc_456"],
    "max_results": 5
  }'
```

## Presentation Generation (V3)

### 1. Generate Presentation

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/presentations/generate/v3" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Kubernetes Architecture",
    "num_slides": 12,
    "duration": 600,
    "style": "modern",
    "language": "en",
    "target_audience": "senior developers"
  }'
```

### 2. Get Job Status (V3)

```bash
curl -X GET "$VIRALIFY_BASE_URL/api/v1/presentations/jobs/v3/$JOB_ID" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

### 3. Get Available Lessons (Progressive Download)

```bash
curl -X GET "$VIRALIFY_BASE_URL/api/v1/presentations/jobs/v3/$JOB_ID/lessons" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

### 4. Retry Failed Lesson

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/presentations/jobs/v3/$JOB_ID/lessons/5/retry" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

## Voice Cloning

### 1. Create Voice Profile

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/voice/profiles" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "name": "My Voice",
    "description": "Professional voice for courses",
    "gender": "male"
  }'
```

### 2. Upload Voice Sample

```bash
PROFILE_ID="profile_123"

curl -X POST "$VIRALIFY_BASE_URL/api/v1/voice/profiles/$PROFILE_ID/samples" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -F "file=@/path/to/sample.mp3"
```

### 3. Start Training

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/voice/profiles/$PROFILE_ID/train" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"consent": true}'
```

### 4. Generate Cloned Speech

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/voice/profiles/$PROFILE_ID/generate" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to this course on Docker.",
    "stability": 0.5,
    "similarity_boost": 0.75
  }'
```

## Configuration

### Get Categories

```bash
curl -X GET "$VIRALIFY_BASE_URL/api/v1/courses/config/categories" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

### Get Elements by Category

```bash
curl -X GET "$VIRALIFY_BASE_URL/api/v1/courses/config/elements/tech" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

### AI-Suggested Elements

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/courses/config/suggest-elements" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Machine Learning with Python",
    "category": "tech"
  }'
```

## Analytics

### Dashboard

```bash
curl -X GET "$VIRALIFY_BASE_URL/api/v1/analytics/dashboard?user_id=user_123&time_range=30d" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY"
```

## Translation

### Translate Text

```bash
curl -X POST "$VIRALIFY_BASE_URL/api/v1/translation/translate" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to this course on Docker",
    "target_language": "fr"
  }'
```

## Polling Script

```bash
#!/bin/bash
# poll_job.sh - Poll job status until completion

JOB_ID=$1
INTERVAL=${2:-15}

while true; do
  STATUS=$(curl -s -X GET "$VIRALIFY_BASE_URL/api/v1/courses/jobs/$JOB_ID" \
    -H "Authorization: Bearer $VIRALIFY_API_KEY" | jq -r '.status')

  PROGRESS=$(curl -s -X GET "$VIRALIFY_BASE_URL/api/v1/courses/jobs/$JOB_ID" \
    -H "Authorization: Bearer $VIRALIFY_API_KEY" | jq -r '.progress')

  echo "[$STATUS] Progress: $PROGRESS%"

  if [ "$STATUS" == "completed" ] || [ "$STATUS" == "failed" ]; then
    curl -s -X GET "$VIRALIFY_BASE_URL/api/v1/courses/jobs/$JOB_ID" \
      -H "Authorization: Bearer $VIRALIFY_API_KEY" | jq
    break
  fi

  sleep $INTERVAL
done
```

Usage:
```bash
chmod +x poll_job.sh
./poll_job.sh abc123-def456 10
```
