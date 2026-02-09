# Error Handling

Guide de gestion des erreurs de l'API Viralify.

## Format des Erreurs

Toutes les erreurs suivent un format standardisé:

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "field": "additional_context"
  }
}
```

## Codes HTTP

| Code | Signification | Action |
|------|---------------|--------|
| 200 | Succès | - |
| 201 | Créé | - |
| 202 | Accepté (job asynchrone) | Polling requis |
| 400 | Bad Request | Vérifier les paramètres |
| 401 | Non authentifié | Vérifier l'API key |
| 403 | Interdit | Vérifier les permissions |
| 404 | Non trouvé | Vérifier l'ID |
| 409 | Conflit | Resource déjà existante |
| 413 | Payload trop large | Réduire la taille |
| 415 | Type non supporté | Vérifier le format |
| 422 | Validation échouée | Corriger les données |
| 429 | Rate limit | Attendre et réessayer |
| 500 | Erreur serveur | Contacter le support |
| 503 | Service indisponible | Réessayer plus tard |

## Erreurs Courantes

### Authentication (401)

```json
{
  "error": "invalid_api_key",
  "message": "The API key provided is invalid or expired"
}
```

**Solutions:**
- Vérifier que la clé est correcte
- Vérifier que la clé n'est pas expirée
- Utiliser le bon header: `Authorization: Bearer YOUR_KEY`

### Validation (422)

```json
{
  "error": "validation_error",
  "message": "Request validation failed",
  "details": [
    {
      "loc": ["body", "topic"],
      "msg": "field required",
      "type": "value_error.missing"
    },
    {
      "loc": ["body", "difficulty_start"],
      "msg": "value is not a valid enumeration member",
      "type": "type_error.enum"
    }
  ]
}
```

**Solutions:**
- Vérifier tous les champs requis
- Utiliser les valeurs valides pour les enums
- Respecter les types (string, integer, etc.)

### Rate Limiting (429)

```json
{
  "error": "rate_limit_exceeded",
  "message": "Too many requests. Please wait before retrying.",
  "retry_after": 60
}
```

**Solutions:**
```python
import time

response = requests.get(url, headers=headers)
if response.status_code == 429:
    retry_after = int(response.headers.get('Retry-After', 60))
    time.sleep(retry_after)
    # Retry...
```

### Resource Not Found (404)

```json
{
  "error": "not_found",
  "message": "Job with ID 'abc123' not found"
}
```

**Solutions:**
- Vérifier l'ID de la resource
- Vérifier que la resource n'a pas été supprimée
- Vérifier les permissions d'accès

### File Upload (413, 415)

```json
{
  "error": "file_too_large",
  "message": "File size exceeds maximum allowed (50MB)"
}
```

```json
{
  "error": "unsupported_media_type",
  "message": "File type 'application/x-executable' is not supported"
}
```

**Limites:**

| Type | Taille max | Formats acceptés |
|------|------------|------------------|
| Documents | 50 MB | PDF, DOCX, PPTX, XLSX, TXT, MD |
| Audio | 50 MB | MP3, WAV, M4A, OGG, FLAC |
| Vidéo | 500 MB | MP4, MOV, AVI, MKV, WEBM |
| Images | 10 MB | JPG, PNG, GIF, WEBP |

## Erreurs Spécifiques aux Services

### Course Generator

#### `insufficient_rag_context`

```json
{
  "error": "insufficient_rag_context",
  "message": "Insufficient source content: 320 tokens (minimum: 500)",
  "details": {
    "tokens_found": 320,
    "minimum_required": 500,
    "suggestions": [
      "Upload more documents covering the topic",
      "Ensure documents contain text (not just images)",
      "Check that documents are relevant to the topic"
    ]
  }
}
```

**Solution:** Uploader plus de documents ou des documents plus complets.

#### `generation_timeout`

```json
{
  "error": "generation_timeout",
  "message": "Course generation timed out after 30 minutes",
  "details": {
    "stage": "generating_lectures",
    "completed_lectures": 8,
    "total_lectures": 12
  }
}
```

**Solution:** Réduire le nombre de sections/lectures ou réessayer.

### Presentation Generator

#### `slide_render_failed`

```json
{
  "error": "slide_render_failed",
  "message": "Failed to render slide 5",
  "details": {
    "scene_index": 5,
    "reason": "Code syntax error in Python block"
  }
}
```

**Solution:** Utiliser l'endpoint de retry avec contenu modifié.

#### `tts_generation_failed`

```json
{
  "error": "tts_generation_failed",
  "message": "Text-to-speech generation failed",
  "details": {
    "provider": "elevenlabs",
    "reason": "Voice model not found"
  }
}
```

### Media Generator

#### `voice_training_failed`

```json
{
  "error": "voice_training_failed",
  "message": "Voice training failed",
  "details": {
    "reason": "Insufficient audio quality",
    "sample_issues": [
      {"sample_id": "s1", "issue": "Too much background noise"},
      {"sample_id": "s2", "issue": "Audio too quiet"}
    ]
  }
}
```

**Solution:** Re-enregistrer les échantillons dans un environnement calme.

#### `consent_required`

```json
{
  "error": "consent_required",
  "message": "Explicit consent is required for voice cloning",
  "details": {
    "consent_type": "voice_cloning",
    "required_fields": ["consent"]
  }
}
```

**Solution:** Ajouter `"consent": true` dans la requête.

## Gestion des Erreurs

### Python

```python
import requests
from requests.exceptions import HTTPError, Timeout, ConnectionError

def api_request(method, endpoint, **kwargs):
    """Make an API request with error handling."""
    try:
        response = requests.request(
            method,
            f"{BASE_URL}{endpoint}",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30,
            **kwargs
        )
        response.raise_for_status()
        return response.json()

    except HTTPError as e:
        error_body = e.response.json()
        error_code = error_body.get("error", "unknown")

        if e.response.status_code == 401:
            raise AuthenticationError(error_body["message"])

        elif e.response.status_code == 422:
            raise ValidationError(error_body["details"])

        elif e.response.status_code == 429:
            retry_after = int(e.response.headers.get("Retry-After", 60))
            raise RateLimitError(f"Rate limited. Retry after {retry_after}s")

        elif e.response.status_code >= 500:
            raise ServerError(error_body["message"])

        raise APIError(error_code, error_body["message"])

    except Timeout:
        raise APIError("timeout", "Request timed out")

    except ConnectionError:
        raise APIError("connection_error", "Could not connect to API")


class APIError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

class AuthenticationError(APIError):
    pass

class ValidationError(APIError):
    pass

class RateLimitError(APIError):
    pass

class ServerError(APIError):
    pass
```

### JavaScript/TypeScript

```typescript
interface APIError {
  error: string;
  message: string;
  details?: Record<string, any>;
}

async function apiRequest<T>(
  method: string,
  endpoint: string,
  body?: any
): Promise<T> {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    method,
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const error: APIError = await response.json();

    switch (response.status) {
      case 401:
        throw new AuthenticationError(error.message);
      case 422:
        throw new ValidationError(error.message, error.details);
      case 429:
        const retryAfter = response.headers.get('Retry-After') || '60';
        throw new RateLimitError(parseInt(retryAfter));
      case 500:
      case 503:
        throw new ServerError(error.message);
      default:
        throw new APIError(error.error, error.message);
    }
  }

  return response.json();
}

class APIError extends Error {
  constructor(public code: string, message: string) {
    super(message);
  }
}

class AuthenticationError extends APIError {
  constructor(message: string) {
    super('authentication_error', message);
  }
}

class ValidationError extends APIError {
  constructor(message: string, public details?: any) {
    super('validation_error', message);
  }
}

class RateLimitError extends APIError {
  constructor(public retryAfter: number) {
    super('rate_limit', `Rate limited. Retry after ${retryAfter}s`);
  }
}

class ServerError extends APIError {
  constructor(message: string) {
    super('server_error', message);
  }
}
```

## Retry Strategy

### Exponential Backoff

```python
import time
import random

def retry_with_backoff(func, max_retries=5, base_delay=1):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            delay = e.retry_after
        except (ServerError, ConnectionError):
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
        except Exception:
            raise  # Don't retry other errors

        if attempt == max_retries - 1:
            raise

        print(f"Attempt {attempt + 1} failed. Retrying in {delay:.1f}s...")
        time.sleep(delay)
```

### Circuit Breaker

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = None
        self.state = "closed"

    def call(self, func):
        if self.state == "open":
            if datetime.now() - self.last_failure > timedelta(seconds=self.reset_timeout):
                self.state = "half-open"
            else:
                raise CircuitBreakerOpen("Service temporarily unavailable")

        try:
            result = func()
            self.failures = 0
            self.state = "closed"
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = datetime.now()

            if self.failures >= self.failure_threshold:
                self.state = "open"

            raise

class CircuitBreakerOpen(Exception):
    pass
```

## Logging des Erreurs

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("viralify_api")

def log_api_error(error, context=None):
    """Log API errors with context."""
    logger.error(
        "API Error",
        extra={
            "error_code": getattr(error, 'code', 'unknown'),
            "error_message": str(error),
            "context": context
        }
    )
```

## Support

Si vous rencontrez des erreurs non documentées:

1. **Vérifier les logs** du service concerné
2. **Consulter le status page**: https://status.viralify.io
3. **Ouvrir un ticket**: support@viralify.io
4. **GitHub Issues**: https://github.com/olsisoft/viralify/issues

Inclure dans votre rapport:
- Code d'erreur complet
- Request body (sans données sensibles)
- Job ID si applicable
- Timestamp de l'erreur
