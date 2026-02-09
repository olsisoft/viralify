# Authentication

L'API Viralify utilise des API keys pour l'authentification.

## Obtenir une API Key

1. Connectez-vous à votre [Dashboard Viralify](https://app.viralify.io/dashboard)
2. Allez dans **Settings > API Keys**
3. Cliquez sur **Generate New Key**
4. Copiez et stockez la clé en lieu sûr

> **Important**: La clé n'est affichée qu'une seule fois. Stockez-la dans un gestionnaire de secrets.

## Utilisation de l'API Key

### Header Authorization (Recommandé)

```bash
curl -X GET "https://api.viralify.io/api/v1/courses/jobs" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Query Parameter (Déconseillé)

```bash
curl -X GET "https://api.viralify.io/api/v1/courses/jobs?api_key=YOUR_API_KEY"
```

## Exemples par Langage

### Python

```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "https://api.viralify.io"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.get(
    f"{BASE_URL}/api/v1/courses/jobs",
    headers=headers
)
```

### JavaScript/TypeScript

```javascript
const API_KEY = process.env.VIRALIFY_API_KEY;
const BASE_URL = 'https://api.viralify.io';

const response = await fetch(`${BASE_URL}/api/v1/courses/jobs`, {
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  }
});
```

### cURL

```bash
export VIRALIFY_API_KEY="your_api_key_here"

curl -X GET "https://api.viralify.io/api/v1/courses/jobs" \
  -H "Authorization: Bearer $VIRALIFY_API_KEY" \
  -H "Content-Type: application/json"
```

## Sécurité des API Keys

### Bonnes Pratiques

1. **Ne jamais exposer** les clés dans le code source
2. **Utiliser des variables d'environnement**
3. **Rotation régulière** des clés (tous les 90 jours)
4. **Clés séparées** par environnement (dev, staging, prod)
5. **Permissions minimales** - créer des clés avec les scopes nécessaires uniquement

### Variables d'Environnement

```bash
# .env (ne pas commiter!)
VIRALIFY_API_KEY=sk_live_xxxxxxxxxxxx
VIRALIFY_BASE_URL=https://api.viralify.io
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("VIRALIFY_API_KEY")
```

## Scopes et Permissions

| Scope | Description | Endpoints |
|-------|-------------|-----------|
| `courses:read` | Lire les cours | GET /courses/* |
| `courses:write` | Créer/modifier des cours | POST/PUT /courses/* |
| `documents:read` | Lire les documents | GET /documents/* |
| `documents:write` | Uploader des documents | POST /documents/* |
| `voice:read` | Lire les profils vocaux | GET /voice/* |
| `voice:write` | Voice cloning | POST /voice/* |
| `analytics:read` | Lire les analytics | GET /analytics/* |
| `billing:read` | Voir la facturation | GET /billing/* |

## Rate Limiting

Les requêtes sont limitées par plan:

| Plan | Requêtes/min | Burst |
|------|--------------|-------|
| Free | 10 | 20 |
| Starter | 30 | 60 |
| Pro | 100 | 200 |
| Enterprise | Illimité | Illimité |

### Headers de Rate Limit

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1612345678
```

### Gestion du Rate Limit

```python
import time

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after}s...")
            time.sleep(retry_after)
            continue

        return response

    raise Exception("Max retries exceeded")
```

## Erreurs d'Authentification

| Code | Message | Solution |
|------|---------|----------|
| 401 | `Invalid API key` | Vérifiez votre clé |
| 401 | `API key expired` | Générez une nouvelle clé |
| 403 | `Insufficient permissions` | Vérifiez les scopes de la clé |
| 429 | `Rate limit exceeded` | Attendez et réessayez |

## Webhooks (Optionnel)

Pour recevoir des notifications asynchrones:

```python
# Configuration du webhook
response = requests.post(
    "https://api.viralify.io/api/v1/webhooks",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "url": "https://your-server.com/webhook",
        "events": ["course.completed", "course.failed"],
        "secret": "your_webhook_secret"
    }
)
```

### Vérification de signature

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

## Support

- Email: security@viralify.io
- Documentation: https://docs.viralify.io/security
