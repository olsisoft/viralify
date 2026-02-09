# Viralify API Documentation

Documentation complète de l'API Viralify pour la génération automatisée de cours vidéo éducatifs.

## Quick Links

- [Getting Started](./guides/getting-started.md) - Commencer avec l'API
- [Authentication](./guides/authentication.md) - Authentification et sécurité
- [Architecture](./guides/architecture.md) - Vue d'ensemble de l'architecture
- [Workflows](./guides/workflows.md) - Workflows de génération
- [Errors](./guides/errors.md) - Gestion des erreurs

## API Reference

### Services Principaux

| Service | Port | Description | OpenAPI Spec |
|---------|------|-------------|--------------|
| **Course Generator** | 8007 | Génération de curriculum et cours | [course-generator.yaml](./openapi/course-generator.yaml) |
| **Presentation Generator** | 8006 | Génération de vidéos et slides | [presentation-generator.yaml](./openapi/presentation-generator.yaml) |
| **Media Generator** | 8004 | TTS, voice cloning, éditeur vidéo | [media-generator.yaml](./openapi/media-generator.yaml) |

### Services Spécialisés

| Service | Port | Description | OpenAPI Spec |
|---------|------|-------------|--------------|
| **Visual Generator** | 8003 | Génération de diagrammes | [visual-generator.yaml](./openapi/visual-generator.yaml) |
| **VQV-HALLU** | 8009 | Validation qualité voiceover | [vqv-hallu.yaml](./openapi/vqv-hallu.yaml) |
| **MAESTRO Engine** | 8010 | Génération avancée sans documents | [maestro-engine.yaml](./openapi/maestro-engine.yaml) |
| **NEXUS Engine** | 8011 | Génération de code pédagogique | [nexus-engine.yaml](./openapi/nexus-engine.yaml) |

### Services Auxiliaires

| Service | Port | Description |
|---------|------|-------------|
| Analytics Service | 8002 | Métriques et analytics |
| Content Generator | 8001 | Génération de scripts |
| Coaching Service | 8005 | Gamification et coaching |
| Practice Agent | 8008 | Exercices interactifs |
| Trend Analyzer | 8000 | Analyse des tendances |

## Interactive Documentation

### Swagger UI

Accédez à la documentation interactive Swagger UI:

```
http://localhost:8007/docs  # Course Generator
http://localhost:8006/docs  # Presentation Generator
http://localhost:8004/docs  # Media Generator
```

### Redoc

Documentation statique Redoc:

```
http://localhost:8007/redoc
http://localhost:8006/redoc
http://localhost:8004/redoc
```

## Code Examples

- [Python Examples](./examples/python/)
- [JavaScript Examples](./examples/javascript/)
- [cURL Examples](./examples/curl/)

## Environments

| Environment | Base URL | Description |
|-------------|----------|-------------|
| Development | `http://localhost:800X` | Local Docker |
| Staging | `https://staging-api.viralify.io` | Test environment |
| Production | `https://api.viralify.io` | Live environment |

## Rate Limits

| Plan | Requests/min | Concurrent Jobs |
|------|--------------|-----------------|
| Free | 10 | 1 |
| Starter | 30 | 3 |
| Pro | 100 | 10 |
| Enterprise | Unlimited | Unlimited |

## Support

- GitHub Issues: https://github.com/olsisoft/viralify/issues
- Email: support@viralify.io
- Discord: https://discord.gg/viralify

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for API version history.
