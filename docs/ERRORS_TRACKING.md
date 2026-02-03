# Viralify - Suivi des Erreurs

Ce document recense les erreurs rencontrées et leur statut de résolution.

---

## Erreurs Ouvertes

### ERR-001: Frontend ne montre pas la progression des outlines
- **Date**: 2026-02-03
- **Statut**: OUVERT (cause identifiée, fix Redis poussé)
- **Service**: frontend / workers / course-generator
- **Description**: Quand les workers génèrent les outlines de leçons, la progression n'apparaît pas sur le frontend.

**Analyse du flux:**
```
Workers → écrivent dans Redis (course_job:{job_id})
       → Main Server API lit Redis → Frontend poll l'API
```

**Fix appliqué**: Redis configuré pour accepter connexions externes (`--bind 0.0.0.0`)

**Checklist de vérification (sur vos serveurs):**

1. **Redémarrer Redis** sur le serveur principal:
   ```bash
   cd /opt/viralify
   git pull
   docker compose up -d redis
   ```

2. **Ouvrir le port Redis** sur le firewall:
   ```bash
   sudo ufw allow from <WORKER_IP> to any port 6379
   ```

3. **Vérifier `.env.workers`** sur les workers:
   ```
   MAIN_SERVER_HOST=<IP_SERVEUR_PRINCIPAL>
   REDIS_PASSWORD=redis_secure_2024
   ```

4. **Tester la connexion** depuis un worker:
   ```bash
   redis-cli -h <MAIN_SERVER_IP> -p 6379 -a redis_secure_2024 PING
   ```

---

### ERR-012: PostgreSQL authentication failed for user "viralify_prod"
- **Date**: 2026-02-03
- **Statut**: OUVERT (configuration)
- **Service**: course-worker
- **Description**:
  ```
  password authentication failed for user "viralify_prod"
  ```
- **Cause**: L'utilisateur dans `.env.workers` ne correspond pas à celui de la base.
- **Solution**: Modifier `.env.workers`:
  ```bash
  DB_USER=tiktok_user  # PAS viralify_prod
  DB_PASSWORD=tiktok_secure_pass_2024  # Mot de passe du main server
  ```

---

### ERR-013: 404 Not Found sur téléchargement vidéos (olsitec.com)
- **Date**: 2026-02-03
- **Statut**: OUVERT (configuration nginx)
- **Service**: presentation-generator (Compositor)
- **Description**:
  ```
  Client error '404 Not Found' sur les URLs de olsitec.com
  ```
- **Cause**: Les fichiers vidéo ne sont pas servis par le reverse proxy.
- **À faire**:
  1. Configurer nginx pour servir les volumes Docker
  2. Ou utiliser les URLs internes entre services (sans passer par le domaine public)

---

### ERR-014: ReadTimeout sur services (Compositor, CodePipeline)
- **Date**: 2026-02-03
- **Statut**: OUVERT
- **Service**: presentation-generator, course-generator
- **Description**:
  ```
  ReadTimeout - Retrying 1/5, 2/5...
  ```
- **Solutions possibles**:
  1. Augmenter `LLM_TIMEOUT=180` dans `.env`
  2. Vérifier la charge avec `docker stats`
  3. Réduire les replicas si les serveurs sont surchargés

---

### ERR-015: [RAG] No RAG context - contenu générique généré
- **Date**: 2026-02-03
- **Statut**: OUVERT (comportement attendu si pas de documents)
- **Service**: course-generator / presentation-generator
- **Description**:
  ```
  [RAG] No RAG context
  ```
- **Note**: Ce message est NORMAL si aucun document n'est uploadé.
- **Pour utiliser RAG**: Uploader des documents via le frontend AVANT de générer le cours.

---

### ERR-016: Prompt Leakage Detection - Titres modifiés automatiquement
- **Date**: 2026-02-03
- **Statut**: INFO (comportement attendu)
- **Service**: presentation-generator (Planner)
- **Description**: Le système modifie automatiquement les titres robotiques.
- **Note**: C'est le **Title Style System** qui fonctionne comme prévu.
- **Pour désactiver**: Utiliser `title_style: "direct"` dans la requête.

---

## Erreurs Résolues

### ERR-017: Rate Limits Groq - pauses de 40-50 secondes
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: nexus-engine
- **Solution**: `GroqRateLimiter` avec rotation de clés API.
- **Configuration**:
  ```bash
  # Plusieurs clés en rotation (recommandé)
  GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3
  ```
- **Fichiers**: `services/shared/groq_rate_limiter.py`, `services/nexus-engine/`

---

### ERR-007: Import Fluentd inexistant dans diagrams library
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: visual-generator
- **Solution**: `ImportCorrector` avec auto-correction de 15+ imports.
- **Fichier**: `services/visual-generator/renderers/diagrams_renderer.py`

---

### ERR-009: TypeError list >> list unsupported operand
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: visual-generator
- **Solution**: Ajouté règle "FORBIDDEN: [list] >> [list]" dans le prompt.

---

### ERR-010: ImportError GenericSQL
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: visual-generator
- **Solution**: Instruction explicite pour GenericSQL dans le prompt.

---

### ERR-011: NameError case sensitivity (postgresql vs PostgreSQL)
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: visual-generator
- **Solution**: Section "VARIABLE NAMING - CASE SENSITIVE" dans le prompt.

---

### ERR-008: TypeError can only concatenate str (not "int") to str
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: presentation-generator
- **Solution**: f-string au lieu de concaténation.
- **Fichier**: `services/presentation-generator/services/code_pipeline/code_generator.py`

---

### ERR-006: ClassificationResult object has no attribute 'lower'
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: presentation-generator
- **Solution**: Passé `description` au lieu de `classification` à `router.route()`.
- **Fichier**: `services/presentation-generator/services/viralify_diagram_service.py`

---

### ERR-002: DiagramRouter.route() - unexpected keyword argument 'complexity'
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: presentation-generator
- **Solution**: Supprimé le paramètre `complexity`.

---

### ERR-003: nexus-engine service not found
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: workers
- **Solution**: Ajouté nexus-engine dans `docker-compose.workers.yml`.

---

### ERR-004: Redis memory overcommit warning
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Solution**: `sudo sysctl vm.overcommit_memory=1`

---

### ERR-005: Durée vidéo plus courte que la cible
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: presentation-generator
- **Solution**: Seuils dynamiques basés sur `words_per_slide`.

---

## Résumé des actions requises sur vos serveurs

### Sur le serveur PRINCIPAL:
```bash
cd /opt/viralify
git pull
docker compose up -d --build redis course-generator
```

### Sur les WORKERS:
```bash
cd /opt/viralify
git pull

# Vérifier .env.workers:
# - DB_USER=tiktok_user (pas viralify_prod)
# - REDIS_PASSWORD=redis_secure_2024
# - GROQ_API_KEYS=gsk_key1,gsk_key2  (optionnel, pour rotation)

docker compose -f docker-compose.workers.yml up -d --build
```

### Firewall (serveur principal):
```bash
sudo ufw allow from <WORKER_IP_1> to any port 6379
sudo ufw allow from <WORKER_IP_2> to any port 6379
```

---

## Statistiques

| Statut | Nombre |
|--------|--------|
| Ouvert | 4 |
| Info | 2 |
| Résolu | 11 |
| **Total** | **17** |

---

*Dernière mise à jour: 2026-02-03*
