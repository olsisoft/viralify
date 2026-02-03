# Viralify - Suivi des Erreurs

Ce document recense les erreurs rencontrées et leur statut de résolution.

---

## Erreurs Ouvertes

### ERR-009: TypeError list >> list unsupported operand
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: visual-generator
- **Description**:
  ```
  TypeError: unsupported operand type(s) for >>: 'list' and 'list'
  ```
- **Cause**: Le LLM génère `[node1, node2] >> [node3, node4]` qui est invalide.
- **Solution**: Ajouté règle explicite "FORBIDDEN: [list] >> [list]" dans le prompt.
- **Fichier modifié**: `services/visual-generator/renderers/diagrams_renderer.py`

---

### ERR-010: ImportError GenericSQL
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: visual-generator
- **Description**:
  ```
  ImportError: cannot import name 'GenericSQL' from 'diagrams.generic.database'
  ```
- **Cause**: Le LLM écrit `import GenericSQL` au lieu de `import SQL as GenericSQL`.
- **Solution**: Ajouté instruction explicite pour GenericSQL dans le prompt.
- **Fichier modifié**: `services/visual-generator/renderers/diagrams_renderer.py`

---

### ERR-011: NameError case sensitivity (postgresql vs PostgreSQL)
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: visual-generator
- **Description**:
  ```
  NameError: name 'postgresql' is not defined
  ```
- **Cause**: Le LLM utilise `postgresql` au lieu de `PostgreSQL` (case sensitive).
- **Solution**: Ajouté section "VARIABLE NAMING - CASE SENSITIVE" dans le prompt.
- **Fichier modifié**: `services/visual-generator/renderers/diagrams_renderer.py`

---

### ERR-008: TypeError can only concatenate str (not "int") to str
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: presentation-generator
- **Description**:
  ```
  File "/app/services/code_pipeline/code_generator.py", line 274, in _validate_against_spec
      {"Exemple I/O: " + spec.example_io.input_value + " → " + spec.example_io.expected_output if spec.example_io else ""}
  TypeError: can only concatenate str (not "int") to str
  ```
- **Cause**: `input_value` ou `expected_output` peut être un int, pas une string.
- **Solution**: Utilisé f-string au lieu de concaténation avec `+`.
- **Fichier modifié**: `services/presentation-generator/services/code_pipeline/code_generator.py`

---

### ERR-007: Import Fluentd inexistant dans diagrams library
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: visual-generator
- **Description**:
  ```
  ImportError: cannot import name 'Fluentd' from 'diagrams.onprem.logging'
  ```
- **Cause**: Le LLM ignore le DIAGRAMS_CHEAT_SHEET et hallucine des imports incorrects.
- **Solution implémentée**:
  1. Ajouté `ImportCorrector` class dans `diagrams_renderer.py`
  2. Auto-correction de 15+ erreurs d'import courantes:
     - Fluentd: logging → aggregator
     - Redis: database → inmemory
     - Elasticsearch/Kibana/Logstash: onprem → elastic
     - Et plus...
  3. Validation AST avant exécution
  4. Logs des corrections appliquées
- **Fichier modifié**: `services/visual-generator/renderers/diagrams_renderer.py`

---

### ERR-015: [RAG] No RAG context - contenu générique généré
- **Date**: 2026-02-03
- **Statut**: OUVERT
- **Service**: course-generator / presentation-generator
- **Description**:
  ```
  [RAG] No RAG context
  ```
- **Cause probable**:
  - Aucun document uploadé pour le cours
  - Documents mal traités ou non vectorisés
  - Connexion au vector store (pgvector) échouée
  - `document_ids` non passés dans la requête de génération
- **Impact**: L'IA génère du contenu générique au lieu d'utiliser les documents source.
- **À vérifier**:
  - [ ] Vérifier que des documents sont uploadés avant génération
  - [ ] Vérifier les logs de vector_store/retrieval_service
  - [ ] Vérifier la connexion pgvector: `VECTOR_BACKEND=pgvector`
  - [ ] S'assurer que `document_ids` est inclus dans la requête API

---

### ERR-016: Prompt Leakage Detection - Titres modifiés automatiquement
- **Date**: 2026-02-03
- **Statut**: INFO (comportement attendu)
- **Service**: presentation-generator (Planner)
- **Description**:
  ```
  Slide 1 'Welcome to...' → Fixed to 'Discovering...'
  ```
- **Explication**: C'est le **Title Style System** qui fonctionne comme prévu.
  - Détecte les patterns robotiques (Introduction, Conclusion, Welcome to...)
  - Les remplace par des titres plus naturels
- **Configuration**: Dans la requête, le champ `title_style` contrôle ce comportement:
  - `engaging` (défaut): Titres dynamiques
  - `direct`: Titres simples et directs
  - `corporate`: Professionnel et formel
- **Action**: Si les titres originaux sont souhaités, utiliser `title_style: "direct"` ou désactiver dans les settings

---

### ERR-017: Rate Limits Groq - pauses de 40-50 secondes
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: nexus-engine
- **Description**:
  ```
  Rate limit exceeded - waiting 45 seconds...
  ```
- **Cause**: L'API Groq a des limites strictes (requests/minute, tokens/minute).
- **Solution implémentée**: `GroqRateLimiter` avec:
  1. **Rotation de clés API** (round-robin entre plusieurs clés)
  2. **Throttling intelligent** (attend juste assez pour rester sous la limite)
  3. **Tracking par fenêtre glissante** (requests/min, tokens/min)
  4. **Buffer de sécurité** (utilise 90% de la limite)
- **Configuration** (dans `.env` ou `.env.workers`):
  ```bash
  # Une seule clé (comportement par défaut)
  GROQ_API_KEY=gsk_xxx

  # OU plusieurs clés en rotation (recommandé)
  GROQ_API_KEYS=gsk_key1,gsk_key2,gsk_key3

  # Optionnel: ajuster les limites
  GROQ_REQUESTS_PER_MINUTE=30
  GROQ_TOKENS_PER_MINUTE=6000
  ```
- **Fichiers créés/modifiés**:
  - `services/shared/groq_rate_limiter.py` (NOUVEAU)
  - `services/nexus-engine/providers/llm_provider.py`
  - `services/nexus-engine/main.py`

---

### ERR-012: PostgreSQL authentication failed for user "viralify_prod"
- **Date**: 2026-02-03
- **Statut**: OUVERT
- **Service**: course-worker
- **Description**:
  ```
  password authentication failed for user "viralify_prod"
  ```
- **Cause probable**: L'utilisateur `viralify_prod` n'existe pas ou le mot de passe est incorrect.
- **Impact**: Le course-worker ne peut pas stocker les composants de cours.
- **À vérifier**:
  - [ ] Vérifier que l'utilisateur existe dans PostgreSQL: `SELECT usename FROM pg_user;`
  - [ ] Vérifier le mot de passe dans `.env.workers` (DB_USER, DB_PASSWORD)
  - [ ] Vérifier que `.env.workers` utilise le bon utilisateur (tiktok_user vs viralify_prod)
  - [ ] Sur le serveur principal: `docker exec -it viralify-postgres psql -U tiktok_user -c "\du"`

---

### ERR-013: 404 Not Found sur téléchargement vidéos (olsitec.com)
- **Date**: 2026-02-03
- **Statut**: OUVERT
- **Service**: presentation-generator (Compositor)
- **Description**:
  ```
  Client error '404 Not Found' sur les URLs de olsitec.com
  ```
- **Cause probable**:
  - Les fichiers vidéo ne sont pas accessibles publiquement
  - L'URL générée est incorrecte
  - Le serveur web n'est pas configuré pour servir les fichiers
- **Impact**: Échec de la composition vidéo finale.
- **À vérifier**:
  - [ ] Vérifier la configuration PUBLIC_BASE_URL et PUBLIC_MEDIA_URL
  - [ ] Vérifier que nginx/reverse proxy sert les fichiers du volume
  - [ ] Tester manuellement l'URL pour voir la réponse

---

### ERR-014: ReadTimeout sur services (Compositor, CodePipeline)
- **Date**: 2026-02-03
- **Statut**: OUVERT
- **Service**: presentation-generator, course-generator
- **Description**:
  ```
  ReadTimeout - Retrying 1/5, 2/5...
  ```
- **Cause probable**:
  - Services surchargés
  - Latence réseau entre workers et serveur principal
  - Timeout trop court pour opérations longues
- **Impact**: Ralentissement et échecs intermittents.
- **À vérifier**:
  - [ ] Vérifier la charge CPU/RAM des services: `docker stats`
  - [ ] Vérifier la latence réseau entre workers et main server
  - [ ] Augmenter les timeouts si nécessaire (LLM_TIMEOUT)

---

### ERR-001: Frontend ne montre pas la progression des outlines
- **Date**: 2026-02-03
- **Statut**: OUVERT (cause identifiée)
- **Service**: frontend / workers / course-generator
- **Description**: Quand les workers génèrent les outlines de leçons, la progression n'apparaît pas sur le frontend.

**Analyse du flux:**
```
Workers → écrivent dans Redis (course_job:{job_id})
       → Main Server API lit Redis → Frontend poll l'API
```

**Cause probable**: Les workers écrivent dans Redis mais le main server ne lit pas du bon Redis ou `USE_QUEUE` n'est pas activé.

**Checklist de vérification:**

1. **Sur le serveur principal** - Vérifier les logs au démarrage:
   ```
   [STARTUP] Queue Mode: enabled  ← doit être "enabled"
   [STARTUP] Redis connected for job status sync  ← doit apparaître
   ```

2. **Sur le serveur principal** - Vérifier que Redis écoute sur toutes les interfaces:
   ```bash
   docker exec viralify-redis redis-cli CONFIG GET bind
   # Doit retourner "0.0.0.0" ou être vide
   ```

3. **Ouvrir le port Redis** sur le firewall du serveur principal:
   ```bash
   sudo ufw allow from <WORKER_IP> to any port 6379
   ```

4. **Sur les workers** - Vérifier `.env.workers`:
   ```
   MAIN_SERVER_HOST=<IP_SERVEUR_PRINCIPAL>
   REDIS_PASSWORD=redis_secure_2024  # Doit matcher le main server!
   ```

5. **Tester la connexion Redis** depuis un worker:
   ```bash
   redis-cli -h <MAIN_SERVER_IP> -p 6379 -a redis_secure_2024 PING
   # Doit retourner PONG
   ```

6. **Vérifier les logs de l'API** quand on poll un job:
   ```
   [STATUS] Getting job xxx: in_memory=False, USE_QUEUE=True, redis=True
   [STATUS] Redis data for xxx: {'status': 'generating_outline', ...}
   ```

**Configuration docker-compose.yml (déjà correcte):**
- Ligne 519: `REDIS_URL=redis://:redis_secure_2024@redis:6379/7`
- Ligne 546: `USE_QUEUE=true`

---

## Erreurs Résolues

### ERR-002: DiagramRouter.route() - unexpected keyword argument 'complexity'
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: presentation-generator
- **Description**:
  ```
  TypeError: DiagramRouter.route() got an unexpected keyword argument 'complexity'
  ```
- **Cause**: Le paramètre `complexity` a été passé à `route()` mais n'est pas supporté.
- **Solution**: Supprimé le paramètre `complexity` de l'appel dans `viralify_diagram_service.py`.
- **Fichier modifié**: `services/presentation-generator/services/viralify_diagram_service.py`

---

### ERR-003: nexus-engine service not found
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: workers / nexus-engine
- **Description**:
  ```
  Name or service not known (nexus-engine)
  ```
- **Cause**: Le service `nexus-engine` n'était pas inclus dans `docker-compose.workers.yml`.
- **Solution**: Ajouté le service `nexus-engine` dans `docker-compose.workers.yml` sur le port 8011.
- **Fichier modifié**: `docker-compose.workers.yml`

---

### ERR-004: Redis memory overcommit warning
- **Date**: 2026-02-03
- **Statut**: RÉSOLU (workaround)
- **Service**: redis-local (workers)
- **Description**:
  ```
  WARNING Memory overcommit must be enabled! Without it, a background save or
  replication may fail under low memory condition.
  ```
- **Solution**: Sur le serveur worker, exécuter:
  ```bash
  sudo sysctl vm.overcommit_memory=1
  echo "vm.overcommit_memory=1" | sudo tee -a /etc/sysctl.conf
  ```

---

### ERR-005: Durée vidéo plus courte que la cible
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: presentation-generator
- **Description**: Les vidéos générées sont plus courtes que la durée demandée (ex: 147s au lieu de 300s).
- **Cause**: Seuils de mots par slide codés en dur (150 mots minimum) au lieu d'être dynamiques.
- **Solution**: Rendu les seuils dynamiques basés sur `words_per_slide` calculé.
- **Fichier modifié**: `services/presentation-generator/services/presentation_planner.py`

---

### ERR-006: ClassificationResult object has no attribute 'lower'
- **Date**: 2026-02-03
- **Statut**: RÉSOLU
- **Service**: presentation-generator
- **Description**:
  ```
  [VIRALIFY] Error in intelligent generation: 'ClassificationResult' object has no attribute 'lower'
  File "/app/services/viralify_diagram_service.py", line 238, in get_slide_recommendation
      routing = self.router.route(
  ```
- **Cause**: `router.route()` attend une string (description) mais recevait un objet `ClassificationResult`.
- **Solution**: Passé `description` au lieu de `classification` à `router.route()`.
- **Fichier modifié**: `services/presentation-generator/services/viralify_diagram_service.py`

---

## Template pour nouvelles erreurs

```markdown
### ERR-XXX: [Titre court]
- **Date**: YYYY-MM-DD
- **Statut**: OUVERT / EN COURS / RÉSOLU
- **Service**: [nom du service]
- **Description**:
  ```
  [Message d'erreur exact]
  ```
- **Cause**: [Description de la cause]
- **Solution**: [Description de la solution]
- **Fichier modifié**: [Chemin du fichier]
```

---

## Statistiques

| Statut | Nombre |
|--------|--------|
| Ouvert | 5 |
| En cours | 0 |
| Résolu | 11 |
| Info | 1 |
| **Total** | **17** |

---

*Dernière mise à jour: 2026-02-03*
