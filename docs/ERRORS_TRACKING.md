# Viralify - Suivi des Erreurs

Ce document recense les erreurs rencontrées et leur statut de résolution.

---

## Erreurs Ouvertes

### ERR-001: Frontend ne montre pas la progression des outlines
- **Date**: 2026-02-03
- **Statut**: OUVERT
- **Service**: frontend / workers
- **Description**: Quand les workers génèrent les outlines de leçons, la progression n'apparaît pas sur le frontend.
- **Cause probable**: Problème de connexion Redis entre workers et serveur principal, ou websocket non configuré.
- **Impact**: L'utilisateur ne voit pas l'avancement en temps réel.
- **À investiguer**:
  - [ ] Vérifier que les workers se connectent au bon Redis (serveur principal)
  - [ ] Vérifier les logs Redis côté serveur principal
  - [ ] Vérifier la configuration websocket du frontend

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
| Ouvert | 1 |
| En cours | 0 |
| Résolu | 5 |
| **Total** | **6** |

---

*Dernière mise à jour: 2026-02-03*
