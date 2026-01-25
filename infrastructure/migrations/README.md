# Database Migrations

Ce dossier contient les scripts de migration SQL pour la base de données PostgreSQL.

## Migrations disponibles

| Fichier | Description | Date |
|---------|-------------|------|
| `002_weave_graph.sql` | Tables WeaveGraph pour Phase 2 & 3 (concepts, edges) | 2026-01-25 |

## Comment exécuter une migration

### Option 1: Script automatique (Linux/Mac)

```bash
cd /opt/viralify/infrastructure/migrations
chmod +x run_migration.sh
./run_migration.sh 002_weave_graph.sql
```

### Option 2: Manuellement

```bash
# Se connecter à PostgreSQL
docker exec -it tiktok-postgres psql -U tiktok_user -d tiktok_platform

# Exécuter le fichier
\i /path/to/002_weave_graph.sql

# Ou depuis l'extérieur du container
docker exec -i tiktok-postgres psql -U tiktok_user -d tiktok_platform < 002_weave_graph.sql
```

### Option 3: Copier dans le container

```bash
# Copier le fichier dans le container
docker cp 002_weave_graph.sql tiktok-postgres:/tmp/

# Exécuter
docker exec -it tiktok-postgres psql -U tiktok_user -d tiktok_platform -f /tmp/002_weave_graph.sql
```

## Vérification

Après l'exécution, vérifier que les tables existent:

```sql
-- Lister les tables WeaveGraph
\dt weave_*

-- Vérifier la structure
\d weave_concepts
\d weave_edges

-- Vérifier les index
\di weave_*
```

## Rollback

Pour supprimer les tables WeaveGraph (⚠️ perte de données):

```sql
DROP TABLE IF EXISTS weave_edges CASCADE;
DROP TABLE IF EXISTS weave_concepts CASCADE;
```

## Notes

- Les migrations sont idempotentes (`IF NOT EXISTS`)
- L'extension pgvector doit être installée (image `pgvector/pgvector:pg16`)
- Les embeddings utilisent E5-large (1024 dimensions)
