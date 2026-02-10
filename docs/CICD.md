# CI/CD Pipeline - Viralify

Documentation complète du pipeline CI/CD pour la plateforme Viralify.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Workflows GitHub Actions](#workflows-github-actions)
4. [Scripts portables](#scripts-portables)
5. [Makefile - Interface unifiée](#makefile---interface-unifiée)
6. [Configuration](#configuration)
7. [Déploiement](#déploiement)
8. [Sécurité](#sécurité)
9. [Portabilité](#portabilité)
10. [Dépannage](#dépannage)

---

## Vue d'ensemble

### Flux principal

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLUX CI/CD VIRALIFY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DÉVELOPPEUR                                                                │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────────────────┐   │
│  │  Code   │────▶│  Push   │────▶│   PR    │────▶│  CI (ci.yml)        │   │
│  │  Local  │     │  Branch │     │  Open   │     │  • Lint             │   │
│  └─────────┘     └─────────┘     └─────────┘     │  • Tests            │   │
│                                                   │  • Type Check       │   │
│                                                   └──────────┬──────────┘   │
│                                                              │              │
│                                                              ▼              │
│                                                   ┌─────────────────────┐   │
│                                                   │  ✅ PR Approved     │   │
│                                                   │  Merge to master    │   │
│                                                   └──────────┬──────────┘   │
│                                                              │              │
│                                                              ▼              │
│                                                   ┌─────────────────────┐   │
│                                                   │  Build & Push       │   │
│                                                   │  (build-push.yml)   │   │
│                                                   │  • Detect changes   │   │
│                                                   │  • Build images     │   │
│                                                   │  • Push to GHCR     │   │
│                                                   └──────────┬──────────┘   │
│                                                              │              │
│                                                              ▼              │
│                                                   ┌─────────────────────┐   │
│                                                   │  Deploy Staging     │   │
│                                                   │  (deploy-staging)   │   │
│                                                   │  • SSH to server    │   │
│                                                   │  • /rebuild.sh      │   │
│                                                   │  • /setup-worker.sh │   │
│                                                   └──────────┬──────────┘   │
│                                                              │              │
│                                                              ▼              │
│                                          ┌────────────────────────────────┐ │
│                                          │  🧪 Test en Staging            │ │
│                                          │  (manuel par l'équipe)         │ │
│                                          └───────────────┬────────────────┘ │
│                                                          │                  │
│                                                          ▼                  │
│                                          ┌────────────────────────────────┐ │
│                                          │  Deploy Production             │ │
│                                          │  (deploy-production.yml)       │ │
│                                          │  • Déclenchement MANUEL        │ │
│                                          │  • Confirmation requise        │ │
│                                          │  • /rebuild.sh                 │ │
│                                          │  • /setup-worker.sh            │ │
│                                          └────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Principes de conception

| Principe | Description |
|----------|-------------|
| **Portabilité** | Logique dans des scripts bash, pas dans le YAML CI |
| **Incrémental** | Ne build que les services modifiés |
| **Sécurisé** | Scans automatiques, secrets externalisés |
| **Reproductible** | Même comportement local et en CI |

---

## Architecture

### Structure des fichiers

```
viralify/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Tests sur PR
│       ├── build-push.yml         # Build + push images
│       ├── deploy-staging.yml     # Deploy auto staging
│       ├── deploy-production.yml  # Deploy manuel prod
│       └── security-scan.yml      # Scan hebdomadaire
├── scripts/
│   └── ci/
│       ├── common.sh              # Fonctions partagées
│       ├── lint.sh                # Linting Python + TS
│       ├── test.sh                # Tests unitaires
│       ├── build.sh               # Build Docker images
│       ├── push.sh                # Push vers registry
│       ├── deploy.sh              # Déploiement SSH
│       └── security-scan.sh       # Scans de sécurité
├── Makefile                       # Interface unifiée
└── ci.env.example                 # Variables documentées
```

### Couches d'abstraction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ARCHITECTURE PORTABLE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COUCHE 1: ORCHESTRATION (interchangeable)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  GitHub Actions │ GitLab CI │ Jenkins │ CircleCI │ Cloud Build      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  COUCHE 2: INTERFACE (Makefile)                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  make test │ make build │ make deploy-stg │ make security           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  COUCHE 3: SCRIPTS PORTABLES (scripts/ci/)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  test.sh │ build.sh │ deploy.sh │ security-scan.sh                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  COUCHE 4: OUTILS STANDARDS                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Docker │ kubectl │ pytest │ npm │ ruff │ trivy │ ssh               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Workflows GitHub Actions

### 1. CI - Tests (`ci.yml`)

**Déclencheur:** Push ou PR vers `master`, `main`, `develop`

**Jobs:**

| Job | Description | Durée estimée |
|-----|-------------|---------------|
| `lint` | Linting Python (ruff) + TypeScript (tsc) | ~1 min |
| `test` | Tests Python en parallèle (matrix) | ~3 min |
| `test-frontend` | Tests et type-check frontend | ~2 min |
| `build-check` | Validation des Dockerfiles (PR only) | ~2 min |

**Exemple de sortie:**

```
✅ lint         (1m 12s)
✅ test         (3m 45s)  [course-generator, presentation-generator]
✅ test-frontend (2m 03s)
✅ build-check  (2m 30s)
```

### 2. Build & Push (`build-push.yml`)

**Déclencheur:** Merge sur `master`/`main`

**Fonctionnement:**

1. **Détection des changements** - Utilise `dorny/paths-filter` pour identifier les services modifiés
2. **Build parallèle** - Matrix build pour chaque service modifié
3. **Push vers GHCR** - Tags: `sha`, `branch`, `latest`
4. **Cache** - Utilise GitHub Actions cache pour les layers Docker

**Tags générés:**

```
ghcr.io/olsisoft/viralify/course-generator:abc1234
ghcr.io/olsisoft/viralify/course-generator:master
ghcr.io/olsisoft/viralify/course-generator:latest
```

### 3. Deploy Staging (`deploy-staging.yml`)

**Déclencheur:** Automatique après `build-push.yml` réussi

**Étapes:**

```bash
# 1. Connexion SSH au serveur staging
ssh $STAGING_USER@$STAGING_HOST

# 2. Pull des nouvelles images
cd /opt/viralify && docker compose pull

# 3. Redémarrage du serveur principal
/rebuild.sh

# 4. Redémarrage des workers
/setup-worker.sh

# 5. Health check
curl -sf https://staging.viralify.app/health
```

### 4. Deploy Production (`deploy-production.yml`)

**Déclencheur:** Manuel uniquement (workflow_dispatch)

**Sécurités:**

- ✅ Confirmation textuelle obligatoire (`deploy-production`)
- ✅ Environment protection (approbation requise)
- ✅ Délai de 5 secondes avant exécution
- ✅ Health check post-déploiement

**Utilisation:**

1. Aller dans Actions → Deploy Production
2. Cliquer "Run workflow"
3. Entrer l'image tag (ex: `abc1234` ou `latest`)
4. Taper `deploy-production` pour confirmer
5. Cliquer "Run workflow"

### 5. Security Scan (`security-scan.yml`)

**Déclencheur:** Chaque dimanche 2h00 UTC + manuel

**Scans effectués:**

| Scan | Outil | Cible |
|------|-------|-------|
| Dépendances Python | pip-audit | `requirements.txt` |
| Dépendances Node | npm audit | `package-lock.json` |
| Code Python | Bandit | `services/` |
| Secrets | Gitleaks | Tout le repo |
| Containers | Trivy | Images Docker |

---

## Scripts portables

### `common.sh` - Fonctions partagées

```bash
# Logging coloré
log_info "Message"      # Bleu
log_success "Message"   # Vert
log_warning "Message"   # Jaune
log_error "Message"     # Rouge

# Utilitaires
require_command "docker"        # Vérifie qu'une commande existe
load_env ".env"                 # Charge les variables d'environnement
get_services                    # Liste tous les services avec Dockerfile
get_modified_services           # Liste les services modifiés (git diff)
build_image_tag "service"       # Génère le tag complet
is_ci                           # Détecte si on est en CI
get_branch                      # Branche git actuelle
get_short_sha                   # SHA court du commit
retry 3 5 "command"             # Retry avec backoff exponentiel
```

### `test.sh` - Tests unitaires

```bash
# Tester tous les services
./scripts/ci/test.sh

# Tester un service spécifique
./scripts/ci/test.sh course-generator

# Avec couverture
./scripts/ci/test.sh --coverage
```

### `build.sh` - Build Docker

```bash
# Build les services modifiés uniquement
./scripts/ci/build.sh

# Build tous les services
./scripts/ci/build.sh --all

# Build un service spécifique
./scripts/ci/build.sh course-generator

# Build et push
./scripts/ci/build.sh --all --push
```

### `deploy.sh` - Déploiement

```bash
# Deploy staging
./scripts/ci/deploy.sh staging

# Deploy production
./scripts/ci/deploy.sh production

# Dry run (simulation)
./scripts/ci/deploy.sh staging --dry-run
```

---

## Makefile - Interface unifiée

### Commandes disponibles

```bash
make help              # Affiche l'aide

# Développement
make install           # Installe les dépendances
make lint              # Lance les linters
make lint-fix          # Lint avec auto-correction
make test              # Lance les tests
make test-cov          # Tests avec couverture

# Build
make build             # Build images modifiées
make build-all         # Build TOUTES les images
make build-service SERVICE=xxx  # Build un service

# Push
make push              # Push toutes les images
make push-service SERVICE=xxx   # Push un service

# Deploy
make deploy-stg        # Deploy staging
make deploy-prod       # Deploy production
make deploy-dry-run    # Simulation

# Sécurité
make security          # Tous les scans
make security-images   # Scan images Docker
make security-code     # Scan code
make security-deps     # Scan dépendances

# Docker local
make docker-up         # Démarrer docker-compose
make docker-down       # Arrêter docker-compose
make logs              # Voir les logs

# Nettoyage
make clean             # Nettoyer les artefacts
```

### Variables d'environnement

```bash
# Personnaliser le registry
REGISTRY_URL=docker.io/myuser make build

# Personnaliser le tag
IMAGE_TAG=v1.0.0 make build

# Personnaliser l'environnement
ENVIRONMENT=production make deploy-dry-run
```

---

## Configuration

### Secrets GitHub requis

| Secret | Description | Exemple |
|--------|-------------|---------|
| `STAGING_HOST` | Hostname/IP du serveur staging | `staging.viralify.app` |
| `STAGING_USER` | Utilisateur SSH staging | `root` |
| `STAGING_SSH_KEY` | Clé SSH privée staging | `-----BEGIN OPENSSH...` |
| `PRODUCTION_HOST` | Hostname/IP du serveur prod | `viralify.app` |
| `PRODUCTION_USER` | Utilisateur SSH production | `root` |
| `PRODUCTION_SSH_KEY` | Clé SSH privée production | `-----BEGIN OPENSSH...` |

### Configuration des secrets

1. Aller dans **Settings** → **Secrets and variables** → **Actions**
2. Cliquer **New repository secret**
3. Ajouter chaque secret

### Environments GitHub

1. Aller dans **Settings** → **Environments**
2. Créer `staging` :
   - Pas de protection particulière
3. Créer `production` :
   - ✅ Required reviewers (ajouter les approbateurs)
   - ✅ Wait timer: 5 minutes (optionnel)

### Variables d'environnement CI

Voir `ci.env.example` pour la liste complète :

```bash
# Registry Docker
REGISTRY_URL=ghcr.io/olsisoft/viralify

# Serveurs
STAGING_HOST=staging.viralify.app
PRODUCTION_HOST=viralify.app

# Sécurité
SEVERITY_THRESHOLD=HIGH
```

---

## Déploiement

### Prérequis serveur

Les serveurs staging et production doivent avoir :

1. **Docker** et **Docker Compose** installés
2. **Scripts de déploiement** :
   - `/rebuild.sh` - Redémarre le serveur principal
   - `/setup-worker.sh` - Configure et démarre les workers
3. **Accès SSH** configuré avec la clé dans les secrets GitHub
4. **Projet cloné** dans `/opt/viralify`

### Processus de déploiement

#### Staging (automatique)

```
Merge PR → Build images → Push GHCR → SSH staging → /rebuild.sh → /setup-worker.sh
```

#### Production (manuel)

```
Actions → Deploy Production → Confirmer → SSH prod → /rebuild.sh → /setup-worker.sh
```

### Rollback

En cas de problème en production :

```bash
# SSH sur le serveur
ssh root@viralify.app

# Revenir à une version précédente
cd /opt/viralify
export IMAGE_TAG=<previous-tag>
docker compose pull
/rebuild.sh
/setup-worker.sh
```

---

## Sécurité

### Scans automatiques

| Type | Fréquence | Outil |
|------|-----------|-------|
| Dépendances | Hebdomadaire + PR | pip-audit, npm audit |
| Code | Hebdomadaire + PR | Bandit |
| Secrets | Chaque push | Gitleaks |
| Containers | Hebdomadaire | Trivy |

### Bonnes pratiques

- ✅ Ne jamais commiter de secrets (`.env`, clés API)
- ✅ Utiliser les GitHub Secrets pour les données sensibles
- ✅ Vérifier les alertes de sécurité dans l'onglet Security
- ✅ Mettre à jour régulièrement les dépendances

### Seuils de sévérité

| Sévérité | Action |
|----------|--------|
| CRITICAL | ❌ Bloque le pipeline |
| HIGH | ⚠️ Warning, à corriger rapidement |
| MEDIUM | 📋 À planifier |
| LOW | 📝 Informatif |

---

## Portabilité

### Pourquoi c'est portable ?

Toute la logique est dans les scripts bash, pas dans le YAML :

```yaml
# GitHub Actions appelle juste make
- run: make test

# GitLab CI ferait pareil
script:
  - make test

# Jenkins aussi
sh 'make test'
```

### Migration vers un autre CI

| Cible | Fichier à créer | Effort |
|-------|-----------------|--------|
| GitLab CI | `.gitlab-ci.yml` | ~2h |
| Jenkins | `Jenkinsfile` | ~1h |
| CircleCI | `.circleci/config.yml` | ~2h |
| AWS CodePipeline | `buildspec.yml` | ~2h |
| Azure DevOps | `azure-pipelines.yml` | ~2h |

### Migration vers un autre cloud

| Changement | Action |
|------------|--------|
| GHCR → ECR | Changer `REGISTRY_URL` |
| GHCR → GCR | Changer `REGISTRY_URL` + auth |
| Serveur → Kubernetes | Utiliser `make deploy-k8s` |

### Exemple GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  script:
    - make lint
    - make test

build:
  stage: build
  script:
    - make build-all
    - make push
  only:
    - master

deploy-staging:
  stage: deploy
  script:
    - make deploy-stg
  only:
    - master
```

---

## Dépannage

### Problèmes courants

#### Les tests échouent en CI mais passent localement

```bash
# Vérifier les dépendances
make install

# Lancer exactement comme en CI
make test
```

#### Build Docker échoue

```bash
# Vérifier la syntaxe Dockerfile
docker build -f services/xxx/Dockerfile services/xxx

# Vérifier les logs complets
docker build --no-cache -f services/xxx/Dockerfile services/xxx
```

#### Déploiement SSH échoue

```bash
# Tester la connexion SSH manuellement
ssh -i ~/.ssh/key user@host

# Vérifier que les scripts existent
ssh user@host "ls -la /rebuild.sh /setup-worker.sh"
```

#### Images non trouvées

```bash
# Vérifier le login au registry
docker login ghcr.io

# Vérifier que l'image existe
docker pull ghcr.io/olsisoft/viralify/service:tag
```

### Logs et debugging

```bash
# Voir les logs GitHub Actions
# → Onglet Actions → Cliquer sur le workflow → Cliquer sur le job

# Logs Docker locaux
make logs

# Logs d'un service spécifique
make docker-logs SERVICE=course-generator
```

### Contact

Pour les problèmes non résolus :
- Ouvrir une issue sur GitHub
- Contacter l'équipe DevOps

---

## Changelog

| Date | Version | Description |
|------|---------|-------------|
| 2026-02-10 | 1.0.0 | Création initiale du pipeline CI/CD |

---

*Documentation générée pour Viralify - Plateforme de création de contenu viral*
