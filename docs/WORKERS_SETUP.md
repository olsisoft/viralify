# Guide de Déploiement des Workers Viralify

Ce guide explique comment déployer des workers Viralify sur des serveurs distants pour la génération de cours en parallèle.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SERVEUR PRINCIPAL                        │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │PostgreSQL│  │  Redis   │  │ RabbitMQ │  │ Frontend │   │
│  │  :5432   │  │  :6379   │  │  :5672   │  │  :3000   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│   WORKER SERVER 1   │     │   WORKER SERVER 2   │
│                     │     │                     │
│  ┌───────────────┐  │     │  ┌───────────────┐  │
│  │ course-worker │  │     │  │ course-worker │  │
│  │    (x4)       │  │     │  │    (x4)       │  │
│  └───────────────┘  │     │  └───────────────┘  │
│  ┌───────────────┐  │     │  ┌───────────────┐  │
│  │ presentation- │  │     │  │ presentation- │  │
│  │  generator    │  │     │  │  generator    │  │
│  └───────────────┘  │     │  └───────────────┘  │
│  ┌───────────────┐  │     │  ┌───────────────┐  │
│  │media-generator│  │     │  │media-generator│  │
│  └───────────────┘  │     │  └───────────────┘  │
└─────────────────────┘     └─────────────────────┘
```

---

## Prérequis

### Sur le serveur principal
- PostgreSQL, Redis, RabbitMQ en fonctionnement
- Ports 5432, 5672, 6379 accessibles depuis les workers
- Fichier `.env` configuré

### Sur chaque serveur worker
- Ubuntu 20.04+ ou Debian 11+
- Accès root (sudo)
- Connexion internet

---

## Étape 1 : Préparer le serveur principal

### 1.1 Ouvrir les ports du firewall

Sur le serveur principal, autorisez les connexions depuis les workers :

```bash
# Remplacez par les IPs de vos workers
WORKER1_IP="xxx.xxx.xxx.xxx"
WORKER2_IP="xxx.xxx.xxx.xxx"

# Ouvrir les ports
sudo ufw allow from $WORKER1_IP to any port 5432  # PostgreSQL
sudo ufw allow from $WORKER1_IP to any port 5672  # RabbitMQ
sudo ufw allow from $WORKER1_IP to any port 6379  # Redis

sudo ufw allow from $WORKER2_IP to any port 5432
sudo ufw allow from $WORKER2_IP to any port 5672
sudo ufw allow from $WORKER2_IP to any port 6379

# Vérifier
sudo ufw status
```

### 1.2 Vérifier que les services écoutent sur toutes les interfaces

```bash
# Vérifier que les services sont accessibles (pas juste localhost)
sudo netstat -tlnp | grep -E "5432|5672|6379"
```

Vous devez voir `0.0.0.0:5432`, `0.0.0.0:5672`, `0.0.0.0:6379` (pas `127.0.0.1`).

### 1.3 Noter les informations de connexion

Récupérez ces informations depuis votre `.env` sur le serveur principal :

```bash
cat /opt/viralify/.env | grep -E "DB_|RABBIT|REDIS"
```

Notez :
- `DB_PASSWORD`
- `RABBITMQ_PASSWORD`
- `REDIS_PASSWORD`
- L'IP publique du serveur principal

---

## Étape 2 : Configurer le premier worker

### 2.1 Se connecter au serveur worker

```bash
ssh user@<IP_WORKER_1>
```

### 2.2 Lancer le script de setup

```bash
curl -fsSL https://raw.githubusercontent.com/olsisoft/viralify/master/setup-worker.sh | sudo bash
```

Le script va :
1. Installer Docker automatiquement
2. Cloner le repository dans `/opt/viralify`
3. Créer le fichier `.env.workers` depuis le template
4. S'arrêter pour vous permettre de configurer

### 2.3 Configurer les variables d'environnement

```bash
sudo nano /opt/viralify/.env.workers
```

Remplissez avec vos vraies valeurs :

```env
# ===========================================
# CONNEXION AU SERVEUR PRINCIPAL
# ===========================================
MAIN_SERVER_HOST=<IP_SERVEUR_PRINCIPAL>

# ===========================================
# BASE DE DONNÉES
# ===========================================
DB_USER=tiktok_user
DB_PASSWORD=<VOTRE_MOT_DE_PASSE_DB>
DB_NAME=tiktok_platform

# ===========================================
# RABBITMQ
# ===========================================
RABBITMQ_USER=tiktok
RABBITMQ_PASSWORD=<VOTRE_MOT_DE_PASSE_RABBITMQ>

# ===========================================
# REDIS
# ===========================================
REDIS_PASSWORD=<VOTRE_MOT_DE_PASSE_REDIS>

# ===========================================
# CLÉS API
# ===========================================
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
ELEVENLABS_API_KEY=xxxxxxxxxxxxxxxx

# Optionnel - autres providers LLM
# LLM_PROVIDER=openai
# GROQ_API_KEY=
# DEEPSEEK_API_KEY=

# ===========================================
# SCALING
# ===========================================
# Nombre de workers en parallèle (recommandé: nb_cores / 2)
WORKER_REPLICAS=4
```

Sauvegardez : `Ctrl+O`, `Enter`, `Ctrl+X`

### 2.4 Relancer le script de setup

```bash
cd /opt/viralify
sudo ./setup-worker.sh
```

Le script va :
1. Tester la connexion au serveur principal
2. Builder les images Docker
3. Démarrer les workers

### 2.5 Vérifier que tout fonctionne

```bash
# Voir les containers
docker compose -f docker-compose.workers.yml ps

# Voir les logs
docker compose -f docker-compose.workers.yml logs -f course-worker
```

---

## Étape 3 : Configurer le deuxième worker

Répétez les étapes 2.1 à 2.5 sur le deuxième serveur.

---

## Commandes utiles

### Gestion des workers

```bash
cd /opt/viralify

# Voir le status
docker compose -f docker-compose.workers.yml ps

# Voir les logs en temps réel
docker compose -f docker-compose.workers.yml logs -f

# Logs d'un service spécifique
docker compose -f docker-compose.workers.yml logs -f course-worker

# Redémarrer les workers
docker compose -f docker-compose.workers.yml restart

# Arrêter les workers
docker compose -f docker-compose.workers.yml down

# Mettre à jour (après un git pull)
git pull origin master
docker compose -f docker-compose.workers.yml up -d --build
```

### Scaling horizontal

```bash
# Augmenter le nombre de workers (ex: 6 workers)
docker compose -f docker-compose.workers.yml up -d --scale course-worker=6

# Réduire (ex: 2 workers)
docker compose -f docker-compose.workers.yml up -d --scale course-worker=2
```

### Monitoring

```bash
# Utilisation CPU/RAM des containers
docker stats

# Espace disque
df -h

# Voir les jobs en cours (sur le serveur principal)
redis-cli -a <REDIS_PASSWORD> KEYS "job:*"
```

---

## Dépannage

### Erreur : Connection refused (PostgreSQL/Redis/RabbitMQ)

1. Vérifiez que les ports sont ouverts sur le serveur principal :
   ```bash
   sudo ufw status
   ```

2. Vérifiez que les services écoutent sur 0.0.0.0 :
   ```bash
   sudo netstat -tlnp | grep -E "5432|5672|6379"
   ```

3. Testez la connexion depuis le worker :
   ```bash
   nc -zv <IP_PRINCIPAL> 5432
   nc -zv <IP_PRINCIPAL> 5672
   nc -zv <IP_PRINCIPAL> 6379
   ```

### Erreur : Docker build failed

```bash
# Nettoyer le cache Docker
docker system prune -a

# Rebuilder
docker compose -f docker-compose.workers.yml build --no-cache
```

### Erreur : Out of memory

Réduisez le nombre de workers :
```bash
# Dans .env.workers
WORKER_REPLICAS=2
```

### Les jobs ne sont pas traités

1. Vérifiez que RabbitMQ est accessible :
   ```bash
   docker compose -f docker-compose.workers.yml logs course-worker | grep -i "rabbit\|amqp"
   ```

2. Vérifiez la queue sur le serveur principal :
   ```bash
   # Sur le serveur principal
   docker exec -it viralify-rabbitmq rabbitmqctl list_queues
   ```

---

## Recommandations de performance

| RAM Serveur | Workers recommandés |
|-------------|---------------------|
| 8 GB        | 2-3 workers         |
| 16 GB       | 4-6 workers         |
| 24 GB       | 6-8 workers         |
| 32 GB       | 8-12 workers        |

Pour vos serveurs OVH (8 cores / 24 GB) :
- **WORKER_REPLICAS=6** est un bon point de départ
- Surveillez avec `docker stats` et ajustez

---

## Mise à jour des workers

Quand une nouvelle version est disponible :

```bash
cd /opt/viralify

# Arrêter les workers
docker compose -f docker-compose.workers.yml down

# Mettre à jour le code
git pull origin master

# Rebuilder et relancer
docker compose -f docker-compose.workers.yml up -d --build

# Vérifier
docker compose -f docker-compose.workers.yml ps
```

---

## Support

- **Logs** : `docker compose -f docker-compose.workers.yml logs -f`
- **GitHub Issues** : https://github.com/olsisoft/viralify/issues
