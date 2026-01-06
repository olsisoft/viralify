# 🚀 ViralTok Platform

> AI-Powered TikTok Content Creation & Optimization Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![Java](https://img.shields.io/badge/Java-21-orange)](https://openjdk.org/)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)

ViralTok is a comprehensive platform that leverages AI to help content creators maximize their TikTok presence through trend analysis, AI-powered content generation, smart scheduling, and detailed analytics.

## ✨ Features

### 🤖 Multi-Agent AI System
- **TrendScout**: Analyzes TikTok trends and identifies viral patterns
- **ScriptGenius**: Generates engaging video scripts with viral hooks
- **ContentOptimizer**: Optimizes captions, hashtags, and posting times
- **StrategyAdvisor**: Develops comprehensive content strategies

### 📊 Real-Time Trend Analysis
- Trending hashtags monitoring
- Viral sound tracking
- Pattern recognition
- Trend lifecycle prediction

### 📅 Smart Scheduling
- Optimal posting time recommendations
- TikTok Content Posting API integration
- Queue management with retry logic
- Rate limit compliance

### 📈 Advanced Analytics
- Performance tracking
- Engagement metrics
- Growth insights
- AI-powered recommendations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│                    Next.js 14 + React 18                        │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                       API GATEWAY                                │
│                  Spring Cloud Gateway                            │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                      MICROSERVICES                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Auth Service│  │   Trend     │  │  Content    │             │
│  │   (Java)    │  │  Analyzer   │  │  Generator  │             │
│  │   :8081     │  │  (Python)   │  │  (Python)   │             │
│  └─────────────┘  │   :8000     │  │   :8001     │             │
│                   └─────────────┘  └─────────────┘             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Scheduler  │  │  Analytics  │  │   TikTok    │             │
│  │   (Java)    │  │  (Python)   │  │  Connector  │             │
│  │   :8082     │  │   :8002     │  │   (Java)    │             │
│  └─────────────┘  └─────────────┘  │   :8083     │             │
│                                    └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  PostgreSQL 16 │ Redis 7 │ Elasticsearch 8 │ RabbitMQ          │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for frontend development)
- Java 21+ (for Java services development)
- Python 3.11+ (for Python services development)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/viraltok-platform.git
cd viraltok-platform
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Configure your API keys in `.env`:
```env
# TikTok API
TIKTOK_CLIENT_KEY=your_tiktok_client_key
TIKTOK_CLIENT_SECRET=your_tiktok_client_secret

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key

# JWT Secret
JWT_SECRET=your-super-secret-jwt-key
```

4. Start all services:
```bash
docker-compose up -d
```

5. Access the platform:
- **Frontend**: http://localhost:3000
- **API Gateway**: http://localhost:8080
- **RabbitMQ Management**: http://localhost:15672

## 🛠️ Development

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Python Services (e.g., Content Generator)

```bash
cd services/content-generator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

### Java Services (e.g., Auth Service)

```bash
cd services/auth-service
./mvnw spring-boot:run
```

## 📁 Project Structure

```
viraltok-platform/
├── frontend/                    # Next.js frontend application
│   ├── src/
│   │   ├── app/                # App router pages
│   │   │   ├── dashboard/      # Dashboard pages
│   │   │   │   ├── ai-chat/    # AI chat interface
│   │   │   │   ├── analytics/  # Analytics dashboard
│   │   │   │   ├── create/     # Content creation
│   │   │   │   ├── scheduler/  # Post scheduling
│   │   │   │   └── trends/     # Trends explorer
│   │   │   └── auth/           # Authentication pages
│   │   ├── components/         # Reusable components
│   │   ├── services/           # API services
│   │   └── stores/             # Zustand stores
│   └── package.json
│
├── services/
│   ├── api-gateway/            # Spring Cloud Gateway
│   ├── auth-service/           # Authentication (Java/Spring)
│   ├── trend-analyzer/         # Trend analysis (Python/FastAPI)
│   ├── content-generator/      # AI content generation (Python/LangChain)
│   ├── scheduler-service/      # Post scheduling (Java/Spring)
│   ├── analytics-service/      # Analytics (Python/FastAPI)
│   ├── tiktok-connector/       # TikTok API integration (Java/Spring)
│   └── notification-service/   # Notifications (Python/FastAPI)
│
├── infrastructure/
│   ├── docker/                 # Docker configurations
│   │   └── init.sql           # Database initialization
│   └── k8s/                   # Kubernetes manifests
│
├── docker-compose.yml          # Docker Compose configuration
└── README.md
```

## 🔌 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Login with email/password |
| GET | `/api/v1/auth/tiktok` | TikTok OAuth redirect |
| POST | `/api/v1/auth/refresh` | Refresh access token |

### Content Generation
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/generate/script` | Generate video script |
| POST | `/api/v1/generate/caption` | Generate caption |
| POST | `/api/v1/generate/hashtags` | Generate hashtags |
| POST | `/api/v1/chat` | Chat with AI agent |

### Scheduling
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/scheduler/posts` | Create scheduled post |
| GET | `/api/v1/scheduler/posts` | Get all posts |
| DELETE | `/api/v1/scheduler/posts/{id}` | Cancel post |

### Trends
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/trends/hashtags` | Get trending hashtags |
| GET | `/api/v1/trends/sounds` | Get trending sounds |
| GET | `/api/v1/trends/viral-patterns` | Get viral patterns |

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/analytics/summary` | Get analytics summary |
| GET | `/api/v1/analytics/insights` | Get AI insights |
| GET | `/api/v1/analytics/dashboard` | Get dashboard metrics |

## 🤖 AI Agents

### TrendScout
Specializes in analyzing TikTok trends and identifying viral patterns.

**Capabilities:**
- Trend detection
- Pattern analysis
- Viral prediction
- Niche matching

### ScriptGenius
Creates engaging TikTok video scripts with viral hooks.

**Capabilities:**
- Script writing
- Hook creation
- Storytelling
- CTA optimization

### ContentOptimizer
Optimizes content for maximum engagement and reach.

**Capabilities:**
- Caption optimization
- Hashtag strategy
- Timing analysis
- A/B testing recommendations

### StrategyAdvisor
Develops comprehensive content strategies for growth.

**Capabilities:**
- Strategy planning
- Competitor analysis
- KPI tracking
- Campaign planning

## 📊 Database Schema

Key entities:
- **users**: User accounts and TikTok connections
- **scheduled_posts**: Scheduled content for publishing
- **post_analytics**: Performance metrics
- **trending_hashtags**: Hashtag trend data
- **trending_sounds**: Sound trend data
- **ai_agents**: AI agent configurations
- **ai_generations**: Generated content history

## 🔒 Security

- TikTok OAuth 2.0 authentication
- JWT-based session management
- AES-256 encryption for tokens
- Rate limiting
- CORS configuration
- Input validation

## 💰 Cost Estimation

| Service | Startup | Scale |
|---------|---------|-------|
| Kubernetes | $200/mo | $800/mo |
| PostgreSQL | $50/mo | $200/mo |
| Redis | $30/mo | $100/mo |
| Elasticsearch | $100/mo | $300/mo |
| OpenAI API | $300/mo | $1,500/mo |
| Anthropic API | $200/mo | $800/mo |
| **Total** | **~$1,000/mo** | **~$4,000/mo** |

## 🗺️ Roadmap

- [x] Core microservices architecture
- [x] Multi-agent AI system
- [x] TikTok OAuth integration
- [x] Content scheduling
- [x] Trend analysis
- [x] Analytics dashboard
- [ ] Mobile app (React Native)
- [ ] Team collaboration features
- [ ] A/B testing automation
- [ ] Advanced ML predictions
- [ ] Multi-platform support (Instagram Reels, YouTube Shorts)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- Documentation: [docs.viraltok.app](https://docs.viraltok.app)
- Email: support@viraltok.app
- Discord: [Join our community](https://discord.gg/viraltok)

---

Built with ❤️ by the ViralTok Team
