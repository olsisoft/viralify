"""
Trend Analyzer Service - Real-time TikTok Trend Analysis
Analyzes hashtags, sounds, and content patterns for viral detection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4
import asyncio
import json
import os
import httpx
from collections import defaultdict

# Database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Integer, BigInteger, Float, Boolean, JSON, select, func, desc
from sqlalchemy.dialects.postgresql import UUID as PGUUID
import uuid

# Redis Cache
from redis.asyncio import Redis

# Elasticsearch
from elasticsearch import AsyncElasticsearch

# ML/Analysis
import numpy as np
from sklearn.linear_model import LinearRegression

# ========================================
# Configuration
# ========================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://tiktok_user:tiktok_secure_pass_2024@localhost:5432/tiktok_platform")
# Ensure async driver is used
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
REDIS_URL = os.getenv("REDIS_URL", "redis://:redis_secure_2024@localhost:6379/0")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
TIKTOK_CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY")
TIKTOK_CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET")

# ========================================
# FastAPI App
# ========================================

app = FastAPI(
    title="Trend Analyzer Service",
    description="Real-time TikTok trend analysis and viral pattern detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Database Setup
# ========================================

class Base(DeclarativeBase):
    pass

class TrendingHashtag(Base):
    __tablename__ = "trending_hashtags"
    
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hashtag: Mapped[str] = mapped_column(String(255), nullable=False)
    hashtag_normalized: Mapped[str] = mapped_column(String(255), nullable=False)
    region: Mapped[str] = mapped_column(String(10), default="global")
    category: Mapped[Optional[str]] = mapped_column(String(100))
    view_count: Mapped[int] = mapped_column(BigInteger, default=0)
    video_count: Mapped[int] = mapped_column(BigInteger, default=0)
    trend_score: Mapped[float] = mapped_column(Float, default=0)
    growth_rate: Mapped[float] = mapped_column(Float, default=0)
    is_trending: Mapped[bool] = mapped_column(Boolean, default=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class TrendingSound(Base):
    __tablename__ = "trending_sounds"
    
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sound_id: Mapped[str] = mapped_column(String(100), nullable=False)
    sound_title: Mapped[Optional[str]] = mapped_column(String(500))
    sound_author: Mapped[Optional[str]] = mapped_column(String(255))
    region: Mapped[str] = mapped_column(String(10), default="global")
    category: Mapped[Optional[str]] = mapped_column(String(100))
    usage_count: Mapped[int] = mapped_column(BigInteger, default=0)
    trend_score: Mapped[float] = mapped_column(Float, default=0)
    growth_rate: Mapped[float] = mapped_column(Float, default=0)
    is_trending: Mapped[bool] = mapped_column(Boolean, default=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class TrendSnapshot(Base):
    __tablename__ = "trend_snapshots"
    
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_type: Mapped[str] = mapped_column(String(50), nullable=False)
    region: Mapped[str] = mapped_column(String(10), default="global")
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    analysis: Mapped[dict] = mapped_column(JSON, default={})
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ========================================
# Redis & Elasticsearch Clients
# ========================================

redis_client: Optional[Redis] = None
es_client: Optional[AsyncElasticsearch] = None

@app.on_event("startup")
async def startup():
    global redis_client, es_client
    redis_client = Redis.from_url(REDIS_URL)
    es_client = AsyncElasticsearch([ELASTICSEARCH_URL])
    
    # Create Elasticsearch indices
    try:
        if not await es_client.indices.exists(index="trends"):
            await es_client.indices.create(
                index="trends",
                body={
                    "mappings": {
                        "properties": {
                            "type": {"type": "keyword"},
                            "name": {"type": "text"},
                            "region": {"type": "keyword"},
                            "category": {"type": "keyword"},
                            "score": {"type": "float"},
                            "growth_rate": {"type": "float"},
                            "timestamp": {"type": "date"}
                        }
                    }
                }
            )
    except Exception as e:
        print(f"Elasticsearch index creation error: {e}")

@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
    if es_client:
        await es_client.close()

# ========================================
# Pydantic Models
# ========================================

class TrendingHashtagResponse(BaseModel):
    id: UUID
    hashtag: str
    region: str
    category: Optional[str]
    view_count: int
    video_count: int
    trend_score: float
    growth_rate: float
    is_trending: bool
    captured_at: datetime

class TrendingSoundResponse(BaseModel):
    id: UUID
    sound_id: str
    sound_title: Optional[str]
    sound_author: Optional[str]
    region: str
    usage_count: int
    trend_score: float
    growth_rate: float
    is_trending: bool
    captured_at: datetime

class TrendAnalysis(BaseModel):
    trend_type: str
    name: str
    current_score: float
    growth_rate: float
    predicted_peak: Optional[datetime]
    longevity_days: int
    recommendation: str
    confidence: float

class NicheTrendsResponse(BaseModel):
    niche: str
    region: str
    hashtags: List[TrendingHashtagResponse]
    sounds: List[TrendingSoundResponse]
    formats: List[Dict[str, Any]]
    analysis: Dict[str, Any]

class ViralPatternsResponse(BaseModel):
    patterns: List[Dict[str, Any]]
    common_elements: List[str]
    optimal_duration: Dict[str, Any]
    best_posting_times: List[Dict[str, Any]]
    engagement_drivers: List[str]

class TrendPrediction(BaseModel):
    trend_name: str
    current_stage: str  # emerging, growing, peak, declining
    predicted_peak_date: Optional[datetime]
    days_until_peak: Optional[int]
    confidence_score: float
    recommendation: str

# ========================================
# Trend Analysis Engine
# ========================================

class TrendAnalyzer:
    """Core trend analysis logic"""
    
    def __init__(self):
        self.cache_ttl = 900  # 15 minutes
    
    async def calculate_trend_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate trend score based on multiple factors"""
        weights = {
            'view_velocity': 0.3,
            'engagement_rate': 0.25,
            'growth_rate': 0.25,
            'recency': 0.2
        }
        
        score = 0.0
        
        # View velocity (views per hour)
        if 'views' in metrics and 'hours_active' in metrics:
            velocity = metrics['views'] / max(metrics['hours_active'], 1)
            normalized_velocity = min(velocity / 100000, 1.0)  # Normalize to 0-1
            score += weights['view_velocity'] * normalized_velocity
        
        # Engagement rate
        if 'engagement_rate' in metrics:
            score += weights['engagement_rate'] * min(metrics['engagement_rate'] / 10, 1.0)
        
        # Growth rate (day over day)
        if 'growth_rate' in metrics:
            normalized_growth = (metrics['growth_rate'] + 1) / 2  # Normalize -1 to 1 → 0 to 1
            score += weights['growth_rate'] * normalized_growth
        
        # Recency boost
        if 'hours_since_start' in metrics:
            recency_factor = max(0, 1 - (metrics['hours_since_start'] / 168))  # Decay over 1 week
            score += weights['recency'] * recency_factor
        
        return round(score * 10, 2)  # Scale to 0-10
    
    async def predict_trend_lifecycle(self, historical_data: List[Dict]) -> TrendPrediction:
        """Predict trend lifecycle stage and peak"""
        if len(historical_data) < 3:
            return TrendPrediction(
                trend_name=historical_data[0].get('name', 'Unknown') if historical_data else 'Unknown',
                current_stage='emerging',
                predicted_peak_date=None,
                days_until_peak=None,
                confidence_score=0.3,
                recommendation="Insufficient data - monitor for 24-48 hours"
            )
        
        # Extract scores over time
        scores = [d.get('score', 0) for d in historical_data]
        times = list(range(len(scores)))
        
        # Calculate trend direction
        if len(scores) >= 2:
            recent_growth = (scores[-1] - scores[-2]) / max(scores[-2], 1)
            overall_growth = (scores[-1] - scores[0]) / max(scores[0], 1)
        else:
            recent_growth = 0
            overall_growth = 0
        
        # Determine lifecycle stage
        if recent_growth > 0.2 and overall_growth > 0.5:
            stage = 'emerging'
            days_to_peak = 3
            recommendation = "Jump on this trend NOW - high growth potential"
        elif recent_growth > 0 and overall_growth > 0:
            stage = 'growing'
            days_to_peak = 5
            recommendation = "Good time to create content - trend is building momentum"
        elif recent_growth < 0 and overall_growth > 0:
            stage = 'peak'
            days_to_peak = 0
            recommendation = "Trend is at peak - create content immediately if relevant"
        else:
            stage = 'declining'
            days_to_peak = None
            recommendation = "Trend is declining - only use if highly relevant to your niche"
        
        predicted_peak = datetime.utcnow() + timedelta(days=days_to_peak) if days_to_peak else None
        
        return TrendPrediction(
            trend_name=historical_data[0].get('name', 'Unknown'),
            current_stage=stage,
            predicted_peak_date=predicted_peak,
            days_until_peak=days_to_peak,
            confidence_score=0.7 if len(historical_data) >= 5 else 0.5,
            recommendation=recommendation
        )
    
    async def detect_viral_patterns(self, niche: str) -> ViralPatternsResponse:
        """Detect common patterns in viral content"""
        # In production, this would analyze actual viral videos
        patterns = [
            {
                "name": "Hook Pattern",
                "description": "Strong opening hook in first 2-3 seconds",
                "effectiveness": 0.85,
                "examples": ["POV:", "Wait for it...", "Nobody talks about this..."]
            },
            {
                "name": "Pattern Interrupt",
                "description": "Visual or audio changes every 3-5 seconds",
                "effectiveness": 0.78,
                "examples": ["Jump cuts", "Text overlays", "Sound effects"]
            },
            {
                "name": "Storytelling Arc",
                "description": "Problem → Solution → Result structure",
                "effectiveness": 0.82,
                "examples": ["Before/After", "Day in my life", "How I..."]
            },
            {
                "name": "Trending Sound Integration",
                "description": "Using trending audio strategically",
                "effectiveness": 0.75,
                "examples": ["Lip sync", "Background music", "Sound effects"]
            }
        ]
        
        return ViralPatternsResponse(
            patterns=patterns,
            common_elements=[
                "Strong hook (0-3 seconds)",
                "Pattern interrupts every 3-5 seconds",
                "Text overlays for accessibility",
                "Clear call-to-action",
                "Relatable content",
                "Trending sounds"
            ],
            optimal_duration={
                "short_form": {"min": 15, "max": 30, "best": 21},
                "medium_form": {"min": 30, "max": 60, "best": 45},
                "long_form": {"min": 60, "max": 180, "best": 90}
            },
            best_posting_times=[
                {"time": "07:00", "day": "all", "engagement_boost": 1.15},
                {"time": "12:00", "day": "weekday", "engagement_boost": 1.25},
                {"time": "19:00", "day": "all", "engagement_boost": 1.40},
                {"time": "21:00", "day": "weekend", "engagement_boost": 1.35}
            ],
            engagement_drivers=[
                "Ask questions in caption",
                "Use controversial hooks (ethically)",
                "Create FOMO",
                "Encourage duets/stitches",
                "Reply to comments with videos"
            ]
        )
    
    async def get_niche_recommendations(self, niche: str, user_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Get personalized trend recommendations for a niche"""
        recommendations = {
            "immediate_opportunities": [
                {
                    "type": "hashtag",
                    "name": f"#{niche}tips",
                    "reason": "Growing hashtag with moderate competition",
                    "action": "Create educational content"
                },
                {
                    "type": "format",
                    "name": "Tutorial/How-to",
                    "reason": "High engagement in this niche",
                    "action": "Share step-by-step guides"
                }
            ],
            "emerging_trends": [
                {
                    "name": f"#{niche}hack",
                    "growth_rate": 2.5,
                    "predicted_peak": "3-5 days",
                    "recommendation": "Create content within 48 hours"
                }
            ],
            "content_gaps": [
                f"Behind-the-scenes {niche} content",
                f"Common {niche} mistakes",
                f"{niche} myth busting"
            ],
            "optimal_strategy": {
                "posting_frequency": "1-2 times daily",
                "best_times": ["7 AM", "12 PM", "7 PM"],
                "content_mix": {
                    "educational": 40,
                    "entertaining": 30,
                    "trending": 20,
                    "personal": 10
                }
            }
        }
        
        return recommendations


trend_analyzer = TrendAnalyzer()

# ========================================
# API Endpoints
# ========================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trend-analyzer"}

@app.get("/api/v1/trends/hashtags", response_model=List[TrendingHashtagResponse])
async def get_trending_hashtags(
    region: str = Query(default="global", description="Region code"),
    category: Optional[str] = Query(default=None, description="Category filter"),
    limit: int = Query(default=20, ge=1, le=100),
    min_score: float = Query(default=0, ge=0, le=10)
):
    """Get trending hashtags with optional filters"""
    
    # Check cache first
    cache_key = f"trends:hashtags:{region}:{category}:{limit}"
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    
    async with async_session() as session:
        query = select(TrendingHashtag).where(
            TrendingHashtag.is_trending == True,
            TrendingHashtag.region == region,
            TrendingHashtag.trend_score >= min_score
        )
        
        if category:
            query = query.where(TrendingHashtag.category == category)
        
        query = query.order_by(desc(TrendingHashtag.trend_score)).limit(limit)
        
        result = await session.execute(query)
        hashtags = result.scalars().all()
        
        response = [
            TrendingHashtagResponse(
                id=h.id,
                hashtag=h.hashtag,
                region=h.region,
                category=h.category,
                view_count=h.view_count,
                video_count=h.video_count,
                trend_score=h.trend_score,
                growth_rate=h.growth_rate,
                is_trending=h.is_trending,
                captured_at=h.captured_at
            ) for h in hashtags
        ]
        
        # Cache result
        if redis_client:
            await redis_client.setex(cache_key, trend_analyzer.cache_ttl, json.dumps([r.dict() for r in response], default=str))
        
        return response

@app.get("/api/v1/trends/sounds", response_model=List[TrendingSoundResponse])
async def get_trending_sounds(
    region: str = Query(default="global"),
    category: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100)
):
    """Get trending sounds"""
    
    cache_key = f"trends:sounds:{region}:{category}:{limit}"
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    
    async with async_session() as session:
        query = select(TrendingSound).where(
            TrendingSound.is_trending == True,
            TrendingSound.region == region
        )
        
        if category:
            query = query.where(TrendingSound.category == category)
        
        query = query.order_by(desc(TrendingSound.trend_score)).limit(limit)
        
        result = await session.execute(query)
        sounds = result.scalars().all()
        
        response = [
            TrendingSoundResponse(
                id=s.id,
                sound_id=s.sound_id,
                sound_title=s.sound_title,
                sound_author=s.sound_author,
                region=s.region,
                usage_count=s.usage_count,
                trend_score=s.trend_score,
                growth_rate=s.growth_rate,
                is_trending=s.is_trending,
                captured_at=s.captured_at
            ) for s in sounds
        ]
        
        if redis_client:
            await redis_client.setex(cache_key, trend_analyzer.cache_ttl, json.dumps([r.dict() for r in response], default=str))
        
        return response

@app.get("/api/v1/trends/niche/{niche}", response_model=NicheTrendsResponse)
async def get_niche_trends(
    niche: str,
    region: str = Query(default="global")
):
    """Get all trends relevant to a specific niche"""
    
    cache_key = f"trends:niche:{niche}:{region}"
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    
    # Get hashtags related to niche
    async with async_session() as session:
        hashtag_query = select(TrendingHashtag).where(
            TrendingHashtag.is_trending == True,
            TrendingHashtag.region == region,
            TrendingHashtag.hashtag.ilike(f"%{niche}%")
        ).order_by(desc(TrendingHashtag.trend_score)).limit(10)
        
        hashtag_result = await session.execute(hashtag_query)
        hashtags = hashtag_result.scalars().all()
        
        # Get sounds
        sound_query = select(TrendingSound).where(
            TrendingSound.is_trending == True,
            TrendingSound.region == region
        ).order_by(desc(TrendingSound.trend_score)).limit(10)
        
        sound_result = await session.execute(sound_query)
        sounds = sound_result.scalars().all()
    
    # Get recommendations
    recommendations = await trend_analyzer.get_niche_recommendations(niche)
    
    response = NicheTrendsResponse(
        niche=niche,
        region=region,
        hashtags=[
            TrendingHashtagResponse(
                id=h.id, hashtag=h.hashtag, region=h.region, category=h.category,
                view_count=h.view_count, video_count=h.video_count,
                trend_score=h.trend_score, growth_rate=h.growth_rate,
                is_trending=h.is_trending, captured_at=h.captured_at
            ) for h in hashtags
        ],
        sounds=[
            TrendingSoundResponse(
                id=s.id, sound_id=s.sound_id, sound_title=s.sound_title,
                sound_author=s.sound_author, region=s.region,
                usage_count=s.usage_count, trend_score=s.trend_score,
                growth_rate=s.growth_rate, is_trending=s.is_trending,
                captured_at=s.captured_at
            ) for s in sounds
        ],
        formats=[
            {"name": "Educational", "popularity": 0.85, "avg_engagement": 7.5},
            {"name": "Tutorial", "popularity": 0.80, "avg_engagement": 7.2},
            {"name": "Storytelling", "popularity": 0.75, "avg_engagement": 8.0}
        ],
        analysis=recommendations
    )
    
    if redis_client:
        await redis_client.setex(cache_key, trend_analyzer.cache_ttl, response.json())
    
    return response

@app.get("/api/v1/trends/viral-patterns", response_model=ViralPatternsResponse)
async def get_viral_patterns(
    niche: str = Query(default="general")
):
    """Get viral content patterns for a niche"""
    return await trend_analyzer.detect_viral_patterns(niche)

@app.get("/api/v1/trends/predict/{trend_name}", response_model=TrendPrediction)
async def predict_trend(trend_name: str):
    """Predict trend lifecycle and provide recommendations"""
    
    # In production, fetch historical data from database
    # Mock historical data for demonstration
    historical_data = [
        {"name": trend_name, "score": 3.0, "timestamp": datetime.utcnow() - timedelta(days=3)},
        {"name": trend_name, "score": 5.0, "timestamp": datetime.utcnow() - timedelta(days=2)},
        {"name": trend_name, "score": 7.0, "timestamp": datetime.utcnow() - timedelta(days=1)},
        {"name": trend_name, "score": 8.5, "timestamp": datetime.utcnow()}
    ]
    
    return await trend_analyzer.predict_trend_lifecycle(historical_data)

@app.post("/api/v1/trends/analyze")
async def analyze_custom_trend(
    trend_data: Dict[str, Any]
):
    """Analyze custom trend data"""
    metrics = {
        'views': trend_data.get('views', 0),
        'hours_active': trend_data.get('hours_active', 24),
        'engagement_rate': trend_data.get('engagement_rate', 5),
        'growth_rate': trend_data.get('growth_rate', 0.1),
        'hours_since_start': trend_data.get('hours_since_start', 24)
    }
    
    score = await trend_analyzer.calculate_trend_score(metrics)
    
    return {
        "trend_score": score,
        "metrics_analyzed": metrics,
        "recommendation": "Strong trend - act now!" if score > 7 else "Moderate trend - consider if relevant" if score > 4 else "Weak trend - monitor only"
    }

@app.get("/api/v1/trends/best-times")
async def get_best_posting_times(
    timezone: str = Query(default="UTC"),
    niche: Optional[str] = Query(default=None)
):
    """Get optimal posting times"""
    
    base_times = [
        {"hour": 7, "minute": 0, "engagement_multiplier": 1.15, "day_type": "all"},
        {"hour": 12, "minute": 0, "engagement_multiplier": 1.25, "day_type": "weekday"},
        {"hour": 15, "minute": 0, "engagement_multiplier": 1.10, "day_type": "all"},
        {"hour": 19, "minute": 0, "engagement_multiplier": 1.40, "day_type": "all"},
        {"hour": 21, "minute": 0, "engagement_multiplier": 1.35, "day_type": "weekend"}
    ]
    
    return {
        "timezone": timezone,
        "niche": niche,
        "optimal_times": base_times,
        "recommendation": "Focus on 7 PM posts for maximum engagement"
    }

@app.post("/api/v1/trends/refresh")
async def refresh_trends(background_tasks: BackgroundTasks):
    """Trigger a refresh of trend data"""
    background_tasks.add_task(fetch_and_update_trends)
    return {"status": "Trend refresh initiated"}

async def fetch_and_update_trends():
    """Background task to fetch and update trends"""
    # In production, this would call TikTok Research API
    # and update the database with fresh trend data
    print("Refreshing trend data...")
    await asyncio.sleep(1)
    print("Trend refresh complete")

# ========================================
# Elasticsearch Search Endpoints
# ========================================

@app.get("/api/v1/search/trends")
async def search_trends(
    query: str = Query(..., min_length=2),
    trend_type: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100)
):
    """Full-text search across all trends"""
    
    if not es_client:
        raise HTTPException(status_code=503, detail="Search service unavailable")
    
    search_body = {
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {"query": query, "fields": ["name^2", "category"]}}
                ]
            }
        },
        "size": limit,
        "sort": [{"score": "desc"}]
    }
    
    if trend_type:
        search_body["query"]["bool"]["filter"] = [{"term": {"type": trend_type}}]
    
    try:
        result = await es_client.search(index="trends", body=search_body)
        hits = result.get("hits", {}).get("hits", [])
        
        return {
            "total": result.get("hits", {}).get("total", {}).get("value", 0),
            "results": [hit["_source"] for hit in hits]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# Run Application
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
