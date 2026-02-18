"""
Analytics Service - Performance tracking and insights
Collects, analyzes, and reports on content performance
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from decimal import Decimal
import asyncio
import json
import os

# Database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Integer, BigInteger, Float, Boolean, JSON, ForeignKey, select, func, desc
from sqlalchemy.dialects.postgresql import UUID as PGUUID
import uuid

# Redis & Elasticsearch
from redis.asyncio import Redis
from elasticsearch import AsyncElasticsearch

# Message Queue
import aio_pika

# ========================================
# Configuration
# ========================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://viralify_prod:password@localhost:5432/viralify_production")
# Ensure async driver is used
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
REDIS_URL = os.getenv("REDIS_URL", "redis://:redis_secure_2024@localhost:6379/2")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://tiktok:rabbitmq_secure_2024@localhost:5672/")

# ========================================
# FastAPI App
# ========================================

app = FastAPI(
    title="Analytics Service",
    description="Performance tracking, insights, and reporting for TikTok content",
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
# Database Models
# ========================================

class Base(DeclarativeBase):
    pass

class PostAnalytics(Base):
    __tablename__ = "post_analytics"
    
    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    user_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    tiktok_post_id: Mapped[Optional[str]] = mapped_column(String(100))
    views: Mapped[int] = mapped_column(BigInteger, default=0)
    likes: Mapped[int] = mapped_column(BigInteger, default=0)
    comments: Mapped[int] = mapped_column(BigInteger, default=0)
    shares: Mapped[int] = mapped_column(BigInteger, default=0)
    saves: Mapped[int] = mapped_column(BigInteger, default=0)
    reach: Mapped[int] = mapped_column(BigInteger, default=0)
    impressions: Mapped[int] = mapped_column(BigInteger, default=0)
    engagement_rate: Mapped[float] = mapped_column(Float, default=0)
    avg_watch_time_seconds: Mapped[float] = mapped_column(Float, default=0)
    completion_rate: Mapped[float] = mapped_column(Float, default=0)
    audience_demographics: Mapped[dict] = mapped_column(JSON, default={})
    traffic_sources: Mapped[dict] = mapped_column(JSON, default={})
    captured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Redis & ES clients
redis_client: Optional[Redis] = None
es_client: Optional[AsyncElasticsearch] = None

@app.on_event("startup")
async def startup():
    global redis_client, es_client
    redis_client = Redis.from_url(REDIS_URL)
    es_client = AsyncElasticsearch([ELASTICSEARCH_URL])

@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
    if es_client:
        await es_client.close()

# ========================================
# Pydantic Models
# ========================================

class PostAnalyticsCreate(BaseModel):
    post_id: UUID
    tiktok_post_id: Optional[str] = None
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    reach: int = 0
    impressions: int = 0
    avg_watch_time_seconds: float = 0
    completion_rate: float = 0
    audience_demographics: Optional[Dict[str, Any]] = None
    traffic_sources: Optional[Dict[str, Any]] = None

class PostAnalyticsResponse(BaseModel):
    id: UUID
    post_id: UUID
    tiktok_post_id: Optional[str]
    views: int
    likes: int
    comments: int
    shares: int
    saves: int
    engagement_rate: float
    avg_watch_time_seconds: float
    completion_rate: float
    captured_at: datetime

class DashboardStats(BaseModel):
    total_views: int
    total_likes: int
    total_comments: int
    total_shares: int
    total_posts: int
    avg_engagement_rate: float
    best_performing_post: Optional[Dict[str, Any]]
    growth_metrics: Dict[str, float]
    posting_frequency: Dict[str, int]

class PerformanceReport(BaseModel):
    period: str
    start_date: datetime
    end_date: datetime
    summary: Dict[str, Any]
    top_posts: List[Dict[str, Any]]
    trends: Dict[str, Any]
    recommendations: List[str]

class ContentInsight(BaseModel):
    insight_type: str
    title: str
    description: str
    data: Dict[str, Any]
    action_items: List[str]

class AudienceAnalytics(BaseModel):
    demographics: Dict[str, Any]
    geography: Dict[str, Any]
    active_times: List[Dict[str, Any]]
    interests: List[Dict[str, float]]
    growth_trend: List[Dict[str, Any]]

# ========================================
# NEW: Viral Prediction Models
# ========================================

class ViralPredictionRequest(BaseModel):
    title: str
    caption: str
    hashtags: List[str] = []
    video_duration_seconds: int = 30
    has_trending_sound: bool = False
    has_hook: bool = True
    posting_hour: int = 19  # 7 PM default
    niche: Optional[str] = None

class ViralPredictionResponse(BaseModel):
    viral_score: float  # 0-100
    predicted_views_low: int
    predicted_views_high: int
    factors: Dict[str, Any]
    recommendations: List[str]
    confidence: float

class NicheBenchmark(BaseModel):
    niche: str
    platform: str
    avg_engagement_rate: float
    avg_views: int
    top_posting_times: List[Dict[str, Any]]
    top_hashtags: List[str]
    content_length_avg: int

class ContentGapAnalysis(BaseModel):
    missing_content_types: List[str]
    underperforming_areas: List[Dict[str, Any]]
    competitor_insights: List[Dict[str, Any]]
    opportunities: List[str]
    recommended_topics: List[str]

class UserPerformanceSnapshot(BaseModel):
    user_id: str
    period: str
    total_views: int
    total_engagement: int
    avg_engagement_rate: float
    best_performing_time: str
    most_used_hashtags: List[str]
    growth_rate: float
    compared_to_niche: Dict[str, Any]

# ========================================
# Analytics Engine
# ========================================

class AnalyticsEngine:
    """Core analytics processing logic"""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes
    
    async def calculate_engagement_rate(self, views: int, likes: int, comments: int, shares: int, saves: int) -> float:
        """Calculate engagement rate"""
        if views == 0:
            return 0.0
        
        total_engagement = likes + comments + shares + saves
        rate = (total_engagement / views) * 100
        return round(rate, 4)
    
    async def get_dashboard_stats(self, user_id: UUID, days: int = 30) -> DashboardStats:
        """Get comprehensive dashboard statistics"""
        
        cache_key = f"analytics:dashboard:{user_id}:{days}"
        if redis_client:
            cached = await redis_client.get(cache_key)
            if cached:
                return DashboardStats(**json.loads(cached))
        
        async with async_session() as session:
            since = datetime.utcnow() - timedelta(days=days)
            
            # Get aggregated metrics
            result = await session.execute(
                select(
                    func.sum(PostAnalytics.views).label('total_views'),
                    func.sum(PostAnalytics.likes).label('total_likes'),
                    func.sum(PostAnalytics.comments).label('total_comments'),
                    func.sum(PostAnalytics.shares).label('total_shares'),
                    func.count(func.distinct(PostAnalytics.post_id)).label('total_posts'),
                    func.avg(PostAnalytics.engagement_rate).label('avg_engagement')
                ).where(
                    PostAnalytics.user_id == user_id,
                    PostAnalytics.captured_at >= since
                )
            )
            
            row = result.first()
            
            # Get best performing post
            best_post_result = await session.execute(
                select(PostAnalytics)
                .where(
                    PostAnalytics.user_id == user_id,
                    PostAnalytics.captured_at >= since
                )
                .order_by(desc(PostAnalytics.views))
                .limit(1)
            )
            best_post = best_post_result.scalar_one_or_none()
            
            # Calculate growth (compare with previous period)
            previous_since = since - timedelta(days=days)
            prev_result = await session.execute(
                select(
                    func.sum(PostAnalytics.views).label('views'),
                    func.sum(PostAnalytics.likes).label('likes')
                ).where(
                    PostAnalytics.user_id == user_id,
                    PostAnalytics.captured_at >= previous_since,
                    PostAnalytics.captured_at < since
                )
            )
            prev_row = prev_result.first()
            
            views_growth = 0.0
            likes_growth = 0.0
            
            if prev_row and prev_row.views and prev_row.views > 0:
                views_growth = ((row.total_views or 0) - prev_row.views) / prev_row.views * 100
            if prev_row and prev_row.likes and prev_row.likes > 0:
                likes_growth = ((row.total_likes or 0) - prev_row.likes) / prev_row.likes * 100
            
            stats = DashboardStats(
                total_views=row.total_views or 0,
                total_likes=row.total_likes or 0,
                total_comments=row.total_comments or 0,
                total_shares=row.total_shares or 0,
                total_posts=row.total_posts or 0,
                avg_engagement_rate=round(row.avg_engagement or 0, 2),
                best_performing_post={
                    "post_id": str(best_post.post_id),
                    "views": best_post.views,
                    "engagement_rate": best_post.engagement_rate
                } if best_post else None,
                growth_metrics={
                    "views_growth": round(views_growth, 2),
                    "likes_growth": round(likes_growth, 2)
                },
                posting_frequency={
                    "daily_avg": round((row.total_posts or 0) / max(days, 1), 1),
                    "total_days": days
                }
            )
            
            if redis_client:
                await redis_client.setex(cache_key, self.cache_ttl, stats.json())
            
            return stats
    
    async def generate_performance_report(self, user_id: UUID, period: str = "weekly") -> PerformanceReport:
        """Generate detailed performance report"""
        
        days = 7 if period == "weekly" else 30 if period == "monthly" else 1
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        async with async_session() as session:
            # Get all analytics for the period
            result = await session.execute(
                select(PostAnalytics)
                .where(
                    PostAnalytics.user_id == user_id,
                    PostAnalytics.captured_at >= start_date,
                    PostAnalytics.captured_at <= end_date
                )
                .order_by(desc(PostAnalytics.views))
            )
            analytics = result.scalars().all()
            
            total_views = sum(a.views for a in analytics)
            total_likes = sum(a.likes for a in analytics)
            total_comments = sum(a.comments for a in analytics)
            total_shares = sum(a.shares for a in analytics)
            avg_engagement = sum(a.engagement_rate for a in analytics) / len(analytics) if analytics else 0
            
            top_posts = [
                {
                    "post_id": str(a.post_id),
                    "views": a.views,
                    "likes": a.likes,
                    "engagement_rate": a.engagement_rate,
                    "captured_at": a.captured_at.isoformat()
                }
                for a in analytics[:5]
            ]
            
            # Generate recommendations based on data
            recommendations = []
            if avg_engagement < 3:
                recommendations.append("Focus on creating more engaging hooks in the first 3 seconds")
            if total_posts := len(set(a.post_id for a in analytics)) < days * 0.5:
                recommendations.append("Increase posting frequency for better algorithm visibility")
            if total_shares < total_likes * 0.1:
                recommendations.append("Add stronger calls-to-action to encourage sharing")
            
            return PerformanceReport(
                period=period,
                start_date=start_date,
                end_date=end_date,
                summary={
                    "total_views": total_views,
                    "total_likes": total_likes,
                    "total_comments": total_comments,
                    "total_shares": total_shares,
                    "avg_engagement_rate": round(avg_engagement, 2),
                    "total_posts": len(set(a.post_id for a in analytics))
                },
                top_posts=top_posts,
                trends={
                    "views_trend": "up" if total_views > 1000 else "stable",
                    "engagement_trend": "up" if avg_engagement > 5 else "stable"
                },
                recommendations=recommendations if recommendations else ["Keep up the great work!"]
            )
    
    async def get_content_insights(self, user_id: UUID) -> List[ContentInsight]:
        """Generate AI-powered content insights"""
        
        insights = []
        
        async with async_session() as session:
            # Get recent analytics
            result = await session.execute(
                select(PostAnalytics)
                .where(PostAnalytics.user_id == user_id)
                .order_by(desc(PostAnalytics.captured_at))
                .limit(50)
            )
            analytics = result.scalars().all()
            
            if not analytics:
                return [ContentInsight(
                    insight_type="onboarding",
                    title="Start Posting!",
                    description="Create your first post to start receiving insights",
                    data={},
                    action_items=["Create your first TikTok post", "Connect your TikTok account"]
                )]
            
            # Best performing content type analysis
            avg_views = sum(a.views for a in analytics) / len(analytics)
            top_performers = [a for a in analytics if a.views > avg_views * 1.5]
            
            if top_performers:
                insights.append(ContentInsight(
                    insight_type="performance",
                    title="Your Top Performing Content",
                    description=f"You have {len(top_performers)} posts performing above average",
                    data={
                        "top_count": len(top_performers),
                        "avg_views": avg_views,
                        "top_avg_views": sum(p.views for p in top_performers) / len(top_performers)
                    },
                    action_items=[
                        "Analyze what made these posts successful",
                        "Create similar content formats",
                        "Post at similar times"
                    ]
                ))
            
            # Engagement optimization insight
            avg_engagement = sum(a.engagement_rate for a in analytics) / len(analytics)
            insights.append(ContentInsight(
                insight_type="engagement",
                title="Engagement Optimization",
                description=f"Your average engagement rate is {avg_engagement:.2f}%",
                data={
                    "current_rate": avg_engagement,
                    "benchmark": 5.0,  # Industry average
                    "gap": max(0, 5.0 - avg_engagement)
                },
                action_items=[
                    "Use pattern interrupts every 3-5 seconds",
                    "Add questions in captions to drive comments",
                    "Include clear CTAs"
                ] if avg_engagement < 5 else ["Maintain your excellent engagement!"]
            ))
            
            # Posting consistency insight
            posting_days = len(set(a.captured_at.date() for a in analytics))
            insights.append(ContentInsight(
                insight_type="consistency",
                title="Posting Consistency",
                description=f"You've posted on {posting_days} different days recently",
                data={
                    "active_days": posting_days,
                    "consistency_score": min(posting_days / 30 * 100, 100)
                },
                action_items=[
                    "Aim for daily posting",
                    "Use the scheduler to maintain consistency",
                    "Batch create content for busy weeks"
                ]
            ))
        
        return insights
    
    async def get_audience_analytics(self, user_id: UUID) -> AudienceAnalytics:
        """Get audience demographics and behavior analytics"""
        
        async with async_session() as session:
            result = await session.execute(
                select(PostAnalytics)
                .where(PostAnalytics.user_id == user_id)
                .order_by(desc(PostAnalytics.captured_at))
                .limit(30)
            )
            analytics = result.scalars().all()
            
            # Aggregate demographics from posts
            age_groups = {"18-24": 0, "25-34": 0, "35-44": 0, "45+": 0}
            genders = {"male": 0, "female": 0, "other": 0}
            
            for a in analytics:
                if a.audience_demographics:
                    for age, pct in a.audience_demographics.get("age", {}).items():
                        if age in age_groups:
                            age_groups[age] += pct
                    for gender, pct in a.audience_demographics.get("gender", {}).items():
                        if gender in genders:
                            genders[gender] += pct
            
            # Normalize
            total_posts = len(analytics) or 1
            age_groups = {k: round(v / total_posts, 1) for k, v in age_groups.items()}
            genders = {k: round(v / total_posts, 1) for k, v in genders.items()}
            
            return AudienceAnalytics(
                demographics={
                    "age_groups": age_groups,
                    "genders": genders
                },
                geography={
                    "top_countries": ["US", "UK", "Canada", "Australia"],
                    "top_cities": ["New York", "Los Angeles", "London"]
                },
                active_times=[
                    {"hour": 7, "engagement_multiplier": 1.15},
                    {"hour": 12, "engagement_multiplier": 1.25},
                    {"hour": 19, "engagement_multiplier": 1.40},
                    {"hour": 21, "engagement_multiplier": 1.35}
                ],
                interests=[
                    {"category": "Entertainment", "score": 0.85},
                    {"category": "Education", "score": 0.72},
                    {"category": "Lifestyle", "score": 0.68}
                ],
                growth_trend=[
                    {"date": (datetime.utcnow() - timedelta(days=i)).isoformat(), "followers": 1000 + i * 50}
                    for i in range(30, 0, -1)
                ]
            )


    async def predict_viral_score(self, request: ViralPredictionRequest) -> ViralPredictionResponse:
        """Predict viral potential of content before posting"""

        # Viral scoring factors (weights sum to 100)
        factors = {
            "hook_strength": 0,      # 25% weight
            "trend_alignment": 0,    # 20% weight
            "posting_time": 0,       # 15% weight
            "hashtag_mix": 0,        # 15% weight
            "caption_quality": 0,    # 10% weight
            "audio_choice": 0,       # 10% weight
            "video_length": 0,       # 5% weight
        }

        recommendations = []

        # Hook strength (25%)
        if request.has_hook:
            factors["hook_strength"] = 20
        else:
            factors["hook_strength"] = 8
            recommendations.append("Add a strong hook in the first 3 seconds to grab attention")

        # Trend alignment (20%)
        if request.has_trending_sound:
            factors["trend_alignment"] = 18
        else:
            factors["trend_alignment"] = 8
            recommendations.append("Consider using a trending sound to boost discoverability")

        # Posting time optimization (15%)
        optimal_hours = [7, 8, 12, 13, 18, 19, 20, 21]
        if request.posting_hour in optimal_hours:
            factors["posting_time"] = 13
        elif request.posting_hour in [h-1 for h in optimal_hours] + [h+1 for h in optimal_hours]:
            factors["posting_time"] = 9
        else:
            factors["posting_time"] = 5
            recommendations.append(f"Consider posting between 7-9 PM for maximum reach")

        # Hashtag mix (15%)
        hashtag_count = len(request.hashtags)
        if 3 <= hashtag_count <= 7:
            factors["hashtag_mix"] = 13
        elif hashtag_count > 0:
            factors["hashtag_mix"] = 8
            if hashtag_count < 3:
                recommendations.append("Add 3-7 relevant hashtags for better discoverability")
            else:
                recommendations.append("Reduce hashtags to 3-7 for optimal performance")
        else:
            factors["hashtag_mix"] = 3
            recommendations.append("Add hashtags to increase content reach")

        # Caption quality (10%)
        caption_len = len(request.caption)
        has_question = "?" in request.caption
        has_cta = any(word in request.caption.lower() for word in ["follow", "like", "comment", "share", "link"])

        caption_score = 3
        if 50 <= caption_len <= 150:
            caption_score += 3
        if has_question:
            caption_score += 2
        if has_cta:
            caption_score += 2
        factors["caption_quality"] = min(caption_score, 10)

        if not has_question and not has_cta:
            recommendations.append("Add a question or call-to-action in your caption")

        # Audio choice (10%)
        factors["audio_choice"] = 8 if request.has_trending_sound else 4

        # Video length (5%)
        if 15 <= request.video_duration_seconds <= 60:
            factors["video_length"] = 5
        elif request.video_duration_seconds < 15:
            factors["video_length"] = 3
            recommendations.append("Videos between 15-60 seconds tend to perform best")
        else:
            factors["video_length"] = 2
            recommendations.append("Consider shorter videos (15-60s) for better completion rates")

        # Calculate final score
        viral_score = sum(factors.values())

        # Predict view ranges based on score
        base_views = 500
        if viral_score >= 80:
            multiplier = 100
        elif viral_score >= 60:
            multiplier = 20
        elif viral_score >= 40:
            multiplier = 5
        else:
            multiplier = 1

        predicted_low = int(base_views * multiplier * 0.5)
        predicted_high = int(base_views * multiplier * 2.5)

        # Calculate confidence based on data completeness
        confidence = 0.6  # Base confidence
        if request.niche:
            confidence += 0.1
        if request.has_trending_sound:
            confidence += 0.1
        if len(request.hashtags) > 0:
            confidence += 0.1

        return ViralPredictionResponse(
            viral_score=viral_score,
            predicted_views_low=predicted_low,
            predicted_views_high=predicted_high,
            factors=factors,
            recommendations=recommendations[:5],  # Top 5 recommendations
            confidence=round(confidence, 2)
        )

    async def get_niche_benchmark(self, niche: str, platform: str = "tiktok") -> NicheBenchmark:
        """Get benchmark data for a specific niche"""

        # In production, this would query the niche_benchmarks table
        # For now, return demo data
        niche_data = {
            "entertainment": {
                "avg_engagement_rate": 6.5,
                "avg_views": 25000,
                "top_hashtags": ["#fyp", "#viral", "#funny", "#comedy", "#entertainment"],
                "content_length_avg": 30
            },
            "education": {
                "avg_engagement_rate": 8.2,
                "avg_views": 15000,
                "top_hashtags": ["#learn", "#education", "#tips", "#howto", "#tutorial"],
                "content_length_avg": 45
            },
            "fitness": {
                "avg_engagement_rate": 7.8,
                "avg_views": 20000,
                "top_hashtags": ["#fitness", "#workout", "#gym", "#health", "#motivation"],
                "content_length_avg": 35
            },
            "food": {
                "avg_engagement_rate": 7.2,
                "avg_views": 30000,
                "top_hashtags": ["#food", "#recipe", "#cooking", "#foodie", "#yummy"],
                "content_length_avg": 40
            },
            "tech": {
                "avg_engagement_rate": 5.8,
                "avg_views": 18000,
                "top_hashtags": ["#tech", "#technology", "#tips", "#gadgets", "#apple"],
                "content_length_avg": 50
            }
        }

        data = niche_data.get(niche.lower(), {
            "avg_engagement_rate": 6.0,
            "avg_views": 20000,
            "top_hashtags": ["#fyp", "#viral", "#trending", "#foryou", "#foryoupage"],
            "content_length_avg": 35
        })

        return NicheBenchmark(
            niche=niche,
            platform=platform,
            avg_engagement_rate=data["avg_engagement_rate"],
            avg_views=data["avg_views"],
            top_posting_times=[
                {"day": "Tuesday", "hour": 19, "engagement_boost": 1.23},
                {"day": "Thursday", "hour": 20, "engagement_boost": 1.18},
                {"day": "Saturday", "hour": 12, "engagement_boost": 1.15},
                {"day": "Sunday", "hour": 18, "engagement_boost": 1.12}
            ],
            top_hashtags=data["top_hashtags"],
            content_length_avg=data["content_length_avg"]
        )

    async def analyze_content_gaps(self, user_id: UUID, niche: str = None) -> ContentGapAnalysis:
        """Analyze content gaps and opportunities"""

        async with async_session() as session:
            # Get user's recent analytics
            result = await session.execute(
                select(PostAnalytics)
                .where(PostAnalytics.user_id == user_id)
                .order_by(desc(PostAnalytics.captured_at))
                .limit(50)
            )
            analytics = result.scalars().all()

            # Analyze content patterns
            missing_types = []
            underperforming = []
            opportunities = []
            topics = []

            if not analytics:
                missing_types = ["Short-form videos", "Tutorials", "Behind-the-scenes", "Trending challenges"]
                opportunities = ["Start posting consistently to gather data"]
            else:
                avg_views = sum(a.views for a in analytics) / len(analytics)
                avg_engagement = sum(a.engagement_rate for a in analytics) / len(analytics)

                # Check for underperforming content
                poor_performers = [a for a in analytics if a.views < avg_views * 0.5]
                if len(poor_performers) > len(analytics) * 0.3:
                    underperforming.append({
                        "area": "Overall reach",
                        "current": int(avg_views),
                        "benchmark": int(avg_views * 1.5),
                        "suggestion": "Focus on hooks and trending sounds"
                    })

                if avg_engagement < 5:
                    underperforming.append({
                        "area": "Engagement rate",
                        "current": round(avg_engagement, 2),
                        "benchmark": 5.0,
                        "suggestion": "Add more CTAs and questions"
                    })

                # Suggest content types
                missing_types = ["Duets/Collabs", "Q&A content", "Tutorial series"]
                opportunities = [
                    "Create a content series for returning viewers",
                    "Engage with trending topics in your niche",
                    "Collaborate with creators of similar size"
                ]
                topics = [
                    "Day in the life content",
                    "Behind the scenes",
                    "Quick tips and hacks",
                    "React to trends",
                    "Answer follower questions"
                ]

            return ContentGapAnalysis(
                missing_content_types=missing_types,
                underperforming_areas=underperforming,
                competitor_insights=[
                    {"metric": "Posting frequency", "you": "5/week", "top_creators": "7/week"},
                    {"metric": "Avg video length", "you": "25s", "top_creators": "35s"},
                    {"metric": "Hashtag usage", "you": "3", "top_creators": "5"}
                ],
                opportunities=opportunities,
                recommended_topics=topics
            )


analytics_engine = AnalyticsEngine()

# ========================================
# API Endpoints
# ========================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "analytics-service"}

@app.post("/api/v1/analytics/record")
async def record_analytics(
    user_id: UUID,
    data: PostAnalyticsCreate
):
    """Record analytics data for a post"""
    
    async with async_session() as session:
        engagement_rate = await analytics_engine.calculate_engagement_rate(
            data.views, data.likes, data.comments, data.shares, data.saves
        )
        
        analytics = PostAnalytics(
            post_id=data.post_id,
            user_id=user_id,
            tiktok_post_id=data.tiktok_post_id,
            views=data.views,
            likes=data.likes,
            comments=data.comments,
            shares=data.shares,
            saves=data.saves,
            engagement_rate=engagement_rate,
            avg_watch_time_seconds=data.avg_watch_time_seconds,
            completion_rate=data.completion_rate,
            audience_demographics=data.audience_demographics or {},
            traffic_sources=data.traffic_sources or {}
        )
        
        session.add(analytics)
        await session.commit()
        await session.refresh(analytics)
        
        # Index in Elasticsearch for advanced queries
        if es_client:
            try:
                await es_client.index(
                    index="post_analytics",
                    id=str(analytics.id),
                    body={
                        "post_id": str(analytics.post_id),
                        "user_id": str(analytics.user_id),
                        "views": analytics.views,
                        "likes": analytics.likes,
                        "engagement_rate": analytics.engagement_rate,
                        "timestamp": analytics.captured_at.isoformat()
                    }
                )
            except Exception as e:
                print(f"ES indexing error: {e}")
        
        return {"id": str(analytics.id), "status": "recorded"}

@app.get("/api/v1/analytics/post/{post_id}", response_model=List[PostAnalyticsResponse])
async def get_post_analytics(
    post_id: UUID,
    user_id: UUID = Query(...)
):
    """Get analytics history for a specific post"""
    
    async with async_session() as session:
        result = await session.execute(
            select(PostAnalytics)
            .where(
                PostAnalytics.post_id == post_id,
                PostAnalytics.user_id == user_id
            )
            .order_by(desc(PostAnalytics.captured_at))
        )
        analytics = result.scalars().all()
        
        return [
            PostAnalyticsResponse(
                id=a.id,
                post_id=a.post_id,
                tiktok_post_id=a.tiktok_post_id,
                views=a.views,
                likes=a.likes,
                comments=a.comments,
                shares=a.shares,
                saves=a.saves,
                engagement_rate=a.engagement_rate,
                avg_watch_time_seconds=a.avg_watch_time_seconds,
                completion_rate=a.completion_rate,
                captured_at=a.captured_at
            )
            for a in analytics
        ]

@app.get("/api/v1/analytics/dashboard", response_model=DashboardStats)
async def get_dashboard(
    user_id: UUID = Query(...),
    days: int = Query(default=30, ge=1, le=365)
):
    """Get dashboard statistics"""
    return await analytics_engine.get_dashboard_stats(user_id, days)

@app.get("/api/v1/analytics/report", response_model=PerformanceReport)
async def get_performance_report(
    user_id: UUID = Query(...),
    period: str = Query(default="weekly", regex="^(daily|weekly|monthly)$")
):
    """Generate performance report"""
    return await analytics_engine.generate_performance_report(user_id, period)

@app.get("/api/v1/analytics/insights", response_model=List[ContentInsight])
async def get_insights(user_id: UUID = Query(...)):
    """Get AI-powered content insights"""
    return await analytics_engine.get_content_insights(user_id)

@app.get("/api/v1/analytics/audience", response_model=AudienceAnalytics)
async def get_audience_analytics(user_id: UUID = Query(...)):
    """Get audience demographics and behavior analytics"""
    return await analytics_engine.get_audience_analytics(user_id)

@app.get("/api/v1/analytics/compare")
async def compare_periods(
    user_id: UUID = Query(...),
    period1_start: datetime = Query(...),
    period1_end: datetime = Query(...),
    period2_start: datetime = Query(...),
    period2_end: datetime = Query(...)
):
    """Compare performance between two periods"""
    
    async with async_session() as session:
        async def get_period_stats(start: datetime, end: datetime):
            result = await session.execute(
                select(
                    func.sum(PostAnalytics.views).label('views'),
                    func.sum(PostAnalytics.likes).label('likes'),
                    func.sum(PostAnalytics.comments).label('comments'),
                    func.sum(PostAnalytics.shares).label('shares'),
                    func.avg(PostAnalytics.engagement_rate).label('engagement'),
                    func.count(func.distinct(PostAnalytics.post_id)).label('posts')
                ).where(
                    PostAnalytics.user_id == user_id,
                    PostAnalytics.captured_at >= start,
                    PostAnalytics.captured_at <= end
                )
            )
            row = result.first()
            return {
                "views": row.views or 0,
                "likes": row.likes or 0,
                "comments": row.comments or 0,
                "shares": row.shares or 0,
                "engagement_rate": round(row.engagement or 0, 2),
                "posts": row.posts or 0
            }
        
        period1 = await get_period_stats(period1_start, period1_end)
        period2 = await get_period_stats(period2_start, period2_end)
        
        def calc_change(old, new):
            if old == 0:
                return 100 if new > 0 else 0
            return round((new - old) / old * 100, 2)
        
        changes = {
            "views_change": calc_change(period1["views"], period2["views"]),
            "likes_change": calc_change(period1["likes"], period2["likes"]),
            "engagement_change": calc_change(period1["engagement_rate"], period2["engagement_rate"]),
            "posts_change": calc_change(period1["posts"], period2["posts"])
        }
        
        return {
            "period1": period1,
            "period2": period2,
            "changes": changes
        }

# ========================================
# NEW: Viral Prediction & Benchmark Endpoints
# ========================================

@app.post("/api/v1/analytics/predict-viral", response_model=ViralPredictionResponse)
async def predict_viral(request: ViralPredictionRequest):
    """Predict viral potential of content before posting"""
    return await analytics_engine.predict_viral_score(request)

@app.get("/api/v1/analytics/benchmark/{niche}", response_model=NicheBenchmark)
async def get_benchmark(
    niche: str,
    platform: str = Query(default="tiktok", regex="^(tiktok|instagram|youtube)$")
):
    """Get benchmark data for a specific niche"""
    return await analytics_engine.get_niche_benchmark(niche, platform)

@app.get("/api/v1/analytics/content-gaps", response_model=ContentGapAnalysis)
async def get_content_gaps(
    user_id: UUID = Query(...),
    niche: Optional[str] = Query(None)
):
    """Analyze content gaps and opportunities"""
    return await analytics_engine.analyze_content_gaps(user_id, niche)

@app.get("/api/v1/analytics/optimal-times")
async def get_optimal_posting_times(
    user_id: UUID = Query(...),
    platform: str = Query(default="tiktok")
):
    """Get optimal posting times based on audience activity"""

    # In production, this would analyze user's actual audience data
    return {
        "platform": platform,
        "optimal_times": [
            {"day": "Monday", "times": ["12:00", "19:00", "21:00"], "engagement_boost": 1.15},
            {"day": "Tuesday", "times": ["07:00", "19:00", "20:00"], "engagement_boost": 1.23},
            {"day": "Wednesday", "times": ["12:00", "19:00"], "engagement_boost": 1.10},
            {"day": "Thursday", "times": ["19:00", "20:00", "21:00"], "engagement_boost": 1.18},
            {"day": "Friday", "times": ["12:00", "17:00", "19:00"], "engagement_boost": 1.12},
            {"day": "Saturday", "times": ["11:00", "12:00", "19:00"], "engagement_boost": 1.15},
            {"day": "Sunday", "times": ["10:00", "18:00", "20:00"], "engagement_boost": 1.12}
        ],
        "best_overall": {
            "day": "Tuesday",
            "time": "19:00",
            "engagement_boost": 1.23
        },
        "timezone": "UTC"
    }

@app.get("/api/v1/analytics/competitor-analysis")
async def get_competitor_analysis(
    user_id: UUID = Query(...),
    niche: str = Query(...)
):
    """Get competitor analysis within the same niche"""

    # Demo data - in production would analyze real competitors
    return {
        "niche": niche,
        "your_stats": {
            "avg_views": 15000,
            "avg_engagement": 5.2,
            "posting_frequency": "5/week",
            "follower_count": 5234
        },
        "niche_average": {
            "avg_views": 20000,
            "avg_engagement": 6.0,
            "posting_frequency": "6/week",
            "follower_count": 12000
        },
        "top_10_percent": {
            "avg_views": 100000,
            "avg_engagement": 8.5,
            "posting_frequency": "7/week",
            "follower_count": 150000
        },
        "your_percentile": 45,
        "growth_potential": "high",
        "recommendations": [
            "Increase posting frequency to match top performers",
            "Focus on trending sounds to boost discoverability",
            "Improve hook quality in first 3 seconds"
        ]
    }

@app.get("/api/v1/analytics/growth-forecast")
async def get_growth_forecast(
    user_id: UUID = Query(...),
    days: int = Query(default=30, ge=7, le=90)
):
    """Forecast growth based on current trends"""

    # Demo forecast - in production would use ML model
    import random

    current_followers = 5234
    growth_rate = 0.02  # 2% daily growth

    forecast = []
    for i in range(days):
        projected = int(current_followers * (1 + growth_rate) ** i)
        # Add some variance
        variance = random.uniform(-0.05, 0.05)
        projected = int(projected * (1 + variance))
        forecast.append({
            "day": i + 1,
            "projected_followers": projected,
            "projected_views": int(projected * 4),  # Rough estimate
            "confidence": max(0.5, 0.95 - (i * 0.01))  # Decreasing confidence
        })

    return {
        "current_followers": current_followers,
        "forecast_days": days,
        "projected_end_followers": forecast[-1]["projected_followers"],
        "growth_rate_daily": growth_rate * 100,
        "forecast": forecast,
        "assumptions": [
            "Consistent posting schedule maintained",
            "Current engagement rate continues",
            "No major algorithm changes"
        ]
    }

# ========================================
# Run Application
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
