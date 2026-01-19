"""
Course Analytics Models

Pydantic models for tracking course generation metrics,
API usage, and content engagement.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class MetricType(str, Enum):
    """Types of metrics tracked"""
    COURSE_CREATED = "course_created"
    COURSE_COMPLETED = "course_completed"
    COURSE_FAILED = "course_failed"
    LECTURE_GENERATED = "lecture_generated"
    VIDEO_RENDERED = "video_rendered"
    DOCUMENT_UPLOADED = "document_uploaded"
    VOICE_CLONED = "voice_cloned"
    API_CALL = "api_call"
    VIEW = "view"
    COMPLETION = "completion"


class APIProvider(str, Enum):
    """External API providers"""
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    DID = "d-id"
    REPLICATE = "replicate"
    PEXELS = "pexels"
    CLOUDINARY = "cloudinary"


class TimeRange(str, Enum):
    """Time ranges for analytics"""
    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    ALL_TIME = "all_time"


# =============================================================================
# Event Tracking Models
# =============================================================================

class AnalyticsEvent(BaseModel):
    """Base analytics event"""
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    event_type: MetricType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict = Field(default_factory=dict)


class CourseEvent(AnalyticsEvent):
    """Course-specific event"""
    course_id: str
    course_title: Optional[str] = None
    category: Optional[str] = None
    lecture_count: Optional[int] = None
    total_duration_seconds: Optional[int] = None


class APIUsageEvent(AnalyticsEvent):
    """API usage event"""
    provider: APIProvider
    endpoint: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


class ViewEvent(AnalyticsEvent):
    """Content view event"""
    course_id: str
    lecture_id: Optional[str] = None
    viewer_id: Optional[str] = None  # Anonymous or authenticated
    watch_duration_seconds: int = 0
    completed: bool = False
    device_type: Optional[str] = None
    country: Optional[str] = None


# =============================================================================
# Aggregated Metrics Models
# =============================================================================

class CourseMetrics(BaseModel):
    """Aggregated course creation metrics"""
    total_courses: int = 0
    courses_completed: int = 0
    courses_failed: int = 0
    total_lectures: int = 0
    total_duration_hours: float = 0.0
    avg_lectures_per_course: float = 0.0
    avg_duration_per_course_minutes: float = 0.0
    categories: Dict[str, int] = Field(default_factory=dict)
    completion_rate: float = 0.0  # courses_completed / total_courses


class APIUsageMetrics(BaseModel):
    """Aggregated API usage metrics"""
    provider: APIProvider
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 0.0


class EngagementMetrics(BaseModel):
    """Content engagement metrics"""
    total_views: int = 0
    unique_viewers: int = 0
    total_watch_time_hours: float = 0.0
    avg_watch_time_minutes: float = 0.0
    completion_rate: float = 0.0
    top_courses: List[Dict] = Field(default_factory=list)
    views_by_country: Dict[str, int] = Field(default_factory=dict)
    views_by_device: Dict[str, int] = Field(default_factory=dict)


class StorageMetrics(BaseModel):
    """Storage usage metrics"""
    total_storage_gb: float = 0.0
    videos_storage_gb: float = 0.0
    documents_storage_gb: float = 0.0
    presentations_storage_gb: float = 0.0
    voice_samples_storage_gb: float = 0.0
    file_count: int = 0


# =============================================================================
# Dashboard Models
# =============================================================================

class DashboardSummary(BaseModel):
    """Main dashboard summary"""
    time_range: TimeRange
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Course stats
    courses: CourseMetrics

    # API usage by provider
    api_usage: List[APIUsageMetrics] = Field(default_factory=list)
    total_api_cost_usd: float = 0.0

    # Engagement
    engagement: EngagementMetrics

    # Storage
    storage: StorageMetrics

    # Trends (daily values for charts)
    daily_courses: List[Dict] = Field(default_factory=list)  # [{date, count}]
    daily_api_costs: List[Dict] = Field(default_factory=list)  # [{date, cost}]
    daily_views: List[Dict] = Field(default_factory=list)  # [{date, views}]


class UserAnalyticsSummary(BaseModel):
    """Per-user analytics summary"""
    user_id: str
    time_range: TimeRange

    # Activity
    courses_created: int = 0
    lectures_generated: int = 0
    documents_uploaded: int = 0
    voice_profiles: int = 0

    # API costs
    total_api_cost_usd: float = 0.0
    openai_tokens: int = 0
    elevenlabs_characters: int = 0

    # Storage
    storage_used_gb: float = 0.0

    # Engagement (for their courses)
    total_views: int = 0
    avg_completion_rate: float = 0.0


# =============================================================================
# Request/Response Models
# =============================================================================

class TrackEventRequest(BaseModel):
    """Request to track an analytics event"""
    event_type: MetricType
    user_id: str
    course_id: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


class GetAnalyticsRequest(BaseModel):
    """Request for analytics data"""
    user_id: Optional[str] = None  # None = admin view
    time_range: TimeRange = TimeRange.MONTH
    include_trends: bool = True


class APIUsageReportRequest(BaseModel):
    """Request for detailed API usage report"""
    user_id: Optional[str] = None
    provider: Optional[APIProvider] = None
    time_range: TimeRange = TimeRange.MONTH
    group_by: str = "day"  # day, week, month


class APIUsageReport(BaseModel):
    """Detailed API usage report"""
    time_range: TimeRange
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_tokens: int = 0

    by_provider: List[APIUsageMetrics] = Field(default_factory=list)
    by_period: List[Dict] = Field(default_factory=list)  # [{period, cost, calls, tokens}]

    # Projections
    projected_monthly_cost_usd: float = 0.0
    cost_trend_percent: float = 0.0  # vs previous period


# =============================================================================
# Quota & Limits Models
# =============================================================================

class UsageQuota(BaseModel):
    """User usage quota and limits"""
    user_id: str
    plan: str = "free"  # free, pro, enterprise

    # Limits
    max_courses_per_month: int = 5
    max_storage_gb: float = 1.0
    max_api_cost_per_month_usd: float = 10.0

    # Current usage
    courses_this_month: int = 0
    storage_used_gb: float = 0.0
    api_cost_this_month_usd: float = 0.0

    # Calculated
    courses_remaining: int = 0
    storage_remaining_gb: float = 0.0
    api_budget_remaining_usd: float = 0.0

    # Flags
    quota_exceeded: bool = False
    warnings: List[str] = Field(default_factory=list)
