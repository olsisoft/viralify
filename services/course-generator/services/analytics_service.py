"""
Course Analytics Service

Tracks, stores, and aggregates analytics events for course generation,
API usage, and content engagement.
"""
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from models.analytics_models import (
    AnalyticsEvent,
    CourseEvent,
    APIUsageEvent,
    ViewEvent,
    MetricType,
    APIProvider,
    TimeRange,
    CourseMetrics,
    APIUsageMetrics,
    EngagementMetrics,
    StorageMetrics,
    DashboardSummary,
    UserAnalyticsSummary,
    APIUsageReport,
    UsageQuota,
)


class AnalyticsRepository:
    """
    In-memory analytics repository.
    In production, use PostgreSQL or ClickHouse for time-series data.
    """

    def __init__(self):
        self.events: List[AnalyticsEvent] = []
        self.course_events: List[CourseEvent] = []
        self.api_events: List[APIUsageEvent] = []
        self.view_events: List[ViewEvent] = []

    async def save_event(self, event: AnalyticsEvent) -> None:
        """Save a generic event"""
        self.events.append(event)

    async def save_course_event(self, event: CourseEvent) -> None:
        """Save a course event"""
        self.course_events.append(event)

    async def save_api_event(self, event: APIUsageEvent) -> None:
        """Save an API usage event"""
        self.api_events.append(event)

    async def save_view_event(self, event: ViewEvent) -> None:
        """Save a view event"""
        self.view_events.append(event)

    def _get_time_filter(self, time_range: TimeRange) -> datetime:
        """Get start datetime for time range"""
        now = datetime.utcnow()
        if time_range == TimeRange.TODAY:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif time_range == TimeRange.WEEK:
            return now - timedelta(days=7)
        elif time_range == TimeRange.MONTH:
            return now - timedelta(days=30)
        elif time_range == TimeRange.QUARTER:
            return now - timedelta(days=90)
        elif time_range == TimeRange.YEAR:
            return now - timedelta(days=365)
        else:  # ALL_TIME
            return datetime.min

    async def get_course_events(
        self,
        user_id: Optional[str] = None,
        time_range: TimeRange = TimeRange.MONTH,
    ) -> List[CourseEvent]:
        """Get course events with filters"""
        start_time = self._get_time_filter(time_range)
        filtered = [e for e in self.course_events if e.timestamp >= start_time]
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        return filtered

    async def get_api_events(
        self,
        user_id: Optional[str] = None,
        provider: Optional[APIProvider] = None,
        time_range: TimeRange = TimeRange.MONTH,
    ) -> List[APIUsageEvent]:
        """Get API events with filters"""
        start_time = self._get_time_filter(time_range)
        filtered = [e for e in self.api_events if e.timestamp >= start_time]
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        if provider:
            filtered = [e for e in filtered if e.provider == provider]
        return filtered

    async def get_view_events(
        self,
        user_id: Optional[str] = None,
        course_id: Optional[str] = None,
        time_range: TimeRange = TimeRange.MONTH,
    ) -> List[ViewEvent]:
        """Get view events with filters"""
        start_time = self._get_time_filter(time_range)
        filtered = [e for e in self.view_events if e.timestamp >= start_time]
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        if course_id:
            filtered = [e for e in filtered if e.course_id == course_id]
        return filtered


class CourseAnalyticsService:
    """
    Main analytics service for course generation tracking.
    """

    def __init__(self):
        self.repository = AnalyticsRepository()
        print("[ANALYTICS] Service initialized", flush=True)

    # =========================================================================
    # Event Tracking
    # =========================================================================

    async def track_course_created(
        self,
        user_id: str,
        course_id: str,
        title: str,
        category: Optional[str] = None,
        lecture_count: int = 0,
    ) -> None:
        """Track course creation event"""
        event = CourseEvent(
            user_id=user_id,
            event_type=MetricType.COURSE_CREATED,
            course_id=course_id,
            course_title=title,
            category=category,
            lecture_count=lecture_count,
        )
        await self.repository.save_course_event(event)
        print(f"[ANALYTICS] Course created: {course_id}", flush=True)

    async def track_course_completed(
        self,
        user_id: str,
        course_id: str,
        total_duration_seconds: int,
    ) -> None:
        """Track course completion event"""
        event = CourseEvent(
            user_id=user_id,
            event_type=MetricType.COURSE_COMPLETED,
            course_id=course_id,
            total_duration_seconds=total_duration_seconds,
        )
        await self.repository.save_course_event(event)
        print(f"[ANALYTICS] Course completed: {course_id}", flush=True)

    async def track_course_failed(
        self,
        user_id: str,
        course_id: str,
        error_message: str,
    ) -> None:
        """Track course failure event"""
        event = CourseEvent(
            user_id=user_id,
            event_type=MetricType.COURSE_FAILED,
            course_id=course_id,
            metadata={"error": error_message},
        )
        await self.repository.save_course_event(event)
        print(f"[ANALYTICS] Course failed: {course_id}", flush=True)

    async def track_api_usage(
        self,
        user_id: str,
        provider: APIProvider,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        success: bool = True,
        endpoint: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Track API usage event"""
        event = APIUsageEvent(
            user_id=user_id,
            event_type=MetricType.API_CALL,
            provider=provider,
            endpoint=endpoint,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
        )
        await self.repository.save_api_event(event)

    async def track_view(
        self,
        course_id: str,
        viewer_id: Optional[str] = None,
        lecture_id: Optional[str] = None,
        watch_duration_seconds: int = 0,
        completed: bool = False,
        device_type: Optional[str] = None,
        country: Optional[str] = None,
        owner_user_id: str = "",
    ) -> None:
        """Track content view event"""
        event = ViewEvent(
            user_id=owner_user_id,  # Course owner for aggregation
            event_type=MetricType.VIEW,
            course_id=course_id,
            lecture_id=lecture_id,
            viewer_id=viewer_id,
            watch_duration_seconds=watch_duration_seconds,
            completed=completed,
            device_type=device_type,
            country=country,
        )
        await self.repository.save_view_event(event)

    # =========================================================================
    # Aggregation Methods
    # =========================================================================

    async def get_course_metrics(
        self,
        user_id: Optional[str] = None,
        time_range: TimeRange = TimeRange.MONTH,
    ) -> CourseMetrics:
        """Get aggregated course metrics"""
        events = await self.repository.get_course_events(user_id, time_range)

        created = [e for e in events if e.event_type == MetricType.COURSE_CREATED]
        completed = [e for e in events if e.event_type == MetricType.COURSE_COMPLETED]
        failed = [e for e in events if e.event_type == MetricType.COURSE_FAILED]

        total_lectures = sum(e.lecture_count or 0 for e in created)
        total_duration_seconds = sum(e.total_duration_seconds or 0 for e in completed)

        # Count by category
        categories = defaultdict(int)
        for e in created:
            if e.category:
                categories[e.category] += 1

        metrics = CourseMetrics(
            total_courses=len(created),
            courses_completed=len(completed),
            courses_failed=len(failed),
            total_lectures=total_lectures,
            total_duration_hours=total_duration_seconds / 3600,
            avg_lectures_per_course=total_lectures / len(created) if created else 0,
            avg_duration_per_course_minutes=(total_duration_seconds / 60) / len(completed) if completed else 0,
            categories=dict(categories),
            completion_rate=len(completed) / len(created) if created else 0,
        )

        return metrics

    async def get_api_usage_metrics(
        self,
        user_id: Optional[str] = None,
        time_range: TimeRange = TimeRange.MONTH,
    ) -> List[APIUsageMetrics]:
        """Get API usage metrics by provider"""
        events = await self.repository.get_api_events(user_id, None, time_range)

        # Group by provider
        by_provider: Dict[APIProvider, List[APIUsageEvent]] = defaultdict(list)
        for e in events:
            by_provider[e.provider].append(e)

        metrics = []
        for provider, provider_events in by_provider.items():
            successful = [e for e in provider_events if e.success]
            failed = [e for e in provider_events if not e.success]

            total_tokens = sum(e.tokens_used for e in provider_events)
            total_cost = sum(e.cost_usd for e in provider_events)
            latencies = [e.duration_ms for e in provider_events if e.duration_ms]

            metrics.append(APIUsageMetrics(
                provider=provider,
                total_calls=len(provider_events),
                successful_calls=len(successful),
                failed_calls=len(failed),
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0,
                success_rate=len(successful) / len(provider_events) if provider_events else 0,
            ))

        return metrics

    async def get_engagement_metrics(
        self,
        user_id: Optional[str] = None,
        time_range: TimeRange = TimeRange.MONTH,
    ) -> EngagementMetrics:
        """Get content engagement metrics"""
        events = await self.repository.get_view_events(user_id, None, time_range)

        if not events:
            return EngagementMetrics()

        total_watch_time = sum(e.watch_duration_seconds for e in events)
        completed_views = [e for e in events if e.completed]
        unique_viewers = len(set(e.viewer_id for e in events if e.viewer_id))

        # Views by country
        by_country = defaultdict(int)
        for e in events:
            if e.country:
                by_country[e.country] += 1

        # Views by device
        by_device = defaultdict(int)
        for e in events:
            if e.device_type:
                by_device[e.device_type] += 1

        # Top courses
        by_course = defaultdict(int)
        for e in events:
            by_course[e.course_id] += 1
        top_courses = [
            {"course_id": k, "views": v}
            for k, v in sorted(by_course.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        return EngagementMetrics(
            total_views=len(events),
            unique_viewers=unique_viewers,
            total_watch_time_hours=total_watch_time / 3600,
            avg_watch_time_minutes=(total_watch_time / 60) / len(events) if events else 0,
            completion_rate=len(completed_views) / len(events) if events else 0,
            top_courses=top_courses,
            views_by_country=dict(by_country),
            views_by_device=dict(by_device),
        )

    async def get_storage_metrics(self, user_id: Optional[str] = None) -> StorageMetrics:
        """Get storage usage metrics (placeholder - integrate with actual storage)"""
        # In production, query actual file storage
        return StorageMetrics(
            total_storage_gb=0.0,
            videos_storage_gb=0.0,
            documents_storage_gb=0.0,
            presentations_storage_gb=0.0,
            voice_samples_storage_gb=0.0,
            file_count=0,
        )

    # =========================================================================
    # Dashboard Methods
    # =========================================================================

    async def get_dashboard(
        self,
        user_id: Optional[str] = None,
        time_range: TimeRange = TimeRange.MONTH,
        include_trends: bool = True,
    ) -> DashboardSummary:
        """Get complete dashboard summary"""
        course_metrics = await self.get_course_metrics(user_id, time_range)
        api_metrics = await self.get_api_usage_metrics(user_id, time_range)
        engagement_metrics = await self.get_engagement_metrics(user_id, time_range)
        storage_metrics = await self.get_storage_metrics(user_id)

        total_api_cost = sum(m.total_cost_usd for m in api_metrics)

        summary = DashboardSummary(
            time_range=time_range,
            courses=course_metrics,
            api_usage=api_metrics,
            total_api_cost_usd=total_api_cost,
            engagement=engagement_metrics,
            storage=storage_metrics,
        )

        # Add trend data
        if include_trends:
            summary.daily_courses = await self._get_daily_course_counts(user_id, time_range)
            summary.daily_api_costs = await self._get_daily_api_costs(user_id, time_range)
            summary.daily_views = await self._get_daily_view_counts(user_id, time_range)

        return summary

    async def _get_daily_course_counts(
        self,
        user_id: Optional[str],
        time_range: TimeRange,
    ) -> List[Dict]:
        """Get daily course creation counts for charts"""
        events = await self.repository.get_course_events(user_id, time_range)
        created = [e for e in events if e.event_type == MetricType.COURSE_CREATED]

        # Group by date
        by_date = defaultdict(int)
        for e in created:
            date_str = e.timestamp.strftime("%Y-%m-%d")
            by_date[date_str] += 1

        return [{"date": k, "count": v} for k, v in sorted(by_date.items())]

    async def _get_daily_api_costs(
        self,
        user_id: Optional[str],
        time_range: TimeRange,
    ) -> List[Dict]:
        """Get daily API costs for charts"""
        events = await self.repository.get_api_events(user_id, None, time_range)

        by_date = defaultdict(float)
        for e in events:
            date_str = e.timestamp.strftime("%Y-%m-%d")
            by_date[date_str] += e.cost_usd

        return [{"date": k, "cost": round(v, 4)} for k, v in sorted(by_date.items())]

    async def _get_daily_view_counts(
        self,
        user_id: Optional[str],
        time_range: TimeRange,
    ) -> List[Dict]:
        """Get daily view counts for charts"""
        events = await self.repository.get_view_events(user_id, None, time_range)

        by_date = defaultdict(int)
        for e in events:
            date_str = e.timestamp.strftime("%Y-%m-%d")
            by_date[date_str] += 1

        return [{"date": k, "views": v} for k, v in sorted(by_date.items())]

    # =========================================================================
    # API Usage Report
    # =========================================================================

    async def get_api_usage_report(
        self,
        user_id: Optional[str] = None,
        provider: Optional[APIProvider] = None,
        time_range: TimeRange = TimeRange.MONTH,
    ) -> APIUsageReport:
        """Get detailed API usage report"""
        events = await self.repository.get_api_events(user_id, provider, time_range)

        total_cost = sum(e.cost_usd for e in events)
        total_tokens = sum(e.tokens_used for e in events)

        # By provider
        by_provider = await self.get_api_usage_metrics(user_id, time_range)
        if provider:
            by_provider = [m for m in by_provider if m.provider == provider]

        # By period (daily)
        by_date = defaultdict(lambda: {"cost": 0.0, "calls": 0, "tokens": 0})
        for e in events:
            date_str = e.timestamp.strftime("%Y-%m-%d")
            by_date[date_str]["cost"] += e.cost_usd
            by_date[date_str]["calls"] += 1
            by_date[date_str]["tokens"] += e.tokens_used

        by_period = [
            {"period": k, **v}
            for k, v in sorted(by_date.items())
        ]

        # Calculate projections
        days_in_range = len(by_date) or 1
        avg_daily_cost = total_cost / days_in_range
        projected_monthly = avg_daily_cost * 30

        # Compare to previous period (simplified)
        cost_trend = 0.0  # Would need previous period data

        return APIUsageReport(
            time_range=time_range,
            total_cost_usd=round(total_cost, 4),
            total_calls=len(events),
            total_tokens=total_tokens,
            by_provider=by_provider,
            by_period=by_period,
            projected_monthly_cost_usd=round(projected_monthly, 2),
            cost_trend_percent=cost_trend,
        )

    # =========================================================================
    # User Summary
    # =========================================================================

    async def get_user_summary(
        self,
        user_id: str,
        time_range: TimeRange = TimeRange.MONTH,
    ) -> UserAnalyticsSummary:
        """Get analytics summary for a specific user"""
        course_metrics = await self.get_course_metrics(user_id, time_range)
        api_metrics = await self.get_api_usage_metrics(user_id, time_range)
        engagement = await self.get_engagement_metrics(user_id, time_range)
        storage = await self.get_storage_metrics(user_id)

        # Extract OpenAI and ElevenLabs specifics
        openai_tokens = 0
        elevenlabs_chars = 0
        for m in api_metrics:
            if m.provider == APIProvider.OPENAI:
                openai_tokens = m.total_tokens
            elif m.provider == APIProvider.ELEVENLABS:
                elevenlabs_chars = m.total_tokens  # Characters in ElevenLabs case

        return UserAnalyticsSummary(
            user_id=user_id,
            time_range=time_range,
            courses_created=course_metrics.total_courses,
            lectures_generated=course_metrics.total_lectures,
            documents_uploaded=0,  # Would need to track separately
            voice_profiles=0,  # Would need to track separately
            total_api_cost_usd=sum(m.total_cost_usd for m in api_metrics),
            openai_tokens=openai_tokens,
            elevenlabs_characters=elevenlabs_chars,
            storage_used_gb=storage.total_storage_gb,
            total_views=engagement.total_views,
            avg_completion_rate=engagement.completion_rate,
        )

    # =========================================================================
    # Quota Management
    # =========================================================================

    async def get_user_quota(self, user_id: str) -> UsageQuota:
        """Get user quota and current usage"""
        # Get current month's usage
        user_summary = await self.get_user_summary(user_id, TimeRange.MONTH)

        # Default limits (would come from user's subscription plan)
        quota = UsageQuota(
            user_id=user_id,
            plan="free",
            max_courses_per_month=5,
            max_storage_gb=1.0,
            max_api_cost_per_month_usd=10.0,
            courses_this_month=user_summary.courses_created,
            storage_used_gb=user_summary.storage_used_gb,
            api_cost_this_month_usd=user_summary.total_api_cost_usd,
        )

        # Calculate remaining
        quota.courses_remaining = max(0, quota.max_courses_per_month - quota.courses_this_month)
        quota.storage_remaining_gb = max(0, quota.max_storage_gb - quota.storage_used_gb)
        quota.api_budget_remaining_usd = max(0, quota.max_api_cost_per_month_usd - quota.api_cost_this_month_usd)

        # Check for exceeded quotas
        warnings = []
        if quota.courses_this_month >= quota.max_courses_per_month:
            warnings.append("Course creation limit reached")
        if quota.storage_used_gb >= quota.max_storage_gb * 0.9:
            warnings.append("Storage almost full")
        if quota.api_cost_this_month_usd >= quota.max_api_cost_per_month_usd:
            warnings.append("API budget exceeded")

        quota.warnings = warnings
        quota.quota_exceeded = len([w for w in warnings if "exceeded" in w.lower() or "reached" in w.lower()]) > 0

        return quota


# Global instance
analytics_service: Optional[CourseAnalyticsService] = None


def get_analytics_service() -> CourseAnalyticsService:
    """Get or create analytics service instance"""
    global analytics_service
    if analytics_service is None:
        analytics_service = CourseAnalyticsService()
    return analytics_service
