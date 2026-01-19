/**
 * Course Analytics Types
 * TypeScript types for the analytics dashboard
 */

// Enums
export type TimeRange = 'today' | 'week' | 'month' | 'quarter' | 'year' | 'all_time';

export type MetricType =
  | 'course_created'
  | 'course_completed'
  | 'course_failed'
  | 'lecture_generated'
  | 'video_rendered'
  | 'document_uploaded'
  | 'voice_cloned'
  | 'api_call'
  | 'view'
  | 'completion';

export type APIProvider = 'openai' | 'elevenlabs' | 'd-id' | 'replicate' | 'pexels' | 'cloudinary';

// Metric Models
export interface CourseMetrics {
  total_courses: number;
  courses_completed: number;
  courses_failed: number;
  total_lectures: number;
  total_duration_hours: number;
  avg_lectures_per_course: number;
  avg_duration_per_course_minutes: number;
  categories: Record<string, number>;
  completion_rate: number;
}

export interface APIUsageMetrics {
  provider: APIProvider;
  total_calls: number;
  successful_calls: number;
  failed_calls: number;
  total_tokens: number;
  total_cost_usd: number;
  avg_latency_ms: number;
  success_rate: number;
}

export interface EngagementMetrics {
  total_views: number;
  unique_viewers: number;
  total_watch_time_hours: number;
  avg_watch_time_minutes: number;
  completion_rate: number;
  top_courses: { course_id: string; views: number }[];
  views_by_country: Record<string, number>;
  views_by_device: Record<string, number>;
}

export interface StorageMetrics {
  total_storage_gb: number;
  videos_storage_gb: number;
  documents_storage_gb: number;
  presentations_storage_gb: number;
  voice_samples_storage_gb: number;
  file_count: number;
}

// Dashboard Models
export interface DashboardSummary {
  time_range: TimeRange;
  generated_at: string;
  courses: CourseMetrics;
  api_usage: APIUsageMetrics[];
  total_api_cost_usd: number;
  engagement: EngagementMetrics;
  storage: StorageMetrics;
  daily_courses: { date: string; count: number }[];
  daily_api_costs: { date: string; cost: number }[];
  daily_views: { date: string; views: number }[];
}

export interface UserAnalyticsSummary {
  user_id: string;
  time_range: TimeRange;
  courses_created: number;
  lectures_generated: number;
  documents_uploaded: number;
  voice_profiles: number;
  total_api_cost_usd: number;
  openai_tokens: number;
  elevenlabs_characters: number;
  storage_used_gb: number;
  total_views: number;
  avg_completion_rate: number;
}

export interface APIUsageReport {
  time_range: TimeRange;
  generated_at: string;
  total_cost_usd: number;
  total_calls: number;
  total_tokens: number;
  by_provider: APIUsageMetrics[];
  by_period: { period: string; cost: number; calls: number; tokens: number }[];
  projected_monthly_cost_usd: number;
  cost_trend_percent: number;
}

export interface UsageQuota {
  user_id: string;
  plan: string;
  max_courses_per_month: number;
  max_storage_gb: number;
  max_api_cost_per_month_usd: number;
  courses_this_month: number;
  storage_used_gb: number;
  api_cost_this_month_usd: number;
  courses_remaining: number;
  storage_remaining_gb: number;
  api_budget_remaining_usd: number;
  quota_exceeded: boolean;
  warnings: string[];
}

// Request Models
export interface TrackEventRequest {
  event_type: MetricType;
  user_id: string;
  course_id?: string;
  metadata?: Record<string, any>;
}

// Helper Functions
export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  }).format(amount);
}

export function formatNumber(num: number): string {
  return new Intl.NumberFormat('en-US').format(num);
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatDuration(hours: number): string {
  if (hours < 1) {
    return `${Math.round(hours * 60)} min`;
  }
  return `${hours.toFixed(1)} hrs`;
}

export function formatStorage(gb: number): string {
  if (gb < 1) {
    return `${Math.round(gb * 1024)} MB`;
  }
  return `${gb.toFixed(2)} GB`;
}

export function getTimeRangeLabel(range: TimeRange): string {
  const labels: Record<TimeRange, string> = {
    today: 'Today',
    week: 'Last 7 days',
    month: 'Last 30 days',
    quarter: 'Last 90 days',
    year: 'Last year',
    all_time: 'All time',
  };
  return labels[range];
}

export function getProviderLabel(provider: APIProvider): string {
  const labels: Record<APIProvider, string> = {
    openai: 'OpenAI',
    elevenlabs: 'ElevenLabs',
    'd-id': 'D-ID',
    replicate: 'Replicate',
    pexels: 'Pexels',
    cloudinary: 'Cloudinary',
  };
  return labels[provider];
}

export function getProviderColor(provider: APIProvider): string {
  const colors: Record<APIProvider, string> = {
    openai: '#10a37f',
    elevenlabs: '#000000',
    'd-id': '#6366f1',
    replicate: '#f59e0b',
    pexels: '#05a081',
    cloudinary: '#3448c5',
  };
  return colors[provider];
}
