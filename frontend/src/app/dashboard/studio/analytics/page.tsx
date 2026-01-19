'use client';

import { useState } from 'react';
import { useAnalytics } from './hooks/useAnalytics';
import {
  TimeRange,
  getTimeRangeLabel,
  formatCurrency,
  formatNumber,
  formatPercentage,
  formatDuration,
  formatStorage,
  getProviderLabel,
  getProviderColor,
} from './lib/analytics-types';

function MiniChart({ data, dataKey, color }: { data: { [key: string]: any }[]; dataKey: string; color: string }) {
  if (!data || data.length === 0) return <div className="h-16 bg-gray-700 rounded animate-pulse" />;

  const values = data.map((d) => d[dataKey] || 0);
  const max = Math.max(...values, 1);
  const width = 100 / data.length;

  return (
    <div className="h-16 flex items-end gap-0.5">
      {values.map((v, i) => (
        <div
          key={i}
          className="rounded-t transition-all hover:opacity-80"
          style={{
            width: `${width}%`,
            height: `${(v / max) * 100}%`,
            backgroundColor: color,
            minHeight: '2px',
          }}
          title={`${data[i]?.date || data[i]?.period}: ${v}`}
        />
      ))}
    </div>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
  color = 'purple',
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: string;
  color?: string;
}) {
  const colorClasses: Record<string, string> = {
    purple: 'bg-purple-500/20 text-purple-400',
    blue: 'bg-blue-500/20 text-blue-400',
    green: 'bg-green-500/20 text-green-400',
    orange: 'bg-orange-500/20 text-orange-400',
    pink: 'bg-pink-500/20 text-pink-400',
  };

  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <span className={`text-2xl p-2.5 rounded-xl ${colorClasses[color]}`}>{icon}</span>
      </div>
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-sm text-gray-400 mt-1">{title}</div>
      {subtitle && <div className="text-xs text-gray-500 mt-1">{subtitle}</div>}
    </div>
  );
}

function ProgressBar({ value, max, label, color }: { value: number; max: number; label: string; color: string }) {
  const percentage = max > 0 ? (value / max) * 100 : 0;
  const isWarning = percentage > 80;
  const isDanger = percentage > 95;

  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <span className={`font-medium ${isDanger ? 'text-red-400' : isWarning ? 'text-orange-400' : 'text-white'}`}>
          {value.toFixed(1)} / {max}
        </span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${isDanger ? 'bg-red-500' : isWarning ? 'bg-orange-500' : ''}`}
          style={{ width: `${Math.min(percentage, 100)}%`, backgroundColor: isDanger || isWarning ? undefined : color }}
        />
      </div>
    </div>
  );
}

export default function CourseAnalyticsDashboard() {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const { dashboard, quota, isLoading, error, timeRange, setTimeRange, refresh } = useAnalytics();

  const timeRanges: TimeRange[] = ['today', 'week', 'month', 'quarter', 'year'];

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-500/20 border border-red-500/30 rounded-xl p-4">
          <h3 className="text-red-400 font-medium">Error loading analytics</h3>
          <p className="text-red-300 text-sm mt-1">{error}</p>
          <button onClick={refresh} className="mt-3 text-red-400 underline text-sm">
            Try again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Course Analytics</h1>
          <p className="text-gray-400 mt-1">Track your course creation and API usage</p>
        </div>
        <div className="flex items-center gap-4 mt-4 md:mt-0">
          {/* Time Range Selector */}
          <div className="flex bg-gray-800 rounded-lg p-1">
            {timeRanges.map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1.5 text-sm rounded-md transition-all ${
                  timeRange === range ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                {getTimeRangeLabel(range)}
              </button>
            ))}
          </div>
          <button
            onClick={refresh}
            disabled={isLoading}
            className="p-2 rounded-lg hover:bg-gray-800 transition-colors disabled:opacity-50 text-gray-400"
          >
            <svg
              className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>
        </div>
      </div>

      {isLoading && !dashboard ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="bg-gray-800 rounded-xl p-5 border border-gray-700 animate-pulse">
              <div className="h-10 w-10 bg-gray-700 rounded-lg mb-4" />
              <div className="h-8 bg-gray-700 rounded w-20 mb-2" />
              <div className="h-4 bg-gray-700 rounded w-32" />
            </div>
          ))}
        </div>
      ) : dashboard ? (
        <>
          {/* Quota Warnings */}
          {quota?.warnings && quota.warnings.length > 0 && (
            <div className="mb-6 bg-orange-500/20 border border-orange-500/30 rounded-xl p-4">
              <div className="flex items-center gap-2 text-orange-400">
                <span className="text-xl">&#9888;</span>
                <span className="font-medium">Usage Warnings</span>
              </div>
              <ul className="mt-2 text-sm text-orange-300">
                {quota.warnings.map((warning, i) => (
                  <li key={i}>• {warning}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Main Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <StatCard
              icon="&#128218;"
              title="Total Courses"
              value={formatNumber(dashboard.courses.total_courses)}
              subtitle={`${formatPercentage(dashboard.courses.completion_rate)} completion rate`}
              color="purple"
            />
            <StatCard
              icon="&#127909;"
              title="Total Lectures"
              value={formatNumber(dashboard.courses.total_lectures)}
              subtitle={`${dashboard.courses.avg_lectures_per_course.toFixed(1)} avg per course`}
              color="blue"
            />
            <StatCard
              icon="&#128065;"
              title="Total Views"
              value={formatNumber(dashboard.engagement.total_views)}
              subtitle={`${dashboard.engagement.unique_viewers} unique viewers`}
              color="green"
            />
            <StatCard
              icon="&#128176;"
              title="API Costs"
              value={formatCurrency(dashboard.total_api_cost_usd)}
              subtitle={getTimeRangeLabel(timeRange)}
              color="orange"
            />
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* Courses Over Time */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="font-medium text-white mb-4">Courses Created</h3>
              <MiniChart data={dashboard.daily_courses} dataKey="count" color="#8b5cf6" />
              <div className="mt-4 text-sm text-gray-400">
                {dashboard.daily_courses.length} days of data
              </div>
            </div>

            {/* Views Over Time */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="font-medium text-white mb-4">Course Views</h3>
              <MiniChart data={dashboard.daily_views} dataKey="views" color="#10b981" />
              <div className="mt-4 text-sm text-gray-400">
                {formatDuration(dashboard.engagement.total_watch_time_hours)} total watch time
              </div>
            </div>

            {/* API Costs Over Time */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="font-medium text-white mb-4">API Costs</h3>
              <MiniChart data={dashboard.daily_api_costs} dataKey="cost" color="#f59e0b" />
              <div className="mt-4 text-sm text-gray-400">
                {formatCurrency(dashboard.total_api_cost_usd)} total
              </div>
            </div>
          </div>

          {/* API Usage & Quota */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            {/* API Usage by Provider */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="font-medium text-white mb-4">API Usage by Provider</h3>
              {dashboard.api_usage.length > 0 ? (
                <div className="space-y-3">
                  {dashboard.api_usage.map((usage) => (
                    <div
                      key={usage.provider}
                      className={`p-4 rounded-lg border cursor-pointer transition-all ${
                        selectedProvider === usage.provider
                          ? 'border-purple-500 bg-purple-500/10'
                          : 'border-gray-700 hover:border-gray-600'
                      }`}
                      onClick={() => setSelectedProvider(selectedProvider === usage.provider ? null : usage.provider)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: getProviderColor(usage.provider) }}
                          />
                          <span className="font-medium text-white">{getProviderLabel(usage.provider)}</span>
                        </div>
                        <span className="font-semibold text-white">{formatCurrency(usage.total_cost_usd)}</span>
                      </div>
                      <div className="mt-2 flex gap-4 text-sm text-gray-400">
                        <span>{formatNumber(usage.total_calls)} calls</span>
                        <span>{formatNumber(usage.total_tokens)} tokens</span>
                        <span>{formatPercentage(usage.success_rate)} success</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-gray-500 text-center py-8">No API usage data</div>
              )}
            </div>

            {/* Quota Usage */}
            {quota && (
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-medium text-white">Usage Quota</h3>
                  <span className="text-sm px-2 py-1 bg-gray-700 rounded-full text-gray-300 capitalize">
                    {quota.plan} plan
                  </span>
                </div>
                <div className="space-y-4">
                  <ProgressBar
                    label="Courses"
                    value={quota.courses_this_month}
                    max={quota.max_courses_per_month}
                    color="#8b5cf6"
                  />
                  <ProgressBar
                    label="Storage"
                    value={quota.storage_used_gb}
                    max={quota.max_storage_gb}
                    color="#06b6d4"
                  />
                  <ProgressBar
                    label="API Budget"
                    value={quota.api_cost_this_month_usd}
                    max={quota.max_api_cost_per_month_usd}
                    color="#f59e0b"
                  />
                </div>
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <div className="grid grid-cols-3 gap-4 text-center">
                    <div>
                      <div className="text-lg font-semibold text-white">{quota.courses_remaining}</div>
                      <div className="text-xs text-gray-500">Courses left</div>
                    </div>
                    <div>
                      <div className="text-lg font-semibold text-white">{formatStorage(quota.storage_remaining_gb)}</div>
                      <div className="text-xs text-gray-500">Storage left</div>
                    </div>
                    <div>
                      <div className="text-lg font-semibold text-white">{formatCurrency(quota.api_budget_remaining_usd)}</div>
                      <div className="text-xs text-gray-500">Budget left</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Categories & Top Courses */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Categories */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="font-medium text-white mb-4">Courses by Category</h3>
              {Object.keys(dashboard.courses.categories).length > 0 ? (
                <div className="space-y-3">
                  {Object.entries(dashboard.courses.categories)
                    .sort(([, a], [, b]) => b - a)
                    .map(([category, count]) => (
                      <div key={category} className="flex items-center justify-between">
                        <span className="text-gray-300 capitalize">{category}</span>
                        <span className="font-medium text-white">{count}</span>
                      </div>
                    ))}
                </div>
              ) : (
                <div className="text-gray-500 text-center py-8">No categories yet</div>
              )}
            </div>

            {/* Top Courses */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="font-medium text-white mb-4">Top Courses by Views</h3>
              {dashboard.engagement.top_courses.length > 0 ? (
                <div className="space-y-3">
                  {dashboard.engagement.top_courses.map((course, i) => (
                    <div key={course.course_id} className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span className="text-sm text-gray-500 w-6">#{i + 1}</span>
                        <span className="text-gray-300 truncate max-w-[200px]">{course.course_id}</span>
                      </div>
                      <span className="font-medium text-white">{formatNumber(course.views)} views</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-gray-500 text-center py-8">No views yet</div>
              )}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
