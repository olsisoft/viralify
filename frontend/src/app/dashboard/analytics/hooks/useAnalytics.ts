'use client';

import { useState, useCallback, useEffect } from 'react';
import {
  DashboardSummary,
  UserAnalyticsSummary,
  APIUsageReport,
  UsageQuota,
  TimeRange,
  APIProvider,
  TrackEventRequest,
} from '../lib/analytics-types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

interface UseAnalyticsReturn {
  // State
  dashboard: DashboardSummary | null;
  userSummary: UserAnalyticsSummary | null;
  apiUsageReport: APIUsageReport | null;
  quota: UsageQuota | null;
  isLoading: boolean;
  error: string | null;
  timeRange: TimeRange;

  // Actions
  setTimeRange: (range: TimeRange) => void;
  fetchDashboard: (userId?: string) => Promise<void>;
  fetchUserSummary: (userId: string) => Promise<void>;
  fetchAPIUsageReport: (userId?: string, provider?: APIProvider) => Promise<void>;
  fetchQuota: (userId: string) => Promise<void>;
  trackEvent: (event: TrackEventRequest) => Promise<void>;
  refresh: () => Promise<void>;
}

export function useAnalytics(initialUserId?: string): UseAnalyticsReturn {
  const [dashboard, setDashboard] = useState<DashboardSummary | null>(null);
  const [userSummary, setUserSummary] = useState<UserAnalyticsSummary | null>(null);
  const [apiUsageReport, setApiUsageReport] = useState<APIUsageReport | null>(null);
  const [quota, setQuota] = useState<UsageQuota | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<TimeRange>('month');

  const fetchDashboard = useCallback(async (userId?: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({ time_range: timeRange, include_trends: 'true' });
      if (userId) params.append('user_id', userId);

      const response = await fetch(`${API_BASE}/api/v1/analytics/dashboard?${params}`);
      if (!response.ok) throw new Error('Failed to fetch dashboard');

      const data: DashboardSummary = await response.json();
      setDashboard(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, [timeRange]);

  const fetchUserSummary = useCallback(async (userId: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({ time_range: timeRange });
      const response = await fetch(`${API_BASE}/api/v1/analytics/user/${userId}?${params}`);
      if (!response.ok) throw new Error('Failed to fetch user summary');

      const data: UserAnalyticsSummary = await response.json();
      setUserSummary(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, [timeRange]);

  const fetchAPIUsageReport = useCallback(async (userId?: string, provider?: APIProvider) => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({ time_range: timeRange });
      if (userId) params.append('user_id', userId);
      if (provider) params.append('provider', provider);

      const response = await fetch(`${API_BASE}/api/v1/analytics/api-usage?${params}`);
      if (!response.ok) throw new Error('Failed to fetch API usage report');

      const data: APIUsageReport = await response.json();
      setApiUsageReport(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, [timeRange]);

  const fetchQuota = useCallback(async (userId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/analytics/quota/${userId}`);
      if (!response.ok) throw new Error('Failed to fetch quota');

      const data: UsageQuota = await response.json();
      setQuota(data);
    } catch (err) {
      console.error('Quota fetch error:', err);
    }
  }, []);

  const trackEvent = useCallback(async (event: TrackEventRequest) => {
    try {
      await fetch(`${API_BASE}/api/v1/analytics/track`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(event),
      });
    } catch (err) {
      console.error('Track event error:', err);
    }
  }, []);

  const refresh = useCallback(async () => {
    await fetchDashboard(initialUserId);
    if (initialUserId) {
      await fetchQuota(initialUserId);
    }
  }, [fetchDashboard, fetchQuota, initialUserId]);

  // Initial load
  useEffect(() => {
    fetchDashboard(initialUserId);
    if (initialUserId) {
      fetchQuota(initialUserId);
    }
  }, [fetchDashboard, fetchQuota, initialUserId, timeRange]);

  return {
    dashboard,
    userSummary,
    apiUsageReport,
    quota,
    isLoading,
    error,
    timeRange,
    setTimeRange,
    fetchDashboard,
    fetchUserSummary,
    fetchAPIUsageReport,
    fetchQuota,
    trackEvent,
    refresh,
  };
}
