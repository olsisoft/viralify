'use client';

import { useState, useCallback, useEffect } from 'react';
import {
  PlanInfo,
  SubscriptionResponse,
  CheckoutSessionResponse,
  SubscriptionPlan,
  BillingInterval,
  PaymentProvider,
} from '../lib/billing-types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

interface UseBillingReturn {
  // State
  plans: PlanInfo[];
  subscription: SubscriptionResponse | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchPlans: () => Promise<void>;
  fetchSubscription: (userId: string) => Promise<void>;
  createCheckout: (
    userId: string,
    plan: SubscriptionPlan,
    interval: BillingInterval,
    provider: PaymentProvider
  ) => Promise<string>;
  cancelSubscription: (userId: string, reason?: string) => Promise<void>;
  openBillingPortal: (userId: string) => Promise<string>;
}

export function useBilling(userId?: string): UseBillingReturn {
  const [plans, setPlans] = useState<PlanInfo[]>([]);
  const [subscription, setSubscription] = useState<SubscriptionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPlans = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/billing/plans`);
      if (!response.ok) throw new Error('Failed to fetch plans');
      const data = await response.json();
      setPlans(data.plans);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  const fetchSubscription = useCallback(async (userId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/billing/subscription/${userId}`);
      if (!response.ok) throw new Error('Failed to fetch subscription');
      const data: SubscriptionResponse = await response.json();
      setSubscription(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const createCheckout = useCallback(async (
    userId: string,
    plan: SubscriptionPlan,
    interval: BillingInterval,
    provider: PaymentProvider
  ): Promise<string> => {
    const response = await fetch(`${API_BASE}/api/v1/billing/checkout`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        plan,
        billing_interval: interval,
        provider,
        success_url: `${window.location.origin}/dashboard/studio/billing?success=true`,
        cancel_url: `${window.location.origin}/dashboard/studio/billing?canceled=true`,
      }),
    });

    if (!response.ok) throw new Error('Failed to create checkout');
    const data: CheckoutSessionResponse = await response.json();
    return data.checkout_url;
  }, []);

  const cancelSubscription = useCallback(async (userId: string, reason?: string) => {
    const response = await fetch(`${API_BASE}/api/v1/billing/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        reason,
        cancel_immediately: false,
      }),
    });

    if (!response.ok) throw new Error('Failed to cancel subscription');
    await fetchSubscription(userId);
  }, [fetchSubscription]);

  const openBillingPortal = useCallback(async (userId: string): Promise<string> => {
    const response = await fetch(`${API_BASE}/api/v1/billing/portal`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        return_url: `${window.location.origin}/dashboard/studio/billing`,
      }),
    });

    if (!response.ok) throw new Error('Failed to open billing portal');
    const data = await response.json();
    return data.portal_url;
  }, []);

  // Initial load
  useEffect(() => {
    fetchPlans();
    if (userId) {
      fetchSubscription(userId);
    }
  }, [fetchPlans, fetchSubscription, userId]);

  return {
    plans,
    subscription,
    isLoading,
    error,
    fetchPlans,
    fetchSubscription,
    createCheckout,
    cancelSubscription,
    openBillingPortal,
  };
}
