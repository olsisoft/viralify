/**
 * Billing Types
 * TypeScript types for subscription and payment management
 */

export type PaymentProvider = 'stripe' | 'paypal';
export type SubscriptionPlan = 'free' | 'starter' | 'pro' | 'enterprise';
export type SubscriptionStatus = 'active' | 'trialing' | 'past_due' | 'canceled' | 'unpaid';
export type BillingInterval = 'monthly' | 'yearly';

export interface PlanFeatures {
  courses_per_month: number;
  max_lectures_per_course: number;
  storage_gb: number;
  api_budget_usd: number;
  voice_cloning: boolean;
  multi_language: number;
  priority_support: boolean;
  custom_branding: boolean;
  analytics: string;
  export_formats: string[];
  team_members?: number;
  sso?: boolean;
  api_access?: boolean;
  dedicated_support?: boolean;
}

export interface PlanInfo {
  id: SubscriptionPlan;
  name: string;
  description: string;
  price_monthly_usd: number;
  price_yearly_usd: number;
  features: PlanFeatures;
  popular?: boolean;
}

export interface Subscription {
  id: string;
  user_id: string;
  plan: SubscriptionPlan;
  status: SubscriptionStatus;
  provider: PaymentProvider;
  billing_interval: BillingInterval;
  current_period_start: string;
  current_period_end: string;
  trial_end?: string;
  canceled_at?: string;
}

export interface SubscriptionResponse {
  subscription: Subscription;
  plan_info: PlanInfo;
  next_invoice_date?: string;
  next_invoice_amount_usd?: number;
}

export interface CheckoutSessionResponse {
  session_id: string;
  checkout_url: string;
  provider: PaymentProvider;
}

// Helper functions
export function formatPrice(amount: number, interval?: BillingInterval): string {
  if (amount === 0) return 'Free';
  const formatted = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
  }).format(amount);
  return interval ? `${formatted}/${interval === 'monthly' ? 'mo' : 'yr'}` : formatted;
}

export function getPlanColor(plan: SubscriptionPlan): string {
  const colors: Record<SubscriptionPlan, string> = {
    free: '#6b7280',
    starter: '#3b82f6',
    pro: '#8b5cf6',
    enterprise: '#f59e0b',
  };
  return colors[plan];
}

export function getStatusLabel(status: SubscriptionStatus): string {
  const labels: Record<SubscriptionStatus, string> = {
    active: 'Active',
    trialing: 'Trial',
    past_due: 'Past Due',
    canceled: 'Canceled',
    unpaid: 'Unpaid',
  };
  return labels[status];
}

export function getStatusColor(status: SubscriptionStatus): string {
  const colors: Record<SubscriptionStatus, string> = {
    active: 'text-green-400',
    trialing: 'text-blue-400',
    past_due: 'text-orange-400',
    canceled: 'text-gray-400',
    unpaid: 'text-red-400',
  };
  return colors[status];
}

export function formatFeatureValue(value: number | boolean | string | string[]): string {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return value === -1 ? 'Unlimited' : value.toString();
  if (Array.isArray(value)) return value.join(', ').toUpperCase();
  return value;
}
