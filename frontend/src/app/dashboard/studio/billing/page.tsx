'use client';

import { useState } from 'react';
import { useBilling } from './hooks/useBilling';
import {
  SubscriptionPlan,
  BillingInterval,
  PaymentProvider,
  formatPrice,
  getPlanColor,
  getStatusLabel,
  getStatusColor,
  formatFeatureValue,
} from './lib/billing-types';

function PlanCard({
  plan,
  interval,
  isCurrentPlan,
  onSelect,
  isLoading,
}: {
  plan: any;
  interval: BillingInterval;
  isCurrentPlan: boolean;
  onSelect: () => void;
  isLoading: boolean;
}) {
  const price = interval === 'monthly' ? plan.price_monthly_usd : plan.price_yearly_usd;
  const monthlyEquivalent = interval === 'yearly' ? plan.price_yearly_usd / 12 : price;
  const savings = interval === 'yearly' && plan.price_monthly_usd > 0
    ? Math.round((1 - plan.price_yearly_usd / (plan.price_monthly_usd * 12)) * 100)
    : 0;

  return (
    <div
      className={`relative bg-gray-800 rounded-2xl p-6 border-2 transition-all ${
        plan.popular ? 'border-purple-500' : 'border-gray-700'
      } ${isCurrentPlan ? 'ring-2 ring-green-500' : ''}`}
    >
      {plan.popular && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 bg-purple-600 rounded-full text-xs font-medium text-white">
          Most Popular
        </div>
      )}

      {isCurrentPlan && (
        <div className="absolute -top-3 right-4 px-3 py-1 bg-green-600 rounded-full text-xs font-medium text-white">
          Current Plan
        </div>
      )}

      <div className="mb-4">
        <h3 className="text-xl font-bold text-white">{plan.name}</h3>
        <p className="text-gray-400 text-sm mt-1">{plan.description}</p>
      </div>

      <div className="mb-6">
        <div className="flex items-baseline">
          <span className="text-4xl font-bold text-white">
            {price === 0 ? 'Free' : `$${monthlyEquivalent.toFixed(0)}`}
          </span>
          {price > 0 && <span className="text-gray-400 ml-2">/month</span>}
        </div>
        {savings > 0 && (
          <div className="mt-1 text-green-400 text-sm">Save {savings}% with yearly</div>
        )}
        {interval === 'yearly' && price > 0 && (
          <div className="text-gray-500 text-sm">Billed ${price}/year</div>
        )}
      </div>

      <ul className="space-y-3 mb-6">
        <li className="flex items-center gap-2 text-gray-300">
          <span className="text-green-400">&#10003;</span>
          {formatFeatureValue(plan.features.courses_per_month)} courses/month
        </li>
        <li className="flex items-center gap-2 text-gray-300">
          <span className="text-green-400">&#10003;</span>
          {formatFeatureValue(plan.features.max_lectures_per_course)} lectures/course
        </li>
        <li className="flex items-center gap-2 text-gray-300">
          <span className="text-green-400">&#10003;</span>
          {plan.features.storage_gb} GB storage
        </li>
        <li className="flex items-center gap-2 text-gray-300">
          <span className="text-green-400">&#10003;</span>
          ${plan.features.api_budget_usd} API budget
        </li>
        {plan.features.voice_cloning && (
          <li className="flex items-center gap-2 text-gray-300">
            <span className="text-green-400">&#10003;</span>
            Voice cloning
          </li>
        )}
        {plan.features.multi_language > 0 && (
          <li className="flex items-center gap-2 text-gray-300">
            <span className="text-green-400">&#10003;</span>
            {formatFeatureValue(plan.features.multi_language)} languages
          </li>
        )}
        {plan.features.priority_support && (
          <li className="flex items-center gap-2 text-gray-300">
            <span className="text-green-400">&#10003;</span>
            Priority support
          </li>
        )}
        {plan.features.team_members && plan.features.team_members > 1 && (
          <li className="flex items-center gap-2 text-gray-300">
            <span className="text-green-400">&#10003;</span>
            {formatFeatureValue(plan.features.team_members)} team members
          </li>
        )}
      </ul>

      <button
        onClick={onSelect}
        disabled={isCurrentPlan || isLoading}
        className={`w-full py-3 rounded-lg font-medium transition-all ${
          isCurrentPlan
            ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
            : plan.popular
            ? 'bg-purple-600 hover:bg-purple-700 text-white'
            : 'bg-gray-700 hover:bg-gray-600 text-white'
        }`}
      >
        {isCurrentPlan ? 'Current Plan' : isLoading ? 'Loading...' : plan.id === 'free' ? 'Get Started' : 'Upgrade'}
      </button>
    </div>
  );
}

export default function BillingPage() {
  const [interval, setInterval] = useState<BillingInterval>('monthly');
  const [selectedProvider, setSelectedProvider] = useState<PaymentProvider>('stripe');
  const [isProcessing, setIsProcessing] = useState(false);
  const [showCancelModal, setShowCancelModal] = useState(false);

  // TODO: Get actual user ID from auth context
  const userId = 'demo-user-123';

  const { plans, subscription, isLoading, error, createCheckout, cancelSubscription, openBillingPortal } = useBilling(userId);

  const handleSelectPlan = async (planId: SubscriptionPlan) => {
    if (planId === 'free') return;

    setIsProcessing(true);
    try {
      const checkoutUrl = await createCheckout(userId, planId, interval, selectedProvider);
      window.location.href = checkoutUrl;
    } catch (err) {
      console.error('Checkout error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleManageBilling = async () => {
    try {
      const portalUrl = await openBillingPortal(userId);
      window.location.href = portalUrl;
    } catch (err) {
      console.error('Portal error:', err);
    }
  };

  const handleCancelSubscription = async () => {
    try {
      await cancelSubscription(userId, 'User requested cancellation');
      setShowCancelModal(false);
    } catch (err) {
      console.error('Cancel error:', err);
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Billing & Subscription</h1>
        <p className="text-gray-400 mt-1">Manage your subscription and billing settings</p>
      </div>

      {/* Current Subscription */}
      {subscription && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-lg font-semibold text-white">
                  {subscription.plan_info.name} Plan
                </h2>
                <span className={`text-sm ${getStatusColor(subscription.subscription.status)}`}>
                  {getStatusLabel(subscription.subscription.status)}
                </span>
              </div>
              <p className="text-gray-400 text-sm mt-1">
                {subscription.subscription.billing_interval === 'monthly' ? 'Billed monthly' : 'Billed yearly'}
                {subscription.next_invoice_date && (
                  <> &middot; Next invoice: {new Date(subscription.next_invoice_date).toLocaleDateString()}</>
                )}
              </p>
            </div>
            <div className="flex gap-3">
              {subscription.subscription.plan !== 'free' && (
                <>
                  <button
                    onClick={handleManageBilling}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white text-sm transition-colors"
                  >
                    Manage Billing
                  </button>
                  {subscription.subscription.status === 'active' && (
                    <button
                      onClick={() => setShowCancelModal(true)}
                      className="px-4 py-2 border border-red-500/50 hover:bg-red-500/10 rounded-lg text-red-400 text-sm transition-colors"
                    >
                      Cancel
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Usage Progress */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Courses this month</span>
                <span className="text-white">0 / {formatFeatureValue(subscription.plan_info.features.courses_per_month)}</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-purple-500 rounded-full" style={{ width: '0%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">Storage used</span>
                <span className="text-white">0 / {subscription.plan_info.features.storage_gb} GB</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full" style={{ width: '0%' }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">API budget</span>
                <span className="text-white">$0 / ${subscription.plan_info.features.api_budget_usd}</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-green-500 rounded-full" style={{ width: '0%' }} />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Billing Interval Toggle */}
      <div className="flex justify-center mb-8">
        <div className="flex bg-gray-800 rounded-lg p-1">
          <button
            onClick={() => setInterval('monthly')}
            className={`px-6 py-2 rounded-md text-sm font-medium transition-all ${
              interval === 'monthly' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            Monthly
          </button>
          <button
            onClick={() => setInterval('yearly')}
            className={`px-6 py-2 rounded-md text-sm font-medium transition-all ${
              interval === 'yearly' ? 'bg-purple-600 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            Yearly
            <span className="ml-2 text-green-400 text-xs">Save up to 17%</span>
          </button>
        </div>
      </div>

      {/* Payment Provider */}
      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => setSelectedProvider('stripe')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all ${
            selectedProvider === 'stripe'
              ? 'border-purple-500 bg-purple-500/10'
              : 'border-gray-700 hover:border-gray-600'
          }`}
        >
          <span className="text-white font-medium">Stripe</span>
          <span className="text-gray-400 text-sm">Cards</span>
        </button>
        <button
          onClick={() => setSelectedProvider('paypal')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-all ${
            selectedProvider === 'paypal'
              ? 'border-purple-500 bg-purple-500/10'
              : 'border-gray-700 hover:border-gray-600'
          }`}
        >
          <span className="text-white font-medium">PayPal</span>
        </button>
      </div>

      {/* Plans Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-gray-800 rounded-2xl p-6 border border-gray-700 animate-pulse">
              <div className="h-6 bg-gray-700 rounded w-24 mb-2" />
              <div className="h-4 bg-gray-700 rounded w-32 mb-6" />
              <div className="h-10 bg-gray-700 rounded w-20 mb-6" />
              <div className="space-y-2">
                {[...Array(5)].map((_, j) => (
                  <div key={j} className="h-4 bg-gray-700 rounded" />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {plans.map((plan) => (
            <PlanCard
              key={plan.id}
              plan={plan}
              interval={interval}
              isCurrentPlan={subscription?.subscription.plan === plan.id}
              onSelect={() => handleSelectPlan(plan.id)}
              isLoading={isProcessing}
            />
          ))}
        </div>
      )}

      {/* Cancel Modal */}
      {showCancelModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-xl font-bold text-white mb-2">Cancel Subscription?</h3>
            <p className="text-gray-400 mb-6">
              Your subscription will remain active until the end of the current billing period.
              You can resubscribe at any time.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowCancelModal(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition-colors"
              >
                Keep Subscription
              </button>
              <button
                onClick={handleCancelSubscription}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-white transition-colors"
              >
                Cancel Subscription
              </button>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-red-500/20 border border-red-500/30 rounded-lg text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
