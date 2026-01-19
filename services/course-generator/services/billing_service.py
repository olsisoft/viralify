"""
Billing Service

Handles subscription management and payments with Stripe and PayPal.
"""
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from models.billing_models import (
    PaymentProvider,
    SubscriptionPlan,
    SubscriptionStatus,
    PaymentStatus,
    BillingInterval,
    PLAN_DETAILS,
    PlanFeatures,
    PlanInfo,
    Subscription,
    Payment,
    Invoice,
    CheckoutSessionResponse,
    PortalSessionResponse,
    SubscriptionResponse,
)


class BillingRepository:
    """In-memory billing repository. Use PostgreSQL in production."""

    def __init__(self):
        self.subscriptions: Dict[str, Subscription] = {}
        self.payments: Dict[str, Payment] = {}
        self.invoices: Dict[str, Invoice] = {}
        self.user_subscriptions: Dict[str, str] = {}  # user_id -> subscription_id

    async def save_subscription(self, subscription: Subscription) -> None:
        self.subscriptions[subscription.id] = subscription
        self.user_subscriptions[subscription.user_id] = subscription.id

    async def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        return self.subscriptions.get(subscription_id)

    async def get_user_subscription(self, user_id: str) -> Optional[Subscription]:
        sub_id = self.user_subscriptions.get(user_id)
        if sub_id:
            return self.subscriptions.get(sub_id)
        return None

    async def save_payment(self, payment: Payment) -> None:
        self.payments[payment.id] = payment

    async def get_user_payments(self, user_id: str) -> List[Payment]:
        return [p for p in self.payments.values() if p.user_id == user_id]


class StripeProvider:
    """Stripe payment provider integration."""

    def __init__(self):
        self.api_key = os.getenv("STRIPE_SECRET_KEY")
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        self.initialized = bool(self.api_key)

        # Stripe price IDs (would be set from Stripe Dashboard)
        self.price_ids = {
            "starter_monthly": os.getenv("STRIPE_PRICE_STARTER_MONTHLY", "price_starter_monthly"),
            "starter_yearly": os.getenv("STRIPE_PRICE_STARTER_YEARLY", "price_starter_yearly"),
            "pro_monthly": os.getenv("STRIPE_PRICE_PRO_MONTHLY", "price_pro_monthly"),
            "pro_yearly": os.getenv("STRIPE_PRICE_PRO_YEARLY", "price_pro_yearly"),
            "enterprise_monthly": os.getenv("STRIPE_PRICE_ENTERPRISE_MONTHLY", "price_enterprise_monthly"),
            "enterprise_yearly": os.getenv("STRIPE_PRICE_ENTERPRISE_YEARLY", "price_enterprise_yearly"),
        }

        if self.initialized:
            print("[BILLING] Stripe provider initialized", flush=True)

    def get_price_id(self, plan: SubscriptionPlan, interval: BillingInterval) -> str:
        """Get Stripe price ID for plan and interval."""
        if plan == SubscriptionPlan.FREE:
            return None
        key = f"{plan.value}_{interval.value}"
        return self.price_ids.get(key)

    async def create_checkout_session(
        self,
        user_id: str,
        customer_email: str,
        plan: SubscriptionPlan,
        interval: BillingInterval,
        success_url: str,
        cancel_url: str,
    ) -> Dict:
        """Create Stripe Checkout session."""
        if not self.initialized:
            # Return mock session for demo
            return {
                "session_id": f"cs_demo_{user_id}_{plan.value}",
                "checkout_url": f"{success_url}?session_id=demo",
            }

        try:
            import stripe
            stripe.api_key = self.api_key

            price_id = self.get_price_id(plan, interval)
            if not price_id:
                raise ValueError(f"No price configured for {plan.value} {interval.value}")

            session = stripe.checkout.Session.create(
                mode="subscription",
                customer_email=customer_email,
                line_items=[{"price": price_id, "quantity": 1}],
                success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
                cancel_url=cancel_url,
                metadata={"user_id": user_id, "plan": plan.value},
                subscription_data={
                    "trial_period_days": 7 if plan != SubscriptionPlan.FREE else 0,
                    "metadata": {"user_id": user_id},
                },
            )

            return {
                "session_id": session.id,
                "checkout_url": session.url,
            }

        except Exception as e:
            print(f"[BILLING] Stripe checkout error: {str(e)}", flush=True)
            raise

    async def create_portal_session(
        self,
        customer_id: str,
        return_url: str,
    ) -> str:
        """Create Stripe Customer Portal session."""
        if not self.initialized:
            return f"{return_url}?portal=demo"

        try:
            import stripe
            stripe.api_key = self.api_key

            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )

            return session.url

        except Exception as e:
            print(f"[BILLING] Stripe portal error: {str(e)}", flush=True)
            raise

    async def cancel_subscription(
        self,
        subscription_id: str,
        cancel_immediately: bool = False,
    ) -> Dict:
        """Cancel a Stripe subscription."""
        if not self.initialized:
            return {"status": "canceled", "id": subscription_id}

        try:
            import stripe
            stripe.api_key = self.api_key

            if cancel_immediately:
                subscription = stripe.Subscription.delete(subscription_id)
            else:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True,
                )

            return subscription

        except Exception as e:
            print(f"[BILLING] Stripe cancel error: {str(e)}", flush=True)
            raise


class PayPalProvider:
    """PayPal payment provider integration."""

    def __init__(self):
        self.client_id = os.getenv("PAYPAL_CLIENT_ID")
        self.client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
        self.sandbox = os.getenv("PAYPAL_SANDBOX", "true").lower() == "true"
        self.initialized = bool(self.client_id and self.client_secret)

        if self.initialized:
            base = "sandbox" if self.sandbox else "api"
            self.api_base = f"https://api-m.{base}.paypal.com"
            print(f"[BILLING] PayPal provider initialized (sandbox={self.sandbox})", flush=True)

    async def create_subscription(
        self,
        user_id: str,
        plan: SubscriptionPlan,
        interval: BillingInterval,
        success_url: str,
        cancel_url: str,
    ) -> Dict:
        """Create PayPal subscription."""
        if not self.initialized:
            return {
                "subscription_id": f"sub_paypal_demo_{user_id}",
                "approval_url": f"{success_url}?subscription_id=demo",
            }

        # In production, use PayPal SDK to create subscription
        # This is a simplified mock
        return {
            "subscription_id": f"sub_paypal_{user_id}_{plan.value}",
            "approval_url": f"{success_url}?subscription_id=paypal_pending",
        }

    async def cancel_subscription(self, subscription_id: str, reason: str = "") -> Dict:
        """Cancel PayPal subscription."""
        if not self.initialized:
            return {"status": "canceled", "id": subscription_id}

        # In production, call PayPal API
        return {"status": "canceled", "id": subscription_id}


class BillingService:
    """
    Main billing service orchestrating Stripe and PayPal.
    """

    def __init__(self):
        self.repository = BillingRepository()
        self.stripe = StripeProvider()
        self.paypal = PayPalProvider()
        print("[BILLING] Service initialized", flush=True)

    def get_plan_info(self, plan: SubscriptionPlan) -> PlanInfo:
        """Get detailed plan information."""
        details = PLAN_DETAILS[plan.value]
        return PlanInfo(
            id=plan,
            name=details["name"],
            description=details["description"],
            price_monthly_usd=details["price_monthly_usd"],
            price_yearly_usd=details["price_yearly_usd"],
            features=PlanFeatures(**details["features"]),
            popular=plan == SubscriptionPlan.PRO,
        )

    def get_all_plans(self) -> List[PlanInfo]:
        """Get all available plans."""
        return [self.get_plan_info(plan) for plan in SubscriptionPlan]

    async def create_checkout_session(
        self,
        user_id: str,
        email: str,
        plan: SubscriptionPlan,
        interval: BillingInterval,
        provider: PaymentProvider,
        success_url: str,
        cancel_url: str,
    ) -> CheckoutSessionResponse:
        """Create a checkout session for subscription."""
        if plan == SubscriptionPlan.FREE:
            # Free plan doesn't need checkout
            await self._create_free_subscription(user_id)
            return CheckoutSessionResponse(
                session_id="free",
                checkout_url=success_url,
                provider=provider,
            )

        if provider == PaymentProvider.STRIPE:
            result = await self.stripe.create_checkout_session(
                user_id=user_id,
                customer_email=email,
                plan=plan,
                interval=interval,
                success_url=success_url,
                cancel_url=cancel_url,
            )
            return CheckoutSessionResponse(
                session_id=result["session_id"],
                checkout_url=result["checkout_url"],
                provider=provider,
            )

        elif provider == PaymentProvider.PAYPAL:
            result = await self.paypal.create_subscription(
                user_id=user_id,
                plan=plan,
                interval=interval,
                success_url=success_url,
                cancel_url=cancel_url,
            )
            return CheckoutSessionResponse(
                session_id=result["subscription_id"],
                checkout_url=result["approval_url"],
                provider=provider,
            )

        raise ValueError(f"Unknown provider: {provider}")

    async def _create_free_subscription(self, user_id: str) -> Subscription:
        """Create a free tier subscription."""
        now = datetime.utcnow()
        subscription = Subscription(
            user_id=user_id,
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
            provider=PaymentProvider.STRIPE,  # Default, not used for free
            billing_interval=BillingInterval.MONTHLY,
            current_period_start=now,
            current_period_end=now + timedelta(days=36500),  # 100 years
        )
        await self.repository.save_subscription(subscription)
        return subscription

    async def get_subscription(self, user_id: str) -> Optional[SubscriptionResponse]:
        """Get user's current subscription."""
        subscription = await self.repository.get_user_subscription(user_id)

        if not subscription:
            # Return free tier by default
            subscription = await self._create_free_subscription(user_id)

        plan_info = self.get_plan_info(subscription.plan)

        return SubscriptionResponse(
            subscription=subscription,
            plan_info=plan_info,
            next_invoice_date=subscription.current_period_end if subscription.status == SubscriptionStatus.ACTIVE else None,
            next_invoice_amount_usd=plan_info.price_monthly_usd if subscription.billing_interval == BillingInterval.MONTHLY else plan_info.price_yearly_usd,
        )

    async def cancel_subscription(
        self,
        user_id: str,
        reason: Optional[str] = None,
        cancel_immediately: bool = False,
    ) -> Subscription:
        """Cancel user's subscription."""
        subscription = await self.repository.get_user_subscription(user_id)

        if not subscription:
            raise ValueError("No active subscription found")

        if subscription.plan == SubscriptionPlan.FREE:
            raise ValueError("Cannot cancel free tier")

        # Cancel with provider
        if subscription.provider == PaymentProvider.STRIPE and subscription.stripe_subscription_id:
            await self.stripe.cancel_subscription(
                subscription.stripe_subscription_id,
                cancel_immediately,
            )

        elif subscription.provider == PaymentProvider.PAYPAL and subscription.paypal_subscription_id:
            await self.paypal.cancel_subscription(
                subscription.paypal_subscription_id,
                reason or "",
            )

        # Update subscription status
        subscription.status = SubscriptionStatus.CANCELED
        subscription.canceled_at = datetime.utcnow()
        subscription.updated_at = datetime.utcnow()

        await self.repository.save_subscription(subscription)

        print(f"[BILLING] Subscription canceled for user {user_id}", flush=True)

        return subscription

    async def create_portal_session(
        self,
        user_id: str,
        return_url: str,
    ) -> PortalSessionResponse:
        """Create a billing portal session for user to manage subscription."""
        subscription = await self.repository.get_user_subscription(user_id)

        if not subscription or not subscription.stripe_customer_id:
            raise ValueError("No Stripe customer found")

        portal_url = await self.stripe.create_portal_session(
            subscription.stripe_customer_id,
            return_url,
        )

        return PortalSessionResponse(portal_url=portal_url)

    async def handle_webhook(
        self,
        provider: PaymentProvider,
        payload: bytes,
        signature: str,
    ) -> Dict:
        """Handle webhook from payment provider."""
        if provider == PaymentProvider.STRIPE:
            return await self._handle_stripe_webhook(payload, signature)
        elif provider == PaymentProvider.PAYPAL:
            return await self._handle_paypal_webhook(payload)

        raise ValueError(f"Unknown provider: {provider}")

    async def _handle_stripe_webhook(self, payload: bytes, signature: str) -> Dict:
        """Handle Stripe webhook events."""
        if not self.stripe.initialized:
            print("[BILLING] Stripe webhook received (demo mode)", flush=True)
            return {"received": True}

        try:
            import stripe
            stripe.api_key = self.stripe.api_key

            event = stripe.Webhook.construct_event(
                payload,
                signature,
                self.stripe.webhook_secret,
            )

            event_type = event["type"]
            data = event["data"]["object"]

            print(f"[BILLING] Stripe webhook: {event_type}", flush=True)

            if event_type == "checkout.session.completed":
                await self._handle_checkout_completed(data)

            elif event_type == "customer.subscription.updated":
                await self._handle_subscription_updated(data)

            elif event_type == "customer.subscription.deleted":
                await self._handle_subscription_deleted(data)

            elif event_type == "invoice.paid":
                await self._handle_invoice_paid(data)

            elif event_type == "invoice.payment_failed":
                await self._handle_payment_failed(data)

            return {"received": True, "event_type": event_type}

        except Exception as e:
            print(f"[BILLING] Stripe webhook error: {str(e)}", flush=True)
            raise

    async def _handle_paypal_webhook(self, payload: bytes) -> Dict:
        """Handle PayPal webhook events."""
        print("[BILLING] PayPal webhook received", flush=True)
        # In production, verify and process PayPal webhooks
        return {"received": True}

    async def _handle_checkout_completed(self, data: Dict) -> None:
        """Handle successful checkout."""
        user_id = data.get("metadata", {}).get("user_id")
        plan_str = data.get("metadata", {}).get("plan")
        stripe_subscription_id = data.get("subscription")
        stripe_customer_id = data.get("customer")

        if not user_id:
            return

        plan = SubscriptionPlan(plan_str) if plan_str else SubscriptionPlan.STARTER
        now = datetime.utcnow()

        subscription = Subscription(
            user_id=user_id,
            plan=plan,
            status=SubscriptionStatus.ACTIVE,
            provider=PaymentProvider.STRIPE,
            stripe_subscription_id=stripe_subscription_id,
            stripe_customer_id=stripe_customer_id,
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
        )

        await self.repository.save_subscription(subscription)
        print(f"[BILLING] Subscription created for user {user_id}: {plan.value}", flush=True)

    async def _handle_subscription_updated(self, data: Dict) -> None:
        """Handle subscription update."""
        # Would update subscription details based on Stripe data
        pass

    async def _handle_subscription_deleted(self, data: Dict) -> None:
        """Handle subscription deletion."""
        # Would mark subscription as canceled
        pass

    async def _handle_invoice_paid(self, data: Dict) -> None:
        """Handle successful invoice payment."""
        # Would record payment and extend subscription
        pass

    async def _handle_payment_failed(self, data: Dict) -> None:
        """Handle failed payment."""
        # Would update subscription status and notify user
        pass

    async def get_user_invoices(self, user_id: str) -> List[Invoice]:
        """Get user's invoice history."""
        # In production, fetch from Stripe API
        return []

    async def get_payment_history(self, user_id: str) -> List[Payment]:
        """Get user's payment history."""
        return await self.repository.get_user_payments(user_id)


# Global instance
billing_service: Optional[BillingService] = None


def get_billing_service() -> BillingService:
    """Get or create billing service instance."""
    global billing_service
    if billing_service is None:
        billing_service = BillingService()
    return billing_service
