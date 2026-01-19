"""
Billing and Subscription Models

Pydantic models for monetization with Stripe and PayPal.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PaymentProvider(str, Enum):
    """Supported payment providers"""
    STRIPE = "stripe"
    PAYPAL = "paypal"


class SubscriptionPlan(str, Enum):
    """Available subscription plans"""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Subscription status"""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"


class PaymentStatus(str, Enum):
    """Payment status"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"


class BillingInterval(str, Enum):
    """Billing interval"""
    MONTHLY = "monthly"
    YEARLY = "yearly"


# Plan features and pricing
PLAN_DETAILS: Dict[str, Dict] = {
    "free": {
        "name": "Free",
        "description": "Get started with basic features",
        "price_monthly_usd": 0,
        "price_yearly_usd": 0,
        "features": {
            "courses_per_month": 3,
            "max_lectures_per_course": 5,
            "storage_gb": 1,
            "api_budget_usd": 5,
            "voice_cloning": False,
            "multi_language": False,
            "priority_support": False,
            "custom_branding": False,
            "analytics": "basic",
            "export_formats": ["mp4"],
        },
    },
    "starter": {
        "name": "Starter",
        "description": "For individual creators",
        "price_monthly_usd": 19,
        "price_yearly_usd": 190,  # ~2 months free
        "features": {
            "courses_per_month": 10,
            "max_lectures_per_course": 15,
            "storage_gb": 10,
            "api_budget_usd": 25,
            "voice_cloning": True,
            "multi_language": 3,  # Number of languages
            "priority_support": False,
            "custom_branding": False,
            "analytics": "standard",
            "export_formats": ["mp4", "webm"],
        },
    },
    "pro": {
        "name": "Pro",
        "description": "For professional educators",
        "price_monthly_usd": 49,
        "price_yearly_usd": 490,
        "features": {
            "courses_per_month": 50,
            "max_lectures_per_course": 30,
            "storage_gb": 50,
            "api_budget_usd": 100,
            "voice_cloning": True,
            "multi_language": 10,
            "priority_support": True,
            "custom_branding": True,
            "analytics": "advanced",
            "export_formats": ["mp4", "webm", "mov"],
            "team_members": 3,
        },
    },
    "enterprise": {
        "name": "Enterprise",
        "description": "For teams and organizations",
        "price_monthly_usd": 199,
        "price_yearly_usd": 1990,
        "features": {
            "courses_per_month": -1,  # Unlimited
            "max_lectures_per_course": -1,
            "storage_gb": 500,
            "api_budget_usd": 500,
            "voice_cloning": True,
            "multi_language": -1,  # All languages
            "priority_support": True,
            "custom_branding": True,
            "analytics": "enterprise",
            "export_formats": ["mp4", "webm", "mov", "mkv"],
            "team_members": -1,  # Unlimited
            "sso": True,
            "api_access": True,
            "dedicated_support": True,
        },
    },
}


class PlanFeatures(BaseModel):
    """Plan features model"""
    courses_per_month: int
    max_lectures_per_course: int
    storage_gb: float
    api_budget_usd: float
    voice_cloning: bool
    multi_language: int  # Number of languages, -1 for unlimited
    priority_support: bool
    custom_branding: bool
    analytics: str
    export_formats: List[str]
    team_members: int = 1
    sso: bool = False
    api_access: bool = False
    dedicated_support: bool = False


class PlanInfo(BaseModel):
    """Complete plan information"""
    id: SubscriptionPlan
    name: str
    description: str
    price_monthly_usd: float
    price_yearly_usd: float
    features: PlanFeatures
    popular: bool = False  # For highlighting recommended plan


class Subscription(BaseModel):
    """User subscription"""
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    plan: SubscriptionPlan
    status: SubscriptionStatus
    provider: PaymentProvider
    billing_interval: BillingInterval = BillingInterval.MONTHLY

    # Provider IDs
    stripe_subscription_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    paypal_subscription_id: Optional[str] = None

    # Dates
    current_period_start: datetime
    current_period_end: datetime
    trial_end: Optional[datetime] = None
    canceled_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Payment(BaseModel):
    """Payment record"""
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    subscription_id: Optional[str] = None
    provider: PaymentProvider
    status: PaymentStatus

    # Amount
    amount_usd: float
    currency: str = "usd"

    # Provider details
    stripe_payment_intent_id: Optional[str] = None
    stripe_invoice_id: Optional[str] = None
    paypal_order_id: Optional[str] = None
    paypal_capture_id: Optional[str] = None

    # Metadata
    description: str = ""
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Invoice(BaseModel):
    """Invoice record"""
    id: str
    user_id: str
    subscription_id: str
    provider: PaymentProvider

    # Amount
    amount_usd: float
    amount_paid_usd: float
    currency: str = "usd"

    # Status
    status: str  # draft, open, paid, void, uncollectible
    paid: bool = False

    # URLs
    invoice_pdf_url: Optional[str] = None
    hosted_invoice_url: Optional[str] = None

    # Dates
    created_at: datetime
    due_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None


# Request/Response Models

class CreateCheckoutRequest(BaseModel):
    """Request to create a checkout session"""
    user_id: str
    plan: SubscriptionPlan
    billing_interval: BillingInterval = BillingInterval.MONTHLY
    provider: PaymentProvider = PaymentProvider.STRIPE
    success_url: str
    cancel_url: str


class CheckoutSessionResponse(BaseModel):
    """Response with checkout session details"""
    session_id: str
    checkout_url: str
    provider: PaymentProvider


class CreatePortalRequest(BaseModel):
    """Request to create a billing portal session"""
    user_id: str
    return_url: str


class PortalSessionResponse(BaseModel):
    """Response with portal session details"""
    portal_url: str


class CancelSubscriptionRequest(BaseModel):
    """Request to cancel subscription"""
    user_id: str
    reason: Optional[str] = None
    cancel_immediately: bool = False  # False = cancel at period end


class UpdateSubscriptionRequest(BaseModel):
    """Request to update subscription (upgrade/downgrade)"""
    user_id: str
    new_plan: SubscriptionPlan
    billing_interval: Optional[BillingInterval] = None


class SubscriptionResponse(BaseModel):
    """Response with subscription details"""
    subscription: Subscription
    plan_info: PlanInfo
    next_invoice_date: Optional[datetime] = None
    next_invoice_amount_usd: Optional[float] = None


class WebhookEvent(BaseModel):
    """Webhook event from payment provider"""
    id: str
    provider: PaymentProvider
    event_type: str
    data: Dict
    created_at: datetime = Field(default_factory=datetime.utcnow)
