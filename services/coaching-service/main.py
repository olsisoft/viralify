"""
Viralify Coaching Service
AI-powered fame coaching with growth plans, missions, and achievements
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from enum import Enum
import os
import uuid
import random
import asyncio
import httpx

# Use shared LLM provider for model name resolution
try:
    from shared.llm_provider import get_model_name as _get_model_name
    _HAS_SHARED_LLM = True
except ImportError:
    _HAS_SHARED_LLM = False
    _get_model_name = lambda tier: {"fast": "gpt-4o-mini", "quality": "gpt-4o"}.get(tier, "gpt-4o-mini")

app = FastAPI(
    title="Viralify Coaching Service",
    description="AI Fame Coaching - Plans, Missions, Achievements",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ============================================================
# ENUMS
# ============================================================

class UserLevel(str, Enum):
    BEGINNER = "beginner"
    CREATOR = "creator"
    RISING_STAR = "rising_star"
    INFLUENCER = "influencer"
    CELEBRITY = "celebrity"

class PlanType(str, Enum):
    THIRTY_DAY = "30_day"
    SIXTY_DAY = "60_day"
    NINETY_DAY = "90_day"

class MissionType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    CHALLENGE = "challenge"
    ONBOARDING = "onboarding"

class MissionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

class BadgeRarity(str, Enum):
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"

# ============================================================
# MODELS - Request/Response
# ============================================================

class SkillProfile(BaseModel):
    user_id: str
    current_level: UserLevel = UserLevel.BEGINNER
    experience_points: int = 0
    skills: Dict[str, int] = Field(default_factory=dict)  # skill_name: level (1-10)
    niche: Optional[str] = None
    next_level_xp: int = 1000
    level_progress_percent: float = 0

class GrowthPlanRequest(BaseModel):
    user_id: str
    plan_type: PlanType = PlanType.THIRTY_DAY
    goals: List[Dict[str, Any]] = Field(default_factory=list)  # [{metric, target}]
    niche: Optional[str] = None
    current_followers: int = 0
    platforms: List[str] = Field(default_factory=lambda: ["tiktok"])

class GrowthPlan(BaseModel):
    id: str
    user_id: str
    plan_type: PlanType
    title: str
    description: str
    goals: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    start_date: date
    end_date: date
    status: str = "active"
    progress_percent: float = 0
    created_at: datetime

class Mission(BaseModel):
    id: str
    title: str
    description: str
    mission_type: MissionType
    category: str  # content, engagement, growth, learning
    difficulty: str  # easy, medium, hard
    xp_reward: int
    requirements: Dict[str, Any]
    badge_reward_id: Optional[str] = None
    progress: Dict[str, Any] = Field(default_factory=dict)
    status: MissionStatus = MissionStatus.ACTIVE
    expires_at: Optional[datetime] = None

class UserMission(BaseModel):
    id: str
    user_id: str
    mission: Mission
    status: MissionStatus
    progress: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None

class Badge(BaseModel):
    id: str
    name: str
    description: str
    icon_url: str
    category: str  # milestone, streak, skill, challenge
    rarity: BadgeRarity
    xp_value: int
    requirements: Dict[str, Any]
    earned: bool = False
    earned_at: Optional[datetime] = None

class CoachingTip(BaseModel):
    id: str
    tip_type: str  # performance, trend, engagement, growth
    title: str
    content: str
    priority: int  # 1-5, higher = more important
    action_url: Optional[str] = None
    is_read: bool = False
    created_at: datetime

class StreakInfo(BaseModel):
    user_id: str
    daily_post_streak: int = 0
    longest_streak: int = 0
    last_post_date: Optional[date] = None
    streak_at_risk: bool = False

class XPGainEvent(BaseModel):
    user_id: str
    amount: int
    source: str  # mission_complete, badge_earned, streak_bonus, content_posted
    source_id: Optional[str] = None

class LevelUpResult(BaseModel):
    new_level: UserLevel
    previous_level: UserLevel
    total_xp: int
    rewards: List[Dict[str, Any]] = Field(default_factory=list)

# ============================================================
# DEMO DATA
# ============================================================

DEMO_BADGES = [
    Badge(
        id="badge-first-post",
        name="First Steps",
        description="Published your first content",
        icon_url="/badges/first-steps.svg",
        category="milestone",
        rarity=BadgeRarity.COMMON,
        xp_value=50,
        requirements={"posts_count": 1}
    ),
    Badge(
        id="badge-viral-hit",
        name="Viral Hit",
        description="Got 10,000+ views on a single video",
        icon_url="/badges/viral-hit.svg",
        category="milestone",
        rarity=BadgeRarity.RARE,
        xp_value=500,
        requirements={"single_video_views": 10000}
    ),
    Badge(
        id="badge-streak-7",
        name="Week Warrior",
        description="7-day posting streak",
        icon_url="/badges/week-warrior.svg",
        category="streak",
        rarity=BadgeRarity.UNCOMMON,
        xp_value=200,
        requirements={"streak_days": 7}
    ),
    Badge(
        id="badge-streak-30",
        name="Consistency King",
        description="30-day posting streak",
        icon_url="/badges/consistency-king.svg",
        category="streak",
        rarity=BadgeRarity.EPIC,
        xp_value=1000,
        requirements={"streak_days": 30}
    ),
    Badge(
        id="badge-engagement-master",
        name="Engagement Master",
        description="Achieved 10% engagement rate",
        icon_url="/badges/engagement-master.svg",
        category="skill",
        rarity=BadgeRarity.RARE,
        xp_value=400,
        requirements={"engagement_rate": 10}
    ),
    Badge(
        id="badge-trend-rider",
        name="Trend Rider",
        description="Used 5 trending sounds/hashtags",
        icon_url="/badges/trend-rider.svg",
        category="skill",
        rarity=BadgeRarity.UNCOMMON,
        xp_value=150,
        requirements={"trends_used": 5}
    ),
    Badge(
        id="badge-1k-followers",
        name="Rising Creator",
        description="Reached 1,000 followers",
        icon_url="/badges/rising-creator.svg",
        category="milestone",
        rarity=BadgeRarity.UNCOMMON,
        xp_value=300,
        requirements={"followers": 1000}
    ),
    Badge(
        id="badge-10k-followers",
        name="Influencer Status",
        description="Reached 10,000 followers",
        icon_url="/badges/influencer-status.svg",
        category="milestone",
        rarity=BadgeRarity.EPIC,
        xp_value=1500,
        requirements={"followers": 10000}
    ),
]

DEMO_MISSIONS = [
    Mission(
        id="mission-daily-post",
        title="Daily Creator",
        description="Post at least 1 piece of content today",
        mission_type=MissionType.DAILY,
        category="content",
        difficulty="easy",
        xp_reward=50,
        requirements={"posts_today": 1}
    ),
    Mission(
        id="mission-engage-10",
        title="Community Builder",
        description="Reply to 10 comments on your content",
        mission_type=MissionType.DAILY,
        category="engagement",
        difficulty="medium",
        xp_reward=75,
        requirements={"replies_today": 10}
    ),
    Mission(
        id="mission-trend-use",
        title="Trend Surfer",
        description="Create content using a trending sound or hashtag",
        mission_type=MissionType.DAILY,
        category="growth",
        difficulty="medium",
        xp_reward=100,
        requirements={"trending_content": 1}
    ),
    Mission(
        id="mission-weekly-5-posts",
        title="Prolific Creator",
        description="Post 5 pieces of content this week",
        mission_type=MissionType.WEEKLY,
        category="content",
        difficulty="medium",
        xp_reward=300,
        requirements={"posts_week": 5}
    ),
    Mission(
        id="mission-weekly-collab",
        title="Network Builder",
        description="Collaborate or duet with another creator",
        mission_type=MissionType.WEEKLY,
        category="growth",
        difficulty="hard",
        xp_reward=500,
        requirements={"collaborations": 1}
    ),
    Mission(
        id="mission-challenge-7day",
        title="7-Day Challenge",
        description="Post every day for 7 consecutive days",
        mission_type=MissionType.CHALLENGE,
        category="content",
        difficulty="hard",
        xp_reward=750,
        requirements={"consecutive_days": 7},
        badge_reward_id="badge-streak-7"
    ),
]

DEMO_TIPS = [
    CoachingTip(
        id="tip-1",
        tip_type="performance",
        title="Your best posting time",
        content="Based on your audience analytics, your followers are most active between 7-9 PM. Try posting during this window for maximum reach.",
        priority=4,
        action_url="/dashboard/scheduler",
        is_read=False,
        created_at=datetime.now()
    ),
    CoachingTip(
        id="tip-2",
        tip_type="trend",
        title="Trending opportunity",
        content="The sound 'Original Sound - trending_creator' is gaining traction in your niche. Consider creating content with it in the next 24 hours.",
        priority=5,
        action_url="/dashboard/trends",
        is_read=False,
        created_at=datetime.now()
    ),
    CoachingTip(
        id="tip-3",
        tip_type="engagement",
        title="Boost your engagement",
        content="Videos with questions in the caption get 2x more comments. Try ending your next video with a question for your audience.",
        priority=3,
        is_read=False,
        created_at=datetime.now()
    ),
    CoachingTip(
        id="tip-4",
        tip_type="growth",
        title="Hashtag strategy",
        content="You're only using broad hashtags. Mix in 2-3 niche-specific hashtags (under 1M posts) to reach your target audience better.",
        priority=4,
        is_read=False,
        created_at=datetime.now()
    ),
]

# Level thresholds
LEVEL_THRESHOLDS = {
    UserLevel.BEGINNER: 0,
    UserLevel.CREATOR: 1000,
    UserLevel.RISING_STAR: 5000,
    UserLevel.INFLUENCER: 15000,
    UserLevel.CELEBRITY: 50000,
}

def get_level_for_xp(xp: int) -> UserLevel:
    """Determine user level based on XP"""
    for level in reversed(list(UserLevel)):
        if xp >= LEVEL_THRESHOLDS[level]:
            return level
    return UserLevel.BEGINNER

def get_next_level_xp(current_level: UserLevel) -> int:
    """Get XP needed for next level"""
    levels = list(UserLevel)
    current_idx = levels.index(current_level)
    if current_idx < len(levels) - 1:
        next_level = levels[current_idx + 1]
        return LEVEL_THRESHOLDS[next_level]
    return LEVEL_THRESHOLDS[UserLevel.CELEBRITY]

# ============================================================
# AI COACHING (OpenAI Integration)
# ============================================================

async def generate_growth_plan_ai(request: GrowthPlanRequest) -> Dict[str, Any]:
    """Generate personalized growth plan using AI"""

    if DEMO_MODE or not OPENAI_API_KEY:
        # Return demo plan
        days = int(request.plan_type.value.replace("_day", ""))
        return {
            "title": f"{days}-Day Fame Accelerator Plan",
            "description": f"A personalized {days}-day plan to grow your {request.niche or 'content'} presence and reach your goals.",
            "milestones": [
                {
                    "day": days // 4,
                    "title": "Foundation Phase Complete",
                    "description": "Establish consistent posting schedule and brand voice",
                    "target_followers": request.current_followers + (request.current_followers * 0.1)
                },
                {
                    "day": days // 2,
                    "title": "Growth Acceleration",
                    "description": "Leverage trends and collaborations for rapid growth",
                    "target_followers": request.current_followers + (request.current_followers * 0.25)
                },
                {
                    "day": (days * 3) // 4,
                    "title": "Community Building",
                    "description": "Focus on engagement and building loyal followers",
                    "target_followers": request.current_followers + (request.current_followers * 0.4)
                },
                {
                    "day": days,
                    "title": "Goal Achievement",
                    "description": "Reach your target metrics and establish sustainable growth",
                    "target_followers": request.current_followers + (request.current_followers * 0.5)
                }
            ],
            "weekly_focus": [
                "Content consistency and quality improvement",
                "Trend identification and participation",
                "Audience engagement and community building",
                "Analytics review and strategy adjustment"
            ],
            "daily_tasks": [
                "Post at least 1 piece of content",
                "Engage with 20 accounts in your niche",
                "Reply to all comments within 1 hour",
                "Research trending topics for 15 minutes"
            ]
        }

    # Real AI generation
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": _get_model_name("quality"),
                "messages": [
                    {
                        "role": "system",
                        "content": """You are an expert social media growth coach. Create detailed, actionable growth plans
                        for content creators looking to build their audience and become famous. Focus on practical strategies
                        that work on TikTok, Instagram Reels, and YouTube Shorts."""
                    },
                    {
                        "role": "user",
                        "content": f"""Create a {request.plan_type.value} growth plan for a creator with these details:
                        - Current followers: {request.current_followers}
                        - Niche: {request.niche or 'General content'}
                        - Platforms: {', '.join(request.platforms)}
                        - Goals: {request.goals}

                        Return a JSON object with: title, description, milestones (array with day, title, description, target_followers),
                        weekly_focus (array of 4 items), daily_tasks (array of 4-5 items)"""
                    }
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.7
            },
            timeout=30.0
        )

        if response.status_code == 200:
            import json
            result = response.json()
            return json.loads(result["choices"][0]["message"]["content"])
        else:
            raise HTTPException(status_code=500, detail="AI plan generation failed")

async def generate_personalized_tips(user_id: str, context: Dict[str, Any]) -> List[CoachingTip]:
    """Generate personalized coaching tips based on user performance"""

    if DEMO_MODE or not OPENAI_API_KEY:
        # Return subset of demo tips based on context
        tips = DEMO_TIPS.copy()
        random.shuffle(tips)
        return tips[:3]

    # Real AI would analyze user data and generate contextual tips
    # For now, return demo tips
    return DEMO_TIPS[:3]

# ============================================================
# API ENDPOINTS - Skill Profile
# ============================================================

@app.get("/api/v1/coaching/profile/{user_id}", response_model=SkillProfile)
async def get_skill_profile(user_id: str):
    """Get user's skill profile and level"""

    if DEMO_MODE:
        xp = random.randint(500, 3000)
        level = get_level_for_xp(xp)
        next_xp = get_next_level_xp(level)
        current_threshold = LEVEL_THRESHOLDS[level]

        return SkillProfile(
            user_id=user_id,
            current_level=level,
            experience_points=xp,
            skills={
                "content_creation": random.randint(3, 8),
                "engagement": random.randint(2, 7),
                "trend_awareness": random.randint(4, 9),
                "consistency": random.randint(3, 8),
                "storytelling": random.randint(2, 7),
                "analytics": random.randint(1, 6)
            },
            niche="entertainment",
            next_level_xp=next_xp,
            level_progress_percent=((xp - current_threshold) / (next_xp - current_threshold)) * 100
        )

    # Database query would go here
    raise HTTPException(status_code=404, detail="Profile not found")

@app.post("/api/v1/coaching/xp/add", response_model=SkillProfile)
async def add_experience_points(event: XPGainEvent):
    """Add XP to user and check for level up"""

    if DEMO_MODE:
        # Simulate XP gain
        current_xp = random.randint(500, 3000)
        new_xp = current_xp + event.amount
        new_level = get_level_for_xp(new_xp)
        next_xp = get_next_level_xp(new_level)
        current_threshold = LEVEL_THRESHOLDS[new_level]

        return SkillProfile(
            user_id=event.user_id,
            current_level=new_level,
            experience_points=new_xp,
            skills={
                "content_creation": random.randint(3, 8),
                "engagement": random.randint(2, 7),
                "trend_awareness": random.randint(4, 9),
                "consistency": random.randint(3, 8),
                "storytelling": random.randint(2, 7),
                "analytics": random.randint(1, 6)
            },
            niche="entertainment",
            next_level_xp=next_xp,
            level_progress_percent=((new_xp - current_threshold) / (next_xp - current_threshold)) * 100
        )

    raise HTTPException(status_code=500, detail="Database not configured")

# ============================================================
# API ENDPOINTS - Growth Plans
# ============================================================

@app.post("/api/v1/coaching/plan/generate", response_model=GrowthPlan)
async def generate_growth_plan(request: GrowthPlanRequest):
    """Generate a personalized AI growth plan"""

    plan_data = await generate_growth_plan_ai(request)
    days = int(request.plan_type.value.replace("_day", ""))

    return GrowthPlan(
        id=str(uuid.uuid4()),
        user_id=request.user_id,
        plan_type=request.plan_type,
        title=plan_data["title"],
        description=plan_data["description"],
        goals=request.goals or [{"metric": "followers", "target": request.current_followers * 1.5}],
        milestones=plan_data["milestones"],
        start_date=date.today(),
        end_date=date.today() + timedelta(days=days),
        status="active",
        progress_percent=0,
        created_at=datetime.now()
    )

@app.get("/api/v1/coaching/plan/active/{user_id}", response_model=Optional[GrowthPlan])
async def get_active_plan(user_id: str):
    """Get user's active growth plan"""

    if DEMO_MODE:
        return GrowthPlan(
            id="plan-demo-001",
            user_id=user_id,
            plan_type=PlanType.THIRTY_DAY,
            title="30-Day Fame Accelerator Plan",
            description="Your personalized path to viral success",
            goals=[
                {"metric": "followers", "target": 10000, "current": 5234},
                {"metric": "avg_views", "target": 50000, "current": 23500},
                {"metric": "engagement_rate", "target": 8, "current": 5.2}
            ],
            milestones=[
                {"day": 7, "title": "Foundation Phase", "status": "completed"},
                {"day": 14, "title": "Growth Phase", "status": "in_progress"},
                {"day": 21, "title": "Engagement Phase", "status": "pending"},
                {"day": 30, "title": "Goal Achievement", "status": "pending"}
            ],
            start_date=date.today() - timedelta(days=10),
            end_date=date.today() + timedelta(days=20),
            status="active",
            progress_percent=33.3,
            created_at=datetime.now() - timedelta(days=10)
        )

    return None

# ============================================================
# API ENDPOINTS - Missions
# ============================================================

@app.get("/api/v1/coaching/missions/today/{user_id}", response_model=List[UserMission])
async def get_today_missions(user_id: str):
    """Get user's missions for today"""

    if DEMO_MODE:
        missions = []
        for i, mission in enumerate(DEMO_MISSIONS[:4]):
            progress = {}
            status = MissionStatus.ACTIVE

            # Simulate some progress
            if i == 0:  # First mission completed
                status = MissionStatus.COMPLETED
                progress = {"posts_today": 1}
            elif i == 1:  # Second mission in progress
                progress = {"replies_today": 6}

            missions.append(UserMission(
                id=f"user-mission-{i}",
                user_id=user_id,
                mission=mission,
                status=status,
                progress=progress,
                started_at=datetime.now().replace(hour=0, minute=0),
                completed_at=datetime.now() if status == MissionStatus.COMPLETED else None
            ))

        return missions

    raise HTTPException(status_code=500, detail="Database not configured")

@app.get("/api/v1/coaching/missions/weekly/{user_id}", response_model=List[UserMission])
async def get_weekly_missions(user_id: str):
    """Get user's weekly missions"""

    if DEMO_MODE:
        weekly = [m for m in DEMO_MISSIONS if m.mission_type == MissionType.WEEKLY]
        return [
            UserMission(
                id=f"user-mission-weekly-{i}",
                user_id=user_id,
                mission=mission,
                status=MissionStatus.ACTIVE,
                progress={"posts_week": random.randint(1, 3)},
                started_at=datetime.now() - timedelta(days=datetime.now().weekday())
            )
            for i, mission in enumerate(weekly)
        ]

    raise HTTPException(status_code=500, detail="Database not configured")

@app.get("/api/v1/coaching/missions/challenges/{user_id}", response_model=List[UserMission])
async def get_active_challenges(user_id: str):
    """Get user's active challenges"""

    if DEMO_MODE:
        challenges = [m for m in DEMO_MISSIONS if m.mission_type == MissionType.CHALLENGE]
        return [
            UserMission(
                id=f"user-challenge-{i}",
                user_id=user_id,
                mission=challenge,
                status=MissionStatus.ACTIVE,
                progress={"consecutive_days": random.randint(2, 5)},
                started_at=datetime.now() - timedelta(days=random.randint(2, 5))
            )
            for i, challenge in enumerate(challenges)
        ]

    raise HTTPException(status_code=500, detail="Database not configured")

@app.post("/api/v1/coaching/missions/{mission_id}/progress")
async def update_mission_progress(
    mission_id: str,
    user_id: str = Query(...),
    progress: Dict[str, Any] = {}
):
    """Update mission progress"""

    if DEMO_MODE:
        # Check if mission is completed based on progress
        is_completed = random.choice([True, False])

        return {
            "mission_id": mission_id,
            "user_id": user_id,
            "progress": progress,
            "status": "completed" if is_completed else "active",
            "xp_earned": 100 if is_completed else 0,
            "badge_earned": None
        }

    raise HTTPException(status_code=500, detail="Database not configured")

# ============================================================
# API ENDPOINTS - Badges/Achievements
# ============================================================

@app.get("/api/v1/coaching/badges", response_model=List[Badge])
async def get_all_badges():
    """Get all available badges"""
    return DEMO_BADGES

@app.get("/api/v1/coaching/badges/user/{user_id}", response_model=List[Badge])
async def get_user_badges(user_id: str):
    """Get user's earned badges"""

    if DEMO_MODE:
        # Return some badges as earned
        earned_badges = []
        for i, badge in enumerate(DEMO_BADGES[:4]):
            earned_badge = badge.model_copy()
            earned_badge.earned = True
            earned_badge.earned_at = datetime.now() - timedelta(days=random.randint(1, 30))
            earned_badges.append(earned_badge)
        return earned_badges

    raise HTTPException(status_code=500, detail="Database not configured")

@app.get("/api/v1/coaching/badges/progress/{user_id}", response_model=List[Dict[str, Any]])
async def get_badge_progress(user_id: str):
    """Get progress towards unearned badges"""

    if DEMO_MODE:
        progress = []
        for badge in DEMO_BADGES[4:]:  # Unearned badges
            badge_progress = {
                "badge": badge,
                "progress_percent": random.randint(20, 80),
                "current_value": 0,
                "target_value": 0
            }

            # Set specific progress based on requirements
            if "streak_days" in badge.requirements:
                badge_progress["current_value"] = random.randint(1, badge.requirements["streak_days"] - 1)
                badge_progress["target_value"] = badge.requirements["streak_days"]
            elif "followers" in badge.requirements:
                badge_progress["current_value"] = int(badge.requirements["followers"] * random.uniform(0.3, 0.9))
                badge_progress["target_value"] = badge.requirements["followers"]

            progress.append(badge_progress)

        return progress

    raise HTTPException(status_code=500, detail="Database not configured")

# ============================================================
# API ENDPOINTS - Streaks
# ============================================================

@app.get("/api/v1/coaching/streak/{user_id}", response_model=StreakInfo)
async def get_streak_info(user_id: str):
    """Get user's streak information"""

    if DEMO_MODE:
        streak = random.randint(3, 15)
        return StreakInfo(
            user_id=user_id,
            daily_post_streak=streak,
            longest_streak=max(streak, random.randint(10, 25)),
            last_post_date=date.today() - timedelta(days=random.choice([0, 1])),
            streak_at_risk=random.choice([True, False])
        )

    raise HTTPException(status_code=500, detail="Database not configured")

@app.post("/api/v1/coaching/streak/{user_id}/update", response_model=StreakInfo)
async def update_streak(user_id: str):
    """Update streak after posting content"""

    if DEMO_MODE:
        new_streak = random.randint(4, 16)
        return StreakInfo(
            user_id=user_id,
            daily_post_streak=new_streak,
            longest_streak=new_streak,
            last_post_date=date.today(),
            streak_at_risk=False
        )

    raise HTTPException(status_code=500, detail="Database not configured")

# ============================================================
# API ENDPOINTS - Coaching Tips
# ============================================================

@app.get("/api/v1/coaching/tips/{user_id}", response_model=List[CoachingTip])
async def get_coaching_tips(user_id: str, limit: int = Query(5, ge=1, le=20)):
    """Get personalized coaching tips"""

    tips = await generate_personalized_tips(user_id, {})
    return tips[:limit]

@app.post("/api/v1/coaching/tips/{tip_id}/read")
async def mark_tip_read(tip_id: str, user_id: str = Query(...)):
    """Mark a coaching tip as read"""

    return {"tip_id": tip_id, "is_read": True}

# ============================================================
# API ENDPOINTS - Leaderboard
# ============================================================

@app.get("/api/v1/coaching/leaderboard")
async def get_leaderboard(
    timeframe: str = Query("weekly", regex="^(daily|weekly|monthly|all_time)$"),
    limit: int = Query(10, ge=1, le=100)
):
    """Get XP leaderboard"""

    if DEMO_MODE:
        usernames = ["viral_queen", "trend_master", "content_king", "growth_guru",
                     "social_star", "fame_seeker", "creator_pro", "engagement_expert"]

        leaderboard = []
        for i in range(min(limit, len(usernames))):
            leaderboard.append({
                "rank": i + 1,
                "user_id": f"user-{i}",
                "username": usernames[i],
                "xp": 5000 - (i * 400) + random.randint(-100, 100),
                "level": list(UserLevel)[min(4, max(0, 4 - i // 2))].value,
                "badges_count": 8 - i
            })

        return {"timeframe": timeframe, "leaderboard": leaderboard}

    raise HTTPException(status_code=500, detail="Database not configured")

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "coaching-service",
        "demo_mode": DEMO_MODE,
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return {
        "service": "Viralify Coaching Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
