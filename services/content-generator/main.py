"""
Content Generator Service - AI-Powered Content Creation
Uses LangChain with multiple AI agents for viral content generation
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
import asyncio
import json
import os
import re

# Database & Cache
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, Integer, JSON, ForeignKey, select
from redis.asyncio import Redis

# AI/LangChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler

# Message Queue
import aio_pika

# ========================================
# Configuration
# ========================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://tiktok_user:tiktok_secure_pass_2024@localhost:5432/tiktok_platform")
REDIS_URL = os.getenv("REDIS_URL", "redis://:redis_secure_2024@localhost:6379/1")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://tiktok:rabbitmq_secure_2024@localhost:5672/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TREND_SERVICE_URL = os.getenv("TREND_SERVICE_URL", "http://localhost:8000")

# ========================================
# FastAPI App
# ========================================

app = FastAPI(
    title="Content Generator Service",
    description="AI-powered content generation with multi-agent architecture",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Database Models
# ========================================

class Base(DeclarativeBase):
    pass

engine = create_async_engine(DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"), echo=True)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ========================================
# Pydantic Models
# ========================================

class CreatorProfileData(BaseModel):
    """Creator profile data for personalized script generation"""
    brandName: Optional[str] = None
    niche: Optional[str] = None
    tone: Optional[str] = None
    hookStyle: Optional[str] = None
    audienceLevel: Optional[str] = None
    ctaStyle: Optional[str] = None

class GenerateScriptRequest(BaseModel):
    user_id: Optional[UUID] = None
    topic: str
    niche: Optional[str] = None
    target_audience: Optional[str] = None
    duration_seconds: int = Field(default=60, ge=15, le=180)
    tone: str = Field(default="engaging", description="engaging, educational, humorous, inspirational")
    include_trends: bool = True
    specific_trends: Optional[List[str]] = None
    # New fields for Creator Profile integration
    system_prompt: Optional[str] = None
    profile: Optional[CreatorProfileData] = None
    style: Optional[str] = None
    platform: Optional[str] = "tiktok"
    format: Optional[str] = "standard"  # "standard" or "structured"

class GenerateCaptionRequest(BaseModel):
    user_id: UUID
    script: str
    max_length: int = Field(default=150, ge=50, le=300)
    include_hashtags: bool = True
    include_cta: bool = True

class GenerateHashtagsRequest(BaseModel):
    user_id: UUID
    content_description: str
    niche: Optional[str] = None
    max_hashtags: int = Field(default=10, ge=5, le=30)

class OptimizeContentRequest(BaseModel):
    user_id: UUID
    script: Optional[str] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    optimization_goals: List[str] = Field(default=["engagement", "reach"])

class ContentStrategyRequest(BaseModel):
    user_id: UUID
    niche: str
    goals: List[str]
    posting_frequency: str = Field(default="daily")
    time_horizon_days: int = Field(default=30, ge=7, le=90)

class AIAgentChatRequest(BaseModel):
    user_id: UUID
    agent_name: str
    message: str
    conversation_id: Optional[UUID] = None
    context: Optional[Dict[str, Any]] = None

class SceneData(BaseModel):
    """Individual scene in a structured script"""
    time: str
    visual: str
    audio: str
    visualType: Optional[str] = "image"  # image, diagram, video, text-overlay
    visualPrompt: Optional[str] = None

class ScriptResponse(BaseModel):
    script_id: UUID
    hook: str
    main_content: str
    cta: str
    full_script: str
    duration_estimate_seconds: int
    suggested_visuals: List[str]
    trending_elements: List[str]
    engagement_score: float
    # New: structured scenes for AI Video workflow
    scenes: Optional[List[SceneData]] = None

class CaptionResponse(BaseModel):
    caption: str
    hashtags: List[str]
    character_count: int
    estimated_reach_multiplier: float

class HashtagResponse(BaseModel):
    hashtags: List[Dict[str, Any]]
    total_reach_potential: int
    mix_analysis: Dict[str, int]

class OptimizationResponse(BaseModel):
    optimized_script: Optional[str]
    optimized_caption: Optional[str]
    optimized_hashtags: Optional[List[str]]
    improvements: List[str]
    expected_performance_lift: float

class StrategyResponse(BaseModel):
    strategy_id: UUID
    content_pillars: List[Dict[str, Any]]
    posting_schedule: List[Dict[str, Any]]
    content_ideas: List[Dict[str, Any]]
    growth_projections: Dict[str, Any]
    key_metrics_to_track: List[str]

class ChatResponse(BaseModel):
    conversation_id: UUID
    agent_name: str
    response: str
    suggested_actions: List[Dict[str, Any]]
    tokens_used: int

# ========================================
# AI Agents System
# ========================================

class TikTokAIAgents:
    """Multi-agent system for TikTok content creation"""

    def __init__(self):
        self.openai_llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )

        # Check if Anthropic key is valid (not a placeholder)
        anthropic_key_valid = ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("sk-ant-your")

        if anthropic_key_valid:
            self.anthropic_llm = ChatAnthropic(
                model="claude-3-opus-20240229",
                temperature=0.8,
                anthropic_api_key=ANTHROPIC_API_KEY
            )
        else:
            # Fallback to OpenAI if no valid Anthropic key
            self.anthropic_llm = self.openai_llm

        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all AI agents with their specific capabilities"""
        
        # ========================================
        # Agent 1: TrendScout - Trend Analysis
        # ========================================
        trend_tools = [
            Tool(
                name="get_trending_hashtags",
                func=self._get_trending_hashtags,
                description="Get current trending hashtags on TikTok for a specific niche or general"
            ),
            Tool(
                name="get_trending_sounds",
                func=self._get_trending_sounds,
                description="Get trending sounds and music on TikTok"
            ),
            Tool(
                name="analyze_viral_patterns",
                func=self._analyze_viral_patterns,
                description="Analyze patterns in viral content for a specific niche"
            ),
            Tool(
                name="predict_trend_longevity",
                func=self._predict_trend_longevity,
                description="Predict how long a trend will remain relevant"
            )
        ]
        
        trend_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are TrendScout, an expert AI agent specialized in analyzing TikTok trends.
Your role is to:
1. Identify emerging trends before they peak
2. Analyze viral patterns in successful content
3. Predict trend longevity and relevance
4. Match trends to user niches and audiences
5. Provide actionable insights for content creation

Always provide data-driven insights with specific examples and metrics when available.
Use your tools to gather real-time trend data before making recommendations."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        trend_agent = create_openai_tools_agent(self.openai_llm, trend_tools, trend_prompt)
        self.agents["TrendScout"] = AgentExecutor(
            agent=trend_agent,
            tools=trend_tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        # ========================================
        # Agent 2: ScriptGenius - Script Generation
        # ========================================
        script_tools = [
            Tool(
                name="generate_hook_variations",
                func=self._generate_hook_variations,
                description="Generate multiple hook variations for the first 3 seconds"
            ),
            Tool(
                name="get_successful_script_templates",
                func=self._get_script_templates,
                description="Get templates from successful viral videos"
            ),
            Tool(
                name="analyze_pacing",
                func=self._analyze_pacing,
                description="Analyze and suggest optimal pacing for video duration"
            )
        ]
        
        script_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are ScriptGenius, a creative AI agent specialized in writing TikTok video scripts.
Your expertise includes:
1. Creating attention-grabbing hooks in the first 3 seconds
2. Writing scripts optimized for different video lengths (15s, 30s, 60s, 3min)
3. Incorporating trending formats and sounds
4. Crafting compelling storytelling arcs
5. Including clear calls-to-action
6. Balancing entertainment with value delivery

IMPORTANT: Always structure your scripts with:
- HOOK (0-3 seconds): Immediate attention grabber
- BODY (main content): Value delivery with retention loops
- CTA (end): Clear call-to-action

Consider the target audience, current trends, and platform best practices."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        script_agent = create_openai_tools_agent(self.anthropic_llm, script_tools, script_prompt)
        self.agents["ScriptGenius"] = AgentExecutor(
            agent=script_agent,
            tools=script_tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        # ========================================
        # Agent 3: ContentOptimizer - Optimization
        # ========================================
        optimizer_tools = [
            Tool(
                name="analyze_caption_seo",
                func=self._analyze_caption_seo,
                description="Analyze caption for SEO and discoverability"
            ),
            Tool(
                name="suggest_posting_time",
                func=self._suggest_posting_time,
                description="Suggest optimal posting times based on audience"
            ),
            Tool(
                name="hashtag_mix_optimizer",
                func=self._optimize_hashtag_mix,
                description="Optimize hashtag mix for reach and engagement"
            ),
            Tool(
                name="content_score_predictor",
                func=self._predict_content_score,
                description="Predict engagement score for content"
            )
        ]
        
        optimizer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are ContentOptimizer, an AI agent focused on maximizing TikTok content performance.
Your capabilities:
1. Optimize captions for discoverability and engagement
2. Suggest optimal posting times based on audience analytics
3. Recommend relevant hashtags with the right mix of popular and niche tags
4. Analyze video length optimization for specific content types
5. Provide A/B testing recommendations
6. Suggest content improvements based on performance data

Base recommendations on platform algorithms, audience behavior, and proven strategies."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        optimizer_agent = create_openai_tools_agent(self.openai_llm, optimizer_tools, optimizer_prompt)
        self.agents["ContentOptimizer"] = AgentExecutor(
            agent=optimizer_agent,
            tools=optimizer_tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        # ========================================
        # Agent 4: StrategyAdvisor - Strategy Planning
        # ========================================
        strategy_tools = [
            Tool(
                name="competitor_analysis",
                func=self._analyze_competitors,
                description="Analyze competitor content and strategies"
            ),
            Tool(
                name="content_calendar_generator",
                func=self._generate_content_calendar,
                description="Generate a content calendar based on strategy"
            ),
            Tool(
                name="growth_projector",
                func=self._project_growth,
                description="Project growth based on posting strategy"
            ),
            Tool(
                name="niche_opportunity_finder",
                func=self._find_niche_opportunities,
                description="Find untapped opportunities in a niche"
            )
        ]
        
        strategy_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are StrategyAdvisor, a strategic AI agent for TikTok growth.
Your expertise covers:
1. Developing long-term content calendars
2. Creating niche positioning strategies
3. Analyzing competitor content and strategies
4. Building audience engagement plans
5. Setting and tracking growth KPIs
6. Recommending collaboration opportunities
7. Planning content series and campaigns

Provide strategic, actionable advice that balances short-term wins with long-term growth."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        strategy_agent = create_openai_tools_agent(self.openai_llm, strategy_tools, strategy_prompt)
        self.agents["StrategyAdvisor"] = AgentExecutor(
            agent=strategy_agent,
            tools=strategy_tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    # ========================================
    # Tool Implementation Methods
    # ========================================
    
    async def _get_trending_hashtags(self, niche: str = "general") -> str:
        """Fetch trending hashtags from trend service"""
        # In production, this would call the trend-analyzer service
        return json.dumps({
            "hashtags": [
                {"tag": "#fyp", "views": "500B+", "growth": "stable"},
                {"tag": "#viral", "views": "200B+", "growth": "stable"},
                {"tag": f"#{niche}", "views": "10B+", "growth": "rising"},
                {"tag": "#trending", "views": "50B+", "growth": "stable"},
            ],
            "niche_specific": [
                {"tag": f"#{niche}tips", "views": "1B+", "growth": "rising"},
                {"tag": f"#{niche}hack", "views": "500M+", "growth": "rising"},
            ]
        })
    
    async def _get_trending_sounds(self, category: str = "all") -> str:
        """Fetch trending sounds"""
        return json.dumps({
            "sounds": [
                {"title": "Original Sound - Creator", "uses": "1.2M", "growth": "viral"},
                {"title": "Trending Beat", "uses": "800K", "growth": "rising"},
            ]
        })
    
    async def _analyze_viral_patterns(self, niche: str) -> str:
        """Analyze patterns in viral content"""
        return json.dumps({
            "patterns": [
                "Strong hook in first 2 seconds",
                "Pattern interrupts every 3-5 seconds",
                "Text overlays for silent viewing",
                "Clear CTA at end",
                "Trending sound usage"
            ],
            "avg_duration": "45 seconds",
            "best_posting_times": ["7-9 AM", "12-2 PM", "7-10 PM"]
        })
    
    async def _predict_trend_longevity(self, trend: str) -> str:
        """Predict trend duration"""
        return json.dumps({
            "trend": trend,
            "predicted_peak": "3-5 days",
            "decline_start": "7-10 days",
            "recommendation": "Act within 48 hours for maximum impact"
        })
    
    async def _generate_hook_variations(self, topic: str) -> str:
        """Generate hook variations"""
        hooks = [
            f"Stop scrolling if you want to know about {topic}...",
            f"Nobody talks about this {topic} secret...",
            f"I learned this {topic} trick the hard way...",
            f"POV: You finally understand {topic}...",
            f"This {topic} hack changed everything..."
        ]
        return json.dumps({"hooks": hooks})
    
    async def _get_script_templates(self, format_type: str) -> str:
        """Get script templates"""
        templates = {
            "educational": "Hook → Problem → Solution → Proof → CTA",
            "storytelling": "Hook → Setup → Conflict → Resolution → CTA",
            "list": "Hook → Item 1 → Item 2 → Item 3 → Bonus → CTA",
            "tutorial": "Hook → Step 1 → Step 2 → Step 3 → Result → CTA"
        }
        return json.dumps({"templates": templates})
    
    async def _analyze_pacing(self, duration: int) -> str:
        """Analyze optimal pacing"""
        return json.dumps({
            "duration": duration,
            "hook_duration": min(3, duration * 0.1),
            "main_content_duration": duration * 0.8,
            "cta_duration": duration * 0.1,
            "pattern_interrupt_frequency": "every 5-7 seconds"
        })
    
    async def _analyze_caption_seo(self, caption: str) -> str:
        """Analyze caption SEO"""
        return json.dumps({
            "length_score": 0.8 if len(caption) < 150 else 0.6,
            "keyword_presence": True,
            "emoji_usage": "optimal" if "✨" in caption or "🔥" in caption else "add_emojis",
            "cta_present": "follow" in caption.lower() or "comment" in caption.lower()
        })
    
    async def _suggest_posting_time(self, timezone: str = "UTC") -> str:
        """Suggest posting times"""
        return json.dumps({
            "best_times": [
                {"time": "07:00", "engagement_multiplier": 1.2},
                {"time": "12:00", "engagement_multiplier": 1.3},
                {"time": "19:00", "engagement_multiplier": 1.5},
                {"time": "21:00", "engagement_multiplier": 1.4}
            ],
            "timezone": timezone
        })
    
    async def _optimize_hashtag_mix(self, hashtags: List[str]) -> str:
        """Optimize hashtag mix"""
        return json.dumps({
            "optimized_mix": {
                "broad": ["#fyp", "#viral"],
                "medium": ["#trending", "#foryou"],
                "niche": hashtags[:3] if hashtags else ["#nichcontent"]
            },
            "recommendation": "Use 30% broad, 40% medium, 30% niche hashtags"
        })
    
    async def _predict_content_score(self, content_data: str) -> str:
        """Predict content performance score"""
        return json.dumps({
            "predicted_score": 7.5,
            "factors": {
                "hook_strength": 8,
                "content_value": 7,
                "trend_alignment": 8,
                "cta_clarity": 7
            }
        })
    
    async def _analyze_competitors(self, niche: str) -> str:
        """Analyze competitors"""
        return json.dumps({
            "top_creators": [
                {"username": "@creator1", "followers": "1M", "avg_engagement": "5%"},
                {"username": "@creator2", "followers": "500K", "avg_engagement": "8%"}
            ],
            "content_gaps": ["Tutorial content", "Behind the scenes"],
            "successful_formats": ["Day in life", "Quick tips", "Storytelling"]
        })
    
    async def _generate_content_calendar(self, days: int) -> str:
        """Generate content calendar"""
        calendar = []
        content_types = ["Educational", "Entertainment", "Trending", "Personal", "Value"]
        for i in range(days):
            calendar.append({
                "day": i + 1,
                "content_type": content_types[i % len(content_types)],
                "posting_time": "19:00"
            })
        return json.dumps({"calendar": calendar[:7]})
    
    async def _project_growth(self, strategy: str) -> str:
        """Project growth"""
        return json.dumps({
            "30_day_projection": {
                "follower_growth": "+15-25%",
                "avg_views": "+30%",
                "engagement_rate": "+10%"
            },
            "assumptions": "Consistent daily posting with optimized content"
        })
    
    async def _find_niche_opportunities(self, niche: str) -> str:
        """Find niche opportunities"""
        return json.dumps({
            "opportunities": [
                f"Micro-niche: {niche} for beginners",
                f"Untapped format: {niche} ASMR",
                f"Cross-niche: {niche} meets productivity"
            ],
            "competition_level": "medium",
            "growth_potential": "high"
        })
    
    # ========================================
    # Agent Execution Methods
    # ========================================
    
    async def generate_script(self, request: GenerateScriptRequest) -> ScriptResponse:
        """Generate a TikTok script using AI agents or custom system prompt"""

        scenes = None
        full_script = ""

        # Check if we have a custom system prompt (from Creator Profile)
        if request.system_prompt and request.format == "structured":
            # Use direct LLM call with custom system prompt for structured output
            try:
                messages = [
                    SystemMessage(content=request.system_prompt),
                    HumanMessage(content=f"Create the script now for topic: {request.topic}")
                ]

                # Use Claude for better structured output
                result = await asyncio.to_thread(
                    self.anthropic_llm.invoke,
                    messages
                )

                response_text = result.content if hasattr(result, 'content') else str(result)

                # Try to parse JSON scenes from the response
                scenes = self._parse_structured_scenes(response_text, request)

                if scenes:
                    full_script = "\n\n".join([
                        f"[{s.time}]\nVisual: {s.visual}\nAudio: {s.audio}"
                        for s in scenes
                    ])
                else:
                    full_script = response_text

            except Exception as e:
                print(f"Error with custom prompt, falling back to agents: {e}")
                # Fall back to agent-based generation
                scenes = None

        # Fallback: Use agent-based generation
        if not scenes:
            # Step 1: Get trend insights from TrendScout
            trend_input = f"Analyze current trends for {request.niche or request.topic} content targeting {request.target_audience or 'general audience'}"
            trend_result = await asyncio.to_thread(
                self.agents["TrendScout"].invoke,
                {"input": trend_input, "chat_history": []}
            )

            # Build the prompt - use custom system prompt context if available
            profile_context = ""
            if request.profile:
                profile_context = f"""
Creator Profile:
- Brand: {request.profile.brandName or 'N/A'}
- Niche: {request.profile.niche or request.niche or 'general'}
- Tone: {request.profile.tone or request.tone}
- Hook Style: {request.profile.hookStyle or 'attention-grabbing'}
- Audience Level: {request.profile.audienceLevel or 'intermediate'}
- CTA Style: {request.profile.ctaStyle or 'Follow for more!'}
"""

            # Step 2: Generate structured script
            script_input = f"""Create a {request.duration_seconds}-second TikTok script about: {request.topic}

{profile_context}
Trending Elements: {trend_result.get('output', 'current trends')}

IMPORTANT: Return the script as a JSON array with this exact format:
[
  {{"time": "0:00-0:05", "visual": "Detailed visual description for image generation", "audio": "Exact voiceover text"}},
  {{"time": "0:05-0:15", "visual": "Next scene visual description", "audio": "Next voiceover text"}},
  ...
]

Requirements for each scene:
1. "visual" must be a detailed description that can be used to generate an image (e.g., "Person holding a red STOP sign with determined expression" not just "Hook visual")
2. "audio" is the exact voiceover/script text to be spoken
3. First scene (0:00-0:05) must be an attention-grabbing hook
4. Last scene should include a clear CTA
5. Include 4-6 scenes total depending on duration

Make visuals specific and descriptive for AI image generation."""

            script_result = await asyncio.to_thread(
                self.agents["ScriptGenius"].invoke,
                {"input": script_input, "chat_history": []}
            )

            response_text = script_result.get('output', '')
            scenes = self._parse_structured_scenes(response_text, request)

            if scenes:
                full_script = "\n\n".join([
                    f"[{s.time}]\nVisual: {s.visual}\nAudio: {s.audio}"
                    for s in scenes
                ])
            else:
                full_script = response_text

        return ScriptResponse(
            script_id=uuid4(),
            hook=self._extract_hook(full_script),
            main_content=self._extract_main_content(full_script),
            cta=self._extract_cta(full_script),
            full_script=full_script,
            duration_estimate_seconds=request.duration_seconds,
            suggested_visuals=self._suggest_visuals(full_script),
            trending_elements=request.specific_trends or [],
            engagement_score=self._calculate_engagement_score(full_script),
            scenes=scenes
        )

    def _parse_structured_scenes(self, response_text: str, request: GenerateScriptRequest) -> Optional[List[SceneData]]:
        """Parse JSON scenes from LLM response"""
        import re

        try:
            # Try to find JSON array in the response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                scenes_json = json.loads(json_match.group())
                scenes = []
                for scene in scenes_json:
                    visual_raw = scene.get('visual', '')
                    audio_raw = scene.get('audio', '')

                    # Normalize visual to string (handle dict case)
                    if isinstance(visual_raw, dict):
                        visual = visual_raw.get('image', '') or visual_raw.get('description', '') or visual_raw.get('visual', '') or str(visual_raw)
                    else:
                        visual = str(visual_raw) if visual_raw else ''
                        # Handle string that looks like a dict: "{'image': 'description'}"
                        if visual.startswith("{'image':") or visual.startswith('{"image":'):
                            try:
                                import ast
                                parsed = ast.literal_eval(visual.replace('null', 'None'))
                                if isinstance(parsed, dict):
                                    visual = parsed.get('image', '') or parsed.get('description', '') or visual
                            except:
                                pass

                    # Normalize audio to string (handle dict case)
                    if isinstance(audio_raw, dict):
                        audio = audio_raw.get('text', '') or audio_raw.get('audio', '') or audio_raw.get('voiceover', '') or str(audio_raw)
                    else:
                        audio = str(audio_raw) if audio_raw else ''

                    visual_type = self._classify_visual_type(visual)
                    visual_prompt = self._generate_image_prompt(visual, request)

                    scenes.append(SceneData(
                        time=scene.get('time', '0:00-0:05'),
                        visual=visual,
                        audio=audio,
                        visualType=visual_type,
                        visualPrompt=visual_prompt
                    ))
                return scenes if scenes else None
        except (json.JSONDecodeError, Exception) as e:
            print(f"Failed to parse scenes JSON: {e}")
        return None

    def _classify_visual_type(self, visual) -> str:
        """Classify visual description into type: image, diagram, video, text-overlay"""
        # Handle case where visual might be a dict or None
        if not visual:
            return 'image'
        if isinstance(visual, dict):
            visual = str(visual.get('description', '') or visual.get('visual', '') or '')
        if not isinstance(visual, str):
            visual = str(visual)
        lower_visual = visual.lower()

        diagram_keywords = ['diagram', 'flowchart', 'architecture', 'pipeline', 'workflow',
                           'schema', 'hierarchy', 'infographic', 'chart showing', 'process flow']
        video_keywords = ['video of', 'footage of', 'clip of', 'b-roll', 'screen recording',
                         'timelapse', 'demo video', 'walkthrough']
        text_keywords = ['text overlay only', 'just text', 'title card', 'text animation']

        if any(kw in lower_visual for kw in diagram_keywords):
            return 'diagram'
        if any(kw in lower_visual for kw in video_keywords):
            return 'video'
        if any(kw in lower_visual for kw in text_keywords):
            return 'text-overlay'
        return 'image'

    def _generate_image_prompt(self, visual, request: GenerateScriptRequest) -> str:
        """Generate an optimized prompt for DALL-E image generation"""
        # Handle case where visual might be a dict or None
        if not visual:
            return "Professional social media style image. High quality, vibrant colors, 9:16 vertical format."
        if isinstance(visual, dict):
            visual = str(visual.get('description', '') or visual.get('visual', '') or '')
        if not isinstance(visual, str):
            visual = str(visual)

        # Clean the visual description
        prompt = visual

        # Remove meta-instructions
        prompt = re.sub(r'(Hook:|CTA:|Visual:|Scene:)\s*', '', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'Text overlay[:\s]*["\'].*?["\']', '', prompt, flags=re.IGNORECASE)

        # Get the core visual description
        prompt = prompt.strip()

        # Add context from profile if available
        style_hints = []
        if request.profile:
            if request.profile.niche:
                style_hints.append(f"{request.profile.niche} theme")
            if request.profile.tone == 'professional':
                style_hints.append("professional corporate style")
            elif request.profile.tone == 'casual':
                style_hints.append("casual friendly style")
            elif request.profile.tone == 'motivational':
                style_hints.append("inspiring empowering style")

        # Build final prompt for DALL-E
        final_prompt = f"{prompt}"
        if style_hints:
            final_prompt += f". Style: {', '.join(style_hints)}"
        final_prompt += ". High quality, professional photography, social media style, vibrant colors, 9:16 vertical format, good lighting, sharp focus."

        return final_prompt
    
    async def generate_caption(self, request: GenerateCaptionRequest) -> CaptionResponse:
        """Generate optimized caption with hashtags"""
        
        input_text = f"""Create an engaging TikTok caption for this script:
{request.script}

Requirements:
- Maximum {request.max_length} characters (excluding hashtags)
- {'Include relevant hashtags' if request.include_hashtags else 'No hashtags'}
- {'Include a call-to-action' if request.include_cta else 'No CTA needed'}
- Use emojis strategically
- Make it scroll-stopping"""
        
        result = await asyncio.to_thread(
            self.agents["ContentOptimizer"].invoke,
            {"input": input_text, "chat_history": []}
        )
        
        caption_text = result.get('output', '')
        hashtags = self._extract_hashtags(caption_text) if request.include_hashtags else []
        clean_caption = self._remove_hashtags(caption_text)
        
        return CaptionResponse(
            caption=clean_caption,
            hashtags=hashtags,
            character_count=len(clean_caption),
            estimated_reach_multiplier=1.2 + (len(hashtags) * 0.05)
        )
    
    async def chat_with_agent(self, request: AIAgentChatRequest) -> ChatResponse:
        """Have a conversation with a specific AI agent"""
        
        if request.agent_name not in self.agents:
            raise HTTPException(status_code=400, detail=f"Agent {request.agent_name} not found")
        
        agent = self.agents[request.agent_name]
        
        # Build context-aware input
        context_str = json.dumps(request.context) if request.context else ""
        full_input = f"{request.message}\n\nContext: {context_str}" if context_str else request.message
        
        result = await asyncio.to_thread(
            agent.invoke,
            {"input": full_input, "chat_history": []}
        )
        
        return ChatResponse(
            conversation_id=request.conversation_id or uuid4(),
            agent_name=request.agent_name,
            response=result.get('output', ''),
            suggested_actions=self._extract_suggested_actions(result.get('output', '')),
            tokens_used=0  # Would be tracked in production
        )
    
    # ========================================
    # Helper Methods
    # ========================================
    
    def _extract_hook(self, script: str) -> str:
        lines = script.split('\n')
        for line in lines:
            if 'hook' in line.lower() or lines.index(line) == 0:
                return line.strip()
        return lines[0] if lines else ""
    
    def _extract_main_content(self, script: str) -> str:
        lines = script.split('\n')
        if len(lines) > 2:
            return '\n'.join(lines[1:-1])
        return script
    
    def _extract_cta(self, script: str) -> str:
        lines = script.split('\n')
        for line in reversed(lines):
            if any(word in line.lower() for word in ['follow', 'like', 'comment', 'share', 'subscribe']):
                return line.strip()
        return lines[-1] if lines else ""
    
    def _suggest_visuals(self, script: str) -> List[str]:
        visuals = []
        if 'tutorial' in script.lower():
            visuals.append("Screen recording or step-by-step demonstration")
        if 'story' in script.lower():
            visuals.append("Personal footage or reenactment")
        visuals.append("Text overlays for key points")
        visuals.append("Trending transition effects")
        return visuals
    
    def _calculate_engagement_score(self, script: str) -> float:
        score = 5.0
        if len(script) > 100:
            score += 1.0
        if any(word in script.lower() for word in ['you', 'your']):
            score += 0.5
        if '?' in script:
            score += 0.5
        if any(word in script.lower() for word in ['secret', 'hack', 'trick', 'mistake']):
            score += 1.0
        return min(score, 10.0)
    
    def _extract_hashtags(self, text: str) -> List[str]:
        import re
        return re.findall(r'#\w+', text)
    
    def _remove_hashtags(self, text: str) -> str:
        import re
        return re.sub(r'#\w+\s*', '', text).strip()
    
    def _extract_suggested_actions(self, response: str) -> List[Dict[str, Any]]:
        actions = []
        if 'script' in response.lower():
            actions.append({"type": "generate_script", "label": "Generate Script"})
        if 'hashtag' in response.lower():
            actions.append({"type": "generate_hashtags", "label": "Get Hashtags"})
        if 'post' in response.lower() or 'schedule' in response.lower():
            actions.append({"type": "schedule_post", "label": "Schedule Post"})
        return actions


# ========================================
# Initialize AI Agents
# ========================================

ai_agents = TikTokAIAgents()

# ========================================
# API Endpoints
# ========================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "content-generator", "agents": list(ai_agents.agents.keys())}

@app.post("/api/v1/generate/script", response_model=ScriptResponse)
async def generate_script(request: GenerateScriptRequest):
    """Generate a TikTok video script using AI agents"""
    try:
        return await ai_agents.generate_script(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/caption", response_model=CaptionResponse)
async def generate_caption(request: GenerateCaptionRequest):
    """Generate an optimized caption with hashtags"""
    try:
        return await ai_agents.generate_caption(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/hashtags", response_model=HashtagResponse)
async def generate_hashtags(request: GenerateHashtagsRequest):
    """Generate optimized hashtag recommendations"""
    try:
        input_text = f"""Recommend {request.max_hashtags} hashtags for: {request.content_description}
        Niche: {request.niche or 'general'}
        
        Provide a mix of:
        - High volume hashtags (broad reach)
        - Medium volume hashtags (targeted)
        - Niche hashtags (specific audience)"""
        
        result = await asyncio.to_thread(
            ai_agents.agents["ContentOptimizer"].invoke,
            {"input": input_text, "chat_history": []}
        )
        
        # Parse hashtags from response
        hashtags = ai_agents._extract_hashtags(result.get('output', ''))
        
        return HashtagResponse(
            hashtags=[{"tag": h, "type": "recommended"} for h in hashtags],
            total_reach_potential=1000000 * len(hashtags),
            mix_analysis={"broad": 3, "medium": 4, "niche": 3}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/optimize", response_model=OptimizationResponse)
async def optimize_content(request: OptimizeContentRequest):
    """Optimize existing content for better performance"""
    try:
        input_text = f"""Optimize the following TikTok content for {', '.join(request.optimization_goals)}:
        
        Script: {request.script or 'N/A'}
        Caption: {request.caption or 'N/A'}
        Hashtags: {', '.join(request.hashtags) if request.hashtags else 'N/A'}
        
        Provide specific improvements for each element."""
        
        result = await asyncio.to_thread(
            ai_agents.agents["ContentOptimizer"].invoke,
            {"input": input_text, "chat_history": []}
        )
        
        return OptimizationResponse(
            optimized_script=request.script,
            optimized_caption=request.caption,
            optimized_hashtags=request.hashtags,
            improvements=[result.get('output', '')],
            expected_performance_lift=1.25
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/strategy", response_model=StrategyResponse)
async def create_content_strategy(request: ContentStrategyRequest):
    """Create a comprehensive content strategy"""
    try:
        input_text = f"""Create a {request.time_horizon_days}-day content strategy for:
        
        Niche: {request.niche}
        Goals: {', '.join(request.goals)}
        Posting Frequency: {request.posting_frequency}
        
        Include:
        1. Content pillars (3-5 main themes)
        2. Weekly posting schedule
        3. 10 specific content ideas
        4. Growth projections
        5. Key metrics to track"""
        
        result = await asyncio.to_thread(
            ai_agents.agents["StrategyAdvisor"].invoke,
            {"input": input_text, "chat_history": []}
        )
        
        return StrategyResponse(
            strategy_id=uuid4(),
            content_pillars=[
                {"name": "Educational", "percentage": 40},
                {"name": "Entertainment", "percentage": 30},
                {"name": "Trending", "percentage": 20},
                {"name": "Personal", "percentage": 10}
            ],
            posting_schedule=[
                {"day": "Monday", "time": "19:00", "content_type": "Educational"},
                {"day": "Wednesday", "time": "12:00", "content_type": "Entertainment"},
                {"day": "Friday", "time": "19:00", "content_type": "Trending"}
            ],
            content_ideas=[{"idea": line.strip()} for line in result.get('output', '').split('\n') if line.strip()][:10],
            growth_projections={
                "followers_30d": "+500-1000",
                "avg_views": "+50%",
                "engagement_rate": "5-8%"
            },
            key_metrics_to_track=["Views", "Watch time", "Shares", "Profile visits", "Follower growth"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_agent(request: AIAgentChatRequest):
    """Have a conversation with a specific AI agent"""
    try:
        return await ai_agents.chat_with_agent(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents")
async def list_agents():
    """List all available AI agents"""
    return {
        "agents": [
            {
                "name": "TrendScout",
                "description": "Analyzes TikTok trends and identifies viral patterns",
                "capabilities": ["trend_detection", "pattern_analysis", "viral_prediction"]
            },
            {
                "name": "ScriptGenius",
                "description": "Generates engaging TikTok video scripts with viral hooks",
                "capabilities": ["script_writing", "hook_creation", "storytelling"]
            },
            {
                "name": "ContentOptimizer",
                "description": "Optimizes content for maximum engagement and reach",
                "capabilities": ["caption_optimization", "hashtag_strategy", "timing_analysis"]
            },
            {
                "name": "StrategyAdvisor",
                "description": "Develops comprehensive content strategies for growth",
                "capabilities": ["strategy_planning", "competitor_analysis", "growth_projection"]
            }
        ]
    }

# ========================================
# Run Application
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
