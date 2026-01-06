"""
Viralify Media Generator Service
Handles AI-powered generation of images, voiceovers, and video compositions
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import uuid
import asyncio
import httpx
import os
import base64
from pathlib import Path
from contextlib import asynccontextmanager

# Create directories for storing generated media
MEDIA_DIR = Path("/tmp/viralify_media")
AUDIO_DIR = MEDIA_DIR / "audio"
IMAGES_DIR = MEDIA_DIR / "images"

for dir_path in [MEDIA_DIR, AUDIO_DIR, IMAGES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========================================
# Configuration
# ========================================

class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://tiktok_user:tiktok_secure_pass_2024@localhost:5432/tiktok_platform")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://:redis_secure_2024@localhost:6379/4")
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://tiktok:rabbitmq_secure_2024@localhost:5672/")

    # AI Providers
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Service URL for generating full URLs
    SERVICE_BASE_URL: str = os.getenv("SERVICE_BASE_URL", "http://localhost:8004")
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    PEXELS_API_KEY: str = os.getenv("PEXELS_API_KEY", "")
    UNSPLASH_API_KEY: str = os.getenv("UNSPLASH_API_KEY", "")
    PIXABAY_API_KEY: str = os.getenv("PIXABAY_API_KEY", "")

    # Storage
    CLOUDINARY_URL: str = os.getenv("CLOUDINARY_URL", "")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "viralify-assets")

    # Demo mode
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() == "true"

settings = Settings()

# ========================================
# AI Video Generator (lazy loaded)
# ========================================
_video_generator = None

def get_video_generator():
    global _video_generator
    if _video_generator is None:
        from services.video_generator import AIVideoGenerator
        _video_generator = AIVideoGenerator(
            openai_api_key=settings.OPENAI_API_KEY,
            elevenlabs_api_key=settings.ELEVENLABS_API_KEY,
            pexels_api_key=settings.PEXELS_API_KEY,
            unsplash_api_key=settings.UNSPLASH_API_KEY,
            pixabay_api_key=settings.PIXABAY_API_KEY
        )
    return _video_generator

# ========================================
# Database (simplified for now)
# ========================================

# In production, use SQLAlchemy with async support
jobs_db: Dict[str, Dict] = {}
assets_db: Dict[str, Dict] = {}

# ========================================
# Enums and Models
# ========================================

class JobType(str, Enum):
    IMAGE = "image"
    THUMBNAIL = "thumbnail"
    VOICEOVER = "voiceover"
    VIDEO = "video"
    ARTICLE = "article"
    VIDEO_COMPOSITION = "video_composition"
    DIAGRAM = "diagram"

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ImageStyle(str, Enum):
    REALISTIC = "realistic"
    ANIME = "anime"
    ILLUSTRATION = "illustration"
    THREE_D = "3d"
    MINIMALIST = "minimalist"
    VIBRANT = "vibrant"

class VoiceProvider(str, Enum):
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"

# ========================================
# Request/Response Models
# ========================================

class GenerateImageRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=1000)
    style: ImageStyle = ImageStyle.REALISTIC
    aspect_ratio: str = Field(default="9:16", pattern=r"^\d+:\d+$")
    quality: str = Field(default="standard", pattern=r"^(standard|hd)$")
    reference_image_url: Optional[str] = None

class GenerateThumbnailRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    style: ImageStyle = ImageStyle.VIBRANT
    include_text: bool = True
    brand_colors: Optional[List[str]] = None

class GenerateDiagramRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=1000)
    style: str = Field(default="modern")  # modern, minimal, technical, colorful
    format: str = Field(default="png")  # png, svg
    aspect_ratio: str = Field(default="9:16")

class GenerateVoiceoverRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000)
    provider: VoiceProvider = VoiceProvider.OPENAI
    voice_id: str = Field(default="alloy")  # OpenAI: alloy, echo, fable, onyx, nova, shimmer
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    emotion: Optional[str] = None  # ElevenLabs only

class GenerateArticleRequest(BaseModel):
    topic: str = Field(..., min_length=10, max_length=500)
    content_type: str = Field(default="blog_post")
    word_count: int = Field(default=800, ge=200, le=3000)
    tone: str = Field(default="conversational")
    source_script: Optional[str] = None
    seo_keywords: Optional[List[str]] = None

class SearchStockVideoRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    duration_max: int = Field(default=30, ge=5, le=120)
    orientation: str = Field(default="portrait")
    per_page: int = Field(default=10, ge=1, le=50)

class ComposeVideoRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    template_id: Optional[str] = None
    script: Optional[str] = None
    voiceover_job_id: Optional[str] = None
    image_assets: Optional[List[str]] = None  # Asset IDs
    stock_video_urls: Optional[List[str]] = None
    music_url: Optional[str] = None
    text_overlays: Optional[List[Dict[str, Any]]] = None
    duration_seconds: int = Field(default=30, ge=5, le=180)

class SimpleComposeVideoRequest(BaseModel):
    video_urls: List[str] = Field(..., min_items=1, max_items=20)
    output_format: str = Field(default="9:16", pattern=r"^(9:16|16:9|1:1)$")
    quality: str = Field(default="1080p", pattern=r"^(720p|1080p|4k)$")

class JobResponse(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus
    created_at: datetime
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    progress_percent: int = 0

class AssetResponse(BaseModel):
    id: str
    asset_type: str
    storage_url: str
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    dimensions: Optional[Dict[str, int]] = None
    prompt: Optional[str] = None
    created_at: datetime

class StockVideoResult(BaseModel):
    id: str
    url: str
    preview_url: str
    duration: int
    width: int
    height: int
    user: str

# ========================================
# AI Provider Integrations
# ========================================

async def generate_image_dalle(prompt: str, style: ImageStyle, aspect_ratio: str, quality: str) -> Dict:
    """Generate image using OpenAI DALL-E 3"""
    if settings.DEMO_MODE:
        await asyncio.sleep(2)  # Simulate processing
        return {
            "url": f"https://picsum.photos/1080/1920?random={uuid.uuid4().hex[:8]}",
            "revised_prompt": f"[DEMO] {prompt}",
            "provider": "dalle3"
        }

    # Map aspect ratio to DALL-E sizes
    size_map = {
        "1:1": "1024x1024",
        "9:16": "1024x1792",
        "16:9": "1792x1024"
    }
    size = size_map.get(aspect_ratio, "1024x1792")

    # Add style to prompt
    style_prompts = {
        ImageStyle.REALISTIC: "photorealistic, high detail",
        ImageStyle.ANIME: "anime style, vibrant colors",
        ImageStyle.ILLUSTRATION: "digital illustration, artistic",
        ImageStyle.THREE_D: "3D rendered, CGI quality",
        ImageStyle.MINIMALIST: "minimalist design, clean lines",
        ImageStyle.VIBRANT: "vibrant colors, eye-catching"
    }
    enhanced_prompt = f"{prompt}, {style_prompts.get(style, '')}"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "dall-e-3",
                "prompt": enhanced_prompt,
                "n": 1,
                "size": size,
                "quality": quality
            },
            timeout=60.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"DALL-E API error: {response.text}")

        data = response.json()
        return {
            "url": data["data"][0]["url"],
            "revised_prompt": data["data"][0].get("revised_prompt", prompt),
            "provider": "dalle3"
        }

async def generate_diagram_dalle(description: str, style: str, aspect_ratio: str) -> Dict:
    """Generate a diagram-style image using DALL-E 3"""
    if settings.DEMO_MODE:
        await asyncio.sleep(2)
        return {
            "url": f"https://placehold.co/1080x1920/1e3a5f/ffffff?text=Diagram",
            "diagram_url": f"https://placehold.co/1080x1920/1e3a5f/ffffff?text=Diagram",
            "provider": "dalle3"
        }

    # Map aspect ratio to DALL-E size
    size_mapping = {
        "9:16": "1024x1792",
        "16:9": "1792x1024",
        "1:1": "1024x1024"
    }
    size = size_mapping.get(aspect_ratio, "1024x1792")

    # Style-specific prompt enhancements
    style_prompts = {
        "modern": "modern, clean design, flat design, professional infographic style",
        "minimal": "minimalist, simple shapes, clean lines, lots of white space",
        "technical": "technical diagram, blueprint style, detailed annotations, engineering style",
        "colorful": "colorful, vibrant, engaging, educational poster style"
    }
    style_enhancement = style_prompts.get(style, style_prompts["modern"])

    # Create a diagram-focused prompt
    diagram_prompt = f"""Create a clear, professional diagram or infographic that visualizes: {description}

Style requirements:
- {style_enhancement}
- Clear visual hierarchy
- Easy to understand at a glance
- Suitable for social media/TikTok
- Use icons, arrows, and visual elements to explain the concept
- Include labeled components
- Professional quality, vector-like appearance"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "dall-e-3",
                "prompt": diagram_prompt,
                "n": 1,
                "size": size,
                "quality": "hd"
            },
            timeout=90.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"DALL-E API error: {response.text}")

        data = response.json()
        url = data["data"][0]["url"]
        return {
            "url": url,
            "diagram_url": url,
            "revised_prompt": data["data"][0].get("revised_prompt", description),
            "provider": "dalle3"
        }

async def generate_voiceover_openai(text: str, voice_id: str, speed: float) -> Dict:
    """Generate voiceover using OpenAI TTS"""
    if settings.DEMO_MODE:
        await asyncio.sleep(1.5)
        return {
            "url": "https://example.com/demo-audio.mp3",
            "duration_seconds": len(text) / 15,  # Rough estimate
            "provider": "openai_tts"
        }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "tts-1-hd",
                "input": text,
                "voice": voice_id,
                "speed": speed
            },
            timeout=120.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"OpenAI TTS error: {response.text}")

        # Save audio to local file
        audio_content = response.content
        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.mp3"
        audio_path = AUDIO_DIR / audio_filename

        with open(audio_path, "wb") as f:
            f.write(audio_content)

        # Return full URL that points to our static file endpoint
        full_url = f"{settings.SERVICE_BASE_URL}/api/v1/media/audio/{audio_id}"
        return {
            "url": full_url,
            "audio_id": audio_id,
            "duration_seconds": len(text) / 15,  # Rough estimate
            "provider": "openai_tts"
        }

async def generate_voiceover_elevenlabs(text: str, voice_id: str, emotion: Optional[str]) -> Dict:
    """Generate voiceover using ElevenLabs"""
    if settings.DEMO_MODE:
        await asyncio.sleep(2)
        return {
            "url": "https://example.com/demo-audio-elevenlabs.mp3",
            "duration_seconds": len(text) / 12,
            "provider": "elevenlabs"
        }

    # ElevenLabs has natural-sounding voices
    default_voice_id = voice_id or "21m00Tcm4TlvDq8ikWAM"  # Rachel voice

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{default_voice_id}",
            headers={
                "xi-api-key": settings.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            },
            timeout=120.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"ElevenLabs error: {response.text}")

        # Save audio to local file
        audio_content = response.content
        audio_id = str(uuid.uuid4())
        audio_filename = f"{audio_id}.mp3"
        audio_path = AUDIO_DIR / audio_filename

        with open(audio_path, "wb") as f:
            f.write(audio_content)

        full_url = f"{settings.SERVICE_BASE_URL}/api/v1/media/audio/{audio_id}"
        return {
            "url": full_url,
            "audio_id": audio_id,
            "duration_seconds": len(text) / 12,
            "provider": "elevenlabs"
        }

async def search_pexels_videos(query: str, duration_max: int, orientation: str, per_page: int) -> List[StockVideoResult]:
    """Search for stock videos on Pexels"""
    if settings.DEMO_MODE:
        # Return mock results
        return [
            StockVideoResult(
                id=f"pexels-{i}",
                url=f"https://example.com/stock-video-{i}.mp4",
                preview_url=f"https://picsum.photos/seed/video{i}/400/711",
                duration=15 + (i * 5),
                width=1080,
                height=1920,
                user=f"Creator {i}"
            )
            for i in range(min(per_page, 5))
        ]

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": settings.PEXELS_API_KEY},
            params={
                "query": query,
                "orientation": orientation,
                "per_page": per_page,
                "size": "medium"
            },
            timeout=30.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Pexels error: {response.text}")

        data = response.json()
        results = []

        for video in data.get("videos", []):
            if video["duration"] <= duration_max:
                # Find the best quality video file
                video_file = next(
                    (f for f in video["video_files"] if f["quality"] == "hd"),
                    video["video_files"][0] if video["video_files"] else None
                )

                if video_file:
                    results.append(StockVideoResult(
                        id=str(video["id"]),
                        url=video_file["link"],
                        preview_url=video["image"],
                        duration=video["duration"],
                        width=video_file["width"],
                        height=video_file["height"],
                        user=video["user"]["name"]
                    ))

        return results

async def generate_article_ai(
    topic: str,
    content_type: str,
    word_count: int,
    tone: str,
    source_script: Optional[str],
    seo_keywords: Optional[List[str]]
) -> Dict:
    """Generate article content using GPT-4"""
    if settings.DEMO_MODE:
        await asyncio.sleep(3)
        return {
            "title": f"Demo Article: {topic}",
            "content": f"This is a demo article about {topic}. " * (word_count // 10),
            "word_count": word_count,
            "seo_keywords": seo_keywords or ["demo", "article"],
            "meta_description": f"Learn about {topic} in this comprehensive guide."
        }

    system_prompt = f"""You are an expert content writer specializing in {content_type} content.
Write in a {tone} tone. Target word count: {word_count} words.
{'Include these SEO keywords naturally: ' + ', '.join(seo_keywords) if seo_keywords else ''}
{'Use this script as the basis for the article: ' + source_script if source_script else ''}"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4-turbo-preview",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Write a {content_type} about: {topic}"}
                ],
                "temperature": 0.7,
                "max_tokens": word_count * 2
            },
            timeout=120.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {response.text}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        return {
            "title": topic,
            "content": content,
            "word_count": len(content.split()),
            "seo_keywords": seo_keywords or [],
            "meta_description": content[:160]
        }

# ========================================
# Job Processing
# ========================================

async def process_image_job(job_id: str, request: GenerateImageRequest, user_id: str):
    """Process image generation job"""
    jobs_db[job_id]["status"] = JobStatus.PROCESSING
    jobs_db[job_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        result = await generate_image_dalle(
            request.prompt,
            request.style,
            request.aspect_ratio,
            request.quality
        )

        # Create asset
        asset_id = str(uuid.uuid4())
        assets_db[asset_id] = {
            "id": asset_id,
            "user_id": user_id,
            "job_id": job_id,
            "asset_type": "image",
            "storage_url": result["url"],
            "prompt": request.prompt,
            "style_preset": request.style,
            "dimensions": {"width": 1080, "height": 1920},
            "created_at": datetime.utcnow().isoformat()
        }

        jobs_db[job_id]["status"] = JobStatus.COMPLETED
        jobs_db[job_id]["output_data"] = {
            "asset_id": asset_id,
            "url": result["url"],
            "revised_prompt": result.get("revised_prompt")
        }
        jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error_message"] = str(e)

async def process_voiceover_job(job_id: str, request: GenerateVoiceoverRequest, user_id: str):
    """Process voiceover generation job"""
    jobs_db[job_id]["status"] = JobStatus.PROCESSING

    try:
        if request.provider == VoiceProvider.ELEVENLABS:
            result = await generate_voiceover_elevenlabs(
                request.text,
                request.voice_id,
                request.emotion
            )
        else:
            result = await generate_voiceover_openai(
                request.text,
                request.voice_id,
                request.speed
            )

        # Create asset
        asset_id = str(uuid.uuid4())
        assets_db[asset_id] = {
            "id": asset_id,
            "user_id": user_id,
            "job_id": job_id,
            "asset_type": "audio",
            "storage_url": result["url"],
            "duration_seconds": result["duration_seconds"],
            "created_at": datetime.utcnow().isoformat()
        }

        jobs_db[job_id]["status"] = JobStatus.COMPLETED
        jobs_db[job_id]["output_data"] = {
            "asset_id": asset_id,
            "url": result["url"],
            "duration_seconds": result["duration_seconds"]
        }

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error_message"] = str(e)

async def process_diagram_job(job_id: str, request: GenerateDiagramRequest, user_id: str):
    """Process diagram generation job"""
    jobs_db[job_id]["status"] = JobStatus.PROCESSING

    try:
        result = await generate_diagram_dalle(
            request.description,
            request.style,
            request.aspect_ratio
        )

        # Create asset
        asset_id = str(uuid.uuid4())
        assets_db[asset_id] = {
            "id": asset_id,
            "user_id": user_id,
            "job_id": job_id,
            "asset_type": "diagram",
            "storage_url": result["url"],
            "description": request.description,
            "style": request.style,
            "dimensions": {"width": 1080, "height": 1920},
            "created_at": datetime.utcnow().isoformat()
        }

        jobs_db[job_id]["status"] = JobStatus.COMPLETED
        jobs_db[job_id]["output_data"] = {
            "asset_id": asset_id,
            "url": result["url"],
            "diagram_url": result["diagram_url"],
            "revised_prompt": result.get("revised_prompt")
        }
        jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error_message"] = str(e)

async def process_article_job(job_id: str, request: GenerateArticleRequest, user_id: str):
    """Process article generation job"""
    jobs_db[job_id]["status"] = JobStatus.PROCESSING

    try:
        result = await generate_article_ai(
            request.topic,
            request.content_type,
            request.word_count,
            request.tone,
            request.source_script,
            request.seo_keywords
        )

        jobs_db[job_id]["status"] = JobStatus.COMPLETED
        jobs_db[job_id]["output_data"] = result

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error_message"] = str(e)

async def process_video_composition_job(job_id: str, request: SimpleComposeVideoRequest, user_id: str):
    """Process video composition job - combines multiple video clips"""
    jobs_db[job_id]["status"] = JobStatus.PROCESSING
    jobs_db[job_id]["progress_percent"] = 10

    try:
        # In production, this would use FFmpeg to:
        # 1. Download all video clips
        # 2. Normalize formats/resolutions
        # 3. Concatenate clips
        # 4. Apply transitions
        # 5. Upload to storage

        # Simulate processing time based on number of clips
        total_clips = len(request.video_urls)
        for i, url in enumerate(request.video_urls):
            await asyncio.sleep(1)  # Simulate downloading/processing each clip
            progress = int(10 + (80 * (i + 1) / total_clips))
            jobs_db[job_id]["progress_percent"] = progress

        # For demo/MVP, return the first video URL as the "composed" output
        # In production, this would be a newly created video file
        composed_url = request.video_urls[0] if request.video_urls else None

        # Create asset
        asset_id = str(uuid.uuid4())
        assets_db[asset_id] = {
            "id": asset_id,
            "user_id": user_id,
            "job_id": job_id,
            "asset_type": "video",
            "storage_url": composed_url,
            "dimensions": {
                "width": 1080 if request.output_format == "9:16" else 1920,
                "height": 1920 if request.output_format == "9:16" else 1080
            },
            "source_clips": len(request.video_urls),
            "created_at": datetime.utcnow().isoformat()
        }

        jobs_db[job_id]["status"] = JobStatus.COMPLETED
        jobs_db[job_id]["progress_percent"] = 100
        jobs_db[job_id]["output_data"] = {
            "asset_id": asset_id,
            "url": composed_url,
            "format": request.output_format,
            "quality": request.quality,
            "clips_used": total_clips,
            "message": "Video composed successfully. In production, this would be a fully merged video file."
        }
        jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error_message"] = str(e)

# ========================================
# FastAPI Application
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Media Generator Service starting...")
    print(f"   Demo Mode: {settings.DEMO_MODE}")
    yield
    # Shutdown
    print("👋 Media Generator Service shutting down...")

app = FastAPI(
    title="Viralify Media Generator",
    description="AI-powered media generation service for images, voiceovers, and videos",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# API Endpoints
# ========================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "media-generator",
        "demo_mode": settings.DEMO_MODE,
        "timestamp": datetime.utcnow().isoformat()
    }

# Static file serving for generated media
@app.get("/api/v1/media/audio/{audio_id}")
async def serve_audio(audio_id: str):
    """Serve generated audio file"""
    audio_path = AUDIO_DIR / f"{audio_id}.mp3"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=f"{audio_id}.mp3"
    )

@app.get("/api/v1/media/files/{file_type}/{file_id}")
async def serve_media_file(file_type: str, file_id: str):
    """Serve generated media files (images, diagrams)"""
    if file_type == "audio":
        file_path = AUDIO_DIR / f"{file_id}.mp3"
        media_type = "audio/mpeg"
    elif file_type == "image":
        file_path = IMAGES_DIR / f"{file_id}.png"
        media_type = "image/png"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=str(file_path), media_type=media_type)

# Image Generation
@app.post("/api/v1/media/image", response_model=JobResponse)
async def generate_image(
    request: GenerateImageRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "demo-user"  # In production, get from auth token
):
    """Generate an AI image using DALL-E 3"""
    job_id = str(uuid.uuid4())

    jobs_db[job_id] = {
        "job_id": job_id,
        "job_type": JobType.IMAGE,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "input_data": request.model_dump(),
        "user_id": user_id
    }

    background_tasks.add_task(process_image_job, job_id, request, user_id)

    return JobResponse(**jobs_db[job_id])

@app.post("/api/v1/media/thumbnail", response_model=JobResponse)
async def generate_thumbnail(
    request: GenerateThumbnailRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "demo-user"
):
    """Generate a thumbnail image for video content"""
    # Convert thumbnail request to image request
    prompt = f"YouTube/TikTok thumbnail for: {request.title}. Eye-catching, bold text overlay ready, {request.style.value} style"

    image_request = GenerateImageRequest(
        prompt=prompt,
        style=request.style,
        aspect_ratio="9:16",
        quality="standard"
    )

    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {
        "job_id": job_id,
        "job_type": JobType.THUMBNAIL,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "input_data": request.model_dump(),
        "user_id": user_id
    }

    background_tasks.add_task(process_image_job, job_id, image_request, user_id)

    return JobResponse(**jobs_db[job_id])

# Diagram Generation
@app.post("/api/v1/media/diagram")
async def generate_diagram(
    request: GenerateDiagramRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "demo-user"
):
    """Generate a diagram/infographic image for technical content"""
    job_id = str(uuid.uuid4())

    jobs_db[job_id] = {
        "job_id": job_id,
        "job_type": JobType.DIAGRAM,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "input_data": request.model_dump(),
        "user_id": user_id
    }

    # Process synchronously for immediate response (diagrams are quick)
    try:
        result = await generate_diagram_dalle(
            request.description,
            request.style,
            request.aspect_ratio
        )

        jobs_db[job_id]["status"] = JobStatus.COMPLETED
        jobs_db[job_id]["output_data"] = {
            "url": result["url"],
            "diagram_url": result["diagram_url"],
            "revised_prompt": result.get("revised_prompt")
        }
        jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()

        # Return the URL directly for immediate use
        return {
            "job_id": job_id,
            "status": "completed",
            "url": result["url"],
            "diagram_url": result["diagram_url"]
        }

    except Exception as e:
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error_message"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

# Voiceover Generation
@app.post("/api/v1/media/voiceover", response_model=JobResponse)
async def generate_voiceover(
    request: GenerateVoiceoverRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "demo-user"
):
    """Generate AI voiceover for video content"""
    job_id = str(uuid.uuid4())

    jobs_db[job_id] = {
        "job_id": job_id,
        "job_type": JobType.VOICEOVER,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "input_data": request.model_dump(),
        "user_id": user_id
    }

    background_tasks.add_task(process_voiceover_job, job_id, request, user_id)

    return JobResponse(**jobs_db[job_id])

@app.get("/api/v1/media/voiceover/voices")
async def get_available_voices():
    """Get list of available voices for voiceover generation"""
    return {
        "openai": [
            {"id": "alloy", "name": "Alloy", "description": "Neutral, balanced"},
            {"id": "echo", "name": "Echo", "description": "Warm, conversational"},
            {"id": "fable", "name": "Fable", "description": "Expressive, storytelling"},
            {"id": "onyx", "name": "Onyx", "description": "Deep, authoritative"},
            {"id": "nova", "name": "Nova", "description": "Friendly, upbeat"},
            {"id": "shimmer", "name": "Shimmer", "description": "Clear, professional"}
        ],
        "elevenlabs": [
            {"id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "description": "American female, calm"},
            {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "description": "American female, soft"},
            {"id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "description": "American male, well-rounded"},
            {"id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli", "description": "American female, emotional"},
            {"id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh", "description": "American male, deep"},
            {"id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "description": "American male, deep and narration"}
        ]
    }

# Stock Video Search
@app.post("/api/v1/media/video/stock/search", response_model=List[StockVideoResult])
async def search_stock_videos(request: SearchStockVideoRequest):
    """Search for stock videos from Pexels"""
    return await search_pexels_videos(
        request.query,
        request.duration_max,
        request.orientation,
        request.per_page
    )

# Video Composition
@app.post("/api/v1/media/video/compose", response_model=JobResponse)
async def compose_video(
    request: SimpleComposeVideoRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "demo-user"
):
    """Compose multiple video clips into a single video"""
    job_id = str(uuid.uuid4())

    jobs_db[job_id] = {
        "job_id": job_id,
        "job_type": JobType.VIDEO_COMPOSITION,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "input_data": request.model_dump(),
        "user_id": user_id,
        "progress_percent": 0
    }

    background_tasks.add_task(process_video_composition_job, job_id, request, user_id)

    return JobResponse(**jobs_db[job_id])

# Article Generation
@app.post("/api/v1/media/article", response_model=JobResponse)
async def generate_article(
    request: GenerateArticleRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "demo-user"
):
    """Generate an article or blog post from a topic or script"""
    job_id = str(uuid.uuid4())

    jobs_db[job_id] = {
        "job_id": job_id,
        "job_type": JobType.ARTICLE,
        "status": JobStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "input_data": request.model_dump(),
        "user_id": user_id
    }

    background_tasks.add_task(process_article_job, job_id, request, user_id)

    return JobResponse(**jobs_db[job_id])

# Job Management
@app.get("/api/v1/media/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get the status of a generation job"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**jobs_db[job_id])

@app.get("/api/v1/media/jobs", response_model=List[JobResponse])
async def get_user_jobs(user_id: str = "demo-user", limit: int = 20):
    """Get all jobs for a user"""
    user_jobs = [
        JobResponse(**job) for job in jobs_db.values()
        if job.get("user_id") == user_id
    ]
    return sorted(user_jobs, key=lambda x: x.created_at, reverse=True)[:limit]

@app.post("/api/v1/media/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a pending job"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    if job["status"] not in [JobStatus.PENDING, JobStatus.PROCESSING]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")

    jobs_db[job_id]["status"] = JobStatus.CANCELLED
    return {"message": "Job cancelled", "job_id": job_id}

# Asset Management
@app.get("/api/v1/media/assets", response_model=List[AssetResponse])
async def get_user_assets(
    user_id: str = "demo-user",
    asset_type: Optional[str] = None,
    limit: int = 50
):
    """Get all generated assets for a user"""
    user_assets = []
    for asset in assets_db.values():
        if asset.get("user_id") == user_id:
            if asset_type is None or asset.get("asset_type") == asset_type:
                user_assets.append(AssetResponse(
                    id=asset["id"],
                    asset_type=asset["asset_type"],
                    storage_url=asset["storage_url"],
                    file_size_bytes=asset.get("file_size_bytes"),
                    duration_seconds=asset.get("duration_seconds"),
                    dimensions=asset.get("dimensions"),
                    prompt=asset.get("prompt"),
                    created_at=datetime.fromisoformat(asset["created_at"])
                ))

    return sorted(user_assets, key=lambda x: x.created_at, reverse=True)[:limit]

@app.delete("/api/v1/media/assets/{asset_id}")
async def delete_asset(asset_id: str, user_id: str = "demo-user"):
    """Delete a generated asset"""
    if asset_id not in assets_db:
        raise HTTPException(status_code=404, detail="Asset not found")

    if assets_db[asset_id].get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    del assets_db[asset_id]
    return {"message": "Asset deleted", "asset_id": asset_id}

# ========================================
# AI Video Generator Endpoints
# ========================================

class CaptionConfig(BaseModel):
    fontColor: str = "white"
    bgColor: str = "black"
    bgOpacity: float = 0.7
    fontSize: str = "medium"  # small, medium, large
    position: str = "bottom"  # top, center, bottom
    animation: str = "none"  # none, fade, pop, slide, highlight
    stroke: bool = False
    strokeColor: str = "black"
    glow: bool = False
    glowColor: str = "white"
    rounded: bool = False
    highlightColor: Optional[str] = None
    gradientColors: Optional[List[str]] = None

class GenerateVideoFromPromptRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=1000, description="Description of the video to generate")
    duration: int = Field(default=30, ge=15, le=2700, description="Video duration in seconds (max 45 minutes)")
    style: str = Field(default="cinematic", description="Visual style: cinematic, energetic, calm, professional")
    format: str = Field(default="9:16", pattern=r"^(9:16|16:9|1:1)$", description="Video aspect ratio")
    voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID")
    voice_provider: str = Field(default="elevenlabs", description="TTS provider: elevenlabs or openai")
    include_music: bool = Field(default=True, description="Include background music")
    music_style: Optional[str] = Field(default=None, description="Music mood: upbeat, calm, epic, etc.")
    prefer_ai_images: bool = Field(default=False, description="Prefer AI-generated images over stock")
    caption_style: Optional[str] = Field(default=None, description="Caption style: classic, bold, neon, minimal, karaoke, boxed, gradient")
    caption_config: Optional[CaptionConfig] = Field(default=None, description="Detailed caption configuration")

class VideoProjectResponse(BaseModel):
    job_id: str
    status: str
    stages: Dict[str, Any]
    project: Optional[Dict[str, Any]] = None
    output_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str

class UpdateSceneRequest(BaseModel):
    description: Optional[str] = None
    duration: Optional[float] = None
    scene_type: Optional[str] = None
    search_keywords: Optional[List[str]] = None
    text_overlay: Optional[str] = None

class AddSceneRequest(BaseModel):
    description: str
    duration: float = 5
    scene_type: str = "video"
    search_keywords: List[str] = []
    text_overlay: Optional[str] = None
    after_scene_id: Optional[str] = None

@app.post("/api/v1/media/video/generate-from-prompt", response_model=VideoProjectResponse)
async def generate_video_from_prompt(
    request: GenerateVideoFromPromptRequest,
    user_id: str = "demo-user"
):
    """
    Generate a complete video from a text prompt.

    The AI will:
    1. Analyze the prompt and plan scenes
    2. Search for stock videos/images or generate with DALL-E
    3. Generate voiceover with ElevenLabs/OpenAI TTS
    4. Find matching background music
    5. Compose everything into a final video

    Returns a job ID to track progress.
    """
    generator = get_video_generator()

    from services.video_generator import VideoGenerationRequest
    gen_request = VideoGenerationRequest(
        prompt=request.prompt,
        duration=request.duration,
        style=request.style,
        format=request.format,
        voice_id=request.voice_id,
        voice_provider=request.voice_provider,
        include_music=request.include_music,
        music_style=request.music_style,
        prefer_ai_images=request.prefer_ai_images
    )

    job = await generator.generate_video(gen_request, user_id)

    return VideoProjectResponse(
        job_id=job.id,
        status=job.status.value,
        stages={k: v.model_dump() for k, v in job.stages.items()},
        project=job.project.model_dump() if job.project else None,
        output_url=job.output_url,
        error_message=job.error_message,
        created_at=job.created_at.isoformat()
    )

@app.get("/api/v1/media/video/generate/{job_id}", response_model=VideoProjectResponse)
async def get_video_generation_status(job_id: str):
    """Get the status of a video generation job"""
    generator = get_video_generator()
    job = generator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return VideoProjectResponse(
        job_id=job.id,
        status=job.status.value,
        stages={k: v.model_dump() for k, v in job.stages.items()},
        project=job.project.model_dump() if job.project else None,
        output_url=job.output_url,
        error_message=job.error_message,
        created_at=job.created_at.isoformat()
    )

@app.put("/api/v1/media/video/generate/{job_id}/scenes/{scene_id}")
async def update_video_scene(
    job_id: str,
    scene_id: str,
    request: UpdateSceneRequest
):
    """Update a scene in a video project (before final composition)"""
    generator = get_video_generator()

    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    scene = await generator.update_scene(job_id, scene_id, updates)

    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")

    return {"message": "Scene updated", "scene": scene.model_dump()}

@app.post("/api/v1/media/video/generate/{job_id}/scenes")
async def add_video_scene(
    job_id: str,
    request: AddSceneRequest
):
    """Add a new scene to a video project"""
    generator = get_video_generator()

    scene = await generator.add_scene(
        job_id,
        request.model_dump(),
        request.after_scene_id
    )

    if not scene:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"message": "Scene added", "scene": scene.model_dump()}

@app.delete("/api/v1/media/video/generate/{job_id}/scenes/{scene_id}")
async def remove_video_scene(job_id: str, scene_id: str):
    """Remove a scene from a video project"""
    generator = get_video_generator()

    success = await generator.remove_scene(job_id, scene_id)

    if not success:
        raise HTTPException(status_code=404, detail="Scene not found")

    return {"message": "Scene removed"}

@app.post("/api/v1/media/video/generate/{job_id}/regenerate")
async def regenerate_video(job_id: str):
    """Re-compose video after editing scenes"""
    generator = get_video_generator()

    job = await generator.regenerate_from_edit(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Project not found")

    return VideoProjectResponse(
        job_id=job.id,
        status=job.status.value,
        stages={k: v.model_dump() for k, v in job.stages.items()},
        project=job.project.model_dump() if job.project else None,
        output_url=job.output_url,
        error_message=job.error_message,
        created_at=job.created_at.isoformat()
    )

@app.get("/api/v1/media/music/library")
async def get_music_library(
    mood: Optional[str] = None,
    genre: Optional[str] = None
):
    """Get available music tracks from the library"""
    from services.music_service import MusicService
    music_service = MusicService()

    tracks = music_service.get_library_tracks(mood=mood, genre=genre)
    return {
        "tracks": [t.model_dump() for t in tracks],
        "stats": music_service.get_library_stats(),
        "moods": music_service.get_all_moods()
    }

@app.get("/api/v1/media/video/download/{job_id}")
async def download_generated_video(job_id: str):
    """Download the generated video file"""
    from pathlib import Path

    generator = get_video_generator()
    job = generator.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.output_url:
        raise HTTPException(status_code=400, detail="Video not ready yet")

    video_path = Path(job.output_url)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"{job.project.title if job.project else job_id}.mp4"
    )

# ========================================
# Script Generator Endpoints
# ========================================

class GenerateScriptRequest(BaseModel):
    topic: str = Field(..., min_length=5, max_length=500, description="Topic for the video script")
    duration: int = Field(default=60, ge=15, le=2700, description="Video duration in seconds (max 45 minutes)")
    style: str = Field(default="educational", description="Script style: educational, entertaining, motivational, tutorial")
    target_audience: str = Field(default="general", description="Target audience description")

class GenerateFromScriptRequest(BaseModel):
    script: Dict[str, Any] = Field(..., description="Generated script from generate-script endpoint")
    format: str = Field(default="9:16", pattern=r"^(9:16|16:9|1:1)$", description="Video aspect ratio")
    voice_id: str = Field(default="21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID")
    voice_provider: str = Field(default="elevenlabs", description="TTS provider")
    caption_style: Optional[str] = Field(default="classic", description="Caption style")
    caption_config: Optional[Dict[str, Any]] = Field(default=None, description="Caption configuration")

@app.post("/api/v1/media/script/generate")
async def generate_script(request: GenerateScriptRequest):
    """
    Generate a structured video script from a topic.
    Returns Time, Visual, Audio structure for each segment.
    Supports videos up to 30 minutes.
    """
    from services.ai_video_planner import AIVideoPlannerService

    planner = AIVideoPlannerService(openai_api_key=settings.OPENAI_API_KEY)

    try:
        script = await planner.generate_script_from_topic(
            topic=request.topic,
            duration=request.duration,
            style=request.style,
            target_audience=request.target_audience
        )
        return script
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/media/video/generate-from-script", response_model=VideoProjectResponse)
async def generate_video_from_script(
    request: GenerateFromScriptRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "demo-user"
):
    """
    Generate a video from a structured script.
    Uses the script's Time, Visual, Audio structure to create scenes.
    """
    from services.ai_video_planner import AIVideoPlannerService
    from services.video_generator import VideoGenerationRequest

    planner = AIVideoPlannerService(openai_api_key=settings.OPENAI_API_KEY)

    # Convert script to video project
    project = await planner.script_to_video_project(
        script_data=request.script,
        format=request.format,
        voice_id=request.voice_id
    )

    generator = get_video_generator()

    # Log caption style received
    print(f"Caption style received: '{request.caption_style}'")
    print(f"Caption config received: {request.caption_config}")

    # Create generation request
    gen_request = VideoGenerationRequest(
        prompt=f"Script-based video: {request.script.get('title', 'Video')}",
        duration=request.script.get("total_duration", 60),
        style="scripted",
        format=request.format,
        voice_id=request.voice_id,
        voice_provider=request.voice_provider,
        include_music=True,
        music_style=request.script.get("music_mood", "cinematic"),
        prefer_ai_images=False,
        caption_style=request.caption_style,
        caption_config=request.caption_config
    )

    # Start generation with pre-built project
    job = await generator.generate_video_from_project(
        project=project,
        request=gen_request,
        user_id=user_id
    )

    return VideoProjectResponse(
        job_id=job.id,
        status=job.status.value if hasattr(job.status, 'value') else str(job.status),
        stages={k: v.model_dump() for k, v in job.stages.items()},
        project=job.project.model_dump() if job.project else None,
        output_url=job.output_url,
        error_message=job.error_message,
        created_at=job.created_at.isoformat()
    )

# ========================================
# Visual Generation System Endpoints
# ========================================

# Lazy load visual services
_visual_router = None
_avatar_service = None

def get_visual_router():
    global _visual_router
    if _visual_router is None:
        from services.visual_router import VisualRouter
        _visual_router = VisualRouter(
            openai_api_key=settings.OPENAI_API_KEY,
            did_api_key=os.getenv("DID_API_KEY", ""),
            heygen_api_key=os.getenv("HEYGEN_API_KEY", ""),
            pexels_api_key=settings.PEXELS_API_KEY
        )
    return _visual_router

def get_avatar_service():
    global _avatar_service
    if _avatar_service is None:
        from services.avatar_service import AvatarService
        did_key = os.getenv("DID_API_KEY", "")
        if did_key:
            _avatar_service = AvatarService(
                did_api_key=did_key,
                heygen_api_key=os.getenv("HEYGEN_API_KEY", "")
            )
    return _avatar_service

# --- Visual Analysis ---

class AnalyzeSceneRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=1000, description="Scene description to analyze")
    script_context: Optional[str] = Field(default=None, description="Surrounding script context")

class VisualAnalysisResponse(BaseModel):
    visual_type: str
    confidence: float
    diagram_type: Optional[str] = None
    requires_avatar: bool
    mermaid_possible: bool
    domain: Optional[str] = None
    reasoning: str
    suggested_prompt: Optional[str] = None
    extracted_elements: List[str] = []

@app.post("/api/v1/media/visual/analyze", response_model=VisualAnalysisResponse)
async def analyze_scene_visual(request: AnalyzeSceneRequest):
    """
    Analyze a scene description to determine the optimal visual type.
    Uses GPT-4 to understand context and recommend: diagram, chart, avatar, concept, or stock.
    """
    router = get_visual_router()

    analysis = await router.analyze_only(
        description=request.description,
        context=request.script_context
    )

    return VisualAnalysisResponse(
        visual_type=analysis.visual_type.value,
        confidence=analysis.confidence,
        diagram_type=analysis.diagram_type.value if analysis.diagram_type else None,
        requires_avatar=analysis.requires_avatar,
        mermaid_possible=analysis.mermaid_possible,
        domain=analysis.domain,
        reasoning=analysis.reasoning,
        suggested_prompt=analysis.suggested_prompt,
        extracted_elements=analysis.extracted_elements
    )

# --- Diagram Generation ---

class GenerateDiagramRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=2000, description="Diagram description")
    diagram_type: Optional[str] = Field(default=None, description="flowchart, sequence, architecture, class, er, mindmap, state, gantt")
    output_format: str = Field(default="9:16", description="Output aspect ratio")
    force_mermaid: bool = Field(default=False, description="Force Mermaid.js rendering")
    force_dalle: bool = Field(default=False, description="Force DALL-E rendering")

class DiagramResponse(BaseModel):
    job_id: str
    image_url: str
    generator: str  # "mermaid" or "dalle"
    mermaid_code: Optional[str] = None
    fallback_used: bool = False

class PreviewMermaidRequest(BaseModel):
    description: str = Field(..., min_length=10, description="Diagram description")
    diagram_type: str = Field(default="flowchart", description="Type of Mermaid diagram")

@app.post("/api/v1/media/diagram/generate", response_model=DiagramResponse)
async def generate_diagram(request: GenerateDiagramRequest):
    """
    Generate a technical diagram using Mermaid.js or DALL-E.
    Automatically chooses the best method based on content analysis.
    """
    from services.diagram_generator import DiagramGenerator
    from models.visual_types import DiagramType

    diagram_gen = DiagramGenerator(openai_api_key=settings.OPENAI_API_KEY)

    # Parse diagram type
    diagram_type = None
    if request.diagram_type:
        try:
            diagram_type = DiagramType(request.diagram_type.lower())
        except ValueError:
            diagram_type = DiagramType.FLOWCHART

    # Get dimensions
    width, height = (1080, 1920) if request.output_format == "9:16" else (1920, 1080)

    result = await diagram_gen.generate(
        description=request.description,
        diagram_type=diagram_type,
        width=width,
        height=height,
        force_mermaid=request.force_mermaid,
        force_dalle=request.force_dalle
    )

    return DiagramResponse(
        job_id=str(uuid.uuid4()),
        image_url=result.image_url,
        generator=result.generator,
        mermaid_code=result.mermaid_code,
        fallback_used=result.fallback_used
    )

@app.post("/api/v1/media/diagram/preview")
async def preview_mermaid(request: PreviewMermaidRequest):
    """
    Generate a Mermaid code preview without rendering.
    Returns the Mermaid code and a link to mermaid.live editor.
    """
    from services.diagram_generator import DiagramGenerator
    from models.visual_types import DiagramType

    diagram_gen = DiagramGenerator(openai_api_key=settings.OPENAI_API_KEY)

    try:
        diagram_type = DiagramType(request.diagram_type.lower())
    except ValueError:
        diagram_type = DiagramType.FLOWCHART

    preview = await diagram_gen.preview_mermaid(
        description=request.description,
        diagram_type=diagram_type
    )

    return preview

# --- Avatar Endpoints ---

class AvatarVideoGenerateRequest(BaseModel):
    avatar_id: str = Field(..., description="Avatar ID from gallery or custom")
    voiceover_url: Optional[str] = Field(default=None, description="URL to voiceover audio")
    script_text: Optional[str] = Field(default=None, description="Text for TTS (if no voiceover)")
    voice_id: Optional[str] = Field(default=None, description="Voice ID for TTS")
    output_format: str = Field(default="9:16", description="Output aspect ratio")
    expression: str = Field(default="neutral", description="Avatar expression: neutral, happy, serious")

class AvatarVideoResponse(BaseModel):
    job_id: str
    status: str
    video_url: Optional[str] = None
    provider: str
    duration: Optional[float] = None
    error_message: Optional[str] = None

class CustomAvatarUploadRequest(BaseModel):
    photo_url: str = Field(..., description="URL to the uploaded photo")
    name: str = Field(..., min_length=2, max_length=50, description="Display name for the avatar")
    style: str = Field(default="professional", description="Avatar style: professional, casual, creative")

@app.get("/api/v1/media/avatars/gallery")
async def get_avatar_gallery(
    style: Optional[str] = None,
    gender: Optional[str] = None,
    user_id: str = "demo-user"
):
    """
    Get the gallery of available avatars.
    Includes predefined avatars and user's custom avatars.
    """
    avatar_service = get_avatar_service()

    if not avatar_service:
        return {
            "avatars": [],
            "total_count": 0,
            "message": "Avatar service not configured. Set DID_API_KEY environment variable.",
            "styles": ["professional", "casual", "creative"],
            "genders": ["male", "female", "neutral"]
        }

    from models.avatar_models import AvatarStyle, AvatarGender

    # Parse filters
    style_filter = None
    gender_filter = None

    if style:
        try:
            style_filter = AvatarStyle(style.lower())
        except ValueError:
            pass

    if gender:
        try:
            gender_filter = AvatarGender(gender.lower())
        except ValueError:
            pass

    gallery = avatar_service.get_avatar_gallery(
        user_id=user_id,
        style=style_filter,
        gender=gender_filter
    )

    return {
        "avatars": [a.model_dump() for a in gallery.avatars],
        "total_count": gallery.total_count,
        "styles": gallery.styles,
        "genders": gallery.genders
    }

@app.post("/api/v1/media/avatars/generate", response_model=AvatarVideoResponse)
async def generate_avatar_video(
    request: AvatarVideoGenerateRequest,
    user_id: str = "demo-user"
):
    """
    Generate an avatar video with lip-sync using D-ID.
    Requires either voiceover_url OR (script_text + voice_id).
    """
    avatar_service = get_avatar_service()

    if not avatar_service:
        raise HTTPException(
            status_code=503,
            detail="Avatar service not configured. Set DID_API_KEY environment variable."
        )

    from models.avatar_models import AvatarVideoRequest

    avatar_request = AvatarVideoRequest(
        avatar_id=request.avatar_id,
        voiceover_url=request.voiceover_url,
        script_text=request.script_text,
        voice_id=request.voice_id,
        output_format=request.output_format,
        expression=request.expression
    )

    try:
        result = await avatar_service.generate_avatar_video(
            request=avatar_request,
            user_id=user_id
        )

        return AvatarVideoResponse(
            job_id=result.job_id,
            status=result.status,
            video_url=result.video_url,
            provider=result.provider.value,
            duration=result.duration
        )

    except Exception as e:
        return AvatarVideoResponse(
            job_id=str(uuid.uuid4()),
            status="failed",
            provider="d-id",
            error_message=str(e)
        )

@app.get("/api/v1/media/avatars/generate/{job_id}")
async def get_avatar_generation_status(job_id: str):
    """Get the status of an avatar video generation job."""
    avatar_service = get_avatar_service()

    if not avatar_service:
        raise HTTPException(status_code=503, detail="Avatar service not configured")

    status = await avatar_service.get_generation_status(job_id)
    return status

@app.post("/api/v1/media/avatars/custom")
async def create_custom_avatar(
    request: CustomAvatarUploadRequest,
    user_id: str = "demo-user"
):
    """
    Create a custom avatar from a user-uploaded photo.
    The photo should contain a clear face for best results.
    """
    avatar_service = get_avatar_service()

    if not avatar_service:
        raise HTTPException(status_code=503, detail="Avatar service not configured")

    from models.avatar_models import CustomAvatarRequest, AvatarStyle

    try:
        style = AvatarStyle(request.style.lower())
    except ValueError:
        style = AvatarStyle.PROFESSIONAL

    custom_request = CustomAvatarRequest(
        photo_url=request.photo_url,
        user_id=user_id,
        name=request.name,
        style=style
    )

    try:
        result = await avatar_service.create_custom_avatar(custom_request)
        return {
            "avatar": result.avatar.model_dump(),
            "status": result.processing_status,
            "message": "Custom avatar created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/media/avatars/custom/{avatar_id}")
async def delete_custom_avatar(avatar_id: str, user_id: str = "demo-user"):
    """Delete a user's custom avatar."""
    avatar_service = get_avatar_service()

    if not avatar_service:
        raise HTTPException(status_code=503, detail="Avatar service not configured")

    success = await avatar_service.delete_custom_avatar(avatar_id, user_id)

    if not success:
        raise HTTPException(status_code=404, detail="Avatar not found")

    return {"message": "Avatar deleted", "avatar_id": avatar_id}

# --- Visual Router (Combined) ---

class GenerateVisualRequest(BaseModel):
    description: str = Field(..., min_length=5, max_length=2000)
    script_context: Optional[str] = None
    preferred_type: Optional[str] = Field(default=None, description="diagram, chart, avatar, concept, stock, ai_image")
    output_format: str = Field(default="9:16")
    style: Optional[str] = None
    avatar_id: Optional[str] = None
    voiceover_url: Optional[str] = None

class GenerateVisualResponse(BaseModel):
    visual_type: str
    asset_url: str
    asset_type: str  # "image" or "video"
    generator_used: str
    mermaid_code: Optional[str] = None
    metadata: Dict[str, Any] = {}

@app.post("/api/v1/media/visual/generate", response_model=GenerateVisualResponse)
async def generate_visual(request: GenerateVisualRequest):
    """
    Intelligently generate a visual based on content analysis.
    Automatically routes to the best generator (Mermaid, DALL-E, D-ID, or stock).
    """
    from models.visual_types import VisualType, VisualGenerationRequest

    router = get_visual_router()

    # Parse preferred type
    preferred_type = None
    if request.preferred_type:
        try:
            preferred_type = VisualType(request.preferred_type.lower())
        except ValueError:
            pass

    gen_request = VisualGenerationRequest(
        description=request.description,
        script_context=request.script_context,
        preferred_type=preferred_type,
        output_format=request.output_format,
        style=request.style,
        avatar_id=request.avatar_id,
        voiceover_url=request.voiceover_url
    )

    result = await router.route_scene(gen_request)

    return GenerateVisualResponse(
        visual_type=result.visual_type.value,
        asset_url=result.asset_url,
        asset_type=result.asset_type,
        generator_used=result.generator_used,
        mermaid_code=result.mermaid_code,
        metadata=result.metadata
    )

# ========================================
# Run Server
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
