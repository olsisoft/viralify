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
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DID_API_KEY: str = os.getenv("DID_API_KEY", "")

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
            pixabay_api_key=settings.PIXABAY_API_KEY,
            did_api_key=settings.DID_API_KEY
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

class AvatarQualityMode(str, Enum):
    """Avatar animation quality modes for cost optimization.

    Cost comparison per 15s video:
    - DRAFT: ~$0.002 (SadTalker - fast preview)
    - PREVIEW: ~$0.01 (SadTalker + upscale - good quality)
    - FINAL: ~$2.80 (OmniHuman - full body, best quality)
    """
    DRAFT = "draft"       # SadTalker - fast, cheap, head motion only
    PREVIEW = "preview"   # SadTalker + face enhancement - good quality, affordable
    FINAL = "final"       # OmniHuman - full body animation, premium quality

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
    language: str = Field(default="en", description="Content language code (en, fr, es, de, etc.)")

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


class SlideshowScene(BaseModel):
    """A single scene in a slideshow"""
    image_url: Optional[str] = Field(None, description="URL or file path to the image")
    video_url: Optional[str] = Field(None, description="URL or file path to a video (alternative to image)")
    audio_url: Optional[str] = Field(None, description="URL to audio for this specific scene (for sync)")
    duration: float = Field(default=5.0, description="Duration in seconds (ignored for video scenes)")
    transition: str = Field(default="fade", description="Transition effect: fade, cut, dissolve")


class ComposeSlideshowRequest(BaseModel):
    """Request to compose a slideshow video from images + audio"""
    scenes: List[SlideshowScene] = Field(..., min_items=1, max_items=50)
    voiceover_url: Optional[str] = Field(None, description="URL to voiceover audio")
    music_url: Optional[str] = Field(None, description="URL to background music")
    music_volume: float = Field(default=0.2, ge=0.0, le=1.0)
    output_format: str = Field(default="16:9", pattern=r"^(9:16|16:9|1:1)$")
    quality: str = Field(default="1080p", pattern=r"^(720p|1080p|4k)$")
    fps: int = Field(default=30, ge=24, le=60)
    ken_burns_effect: bool = Field(default=False, description="Enable Ken Burns zoom/pan effect on images. False for static display.")

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
    """Generate photorealistic image using GPT-4o via Chat Completions API"""
    if settings.DEMO_MODE:
        await asyncio.sleep(2)
        return {
            "url": f"https://picsum.photos/1080/1920?random={uuid.uuid4().hex[:8]}",
            "revised_prompt": f"[DEMO] {prompt}",
            "provider": "gpt-4o"
        }

    # Add style to prompt for photorealistic results - CRITICAL for realistic output
    style_prompts = {
        ImageStyle.REALISTIC: "Ultra-realistic photograph, shot on Canon EOS R5 with 85mm lens, natural ambient lighting, real human subjects, real-world environment, NOT AI-generated, NOT illustration, NOT cartoon, NOT digital art, documentary style, photojournalistic quality",
        ImageStyle.ANIME: "anime style, vibrant colors, detailed illustration, Japanese animation aesthetic",
        ImageStyle.ILLUSTRATION: "professional digital illustration, artistic, detailed, concept art quality",
        ImageStyle.THREE_D: "3D rendered, CGI quality, cinematic lighting, Unreal Engine quality",
        ImageStyle.MINIMALIST: "minimalist design, clean lines, modern aesthetic, subtle elegance",
        ImageStyle.VIBRANT: "vibrant colors, eye-catching, dynamic composition, high contrast"
    }

    # For realistic style, use stronger photorealistic prompt
    if style == ImageStyle.REALISTIC:
        enhanced_prompt = f"I NEED a real photograph, not AI art. {prompt}. {style_prompts[style]}. This must look like a genuine photo taken by a professional photographer, with real textures, natural imperfections, and authentic lighting."
    else:
        enhanced_prompt = f"{prompt}. Style: {style_prompts.get(style, 'photorealistic, real photograph')}"

    async with httpx.AsyncClient() as client:
        # Use GPT-4o for photorealistic image generation
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Generate a photorealistic image: {enhanced_prompt}"
                        }
                    ],
                    "modalities": ["text", "image"],
                    "max_tokens": 1000
                },
                timeout=120.0
            )

            if response.status_code == 200:
                data = response.json()
                # Extract image from response
                for choice in data.get("choices", []):
                    message = choice.get("message", {})
                    for content in message.get("content", []):
                        if content.get("type") == "image":
                            image_data = content.get("image", {})
                            if "url" in image_data:
                                return {
                                    "url": image_data["url"],
                                    "revised_prompt": enhanced_prompt,
                                    "provider": "gpt-4o"
                                }
                            elif "b64_json" in image_data or "data" in image_data:
                                import base64
                                os.makedirs("/tmp/viralify/images", exist_ok=True)
                                image_path = f"/tmp/viralify/images/{uuid.uuid4().hex}.png"
                                b64_data = image_data.get("b64_json") or image_data.get("data")
                                with open(image_path, "wb") as f:
                                    f.write(base64.b64decode(b64_data))
                                return {
                                    "url": image_path,
                                    "revised_prompt": enhanced_prompt,
                                    "provider": "gpt-4o"
                                }
                print(f"GPT-4o response didn't contain image, trying images API...")
            else:
                print(f"GPT-4o chat failed ({response.status_code}), trying images API...")

        except Exception as e:
            print(f"GPT-4o chat failed: {e}, trying images API...")

        # Try gpt-image-1.5 model (most photorealistic)
        try:
            size_map = {
                "1:1": "1024x1024",
                "9:16": "1024x1536",
                "16:9": "1536x1024"
            }
            size = size_map.get(aspect_ratio, "1024x1536")

            # Try gpt-image-1.5 first, then gpt-image-1
            for model in ["gpt-image-1.5", "gpt-image-1"]:
                try:
                    response = await client.post(
                        "https://api.openai.com/v1/images/generations",
                        headers={
                            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": model,
                            "prompt": enhanced_prompt,
                            "n": 1,
                            "size": size,
                            "quality": "high"
                        },
                        timeout=120.0
                    )
                    if response.status_code == 200:
                        print(f"Using {model} for image generation")
                        break
                except Exception as e:
                    print(f"{model} failed: {e}")
                    continue

            if response.status_code == 200:
                data = response.json()
                image_data = data["data"][0]
                if "b64_json" in image_data:
                    import base64
                    os.makedirs("/tmp/viralify/images", exist_ok=True)
                    image_path = f"/tmp/viralify/images/{uuid.uuid4().hex}.png"
                    with open(image_path, "wb") as f:
                        f.write(base64.b64decode(image_data["b64_json"]))
                    return {
                        "url": image_path,
                        "revised_prompt": image_data.get("revised_prompt", prompt),
                        "provider": "gpt-image-1"
                    }
                else:
                    return {
                        "url": image_data["url"],
                        "revised_prompt": image_data.get("revised_prompt", prompt),
                        "provider": "gpt-image-1"
                    }
        except Exception as e:
            print(f"gpt-image-1 failed: {e}, falling back to DALL-E 3...")

        # Try Gemini as backup (photorealistic)
        if settings.GEMINI_API_KEY:
            try:
                print("Trying Gemini 2.0 Flash for image generation...")
                # Use Gemini 2.0 Flash with image generation capabilities
                gemini_response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={settings.GEMINI_API_KEY}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{
                            "parts": [{"text": f"Generate a photorealistic image: {enhanced_prompt}"}]
                        }],
                        "generationConfig": {
                            "responseModalities": ["TEXT", "IMAGE"]
                        }
                    },
                    timeout=120.0
                )

                if gemini_response.status_code == 200:
                    gemini_data = gemini_response.json()
                    # Extract image from Gemini response
                    candidates = gemini_data.get("candidates", [])
                    for candidate in candidates:
                        content = candidate.get("content", {})
                        for part in content.get("parts", []):
                            if "inlineData" in part:
                                inline_data = part["inlineData"]
                                if inline_data.get("mimeType", "").startswith("image/"):
                                    os.makedirs("/tmp/viralify/images", exist_ok=True)
                                    image_path = f"/tmp/viralify/images/gemini_{uuid.uuid4().hex}.png"
                                    with open(image_path, "wb") as f:
                                        f.write(base64.b64decode(inline_data["data"]))
                                    print(f"Gemini 2.0 Flash generated image successfully")
                                    return {
                                        "url": image_path,
                                        "revised_prompt": enhanced_prompt,
                                        "provider": "gemini-2.0-flash"
                                    }
                    print(f"Gemini response didn't contain image data")
                elif gemini_response.status_code == 429:
                    print(f"Gemini quota exceeded, falling back to DALL-E 3...")
                else:
                    print(f"Gemini failed ({gemini_response.status_code}): {gemini_response.text[:300]}")
            except Exception as e:
                print(f"Gemini failed: {e}")

        # Final fallback to DALL-E 3
        print("Falling back to DALL-E 3...")
        dalle_size_map = {
            "1:1": "1024x1024",
            "9:16": "1024x1792",
            "16:9": "1792x1024"
        }
        dalle_size = dalle_size_map.get(aspect_ratio, "1024x1792")

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
                "size": dalle_size,
                "quality": quality
            },
            timeout=90.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Image generation error: {response.text}")

        data = response.json()
        return {
            "url": data["data"][0]["url"],
            "revised_prompt": data["data"][0].get("revised_prompt", prompt),
            "provider": "dalle3"
        }

async def generate_diagram_dalle(description: str, style: str, aspect_ratio: str) -> Dict:
    """Generate a diagram-style image using GPT-Image-1 (photorealistic) with DALL-E 3 fallback"""
    if settings.DEMO_MODE:
        await asyncio.sleep(2)
        return {
            "url": f"https://placehold.co/1080x1920/1e3a5f/ffffff?text=Diagram",
            "diagram_url": f"https://placehold.co/1080x1920/1e3a5f/ffffff?text=Diagram",
            "provider": "gpt-image-1"
        }

    # Map aspect ratio to GPT-Image-1 sizes
    size_mapping = {
        "9:16": "1024x1536",
        "16:9": "1536x1024",
        "1:1": "1024x1024"
    }
    size = size_mapping.get(aspect_ratio, "1024x1536")

    # Style-specific prompt enhancements for realistic diagrams
    style_prompts = {
        "modern": "modern professional infographic, clean corporate design, realistic 3D elements, glass and metal textures",
        "minimal": "minimalist design, elegant simplicity, subtle shadows, premium aesthetic",
        "technical": "technical visualization, realistic engineering diagram, precise details, professional presentation",
        "colorful": "vibrant professional presentation, dynamic composition, engaging visual hierarchy"
    }
    style_enhancement = style_prompts.get(style, style_prompts["modern"])

    # Create an enhanced prompt for photorealistic diagrams
    diagram_prompt = f"""I NEED a real photograph of a professional presentation screen or digital display showing: {description}

CRITICAL REQUIREMENTS - This must look like a REAL PHOTO:
- {style_enhancement}
- Photo of an actual LED screen, monitor, or presentation display in a real environment
- Real ambient lighting, slight reflections on the screen surface
- Visible screen bezels and real-world context (office, conference room, studio)
- The content ON the screen can be digital/graphical, but the PHOTO itself must be realistic
- Shot with professional camera, shallow depth of field
- Natural imperfections that real photos have
- NOT an AI-generated illustration
- NOT cartoon or stylized art
- Must pass as a genuine photograph"""

    async with httpx.AsyncClient() as client:
        # Try GPT-Image-1 first (more photorealistic)
        try:
            response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-image-1",
                    "prompt": diagram_prompt,
                    "n": 1,
                    "size": size,
                    "quality": "high"
                },
                timeout=120.0
            )

            if response.status_code == 200:
                data = response.json()
                image_data = data["data"][0]
                if "b64_json" in image_data:
                    import base64
                    os.makedirs("/tmp/viralify/images", exist_ok=True)
                    image_path = f"/tmp/viralify/images/diagram_{uuid.uuid4().hex}.png"
                    with open(image_path, "wb") as f:
                        f.write(base64.b64decode(image_data["b64_json"]))
                    return {
                        "url": image_path,
                        "diagram_url": image_path,
                        "revised_prompt": image_data.get("revised_prompt", description),
                        "provider": "gpt-image-1"
                    }
                else:
                    url = image_data["url"]
                    return {
                        "url": url,
                        "diagram_url": url,
                        "revised_prompt": image_data.get("revised_prompt", description),
                        "provider": "gpt-image-1"
                    }
        except Exception as e:
            print(f"GPT-Image-1 diagram failed, falling back to DALL-E 3: {e}")

        # Fallback to DALL-E 3
        dalle_size_mapping = {
            "9:16": "1024x1792",
            "16:9": "1792x1024",
            "1:1": "1024x1024"
        }
        dalle_size = dalle_size_mapping.get(aspect_ratio, "1024x1792")

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
                "size": dalle_size,
                "quality": "hd"
            },
            timeout=90.0
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Diagram generation error: {response.text}")

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
    # Valid OpenAI TTS voices
    VALID_OPENAI_VOICES = ['nova', 'shimmer', 'echo', 'onyx', 'fable', 'alloy', 'ash', 'sage', 'coral']

    # Validate voice_id - fallback to 'alloy' if invalid
    if voice_id not in VALID_OPENAI_VOICES:
        print(f"[TTS] Invalid OpenAI voice '{voice_id}', falling back to 'alloy'", flush=True)
        voice_id = 'alloy'

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

async def generate_voiceover_elevenlabs(text: str, voice_id: str, emotion: Optional[str], language: str = "en") -> Dict:
    """Generate voiceover using ElevenLabs with multilingual support"""
    if settings.DEMO_MODE:
        await asyncio.sleep(2)
        return {
            "url": "https://example.com/demo-audio-elevenlabs.mp3",
            "duration_seconds": len(text) / 12,
            "provider": "elevenlabs"
        }

    # Import voice service for language-based voice selection
    from services.voice_service import get_voice_service
    voice_service = get_voice_service()

    # Get appropriate voice for the language if not specified
    if not voice_id or voice_id == "21m00Tcm4TlvDq8ikWAM":  # Default Rachel voice
        voice_id = voice_service.get_voice_by_gender("male", language, "elevenlabs")

    # Use multilingual model for non-English content
    # eleven_multilingual_v2 supports: en, de, pl, es, it, fr, pt, hi, zh, ar, ko, nl, tr, sv, id, fil, ja, uk, el, cs, fi, ro, da, bg, ms, sk, hr, ca, ar
    model_id = "eleven_multilingual_v2" if language != "en" else "eleven_monolingual_v1"

    print(f"[TTS] Generating voiceover: language={language}, voice_id={voice_id}, model={model_id}", flush=True)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": settings.ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "text": text,
                "model_id": model_id,
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
        # Use ElevenLabs for non-English content (better multilingual support)
        language = getattr(request, 'language', 'en') or 'en'

        if request.provider == VoiceProvider.ELEVENLABS or language != "en":
            result = await generate_voiceover_elevenlabs(
                request.text,
                request.voice_id,
                request.emotion,
                language
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
    print("[STARTUP] Media Generator Service starting...")
    print(f"[STARTUP] Demo Mode: {settings.DEMO_MODE}")
    yield
    # Shutdown
    print("[SHUTDOWN] Media Generator Service shutting down...")

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


# ========================================
# Cache Management Endpoints
# ========================================

@app.get("/api/v1/media/cache/stats")
async def get_cache_stats():
    """Get avatar video cache statistics."""
    from services.local_avatar_service import get_local_avatar_service
    local_avatar = get_local_avatar_service()
    stats = local_avatar.get_cache_stats()
    return {
        "status": "success",
        "cache": stats
    }


@app.delete("/api/v1/media/cache")
async def clear_cache():
    """Clear avatar video cache."""
    from services.local_avatar_service import get_local_avatar_service
    local_avatar = get_local_avatar_service()
    result = local_avatar.clear_cache()
    return {
        "status": "success",
        "message": f"Cleared {result['cleared']} cached videos"
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

@app.get("/api/v1/media/files/{filename}")
async def serve_generated_image(filename: str):
    """Serve generated images from /tmp/viralify/images"""
    # Check in /tmp/viralify/images first
    tmp_path = Path(f"/tmp/viralify/images/{filename}")
    if tmp_path.exists():
        return FileResponse(path=str(tmp_path), media_type="image/png")

    # Check in IMAGES_DIR
    img_path = IMAGES_DIR / filename
    if img_path.exists():
        return FileResponse(path=str(img_path), media_type="image/png")

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/files/videos/{filename}")
async def serve_video_file(filename: str):
    """Serve generated videos from /tmp/viralify/videos"""
    video_path = Path(f"/tmp/viralify/videos/{filename}")
    if video_path.exists():
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=filename
        )
    raise HTTPException(status_code=404, detail="Video not found")

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


# Slideshow Composition (Images + Audio)
async def process_slideshow_job(job_id: str, request: ComposeSlideshowRequest, user_id: str):
    """Process slideshow composition job - combines images with audio into video"""
    from services.video_compositor import VideoCompositorService, CompositionRequest, CompositionScene

    try:
        jobs_db[job_id]["status"] = JobStatus.PROCESSING
        jobs_db[job_id]["progress_percent"] = 10

        print(f"[SLIDESHOW] Starting composition for job {job_id}", flush=True)
        print(f"[SLIDESHOW] {len(request.scenes)} scenes, voiceover: {bool(request.voiceover_url)}", flush=True)

        # Build composition scenes
        composition_scenes = []
        current_time = 0.0

        for i, scene in enumerate(request.scenes):
            # Determine if this is a video or image scene
            if scene.video_url:
                media_url = scene.video_url
                media_type = "video"
            else:
                media_url = scene.image_url
                media_type = "image"

            composition_scenes.append(CompositionScene(
                id=f"slide_{i:03d}",
                order=i,
                media_url=media_url,
                media_type=media_type,
                duration=scene.duration,
                start_time=current_time,
                audio_url=scene.audio_url,  # Per-scene audio for sync
                transition=scene.transition
            ))
            current_time += scene.duration

        # Create composition request
        comp_request = CompositionRequest(
            project_id=job_id,
            scenes=composition_scenes,
            voiceover_url=request.voiceover_url,
            music_url=request.music_url,
            music_volume=request.music_volume,
            format=request.output_format,
            quality=request.quality,
            fps=request.fps,
            ken_burns_effect=request.ken_burns_effect
        )

        jobs_db[job_id]["progress_percent"] = 30

        # Create compositor and compose video
        compositor = VideoCompositorService()

        def progress_callback(percent, message):
            # Scale progress from 30-90
            scaled = 30 + int(percent * 0.6)
            jobs_db[job_id]["progress_percent"] = scaled
            print(f"[SLIDESHOW] {message} ({scaled}%)", flush=True)

        result = await compositor.compose_video(comp_request, progress_callback)

        if result.success:
            jobs_db[job_id]["status"] = JobStatus.COMPLETED
            jobs_db[job_id]["progress_percent"] = 100
            jobs_db[job_id]["output_data"] = {
                "video_url": result.output_url,
                "duration_seconds": result.duration,
                "file_size_bytes": result.file_size_bytes
            }
            print(f"[SLIDESHOW] Completed! URL: {result.output_url}", flush=True)
        else:
            jobs_db[job_id]["status"] = JobStatus.FAILED
            jobs_db[job_id]["error_message"] = result.error_message
            print(f"[SLIDESHOW] Failed: {result.error_message}", flush=True)

    except Exception as e:
        print(f"[SLIDESHOW] Error: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        jobs_db[job_id]["status"] = JobStatus.FAILED
        jobs_db[job_id]["error_message"] = str(e)


@app.post("/api/v1/media/slideshow/compose", response_model=JobResponse)
async def compose_slideshow(
    request: ComposeSlideshowRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "demo-user"
):
    """
    Compose a slideshow video from images and audio.

    Creates a video from a series of images with optional voiceover and background music.
    Each scene can have its own duration and transition effect.
    """
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

    background_tasks.add_task(process_slideshow_job, job_id, request, user_id)

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

# Voice IDs for gender matching
MALE_VOICES = {
    "elevenlabs": "pNInz6obpgDQGcFmaJgB",  # Adam - deep male voice
    "openai": "onyx"  # Deep male voice
}
FEMALE_VOICES = {
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",  # Rachel - female voice
    "openai": "nova"  # Female voice
}


def match_voice_to_avatar_gender(avatar_id: Optional[str], voice_provider: str = "elevenlabs") -> str:
    """Get appropriate voice ID based on avatar gender."""
    if not avatar_id:
        return FEMALE_VOICES.get(voice_provider, "21m00Tcm4TlvDq8ikWAM")

    # Load avatar config and check gender
    try:
        import json
        config_path = "/app/config/avatars.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        for avatar in config.get("avatars", []):
            if avatar.get("id") == avatar_id:
                gender = avatar.get("gender", "female").lower()
                print(f"[Voice Match] Avatar '{avatar_id}' gender: {gender}", flush=True)
                if gender == "male":
                    voice = MALE_VOICES.get(voice_provider, "pNInz6obpgDQGcFmaJgB")
                    print(f"[Voice Match] Using MALE voice: {voice}", flush=True)
                    return voice
                else:
                    voice = FEMALE_VOICES.get(voice_provider, "21m00Tcm4TlvDq8ikWAM")
                    print(f"[Voice Match] Using FEMALE voice: {voice}", flush=True)
                    return voice
    except Exception as e:
        print(f"[Voice Match] Error loading avatar config: {e}", flush=True)

    # Default to female voice
    return FEMALE_VOICES.get(voice_provider, "21m00Tcm4TlvDq8ikWAM")


class GenerateVideoFromPromptRequest(BaseModel):
    prompt: str = Field(..., min_length=10, max_length=1000, description="Description of the video to generate")
    duration: int = Field(default=30, ge=15, le=2700, description="Video duration in seconds (max 45 minutes)")
    style: str = Field(default="cinematic", description="Visual style: cinematic, energetic, calm, professional")
    format: str = Field(default="9:16", pattern=r"^(9:16|16:9|1:1)$", description="Video aspect ratio")
    voice_id: Optional[str] = Field(default=None, description="ElevenLabs voice ID (auto-matched to avatar gender if not provided)")
    voice_provider: str = Field(default="elevenlabs", description="TTS provider: elevenlabs or openai")
    auto_match_voice: bool = Field(default=True, description="Automatically match voice gender to avatar gender")
    include_music: bool = Field(default=True, description="Include background music")
    music_style: Optional[str] = Field(default=None, description="Music mood: upbeat, calm, epic, etc.")
    prefer_ai_images: bool = Field(default=False, description="Prefer AI-generated images over stock")
    caption_style: Optional[str] = Field(default=None, description="Caption style: classic, bold, neon, minimal, karaoke, boxed, gradient")
    caption_config: Optional[CaptionConfig] = Field(default=None, description="Detailed caption configuration")
    # Lip-sync and Avatar options
    enable_lipsync: bool = Field(default=False, description="Enable lip-sync avatar animation (PIP overlay)")
    avatar_id: Optional[str] = Field(default=None, description="Avatar ID from gallery (e.g., 'avatar-pro-male-alex')")
    lipsync_expression: str = Field(default="neutral", description="Avatar expression: neutral, happy, serious")
    enable_body_motion: bool = Field(default=True, description="Enable natural body movements during speech")
    use_presenter: bool = Field(default=False, description="Use D-ID presenter (real actor with natural movements)")
    # PIP (Picture-in-Picture) avatar overlay options
    pip_position: str = Field(default="bottom-right", description="PIP position: bottom-right, bottom-left, top-right, top-left")
    pip_size: float = Field(default=0.35, ge=0.2, le=0.5, description="PIP size as fraction of screen width (0.2-0.5)")
    pip_remove_background: bool = Field(default=True, description="Remove avatar background for seamless video blending")
    # Face Swap options
    face_swap_image: Optional[str] = Field(default=None, description="URL or base64 of user's face to swap onto avatar")
    face_swap_hair_source: str = Field(default="user", description="Hair source: 'user' (keep user's hair) or 'target' (keep avatar's hair)")
    # Quality/Cost optimization
    avatar_quality: str = Field(
        default="final",
        description="Avatar quality mode for cost optimization: 'draft' (~$0.002, SadTalker), 'preview' (~$0.01), 'final' (~$2.80, OmniHuman)"
    )

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

    # Auto-match voice to avatar gender if enabled and no voice_id provided
    voice_id = request.voice_id
    if request.auto_match_voice and request.enable_lipsync and request.avatar_id:
        if not voice_id:
            voice_id = match_voice_to_avatar_gender(request.avatar_id, request.voice_provider)
            print(f"[API] Auto-matched voice for avatar '{request.avatar_id}': {voice_id}", flush=True)
    elif not voice_id:
        # Default voice if not specified
        voice_id = FEMALE_VOICES.get(request.voice_provider, "21m00Tcm4TlvDq8ikWAM")

    from services.video_generator import VideoGenerationRequest
    gen_request = VideoGenerationRequest(
        prompt=request.prompt,
        duration=request.duration,
        style=request.style,
        format=request.format,
        voice_id=voice_id,
        voice_provider=request.voice_provider,
        include_music=request.include_music,
        music_style=request.music_style,
        prefer_ai_images=request.prefer_ai_images,
        caption_style=request.caption_style,
        caption_config=request.caption_config.model_dump() if request.caption_config else None,
        # Lip-sync and Avatar options
        enable_lipsync=request.enable_lipsync,
        avatar_id=request.avatar_id,
        lipsync_expression=request.lipsync_expression,
        enable_body_motion=request.enable_body_motion,
        use_presenter=request.use_presenter,
        # Face Swap options
        face_swap_image=request.face_swap_image,
        face_swap_hair_source=request.face_swap_hair_source,
        # PIP options
        pip_position=request.pip_position,
        pip_size=request.pip_size,
        pip_shadow=request.pip_shadow,
        pip_remove_background=request.pip_remove_background,
        # Quality/Cost optimization
        avatar_quality=request.avatar_quality
    )

    # Log lip-sync options
    if request.enable_lipsync:
        print(f"[API] Lip-sync enabled with avatar: {request.avatar_id}", flush=True)
        print(f"[API] Voice matched: {voice_id} (provider: {request.voice_provider})", flush=True)
        print(f"[API] PIP position: {request.pip_position}, size: {request.pip_size}", flush=True)

    # Debug face swap
    face_swap_value = request.face_swap_image
    print(f"[API] Face swap image received: {bool(face_swap_value)} (length: {len(face_swap_value) if face_swap_value else 0})", flush=True)
    if face_swap_value:
        print(f"[API] Face swap enabled with hair_source: {request.face_swap_hair_source}", flush=True)
        print(f"[API] Face swap image preview: {face_swap_value[:100]}...", flush=True)

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
    voice_id: Optional[str] = Field(default=None, description="ElevenLabs voice ID (auto-selected if not provided)")
    voice_provider: str = Field(default="elevenlabs", description="TTS provider")
    language: str = Field(default="en", description="Voice language code (en, es, fr, de, pt, etc.)")
    auto_match_voice: bool = Field(default=True, description="Automatically match voice to avatar gender")
    # Visual options
    prefer_ai_images: bool = Field(default=False, description="Prefer AI-generated images over stock footage")
    caption_style: Optional[str] = Field(default="classic", description="Caption style")
    caption_config: Optional[Dict[str, Any]] = Field(default=None, description="Caption configuration")
    # Lip-sync options
    enable_lipsync: bool = Field(default=False, description="Enable lip-sync animation for presenter scenes")
    avatar_id: Optional[str] = Field(default=None, description="Avatar ID for lip-sync (use gallery or custom)")
    lipsync_expression: str = Field(default="neutral", description="Avatar expression: neutral, happy, serious")
    # Body motion options
    enable_body_motion: bool = Field(default=True, description="Enable natural body movements during speech")
    use_presenter: bool = Field(default=False, description="Use D-ID presenter (real actor with natural movements)")
    # PIP (Picture-in-Picture) avatar overlay options
    pip_position: str = Field(default="bottom-right", description="PIP position: bottom-right, bottom-left, top-right, top-left")
    pip_size: float = Field(default=0.35, ge=0.2, le=0.5, description="PIP size as fraction of screen width (0.2-0.5)")
    pip_shadow: bool = Field(default=True, description="Add drop shadow to PIP overlay")
    # Background removal for seamless avatar blending
    pip_remove_background: bool = Field(default=True, description="Remove avatar background for seamless video blending")
    pip_bg_color: Optional[str] = Field(default=None, description="Background color to remove (hex format like 0xRRGGBB, auto-detect if None)")
    pip_bg_similarity: float = Field(default=0.3, ge=0.1, le=0.5, description="Color similarity threshold for background removal (0.1-0.5)")
    # Face Swap options
    face_swap_image: Optional[str] = Field(default=None, description="URL or base64 of user's face to swap onto avatar")
    face_swap_hair_source: str = Field(default="user", description="Hair source: 'user' (keep user's hair) or 'target' (keep avatar's hair)")
    # Quality/Cost optimization
    avatar_quality: str = Field(default="final", description="Avatar quality mode: 'draft', 'preview', 'standard', 'final'")

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
    Automatically matches voice to avatar gender if auto_match_voice is True.
    """
    from services.ai_video_planner import AIVideoPlannerService
    from services.video_generator import VideoGenerationRequest
    from services.voice_service import get_voice_service

    # Determine the voice ID to use
    voice_id = request.voice_id
    if request.enable_lipsync and request.avatar_id and request.auto_match_voice:
        # Auto-match voice to avatar gender
        voice_service = get_voice_service()
        voice_id = voice_service.get_voice_for_avatar(
            avatar_id=request.avatar_id,
            language=request.language,
            provider=request.voice_provider
        )
        print(f"Auto-matched voice '{voice_id}' for avatar '{request.avatar_id}' (language: {request.language})")
    elif not voice_id:
        # Default voice if none specified
        voice_id = "21m00Tcm4TlvDq8ikWAM"

    planner = AIVideoPlannerService(openai_api_key=settings.OPENAI_API_KEY)

    # Convert script to video project
    project = await planner.script_to_video_project(
        script_data=request.script,
        format=request.format,
        voice_id=voice_id
    )

    generator = get_video_generator()

    # Log options received
    print(f"Prefer AI images: {request.prefer_ai_images}")
    print(f"Caption style received: '{request.caption_style}'")
    print(f"Caption config received: {request.caption_config}")
    print(f"Lip-sync enabled: {request.enable_lipsync}, Avatar ID: {request.avatar_id}")
    print(f"Voice ID: {voice_id}, Language: {request.language}, Auto-match: {request.auto_match_voice}")
    print(f"Body motion: {request.enable_body_motion}, Use presenter: {request.use_presenter}")

    # Debug face swap
    face_swap_value = request.face_swap_image
    print(f"[API-Script] Face swap image received: {bool(face_swap_value)} (length: {len(face_swap_value) if face_swap_value else 0})", flush=True)
    if face_swap_value:
        print(f"[API-Script] Face swap enabled with hair_source: {request.face_swap_hair_source}", flush=True)
        print(f"[API-Script] Face swap image preview: {face_swap_value[:100]}...", flush=True)

    # Create generation request
    gen_request = VideoGenerationRequest(
        prompt=f"Script-based video: {request.script.get('title', 'Video')}",
        duration=request.script.get("total_duration", 60),
        style="scripted",
        format=request.format,
        voice_id=voice_id,
        voice_provider=request.voice_provider,
        include_music=True,
        music_style=request.script.get("music_mood", "cinematic"),
        prefer_ai_images=request.prefer_ai_images,
        caption_style=request.caption_style,
        caption_config=request.caption_config,
        # Lip-sync options
        enable_lipsync=request.enable_lipsync,
        avatar_id=request.avatar_id,
        lipsync_expression=request.lipsync_expression,
        # Body motion options
        enable_body_motion=request.enable_body_motion,
        use_presenter=request.use_presenter,
        # Face Swap options
        face_swap_image=request.face_swap_image,
        face_swap_hair_source=request.face_swap_hair_source,
        # PIP options
        pip_position=request.pip_position,
        pip_size=request.pip_size,
        pip_shadow=request.pip_shadow,
        pip_remove_background=request.pip_remove_background,
        # Quality/Cost optimization
        avatar_quality=request.avatar_quality
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
                heygen_api_key=os.getenv("HEYGEN_API_KEY", ""),
                elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
                replicate_api_key=os.getenv("REPLICATE_API_KEY", "")
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

# --- Voice Endpoints ---

@app.get("/api/v1/media/voices")
async def get_available_voices_for_avatars(
    language: str = "en",
    provider: str = "elevenlabs",
    gender: Optional[str] = None
):
    """
    Get available voices for avatar video generation.
    Supports multiple languages and providers.
    """
    from services.voice_service import get_voice_service

    voice_service = get_voice_service()
    voices = voice_service.get_available_voices(
        language=language,
        provider=provider,
        gender=gender
    )

    return {
        "voices": [
            {
                "id": v.id,
                "name": v.name,
                "provider": v.provider,
                "gender": v.gender,
                "language": v.language,
                "style": v.style,
                "description": v.description
            }
            for v in voices
        ],
        "supported_languages": voice_service.get_supported_languages(),
        "current_language": language,
        "current_provider": provider
    }

@app.get("/api/v1/media/voices/for-avatar/{avatar_id}")
async def get_voice_for_avatar(
    avatar_id: str,
    language: str = "en",
    provider: str = "elevenlabs"
):
    """
    Get the recommended voice for a specific avatar based on gender matching.
    """
    from services.voice_service import get_voice_service

    voice_service = get_voice_service()
    voice_id = voice_service.get_voice_for_avatar(
        avatar_id=avatar_id,
        language=language,
        provider=provider
    )
    gender = voice_service.get_avatar_gender(avatar_id)

    return {
        "avatar_id": avatar_id,
        "recommended_voice_id": voice_id,
        "avatar_gender": gender,
        "language": language,
        "provider": provider
    }

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
# Face Swap Endpoints
# ========================================

class FaceSwapRequest(BaseModel):
    """Request for face swap operation."""
    target_image: str = Field(..., description="Avatar/target image URL, path, or base64")
    swap_image: str = Field(..., description="User's face image URL, path, or base64")
    gender: str = Field(default="auto", description="Gender: 'male', 'female', or 'auto'")
    hair_source: str = Field(default="user", description="Hair source: 'user' or 'target'")
    upscale: bool = Field(default=True, description="Apply 2x upscale for better quality")

class FaceSwapResponse(BaseModel):
    """Response from face swap operation."""
    success: bool
    image_url: Optional[str] = None
    error_message: Optional[str] = None

class CreateFaceSwapAvatarRequest(BaseModel):
    """Request to create a custom avatar using face swap."""
    base_avatar_id: str = Field(..., description="Base avatar ID from gallery (e.g., 'avatar-pro-male-alex')")
    face_image: str = Field(..., description="User's face image URL, path, or base64")
    avatar_name: str = Field(..., min_length=1, max_length=100, description="Name for the custom avatar")
    gender: str = Field(default="auto", description="Gender: 'male', 'female', or 'auto'")
    hair_source: str = Field(default="user", description="Hair source: 'user' or 'target'")

class CreateFaceSwapAvatarResponse(BaseModel):
    """Response from custom avatar creation."""
    success: bool
    custom_avatar_id: Optional[str] = None
    preview_url: Optional[str] = None
    avatar_name: Optional[str] = None
    error_message: Optional[str] = None

@app.post("/api/v1/media/face-swap", response_model=FaceSwapResponse)
async def swap_face(request: FaceSwapRequest):
    """
    Swap a face onto a target image.

    Uses Replicate's easel/advanced-face-swap model for high-quality results.
    Cost: ~$0.014 per swap
    """
    from services.face_swap_service import get_face_swap_service, HairSource

    face_swap = get_face_swap_service()

    if not face_swap.is_available():
        raise HTTPException(
            status_code=503,
            detail="Face swap service not available. Check REPLICATE_API_KEY."
        )

    try:
        hair_source = HairSource.USER if request.hair_source == "user" else HairSource.TARGET

        result_path = await face_swap.swap_face(
            target_image=request.target_image,
            swap_image=request.swap_image,
            gender=request.gender,
            hair_source=hair_source,
            upscale=request.upscale
        )

        if result_path:
            return FaceSwapResponse(
                success=True,
                image_url=result_path
            )
        else:
            return FaceSwapResponse(
                success=False,
                error_message="Face swap failed. Check logs for details."
            )

    except Exception as e:
        return FaceSwapResponse(
            success=False,
            error_message=str(e)
        )

@app.post("/api/v1/media/avatars/create-from-faceswap", response_model=CreateFaceSwapAvatarResponse)
async def create_avatar_from_faceswap(
    request: CreateFaceSwapAvatarRequest,
    user_id: str = "demo-user"
):
    """
    Create a reusable custom avatar by face-swapping user's face onto a base avatar.

    This allows users to:
    1. Choose a base avatar (body/pose template)
    2. Swap their own face onto it
    3. Save it as a custom avatar for future videos

    Cost: ~$0.014 per face swap
    """
    from services.face_swap_service import get_face_swap_service, HairSource
    import json

    face_swap = get_face_swap_service()

    if not face_swap.is_available():
        raise HTTPException(
            status_code=503,
            detail="Face swap service not available. Check REPLICATE_API_KEY."
        )

    # Load avatar config to get base avatar image
    config_path = Path(__file__).parent / "config" / "avatars.json"
    try:
        with open(config_path, "r") as f:
            avatars_config = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load avatars config: {e}")

    # Find base avatar
    base_avatar = None
    for avatar in avatars_config.get("avatars", []):
        if avatar["id"] == request.base_avatar_id:
            base_avatar = avatar
            break

    if not base_avatar:
        raise HTTPException(
            status_code=404,
            detail=f"Base avatar '{request.base_avatar_id}' not found"
        )

    # Get base avatar image URL
    base_image_url = base_avatar.get("did_presenter_id") or base_avatar.get("preview_url")

    if not base_image_url:
        raise HTTPException(
            status_code=400,
            detail=f"Base avatar '{request.base_avatar_id}' has no image URL"
        )

    try:
        hair_source = HairSource.USER if request.hair_source == "user" else HairSource.TARGET

        result = await face_swap.create_custom_avatar(
            base_avatar_image=base_image_url,
            user_face_image=request.face_image,
            avatar_name=request.avatar_name,
            user_id=user_id,
            gender=request.gender,
            hair_source=hair_source
        )

        if result:
            print(f"[API] Custom avatar created: {result['custom_avatar_id']} for user {user_id}", flush=True)
            return CreateFaceSwapAvatarResponse(
                success=True,
                custom_avatar_id=result["custom_avatar_id"],
                preview_url=result["image_path"],
                avatar_name=result["name"]
            )
        else:
            return CreateFaceSwapAvatarResponse(
                success=False,
                error_message="Failed to create custom avatar. Check logs for details."
            )

    except Exception as e:
        return CreateFaceSwapAvatarResponse(
            success=False,
            error_message=str(e)
        )


# ========================================
# Video Editor API - Phase 3
# ========================================

from fastapi import UploadFile, File, Form
from models.video_editor_models import (
    VideoProject,
    VideoSegment,
    SegmentType,
    SegmentStatus,
    ProjectStatus,
    TextOverlay,
    ImageOverlay,
    CreateProjectRequest,
    CreateProjectResponse,
    AddSegmentRequest,
    UpdateSegmentRequest,
    ReorderSegmentsRequest,
    RenderProjectRequest,
    ProjectListResponse,
    UploadSegmentResponse,
)

# Lazy load video editor services
_timeline_service = None
_segment_manager = None
_video_merge_service = None


def get_timeline_service():
    global _timeline_service
    if _timeline_service is None:
        from services.timeline_service import TimelineService
        _timeline_service = TimelineService()
    return _timeline_service


def get_segment_manager():
    global _segment_manager
    if _segment_manager is None:
        from services.segment_manager import SegmentManager
        _segment_manager = SegmentManager()
    return _segment_manager


def get_video_merge_service():
    global _video_merge_service
    if _video_merge_service is None:
        from services.video_merge_service import VideoMergeService
        _video_merge_service = VideoMergeService()
    return _video_merge_service


# ------------------------------------
# Project Endpoints
# ------------------------------------

@app.post("/api/v1/editor/projects", response_model=CreateProjectResponse)
async def create_editor_project(
    request: CreateProjectRequest,
):
    """
    Create a new video editing project.
    Can optionally import videos from a completed course generation.
    """
    timeline = get_timeline_service()

    # If importing from course, fetch course videos
    course_videos = None
    if request.course_job_id and request.import_course_videos:
        # TODO: Fetch videos from course-generator service
        # For now, this will create an empty project
        pass

    project = await timeline.create_project(request, course_videos)

    return CreateProjectResponse(
        project_id=project.id,
        title=project.title,
        status=project.status,
        segment_count=len(project.segments),
        message=f"Project created with {len(project.segments)} segments"
    )


@app.get("/api/v1/editor/projects", response_model=ProjectListResponse)
async def list_editor_projects(
    user_id: str = "demo-user",
    page: int = 1,
    page_size: int = 20,
):
    """List all video editing projects for a user"""
    timeline = get_timeline_service()
    projects, total = await timeline.list_projects(user_id, page, page_size)

    return ProjectListResponse(
        projects=projects,
        total=total,
        page=page,
        page_size=page_size
    )


@app.get("/api/v1/editor/projects/{project_id}")
async def get_editor_project(
    project_id: str,
    user_id: str = "demo-user",
):
    """Get a video editing project with all segments"""
    timeline = get_timeline_service()
    project = await timeline.get_project(project_id, user_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return project


@app.delete("/api/v1/editor/projects/{project_id}")
async def delete_editor_project(
    project_id: str,
    user_id: str = "demo-user",
):
    """Delete a video editing project"""
    timeline = get_timeline_service()
    deleted = await timeline.delete_project(project_id, user_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"message": "Project deleted", "project_id": project_id}


@app.patch("/api/v1/editor/projects/{project_id}/settings")
async def update_project_settings(
    project_id: str,
    settings: Dict[str, Any],
    user_id: str = "demo-user",
):
    """Update project output settings (resolution, fps, quality, etc.)"""
    timeline = get_timeline_service()
    project = await timeline.update_project_settings(project_id, user_id, settings)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return project


# ------------------------------------
# Segment Endpoints
# ------------------------------------

@app.post("/api/v1/editor/projects/{project_id}/segments")
async def add_segment(
    project_id: str,
    request: AddSegmentRequest,
    user_id: str = "demo-user",
):
    """Add a new segment to the timeline"""
    timeline = get_timeline_service()
    segment = await timeline.add_segment(project_id, user_id, request)

    if not segment:
        raise HTTPException(status_code=404, detail="Project not found")

    return segment


@app.post("/api/v1/editor/projects/{project_id}/segments/upload", response_model=UploadSegmentResponse)
async def upload_segment(
    project_id: str,
    file: UploadFile = File(...),
    insert_after_segment_id: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    user_id: str = Form("demo-user"),
):
    """
    Upload a user video/audio/image as a new segment.

    Supported formats:
    - Video: mp4, mov, avi, mkv, webm, m4v
    - Audio: mp3, wav, aac, m4a, ogg
    - Image: jpg, jpeg, png, gif, webp
    """
    segment_manager = get_segment_manager()
    timeline = get_timeline_service()

    # Read file content
    content = await file.read()

    # Process upload
    try:
        source_url, duration, thumbnail_url, segment_type = await segment_manager.process_upload(
            content, file.filename, project_id, user_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Add segment to timeline
    segment_request = AddSegmentRequest(
        segment_type=segment_type,
        source_url=source_url,
        insert_after_segment_id=insert_after_segment_id,
        title=title or file.filename,
        slide_duration=duration if segment_type == SegmentType.SLIDE else 5.0,
    )

    segment = await timeline.add_segment(project_id, user_id, segment_request)

    if not segment:
        raise HTTPException(status_code=404, detail="Project not found")

    # Update segment with actual duration and thumbnail
    await timeline.update_segment_duration(
        project_id, segment.id, duration, thumbnail_url
    )

    return UploadSegmentResponse(
        segment_id=segment.id,
        status=SegmentStatus.READY,
        source_url=source_url,
        duration=duration,
        thumbnail_url=thumbnail_url,
        message=f"Uploaded {file.filename} ({duration:.1f}s)"
    )


@app.patch("/api/v1/editor/projects/{project_id}/segments/{segment_id}")
async def update_segment(
    project_id: str,
    segment_id: str,
    request: UpdateSegmentRequest,
    user_id: str = "demo-user",
):
    """Update segment properties (trim, volume, transitions, etc.)"""
    timeline = get_timeline_service()
    segment = await timeline.update_segment(project_id, segment_id, user_id, request)

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    return segment


@app.delete("/api/v1/editor/projects/{project_id}/segments/{segment_id}")
async def remove_segment(
    project_id: str,
    segment_id: str,
    user_id: str = "demo-user",
):
    """Remove a segment from the timeline"""
    timeline = get_timeline_service()
    segment_manager = get_segment_manager()

    # Get segment to cleanup files
    project = await timeline.get_project(project_id, user_id)
    if project:
        for seg in project.segments:
            if seg.id == segment_id and seg.segment_type == SegmentType.USER_VIDEO:
                await segment_manager.delete_segment_files(seg.source_url, seg.thumbnail_url)
                break

    removed = await timeline.remove_segment(project_id, segment_id, user_id)

    if not removed:
        raise HTTPException(status_code=404, detail="Segment not found")

    return {"message": "Segment removed", "segment_id": segment_id}


@app.post("/api/v1/editor/projects/{project_id}/segments/reorder")
async def reorder_segments(
    project_id: str,
    request: ReorderSegmentsRequest,
    user_id: str = "demo-user",
):
    """Reorder segments in the timeline"""
    timeline = get_timeline_service()
    success = await timeline.reorder_segments(project_id, user_id, request.segment_ids)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to reorder segments")

    project = await timeline.get_project(project_id, user_id)
    return project


@app.post("/api/v1/editor/projects/{project_id}/segments/{segment_id}/split")
async def split_segment(
    project_id: str,
    segment_id: str,
    split_time: float,
    user_id: str = "demo-user",
):
    """Split a segment at a specific time"""
    timeline = get_timeline_service()
    result = await timeline.split_segment(project_id, segment_id, user_id, split_time)

    if not result:
        raise HTTPException(status_code=400, detail="Failed to split segment")

    first_half, second_half = result
    return {
        "message": "Segment split successfully",
        "first_half": first_half,
        "second_half": second_half
    }


# ------------------------------------
# Overlay Endpoints
# ------------------------------------

@app.post("/api/v1/editor/projects/{project_id}/overlays/text")
async def add_text_overlay(
    project_id: str,
    overlay: TextOverlay,
    user_id: str = "demo-user",
):
    """Add a text overlay to the project"""
    timeline = get_timeline_service()
    success = await timeline.add_text_overlay(project_id, user_id, overlay)

    if not success:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"message": "Text overlay added", "overlay_id": overlay.id}


@app.post("/api/v1/editor/projects/{project_id}/overlays/image")
async def add_image_overlay(
    project_id: str,
    overlay: ImageOverlay,
    user_id: str = "demo-user",
):
    """Add an image overlay (logo, watermark) to the project"""
    timeline = get_timeline_service()
    success = await timeline.add_image_overlay(project_id, user_id, overlay)

    if not success:
        raise HTTPException(status_code=404, detail="Project not found")

    return {"message": "Image overlay added", "overlay_id": overlay.id}


# ------------------------------------
# Render Endpoints
# ------------------------------------

# Render job storage
render_jobs_db: Dict[str, Dict] = {}


@app.post("/api/v1/editor/projects/{project_id}/render")
async def start_render(
    project_id: str,
    request: RenderProjectRequest,
    background_tasks: BackgroundTasks,
    user_id: str = "demo-user",
):
    """
    Start rendering the final video.
    Returns a job ID to track progress.
    """
    timeline = get_timeline_service()
    project = await timeline.get_project(project_id, user_id)

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.segments:
        raise HTTPException(status_code=400, detail="No segments to render")

    # Apply render overrides
    if request.output_resolution:
        project.output_resolution = request.output_resolution
    if request.output_fps:
        project.output_fps = request.output_fps
    if request.output_quality:
        project.output_quality = request.output_quality

    # Add watermark if requested
    if request.include_watermark and request.watermark_url:
        watermark = ImageOverlay(
            image_url=request.watermark_url,
            position_x=0.95,
            position_y=0.05,
            scale=0.1,
            opacity=0.7
        )
        project.image_overlays.append(watermark)

    # Create render job
    job_id = str(uuid.uuid4())
    render_jobs_db[job_id] = {
        "job_id": job_id,
        "project_id": project_id,
        "user_id": user_id,
        "status": "pending",
        "progress": 0,
        "message": "Queued for rendering",
        "created_at": datetime.utcnow().isoformat(),
        "output_url": None,
        "error": None
    }

    # Start render in background
    background_tasks.add_task(process_render_job, job_id, project)

    return {
        "job_id": job_id,
        "project_id": project_id,
        "status": "pending",
        "message": "Render job started"
    }


async def process_render_job(job_id: str, project: VideoProject):
    """Background task to render the video project"""
    merge_service = get_video_merge_service()
    timeline = get_timeline_service()

    def progress_callback(progress: float, message: str):
        render_jobs_db[job_id]["progress"] = progress
        render_jobs_db[job_id]["message"] = message

    try:
        render_jobs_db[job_id]["status"] = "processing"
        render_jobs_db[job_id]["message"] = "Starting render..."

        # Update project status
        project.status = ProjectStatus.RENDERING
        await timeline.repository.save(project)

        # Render the video
        output_path = await merge_service.render_project(project, progress_callback)

        # Generate URL for the output
        output_url = f"{settings.SERVICE_BASE_URL}/api/v1/editor/videos/{Path(output_path).name}"

        render_jobs_db[job_id]["status"] = "completed"
        render_jobs_db[job_id]["progress"] = 100
        render_jobs_db[job_id]["message"] = "Render complete"
        render_jobs_db[job_id]["output_url"] = output_url
        render_jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()

        # Update project
        project.status = ProjectStatus.COMPLETED
        project.output_url = output_url
        project.rendered_at = datetime.utcnow()
        await timeline.repository.save(project)

    except Exception as e:
        print(f"[RENDER] Error: {e}", flush=True)
        render_jobs_db[job_id]["status"] = "failed"
        render_jobs_db[job_id]["error"] = str(e)
        render_jobs_db[job_id]["message"] = f"Render failed: {str(e)}"

        project.status = ProjectStatus.FAILED
        await timeline.repository.save(project)


@app.get("/api/v1/editor/render-jobs/{job_id}")
async def get_render_status(job_id: str):
    """Get the status of a render job"""
    if job_id not in render_jobs_db:
        raise HTTPException(status_code=404, detail="Render job not found")

    return render_jobs_db[job_id]


@app.post("/api/v1/editor/projects/{project_id}/preview")
async def create_preview(
    project_id: str,
    start_time: float = 0.0,
    duration: float = 10.0,
    user_id: str = "demo-user",
):
    """
    Create a short preview of the project at a specific timestamp.
    Useful for reviewing edits before full render.
    """
    timeline = get_timeline_service()
    merge_service = get_video_merge_service()

    project = await timeline.get_project(project_id, user_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        preview_path = await merge_service.create_preview(project, start_time, duration)
        preview_url = f"{settings.SERVICE_BASE_URL}/api/v1/editor/videos/{Path(preview_path).name}"

        return {
            "preview_url": preview_url,
            "start_time": start_time,
            "duration": duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


# ------------------------------------
# Video File Serving
# ------------------------------------

@app.get("/api/v1/editor/videos/{filename}")
async def serve_editor_video(filename: str):
    """Serve rendered video files"""
    video_path = Path("/tmp/viralify/editor/output") / filename

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=filename
    )


@app.get("/api/v1/editor/thumbnails/{filename}")
async def serve_editor_thumbnail(filename: str):
    """Serve segment thumbnail images"""
    thumb_path = Path("/tmp/viralify/editor/thumbnails") / filename

    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(
        path=str(thumb_path),
        media_type="image/jpeg",
        filename=filename
    )


# ------------------------------------
# Utility Endpoints
# ------------------------------------

@app.get("/api/v1/editor/supported-formats")
async def get_supported_formats():
    """Get supported file formats and size limits"""
    segment_manager = get_segment_manager()
    return segment_manager.get_supported_formats()


# ========================================
# Voice Cloning API - Phase 4
# ========================================

from models.voice_cloning_models import (
    VoiceProfile,
    VoiceSample,
    VoiceProfileStatus,
    SampleStatus,
    VoiceGenerationSettings,
    CreateVoiceProfileRequest,
    CreateVoiceProfileResponse,
    UploadSampleResponse,
    StartTrainingRequest,
    StartTrainingResponse,
    GenerateClonedSpeechRequest,
    GenerateClonedSpeechResponse,
    VoiceProfileListResponse,
    VoiceProfileDetailResponse,
    PreviewVoiceRequest,
    VoiceSampleRequirements,
)

# Lazy load voice cloning services
_voice_profile_manager = None


def get_voice_profile_manager():
    global _voice_profile_manager
    if _voice_profile_manager is None:
        from services.voice_profile_manager import VoiceProfileManager
        _voice_profile_manager = VoiceProfileManager()
    return _voice_profile_manager


# ------------------------------------
# Voice Profile Endpoints
# ------------------------------------

@app.post("/api/v1/voice/profiles", response_model=CreateVoiceProfileResponse)
async def create_voice_profile(
    request: CreateVoiceProfileRequest,
    user_id: str = "demo-user",
):
    """
    Create a new voice cloning profile.
    Start here to begin the voice cloning process.
    """
    manager = get_voice_profile_manager()
    profile = await manager.create_profile(request, user_id)

    requirements = manager.get_training_requirements(profile)

    return CreateVoiceProfileResponse(
        profile_id=profile.id,
        name=profile.name,
        status=profile.status,
        message="Profile created. Upload voice samples to continue.",
        min_samples_required=requirements["min_samples"],
        min_duration_seconds=requirements["min_duration_seconds"],
    )


@app.get("/api/v1/voice/profiles", response_model=VoiceProfileListResponse)
async def list_voice_profiles(user_id: str = "demo-user"):
    """List all voice profiles for a user"""
    manager = get_voice_profile_manager()
    profiles = await manager.list_profiles(user_id)

    return VoiceProfileListResponse(
        profiles=profiles,
        total=len(profiles),
    )


@app.get("/api/v1/voice/profiles/{profile_id}")
async def get_voice_profile(
    profile_id: str,
    user_id: str = "demo-user",
):
    """Get detailed voice profile information"""
    manager = get_voice_profile_manager()
    profile = await manager.get_profile(profile_id, user_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    requirements = manager.get_training_requirements(profile)

    return VoiceProfileDetailResponse(
        profile=profile,
        samples=profile.samples,
        can_train=requirements.get("can_train", False),
        training_requirements=requirements,
    )


@app.delete("/api/v1/voice/profiles/{profile_id}")
async def delete_voice_profile(
    profile_id: str,
    user_id: str = "demo-user",
):
    """Delete a voice profile and all associated data"""
    manager = get_voice_profile_manager()
    deleted = await manager.delete_profile(profile_id, user_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {"message": "Profile deleted", "profile_id": profile_id}


@app.patch("/api/v1/voice/profiles/{profile_id}")
async def update_voice_profile(
    profile_id: str,
    updates: Dict[str, Any],
    user_id: str = "demo-user",
):
    """Update voice profile settings"""
    manager = get_voice_profile_manager()
    profile = await manager.update_profile(profile_id, user_id, updates)

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return profile


# ------------------------------------
# Voice Sample Endpoints
# ------------------------------------

@app.post("/api/v1/voice/profiles/{profile_id}/samples", response_model=UploadSampleResponse)
async def upload_voice_sample(
    profile_id: str,
    file: UploadFile = File(...),
    user_id: str = Form("demo-user"),
):
    """
    Upload a voice sample for cloning.

    Requirements:
    - Format: MP3, WAV, M4A, OGG, WEBM, FLAC, AAC
    - Duration: 5-300 seconds per sample
    - Total: At least 30 seconds across all samples
    - Quality: Record in quiet environment with good microphone
    """
    manager = get_voice_profile_manager()

    # Read file content
    content = await file.read()

    # Add sample to profile
    sample, message = await manager.add_sample(
        profile_id, user_id, content, file.filename
    )

    if not sample:
        raise HTTPException(status_code=400, detail=message)

    # Get updated profile for stats
    profile = await manager.get_profile(profile_id, user_id)
    requirements = manager.get_training_requirements(profile)

    return UploadSampleResponse(
        sample_id=sample.id,
        profile_id=profile_id,
        duration_seconds=sample.duration_seconds,
        quality_score=sample.quality_score,
        status=sample.status,
        message=message,
        total_duration=profile.total_sample_duration if profile else 0,
        can_start_training=requirements.get("can_train", False),
    )


@app.delete("/api/v1/voice/profiles/{profile_id}/samples/{sample_id}")
async def delete_voice_sample(
    profile_id: str,
    sample_id: str,
    user_id: str = "demo-user",
):
    """Delete a voice sample"""
    manager = get_voice_profile_manager()
    deleted = await manager.remove_sample(profile_id, sample_id, user_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Sample not found")

    return {"message": "Sample deleted", "sample_id": sample_id}


# ------------------------------------
# Training Endpoints
# ------------------------------------

@app.post("/api/v1/voice/profiles/{profile_id}/train", response_model=StartTrainingResponse)
async def start_voice_training(
    profile_id: str,
    request: StartTrainingRequest,
    user_id: str = "demo-user",
):
    """
    Start voice cloning training.

    IMPORTANT: You must confirm that you own this voice or have explicit
    permission from the voice owner to create a cloned voice.
    """
    if request.profile_id != profile_id:
        raise HTTPException(status_code=400, detail="Profile ID mismatch")

    manager = get_voice_profile_manager()

    success, message = await manager.start_training(
        profile_id,
        user_id,
        request.consent_confirmed,
        ip_address=None,  # Would get from request in production
    )

    profile = await manager.get_profile(profile_id, user_id)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return StartTrainingResponse(
        profile_id=profile_id,
        status=profile.status if profile else VoiceProfileStatus.FAILED,
        estimated_time_seconds=30,  # ElevenLabs instant cloning is fast
        message=message,
    )


@app.get("/api/v1/voice/profiles/{profile_id}/training-status")
async def get_training_status(
    profile_id: str,
    user_id: str = "demo-user",
):
    """Get the training status of a voice profile"""
    manager = get_voice_profile_manager()
    profile = await manager.get_profile(profile_id, user_id)

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {
        "profile_id": profile_id,
        "status": profile.status,
        "progress": profile.training_progress,
        "error_message": profile.error_message,
        "trained_at": profile.trained_at.isoformat() if profile.trained_at else None,
    }


# ------------------------------------
# Speech Generation Endpoints
# ------------------------------------

@app.post("/api/v1/voice/profiles/{profile_id}/generate", response_model=GenerateClonedSpeechResponse)
async def generate_cloned_speech(
    profile_id: str,
    request: GenerateClonedSpeechRequest,
    user_id: str = "demo-user",
):
    """
    Generate speech using a cloned voice.

    The voice profile must be in 'ready' status after training.
    """
    if request.profile_id != profile_id:
        raise HTTPException(status_code=400, detail="Profile ID mismatch")

    manager = get_voice_profile_manager()

    audio_path, message = await manager.generate_speech(
        profile_id,
        user_id,
        request.text,
        request.settings,
    )

    if not audio_path:
        raise HTTPException(status_code=400, detail=message)

    # Generate URL
    audio_filename = Path(audio_path).name
    audio_url = f"{settings.SERVICE_BASE_URL}/api/v1/voice/audio/{audio_filename}"

    # Get duration
    from services.voice_cloning_service import get_voice_cloning_service
    cloning_service = get_voice_cloning_service()
    duration = await cloning_service._get_audio_duration(Path(audio_path))

    return GenerateClonedSpeechResponse(
        audio_url=audio_url,
        duration_seconds=duration,
        characters_used=len(request.text),
        profile_id=profile_id,
    )


@app.post("/api/v1/voice/profiles/{profile_id}/preview")
async def preview_cloned_voice(
    profile_id: str,
    request: Optional[PreviewVoiceRequest] = None,
    user_id: str = "demo-user",
):
    """
    Generate a short preview of the cloned voice.
    Uses a default or custom sample text.
    """
    manager = get_voice_profile_manager()

    text = request.text if request else "Hello! This is a preview of my cloned voice. How does it sound?"

    audio_path, message = await manager.preview_voice(profile_id, user_id, text)

    if not audio_path:
        raise HTTPException(status_code=400, detail=message)

    audio_filename = Path(audio_path).name
    audio_url = f"{settings.SERVICE_BASE_URL}/api/v1/voice/audio/{audio_filename}"

    return {
        "audio_url": audio_url,
        "text": text,
        "message": "Preview generated",
    }


# ------------------------------------
# Voice Audio Serving
# ------------------------------------

@app.get("/api/v1/voice/audio/{filename}")
async def serve_cloned_audio(filename: str):
    """Serve generated cloned voice audio files"""
    audio_path = Path("/tmp/viralify/cloned_voices") / filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=filename,
    )


# ------------------------------------
# Voice Requirements & Info
# ------------------------------------

@app.get("/api/v1/voice/requirements", response_model=VoiceSampleRequirements)
async def get_voice_requirements():
    """Get requirements for voice sample uploads"""
    from services.voice_sample_service import get_voice_sample_service
    sample_service = get_voice_sample_service()
    return sample_service.get_requirements()


@app.get("/api/v1/voice/usage")
async def get_voice_usage():
    """Get voice cloning API usage statistics"""
    from services.voice_cloning_service import get_voice_cloning_service
    cloning_service = get_voice_cloning_service()

    if not cloning_service.is_available():
        return {"error": "Voice cloning service not configured"}

    return await cloning_service.get_usage_stats()


# ========================================
# Hybrid TTS Service Endpoints
# ========================================

class HybridTTSRequest(BaseModel):
    """Request for hybrid TTS generation"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    language: str = Field(default="en", description="Language code (en, fr, es, de, etc.)")
    voice_id: Optional[str] = Field(default=None, description="Specific voice ID")
    voice_gender: str = Field(default="neutral", description="Voice gender: male, female, neutral")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    quality: str = Field(default="standard", description="Quality level: draft, standard, premium")
    provider: Optional[str] = Field(default=None, description="Force specific provider: kokoro, chatterbox, elevenlabs, openai")
    prefer_self_hosted: bool = Field(default=True, description="Prefer self-hosted providers over API")


class HybridTTSWithCloningRequest(BaseModel):
    """Request for TTS with voice cloning"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    language: str = Field(default="en", description="Language code")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")


@app.get("/api/v1/tts/providers")
async def get_tts_providers():
    """Get information about available TTS providers"""
    try:
        from services.tts_providers import get_tts_service
        service = get_tts_service()
        await service.initialize()
        return service.get_provider_info()
    except Exception as e:
        return {
            "error": str(e),
            "available_providers": [],
            "fallback_mode": True
        }


@app.get("/api/v1/tts/voices")
async def get_tts_voices(language: Optional[str] = None):
    """Get all available voices from all TTS providers"""
    try:
        from services.tts_providers import get_tts_service
        service = get_tts_service()
        await service.initialize()
        voices = service.get_all_voices(language)
        return {
            "voices": [
                {
                    "id": v.voice_id,
                    "name": v.name,
                    "provider": v.provider.value,
                    "language": v.language,
                    "gender": v.gender.value,
                    "supports_cloning": v.supports_cloning,
                    "description": v.description,
                }
                for v in voices
            ],
            "supported_languages": service.get_supported_languages(),
        }
    except Exception as e:
        # Fallback to existing voices endpoint
        return await get_available_voices()


@app.post("/api/v1/tts/generate")
async def generate_hybrid_tts(request: HybridTTSRequest):
    """
    Generate TTS audio using the hybrid provider system.

    Automatically selects the best provider based on:
    - Language (multilingual support)
    - Quality level (draft/standard/premium)
    - Voice cloning requirements
    - Provider availability

    Routing logic:
    - draft: Kokoro (fastest)
    - standard: Kokoro → Chatterbox (if GPU available)
    - premium: Chatterbox → ElevenLabs
    - voice_cloning: Chatterbox → ElevenLabs
    """
    try:
        from services.tts_providers import get_tts_service
        from services.tts_providers.base_provider import VoiceGender, TTSProviderType
        from services.tts_providers.provider_service import TTSQuality

        service = get_tts_service()

        # Map string values to enums
        gender = VoiceGender(request.voice_gender) if request.voice_gender in ["male", "female", "neutral"] else VoiceGender.NEUTRAL
        quality = TTSQuality(request.quality) if request.quality in ["draft", "standard", "premium"] else TTSQuality.STANDARD
        preferred_provider = TTSProviderType(request.provider) if request.provider else None

        result = await service.generate(
            text=request.text,
            language=request.language,
            voice_id=request.voice_id,
            voice_gender=gender,
            speed=request.speed,
            quality=quality,
            preferred_provider=preferred_provider,
            prefer_self_hosted=request.prefer_self_hosted,
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        # Save to file and return URL
        from pathlib import Path
        import uuid

        output_dir = Path("/tmp/viralify/tts")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{uuid.uuid4()}.mp3"
        output_path = output_dir / filename

        with open(output_path, "wb") as f:
            f.write(result.audio_data)

        return {
            "success": True,
            "audio_url": f"{settings.SERVICE_BASE_URL}/api/v1/tts/audio/{filename}",
            "duration_seconds": result.duration_seconds,
            "provider_used": result.provider_used.value if result.provider_used else "unknown",
            "metadata": result.metadata,
        }

    except ImportError:
        # Fallback to existing voiceover generation
        from services.tts_providers.elevenlabs_provider import ElevenLabsProvider
        from services.tts_providers.openai_provider import OpenAIProvider
        from services.tts_providers.base_provider import TTSConfig, VoiceGender

        # Try ElevenLabs first, then OpenAI
        providers = [ElevenLabsProvider(), OpenAIProvider()]

        config = TTSConfig(
            text=request.text,
            language=request.language,
            voice_gender=VoiceGender(request.voice_gender) if request.voice_gender in ["male", "female", "neutral"] else VoiceGender.NEUTRAL,
            speed=request.speed,
        )

        for provider in providers:
            if await provider.is_available():
                result = await provider.generate(config)
                if result.success:
                    from pathlib import Path
                    import uuid

                    output_dir = Path("/tmp/viralify/tts")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    filename = f"{uuid.uuid4()}.mp3"
                    output_path = output_dir / filename

                    with open(output_path, "wb") as f:
                        f.write(result.audio_data)

                    return {
                        "success": True,
                        "audio_url": f"{settings.SERVICE_BASE_URL}/api/v1/tts/audio/{filename}",
                        "duration_seconds": result.duration_seconds,
                        "provider_used": result.provider_used.value if result.provider_used else "fallback",
                    }

        raise HTTPException(status_code=500, detail="No TTS providers available")


@app.post("/api/v1/tts/generate-with-cloning")
async def generate_tts_with_cloning(
    text: str = Form(...),
    language: str = Form("en"),
    speed: float = Form(1.0),
    audio_file: UploadFile = File(...),
):
    """
    Generate TTS with voice cloning from an uploaded audio sample.

    Requires GPU with Chatterbox installed, or falls back to ElevenLabs API.
    """
    try:
        from services.tts_providers import get_tts_service
        from services.tts_providers.provider_service import TTSQuality

        service = get_tts_service()

        # Read uploaded audio
        audio_bytes = await audio_file.read()

        result = await service.generate(
            text=text,
            language=language,
            speed=speed,
            quality=TTSQuality.PREMIUM,
            clone_audio_bytes=audio_bytes,
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        # Save to file and return URL
        from pathlib import Path
        import uuid

        output_dir = Path("/tmp/viralify/tts")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"cloned_{uuid.uuid4()}.mp3"
        output_path = output_dir / filename

        with open(output_path, "wb") as f:
            f.write(result.audio_data)

        return {
            "success": True,
            "audio_url": f"{settings.SERVICE_BASE_URL}/api/v1/tts/audio/{filename}",
            "duration_seconds": result.duration_seconds,
            "provider_used": result.provider_used.value if result.provider_used else "unknown",
            "voice_cloning": True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tts/audio/{filename}")
async def serve_tts_audio(filename: str):
    """Serve generated TTS audio files"""
    audio_path = Path("/tmp/viralify/tts") / filename

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg",
        filename=filename,
    )


# ========================================
# Run Server
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
