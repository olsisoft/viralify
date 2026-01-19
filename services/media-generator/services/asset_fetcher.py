"""
Multi-Source Asset Fetcher Service
Fetches videos and images from multiple sources:
- Pexels (videos + images)
- Unsplash (high-quality images)
- Pixabay (videos + images)
- DALL-E 3 (AI-generated images)
- Technical Diagrams (Mermaid, Python Diagrams, Graphviz)
"""

import asyncio
import httpx
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel
from enum import Enum
import os

from services.diagram_generator import DiagramGenerator, DiagramType


class MediaType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"


class FetchedAsset(BaseModel):
    id: str
    source: str  # pexels, unsplash, pixabay, dalle
    media_type: MediaType
    url: str  # Full quality URL
    preview_url: str  # Thumbnail/preview
    width: int
    height: int
    duration: Optional[float] = None  # For videos
    author: str
    quality_score: float = 0.0  # 0-1, for ranking


class AssetFetcherService:
    """Fetches assets from multiple sources and ranks them"""

    def __init__(
        self,
        pexels_api_key: str,
        unsplash_api_key: str = "",
        pixabay_api_key: str = "",
        openai_api_key: str = ""
    ):
        self.pexels_key = pexels_api_key
        self.unsplash_key = unsplash_api_key
        self.pixabay_key = pixabay_api_key
        self.openai_key = openai_api_key

        # Initialize diagram generator for technical content
        self.diagram_generator = DiagramGenerator(openai_api_key) if openai_api_key else None

    async def search_all_sources(
        self,
        keywords: List[str],
        media_type: MediaType = MediaType.VIDEO,
        orientation: str = "portrait",
        duration_max: int = 30,
        limit_per_source: int = 5
    ) -> List[FetchedAsset]:
        """
        Search all available sources in parallel and return ranked results
        """
        query = " ".join(keywords)

        tasks = []

        # Always search Pexels
        if self.pexels_key:
            if media_type == MediaType.VIDEO:
                tasks.append(self._search_pexels_videos(query, orientation, duration_max, limit_per_source))
            else:
                tasks.append(self._search_pexels_images(query, orientation, limit_per_source))

        # Search Unsplash for images
        if self.unsplash_key and media_type == MediaType.IMAGE:
            tasks.append(self._search_unsplash(query, orientation, limit_per_source))

        # Search Pixabay
        if self.pixabay_key:
            if media_type == MediaType.VIDEO:
                tasks.append(self._search_pixabay_videos(query, orientation, limit_per_source))
            else:
                tasks.append(self._search_pixabay_images(query, orientation, limit_per_source))

        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter errors
        all_assets = []
        for result in results:
            if isinstance(result, list):
                all_assets.extend(result)
            elif isinstance(result, Exception):
                print(f"Search error: {result}")

        # Rank and sort by quality
        ranked_assets = self._rank_assets(all_assets, keywords)
        return ranked_assets

    async def generate_ai_image(
        self,
        prompt: str,
        style: str = "cinematic",
        aspect_ratio: str = "9:16"
    ) -> FetchedAsset:
        """Generate photorealistic image using GPT-4o with fallbacks"""

        if not self.openai_key:
            raise Exception("OpenAI API key not configured")

        # Enhance prompt for photorealistic quality
        enhanced_prompt = f"{prompt}. Style: {style}, photorealistic, real photograph, natural lighting, shot on professional camera, ultra high detail"

        async with httpx.AsyncClient() as client:
            # Try GPT-4o first via chat completions (most photorealistic)
            try:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.openai_key}",
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
                    for choice in data.get("choices", []):
                        message = choice.get("message", {})
                        for content in message.get("content", []):
                            if content.get("type") == "image":
                                image_data = content.get("image", {})
                                if "url" in image_data:
                                    import uuid as uuid_module
                                    return FetchedAsset(
                                        id=f"gpt4o-{uuid_module.uuid4().hex[:8]}",
                                        source="gpt-4o",
                                        media_type=MediaType.IMAGE,
                                        url=image_data["url"],
                                        preview_url=image_data["url"],
                                        width=1024,
                                        height=1536 if aspect_ratio == "9:16" else 1024,
                                        author="GPT-4o",
                                        quality_score=1.0
                                    )
                                elif "b64_json" in image_data or "data" in image_data:
                                    import base64
                                    os.makedirs("/tmp/viralify/images", exist_ok=True)
                                    import uuid as uuid_module
                                    image_path = f"/tmp/viralify/images/{uuid_module.uuid4().hex}.png"
                                    b64_data = image_data.get("b64_json") or image_data.get("data")
                                    with open(image_path, "wb") as f:
                                        f.write(base64.b64decode(b64_data))
                                    return FetchedAsset(
                                        id=f"gpt4o-{uuid_module.uuid4().hex[:8]}",
                                        source="gpt-4o",
                                        media_type=MediaType.IMAGE,
                                        url=image_path,
                                        preview_url=image_path,
                                        width=1024,
                                        height=1536 if aspect_ratio == "9:16" else 1024,
                                        author="GPT-4o",
                                        quality_score=1.0
                                    )
                    print("GPT-4o response didn't contain image, trying gpt-image-1...")
                else:
                    print(f"GPT-4o chat failed ({response.status_code}), trying gpt-image-1...")
            except Exception as e:
                print(f"GPT-4o failed: {e}, trying gpt-image-1...")

            # Try gpt-image-1.5 model (most photorealistic)
            try:
                size_map = {
                    "9:16": "1024x1536",
                    "16:9": "1536x1024",
                    "1:1": "1024x1024"
                }
                size = size_map.get(aspect_ratio, "1024x1536")

                # Try gpt-image-1.5 first, then gpt-image-1
                response = None
                for model in ["gpt-image-1.5", "gpt-image-1"]:
                    try:
                        response = await client.post(
                            "https://api.openai.com/v1/images/generations",
                            headers={
                                "Authorization": f"Bearer {self.openai_key}",
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
                        import uuid as uuid_module
                        image_path = f"/tmp/viralify/images/{uuid_module.uuid4().hex}.png"
                        with open(image_path, "wb") as f:
                            f.write(base64.b64decode(image_data["b64_json"]))
                        width, height = map(int, size.split("x"))
                        return FetchedAsset(
                            id=f"gpt-image-{uuid_module.uuid4().hex[:8]}",
                            source="gpt-image-1",
                            media_type=MediaType.IMAGE,
                            url=image_path,
                            preview_url=image_path,
                            width=width,
                            height=height,
                            author="GPT-Image-1",
                            quality_score=1.0
                        )
                    else:
                        width, height = map(int, size.split("x"))
                        import uuid as uuid_module
                        return FetchedAsset(
                            id=f"gpt-image-{uuid_module.uuid4().hex[:8]}",
                            source="gpt-image-1",
                            media_type=MediaType.IMAGE,
                            url=image_data["url"],
                            preview_url=image_data["url"],
                            width=width,
                            height=height,
                            author="GPT-Image-1",
                            quality_score=1.0
                        )
            except Exception as e:
                print(f"gpt-image-1 failed: {e}, falling back to DALL-E 3...")

            # Final fallback to DALL-E 3
            dalle_size_map = {
                "9:16": "1024x1792",
                "16:9": "1792x1024",
                "1:1": "1024x1024"
            }
            dalle_size = dalle_size_map.get(aspect_ratio, "1024x1792")

            response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.openai_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "dall-e-3",
                    "prompt": enhanced_prompt,
                    "n": 1,
                    "size": dalle_size,
                    "quality": "hd"
                },
                timeout=120.0
            )

            if response.status_code != 200:
                raise Exception(f"Image generation error: {response.text}")

            data = response.json()
            image_url = data["data"][0]["url"]

        width, height = map(int, dalle_size.split("x"))
        import uuid
        return FetchedAsset(
            id=f"dalle-{uuid.uuid4().hex[:8]}",
            source="dalle",
            media_type=MediaType.IMAGE,
            url=image_url,
            preview_url=image_url,
            width=width,
            height=height,
            author="DALL-E 3",
            quality_score=1.0
        )

    async def _search_pexels_videos(
        self,
        query: str,
        orientation: str,
        duration_max: int,
        limit: int
    ) -> List[FetchedAsset]:
        """Search Pexels for videos"""

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": self.pexels_key},
                params={
                    "query": query,
                    "orientation": orientation,
                    "per_page": limit,
                    "size": "medium"
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise Exception(f"Pexels error: {response.text}")

            data = response.json()
            assets = []

            for video in data.get("videos", []):
                if video["duration"] <= duration_max:
                    # Find HD video file
                    video_file = next(
                        (f for f in video["video_files"] if f["quality"] == "hd"),
                        video["video_files"][0] if video["video_files"] else None
                    )

                    if video_file:
                        assets.append(FetchedAsset(
                            id=f"pexels-v-{video['id']}",
                            source="pexels",
                            media_type=MediaType.VIDEO,
                            url=video_file["link"],
                            preview_url=video["image"],
                            width=video_file["width"],
                            height=video_file["height"],
                            duration=video["duration"],
                            author=video["user"]["name"],
                            quality_score=0.8
                        ))

            return assets

    async def _search_pexels_images(
        self,
        query: str,
        orientation: str,
        limit: int
    ) -> List[FetchedAsset]:
        """Search Pexels for images"""

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": self.pexels_key},
                params={
                    "query": query,
                    "orientation": orientation,
                    "per_page": limit
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise Exception(f"Pexels error: {response.text}")

            data = response.json()
            assets = []

            for photo in data.get("photos", []):
                assets.append(FetchedAsset(
                    id=f"pexels-i-{photo['id']}",
                    source="pexels",
                    media_type=MediaType.IMAGE,
                    url=photo["src"]["original"],
                    preview_url=photo["src"]["medium"],
                    width=photo["width"],
                    height=photo["height"],
                    author=photo["photographer"],
                    quality_score=0.75
                ))

            return assets

    async def _search_unsplash(
        self,
        query: str,
        orientation: str,
        limit: int
    ) -> List[FetchedAsset]:
        """Search Unsplash for high-quality images"""

        if not self.unsplash_key:
            return []

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.unsplash.com/search/photos",
                headers={"Authorization": f"Client-ID {self.unsplash_key}"},
                params={
                    "query": query,
                    "orientation": orientation,
                    "per_page": limit
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise Exception(f"Unsplash error: {response.text}")

            data = response.json()
            assets = []

            for photo in data.get("results", []):
                assets.append(FetchedAsset(
                    id=f"unsplash-{photo['id']}",
                    source="unsplash",
                    media_type=MediaType.IMAGE,
                    url=photo["urls"]["full"],
                    preview_url=photo["urls"]["small"],
                    width=photo["width"],
                    height=photo["height"],
                    author=photo["user"]["name"],
                    quality_score=0.9  # Unsplash has very high quality
                ))

            return assets

    async def _search_pixabay_videos(
        self,
        query: str,
        orientation: str,
        limit: int
    ) -> List[FetchedAsset]:
        """Search Pixabay for videos"""

        if not self.pixabay_key:
            return []

        # Map orientation
        video_type = "all"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://pixabay.com/api/videos/",
                params={
                    "key": self.pixabay_key,
                    "q": query,
                    "per_page": limit,
                    "safesearch": "true"
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise Exception(f"Pixabay error: {response.text}")

            data = response.json()
            assets = []

            for video in data.get("hits", []):
                # Get medium quality video
                video_url = video.get("videos", {}).get("medium", {}).get("url", "")
                if video_url:
                    assets.append(FetchedAsset(
                        id=f"pixabay-v-{video['id']}",
                        source="pixabay",
                        media_type=MediaType.VIDEO,
                        url=video_url,
                        preview_url=f"https://i.vimeocdn.com/video/{video['picture_id']}_640x360.jpg",
                        width=video.get("videos", {}).get("medium", {}).get("width", 1920),
                        height=video.get("videos", {}).get("medium", {}).get("height", 1080),
                        duration=video.get("duration", 10),
                        author=video.get("user", "Pixabay"),
                        quality_score=0.7
                    ))

            return assets

    async def _search_pixabay_images(
        self,
        query: str,
        orientation: str,
        limit: int
    ) -> List[FetchedAsset]:
        """Search Pixabay for images"""

        if not self.pixabay_key:
            return []

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://pixabay.com/api/",
                params={
                    "key": self.pixabay_key,
                    "q": query,
                    "orientation": "vertical" if orientation == "portrait" else "horizontal",
                    "per_page": limit,
                    "safesearch": "true",
                    "image_type": "photo"
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise Exception(f"Pixabay error: {response.text}")

            data = response.json()
            assets = []

            for photo in data.get("hits", []):
                assets.append(FetchedAsset(
                    id=f"pixabay-i-{photo['id']}",
                    source="pixabay",
                    media_type=MediaType.IMAGE,
                    url=photo["largeImageURL"],
                    preview_url=photo["previewURL"],
                    width=photo["imageWidth"],
                    height=photo["imageHeight"],
                    author=photo.get("user", "Pixabay"),
                    quality_score=0.7
                ))

            return assets

    def _rank_assets(
        self,
        assets: List[FetchedAsset],
        keywords: List[str]
    ) -> List[FetchedAsset]:
        """Rank assets by quality and relevance"""

        # Simple ranking: sort by quality_score
        # In production, could use ML model for better ranking
        sorted_assets = sorted(assets, key=lambda x: x.quality_score, reverse=True)

        return sorted_assets

    async def fetch_best_asset(
        self,
        keywords: List[str],
        media_type: MediaType,
        orientation: str = "portrait",
        prefer_ai: bool = False,
        visual_description: str = "",
        style: str = "cinematic",
        fallback_to_ai: bool = True,
        aspect_ratio: str = None
    ) -> Optional[FetchedAsset]:
        """
        Fetch the single best asset for given keywords.
        Uses intelligent visual generation based on content type.
        Falls back to DALL-E image generation if no stock footage found.
        """
        # Use provided aspect_ratio or derive from orientation
        if aspect_ratio is None:
            if orientation == "portrait":
                aspect_ratio = "9:16"
            elif orientation == "square":
                aspect_ratio = "1:1"
            else:
                aspect_ratio = "16:9"

        print(f"Fetching asset - orientation: {orientation}, aspect_ratio: {aspect_ratio}")

        # Analyze visual description to determine content type
        content_type = self._detect_content_type(visual_description)
        print(f"Detected content type: {content_type} for '{visual_description[:50]}...'")

        # If content requires technical diagrams (architecture, flowcharts, etc.)
        if content_type in ["diagram", "architecture", "chart", "infographic"]:
            if self.diagram_generator:
                print(f"Generating {content_type} with DiagramGenerator (Mermaid/Diagrams/Graphviz)...")
                try:
                    diagram_path = await self.diagram_generator.generate_diagram(
                        description=visual_description,
                        style=style,
                        aspect_ratio=aspect_ratio
                    )
                    if diagram_path:
                        import uuid
                        return FetchedAsset(
                            id=f"diagram-{uuid.uuid4().hex[:8]}",
                            source="diagram_generator",
                            media_type=MediaType.IMAGE,
                            url=diagram_path,
                            preview_url=diagram_path,
                            width=1920 if aspect_ratio == "16:9" else 1080,
                            height=1080 if aspect_ratio == "16:9" else (1920 if aspect_ratio == "9:16" else 1080),
                            author="DiagramGenerator",
                            quality_score=1.0
                        )
                except Exception as e:
                    print(f"DiagramGenerator failed, falling back to DALL-E: {e}")
                    # Fallback to DALL-E
                    if self.openai_key:
                        enhanced_prompt = self._create_diagram_prompt(visual_description, content_type, style)
                        return await self.generate_ai_image(enhanced_prompt, style="professional", aspect_ratio=aspect_ratio)

        # If content requires human presence
        if content_type == "human_presenter":
            if self.openai_key:
                print("Generating human presenter image with DALL-E...")
                enhanced_prompt = self._create_human_prompt(visual_description, style)
                return await self.generate_ai_image(enhanced_prompt, style=style, aspect_ratio=aspect_ratio)

        # If AI images preferred, generate with DALL-E
        if prefer_ai and media_type == MediaType.IMAGE and self.openai_key:
            prompt = visual_description if visual_description else " ".join(keywords)
            return await self.generate_ai_image(prompt, style=style, aspect_ratio=aspect_ratio)

        # Search stock sources with multiple keyword combinations
        assets = await self.search_all_sources(
            keywords=keywords,
            media_type=media_type,
            orientation=orientation,
            limit_per_source=5
        )

        if assets:
            return assets[0]

        # If no stock found, try with visual description as query
        if visual_description and not assets:
            # Extract key phrases from visual description
            simplified_keywords = self._extract_search_terms(visual_description)
            assets = await self.search_all_sources(
                keywords=simplified_keywords,
                media_type=media_type,
                orientation=orientation,
                limit_per_source=5
            )
            if assets:
                return assets[0]

        # Fallback to DALL-E for images if nothing found
        if fallback_to_ai and self.openai_key:
            print(f"No stock found for '{keywords}', generating with DALL-E...")
            prompt = visual_description if visual_description else " ".join(keywords)
            try:
                return await self.generate_ai_image(prompt, style=style, aspect_ratio=aspect_ratio)
            except Exception as e:
                print(f"DALL-E fallback failed: {e}")
                return None

        return None

    def _detect_content_type(self, description: str) -> str:
        """Detect the type of content needed based on visual description"""
        desc_lower = description.lower()

        # Diagram/architecture keywords
        diagram_keywords = ["diagram", "architecture", "flowchart", "workflow", "system", "infrastructure",
                          "network", "database", "api", "microservices", "schema", "structure",
                          "pipeline", "flow", "process", "layer", "component", "service"]
        if any(kw in desc_lower for kw in diagram_keywords):
            return "diagram"

        # Chart/infographic keywords
        chart_keywords = ["chart", "graph", "statistics", "data", "metrics", "comparison",
                         "infographic", "percentage", "growth", "trend", "analytics"]
        if any(kw in desc_lower for kw in chart_keywords):
            return "chart"

        # Human presenter keywords
        human_keywords = ["person", "human", "presenter", "speaker", "expert", "professional",
                         "man", "woman", "people", "team", "explaining", "teaching", "presenting"]
        if any(kw in desc_lower for kw in human_keywords):
            return "human_presenter"

        return "generic"

    def _create_diagram_prompt(self, description: str, content_type: str, style: str) -> str:
        """Create an enhanced prompt for diagram/architecture generation"""
        base = f"""Create a professional, clean {content_type} visualization:
{description}

Style requirements:
- Clean, modern design with clear visual hierarchy
- Use a dark or gradient background for contrast
- Include clear labels and annotations
- Professional color scheme (blues, purples, teals)
- High resolution, 8K quality
- Make it visually appealing for social media
- {style} aesthetic"""
        return base

    def _create_human_prompt(self, description: str, style: str) -> str:
        """Create an enhanced prompt for human presenter images"""
        base = f"""Create a professional image of a person:
{description}

Style requirements:
- Professional, confident appearance
- Good lighting and composition
- Modern, clean background
- Suitable for educational/professional content
- High resolution, photorealistic quality
- {style} style
- Person should appear approachable and knowledgeable"""
        return base

    def _extract_search_terms(self, description: str) -> List[str]:
        """Extract key search terms from a visual description"""
        import re
        # Remove common filler words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'of', 'to', 'in', 'on', 'with', 'for',
            'and', 'or', 'showing', 'shows', 'displayed', 'featuring', 'with',
            'that', 'this', 'being', 'has', 'have', 'shot', 'scene', 'view'
        }

        # Clean and split
        words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Return top 5 most relevant terms
        return keywords[:5] if keywords else ["video", "content"]
