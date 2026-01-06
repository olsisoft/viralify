"""
Music Service
Provides background music from:
- Free Music Library (royalty-free tracks)
- Suno AI (AI-generated music)
- Pixabay Music (free)
"""

import asyncio
import httpx
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum
import os


class MusicMood(str, Enum):
    UPBEAT = "upbeat"
    CALM = "calm"
    EPIC = "epic"
    EMOTIONAL = "emotional"
    ENERGETIC = "energetic"
    INSPIRATIONAL = "inspirational"
    CORPORATE = "corporate"
    CINEMATIC = "cinematic"
    HAPPY = "happy"
    SAD = "sad"
    MYSTERIOUS = "mysterious"
    ROMANTIC = "romantic"


class MusicTrack(BaseModel):
    id: str
    title: str
    artist: str
    source: str  # "library", "suno", "pixabay"
    url: str
    preview_url: Optional[str] = None
    duration: float  # seconds
    mood: Optional[str] = None
    genre: Optional[str] = None
    bpm: Optional[int] = None
    is_loopable: bool = False


# Built-in royalty-free music library
# Using SoundHelix sample music (reliable, always available)
FREE_MUSIC_LIBRARY = [
    {
        "id": "lib-001",
        "title": "Upbeat Corporate",
        "artist": "SoundHelix",
        "mood": "corporate",
        "genre": "corporate",
        "bpm": 120,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "is_loopable": True
    },
    {
        "id": "lib-002",
        "title": "Inspirational Piano",
        "artist": "SoundHelix",
        "mood": "inspirational",
        "genre": "piano",
        "bpm": 80,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
        "is_loopable": True
    },
    {
        "id": "lib-003",
        "title": "Epic Cinematic",
        "artist": "SoundHelix",
        "mood": "epic",
        "genre": "orchestral",
        "bpm": 100,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "is_loopable": False
    },
    {
        "id": "lib-004",
        "title": "Calm Ambient",
        "artist": "SoundHelix",
        "mood": "calm",
        "genre": "ambient",
        "bpm": 70,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
        "is_loopable": True
    },
    {
        "id": "lib-005",
        "title": "Energetic Electronic",
        "artist": "SoundHelix",
        "mood": "energetic",
        "genre": "electronic",
        "bpm": 140,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
        "is_loopable": True
    },
    {
        "id": "lib-006",
        "title": "Happy Acoustic",
        "artist": "SoundHelix",
        "mood": "happy",
        "genre": "acoustic",
        "bpm": 110,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
        "is_loopable": True
    },
    {
        "id": "lib-007",
        "title": "Emotional Strings",
        "artist": "SoundHelix",
        "mood": "emotional",
        "genre": "orchestral",
        "bpm": 60,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3",
        "is_loopable": False
    },
    {
        "id": "lib-008",
        "title": "Mysterious Dark",
        "artist": "SoundHelix",
        "mood": "mysterious",
        "genre": "cinematic",
        "bpm": 90,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3",
        "is_loopable": True
    },
    {
        "id": "lib-009",
        "title": "Cinematic Inspirational",
        "artist": "SoundHelix",
        "mood": "cinematic",
        "genre": "cinematic",
        "bpm": 85,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-9.mp3",
        "is_loopable": True
    },
    {
        "id": "lib-010",
        "title": "Upbeat Pop",
        "artist": "SoundHelix",
        "mood": "upbeat",
        "genre": "pop",
        "bpm": 128,
        "duration": 300,
        "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-10.mp3",
        "is_loopable": True
    }
]


class MusicService:
    """Provides background music for videos"""

    def __init__(
        self,
        pixabay_api_key: str = "",
        suno_api_key: str = ""
    ):
        self.pixabay_key = pixabay_api_key
        self.suno_key = suno_api_key
        self.library = FREE_MUSIC_LIBRARY

    def get_library_tracks(
        self,
        mood: Optional[str] = None,
        genre: Optional[str] = None,
        min_duration: float = 0
    ) -> List[MusicTrack]:
        """Get tracks from built-in library"""

        tracks = []
        for item in self.library:
            # Filter by mood
            if mood and item.get("mood") != mood:
                continue
            # Filter by genre
            if genre and item.get("genre") != genre:
                continue
            # Filter by duration
            if item.get("duration", 0) < min_duration:
                continue

            tracks.append(MusicTrack(
                id=item["id"],
                title=item["title"],
                artist=item["artist"],
                source="library",
                url=item["url"],
                duration=item["duration"],
                mood=item.get("mood"),
                genre=item.get("genre"),
                bpm=item.get("bpm"),
                is_loopable=item.get("is_loopable", False)
            ))

        return tracks

    async def search_pixabay_music(
        self,
        query: str,
        min_duration: float = 30
    ) -> List[MusicTrack]:
        """Search Pixabay for royalty-free music"""

        if not self.pixabay_key:
            return []

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://pixabay.com/api/",
                params={
                    "key": self.pixabay_key,
                    "q": query,
                    "media_type": "music",
                    "per_page": 10,
                    "safesearch": "true"
                },
                timeout=30.0
            )

            # Note: Pixabay music API requires different endpoint
            # This is a placeholder - actual implementation would use their audio API
            return []

    async def generate_ai_music(
        self,
        prompt: str,
        duration: int = 30,
        style: str = "cinematic"
    ) -> Optional[MusicTrack]:
        """Generate music using Suno AI"""

        if not self.suno_key:
            # Fallback to library if Suno not configured
            return None

        # Suno AI API integration
        # Note: Suno API is still in beta/limited access
        # This is a placeholder for when it becomes available

        try:
            async with httpx.AsyncClient() as client:
                # Create generation request
                response = await client.post(
                    "https://api.suno.ai/v1/generate",
                    headers={
                        "Authorization": f"Bearer {self.suno_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "prompt": f"{prompt}, {style} style, instrumental, {duration} seconds",
                        "duration": duration,
                        "instrumental": True
                    },
                    timeout=120.0
                )

                if response.status_code != 200:
                    print(f"Suno API error: {response.text}")
                    return None

                data = response.json()

                import uuid
                return MusicTrack(
                    id=f"suno-{uuid.uuid4().hex[:8]}",
                    title=f"AI Generated: {prompt[:30]}",
                    artist="Suno AI",
                    source="suno",
                    url=data.get("audio_url", ""),
                    duration=duration,
                    mood=style,
                    is_loopable=False
                )

        except Exception as e:
            print(f"Suno generation error: {e}")
            return None

    async def get_best_track_for_mood(
        self,
        mood_description: str,
        min_duration: float = 30,
        prefer_ai: bool = False
    ) -> Optional[MusicTrack]:
        """Get the best matching track for a mood/style description"""

        # Map description to mood enum
        mood_mapping = {
            "upbeat": MusicMood.UPBEAT,
            "happy": MusicMood.HAPPY,
            "energetic": MusicMood.ENERGETIC,
            "calm": MusicMood.CALM,
            "relaxing": MusicMood.CALM,
            "peaceful": MusicMood.CALM,
            "epic": MusicMood.EPIC,
            "dramatic": MusicMood.EPIC,
            "cinematic": MusicMood.CINEMATIC,
            "emotional": MusicMood.EMOTIONAL,
            "sad": MusicMood.SAD,
            "inspiring": MusicMood.INSPIRATIONAL,
            "motivational": MusicMood.INSPIRATIONAL,
            "corporate": MusicMood.CORPORATE,
            "professional": MusicMood.CORPORATE,
            "mysterious": MusicMood.MYSTERIOUS,
            "dark": MusicMood.MYSTERIOUS,
            "romantic": MusicMood.ROMANTIC,
            "love": MusicMood.ROMANTIC
        }

        # Find matching mood
        detected_mood = None
        description_lower = mood_description.lower()

        for keyword, mood in mood_mapping.items():
            if keyword in description_lower:
                detected_mood = mood.value
                break

        # Try AI generation first if preferred
        if prefer_ai and self.suno_key:
            ai_track = await self.generate_ai_music(
                prompt=mood_description,
                duration=int(min_duration) + 30,  # Generate slightly longer
                style=detected_mood or "cinematic"
            )
            if ai_track:
                return ai_track

        # Search library
        library_tracks = self.get_library_tracks(
            mood=detected_mood,
            min_duration=min_duration
        )

        if library_tracks:
            return library_tracks[0]

        # Fallback: return any track that's long enough
        all_tracks = self.get_library_tracks(min_duration=min_duration)
        return all_tracks[0] if all_tracks else None

    def get_all_moods(self) -> List[str]:
        """Get list of available moods"""
        return [mood.value for mood in MusicMood]

    def get_library_stats(self) -> Dict[str, Any]:
        """Get statistics about the music library"""
        moods = {}
        genres = {}

        for track in self.library:
            mood = track.get("mood", "unknown")
            genre = track.get("genre", "unknown")

            moods[mood] = moods.get(mood, 0) + 1
            genres[genre] = genres.get(genre, 0) + 1

        return {
            "total_tracks": len(self.library),
            "moods": moods,
            "genres": genres
        }
