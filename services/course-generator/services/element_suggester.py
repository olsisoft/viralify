"""
AI Element Suggester Service

Uses GPT to analyze the course topic and suggest the most relevant
lesson elements based on the content.

OPTIMIZED: Includes caching layer to avoid repeated AI calls for similar topics.
"""
import json
import os
from typing import List, Optional, Tuple

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    _USE_SHARED_LLM = False

from models.course_models import ProfileCategory, CourseContext
from models.lesson_elements import (
    LessonElementType,
    LessonElement,
    COMMON_ELEMENTS,
    CATEGORY_ELEMENTS,
    get_elements_for_category,
)
from services.cache_service import get_cache


class ElementSuggester:
    """AI-powered lesson element suggester (Option C)"""

    # Cache TTL for different operations
    CACHE_TTL_SUGGESTIONS = 3600 * 24  # 24 hours for suggestions
    CACHE_TTL_CATEGORY = 3600 * 24  # 24 hours for category detection
    CACHE_TTL_DOMAIN = 3600 * 24  # 24 hours for domain detection

    def __init__(self, openai_api_key: Optional[str] = None):
        if _USE_SHARED_LLM:
            self.client = get_llm_client()
        else:
            self.client = AsyncOpenAI(
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
                timeout=60.0,
                max_retries=2
            )
        self.cache = get_cache()

    async def suggest_elements(
        self,
        topic: str,
        description: Optional[str],
        category: ProfileCategory,
        context: Optional[CourseContext] = None,
    ) -> List[Tuple[LessonElementType, float, str]]:
        """
        Analyze topic and suggest relevant elements with confidence scores.

        OPTIMIZED: Results are cached for 24 hours to avoid repeated AI calls.

        Returns:
            List of (element_id, confidence_score, reason)
        """
        # Check cache first
        cache_key = self.cache._make_key("suggest", topic, description, category.value)
        cached = await self.cache.get(cache_key)
        if cached:
            print(f"[SUGGESTER] Cache hit for topic: {topic}", flush=True)
            # Reconstruct tuples from cached data
            return [(LessonElementType(s[0]), s[1], s[2]) for s in cached]

        print(f"[SUGGESTER] Analyzing topic: {topic} for category: {category.value}", flush=True)

        # Get available elements for this category
        available_elements = get_elements_for_category(category)
        category_specific = CATEGORY_ELEMENTS.get(category, [])

        # Build element descriptions for the prompt
        element_descriptions = []
        for el in category_specific:
            element_descriptions.append(f"- {el.id.value}: {el.name} - {el.description}")

        context_info = ""
        if context:
            context_info = f"""
Context additionnel:
- Niche: {context.profile_niche}
- Ton: {context.profile_tone}
- Niveau audience: {context.profile_audience_level}
- Objectif: {context.profile_primary_goal}
"""
            if context.context_answers:
                context_info += "- Réponses contextuelles: " + ", ".join(
                    f"{k}={v}" for k, v in context.context_answers.items()
                )

        prompt = f"""Tu es un expert en conception pédagogique.

Analyse ce sujet de cours et suggère les éléments de leçon les plus pertinents.

SUJET: {topic}
{f'DESCRIPTION: {description}' if description else ''}
CATÉGORIE: {category.value}
{context_info}

ÉLÉMENTS DISPONIBLES pour cette catégorie:
{chr(10).join(element_descriptions)}

Pour chaque élément, évalue sa pertinence pour ce sujet spécifique.

Réponds en JSON avec ce format:
{{
    "suggestions": [
        {{
            "element_id": "identifiant_element",
            "confidence": 0.0-1.0,
            "reason": "Raison de la suggestion"
        }}
    ],
    "detected_category": "catégorie détectée si différente",
    "additional_recommendations": "Recommandations supplémentaires"
}}

Inclus TOUS les éléments de la catégorie avec leur score de pertinence.
Un score > 0.7 signifie très pertinent, 0.4-0.7 moyennement pertinent, < 0.4 peu pertinent.

Réponds UNIQUEMENT avec le JSON, sans markdown."""

        try:
            response = await self.client.chat.completions.create(
                model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()

            # Clean up potential markdown formatting
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            suggestions = []

            for item in data.get("suggestions", []):
                try:
                    element_id = LessonElementType(item["element_id"])
                    confidence = float(item.get("confidence", 0.5))
                    reason = item.get("reason", "")
                    suggestions.append((element_id, confidence, reason))
                except (ValueError, KeyError) as e:
                    print(f"[SUGGESTER] Skipping invalid element: {item}, error: {e}", flush=True)
                    continue

            # Sort by confidence descending
            suggestions.sort(key=lambda x: x[1], reverse=True)

            print(f"[SUGGESTER] Suggested {len(suggestions)} elements", flush=True)

            # Cache the results (convert to serializable format)
            cache_data = [(s[0].value, s[1], s[2]) for s in suggestions]
            await self.cache.set(cache_key, cache_data, self.CACHE_TTL_SUGGESTIONS)

            return suggestions

        except Exception as e:
            print(f"[SUGGESTER] Error suggesting elements: {e}", flush=True)
            # Return default suggestions based on category
            return self._get_default_suggestions(category)

    def _get_default_suggestions(
        self, category: ProfileCategory
    ) -> List[Tuple[LessonElementType, float, str]]:
        """Return default suggestions if AI fails"""
        suggestions = []
        category_elements = CATEGORY_ELEMENTS.get(category, [])

        for i, el in enumerate(category_elements):
            # First 3 elements get high confidence, rest medium
            confidence = 0.9 - (i * 0.1) if i < 3 else 0.5
            suggestions.append((el.id, confidence, f"Élément standard pour {category.value}"))

        return suggestions

    async def detect_category(
        self,
        topic: str,
        description: Optional[str] = None,
    ) -> Tuple[ProfileCategory, float]:
        """
        Auto-detect the best category for a topic.

        OPTIMIZED: Results are cached for 24 hours.

        Returns:
            (detected_category, confidence_score)
        """
        # Check cache first
        cache_key = self.cache._make_key("category", topic, description)
        cached = await self.cache.get(cache_key)
        if cached:
            print(f"[SUGGESTER] Cache hit for category detection: {topic}", flush=True)
            return ProfileCategory(cached["category"]), cached["confidence"]

        print(f"[SUGGESTER] Detecting category for: {topic}", flush=True)

        categories_desc = """
- tech: Programmation, développement, IA, data science, cybersécurité, DevOps
- business: Entrepreneuriat, marketing, vente, management, finance, e-commerce
- health: Fitness, nutrition, yoga, méditation, santé mentale, sports
- creative: Design, illustration, vidéo, photo, écriture, musique
- education: Enseignement, langues, sciences, mathématiques, préparation examens
- lifestyle: Productivité, développement personnel, relations, parentalité, cuisine
"""

        prompt = f"""Analyse ce sujet de cours et détermine la catégorie la plus appropriée.

SUJET: {topic}
{f'DESCRIPTION: {description}' if description else ''}

CATÉGORIES DISPONIBLES:
{categories_desc}

Réponds en JSON:
{{
    "category": "nom_categorie",
    "confidence": 0.0-1.0,
    "reason": "Raison du choix"
}}

Réponds UNIQUEMENT avec le JSON."""

        try:
            response = await self.client.chat.completions.create(
                model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            category = ProfileCategory(data["category"])
            confidence = float(data.get("confidence", 0.8))

            print(f"[SUGGESTER] Detected category: {category.value} (confidence: {confidence})", flush=True)

            # Cache the result
            await self.cache.set(cache_key, {
                "category": category.value,
                "confidence": confidence
            }, self.CACHE_TTL_CATEGORY)

            return category, confidence

        except Exception as e:
            print(f"[SUGGESTER] Error detecting category: {e}, defaulting to lifestyle", flush=True)
            return ProfileCategory.LIFESTYLE, 0.5

    async def detect_domain_and_keywords(
        self,
        topic: str,
        description: Optional[str],
        category: ProfileCategory,
    ) -> dict:
        """
        Detect the specific domain within a category and suggest relevant keywords.
        Keywords become the dynamic domain options.

        OPTIMIZED: Results are cached for 24 hours.

        Returns:
            {
                "domain": "detected domain (first keyword)",
                "domain_options": ["keyword1", "keyword2", ...],  # Dynamic based on topic
                "keywords": ["keyword1", "keyword2", ...]
            }
        """
        # Check cache first
        cache_key = self.cache._make_key("domain", topic, description, category.value)
        cached = await self.cache.get(cache_key)
        if cached:
            print(f"[SUGGESTER] Cache hit for domain detection: {topic}", flush=True)
            return cached

        print(f"[SUGGESTER] Detecting domain for: {topic} (category: {category.value})", flush=True)

        prompt = f"""Analyse ce sujet de cours et génère:
1. Le domaine technique principal spécifique à ce sujet (pas générique)
2. 5-6 sous-domaines ou concepts clés qui caractérisent ce sujet
3. Des outils/technologies associés

SUJET: {topic}
{f'DESCRIPTION: {description}' if description else ''}
CATÉGORIE: {category.value}

Réponds en JSON:
{{
    "main_domain": "le domaine technique principal spécifique au sujet",
    "sub_domains": ["sous-domaine 1", "sous-domaine 2", "sous-domaine 3", "sous-domaine 4", "sous-domaine 5"],
    "tools": ["outil 1", "outil 2", "outil 3"]
}}

IMPORTANT:
- Le main_domain doit être SPÉCIFIQUE au sujet (ex: pour "Enterprise Architecture Integration" -> "Architecture d'entreprise")
- Les sub_domains doivent être des concepts clés du sujet, pas des catégories génériques
- Exemples de bons sub_domains: "microservices", "intégration API", "patterns d'architecture", "ESB", "SOA"

Réponds UNIQUEMENT avec le JSON."""

        try:
            response = await self.client.chat.completions.create(
                model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            main_domain = data.get("main_domain", topic)
            sub_domains = data.get("sub_domains", [])
            tools = data.get("tools", [])

            # Build dynamic domain options: main domain + sub domains
            domain_options = [main_domain] + sub_domains[:5]
            # Remove duplicates while preserving order
            seen = set()
            unique_options = []
            for opt in domain_options:
                if opt.lower() not in seen:
                    seen.add(opt.lower())
                    unique_options.append(opt)

            print(f"[SUGGESTER] Detected domain: {main_domain}, options: {unique_options}", flush=True)
            result = {
                "domain": main_domain,
                "domain_options": unique_options[:6],  # Limit to 6 options
                "keywords": sub_domains[:8],  # Keywords are the sub-domains
                "tools": tools[:5],
            }

            # Cache the result
            await self.cache.set(cache_key, result, self.CACHE_TTL_DOMAIN)

            return result

        except Exception as e:
            print(f"[SUGGESTER] Error detecting domain: {e}", flush=True)
            return {
                "domain": topic,
                "domain_options": [topic],
                "keywords": [],
                "tools": [],
            }

    async def get_smart_defaults(
        self,
        topic: str,
        description: Optional[str],
        category: Optional[ProfileCategory] = None,
        context: Optional[CourseContext] = None,
    ) -> dict:
        """
        Get smart default element configuration based on AI analysis.

        Returns a complete configuration with suggested elements enabled.
        """
        # Detect category if not provided
        if not category:
            category, _ = await self.detect_category(topic, description)

        # Get AI suggestions
        suggestions = await self.suggest_elements(topic, description, category, context)

        # Build configuration
        from models.lesson_elements import (
            AdaptiveLessonElementConfig,
            QuizConfig,
            QuizFrequency,
        )

        # Common elements always enabled
        common_elements = {
            LessonElementType.CONCEPT_INTRO: True,
            LessonElementType.VOICEOVER: True,
            LessonElementType.CURRICULUM_SLIDE: True,
            LessonElementType.CONCLUSION: True,
            LessonElementType.QUIZ: True,
        }

        # Category elements based on AI confidence
        category_elements = {}
        ai_suggested = []

        for element_id, confidence, reason in suggestions:
            # Enable if confidence > 0.6
            category_elements[element_id] = confidence > 0.6
            if confidence > 0.7:
                ai_suggested.append(element_id)

        config = AdaptiveLessonElementConfig(
            common_elements=common_elements,
            category_elements=category_elements,
            ai_suggested_elements=ai_suggested,
            quiz_config=QuizConfig(
                enabled=True,
                frequency=QuizFrequency.PER_SECTION,
                questions_per_quiz=5,
            ),
        )

        return {
            "detected_category": category.value,
            "config": config.model_dump(),
            "suggestions": [
                {
                    "element_id": el_id.value,
                    "confidence": conf,
                    "reason": reason,
                    "enabled": conf > 0.6,
                }
                for el_id, conf, reason in suggestions
            ],
        }
