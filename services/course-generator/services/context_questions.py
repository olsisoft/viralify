"""
Context Questions Service

Provides contextual questions based on profile category and generates
AI-powered questions for specific topics.
"""
import os
from typing import Dict, List, Optional

# Try to import shared LLM provider, fallback to direct OpenAI
try:
    from shared.llm_provider import get_llm_client, get_model_name
    _USE_SHARED_LLM = True
except ImportError:
    from openai import AsyncOpenAI
    _USE_SHARED_LLM = False

from models.course_models import (
    ContextQuestion,
    ProfileCategory,
)


# Base questions for each profile category
BASE_QUESTIONS: Dict[ProfileCategory, List[ContextQuestion]] = {
    ProfileCategory.BUSINESS: [
        ContextQuestion(
            id="target_role",
            question="Quel rôle professionnel ciblez-vous ?",
            type="select",
            options=["Entrepreneurs", "Managers", "Commerciaux", "Freelances", "Cadres dirigeants"],
        ),
        ContextQuestion(
            id="industry_focus",
            question="Secteur d'industrie principal ?",
            type="select",
            options=["E-commerce", "SaaS/Tech", "Services", "Consulting", "Tous secteurs"],
        ),
        ContextQuestion(
            id="expected_outcome",
            question="Résultat concret attendu après le cours ?",
            type="text",
            placeholder="Ex: Augmenter mes ventes de 20%",
        ),
    ],
    ProfileCategory.TECH: [
        ContextQuestion(
            id="tech_domain",
            question="Domaine technique principal ?",
            type="select",
            options=["Développement Web", "Mobile", "Data/IA", "DevOps", "Cybersécurité", "No-code"],
        ),
        ContextQuestion(
            id="specific_tools",
            question="Outils/technologies spécifiques à couvrir ?",
            type="text",
            placeholder="Ex: React, Python, AWS...",
        ),
        ContextQuestion(
            id="practical_focus",
            question="Niveau de pratique souhaité ?",
            type="select",
            options=["Théorique (concepts)", "Équilibré (50/50)", "Très pratique (projets)"],
        ),
    ],
    ProfileCategory.HEALTH: [
        ContextQuestion(
            id="health_domain",
            question="Domaine de santé/bien-être ?",
            type="select",
            options=["Fitness/Musculation", "Nutrition", "Yoga/Méditation", "Santé mentale", "Sport spécifique"],
        ),
        ContextQuestion(
            id="equipment_needed",
            question="Équipement nécessaire ?",
            type="select",
            options=["Aucun (poids du corps)", "Équipement basique", "Salle de sport complète"],
        ),
        ContextQuestion(
            id="physical_level",
            question="Condition physique de l'audience ?",
            type="select",
            options=["Débutant complet", "Actif occasionnel", "Sportif régulier"],
        ),
    ],
    ProfileCategory.CREATIVE: [
        ContextQuestion(
            id="creative_domain",
            question="Domaine créatif ?",
            type="select",
            options=["Design graphique", "Illustration", "Vidéo/Montage", "Photographie", "Écriture", "Musique"],
        ),
        ContextQuestion(
            id="tools_software",
            question="Logiciels/outils utilisés ?",
            type="text",
            placeholder="Ex: Photoshop, Figma, Premiere...",
        ),
        ContextQuestion(
            id="output_type",
            question="Type de création finale ?",
            type="text",
            placeholder="Ex: Logo, portfolio, court-métrage...",
        ),
    ],
    ProfileCategory.EDUCATION: [
        ContextQuestion(
            id="teaching_context",
            question="Contexte d'enseignement ?",
            type="select",
            options=["Scolaire", "Universitaire", "Formation pro", "Auto-formation"],
        ),
        ContextQuestion(
            id="subject_area",
            question="Matière/discipline ?",
            type="text",
            placeholder="Ex: Mathématiques, Anglais, Histoire...",
        ),
        ContextQuestion(
            id="assessment_type",
            question="Type d'évaluation visée ?",
            type="select",
            options=["Examen", "Projet pratique", "Certification", "Compréhension générale"],
        ),
    ],
    ProfileCategory.LIFESTYLE: [
        ContextQuestion(
            id="life_area",
            question="Domaine de vie concerné ?",
            type="select",
            options=["Productivité", "Relations", "Finances personnelles", "Développement personnel", "Parentalité"],
        ),
        ContextQuestion(
            id="current_situation",
            question="Situation actuelle de l'audience ?",
            type="text",
            placeholder="Ex: Employés stressés cherchant un équilibre...",
        ),
        ContextQuestion(
            id="transformation_goal",
            question="Transformation souhaitée ?",
            type="text",
            placeholder="Ex: Être plus organisé et serein au quotidien",
        ),
    ],
}

# Mapping from niche names to categories
NICHE_TO_CATEGORY: Dict[str, ProfileCategory] = {
    # Tech
    "Technology": ProfileCategory.TECH,
    "Tech": ProfileCategory.TECH,
    "Programming": ProfileCategory.TECH,
    "Data Science": ProfileCategory.TECH,
    "Artificial Intelligence": ProfileCategory.TECH,
    "Cybersecurity": ProfileCategory.TECH,
    "Web Development": ProfileCategory.TECH,
    "Mobile Development": ProfileCategory.TECH,

    # Business
    "Business & Entrepreneurship": ProfileCategory.BUSINESS,
    "Business": ProfileCategory.BUSINESS,
    "Entrepreneurship": ProfileCategory.BUSINESS,
    "Finance & Investing": ProfileCategory.BUSINESS,
    "Finance": ProfileCategory.BUSINESS,
    "Marketing & Sales": ProfileCategory.BUSINESS,
    "Marketing": ProfileCategory.BUSINESS,
    "Sales": ProfileCategory.BUSINESS,
    "Leadership": ProfileCategory.BUSINESS,
    "Management": ProfileCategory.BUSINESS,
    "Real Estate": ProfileCategory.BUSINESS,
    "E-commerce": ProfileCategory.BUSINESS,

    # Health
    "Health & Fitness": ProfileCategory.HEALTH,
    "Health": ProfileCategory.HEALTH,
    "Fitness": ProfileCategory.HEALTH,
    "Wellness": ProfileCategory.HEALTH,
    "Nutrition": ProfileCategory.HEALTH,
    "Mental Health": ProfileCategory.HEALTH,
    "Yoga": ProfileCategory.HEALTH,
    "Meditation": ProfileCategory.HEALTH,
    "Sports": ProfileCategory.HEALTH,

    # Creative
    "Art & Design": ProfileCategory.CREATIVE,
    "Art": ProfileCategory.CREATIVE,
    "Design": ProfileCategory.CREATIVE,
    "Music & Audio": ProfileCategory.CREATIVE,
    "Music": ProfileCategory.CREATIVE,
    "Photography": ProfileCategory.CREATIVE,
    "Video Production": ProfileCategory.CREATIVE,
    "Writing": ProfileCategory.CREATIVE,
    "Creative Writing": ProfileCategory.CREATIVE,
    "Illustration": ProfileCategory.CREATIVE,
    "Animation": ProfileCategory.CREATIVE,

    # Education
    "Education & Teaching": ProfileCategory.EDUCATION,
    "Education": ProfileCategory.EDUCATION,
    "Teaching": ProfileCategory.EDUCATION,
    "Language Learning": ProfileCategory.EDUCATION,
    "Languages": ProfileCategory.EDUCATION,
    "Academic": ProfileCategory.EDUCATION,
    "Test Prep": ProfileCategory.EDUCATION,
    "Science": ProfileCategory.EDUCATION,
    "Mathematics": ProfileCategory.EDUCATION,

    # Lifestyle
    "Personal Development": ProfileCategory.LIFESTYLE,
    "Lifestyle": ProfileCategory.LIFESTYLE,
    "Productivity": ProfileCategory.LIFESTYLE,
    "Relationships": ProfileCategory.LIFESTYLE,
    "Parenting": ProfileCategory.LIFESTYLE,
    "Self-Improvement": ProfileCategory.LIFESTYLE,
    "Motivation": ProfileCategory.LIFESTYLE,
    "Spirituality": ProfileCategory.LIFESTYLE,
    "Travel": ProfileCategory.LIFESTYLE,
    "Food & Cooking": ProfileCategory.LIFESTYLE,
    "Home & Garden": ProfileCategory.LIFESTYLE,
    "Fashion & Beauty": ProfileCategory.LIFESTYLE,
    "Gaming": ProfileCategory.LIFESTYLE,
    "Entertainment": ProfileCategory.LIFESTYLE,
}


class CourseContextBuilder:
    """Builds course context from profile and user answers"""

    def __init__(self):
        if _USE_SHARED_LLM:
            self.openai_client = get_llm_client()
        else:
            self.openai_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=120.0,  # 2 minutes timeout for GPT calls
                max_retries=2
            )

    def get_category_from_niche(self, niche: str) -> ProfileCategory:
        """Detect profile category from niche name"""
        # Try exact match first
        if niche in NICHE_TO_CATEGORY:
            return NICHE_TO_CATEGORY[niche]

        # Try case-insensitive match
        niche_lower = niche.lower()
        for key, category in NICHE_TO_CATEGORY.items():
            if key.lower() == niche_lower:
                return category

        # Try partial match
        for key, category in NICHE_TO_CATEGORY.items():
            if key.lower() in niche_lower or niche_lower in key.lower():
                return category

        # Default to lifestyle if no match found
        return ProfileCategory.LIFESTYLE

    def get_base_questions(self, category: ProfileCategory) -> List[ContextQuestion]:
        """Get base questions for a profile category"""
        return BASE_QUESTIONS.get(category, BASE_QUESTIONS[ProfileCategory.LIFESTYLE])

    async def generate_topic_questions(
        self,
        topic: str,
        category: ProfileCategory,
        existing_questions: List[ContextQuestion],
        max_questions: int = 2
    ) -> List[ContextQuestion]:
        """Generate AI-powered questions specific to the topic"""

        existing_ids = [q.id for q in existing_questions]
        existing_texts = [q.question for q in existing_questions]

        prompt = f"""Tu es un expert en conception de cours.

Génère {max_questions} questions pertinentes pour mieux comprendre les besoins de l'utilisateur qui veut créer un cours sur:

SUJET: {topic}
CATÉGORIE: {category.value}

Questions déjà posées (ne pas répéter):
{chr(10).join(f"- {q}" for q in existing_texts)}

Génère des questions qui aident à:
- Préciser le niveau de détail souhaité
- Identifier les cas d'usage spécifiques
- Comprendre les attentes de l'audience

Réponds en JSON avec ce format:
[
  {{
    "id": "unique_id",
    "question": "La question ?",
    "type": "text" ou "select",
    "options": ["opt1", "opt2"] (si type=select),
    "placeholder": "Exemple..." (si type=text)
  }}
]

Réponds UNIQUEMENT avec le JSON, sans markdown."""

        try:
            response = await self.openai_client.chat.completions.create(
                model=get_model_name("fast") if _USE_SHARED_LLM else "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
            )

            import json
            content = response.choices[0].message.content.strip()

            # Clean up potential markdown formatting
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            questions_data = json.loads(content)

            ai_questions = []
            for q_data in questions_data[:max_questions]:
                # Ensure unique ID
                q_id = q_data.get("id", f"ai_{len(ai_questions)}")
                if q_id in existing_ids:
                    q_id = f"ai_{q_id}"

                ai_questions.append(ContextQuestion(
                    id=q_id,
                    question=q_data.get("question", ""),
                    type=q_data.get("type", "text"),
                    options=q_data.get("options"),
                    placeholder=q_data.get("placeholder"),
                    required=False,  # AI questions are optional
                ))

            return ai_questions

        except Exception as e:
            print(f"[CONTEXT] Error generating AI questions: {e}", flush=True)
            return []

    def build_context_summary(
        self,
        category: ProfileCategory,
        profile_niche: str,
        profile_tone: str,
        profile_audience_level: str,
        context_answers: Dict[str, str]
    ) -> str:
        """Build a human-readable context summary"""

        category_labels = {
            ProfileCategory.BUSINESS: "Business",
            ProfileCategory.TECH: "Technique",
            ProfileCategory.HEALTH: "Santé/Fitness",
            ProfileCategory.CREATIVE: "Créatif",
            ProfileCategory.EDUCATION: "Éducation",
            ProfileCategory.LIFESTYLE: "Lifestyle",
        }

        parts = [
            f"Cours {category_labels.get(category, 'Général')}",
            f"Ton: {profile_tone}",
            f"Audience: {profile_audience_level}",
        ]

        # Add key context answers
        if context_answers:
            key_answers = list(context_answers.values())[:2]
            if key_answers:
                parts.append(f"Focus: {', '.join(key_answers)}")

        return " • ".join(parts)
