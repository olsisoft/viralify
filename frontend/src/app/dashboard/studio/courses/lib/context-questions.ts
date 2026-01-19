/**
 * Context Questions Configuration
 *
 * Maps profile niches to categories and defines base questions for each category.
 */

import type { ProfileCategory, ContextQuestion } from './course-types';

// Mapping from niche names to categories
export const NICHE_TO_CATEGORY: Record<string, ProfileCategory> = {
  // Tech
  Technology: 'tech',
  Tech: 'tech',
  Programming: 'tech',
  'Data Science': 'tech',
  'Artificial Intelligence': 'tech',
  Cybersecurity: 'tech',
  'Web Development': 'tech',
  'Mobile Development': 'tech',

  // Business
  'Business & Entrepreneurship': 'business',
  Business: 'business',
  Entrepreneurship: 'business',
  'Finance & Investing': 'business',
  Finance: 'business',
  'Marketing & Sales': 'business',
  Marketing: 'business',
  Sales: 'business',
  Leadership: 'business',
  Management: 'business',
  'Real Estate': 'business',
  'E-commerce': 'business',

  // Health
  'Health & Fitness': 'health',
  Health: 'health',
  Fitness: 'health',
  Wellness: 'health',
  Nutrition: 'health',
  'Mental Health': 'health',
  Yoga: 'health',
  Meditation: 'health',
  Sports: 'health',

  // Creative
  'Art & Design': 'creative',
  Art: 'creative',
  Design: 'creative',
  'Music & Audio': 'creative',
  Music: 'creative',
  Photography: 'creative',
  'Video Production': 'creative',
  Writing: 'creative',
  'Creative Writing': 'creative',
  Illustration: 'creative',
  Animation: 'creative',

  // Education
  'Education & Teaching': 'education',
  Education: 'education',
  Teaching: 'education',
  'Language Learning': 'education',
  Languages: 'education',
  Academic: 'education',
  'Test Prep': 'education',
  Science: 'education',
  Mathematics: 'education',

  // Lifestyle
  'Personal Development': 'lifestyle',
  Lifestyle: 'lifestyle',
  Productivity: 'lifestyle',
  Relationships: 'lifestyle',
  Parenting: 'lifestyle',
  'Self-Improvement': 'lifestyle',
  Motivation: 'lifestyle',
  Spirituality: 'lifestyle',
  Travel: 'lifestyle',
  'Food & Cooking': 'lifestyle',
  'Home & Garden': 'lifestyle',
  'Fashion & Beauty': 'lifestyle',
  Gaming: 'lifestyle',
  Entertainment: 'lifestyle',
};

// Base questions for each category
export const BASE_QUESTIONS: Record<ProfileCategory, ContextQuestion[]> = {
  business: [
    {
      id: 'target_role',
      question: 'Quel rôle professionnel ciblez-vous ?',
      type: 'select',
      options: [
        'Entrepreneurs',
        'Managers',
        'Commerciaux',
        'Freelances',
        'Cadres dirigeants',
      ],
      required: true,
    },
    {
      id: 'industry_focus',
      question: "Secteur d'industrie principal ?",
      type: 'select',
      options: ['E-commerce', 'SaaS/Tech', 'Services', 'Consulting', 'Tous secteurs'],
      required: true,
    },
    {
      id: 'expected_outcome',
      question: 'Résultat concret attendu après le cours ?',
      type: 'text',
      placeholder: 'Ex: Augmenter mes ventes de 20%',
      required: true,
    },
  ],

  tech: [
    {
      id: 'tech_domain',
      question: 'Domaine technique principal ?',
      type: 'select',
      options: [
        'Développement Web',
        'Mobile',
        'Data/IA',
        'DevOps',
        'Cybersécurité',
        'No-code',
      ],
      required: true,
    },
    {
      id: 'specific_tools',
      question: 'Outils/technologies spécifiques à couvrir ?',
      type: 'text',
      placeholder: 'Ex: React, Python, AWS...',
      required: true,
    },
    {
      id: 'practical_focus',
      question: 'Niveau de pratique souhaité ?',
      type: 'select',
      options: ['Théorique (concepts)', 'Équilibré (50/50)', 'Très pratique (projets)'],
      required: true,
    },
  ],

  health: [
    {
      id: 'health_domain',
      question: 'Domaine de santé/bien-être ?',
      type: 'select',
      options: [
        'Fitness/Musculation',
        'Nutrition',
        'Yoga/Méditation',
        'Santé mentale',
        'Sport spécifique',
      ],
      required: true,
    },
    {
      id: 'equipment_needed',
      question: 'Équipement nécessaire ?',
      type: 'select',
      options: ['Aucun (poids du corps)', 'Équipement basique', 'Salle de sport complète'],
      required: true,
    },
    {
      id: 'physical_level',
      question: "Condition physique de l'audience ?",
      type: 'select',
      options: ['Débutant complet', 'Actif occasionnel', 'Sportif régulier'],
      required: true,
    },
  ],

  creative: [
    {
      id: 'creative_domain',
      question: 'Domaine créatif ?',
      type: 'select',
      options: [
        'Design graphique',
        'Illustration',
        'Vidéo/Montage',
        'Photographie',
        'Écriture',
        'Musique',
      ],
      required: true,
    },
    {
      id: 'tools_software',
      question: 'Logiciels/outils utilisés ?',
      type: 'text',
      placeholder: 'Ex: Photoshop, Figma, Premiere...',
      required: true,
    },
    {
      id: 'output_type',
      question: 'Type de création finale ?',
      type: 'text',
      placeholder: 'Ex: Logo, portfolio, court-métrage...',
      required: true,
    },
  ],

  education: [
    {
      id: 'teaching_context',
      question: "Contexte d'enseignement ?",
      type: 'select',
      options: ['Scolaire', 'Universitaire', 'Formation pro', 'Auto-formation'],
      required: true,
    },
    {
      id: 'subject_area',
      question: 'Matière/discipline ?',
      type: 'text',
      placeholder: 'Ex: Mathématiques, Anglais, Histoire...',
      required: true,
    },
    {
      id: 'assessment_type',
      question: "Type d'évaluation visée ?",
      type: 'select',
      options: ['Examen', 'Projet pratique', 'Certification', 'Compréhension générale'],
      required: true,
    },
  ],

  lifestyle: [
    {
      id: 'life_area',
      question: 'Domaine de vie concerné ?',
      type: 'select',
      options: [
        'Productivité',
        'Relations',
        'Finances personnelles',
        'Développement personnel',
        'Parentalité',
      ],
      required: true,
    },
    {
      id: 'current_situation',
      question: "Situation actuelle de l'audience ?",
      type: 'text',
      placeholder: 'Ex: Employés stressés cherchant un équilibre...',
      required: true,
    },
    {
      id: 'transformation_goal',
      question: 'Transformation souhaitée ?',
      type: 'text',
      placeholder: 'Ex: Être plus organisé et serein au quotidien',
      required: true,
    },
  ],
};

/**
 * Get category from niche name
 */
export function getCategoryFromNiche(niche: string): ProfileCategory {
  // Try exact match first
  if (niche in NICHE_TO_CATEGORY) {
    return NICHE_TO_CATEGORY[niche];
  }

  // Try case-insensitive match
  const nicheLower = niche.toLowerCase();
  for (const [key, category] of Object.entries(NICHE_TO_CATEGORY)) {
    if (key.toLowerCase() === nicheLower) {
      return category;
    }
  }

  // Try partial match
  for (const [key, category] of Object.entries(NICHE_TO_CATEGORY)) {
    if (key.toLowerCase().includes(nicheLower) || nicheLower.includes(key.toLowerCase())) {
      return category;
    }
  }

  // Default to lifestyle
  return 'lifestyle';
}

/**
 * Get base questions for a category
 */
export function getBaseQuestions(category: ProfileCategory): ContextQuestion[] {
  return BASE_QUESTIONS[category] || BASE_QUESTIONS.lifestyle;
}

/**
 * Check if all required questions are answered
 */
export function areQuestionsAnswered(
  questions: ContextQuestion[],
  answers: Record<string, string>
): boolean {
  return questions
    .filter((q) => q.required !== false)
    .every((q) => answers[q.id] && answers[q.id].trim() !== '');
}
