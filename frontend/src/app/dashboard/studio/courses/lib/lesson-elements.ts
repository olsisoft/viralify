/**
 * Category-based Lesson Elements Configuration
 *
 * Defines lesson elements specific to each profile category
 * and quiz configuration types.
 */

import type { ProfileCategory } from './course-types';

// Lesson element types
export type LessonElementType =
  // Common elements (all categories)
  | 'concept_intro'
  | 'voiceover'
  | 'curriculum_slide'
  | 'conclusion'
  | 'quiz'
  // Tech elements
  | 'code_demo'
  | 'terminal_output'
  | 'architecture_diagram'
  | 'debug_tips'
  | 'code_execution'
  // Business elements
  | 'case_study'
  | 'framework_template'
  | 'roi_metrics'
  | 'action_checklist'
  | 'market_analysis'
  // Health elements
  | 'exercise_demo'
  | 'safety_warning'
  | 'body_diagram'
  | 'progression_plan'
  | 'rest_guidance'
  // Creative elements
  | 'before_after'
  | 'technique_demo'
  | 'tool_tutorial'
  | 'creative_exercise'
  | 'critique_section'
  // Education elements
  | 'memory_aid'
  | 'practice_problem'
  | 'multiple_explanations'
  | 'summary_card'
  // Lifestyle elements
  | 'daily_routine'
  | 'reflection_exercise'
  | 'goal_setting'
  | 'habit_tracker'
  | 'milestone';

// Lesson element definition
export interface LessonElement {
  id: LessonElementType;
  name: string;
  description: string;
  icon: string;
  isRequired: boolean;
  enabled: boolean;
  presentationType?: string;
}

// Quiz frequency options
export type QuizFrequency = 'per_lecture' | 'per_section' | 'end_of_course' | 'custom';

// Quiz question types (Udemy style)
export type QuizQuestionType =
  | 'multiple_choice'
  | 'multi_select'
  | 'true_false'
  | 'fill_blank'
  | 'matching';

// Quiz configuration
export interface QuizConfig {
  enabled: boolean; // Always true (quizzes are required)
  frequency: QuizFrequency;
  customFrequency?: number; // Every N lectures (if frequency = 'custom')
  questionsPerQuiz: number;
  questionTypes: QuizQuestionType[];
  passingScore: number;
  showExplanations: boolean;
  allowRetry: boolean;
}

// Adaptive lesson elements configuration
export interface AdaptiveLessonElementConfig {
  commonElements: Record<LessonElementType, boolean>;
  categoryElements: Record<LessonElementType, boolean>;
  aiSuggestedElements: LessonElementType[];
  quizConfig: QuizConfig;
}

// API response for category elements
export interface CategoryElementsResponse {
  category: ProfileCategory;
  commonElements: LessonElement[];
  categoryElements: LessonElement[];
}

// AI suggestion response
export interface ElementSuggestion {
  elementId: LessonElementType;
  confidence: number;
  reason: string;
  enabled: boolean;
}

export interface ElementSuggestionResponse {
  detectedCategory: ProfileCategory;
  config: AdaptiveLessonElementConfig;
  suggestions: ElementSuggestion[];
}

// Category info
export interface CategoryInfo {
  id: ProfileCategory;
  name: string;
  icon: string;
  description: string;
}

// Categories data
export const CATEGORIES: CategoryInfo[] = [
  { id: 'tech', name: 'Technique', icon: '💻', description: 'Programmation, développement, IA, data science' },
  { id: 'business', name: 'Business', icon: '💼', description: 'Entrepreneuriat, marketing, vente, management' },
  { id: 'health', name: 'Santé/Fitness', icon: '🏃', description: 'Fitness, nutrition, yoga, bien-être' },
  { id: 'creative', name: 'Créatif', icon: '🎨', description: 'Design, illustration, vidéo, photo, musique' },
  { id: 'education', name: 'Éducation', icon: '📚', description: 'Enseignement, langues, sciences, examens' },
  { id: 'lifestyle', name: 'Lifestyle', icon: '✨', description: 'Productivité, développement personnel, relations' },
];

// Quiz frequency options
export const QUIZ_FREQUENCIES: { id: QuizFrequency; name: string; description: string }[] = [
  { id: 'per_lecture', name: 'Par lecture', description: 'Quiz à la fin de chaque lecture' },
  { id: 'per_section', name: 'Par section', description: 'Quiz à la fin de chaque section' },
  { id: 'end_of_course', name: 'Fin de cours', description: 'Un seul quiz final' },
  { id: 'custom', name: 'Personnalisé', description: 'Toutes les N lectures' },
];

// Quiz question type options
export const QUIZ_QUESTION_TYPES: { id: QuizQuestionType; name: string; description: string }[] = [
  { id: 'multiple_choice', name: 'QCM', description: 'Une seule bonne réponse' },
  { id: 'multi_select', name: 'Choix multiples', description: 'Plusieurs bonnes réponses' },
  { id: 'true_false', name: 'Vrai/Faux', description: 'Vrai ou faux' },
  { id: 'fill_blank', name: 'Texte à trous', description: 'Compléter le texte' },
  { id: 'matching', name: 'Association', description: 'Associer des éléments' },
];

// Default quiz configuration
export const DEFAULT_QUIZ_CONFIG: QuizConfig = {
  enabled: true,
  frequency: 'per_section',
  questionsPerQuiz: 5,
  questionTypes: ['multiple_choice', 'true_false'],
  passingScore: 70,
  showExplanations: true,
  allowRetry: true,
};

// Get category info by ID
export function getCategoryInfo(categoryId: ProfileCategory): CategoryInfo | undefined {
  return CATEGORIES.find((c) => c.id === categoryId);
}
