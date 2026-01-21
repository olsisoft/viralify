/**
 * Course Generator TypeScript Types
 */

import type { Document } from './document-types';

// Profile Categories
export type ProfileCategory =
  | 'business'
  | 'tech'
  | 'health'
  | 'creative'
  | 'education'
  | 'lifestyle';

// Category display info
export const CATEGORY_INFO: Record<ProfileCategory, { icon: string; label: string }> = {
  business: { icon: '💼', label: 'Business' },
  tech: { icon: '💻', label: 'Technique' },
  health: { icon: '🏃', label: 'Santé/Fitness' },
  creative: { icon: '🎨', label: 'Créatif' },
  education: { icon: '📚', label: 'Éducation' },
  lifestyle: { icon: '✨', label: 'Lifestyle' },
};

// Context Question type
export interface ContextQuestion {
  id: string;
  question: string;
  type: 'text' | 'select';
  options?: string[];
  placeholder?: string;
  required?: boolean;
}

// Course Context built from profile + answers
export interface CourseContext {
  category: ProfileCategory;
  profileNiche: string;
  profileTone: string;
  profileAudienceLevel: string;
  profileLanguageLevel: string;
  profilePrimaryGoal: string;
  profileAudienceDescription: string;
  contextAnswers: Record<string, string>;
  specificTools?: string;
  practicalFocus?: string;
  expectedOutcome?: string;
}

export type DifficultyLevel =
  | 'beginner'
  | 'intermediate'
  | 'advanced'
  | 'very_advanced'
  | 'expert';

export type CourseStage =
  | 'queued'
  | 'planning'
  | 'generating_lectures'
  | 'compiling'
  | 'completed'
  | 'failed';

export interface LessonElementConfig {
  conceptIntro: boolean;
  diagramSchema: boolean;
  codeTyping: boolean;
  codeExecution: boolean;
  voiceoverExplanation: boolean;
  curriculumSlide: boolean; // Always true, readonly
}

export interface CourseStructureConfig {
  totalDurationMinutes: number;
  numberOfSections: number;
  lecturesPerSection: number;
  randomStructure: boolean;
}

export type LectureStatus = 'pending' | 'generating' | 'completed' | 'failed' | 'retrying';

export interface Lecture {
  id: string;
  title: string;
  description: string;
  objectives: string[];
  difficulty: DifficultyLevel;
  durationSeconds: number;
  order: number;
  status: LectureStatus;
  presentationJobId?: string;
  videoUrl?: string;
  error?: string;
  // Progress tracking
  progressPercent: number;
  currentStage?: string;
  retryCount: number;
}

export interface Section {
  id: string;
  title: string;
  description: string;
  order: number;
  lectures: Lecture[];
}

export interface CourseOutline {
  title: string;
  description: string;
  targetAudience: string;
  language: string;
  difficultyStart: DifficultyLevel;
  difficultyEnd: DifficultyLevel;
  totalDurationMinutes: number;
  sections: Section[];
}

// OPTIMIZED: Preview response includes RAG context to avoid double-fetching
export interface PreviewOutlineResponse {
  outline: CourseOutline;
  ragContext?: string; // Pre-fetched RAG context to pass to generate
}

export interface GenerateCourseRequest {
  profileId: string;
  topic: string;
  description?: string;
  difficultyStart: DifficultyLevel;
  difficultyEnd: DifficultyLevel;
  structure: {
    total_duration_minutes: number;
    number_of_sections: number;
    lectures_per_section: number;
    random_structure: boolean;
  };
  lessonElements: {
    concept_intro: boolean;
    diagram_schema: boolean;
    code_typing: boolean;
    code_execution: boolean;
    voiceover_explanation: boolean;
    curriculum_slide: boolean;
  };
  language: string;
  voiceId: string;
  style: string;
  typingSpeed: string;
  includeAvatar: boolean;
  avatarId?: string;
  approvedOutline?: CourseOutline;
  // RAG document references (Phase 2)
  document_ids?: string[];
  // OPTIMIZED: Pre-fetched RAG context from preview (avoids double-fetching)
  rag_context?: string;
  // Custom keywords for context refinement (max 5)
  keywords?: string[];
}

export interface PreviewOutlineRequest {
  profileId?: string;
  topic: string;
  description?: string;
  difficultyStart: DifficultyLevel;
  difficultyEnd: DifficultyLevel;
  structure: {
    total_duration_minutes: number;
    number_of_sections: number;
    lectures_per_section: number;
    random_structure: boolean;
  };
  language: string;
  // RAG document references (Phase 2)
  document_ids?: string[];
  // Custom keywords for context refinement (max 5)
  keywords?: string[];
}

export interface CourseJob {
  jobId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  currentStage: CourseStage;
  progress: number;
  message: string;
  outline?: CourseOutline;
  lecturesTotal: number;
  lecturesCompleted: number;
  currentLectureTitle?: string;
  outputUrls: string[];
  zipUrl?: string;
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
  error?: string;
}

export interface DifficultyOption {
  id: DifficultyLevel;
  name: string;
  description: string;
}

export interface LessonElementOption {
  id: keyof LessonElementConfig;
  name: string;
  description: string;
  default: boolean;
  readonly?: boolean;
}

// Quiz configuration
export type QuizFrequency = 'per_lecture' | 'per_section' | 'end_of_course' | 'custom';
export type QuizQuestionType = 'multiple_choice' | 'multi_select' | 'true_false' | 'fill_blank' | 'matching';

export interface QuizConfig {
  enabled: boolean;
  frequency: QuizFrequency;
  customFrequency?: number;
  questionsPerQuiz: number;
  questionTypes: QuizQuestionType[];
  passingScore: number;
  showExplanations: boolean;
  allowRetry: boolean;
}

// Adaptive lesson elements
export interface AdaptiveElementsConfig {
  commonElements: Record<string, boolean>;
  categoryElements: Record<string, boolean>;
  useAiSuggestions: boolean;
}

// Detected category from AI
export interface DetectedCategory {
  category: ProfileCategory;
  confidence: number;
  domain?: string;
  domainOptions?: string[];
  keywords?: string[];
  tools?: string[];
}

// Form state types
export interface CourseFormState {
  profileId: string;
  topic: string;
  description: string;
  difficultyStart: DifficultyLevel;
  difficultyEnd: DifficultyLevel;
  structure: CourseStructureConfig;
  lessonElements: LessonElementConfig;
  // NEW: Adaptive elements based on category
  adaptiveElements: AdaptiveElementsConfig;
  // NEW: Quiz configuration (required)
  quizConfig: QuizConfig;
  language: string;
  voiceId: string;
  style: string;
  typingSpeed: string;
  includeAvatar: boolean;
  avatarId: string;
  // Context questions
  contextAnswers: Record<string, string>;
  context: CourseContext | null;
  // RAG Documents (Phase 2 - Legacy)
  documents: Document[];
  // Source Library IDs (Multi-Source RAG)
  sourceIds?: string[];
  // Auto-detected category from topic (Phase 1)
  detectedCategory: DetectedCategory | null;
  // Custom keywords for context refinement (max 5)
  customKeywords: string[];
}

// Helper to convert frontend state to API request
export function toApiRequest(state: CourseFormState, outline?: CourseOutline): GenerateCourseRequest {
  // Extract document IDs from uploaded documents (legacy)
  const documentIds = state.documents
    .filter(doc => doc.status === 'ready')
    .map(doc => doc.id);

  // Combine legacy document IDs with source library IDs
  const allSourceIds = [
    ...documentIds,
    ...(state.sourceIds || []),
  ];

  return {
    profileId: state.profileId,
    topic: state.topic,
    description: state.description || undefined,
    difficultyStart: state.difficultyStart,
    difficultyEnd: state.difficultyEnd,
    structure: {
      total_duration_minutes: state.structure.totalDurationMinutes,
      number_of_sections: state.structure.numberOfSections,
      lectures_per_section: state.structure.lecturesPerSection,
      random_structure: state.structure.randomStructure,
    },
    lessonElements: {
      concept_intro: state.lessonElements.conceptIntro,
      diagram_schema: state.lessonElements.diagramSchema,
      code_typing: state.lessonElements.codeTyping,
      code_execution: state.lessonElements.codeExecution,
      voiceover_explanation: state.lessonElements.voiceoverExplanation,
      curriculum_slide: state.lessonElements.curriculumSlide,
    },
    language: state.language,
    voiceId: state.voiceId,
    style: state.style,
    typingSpeed: state.typingSpeed,
    includeAvatar: state.includeAvatar,
    avatarId: state.avatarId || undefined,
    approvedOutline: outline,
    // RAG document IDs (includes both legacy and source library IDs)
    document_ids: allSourceIds.length > 0 ? allSourceIds : undefined,
    // Custom keywords for context refinement
    keywords: state.customKeywords.length > 0 ? state.customKeywords : undefined,
  };
}

// Helper to convert frontend state to preview request
export function toPreviewRequest(state: CourseFormState): PreviewOutlineRequest {
  // Extract document IDs from uploaded documents
  const documentIds = state.documents
    .filter(doc => doc.status === 'ready')
    .map(doc => doc.id);

  return {
    profileId: state.profileId || undefined,
    topic: state.topic,
    description: state.description || undefined,
    difficultyStart: state.difficultyStart,
    difficultyEnd: state.difficultyEnd,
    structure: {
      total_duration_minutes: state.structure.totalDurationMinutes,
      number_of_sections: state.structure.numberOfSections,
      lectures_per_section: state.structure.lecturesPerSection,
      random_structure: state.structure.randomStructure,
    },
    language: state.language,
    // RAG document IDs (Phase 2)
    document_ids: documentIds.length > 0 ? documentIds : undefined,
    // Custom keywords for context refinement
    keywords: state.customKeywords.length > 0 ? state.customKeywords : undefined,
  };
}

// Default quiz configuration
export const defaultQuizConfig: QuizConfig = {
  enabled: true,
  frequency: 'per_section',
  questionsPerQuiz: 5,
  questionTypes: ['multiple_choice', 'true_false'],
  passingScore: 70,
  showExplanations: true,
  allowRetry: true,
};

// Default adaptive elements
export const defaultAdaptiveElements: AdaptiveElementsConfig = {
  commonElements: {
    concept_intro: true,
    voiceover: true,
    curriculum_slide: true,
    conclusion: true,
    quiz: true,
  },
  categoryElements: {},
  useAiSuggestions: true,
};

// Default form state
export const defaultCourseFormState: CourseFormState = {
  profileId: '',
  topic: '',
  description: '',
  difficultyStart: 'beginner',
  difficultyEnd: 'intermediate',
  structure: {
    totalDurationMinutes: 60,
    numberOfSections: 5,
    lecturesPerSection: 3,
    randomStructure: false,
  },
  lessonElements: {
    conceptIntro: true,
    diagramSchema: true,
    codeTyping: true,
    codeExecution: false,
    voiceoverExplanation: true,
    curriculumSlide: true,
  },
  adaptiveElements: defaultAdaptiveElements,
  quizConfig: defaultQuizConfig,
  language: 'fr',
  voiceId: 'alloy',
  style: 'dark',
  typingSpeed: 'natural',
  includeAvatar: false,
  avatarId: '',
  contextAnswers: {},
  context: null,
  documents: [],
  sourceIds: [],
  detectedCategory: null,
};
