/**
 * Course Generator TypeScript Types
 */

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

export type ProfileCategory =
  | 'business'
  | 'tech'
  | 'creative'
  | 'health'
  | 'education'
  | 'lifestyle';

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
  category?: ProfileCategory;
  contextSummary?: string;
  difficultyStart: DifficultyLevel;
  difficultyEnd: DifficultyLevel;
  totalDurationMinutes: number;
  sections: Section[];
}

// Context Question Types
export interface ContextQuestion {
  id: string;
  question: string;
  type: 'select' | 'text' | 'multiselect';
  options?: string[];
  placeholder?: string;
  required?: boolean;
}

export interface ContextQuestionsResponse {
  category: ProfileCategory;
  baseQuestions: ContextQuestion[];
  aiQuestions: ContextQuestion[];
}

// Course Context (replaces simple language field)
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
  context?: {
    category: ProfileCategory;
    profile_niche: string;
    profile_tone: string;
    profile_audience_level: string;
    profile_language_level: string;
    profile_primary_goal: string;
    profile_audience_description: string;
    context_answers: Record<string, string>;
    specific_tools?: string;
    practical_focus?: string;
    expected_outcome?: string;
  };
  voiceId: string;
  style: string;
  typingSpeed: string;
  includeAvatar: boolean;
  avatarId?: string;
  approvedOutline?: CourseOutline;
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
  context?: {
    category: ProfileCategory;
    profile_niche: string;
    profile_tone: string;
    profile_audience_level: string;
    profile_language_level: string;
    profile_primary_goal: string;
    profile_audience_description: string;
    context_answers: Record<string, string>;
  };
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

// Form state types
export interface CourseFormState {
  profileId: string;
  topic: string;
  description: string;
  difficultyStart: DifficultyLevel;
  difficultyEnd: DifficultyLevel;
  structure: CourseStructureConfig;
  lessonElements: LessonElementConfig;
  // Context replaces language
  context: CourseContext | null;
  contextAnswers: Record<string, string>;
  // Presentation options
  voiceId: string;
  style: string;
  typingSpeed: string;
  includeAvatar: boolean;
  avatarId: string;
}

// Helper to convert frontend state to API request
export function toApiRequest(
  state: CourseFormState,
  outline?: CourseOutline
): GenerateCourseRequest {
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
    context: state.context
      ? {
          category: state.context.category,
          profile_niche: state.context.profileNiche,
          profile_tone: state.context.profileTone,
          profile_audience_level: state.context.profileAudienceLevel,
          profile_language_level: state.context.profileLanguageLevel,
          profile_primary_goal: state.context.profilePrimaryGoal,
          profile_audience_description: state.context.profileAudienceDescription,
          context_answers: state.contextAnswers,
          specific_tools: state.context.specificTools,
          practical_focus: state.context.practicalFocus,
          expected_outcome: state.context.expectedOutcome,
        }
      : undefined,
    voiceId: state.voiceId,
    style: state.style,
    typingSpeed: state.typingSpeed,
    includeAvatar: state.includeAvatar,
    avatarId: state.avatarId || undefined,
    approvedOutline: outline,
  };
}

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
  context: null,
  contextAnswers: {},
  voiceId: 'alloy',
  style: 'dark',
  typingSpeed: 'natural',
  includeAvatar: false,
  avatarId: '',
};

// Category display info
export const CATEGORY_INFO: Record<
  ProfileCategory,
  { label: string; icon: string; color: string }
> = {
  business: { label: 'Business', icon: '📊', color: 'blue' },
  tech: { label: 'Tech', icon: '💻', color: 'purple' },
  creative: { label: 'Creative', icon: '🎨', color: 'pink' },
  health: { label: 'Health', icon: '🏋️', color: 'green' },
  education: { label: 'Education', icon: '📚', color: 'yellow' },
  lifestyle: { label: 'Lifestyle', icon: '✨', color: 'orange' },
};
