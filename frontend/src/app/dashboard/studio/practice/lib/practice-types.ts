/**
 * Practice Agent Types
 * TypeScript types for the practice training system
 */

// Enums
export enum DifficultyLevel {
  BEGINNER = 'beginner',
  INTERMEDIATE = 'intermediate',
  ADVANCED = 'advanced',
  EXPERT = 'expert',
}

export enum ExerciseCategory {
  DOCKER = 'docker',
  KUBERNETES = 'kubernetes',
  CI_CD = 'ci_cd',
  TERRAFORM = 'terraform',
  ANSIBLE = 'ansible',
  LINUX = 'linux',
  NETWORKING = 'networking',
  SECURITY = 'security',
  MONITORING = 'monitoring',
  PYTHON = 'python',
  JAVASCRIPT = 'javascript',
  GO = 'go',
  RUST = 'rust',
}

export enum ExerciseType {
  CODING = 'coding',
  DEBUGGING = 'debugging',
  CONFIGURATION = 'configuration',
  TROUBLESHOOTING = 'troubleshooting',
  QUIZ = 'quiz',
  PROJECT = 'project',
}

export enum SessionStatus {
  ACTIVE = 'active',
  PAUSED = 'paused',
  COMPLETED = 'completed',
}

export enum SandboxType {
  DOCKER = 'docker',
  KUBERNETES = 'kubernetes',
  PYTHON = 'python',
  BASH = 'bash',
}

// Interfaces
export interface ValidationCheck {
  name: string;
  check_type: string;
  patterns?: string[];
  expected_output?: string;
  points: number;
  required?: boolean;
}

export interface Exercise {
  id: string;
  title: string;
  description: string;
  instructions: string;
  difficulty: DifficultyLevel;
  type: ExerciseType;
  category: ExerciseCategory;
  tags: string[];
  starter_code?: string;
  hints: string[];
  solution?: string;
  solution_explanation?: string;
  validation_checks: ValidationCheck[];
  sandbox_type: SandboxType;
  estimated_minutes: number;
  points: number;
  course_id?: string;
  lecture_id?: string;
}

export interface ExerciseAttempt {
  exercise_id: string;
  code: string;
  passed: boolean;
  score: number;
  feedback: string;
  timestamp: string;
}

export interface LearnerProgress {
  user_id: string;
  total_points: number;
  exercises_completed: number;
  exercises_by_category: Record<string, number>;
  exercises_by_difficulty: Record<string, number>;
  current_streak: number;
  longest_streak: number;
  badges: string[];
  last_active: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
}

export interface PracticeSession {
  id: string;
  user_id: string;
  course_id?: string;
  status: SessionStatus;
  difficulty_preference: DifficultyLevel;
  categories_focus?: ExerciseCategory[];
  current_exercise?: Exercise;
  exercises_completed: string[];
  hints_used_total: number;
  points_earned: number;
  pair_programming_enabled: boolean;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

export interface SessionSummary {
  session_id: string;
  duration_minutes: number;
  exercises_attempted: number;
  exercises_passed: number;
  total_points: number;
  hints_used: number;
  categories_practiced: string[];
  improvement_areas: string[];
}

// API Request/Response Types
export interface CreateSessionRequest {
  user_id: string;
  course_id?: string;
  difficulty_preference?: DifficultyLevel;
  categories_focus?: ExerciseCategory[];
  pair_programming_enabled?: boolean;
}

export interface CreateSessionResponse {
  session_id: string;
  session: PracticeSession;
  welcome_message: string;
  suggested_exercise?: Exercise;
}

export interface InteractionRequest {
  session_id: string;
  message?: string;
  code?: string;
}

export interface InteractionResponse {
  response: string;
  exercise?: Exercise;
  assessment?: AssessmentResult;
  next_action?: string;
}

export interface SubmitCodeRequest {
  code: string;
  language?: string;
}

export interface SubmitCodeResponse {
  passed: boolean;
  score: number;
  feedback: string;
  checks_passed: string[];
  checks_failed: string[];
  execution_output?: string;
  next_exercise?: Exercise;
}

export interface HintRequest {
  hint_level?: number;
}

export interface HintResponse {
  hint: string;
  hint_number: number;
  hints_remaining: number;
  points_deduction: number;
}

export interface AssessmentResult {
  passed: boolean;
  score: number;
  max_score: number;
  checks_passed: string[];
  checks_failed: string[];
  summary_feedback: string;
  detailed_feedback?: string;
}

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  earned_at?: string;
}

export interface LeaderboardEntry {
  user_id: string;
  username: string;
  total_points: number;
  exercises_completed: number;
  rank: number;
}

// WebSocket Message Types
export interface WSMessage {
  type: 'message' | 'code' | 'hint' | 'ping' | 'pair_start' | 'pair_stop';
  content?: string;
  code?: string;
}

export interface WSResponse {
  type: 'response' | 'evaluation' | 'hint' | 'pong' | 'error';
  content?: string;
  exercise?: Exercise;
  assessment?: AssessmentResult;
  passed?: boolean;
  hint_number?: number;
}

// Helper functions
export function getDifficultyLabel(difficulty: DifficultyLevel): string {
  const labels: Record<DifficultyLevel, string> = {
    [DifficultyLevel.BEGINNER]: 'Débutant',
    [DifficultyLevel.INTERMEDIATE]: 'Intermédiaire',
    [DifficultyLevel.ADVANCED]: 'Avancé',
    [DifficultyLevel.EXPERT]: 'Expert',
  };
  return labels[difficulty] || difficulty;
}

export function getDifficultyColor(difficulty: DifficultyLevel): string {
  const colors: Record<DifficultyLevel, string> = {
    [DifficultyLevel.BEGINNER]: 'text-green-500 bg-green-500/10',
    [DifficultyLevel.INTERMEDIATE]: 'text-yellow-500 bg-yellow-500/10',
    [DifficultyLevel.ADVANCED]: 'text-orange-500 bg-orange-500/10',
    [DifficultyLevel.EXPERT]: 'text-red-500 bg-red-500/10',
  };
  return colors[difficulty] || 'text-gray-500 bg-gray-500/10';
}

export function getCategoryLabel(category: ExerciseCategory): string {
  const labels: Record<ExerciseCategory, string> = {
    [ExerciseCategory.DOCKER]: 'Docker',
    [ExerciseCategory.KUBERNETES]: 'Kubernetes',
    [ExerciseCategory.CI_CD]: 'CI/CD',
    [ExerciseCategory.TERRAFORM]: 'Terraform',
    [ExerciseCategory.ANSIBLE]: 'Ansible',
    [ExerciseCategory.LINUX]: 'Linux',
    [ExerciseCategory.NETWORKING]: 'Networking',
    [ExerciseCategory.SECURITY]: 'Sécurité',
    [ExerciseCategory.MONITORING]: 'Monitoring',
    [ExerciseCategory.PYTHON]: 'Python',
    [ExerciseCategory.JAVASCRIPT]: 'JavaScript',
    [ExerciseCategory.GO]: 'Go',
    [ExerciseCategory.RUST]: 'Rust',
  };
  return labels[category] || category;
}

export function getCategoryIcon(category: ExerciseCategory): string {
  const icons: Record<ExerciseCategory, string> = {
    [ExerciseCategory.DOCKER]: '🐳',
    [ExerciseCategory.KUBERNETES]: '☸️',
    [ExerciseCategory.CI_CD]: '🔄',
    [ExerciseCategory.TERRAFORM]: '🏗️',
    [ExerciseCategory.ANSIBLE]: '📦',
    [ExerciseCategory.LINUX]: '🐧',
    [ExerciseCategory.NETWORKING]: '🌐',
    [ExerciseCategory.SECURITY]: '🔒',
    [ExerciseCategory.MONITORING]: '📊',
    [ExerciseCategory.PYTHON]: '🐍',
    [ExerciseCategory.JAVASCRIPT]: '💛',
    [ExerciseCategory.GO]: '🔵',
    [ExerciseCategory.RUST]: '🦀',
  };
  return icons[category] || '📚';
}

export function formatDuration(minutes: number): string {
  if (minutes < 60) {
    return `${minutes} min`;
  }
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  if (remainingMinutes === 0) {
    return `${hours}h`;
  }
  return `${hours}h ${remainingMinutes}min`;
}

export function formatPoints(points: number): string {
  if (points >= 1000) {
    return `${(points / 1000).toFixed(1)}k`;
  }
  return points.toString();
}
