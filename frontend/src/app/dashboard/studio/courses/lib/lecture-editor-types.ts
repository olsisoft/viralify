/**
 * Lecture Editor TypeScript Types
 * Types for editing lecture components (slides, voiceover, diagrams)
 */

// Slide types matching backend SlideType enum
export type SlideType =
  | 'title'
  | 'content'
  | 'code'
  | 'code_demo'
  | 'diagram'
  | 'split'
  | 'terminal'
  | 'conclusion';

// Component status
export type ComponentStatus = 'pending' | 'generating' | 'completed' | 'failed' | 'edited';

// Code block within a slide
export interface CodeBlockComponent {
  id: string;
  language: string;
  code: string;
  filename?: string;
  highlightLines: number[];
  executionOrder: number;
  expectedOutput?: string;
  actualOutput?: string;
  showLineNumbers: boolean;
}

// Editable slide component
export interface SlideComponent {
  id: string;
  index: number;
  type: SlideType;
  status: ComponentStatus;
  // Content
  title?: string;
  subtitle?: string;
  content?: string;
  bulletPoints: string[];
  codeBlocks: CodeBlockComponent[];
  // Voiceover
  voiceoverText: string;
  // Timing
  duration: number;
  transition: string;
  // Diagram specific
  diagramType?: string;
  diagramData?: Record<string, unknown>;
  // Generated assets
  imageUrl?: string;
  animationUrl?: string;
  // Edit tracking
  isEdited: boolean;
  editedAt?: string;
  editedFields: string[];
  // Error info
  error?: string;
}

// Voiceover component
export interface VoiceoverComponent {
  id: string;
  status: ComponentStatus;
  audioUrl?: string;
  durationSeconds: number;
  voiceId: string;
  voiceSettings: Record<string, unknown>;
  fullText: string;
  isCustomAudio: boolean;
  originalFilename?: string;
  isEdited: boolean;
  editedAt?: string;
  error?: string;
}

// Complete lecture components
export interface LectureComponents {
  id: string;
  lectureId: string;
  jobId: string;
  slides: SlideComponent[];
  voiceover?: VoiceoverComponent;
  totalDuration: number;
  generationParams: Record<string, unknown>;
  presentationJobId?: string;
  videoUrl?: string;
  status: ComponentStatus;
  isEdited: boolean;
  createdAt: string;
  updatedAt: string;
  error?: string;
}

// API Request types
export interface UpdateSlideRequest {
  title?: string;
  subtitle?: string;
  content?: string;
  bulletPoints?: string[];
  voiceoverText?: string;
  duration?: number;
  codeBlocks?: CodeBlockComponent[];
  diagramType?: string;
  diagramData?: Record<string, unknown>;
}

export interface RegenerateSlideRequest {
  regenerateImage: boolean;
  regenerateAnimation: boolean;
  useEditedContent: boolean;
}

export interface RegenerateLectureRequest {
  useEditedComponents: boolean;
  regenerateVoiceover: boolean;
  voiceId?: string;
}

export interface RegenerateVoiceoverRequest {
  voiceId?: string;
  voiceSettings?: Record<string, unknown>;
}

export interface RecomposeVideoRequest {
  quality: 'low' | 'medium' | 'high';
  includeTransitions: boolean;
}

// API Response types
export interface LectureComponentsResponse {
  lectureId: string;
  jobId: string;
  status: ComponentStatus;
  slides: SlideComponent[];
  voiceover?: VoiceoverComponent;
  totalDuration: number;
  videoUrl?: string;
  isEdited: boolean;
  createdAt: string;
  updatedAt: string;
  error?: string;
}

export interface SlideComponentResponse {
  slide: SlideComponent;
  lectureId: string;
  message: string;
}

export interface RegenerateResponse {
  success: boolean;
  message: string;
  jobId?: string;
  result?: Record<string, unknown>;
}

// Helper functions
export function getSlideTypeLabel(type: SlideType): string {
  const labels: Record<SlideType, string> = {
    title: 'Titre',
    content: 'Contenu',
    code: 'Code',
    code_demo: 'D\u00e9mo Code',
    diagram: 'Diagramme',
    split: 'Split',
    terminal: 'Terminal',
    conclusion: 'Conclusion',
  };
  return labels[type] || type;
}

export function getSlideTypeIcon(type: SlideType): string {
  const icons: Record<SlideType, string> = {
    title: '\ud83c\udfa8',
    content: '\ud83d\udcdd',
    code: '\ud83d\udcbb',
    code_demo: '\u25b6\ufe0f',
    diagram: '\ud83d\udcc8',
    split: '\u2195\ufe0f',
    terminal: '\ud83d\udcdf',
    conclusion: '\u2705',
  };
  return icons[type] || '\ud83d\udcc4';
}

export function getStatusColor(status: ComponentStatus): string {
  const colors: Record<ComponentStatus, string> = {
    pending: 'text-gray-500',
    generating: 'text-blue-500',
    completed: 'text-green-500',
    failed: 'text-red-500',
    edited: 'text-yellow-500',
  };
  return colors[status] || 'text-gray-500';
}

export function getStatusLabel(status: ComponentStatus): string {
  const labels: Record<ComponentStatus, string> = {
    pending: 'En attente',
    generating: 'G\u00e9n\u00e9ration...',
    completed: 'Termin\u00e9',
    failed: '\u00c9chou\u00e9',
    edited: 'Modifi\u00e9',
  };
  return labels[status] || status;
}

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function formatTotalDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  if (mins === 0) {
    return `${secs}s`;
  }
  return `${mins}m ${secs}s`;
}
