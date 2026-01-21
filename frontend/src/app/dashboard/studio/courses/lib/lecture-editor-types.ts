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
  | 'conclusion'
  | 'media'; // New type for inserted media (image/video)

// Media types for insertion
export type MediaType = 'image' | 'video' | 'audio';

// Drag & Drop item type
export interface DragItem {
  id: string;
  index: number;
  type: 'slide';
}

// Editor action for undo/redo
export type EditorActionType =
  | 'update_slide'
  | 'reorder_slide'
  | 'insert_media'
  | 'delete_slide'
  | 'update_voiceover';

export interface EditorAction {
  type: EditorActionType;
  timestamp: number;
  previousState: unknown;
  newState: unknown;
  slideId?: string;
}

// Quick action button type
export interface QuickAction {
  id: string;
  icon: string;
  label: string;
  type: MediaType | 'regenerate';
  tooltip: string;
}

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
    code_demo: 'Démo Code',
    diagram: 'Diagramme',
    split: 'Split',
    terminal: 'Terminal',
    conclusion: 'Conclusion',
    media: 'Média',
  };
  return labels[type] || type;
}

export function getSlideTypeIcon(type: SlideType): string {
  const icons: Record<SlideType, string> = {
    title: '🎨',
    content: '📝',
    code: '💻',
    code_demo: '▶️',
    diagram: '📈',
    split: '↕️',
    terminal: '📟',
    conclusion: '✅',
    media: '🎬',
  };
  return icons[type] || '📄';
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

// Quick actions configuration
export const QUICK_ACTIONS: QuickAction[] = [
  { id: 'add-image', icon: '🖼️', label: 'Image', type: 'image', tooltip: 'Ajouter une image' },
  { id: 'add-video', icon: '🎬', label: 'Vidéo', type: 'video', tooltip: 'Ajouter une vidéo' },
  { id: 'add-audio', icon: '🎵', label: 'Audio', type: 'audio', tooltip: 'Remplacer l\'audio' },
  { id: 'regenerate', icon: '🔄', label: 'Régénérer', type: 'regenerate', tooltip: 'Régénérer le slide' },
];

// Keyboard shortcuts configuration
export const KEYBOARD_SHORTCUTS = {
  DELETE: ['Delete', 'Backspace'],
  UNDO: ['Control+z', 'Meta+z'],
  REDO: ['Control+y', 'Meta+y', 'Control+Shift+z', 'Meta+Shift+z'],
  SAVE: ['Control+s', 'Meta+s'],
  ESCAPE: ['Escape'],
  PLAY: ['Space'],
} as const;

// Media upload configuration
export interface MediaUploadConfig {
  type: MediaType;
  accept: string;
  maxSizeMB: number;
}

export const MEDIA_UPLOAD_CONFIG: Record<MediaType, MediaUploadConfig> = {
  image: { type: 'image', accept: 'image/jpeg,image/png,image/gif,image/webp', maxSizeMB: 10 },
  video: { type: 'video', accept: 'video/mp4,video/webm,video/mov', maxSizeMB: 100 },
  audio: { type: 'audio', accept: 'audio/mp3,audio/wav,audio/m4a,audio/ogg', maxSizeMB: 50 },
};

// Insert media request
export interface InsertMediaRequest {
  type: MediaType;
  insertAfterSlideId?: string;
  file?: File;
  url?: string;
  duration?: number;
}

// Reorder slides request
export interface ReorderSlidesRequest {
  slideId: string;
  newIndex: number;
}

// Delete slide request
export interface DeleteSlideRequest {
  slideId: string;
}

// Editor state for undo/redo
export interface EditorHistory {
  past: EditorAction[];
  future: EditorAction[];
  canUndo: boolean;
  canRedo: boolean;
}

// Media slide content
export interface MediaSlideContent {
  mediaType: MediaType;
  mediaUrl: string;
  thumbnailUrl?: string;
  originalFilename?: string;
  duration?: number;
  width?: number;
  height?: number;
}
