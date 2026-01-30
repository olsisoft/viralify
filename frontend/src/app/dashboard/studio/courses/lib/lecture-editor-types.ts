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
  | 'media'; // Type for inserted media (image/video)

// Media slide fields
export interface MediaSlideFields {
  mediaType?: MediaType;
  mediaUrl?: string;
  mediaThumbnailUrl?: string;
  mediaOriginalFilename?: string;
}

// Media types for insertion
export type MediaType = 'image' | 'video' | 'audio';

// =============================================================================
// SLIDE ELEMENT SYSTEM (for positionable images, text, shapes)
// User doesn't see "layers" - just drag and drop on canvas
// =============================================================================

export type ElementType = 'image' | 'text_block' | 'shape';
export type ElementFit = 'cover' | 'contain' | 'fill';
export type ShapeType = 'rectangle' | 'circle' | 'rounded_rect' | 'arrow' | 'line';

export interface ImageElementContent {
  url: string;
  originalFilename?: string;
  fit: ElementFit;
  opacity: number;
  borderRadius: number;
  crop?: { x: number; y: number; width: number; height: number };
}

export interface TextBlockContent {
  text: string;
  fontSize: number;
  fontWeight: 'normal' | 'bold';
  fontFamily: string;
  color: string;
  backgroundColor?: string;
  textAlign: 'left' | 'center' | 'right';
  lineHeight: number;
  padding: number;
}

export interface ShapeContent {
  shape: ShapeType;
  fillColor: string;
  strokeColor?: string;
  strokeWidth: number;
  opacity: number;
  borderRadius: number;
}

export interface SlideElement {
  id: string;
  type: ElementType;
  // Position (% of slide, 0-100)
  x: number;
  y: number;
  width: number;
  height: number;
  // Transform
  rotation: number;
  zIndex: number;
  // State
  locked: boolean;
  visible: boolean;
  // Content (one based on type)
  imageContent?: ImageElementContent;
  textContent?: TextBlockContent;
  shapeContent?: ShapeContent;
  // Timestamps
  createdAt?: string;
  updatedAt?: string;
}

export interface AddElementRequest {
  type: ElementType;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  imageContent?: ImageElementContent;
  textContent?: TextBlockContent;
  shapeContent?: ShapeContent;
}

export interface UpdateElementRequest {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  rotation?: number;
  locked?: boolean;
  visible?: boolean;
  imageContent?: ImageElementContent;
  textContent?: TextBlockContent;
  shapeContent?: ShapeContent;
}

// Default element sizes (% of slide)
export const DEFAULT_ELEMENT_SIZES: Record<ElementType, { width: number; height: number }> = {
  image: { width: 30, height: 30 },
  text_block: { width: 40, height: 15 },
  shape: { width: 20, height: 20 },
};

// Default positions (center of slide)
export const DEFAULT_ELEMENT_POSITION = { x: 35, y: 35 };

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
  // Media slide specific
  mediaType?: MediaType;
  mediaUrl?: string;
  mediaThumbnailUrl?: string;
  mediaOriginalFilename?: string;
  // Positionable elements (images, text, shapes)
  elements: SlideElement[];
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
  quality: '720p' | '1080p' | '4k';
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

// ============================================================================
// SUBTITLES TYPES
// ============================================================================

export interface SubtitleCue {
  id: string;
  startTime: number; // in seconds
  endTime: number;
  text: string;
  // Styling
  position: SubtitlePosition;
  style: SubtitleStyle;
}

export type SubtitlePosition = 'top' | 'middle' | 'bottom';

export interface SubtitleStyle {
  fontFamily: string;
  fontSize: number; // in pixels
  fontWeight: 'normal' | 'bold';
  fontStyle: 'normal' | 'italic';
  color: string; // hex color
  backgroundColor: string; // hex color with alpha
  textAlign: 'left' | 'center' | 'right';
  textShadow: boolean;
  outline: boolean;
  outlineColor: string;
}

export interface SubtitleTrack {
  id: string;
  language: string;
  label: string;
  cues: SubtitleCue[];
  isDefault: boolean;
}

export const DEFAULT_SUBTITLE_STYLE: SubtitleStyle = {
  fontFamily: 'Arial',
  fontSize: 24,
  fontWeight: 'bold',
  fontStyle: 'normal',
  color: '#FFFFFF',
  backgroundColor: 'rgba(0, 0, 0, 0.75)',
  textAlign: 'center',
  textShadow: true,
  outline: true,
  outlineColor: '#000000',
};

export const SUBTITLE_FONTS = [
  'Arial',
  'Helvetica',
  'Verdana',
  'Roboto',
  'Open Sans',
  'Montserrat',
  'Lato',
  'Poppins',
  'Inter',
  'Georgia',
  'Times New Roman',
];

// ============================================================================
// AUDIO TYPES
// ============================================================================

export interface AudioTrack {
  id: string;
  name: string;
  type: 'voiceover' | 'music' | 'sfx';
  url: string;
  duration: number;
  startTime: number; // offset in timeline
  volume: number; // 0-1
  isMuted: boolean;
  isSolo: boolean;
  fadeIn: number; // duration in seconds
  fadeOut: number;
  waveformData?: number[]; // normalized waveform peaks
}

export interface AudioMixerState {
  masterVolume: number;
  tracks: AudioTrack[];
  voiceoverVolume: number;
  musicVolume: number;
  sfxVolume: number;
}

export interface AudioSegment {
  id: string;
  trackId: string;
  startTime: number;
  endTime: number;
  volume: number;
  fadeIn: number;
  fadeOut: number;
}

// ============================================================================
// TRANSITION TYPES
// ============================================================================

export type TransitionType =
  | 'none'
  | 'fade'
  | 'dissolve'
  | 'slide-left'
  | 'slide-right'
  | 'slide-up'
  | 'slide-down'
  | 'zoom-in'
  | 'zoom-out'
  | 'wipe-left'
  | 'wipe-right'
  | 'wipe-up'
  | 'wipe-down'
  | 'blur'
  | 'flash';

export interface Transition {
  id: string;
  type: TransitionType;
  duration: number; // in seconds (0.25 - 3)
  easing: 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out';
}

export interface SlideTransition {
  slideId: string;
  inTransition?: Transition;
  outTransition?: Transition;
}

export const TRANSITION_PRESETS: Record<TransitionType, { label: string; icon: string; description: string }> = {
  'none': { label: 'Aucune', icon: '⏹️', description: 'Pas de transition' },
  'fade': { label: 'Fondu', icon: '🌅', description: 'Fondu progressif' },
  'dissolve': { label: 'Dissolution', icon: '✨', description: 'Dissolution douce' },
  'slide-left': { label: 'Glissement gauche', icon: '⬅️', description: 'Glisse vers la gauche' },
  'slide-right': { label: 'Glissement droite', icon: '➡️', description: 'Glisse vers la droite' },
  'slide-up': { label: 'Glissement haut', icon: '⬆️', description: 'Glisse vers le haut' },
  'slide-down': { label: 'Glissement bas', icon: '⬇️', description: 'Glisse vers le bas' },
  'zoom-in': { label: 'Zoom avant', icon: '🔍', description: 'Zoom progressif' },
  'zoom-out': { label: 'Zoom arrière', icon: '🔎', description: 'Dézoom progressif' },
  'wipe-left': { label: 'Balayage gauche', icon: '◀️', description: 'Balayage vers la gauche' },
  'wipe-right': { label: 'Balayage droite', icon: '▶️', description: 'Balayage vers la droite' },
  'wipe-up': { label: 'Balayage haut', icon: '🔼', description: 'Balayage vers le haut' },
  'wipe-down': { label: 'Balayage bas', icon: '🔽', description: 'Balayage vers le bas' },
  'blur': { label: 'Flou', icon: '💨', description: 'Transition floue' },
  'flash': { label: 'Flash', icon: '⚡', description: 'Flash lumineux' },
};

// ============================================================================
// VISUAL EFFECTS TYPES
// ============================================================================

export interface KenBurnsEffect {
  enabled: boolean;
  startScale: number; // 1.0 - 2.0
  endScale: number;
  startPosition: { x: number; y: number }; // -1 to 1 (center = 0)
  endPosition: { x: number; y: number };
}

export interface ColorGrading {
  brightness: number; // -100 to 100
  contrast: number; // -100 to 100
  saturation: number; // -100 to 100
  temperature: number; // -100 (cool) to 100 (warm)
  tint: number; // -100 (green) to 100 (magenta)
  highlights: number; // -100 to 100
  shadows: number; // -100 to 100
  vignette: number; // 0 to 100
}

export type FilterPreset =
  | 'none'
  | 'cinematic'
  | 'vintage'
  | 'noir'
  | 'vivid'
  | 'muted'
  | 'warm'
  | 'cool'
  | 'dramatic'
  | 'soft';

export interface VisualEffect {
  id: string;
  slideId: string;
  kenBurns?: KenBurnsEffect;
  colorGrading?: ColorGrading;
  filterPreset: FilterPreset;
  speed: number; // 0.25 to 4.0
  reverse: boolean;
}

export const FILTER_PRESETS: Record<FilterPreset, { label: string; grading: Partial<ColorGrading> }> = {
  'none': { label: 'Aucun', grading: {} },
  'cinematic': { label: 'Cinématique', grading: { contrast: 15, saturation: -10, temperature: -5, vignette: 30 } },
  'vintage': { label: 'Vintage', grading: { saturation: -20, temperature: 15, contrast: 10, vignette: 40 } },
  'noir': { label: 'Noir & Blanc', grading: { saturation: -100, contrast: 20 } },
  'vivid': { label: 'Vivide', grading: { saturation: 30, contrast: 10, brightness: 5 } },
  'muted': { label: 'Désaturé', grading: { saturation: -30, contrast: -5 } },
  'warm': { label: 'Chaud', grading: { temperature: 25, tint: 5 } },
  'cool': { label: 'Froid', grading: { temperature: -25, tint: -5 } },
  'dramatic': { label: 'Dramatique', grading: { contrast: 30, shadows: -20, highlights: 20, vignette: 50 } },
  'soft': { label: 'Doux', grading: { contrast: -10, brightness: 10, saturation: -10 } },
};

export const DEFAULT_COLOR_GRADING: ColorGrading = {
  brightness: 0,
  contrast: 0,
  saturation: 0,
  temperature: 0,
  tint: 0,
  highlights: 0,
  shadows: 0,
  vignette: 0,
};

// ============================================================================
// OVERLAY TYPES
// ============================================================================

export type OverlayType = 'text' | 'lower-third' | 'callout' | 'shape' | 'image' | 'watermark';

export interface OverlayPosition {
  x: number; // 0-100 percentage
  y: number;
  width: number;
  height: number;
  rotation: number; // degrees
}

export interface OverlayTiming {
  startTime: number;
  endTime: number;
  fadeIn: number;
  fadeOut: number;
}

export interface OverlayAnimation {
  type: 'none' | 'fade' | 'slide-in' | 'zoom' | 'bounce' | 'typewriter';
  direction?: 'left' | 'right' | 'top' | 'bottom';
  duration: number;
}

export interface TextOverlay {
  type: 'text';
  text: string;
  style: SubtitleStyle;
}

export interface LowerThirdOverlay {
  type: 'lower-third';
  title: string;
  subtitle: string;
  style: 'classic' | 'modern' | 'minimal' | 'bold';
  primaryColor: string;
  secondaryColor: string;
}

export interface CalloutOverlay {
  type: 'callout';
  text: string;
  shape: 'rectangle' | 'rounded' | 'pill' | 'arrow-left' | 'arrow-right';
  backgroundColor: string;
  borderColor: string;
  textColor: string;
  arrowPosition?: { x: number; y: number };
}

export interface ShapeOverlay {
  type: 'shape';
  shape: 'rectangle' | 'circle' | 'arrow' | 'line' | 'highlight';
  fillColor: string;
  strokeColor: string;
  strokeWidth: number;
  opacity: number;
}

export interface ImageOverlay {
  type: 'image';
  imageUrl: string;
  opacity: number;
  borderRadius: number;
}

export interface WatermarkOverlay {
  type: 'watermark';
  imageUrl?: string;
  text?: string;
  opacity: number;
  position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center';
}

export interface Overlay {
  id: string;
  slideId?: string; // null = global overlay
  position: OverlayPosition;
  timing: OverlayTiming;
  animation: OverlayAnimation;
  content: TextOverlay | LowerThirdOverlay | CalloutOverlay | ShapeOverlay | ImageOverlay | WatermarkOverlay;
  isLocked: boolean;
  isVisible: boolean;
}

export const LOWER_THIRD_STYLES: Record<string, { label: string; description: string }> = {
  'classic': { label: 'Classique', description: 'Style professionnel traditionnel' },
  'modern': { label: 'Moderne', description: 'Design épuré et minimaliste' },
  'minimal': { label: 'Minimal', description: 'Texte simple avec fond subtil' },
  'bold': { label: 'Bold', description: 'Design accrocheur et coloré' },
};

// ============================================================================
// EXPORT TYPES
// ============================================================================

export type ExportResolution = '720p' | '1080p' | '1440p' | '4k';
export type ExportFormat = 'mp4' | 'webm' | 'mov';
export type ExportAspectRatio = '16:9' | '9:16' | '1:1' | '4:3' | '4:5';
export type ExportQuality = 'draft' | 'standard' | 'high' | 'ultra';

export interface ExportSettings {
  resolution: ExportResolution;
  format: ExportFormat;
  aspectRatio: ExportAspectRatio;
  quality: ExportQuality;
  fps: 24 | 30 | 60;
  videoBitrate: number; // kbps
  audioBitrate: number; // kbps
  includeSubtitles: boolean;
  burnSubtitles: boolean; // hardcode subtitles into video
  watermark?: WatermarkOverlay;
}

export const RESOLUTION_CONFIG: Record<ExportResolution, { width: number; height: number; label: string }> = {
  '720p': { width: 1280, height: 720, label: 'HD (720p)' },
  '1080p': { width: 1920, height: 1080, label: 'Full HD (1080p)' },
  '1440p': { width: 2560, height: 1440, label: 'QHD (1440p)' },
  '4k': { width: 3840, height: 2160, label: '4K UHD' },
};

export const ASPECT_RATIO_CONFIG: Record<ExportAspectRatio, { label: string; description: string }> = {
  '16:9': { label: '16:9', description: 'YouTube, Standard' },
  '9:16': { label: '9:16', description: 'TikTok, Reels, Shorts' },
  '1:1': { label: '1:1', description: 'Instagram, Facebook' },
  '4:3': { label: '4:3', description: 'Présentation classique' },
  '4:5': { label: '4:5', description: 'Instagram Portrait' },
};

export const QUALITY_PRESETS: Record<ExportQuality, { videoBitrate: number; audioBitrate: number; label: string }> = {
  'draft': { videoBitrate: 2000, audioBitrate: 96, label: 'Brouillon (rapide)' },
  'standard': { videoBitrate: 5000, audioBitrate: 128, label: 'Standard' },
  'high': { videoBitrate: 10000, audioBitrate: 192, label: 'Haute qualité' },
  'ultra': { videoBitrate: 20000, audioBitrate: 320, label: 'Ultra (lent)' },
};

export const DEFAULT_EXPORT_SETTINGS: ExportSettings = {
  resolution: '1080p',
  format: 'mp4',
  aspectRatio: '16:9',
  quality: 'high',
  fps: 30,
  videoBitrate: 10000,
  audioBitrate: 192,
  includeSubtitles: true,
  burnSubtitles: false,
};

// ============================================================================
// COLLABORATION TYPES
// ============================================================================

export interface TimecodedComment {
  id: string;
  userId: string;
  userName: string;
  userAvatar?: string;
  timestamp: number; // video timecode in seconds
  slideId?: string;
  text: string;
  createdAt: string;
  updatedAt?: string;
  isResolved: boolean;
  replies: CommentReply[];
}

export interface CommentReply {
  id: string;
  userId: string;
  userName: string;
  userAvatar?: string;
  text: string;
  createdAt: string;
}

export interface ProjectVersion {
  id: string;
  versionNumber: number;
  name: string;
  description?: string;
  createdAt: string;
  createdBy: string;
  snapshot: LectureComponents;
  isAutoSave: boolean;
}

export interface ShareLink {
  id: string;
  url: string;
  expiresAt?: string;
  permissions: 'view' | 'comment' | 'edit';
  password?: string;
  isActive: boolean;
  createdAt: string;
  accessCount: number;
}

export interface CollaborationState {
  comments: TimecodedComment[];
  versions: ProjectVersion[];
  shareLinks: ShareLink[];
  activeCollaborators: Collaborator[];
}

export interface Collaborator {
  userId: string;
  userName: string;
  userAvatar?: string;
  role: 'owner' | 'editor' | 'commenter' | 'viewer';
  isOnline: boolean;
  lastSeen?: string;
  currentSlideId?: string;
}

// ============================================================================
// EDITOR PANEL TYPES
// ============================================================================

export type EditorPanelType =
  | 'properties'
  | 'subtitles'
  | 'audio'
  | 'transitions'
  | 'effects'
  | 'overlays'
  | 'export'
  | 'collaboration';

export interface EditorPanelConfig {
  id: EditorPanelType;
  label: string;
  icon: string;
  shortcut?: string;
}

export const EDITOR_PANELS: EditorPanelConfig[] = [
  { id: 'properties', label: 'Propriétés', icon: '⚙️', shortcut: '1' },
  { id: 'subtitles', label: 'Sous-titres', icon: '💬', shortcut: '2' },
  { id: 'audio', label: 'Audio', icon: '🎵', shortcut: '3' },
  { id: 'transitions', label: 'Transitions', icon: '✨', shortcut: '4' },
  { id: 'effects', label: 'Effets', icon: '🎨', shortcut: '5' },
  { id: 'overlays', label: 'Overlays', icon: '📝', shortcut: '6' },
  { id: 'export', label: 'Export', icon: '📤', shortcut: '7' },
  { id: 'collaboration', label: 'Collaboration', icon: '👥', shortcut: '8' },
];

// ============================================================================
// EXTENDED LECTURE COMPONENTS
// ============================================================================

export interface ExtendedLectureComponents extends LectureComponents {
  subtitleTracks: SubtitleTrack[];
  audioTracks: AudioTrack[];
  transitions: SlideTransition[];
  effects: VisualEffect[];
  overlays: Overlay[];
  exportSettings: ExportSettings;
  collaboration: CollaborationState;
}

// Helper to create default extended components
export function createDefaultExtendedComponents(base: LectureComponents): ExtendedLectureComponents {
  return {
    ...base,
    subtitleTracks: [],
    audioTracks: base.voiceover ? [{
      id: 'voiceover-main',
      name: 'Voiceover',
      type: 'voiceover',
      url: base.voiceover.audioUrl || '',
      duration: base.voiceover.durationSeconds,
      startTime: 0,
      volume: 1,
      isMuted: false,
      isSolo: false,
      fadeIn: 0,
      fadeOut: 0,
    }] : [],
    transitions: base.slides.map(slide => ({
      slideId: slide.id,
      inTransition: { id: `trans-${slide.id}-in`, type: 'fade', duration: 0.5, easing: 'ease-in-out' },
      outTransition: undefined,
    })),
    effects: base.slides.map(slide => ({
      id: `effect-${slide.id}`,
      slideId: slide.id,
      filterPreset: 'none',
      speed: 1,
      reverse: false,
    })),
    overlays: [],
    exportSettings: { ...DEFAULT_EXPORT_SETTINGS },
    collaboration: {
      comments: [],
      versions: [],
      shareLinks: [],
      activeCollaborators: [],
    },
  };
}
