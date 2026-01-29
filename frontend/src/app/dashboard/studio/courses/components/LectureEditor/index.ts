/**
 * LectureEditor Components
 *
 * A complete professional video lecture editor with:
 * - Multi-level undo/redo with history (Ctrl+Z, Ctrl+Y)
 * - Drag & drop timeline with quick actions
 * - Inline editing for slides and voiceover
 * - Full code block editing with syntax highlighting
 * - Editable bullet points with reordering
 * - Image replacement on existing slides
 * - Media upload (image/video) support
 * - Video preview player
 * - Professional toolbar with history counter
 * - Keyboard shortcuts (Ctrl+S, Escape, Arrow keys, Delete)
 * - Subtitles editor with SRT import/export
 * - Multi-track audio timeline with waveform
 * - Audio mixer with volume controls
 * - Transition effects between slides
 * - Visual effects (filters, Ken Burns, speed)
 * - Text overlays and lower thirds
 * - Export settings (resolution, format, quality)
 * - Collaboration with comments and version history
 */

// Core components
export { LectureEditor } from './LectureEditor';
export { SlideTimeline } from './SlideTimeline';
export { SlidePreview } from './SlidePreview';
export { SlideProperties } from './SlideProperties';
export { EditorToolbar } from './EditorToolbar';
export { CodeBlockEditor } from './CodeBlockEditor';

// Professional editor panels
export { SubtitlesEditor } from './SubtitlesEditor';
export { AudioTimeline } from './AudioTimeline';
export { AudioMixer } from './AudioMixer';
export { TransitionsPanel } from './TransitionsPanel';
export { VisualEffectsPanel } from './VisualEffectsPanel';
export { OverlaysEditor } from './OverlaysEditor';
export { ExportPanel } from './ExportPanel';
export { CollaborationPanel } from './CollaborationPanel';
