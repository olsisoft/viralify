/**
 * Viralify PPTX Service - Slide Data Models
 *
 * These models define the structure for slides received from
 * presentation-generator and used to create PPTX files.
 */

// ===========================================
// ENUMS
// ===========================================

export enum SlideType {
  TITLE = 'title',
  CONTENT = 'content',
  CODE = 'code',
  CODE_DEMO = 'code_demo',
  DIAGRAM = 'diagram',
  COMPARISON = 'comparison',
  QUOTE = 'quote',
  IMAGE = 'image',
  VIDEO = 'video',
  QUIZ = 'quiz',
  CONCLUSION = 'conclusion',
  SECTION_HEADER = 'section_header',
  TWO_COLUMN = 'two_column',
  BULLET_POINTS = 'bullet_points',
}

export enum TransitionType {
  NONE = 'none',
  FADE = 'fade',
  PUSH = 'push',
  WIPE = 'wipe',
  ZOOM = 'zoom',
  SPLIT = 'split',
  REVEAL = 'reveal',
  COVER = 'cover',
}

export enum ThemeStyle {
  DARK = 'dark',
  LIGHT = 'light',
  CORPORATE = 'corporate',
  GRADIENT = 'gradient',
  OCEAN = 'ocean',
  NEON = 'neon',
  MINIMAL = 'minimal',
}

export enum CodeLanguage {
  PYTHON = 'python',
  JAVASCRIPT = 'javascript',
  TYPESCRIPT = 'typescript',
  JAVA = 'java',
  GO = 'go',
  RUST = 'rust',
  CPP = 'cpp',
  CSHARP = 'csharp',
  SQL = 'sql',
  BASH = 'bash',
  YAML = 'yaml',
  JSON = 'json',
  HTML = 'html',
  CSS = 'css',
}

// ===========================================
// CONTENT ELEMENTS
// ===========================================

export interface TextElement {
  text: string;
  fontSize?: number;
  fontFace?: string;
  bold?: boolean;
  italic?: boolean;
  underline?: boolean;
  color?: string;
  align?: 'left' | 'center' | 'right' | 'justify';
  valign?: 'top' | 'middle' | 'bottom';
  x?: number | string;
  y?: number | string;
  w?: number | string;
  h?: number | string;
}

export interface BulletPoint {
  text: string;
  level?: number;
  bullet?: boolean | { type?: string; code?: string };
  color?: string;
  fontSize?: number;
}

export interface CodeBlock {
  code: string;
  language: CodeLanguage | string;
  title?: string;
  highlightLines?: number[];
  showLineNumbers?: boolean;
  fontSize?: number;
  theme?: 'dark' | 'light';
}

export interface ImageElement {
  path?: string;
  data?: string; // Base64
  url?: string;
  x?: number | string;
  y?: number | string;
  w?: number | string;
  h?: number | string;
  sizing?: {
    type: 'contain' | 'cover' | 'crop';
    w?: number | string;
    h?: number | string;
  };
  hyperlink?: { url: string };
  altText?: string;
}

export interface ShapeElement {
  type: 'rect' | 'roundRect' | 'ellipse' | 'triangle' | 'line' | 'arrow';
  x: number | string;
  y: number | string;
  w: number | string;
  h: number | string;
  fill?: { color: string; transparency?: number };
  line?: { color: string; width?: number; dashType?: string };
  shadow?: { type: string; blur: number; offset: number; angle: number; color: string };
}

export interface TableCell {
  text: string;
  options?: {
    fill?: { color: string };
    color?: string;
    bold?: boolean;
    align?: 'left' | 'center' | 'right';
    valign?: 'top' | 'middle' | 'bottom';
    colspan?: number;
    rowspan?: number;
  };
}

export interface TableElement {
  rows: TableCell[][];
  x?: number | string;
  y?: number | string;
  w?: number | string;
  colW?: number[];
  rowH?: number | number[];
  border?: { pt: number; color: string };
  fontFace?: string;
  fontSize?: number;
  autoPage?: boolean;
}

export interface ChartData {
  name: string;
  labels: string[];
  values: number[];
}

export interface ChartElement {
  type: 'bar' | 'line' | 'pie' | 'doughnut' | 'area' | 'scatter';
  data: ChartData[];
  title?: string;
  x?: number | string;
  y?: number | string;
  w?: number | string;
  h?: number | string;
  showLegend?: boolean;
  showTitle?: boolean;
  showValue?: boolean;
  catAxisTitle?: string;
  valAxisTitle?: string;
}

// ===========================================
// SLIDE DEFINITION
// ===========================================

export interface SlideBackground {
  color?: string;
  image?: string; // URL or base64
  gradient?: {
    colors: string[];
    direction?: 'horizontal' | 'vertical' | 'diagonal';
  };
}

export interface SlideTransition {
  type: TransitionType;
  duration?: number; // seconds
}

export interface Slide {
  id?: string;
  type: SlideType;
  title?: string;
  subtitle?: string;
  content?: string;

  // Content elements
  textElements?: TextElement[];
  bulletPoints?: BulletPoint[];
  codeBlocks?: CodeBlock[];
  images?: ImageElement[];
  shapes?: ShapeElement[];
  tables?: TableElement[];
  charts?: ChartElement[];

  // Styling
  background?: SlideBackground;
  transition?: SlideTransition;

  // Layout
  layout?: 'default' | 'title' | 'section' | 'two_column' | 'blank';

  // Notes for speaker
  speakerNotes?: string;

  // Voiceover text (for video sync)
  voiceover?: string;

  // Duration hint (seconds)
  duration?: number;
}

// ===========================================
// PRESENTATION REQUEST
// ===========================================

export interface PresentationTheme {
  style: ThemeStyle;
  primaryColor?: string;
  secondaryColor?: string;
  accentColor?: string;
  backgroundColor?: string;
  textColor?: string;
  fontFamily?: string;
  headingFontFamily?: string;
  codeFontFamily?: string;
}

export interface PresentationMetadata {
  title: string;
  author?: string;
  company?: string;
  subject?: string;
  category?: string;
  keywords?: string[];
  revision?: number;
}

export interface GeneratePptxRequest {
  job_id: string;
  slides: Slide[];
  theme?: PresentationTheme;
  metadata?: PresentationMetadata;

  // Output options
  outputFormat?: 'pptx' | 'png' | 'both';
  pngWidth?: number;
  pngHeight?: number;

  // Default transition for all slides
  defaultTransition?: SlideTransition;

  // Slide dimensions (16:9 by default)
  width?: number; // inches
  height?: number; // inches
}

export interface GeneratePptxResponse {
  success: boolean;
  job_id: string;
  pptx_url?: string;
  png_urls?: string[];
  error?: string;
  processing_time_ms?: number;
}

// ===========================================
// THEME PRESETS
// ===========================================

export const THEME_PRESETS: Record<ThemeStyle, PresentationTheme> = {
  [ThemeStyle.DARK]: {
    style: ThemeStyle.DARK,
    primaryColor: '#1a1a2e',
    secondaryColor: '#16213e',
    accentColor: '#e94560',
    backgroundColor: '#0f0f1a',
    textColor: '#ffffff',
    fontFamily: 'Inter',
    headingFontFamily: 'Poppins',
    codeFontFamily: 'JetBrains Mono',
  },
  [ThemeStyle.LIGHT]: {
    style: ThemeStyle.LIGHT,
    primaryColor: '#ffffff',
    secondaryColor: '#f5f5f5',
    accentColor: '#2563eb',
    backgroundColor: '#ffffff',
    textColor: '#1f2937',
    fontFamily: 'Inter',
    headingFontFamily: 'Poppins',
    codeFontFamily: 'JetBrains Mono',
  },
  [ThemeStyle.CORPORATE]: {
    style: ThemeStyle.CORPORATE,
    primaryColor: '#1e3a5f',
    secondaryColor: '#2c5282',
    accentColor: '#ed8936',
    backgroundColor: '#f7fafc',
    textColor: '#2d3748',
    fontFamily: 'Roboto',
    headingFontFamily: 'Roboto Slab',
    codeFontFamily: 'Fira Code',
  },
  [ThemeStyle.GRADIENT]: {
    style: ThemeStyle.GRADIENT,
    primaryColor: '#667eea',
    secondaryColor: '#764ba2',
    accentColor: '#f093fb',
    backgroundColor: '#1a1a2e',
    textColor: '#ffffff',
    fontFamily: 'Inter',
    headingFontFamily: 'Montserrat',
    codeFontFamily: 'JetBrains Mono',
  },
  [ThemeStyle.OCEAN]: {
    style: ThemeStyle.OCEAN,
    primaryColor: '#0077b6',
    secondaryColor: '#00b4d8',
    accentColor: '#90e0ef',
    backgroundColor: '#03045e',
    textColor: '#caf0f8',
    fontFamily: 'Open Sans',
    headingFontFamily: 'Raleway',
    codeFontFamily: 'Source Code Pro',
  },
  [ThemeStyle.NEON]: {
    style: ThemeStyle.NEON,
    primaryColor: '#0a0a0a',
    secondaryColor: '#1a1a1a',
    accentColor: '#00ff88',
    backgroundColor: '#000000',
    textColor: '#00ff88',
    fontFamily: 'Space Grotesk',
    headingFontFamily: 'Orbitron',
    codeFontFamily: 'Fira Code',
  },
  [ThemeStyle.MINIMAL]: {
    style: ThemeStyle.MINIMAL,
    primaryColor: '#ffffff',
    secondaryColor: '#fafafa',
    accentColor: '#000000',
    backgroundColor: '#ffffff',
    textColor: '#333333',
    fontFamily: 'Helvetica Neue',
    headingFontFamily: 'Helvetica Neue',
    codeFontFamily: 'Monaco',
  },
};
