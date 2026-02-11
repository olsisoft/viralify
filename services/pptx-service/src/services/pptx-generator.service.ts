/**
 * Viralify PPTX Generator Service
 *
 * Production-ready PPTX generation using PptxGenJS
 * with support for code highlighting, diagrams, and transitions.
 */

import PptxGenJS from 'pptxgenjs';
import { v4 as uuidv4 } from 'uuid';
import * as fs from 'fs';
import * as path from 'path';
import { logger } from '../utils/logger';
import {
  Slide,
  SlideType,
  TransitionType,
  GeneratePptxRequest,
  PresentationTheme,
  THEME_PRESETS,
  ThemeStyle,
  CodeBlock,
  BulletPoint,
  ImageElement,
  TableElement,
  ChartElement,
  TextElement,
} from '../models/slide.model';

// ===========================================
// TYPE UTILITIES
// ===========================================

/**
 * Position/size type for internal use
 * PptxGenJS accepts both numbers (inches) and strings (percentages like '10%')
 * We use 'any' for coordinates to work around strict PptxGenJS typing
 */
interface Position {
  x: string | number;
  y: string | number;
  w: string | number;
  h: string | number;
}

/**
 * Parse percentage string to extract numeric value
 */
function parsePercentage(value: string): number {
  const match = value.match(/^([\d.]+)%?$/);
  return match ? parseFloat(match[1]) : 0;
}

// ===========================================
// CODE SYNTAX HIGHLIGHTING
// ===========================================

interface TokenStyle {
  color: string;
  bold?: boolean;
  italic?: boolean;
}

const CODE_THEME_DARK: Record<string, TokenStyle> = {
  keyword: { color: '#c678dd', bold: true },
  string: { color: '#98c379' },
  number: { color: '#d19a66' },
  comment: { color: '#5c6370', italic: true },
  function: { color: '#61afef' },
  class: { color: '#e5c07b', bold: true },
  operator: { color: '#56b6c2' },
  variable: { color: '#e06c75' },
  type: { color: '#e5c07b' },
  default: { color: '#abb2bf' },
};

const CODE_THEME_LIGHT: Record<string, TokenStyle> = {
  keyword: { color: '#a626a4', bold: true },
  string: { color: '#50a14f' },
  number: { color: '#986801' },
  comment: { color: '#a0a1a7', italic: true },
  function: { color: '#4078f2' },
  class: { color: '#c18401', bold: true },
  operator: { color: '#0184bc' },
  variable: { color: '#e45649' },
  type: { color: '#c18401' },
  default: { color: '#383a42' },
};

// Language keywords for syntax highlighting
const LANGUAGE_KEYWORDS: Record<string, string[]> = {
  python: ['def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with', 'as', 'import', 'from', 'return', 'yield', 'raise', 'pass', 'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False', 'lambda', 'async', 'await', 'global', 'nonlocal'],
  javascript: ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return', 'try', 'catch', 'finally', 'throw', 'new', 'class', 'extends', 'import', 'export', 'default', 'from', 'async', 'await', 'this', 'super', 'null', 'undefined', 'true', 'false', 'typeof', 'instanceof'],
  typescript: ['function', 'const', 'let', 'var', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return', 'try', 'catch', 'finally', 'throw', 'new', 'class', 'extends', 'implements', 'interface', 'type', 'enum', 'import', 'export', 'default', 'from', 'async', 'await', 'this', 'super', 'null', 'undefined', 'true', 'false', 'typeof', 'instanceof', 'as', 'readonly', 'private', 'public', 'protected', 'static', 'abstract'],
  java: ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'static', 'final', 'void', 'int', 'long', 'double', 'float', 'boolean', 'char', 'String', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue', 'return', 'try', 'catch', 'finally', 'throw', 'throws', 'new', 'this', 'super', 'null', 'true', 'false', 'import', 'package'],
  go: ['func', 'package', 'import', 'type', 'struct', 'interface', 'map', 'chan', 'go', 'select', 'case', 'default', 'if', 'else', 'for', 'range', 'switch', 'break', 'continue', 'return', 'defer', 'var', 'const', 'nil', 'true', 'false', 'make', 'new', 'len', 'cap', 'append', 'copy', 'delete'],
  rust: ['fn', 'let', 'mut', 'const', 'static', 'struct', 'enum', 'impl', 'trait', 'type', 'where', 'if', 'else', 'match', 'for', 'while', 'loop', 'break', 'continue', 'return', 'use', 'mod', 'pub', 'crate', 'self', 'super', 'async', 'await', 'move', 'ref', 'true', 'false', 'Some', 'None', 'Ok', 'Err'],
  sql: ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON', 'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC', 'LIMIT', 'OFFSET', 'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'CREATE', 'TABLE', 'ALTER', 'DROP', 'INDEX', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'NULL', 'NOT NULL', 'UNIQUE', 'DEFAULT', 'AS', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN'],
  bash: ['if', 'then', 'else', 'elif', 'fi', 'for', 'while', 'do', 'done', 'case', 'esac', 'function', 'return', 'exit', 'echo', 'read', 'export', 'source', 'cd', 'pwd', 'ls', 'cp', 'mv', 'rm', 'mkdir', 'rmdir', 'cat', 'grep', 'sed', 'awk', 'chmod', 'chown', 'sudo', 'apt', 'yum', 'pip', 'npm', 'docker', 'kubectl', 'git'],
};

// ===========================================
// PPTX GENERATOR SERVICE
// ===========================================

export class PptxGeneratorService {
  private outputDir: string;

  constructor(outputDir: string = '/tmp/viralify/pptx') {
    this.outputDir = outputDir;
    this.ensureOutputDir();
  }

  private ensureOutputDir(): void {
    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
      logger.info(`Created output directory: ${this.outputDir}`);
    }
  }

  /**
   * Generate a PPTX file from the request
   */
  async generate(request: GeneratePptxRequest): Promise<string> {
    const startTime = Date.now();
    const pptx = new PptxGenJS();

    // Apply theme
    const theme = this.resolveTheme(request.theme);
    this.applyTheme(pptx, theme, request);

    // Set metadata
    if (request.metadata) {
      pptx.title = request.metadata.title;
      pptx.author = request.metadata.author || 'Viralify';
      pptx.company = request.metadata.company || '';
      pptx.subject = request.metadata.subject || '';
    }

    // Set slide dimensions (16:9 default)
    pptx.defineLayout({
      name: 'VIRALIFY_16_9',
      width: request.width || 10,
      height: request.height || 5.625,
    });
    pptx.layout = 'VIRALIFY_16_9';

    // Generate each slide
    for (let i = 0; i < request.slides.length; i++) {
      const slideData = request.slides[i];
      const slide = pptx.addSlide();

      // Apply transition
      const transition = slideData.transition || request.defaultTransition;
      if (transition && transition.type !== TransitionType.NONE) {
        this.applyTransition(slide, transition);
      }

      // Apply background
      this.applyBackground(slide, slideData, theme);

      // Generate slide content based on type
      await this.generateSlideContent(slide, slideData, theme, i);

      // Add speaker notes
      if (slideData.speakerNotes) {
        slide.addNotes(slideData.speakerNotes);
      }

      logger.debug(`Generated slide ${i + 1}/${request.slides.length}: ${slideData.type}`);
    }

    // Save to file
    const filename = `${request.job_id}_${Date.now()}.pptx`;
    const outputPath = path.join(this.outputDir, filename);

    await pptx.writeFile({ fileName: outputPath });

    const processingTime = Date.now() - startTime;
    logger.info(`Generated PPTX: ${outputPath} (${request.slides.length} slides, ${processingTime}ms)`);

    return outputPath;
  }

  /**
   * Resolve theme from request or use preset
   */
  private resolveTheme(theme?: PresentationTheme): PresentationTheme {
    if (!theme) {
      return THEME_PRESETS[ThemeStyle.DARK];
    }
    const preset = THEME_PRESETS[theme.style] || THEME_PRESETS[ThemeStyle.DARK];
    return { ...preset, ...theme };
  }

  /**
   * Apply theme to presentation
   */
  private applyTheme(pptx: PptxGenJS, theme: PresentationTheme, request: GeneratePptxRequest): void {
    // Define master slide with theme colors
    pptx.defineSlideMaster({
      title: 'VIRALIFY_MASTER',
      background: { color: theme.backgroundColor?.replace('#', '') || '0f0f1a' },
      objects: [
        // Footer with page number
        {
          placeholder: {
            options: {
              name: 'footer',
              type: 'body',
              x: '90%' as any,
              y: '95%' as any,
              w: '10%' as any,
              h: '5%' as any,
              fontSize: 10,
              color: theme.textColor?.replace('#', '') || 'ffffff',
            },
            text: '',
          },
        },
      ],
    });
  }

  /**
   * Apply transition to slide
   */
  private applyTransition(slide: PptxGenJS.Slide, transition: { type: TransitionType; duration?: number }): void {
    const transitionMap: Record<TransitionType, string> = {
      [TransitionType.NONE]: 'none',
      [TransitionType.FADE]: 'fade',
      [TransitionType.PUSH]: 'push',
      [TransitionType.WIPE]: 'wipe',
      [TransitionType.ZOOM]: 'zoom',
      [TransitionType.SPLIT]: 'split',
      [TransitionType.REVEAL]: 'reveal',
      [TransitionType.COVER]: 'cover',
    };

    const transType = transitionMap[transition.type] || 'fade';
    const duration = (transition.duration || 0.5) * 1000; // Convert to ms

    // PptxGenJS transition options
    (slide as any).transition = {
      type: transType,
      speed: duration <= 500 ? 'fast' : duration <= 1000 ? 'med' : 'slow',
    };
  }

  /**
   * Apply background to slide
   */
  private applyBackground(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    if (slideData.background) {
      if (slideData.background.color) {
        slide.background = { color: slideData.background.color.replace('#', '') };
      } else if (slideData.background.image) {
        slide.background = { path: slideData.background.image };
      } else if (slideData.background.gradient) {
        // PptxGenJS doesn't support gradients directly, use primary color
        slide.background = { color: theme.primaryColor?.replace('#', '') || '1a1a2e' };
      }
    } else {
      slide.background = { color: theme.backgroundColor?.replace('#', '') || '0f0f1a' };
    }
  }

  /**
   * Generate slide content based on type
   */
  private async generateSlideContent(
    slide: PptxGenJS.Slide,
    slideData: Slide,
    theme: PresentationTheme,
    index: number
  ): Promise<void> {
    switch (slideData.type) {
      case SlideType.TITLE:
        this.generateTitleSlide(slide, slideData, theme);
        break;
      case SlideType.SECTION_HEADER:
        this.generateSectionHeaderSlide(slide, slideData, theme);
        break;
      case SlideType.CODE:
      case SlideType.CODE_DEMO:
        this.generateCodeSlide(slide, slideData, theme);
        break;
      case SlideType.DIAGRAM:
        await this.generateDiagramSlide(slide, slideData, theme);
        break;
      case SlideType.COMPARISON:
        this.generateComparisonSlide(slide, slideData, theme);
        break;
      case SlideType.TWO_COLUMN:
        this.generateTwoColumnSlide(slide, slideData, theme);
        break;
      case SlideType.BULLET_POINTS:
        this.generateBulletPointsSlide(slide, slideData, theme);
        break;
      case SlideType.QUOTE:
        this.generateQuoteSlide(slide, slideData, theme);
        break;
      case SlideType.IMAGE:
        await this.generateImageSlide(slide, slideData, theme);
        break;
      case SlideType.VIDEO:
        // VIDEO slides render as image slides with video placeholder
        await this.generateImageSlide(slide, slideData, theme);
        break;
      case SlideType.QUIZ:
        // QUIZ slides render as bullet points with question/answers
        this.generateQuizSlide(slide, slideData, theme);
        break;
      case SlideType.CONCLUSION:
        this.generateConclusionSlide(slide, slideData, theme);
        break;
      default:
        this.generateContentSlide(slide, slideData, theme);
    }

    // Add custom elements if present
    this.addCustomElements(slide, slideData, theme);
  }

  /**
   * Generate title slide
   */
  private generateTitleSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Main title
    slide.addText(slideData.title || 'Untitled', {
      x: '5%' as any,
      y: '35%' as any,
      w: '90%' as any,
      h: '20%' as any,
      fontSize: 44,
      fontFace: theme.headingFontFamily || 'Poppins',
      color: theme.textColor?.replace('#', '') || 'ffffff',
      bold: true,
      align: 'center',
      valign: 'middle',
    });

    // Subtitle
    if (slideData.subtitle) {
      slide.addText(slideData.subtitle, {
        x: '10%' as any,
        y: '55%' as any,
        w: '80%' as any,
        h: '10%' as any,
        fontSize: 24,
        fontFace: theme.fontFamily || 'Inter',
        color: theme.accentColor?.replace('#', '') || 'e94560',
        align: 'center',
        valign: 'middle',
      });
    }

    // Accent line
    slide.addShape('rect', {
      x: '35%' as any,
      y: '52%' as any,
      w: '30%' as any,
      h: '0.5%' as any,
      fill: { color: theme.accentColor?.replace('#', '') || 'e94560' },
    });
  }

  /**
   * Generate section header slide
   */
  private generateSectionHeaderSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Section number/indicator
    slide.addShape('ellipse', {
      x: '45%' as any,
      y: '25%' as any,
      w: '10%' as any,
      h: '17.8%' as any,
      fill: { color: theme.accentColor?.replace('#', '') || 'e94560' },
    });

    // Section title
    slide.addText(slideData.title || 'Section', {
      x: '5%' as any,
      y: '50%' as any,
      w: '90%' as any,
      h: '15%' as any,
      fontSize: 40,
      fontFace: theme.headingFontFamily || 'Poppins',
      color: theme.textColor?.replace('#', '') || 'ffffff',
      bold: true,
      align: 'center',
      valign: 'middle',
    });

    // Subtitle
    if (slideData.subtitle) {
      slide.addText(slideData.subtitle, {
        x: '10%' as any,
        y: '65%' as any,
        w: '80%' as any,
        h: '10%' as any,
        fontSize: 20,
        fontFace: theme.fontFamily || 'Inter',
        color: (theme.textColor?.replace('#', '') || 'ffffff') + '99',
        align: 'center',
      });
    }
  }

  /**
   * Generate code slide with syntax highlighting
   */
  private generateCodeSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Title
    if (slideData.title) {
      slide.addText(slideData.title, {
        x: '3%' as any,
        y: '3%' as any,
        w: '94%' as any,
        h: '10%' as any,
        fontSize: 28,
        fontFace: theme.headingFontFamily || 'Poppins',
        color: theme.textColor?.replace('#', '') || 'ffffff',
        bold: true,
      });
    }

    // Code blocks
    const codeBlocks = slideData.codeBlocks || [];
    const yStart = slideData.title ? 0.15 : 0.05;
    const blockHeight = codeBlocks.length > 1 ? 0.4 : 0.8;

    codeBlocks.forEach((block, i) => {
      this.addCodeBlock(slide, block, {
        x: '3%',
        y: `${(yStart + i * (blockHeight + 0.02)) * 100}%`,
        w: '94%',
        h: `${blockHeight * 100}%`,
      }, theme);
    });
  }

  /**
   * Add a code block with syntax highlighting
   */
  private addCodeBlock(
    slide: PptxGenJS.Slide,
    block: CodeBlock,
    position: Position,
    theme: PresentationTheme
  ): void {
    const codeTheme = block.theme === 'light' ? CODE_THEME_LIGHT : CODE_THEME_DARK;
    const bgColor = block.theme === 'light' ? 'f5f5f5' : '1e1e1e';
    const language = block.language.toLowerCase();
    const keywords = LANGUAGE_KEYWORDS[language] || [];

    // Code background
    slide.addShape('roundRect', {
      x: position.x as any,
      y: position.y as any,
      w: position.w as any,
      h: position.h as any,
      fill: { color: bgColor },
      rectRadius: 0.1,
    });

    // Language badge
    if (block.title || block.language) {
      slide.addText(block.title || block.language.toUpperCase(), {
        x: position.x as any,
        y: position.y as any,
        w: '15%' as any,
        h: '6%' as any,
        fontSize: 10,
        fontFace: theme.codeFontFamily || 'JetBrains Mono',
        color: theme.accentColor?.replace('#', '') || 'e94560',
        bold: true,
        fill: { color: bgColor },
      });
    }

    // Parse and highlight code
    const lines = block.code.split('\n');
    const highlightedLines = this.highlightCode(lines, keywords, codeTheme);

    // Calculate font size based on line count
    const fontSize = Math.max(8, Math.min(14, 180 / lines.length));
    const lineHeight = fontSize * 1.5;

    // Add code text with highlighting
    const textRuns: PptxGenJS.TextProps[] = [];
    highlightedLines.forEach((line, lineIdx) => {
      if (lineIdx > 0) {
        textRuns.push({ text: '\n', options: { breakLine: true } });
      }
      if (block.showLineNumbers) {
        textRuns.push({
          text: `${String(lineIdx + 1).padStart(3, ' ')} `,
          options: {
            fontSize,
            fontFace: theme.codeFontFamily || 'JetBrains Mono',
            color: '666666',
          },
        });
      }
      line.forEach(token => {
        textRuns.push({
          text: token.text,
          options: {
            fontSize,
            fontFace: theme.codeFontFamily || 'JetBrains Mono',
            color: token.style.color.replace('#', ''),
            bold: token.style.bold,
            italic: token.style.italic,
          },
        });
      });
    });

    // Calculate adjusted Y position for code content
    const yValue = typeof position.y === 'string' ? parsePercentage(position.y) + 5 : (position.y as number) + 0.3;
    const hValue = typeof position.h === 'string' ? parsePercentage(position.h) - 8 : (position.h as number) - 0.4;

    slide.addText(textRuns, {
      x: position.x as any,
      y: (typeof position.y === 'string' ? `${yValue}%` : yValue) as any,
      w: position.w as any,
      h: (typeof position.h === 'string' ? `${hValue}%` : hValue) as any,
      valign: 'top',
      margin: [5, 10, 5, 10],
    });
  }

  /**
   * Simple syntax highlighting
   */
  private highlightCode(
    lines: string[],
    keywords: string[],
    codeTheme: Record<string, TokenStyle>
  ): Array<Array<{ text: string; style: TokenStyle }>> {
    return lines.map(line => {
      const tokens: Array<{ text: string; style: TokenStyle }> = [];
      let remaining = line;

      while (remaining.length > 0) {
        let matched = false;

        // Comments
        const commentMatch = remaining.match(/^(\/\/.*|#.*|\/\*[\s\S]*?\*\/)/);
        if (commentMatch) {
          tokens.push({ text: commentMatch[0], style: codeTheme.comment });
          remaining = remaining.slice(commentMatch[0].length);
          matched = true;
          continue;
        }

        // Strings
        const stringMatch = remaining.match(/^(["'`])(?:(?!\1|\\).|\\.)*\1/);
        if (stringMatch) {
          tokens.push({ text: stringMatch[0], style: codeTheme.string });
          remaining = remaining.slice(stringMatch[0].length);
          matched = true;
          continue;
        }

        // Numbers
        const numberMatch = remaining.match(/^-?\d+\.?\d*([eE][+-]?\d+)?/);
        if (numberMatch) {
          tokens.push({ text: numberMatch[0], style: codeTheme.number });
          remaining = remaining.slice(numberMatch[0].length);
          matched = true;
          continue;
        }

        // Keywords
        for (const kw of keywords) {
          const kwRegex = new RegExp(`^\\b${kw}\\b`, 'i');
          const kwMatch = remaining.match(kwRegex);
          if (kwMatch) {
            tokens.push({ text: kwMatch[0], style: codeTheme.keyword });
            remaining = remaining.slice(kwMatch[0].length);
            matched = true;
            break;
          }
        }
        if (matched) continue;

        // Function calls
        const funcMatch = remaining.match(/^(\w+)(?=\s*\()/);
        if (funcMatch) {
          tokens.push({ text: funcMatch[0], style: codeTheme.function });
          remaining = remaining.slice(funcMatch[0].length);
          matched = true;
          continue;
        }

        // Operators
        const opMatch = remaining.match(/^[+\-*/%=<>!&|^~?:]+/);
        if (opMatch) {
          tokens.push({ text: opMatch[0], style: codeTheme.operator });
          remaining = remaining.slice(opMatch[0].length);
          matched = true;
          continue;
        }

        // Default: single character
        if (!matched) {
          tokens.push({ text: remaining[0], style: codeTheme.default });
          remaining = remaining.slice(1);
        }
      }

      return tokens;
    });
  }

  /**
   * Generate diagram slide
   */
  private async generateDiagramSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): Promise<void> {
    // Title
    if (slideData.title) {
      slide.addText(slideData.title, {
        x: '3%' as any,
        y: '3%' as any,
        w: '94%' as any,
        h: '10%' as any,
        fontSize: 28,
        fontFace: theme.headingFontFamily || 'Poppins',
        color: theme.textColor?.replace('#', '') || 'ffffff',
        bold: true,
      });
    }

    // Add images (diagrams)
    if (slideData.images && slideData.images.length > 0) {
      for (const img of slideData.images) {
        await this.addImage(slide, img, theme);
      }
    } else {
      // Placeholder for diagram
      slide.addText('Diagram placeholder', {
        x: '10%' as any,
        y: '20%' as any,
        w: '80%' as any,
        h: '70%' as any,
        fontSize: 20,
        color: '666666',
        align: 'center',
        valign: 'middle',
        fill: { color: theme.secondaryColor?.replace('#', '') || '16213e' },
      });
    }
  }

  /**
   * Add image to slide
   */
  private async addImage(slide: PptxGenJS.Slide, img: ImageElement, theme: PresentationTheme): Promise<void> {
    const imgOptions: any = {
      x: img.x || '10%',
      y: img.y || '15%',
      w: img.w || '80%',
      h: img.h || '75%',
    };

    if (img.sizing) {
      imgOptions.sizing = img.sizing;
    }

    if (img.data) {
      imgOptions.data = img.data;
      slide.addImage(imgOptions);
    } else if (img.path) {
      if (fs.existsSync(img.path)) {
        imgOptions.path = img.path;
        slide.addImage(imgOptions);
      } else {
        logger.warn(`Image not found: ${img.path}`);
      }
    } else if (img.url) {
      imgOptions.path = img.url;
      slide.addImage(imgOptions);
    }
  }

  /**
   * Generate comparison slide (two columns)
   */
  private generateComparisonSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Title
    if (slideData.title) {
      slide.addText(slideData.title, {
        x: '3%' as any,
        y: '3%' as any,
        w: '94%' as any,
        h: '12%' as any,
        fontSize: 28,
        fontFace: theme.headingFontFamily || 'Poppins',
        color: theme.textColor?.replace('#', '') || 'ffffff',
        bold: true,
        align: 'center',
      });
    }

    // VS divider
    slide.addText('VS', {
      x: '47%' as any,
      y: '50%' as any,
      w: '6%' as any,
      h: '10%' as any,
      fontSize: 20,
      fontFace: theme.headingFontFamily || 'Poppins',
      color: theme.accentColor?.replace('#', '') || 'e94560',
      bold: true,
      align: 'center',
      valign: 'middle',
    });

    // Left column
    slide.addShape('roundRect', {
      x: '3%' as any,
      y: '18%' as any,
      w: '43%' as any,
      h: '75%' as any,
      fill: { color: theme.secondaryColor?.replace('#', '') || '16213e' },
      rectRadius: 0.1,
    });

    // Right column
    slide.addShape('roundRect', {
      x: '54%' as any,
      y: '18%' as any,
      w: '43%' as any,
      h: '75%' as any,
      fill: { color: theme.secondaryColor?.replace('#', '') || '16213e' },
      rectRadius: 0.1,
    });

    // Add content if provided in bullet points
    if (slideData.bulletPoints && slideData.bulletPoints.length >= 2) {
      const midPoint = Math.ceil(slideData.bulletPoints.length / 2);
      const leftPoints = slideData.bulletPoints.slice(0, midPoint);
      const rightPoints = slideData.bulletPoints.slice(midPoint);

      this.addBulletList(slide, leftPoints, { x: '5%', y: '22%', w: '39%', h: '68%' }, theme);
      this.addBulletList(slide, rightPoints, { x: '56%', y: '22%', w: '39%', h: '68%' }, theme);
    }
  }

  /**
   * Generate two-column slide
   */
  private generateTwoColumnSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Title
    if (slideData.title) {
      slide.addText(slideData.title, {
        x: '3%' as any,
        y: '3%' as any,
        w: '94%' as any,
        h: '12%' as any,
        fontSize: 28,
        fontFace: theme.headingFontFamily || 'Poppins',
        color: theme.textColor?.replace('#', '') || 'ffffff',
        bold: true,
      });
    }

    // Left column background
    slide.addShape('rect', {
      x: '3%' as any,
      y: '18%' as any,
      w: '45%' as any,
      h: '75%' as any,
      fill: { color: theme.secondaryColor?.replace('#', '') || '16213e', transparency: 50 },
    });

    // Right column background
    slide.addShape('rect', {
      x: '52%' as any,
      y: '18%' as any,
      w: '45%' as any,
      h: '75%' as any,
      fill: { color: theme.secondaryColor?.replace('#', '') || '16213e', transparency: 50 },
    });
  }

  /**
   * Generate bullet points slide
   */
  private generateBulletPointsSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Title
    if (slideData.title) {
      slide.addText(slideData.title, {
        x: '3%' as any,
        y: '5%' as any,
        w: '94%' as any,
        h: '12%' as any,
        fontSize: 28,
        fontFace: theme.headingFontFamily || 'Poppins',
        color: theme.textColor?.replace('#', '') || 'ffffff',
        bold: true,
      });
    }

    // Bullet points
    if (slideData.bulletPoints) {
      this.addBulletList(slide, slideData.bulletPoints, { x: '5%', y: '20%', w: '90%', h: '75%' }, theme);
    }
  }

  /**
   * Add bullet list to slide
   */
  private addBulletList(
    slide: PptxGenJS.Slide,
    points: BulletPoint[],
    position: Position,
    theme: PresentationTheme
  ): void {
    const textRuns: PptxGenJS.TextProps[] = [];

    points.forEach((point, i) => {
      if (i > 0) textRuns.push({ text: '\n' });

      const indent = (point.level || 0) * 20;
      const bullet = point.bullet !== false ? '• ' : '';
      const fontSize = 18 - (point.level || 0) * 2;

      textRuns.push({
        text: `${' '.repeat((point.level || 0) * 4)}${bullet}${point.text}`,
        options: {
          fontSize: point.fontSize || fontSize,
          fontFace: theme.fontFamily || 'Inter',
          color: point.color?.replace('#', '') || theme.textColor?.replace('#', '') || 'ffffff',
          bullet: false, // We're handling bullets manually
        },
      });
    });

    slide.addText(textRuns, {
      x: position.x as any,
      y: position.y as any,
      w: position.w as any,
      h: position.h as any,
      valign: 'top',
      lineSpacing: 32,
    });
  }

  /**
   * Generate quote slide
   */
  private generateQuoteSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Quote marks
    slide.addText('"', {
      x: '5%' as any,
      y: '15%' as any,
      w: '10%' as any,
      h: '20%' as any,
      fontSize: 120,
      fontFace: 'Georgia',
      color: theme.accentColor?.replace('#', '') || 'e94560',
    });

    // Quote text
    slide.addText(slideData.content || slideData.title || '', {
      x: '10%' as any,
      y: '30%' as any,
      w: '80%' as any,
      h: '40%' as any,
      fontSize: 28,
      fontFace: theme.fontFamily || 'Inter',
      color: theme.textColor?.replace('#', '') || 'ffffff',
      italic: true,
      align: 'center',
      valign: 'middle',
    });

    // Attribution
    if (slideData.subtitle) {
      slide.addText(`— ${slideData.subtitle}`, {
        x: '50%' as any,
        y: '75%' as any,
        w: '45%' as any,
        h: '10%' as any,
        fontSize: 18,
        fontFace: theme.fontFamily || 'Inter',
        color: theme.accentColor?.replace('#', '') || 'e94560',
        align: 'right',
      });
    }
  }

  /**
   * Generate image slide
   */
  private async generateImageSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): Promise<void> {
    // Title
    if (slideData.title) {
      slide.addText(slideData.title, {
        x: '3%' as any,
        y: '3%' as any,
        w: '94%' as any,
        h: '10%' as any,
        fontSize: 24,
        fontFace: theme.headingFontFamily || 'Poppins',
        color: theme.textColor?.replace('#', '') || 'ffffff',
        bold: true,
      });
    }

    // Images
    if (slideData.images) {
      for (const img of slideData.images) {
        await this.addImage(slide, img, theme);
      }
    }
  }

  /**
   * Generate conclusion slide
   */
  private generateConclusionSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Title
    slide.addText(slideData.title || 'Conclusion', {
      x: '5%' as any,
      y: '10%' as any,
      w: '90%' as any,
      h: '15%' as any,
      fontSize: 36,
      fontFace: theme.headingFontFamily || 'Poppins',
      color: theme.textColor?.replace('#', '') || 'ffffff',
      bold: true,
      align: 'center',
    });

    // Key takeaways
    if (slideData.bulletPoints) {
      this.addBulletList(slide, slideData.bulletPoints, { x: '10%', y: '30%', w: '80%', h: '55%' }, theme);
    }

    // Thank you message
    slide.addText('Thank you!', {
      x: '0%' as any,
      y: '88%' as any,
      w: '100%' as any,
      h: '10%' as any,
      fontSize: 24,
      fontFace: theme.headingFontFamily || 'Poppins',
      color: theme.accentColor?.replace('#', '') || 'e94560',
      align: 'center',
    });
  }

  /**
   * Generate quiz slide with question and answer options
   */
  private generateQuizSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Question title
    slide.addText(slideData.title || 'Quiz', {
      x: '5%' as any,
      y: '8%' as any,
      w: '90%' as any,
      h: '12%' as any,
      fontSize: 32,
      fontFace: theme.headingFontFamily || 'Poppins',
      color: theme.accentColor?.replace('#', '') || 'e94560',
      bold: true,
      align: 'center',
    });

    // Question content
    if (slideData.content) {
      slide.addText(slideData.content, {
        x: '5%' as any,
        y: '22%' as any,
        w: '90%' as any,
        h: '15%' as any,
        fontSize: 24,
        fontFace: theme.bodyFontFamily || 'Open Sans',
        color: theme.textColor?.replace('#', '') || 'ffffff',
        align: 'center',
      });
    }

    // Answer options as bullet points
    if (slideData.bulletPoints) {
      this.addBulletList(slide, slideData.bulletPoints, { x: '10%', y: '40%', w: '80%', h: '50%' }, theme);
    }
  }

  /**
   * Generate generic content slide
   */
  private generateContentSlide(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Title
    if (slideData.title) {
      slide.addText(slideData.title, {
        x: '3%' as any,
        y: '5%' as any,
        w: '94%' as any,
        h: '12%' as any,
        fontSize: 28,
        fontFace: theme.headingFontFamily || 'Poppins',
        color: theme.textColor?.replace('#', '') || 'ffffff',
        bold: true,
      });
    }

    // Content text
    if (slideData.content) {
      slide.addText(slideData.content, {
        x: '5%' as any,
        y: '20%' as any,
        w: '90%' as any,
        h: '70%' as any,
        fontSize: 18,
        fontFace: theme.fontFamily || 'Inter',
        color: theme.textColor?.replace('#', '') || 'ffffff',
        valign: 'top',
        lineSpacing: 28,
      });
    }

    // Bullet points if no content
    if (!slideData.content && slideData.bulletPoints) {
      this.addBulletList(slide, slideData.bulletPoints, { x: '5%', y: '20%', w: '90%', h: '75%' }, theme);
    }
  }

  /**
   * Add custom elements to slide
   */
  private addCustomElements(slide: PptxGenJS.Slide, slideData: Slide, theme: PresentationTheme): void {
    // Custom text elements
    if (slideData.textElements) {
      for (const text of slideData.textElements) {
        slide.addText(text.text, {
          x: (text.x ?? 0) as any,
          y: (text.y ?? 0) as any,
          w: (text.w ?? '100%') as any,
          h: (text.h ?? '10%') as any,
          fontSize: text.fontSize || 18,
          fontFace: text.fontFace || theme.fontFamily || 'Inter',
          color: text.color?.replace('#', '') || theme.textColor?.replace('#', '') || 'ffffff',
          bold: text.bold,
          italic: text.italic,
          underline: text.underline ? { style: 'sng' } : undefined,
          align: text.align,
          valign: text.valign,
        });
      }
    }

    // Custom shapes
    if (slideData.shapes) {
      for (const shape of slideData.shapes) {
        const shapeOptions: any = {
          x: shape.x,
          y: shape.y,
          w: shape.w,
          h: shape.h,
          fill: shape.fill,
        };

        // Add line properties if present
        if (shape.line) {
          shapeOptions.line = {
            color: shape.line.color,
            width: shape.line.width,
            dashType: shape.line.dashType || 'solid',
          };
        }

        // Add shadow properties if present
        if (shape.shadow) {
          shapeOptions.shadow = {
            type: shape.shadow.type || 'outer',
            blur: shape.shadow.blur,
            offset: shape.shadow.offset,
            angle: shape.shadow.angle,
            color: shape.shadow.color,
          };
        }

        slide.addShape(shape.type as any, shapeOptions);
      }
    }

    // Tables
    if (slideData.tables) {
      for (const table of slideData.tables) {
        this.addTable(slide, table, theme);
      }
    }

    // Charts
    if (slideData.charts) {
      for (const chart of slideData.charts) {
        this.addChart(slide, chart, theme);
      }
    }
  }

  /**
   * Add table to slide
   */
  private addTable(slide: PptxGenJS.Slide, table: TableElement, theme: PresentationTheme): void {
    const rows = table.rows.map((row, rowIdx) =>
      row.map(cell => ({
        text: cell.text,
        options: {
          fill: cell.options?.fill || (rowIdx === 0 ? { color: theme.accentColor?.replace('#', '') || 'e94560' } : undefined),
          color: cell.options?.color?.replace('#', '') || theme.textColor?.replace('#', '') || 'ffffff',
          bold: cell.options?.bold ?? rowIdx === 0,
          align: cell.options?.align || 'center',
          valign: cell.options?.valign || 'middle',
          colspan: cell.options?.colspan,
          rowspan: cell.options?.rowspan,
        },
      }))
    );

    slide.addTable(rows, {
      x: (table.x || '5%') as any,
      y: (table.y || '20%') as any,
      w: (table.w || '90%') as any,
      colW: table.colW,
      rowH: table.rowH || 0.5,
      fontFace: table.fontFace || theme.fontFamily || 'Inter',
      fontSize: table.fontSize || 14,
      border: table.border || { pt: 1, color: theme.secondaryColor?.replace('#', '') || '16213e' },
      autoPage: table.autoPage,
    });
  }

  /**
   * Add chart to slide
   */
  private addChart(slide: PptxGenJS.Slide, chart: ChartElement, theme: PresentationTheme): void {
    const chartTypeMap: Record<string, PptxGenJS.CHART_NAME> = {
      bar: 'bar',
      line: 'line',
      pie: 'pie',
      doughnut: 'doughnut',
      area: 'area',
      scatter: 'scatter',
    };

    const chartType = chartTypeMap[chart.type] || 'bar';

    slide.addChart(chartType as any, chart.data, {
      x: (chart.x || '10%') as any,
      y: (chart.y || '20%') as any,
      w: (chart.w || '80%') as any,
      h: (chart.h || '65%') as any,
      showLegend: chart.showLegend ?? true,
      showTitle: chart.showTitle ?? !!chart.title,
      title: chart.title,
      showValue: chart.showValue ?? false,
      catAxisTitle: chart.catAxisTitle,
      valAxisTitle: chart.valAxisTitle,
      chartColors: [
        theme.accentColor?.replace('#', '') || 'e94560',
        theme.primaryColor?.replace('#', '') || '1a1a2e',
        theme.secondaryColor?.replace('#', '') || '16213e',
      ],
    });
  }

  /**
   * Clean up old files
   */
  async cleanup(maxAgeMs: number = 3600000): Promise<number> {
    let cleaned = 0;
    const now = Date.now();

    const files = fs.readdirSync(this.outputDir);
    for (const file of files) {
      const filePath = path.join(this.outputDir, file);
      const stats = fs.statSync(filePath);
      if (now - stats.mtimeMs > maxAgeMs) {
        fs.unlinkSync(filePath);
        cleaned++;
      }
    }

    if (cleaned > 0) {
      logger.info(`Cleaned up ${cleaned} old PPTX files`);
    }

    return cleaned;
  }
}

// Singleton instance
let instance: PptxGeneratorService | null = null;

export function getPptxGenerator(): PptxGeneratorService {
  if (!instance) {
    instance = new PptxGeneratorService(process.env.PPTX_OUTPUT_DIR || '/tmp/viralify/pptx');
  }
  return instance;
}
