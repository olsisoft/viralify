/**
 * LibreOffice Converter Service
 *
 * Converts PPTX files to PNG images using LibreOffice headless mode.
 * This enables high-quality slide rendering for video composition.
 */

import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../utils/logger';

const execAsync = promisify(exec);

// ===========================================
// CONFIGURATION
// ===========================================

interface ConversionOptions {
  width?: number;
  height?: number;
  format?: 'png' | 'jpg' | 'pdf';
  quality?: number; // 1-100 for JPEG
  dpi?: number;
}

interface ConversionResult {
  success: boolean;
  outputFiles: string[];
  error?: string;
  processingTimeMs: number;
}

// ===========================================
// LIBREOFFICE CONVERTER SERVICE
// ===========================================

export class LibreOfficeConverterService {
  private outputDir: string;
  private libreOfficePath: string;
  private isAvailable: boolean = false;
  private checkPromise: Promise<boolean> | null = null;

  constructor(outputDir: string = '/tmp/viralify/pptx-images') {
    this.outputDir = outputDir;
    this.libreOfficePath = process.env.LIBREOFFICE_PATH || this.findLibreOffice();
    this.ensureOutputDir();
  }

  /**
   * Find LibreOffice installation
   */
  private findLibreOffice(): string {
    const paths = [
      '/usr/bin/libreoffice',
      '/usr/bin/soffice',
      '/usr/local/bin/libreoffice',
      '/usr/local/bin/soffice',
      '/opt/libreoffice/program/soffice',
      '/Applications/LibreOffice.app/Contents/MacOS/soffice',
      'C:\\Program Files\\LibreOffice\\program\\soffice.exe',
      'C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe',
    ];

    for (const p of paths) {
      if (fs.existsSync(p)) {
        return p;
      }
    }

    return 'soffice'; // Fallback to PATH
  }

  /**
   * Ensure output directory exists
   */
  private ensureOutputDir(): void {
    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
      logger.info(`Created output directory: ${this.outputDir}`);
    }
  }

  /**
   * Check if LibreOffice is available
   */
  async checkAvailability(): Promise<boolean> {
    if (this.checkPromise) {
      return this.checkPromise;
    }

    this.checkPromise = new Promise(async (resolve) => {
      try {
        const { stdout } = await execAsync(`"${this.libreOfficePath}" --version`, {
          timeout: 10000,
        });
        this.isAvailable = stdout.includes('LibreOffice');
        logger.info(`LibreOffice available: ${this.isAvailable} (${stdout.trim()})`);
        resolve(this.isAvailable);
      } catch (error) {
        logger.warn('LibreOffice not available:', error);
        this.isAvailable = false;
        resolve(false);
      }
    });

    return this.checkPromise;
  }

  /**
   * Convert PPTX to PNG images
   */
  async convertToPng(
    pptxPath: string,
    options: ConversionOptions = {}
  ): Promise<ConversionResult> {
    const startTime = Date.now();
    const jobId = uuidv4();
    const jobDir = path.join(this.outputDir, jobId);

    // Check availability
    if (!this.isAvailable && !(await this.checkAvailability())) {
      return {
        success: false,
        outputFiles: [],
        error: 'LibreOffice is not available',
        processingTimeMs: Date.now() - startTime,
      };
    }

    // Verify input file
    if (!fs.existsSync(pptxPath)) {
      return {
        success: false,
        outputFiles: [],
        error: `Input file not found: ${pptxPath}`,
        processingTimeMs: Date.now() - startTime,
      };
    }

    try {
      // Create job directory
      fs.mkdirSync(jobDir, { recursive: true });

      // Step 1: Convert PPTX to PDF (better quality than direct PNG)
      const pdfPath = path.join(jobDir, 'presentation.pdf');
      await this.convertToPdf(pptxPath, jobDir);

      // Step 2: Convert PDF to PNG images using pdftoppm or ImageMagick
      const pngFiles = await this.pdfToPng(pdfPath, jobDir, options);

      const result: ConversionResult = {
        success: true,
        outputFiles: pngFiles,
        processingTimeMs: Date.now() - startTime,
      };

      logger.info(`Converted PPTX to ${pngFiles.length} PNG files in ${result.processingTimeMs}ms`);
      return result;

    } catch (error: any) {
      logger.error('PPTX conversion failed:', error);
      return {
        success: false,
        outputFiles: [],
        error: error.message || 'Conversion failed',
        processingTimeMs: Date.now() - startTime,
      };
    }
  }

  /**
   * Convert PPTX to PDF using LibreOffice
   */
  private async convertToPdf(pptxPath: string, outputDir: string): Promise<string> {
    const args = [
      '--headless',
      '--invisible',
      '--nologo',
      '--nofirststartwizard',
      '--convert-to', 'pdf',
      '--outdir', outputDir,
      pptxPath,
    ];

    return new Promise((resolve, reject) => {
      const process = spawn(this.libreOfficePath, args, {
        timeout: 120000, // 2 minutes timeout
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stderr = '';
      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) {
          const pdfPath = path.join(outputDir, 'presentation.pdf');
          if (fs.existsSync(pdfPath)) {
            resolve(pdfPath);
          } else {
            // Try to find the output file with original name
            const baseName = path.basename(pptxPath, path.extname(pptxPath));
            const altPdfPath = path.join(outputDir, `${baseName}.pdf`);
            if (fs.existsSync(altPdfPath)) {
              fs.renameSync(altPdfPath, pdfPath);
              resolve(pdfPath);
            } else {
              reject(new Error('PDF output file not found'));
            }
          }
        } else {
          reject(new Error(`LibreOffice exited with code ${code}: ${stderr}`));
        }
      });

      process.on('error', (err) => {
        reject(err);
      });
    });
  }

  /**
   * Convert PDF to PNG images
   * Uses pdftoppm (poppler-utils) or ImageMagick as fallback
   */
  private async pdfToPng(
    pdfPath: string,
    outputDir: string,
    options: ConversionOptions
  ): Promise<string[]> {
    const width = options.width || 1920;
    const height = options.height || 1080;
    const dpi = options.dpi || 150;

    // Try pdftoppm first (faster, better quality)
    try {
      return await this.pdfToPngWithPdftoppm(pdfPath, outputDir, dpi);
    } catch (error) {
      logger.warn('pdftoppm not available, trying ImageMagick...');
    }

    // Fallback to ImageMagick
    try {
      return await this.pdfToPngWithImageMagick(pdfPath, outputDir, width, height);
    } catch (error) {
      logger.warn('ImageMagick not available, trying GraphicsMagick...');
    }

    // Fallback to GraphicsMagick
    return await this.pdfToPngWithGm(pdfPath, outputDir, width, height);
  }

  /**
   * Convert PDF to PNG using pdftoppm (poppler-utils)
   */
  private async pdfToPngWithPdftoppm(
    pdfPath: string,
    outputDir: string,
    dpi: number
  ): Promise<string[]> {
    const outputPrefix = path.join(outputDir, 'slide');

    await execAsync(
      `pdftoppm -png -r ${dpi} "${pdfPath}" "${outputPrefix}"`,
      { timeout: 120000 }
    );

    return this.collectOutputFiles(outputDir, 'slide-', '.png');
  }

  /**
   * Convert PDF to PNG using ImageMagick
   */
  private async pdfToPngWithImageMagick(
    pdfPath: string,
    outputDir: string,
    width: number,
    height: number
  ): Promise<string[]> {
    const outputPattern = path.join(outputDir, 'slide-%03d.png');

    await execAsync(
      `convert -density 150 -resize ${width}x${height} "${pdfPath}" "${outputPattern}"`,
      { timeout: 120000 }
    );

    return this.collectOutputFiles(outputDir, 'slide-', '.png');
  }

  /**
   * Convert PDF to PNG using GraphicsMagick
   */
  private async pdfToPngWithGm(
    pdfPath: string,
    outputDir: string,
    width: number,
    height: number
  ): Promise<string[]> {
    const outputPattern = path.join(outputDir, 'slide-%03d.png');

    await execAsync(
      `gm convert -density 150 -resize ${width}x${height} "${pdfPath}" "${outputPattern}"`,
      { timeout: 120000 }
    );

    return this.collectOutputFiles(outputDir, 'slide-', '.png');
  }

  /**
   * Collect output files matching pattern
   */
  private collectOutputFiles(dir: string, prefix: string, suffix: string): string[] {
    const files = fs.readdirSync(dir)
      .filter(f => f.startsWith(prefix) && f.endsWith(suffix))
      .sort((a, b) => {
        const numA = parseInt(a.replace(prefix, '').replace(suffix, '')) || 0;
        const numB = parseInt(b.replace(prefix, '').replace(suffix, '')) || 0;
        return numA - numB;
      })
      .map(f => path.join(dir, f));

    return files;
  }

  /**
   * Get slide count from PPTX without converting
   */
  async getSlideCount(pptxPath: string): Promise<number> {
    // This would require parsing the PPTX file
    // For now, return -1 to indicate unknown
    return -1;
  }

  /**
   * Clean up temporary files
   */
  async cleanup(jobId?: string, maxAgeMs: number = 3600000): Promise<number> {
    let cleaned = 0;

    if (jobId) {
      // Clean specific job
      const jobDir = path.join(this.outputDir, jobId);
      if (fs.existsSync(jobDir)) {
        fs.rmSync(jobDir, { recursive: true, force: true });
        cleaned = 1;
      }
    } else {
      // Clean old jobs
      const now = Date.now();
      const dirs = fs.readdirSync(this.outputDir);

      for (const dir of dirs) {
        const dirPath = path.join(this.outputDir, dir);
        try {
          const stats = fs.statSync(dirPath);
          if (stats.isDirectory() && now - stats.mtimeMs > maxAgeMs) {
            fs.rmSync(dirPath, { recursive: true, force: true });
            cleaned++;
          }
        } catch (error) {
          // Ignore errors for individual directories
        }
      }
    }

    if (cleaned > 0) {
      logger.info(`Cleaned up ${cleaned} conversion job(s)`);
    }

    return cleaned;
  }
}

// Singleton instance
let instance: LibreOfficeConverterService | null = null;

export function getLibreOfficeConverter(): LibreOfficeConverterService {
  if (!instance) {
    instance = new LibreOfficeConverterService(
      process.env.PPTX_IMAGES_DIR || '/tmp/viralify/pptx-images'
    );
  }
  return instance;
}
