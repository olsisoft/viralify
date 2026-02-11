/**
 * PPTX API Routes
 *
 * Production-ready API for PPTX generation with support for:
 * - Slide generation from JSON
 * - PNG export via LibreOffice
 * - File download
 * - Job management
 */

import { Router, Request, Response, NextFunction } from 'express';
import multer from 'multer';
import * as fs from 'fs';
import * as path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { getPptxGenerator } from '../services/pptx-generator.service';
import { getLibreOfficeConverter } from '../services/libreoffice-converter.service';
import { logger } from '../utils/logger';
import {
  GeneratePptxRequest,
  GeneratePptxResponse,
  Slide,
  SlideType,
  ThemeStyle,
  THEME_PRESETS,
} from '../models/slide.model';

const router = Router();

// ===========================================
// JOB STORAGE (In-memory for now, Redis in production)
// ===========================================

interface Job {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  createdAt: Date;
  completedAt?: Date;
  pptxPath?: string;
  pngPaths?: string[];
  error?: string;
  slideCount?: number;
  processingTimeMs?: number;
}

const jobs = new Map<string, Job>();

// ===========================================
// MULTER CONFIGURATION
// ===========================================

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB max
  },
});

// ===========================================
// ROUTES
// ===========================================

/**
 * Health check
 */
router.get('/health', async (req: Request, res: Response) => {
  const converter = getLibreOfficeConverter();
  const libreOfficeAvailable = await converter.checkAvailability();

  res.json({
    status: 'healthy',
    service: 'pptx-service',
    version: '1.0.0',
    libreoffice_available: libreOfficeAvailable,
    jobs_in_memory: jobs.size,
    timestamp: new Date().toISOString(),
  });
});

/**
 * Get available themes
 */
router.get('/themes', (req: Request, res: Response) => {
  const themes = Object.entries(THEME_PRESETS).map(([key, value]) => ({
    id: key,
    name: key.charAt(0).toUpperCase() + key.slice(1),
    ...value,
  }));

  res.json({
    themes,
    default: ThemeStyle.DARK,
  });
});

/**
 * Get available slide types
 */
router.get('/slide-types', (req: Request, res: Response) => {
  res.json({
    types: Object.values(SlideType),
    descriptions: {
      [SlideType.TITLE]: 'Title slide with main heading and subtitle',
      [SlideType.CONTENT]: 'Generic content slide',
      [SlideType.CODE]: 'Code slide with syntax highlighting',
      [SlideType.CODE_DEMO]: 'Code demonstration slide',
      [SlideType.DIAGRAM]: 'Diagram or architecture slide',
      [SlideType.COMPARISON]: 'Side-by-side comparison',
      [SlideType.QUOTE]: 'Quote or testimonial slide',
      [SlideType.IMAGE]: 'Full image slide',
      [SlideType.VIDEO]: 'Video embed slide',
      [SlideType.QUIZ]: 'Quiz or assessment slide',
      [SlideType.CONCLUSION]: 'Conclusion with key takeaways',
      [SlideType.SECTION_HEADER]: 'Section divider slide',
      [SlideType.TWO_COLUMN]: 'Two-column layout',
      [SlideType.BULLET_POINTS]: 'Bullet points slide',
    },
  });
});

/**
 * Generate PPTX (synchronous)
 *
 * POST /api/v1/pptx/generate
 */
router.post('/generate', async (req: Request, res: Response, next: NextFunction) => {
  const startTime = Date.now();

  try {
    const request: GeneratePptxRequest = req.body;

    // Validate request
    if (!request.slides || !Array.isArray(request.slides) || request.slides.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'slides array is required and must not be empty',
      });
    }

    // Validate each slide has a type
    const validSlideTypes = Object.values(SlideType);
    for (let i = 0; i < request.slides.length; i++) {
      const slide = request.slides[i];
      if (!slide.type) {
        return res.status(400).json({
          success: false,
          error: `Slide ${i + 1} is missing required 'type' field`,
        });
      }
      // Log warning for unknown types (they'll fall back to content)
      if (!validSlideTypes.includes(slide.type as SlideType)) {
        logger.warn(`Slide ${i + 1} has unknown type '${slide.type}', will use default content rendering`);
      }
    }

    if (!request.job_id) {
      request.job_id = uuidv4();
    }

    logger.info(`Generating PPTX for job ${request.job_id} with ${request.slides.length} slides`);

    // Generate PPTX
    const generator = getPptxGenerator();
    const pptxPath = await generator.generate(request);

    // Convert to PNG if requested
    let pngUrls: string[] | undefined;
    if (request.outputFormat === 'png' || request.outputFormat === 'both') {
      const converter = getLibreOfficeConverter();
      const result = await converter.convertToPng(pptxPath, {
        width: request.pngWidth || 1920,
        height: request.pngHeight || 1080,
      });

      if (result.success) {
        pngUrls = result.outputFiles.map(f =>
          `/api/v1/pptx/files/${path.basename(path.dirname(f))}/${path.basename(f)}`
        );
      } else {
        logger.warn(`PNG conversion failed: ${result.error}`);
      }
    }

    const response: GeneratePptxResponse = {
      success: true,
      job_id: request.job_id,
      pptx_url: `/api/v1/pptx/files/${path.basename(pptxPath)}`,
      png_urls: pngUrls,
      processing_time_ms: Date.now() - startTime,
    };

    // Store job
    jobs.set(request.job_id, {
      id: request.job_id,
      status: 'completed',
      createdAt: new Date(),
      completedAt: new Date(),
      pptxPath,
      pngPaths: pngUrls ? pngUrls.map(u => u.replace('/api/v1/pptx/files/', '')) : undefined,
      slideCount: request.slides.length,
      processingTimeMs: Date.now() - startTime,
    });

    logger.info(`PPTX generated successfully: ${pptxPath}`);
    res.json(response);

  } catch (error: any) {
    logger.error('PPTX generation failed:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'PPTX generation failed',
      processing_time_ms: Date.now() - startTime,
    });
  }
});

/**
 * Generate PPTX (asynchronous - returns job ID immediately)
 *
 * POST /api/v1/pptx/generate-async
 */
router.post('/generate-async', async (req: Request, res: Response) => {
  const request: GeneratePptxRequest = req.body;

  // Validate request
  if (!request.slides || !Array.isArray(request.slides) || request.slides.length === 0) {
    return res.status(400).json({
      success: false,
      error: 'slides array is required and must not be empty',
    });
  }

  if (!request.job_id) {
    request.job_id = uuidv4();
  }

  // Create pending job
  const job: Job = {
    id: request.job_id,
    status: 'pending',
    createdAt: new Date(),
    slideCount: request.slides.length,
  };
  jobs.set(request.job_id, job);

  // Return immediately
  res.status(202).json({
    success: true,
    job_id: request.job_id,
    status: 'pending',
    status_url: `/api/v1/pptx/jobs/${request.job_id}`,
  });

  // Process in background
  setImmediate(async () => {
    const startTime = Date.now();
    job.status = 'processing';

    try {
      const generator = getPptxGenerator();
      const pptxPath = await generator.generate(request);

      let pngPaths: string[] | undefined;
      if (request.outputFormat === 'png' || request.outputFormat === 'both') {
        const converter = getLibreOfficeConverter();
        const result = await converter.convertToPng(pptxPath, {
          width: request.pngWidth || 1920,
          height: request.pngHeight || 1080,
        });

        if (result.success) {
          pngPaths = result.outputFiles;
        }
      }

      job.status = 'completed';
      job.completedAt = new Date();
      job.pptxPath = pptxPath;
      job.pngPaths = pngPaths;
      job.processingTimeMs = Date.now() - startTime;

      logger.info(`Async job ${job.id} completed in ${job.processingTimeMs}ms`);

    } catch (error: any) {
      job.status = 'failed';
      job.error = error.message || 'PPTX generation failed';
      job.processingTimeMs = Date.now() - startTime;

      logger.error(`Async job ${job.id} failed:`, error);
    }
  });
});

/**
 * Get job status
 *
 * GET /api/v1/pptx/jobs/:jobId
 */
router.get('/jobs/:jobId', (req: Request, res: Response) => {
  const { jobId } = req.params;
  const job = jobs.get(jobId);

  if (!job) {
    return res.status(404).json({
      success: false,
      error: 'Job not found',
    });
  }

  const response: any = {
    success: true,
    job_id: job.id,
    status: job.status,
    created_at: job.createdAt.toISOString(),
    slide_count: job.slideCount,
  };

  if (job.status === 'completed') {
    response.completed_at = job.completedAt?.toISOString();
    response.processing_time_ms = job.processingTimeMs;
    response.pptx_url = `/api/v1/pptx/files/${path.basename(job.pptxPath || '')}`;

    if (job.pngPaths) {
      response.png_urls = job.pngPaths.map(p =>
        `/api/v1/pptx/files/${path.basename(path.dirname(p))}/${path.basename(p)}`
      );
    }
  } else if (job.status === 'failed') {
    response.error = job.error;
    response.processing_time_ms = job.processingTimeMs;
  }

  res.json(response);
});

/**
 * Download generated file
 *
 * GET /api/v1/pptx/files/:filename
 * GET /api/v1/pptx/files/:jobId/:filename
 */
router.get('/files/:param1/:param2?', (req: Request, res: Response) => {
  const { param1, param2 } = req.params;

  let filePath: string;

  if (param2) {
    // /files/:jobId/:filename
    filePath = path.join(
      process.env.PPTX_IMAGES_DIR || '/tmp/viralify/pptx-images',
      param1,
      param2
    );
  } else {
    // /files/:filename
    filePath = path.join(
      process.env.PPTX_OUTPUT_DIR || '/tmp/viralify/pptx',
      param1
    );
  }

  if (!fs.existsSync(filePath)) {
    return res.status(404).json({
      success: false,
      error: 'File not found',
    });
  }

  // Determine content type
  const ext = path.extname(filePath).toLowerCase();
  const contentTypes: Record<string, string> = {
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.pdf': 'application/pdf',
  };

  const contentType = contentTypes[ext] || 'application/octet-stream';
  res.setHeader('Content-Type', contentType);
  res.setHeader('Content-Disposition', `attachment; filename="${path.basename(filePath)}"`);

  const stream = fs.createReadStream(filePath);
  stream.pipe(res);
});

/**
 * Delete job and its files
 *
 * DELETE /api/v1/pptx/jobs/:jobId
 */
router.delete('/jobs/:jobId', async (req: Request, res: Response) => {
  const { jobId } = req.params;
  const job = jobs.get(jobId);

  if (!job) {
    return res.status(404).json({
      success: false,
      error: 'Job not found',
    });
  }

  // Delete files
  try {
    if (job.pptxPath && fs.existsSync(job.pptxPath)) {
      fs.unlinkSync(job.pptxPath);
    }

    if (job.pngPaths) {
      const jobDir = path.dirname(job.pngPaths[0]);
      if (fs.existsSync(jobDir)) {
        fs.rmSync(jobDir, { recursive: true, force: true });
      }
    }
  } catch (error) {
    logger.warn(`Failed to delete files for job ${jobId}:`, error);
  }

  // Remove from memory
  jobs.delete(jobId);

  res.json({
    success: true,
    message: `Job ${jobId} deleted`,
  });
});

/**
 * Cleanup old jobs and files
 *
 * POST /api/v1/pptx/cleanup
 */
router.post('/cleanup', async (req: Request, res: Response) => {
  const maxAgeMs = parseInt(req.query.max_age_ms as string) || 3600000; // 1 hour default

  try {
    const generator = getPptxGenerator();
    const converter = getLibreOfficeConverter();

    const pptxCleaned = await generator.cleanup(maxAgeMs);
    const pngCleaned = await converter.cleanup(undefined, maxAgeMs);

    // Clean old jobs from memory
    const now = Date.now();
    let jobsCleaned = 0;
    for (const [id, job] of jobs) {
      if (now - job.createdAt.getTime() > maxAgeMs) {
        jobs.delete(id);
        jobsCleaned++;
      }
    }

    res.json({
      success: true,
      cleaned: {
        pptx_files: pptxCleaned,
        png_jobs: pngCleaned,
        memory_jobs: jobsCleaned,
      },
    });

  } catch (error: any) {
    logger.error('Cleanup failed:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Cleanup failed',
    });
  }
});

/**
 * Generate preview (single slide as PNG)
 *
 * POST /api/v1/pptx/preview
 */
router.post('/preview', async (req: Request, res: Response) => {
  const startTime = Date.now();

  try {
    const slide: Slide = req.body.slide;
    const theme = req.body.theme;

    if (!slide) {
      return res.status(400).json({
        success: false,
        error: 'slide is required',
      });
    }

    const request: GeneratePptxRequest = {
      job_id: `preview_${uuidv4()}`,
      slides: [slide],
      theme,
      outputFormat: 'png',
    };

    // Generate PPTX
    const generator = getPptxGenerator();
    const pptxPath = await generator.generate(request);

    // Convert to PNG
    const converter = getLibreOfficeConverter();
    const result = await converter.convertToPng(pptxPath, {
      width: req.body.width || 1920,
      height: req.body.height || 1080,
    });

    // Clean up PPTX
    fs.unlinkSync(pptxPath);

    if (!result.success || result.outputFiles.length === 0) {
      return res.status(500).json({
        success: false,
        error: result.error || 'PNG conversion failed',
      });
    }

    // Return PNG directly
    const pngPath = result.outputFiles[0];
    const pngData = fs.readFileSync(pngPath);

    // Clean up PNG
    fs.rmSync(path.dirname(pngPath), { recursive: true, force: true });

    res.setHeader('Content-Type', 'image/png');
    res.setHeader('X-Processing-Time-Ms', String(Date.now() - startTime));
    res.send(pngData);

  } catch (error: any) {
    logger.error('Preview generation failed:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Preview generation failed',
    });
  }
});

export default router;
