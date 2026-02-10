/**
 * Viralify PPTX Service
 *
 * Production-ready microservice for PowerPoint generation using PptxGenJS.
 * Supports PPTX generation, PNG export via LibreOffice, and theme customization.
 *
 * Port: 8013 (configurable via PORT env)
 *
 * @author Viralify Team
 * @version 1.0.0
 */

import express, { Express, Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { logger, requestLogger } from './utils/logger';
import pptxRoutes from './routes/pptx.routes';
import { getLibreOfficeConverter } from './services/libreoffice-converter.service';

// ===========================================
// CONFIGURATION
// ===========================================

const PORT = parseInt(process.env.PORT || '8013', 10);
const HOST = process.env.HOST || '0.0.0.0';
const NODE_ENV = process.env.NODE_ENV || 'development';

// ===========================================
// EXPRESS APP
// ===========================================

const app: Express = express();

// ===========================================
// MIDDLEWARE
// ===========================================

// Security headers
app.use(helmet({
  contentSecurityPolicy: false, // Allow inline styles for slide previews
}));

// CORS
app.use(cors({
  origin: process.env.CORS_ORIGIN || '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Job-ID'],
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100, // 100 requests per minute
  message: { error: 'Too many requests, please try again later.' },
  standardHeaders: true,
  legacyHeaders: false,
});
app.use('/api/', limiter);

// Body parsing
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Request logging
app.use(requestLogger);

// ===========================================
// ROUTES
// ===========================================

// Health check (outside of /api for k8s probes)
app.get('/health', async (req: Request, res: Response) => {
  const converter = getLibreOfficeConverter();
  const libreOfficeAvailable = await converter.checkAvailability();

  res.json({
    status: 'healthy',
    service: 'pptx-service',
    version: '1.0.0',
    environment: NODE_ENV,
    libreoffice: libreOfficeAvailable ? 'available' : 'unavailable',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    timestamp: new Date().toISOString(),
  });
});

// Readiness probe
app.get('/ready', async (req: Request, res: Response) => {
  // Check if LibreOffice is available (optional for PPTX-only mode)
  res.json({ status: 'ready' });
});

// Liveness probe
app.get('/live', (req: Request, res: Response) => {
  res.json({ status: 'alive' });
});

// API routes
app.use('/api/v1/pptx', pptxRoutes);

// Root redirect
app.get('/', (req: Request, res: Response) => {
  res.redirect('/health');
});

// ===========================================
// ERROR HANDLING
// ===========================================

// 404 handler
app.use((req: Request, res: Response) => {
  res.status(404).json({
    error: 'Not Found',
    path: req.originalUrl,
    method: req.method,
  });
});

// Global error handler
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  logger.error('Unhandled error:', err);

  res.status(500).json({
    error: 'Internal Server Error',
    message: NODE_ENV === 'development' ? err.message : 'Something went wrong',
  });
});

// ===========================================
// STARTUP
// ===========================================

async function startServer(): Promise<void> {
  try {
    // Check LibreOffice availability
    const converter = getLibreOfficeConverter();
    const libreOfficeAvailable = await converter.checkAvailability();

    if (!libreOfficeAvailable) {
      logger.warn('LibreOffice is not available. PNG export will be disabled.');
      logger.warn('Install LibreOffice for PNG export: apt-get install libreoffice');
    }

    // Start server
    app.listen(PORT, HOST, () => {
      logger.info(`
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🎨 Viralify PPTX Service v1.0.0                            ║
║                                                               ║
║   Port:        ${PORT}                                          ║
║   Host:        ${HOST}                                       ║
║   Environment: ${NODE_ENV.padEnd(12)}                           ║
║   LibreOffice: ${(libreOfficeAvailable ? 'Available' : 'Not Available').padEnd(12)}                         ║
║                                                               ║
║   Endpoints:                                                  ║
║   - POST /api/v1/pptx/generate       Generate PPTX           ║
║   - POST /api/v1/pptx/generate-async Async generation        ║
║   - POST /api/v1/pptx/preview        Single slide preview    ║
║   - GET  /api/v1/pptx/jobs/:id       Job status              ║
║   - GET  /api/v1/pptx/themes         Available themes        ║
║   - GET  /api/v1/pptx/slide-types    Available slide types   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
      `);
    });

    // Graceful shutdown
    const shutdown = async (signal: string) => {
      logger.info(`Received ${signal}, shutting down gracefully...`);

      // Cleanup old files
      try {
        const { getPptxGenerator } = await import('./services/pptx-generator.service');
        await getPptxGenerator().cleanup(0); // Clean all
        await converter.cleanup(undefined, 0);
      } catch (error) {
        logger.error('Cleanup failed during shutdown:', error);
      }

      process.exit(0);
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));

    // Scheduled cleanup (every hour)
    setInterval(async () => {
      try {
        const { getPptxGenerator } = await import('./services/pptx-generator.service');
        await getPptxGenerator().cleanup();
        await converter.cleanup();
      } catch (error) {
        logger.error('Scheduled cleanup failed:', error);
      }
    }, 3600000); // 1 hour

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server
startServer();

export default app;
