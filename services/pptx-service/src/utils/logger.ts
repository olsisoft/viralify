/**
 * Logger utility for PPTX Service
 *
 * Uses Winston for structured logging with support for
 * multiple transports (console, file).
 */

import winston from 'winston';

const LOG_LEVEL = process.env.LOG_LEVEL || 'info';
const NODE_ENV = process.env.NODE_ENV || 'development';

// Custom format for console output
const consoleFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.colorize(),
  winston.format.printf(({ level, message, timestamp, ...meta }) => {
    const metaStr = Object.keys(meta).length ? ` ${JSON.stringify(meta)}` : '';
    return `[${timestamp}] [PPTX-SERVICE] ${level}: ${message}${metaStr}`;
  })
);

// JSON format for production/file output
const jsonFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.json()
);

// Create transports
const transports: winston.transport[] = [
  new winston.transports.Console({
    format: NODE_ENV === 'production' ? jsonFormat : consoleFormat,
  }),
];

// Add file transport in production
if (NODE_ENV === 'production') {
  transports.push(
    new winston.transports.File({
      filename: '/var/log/viralify/pptx-service-error.log',
      level: 'error',
      format: jsonFormat,
      maxsize: 10 * 1024 * 1024, // 10MB
      maxFiles: 5,
    }),
    new winston.transports.File({
      filename: '/var/log/viralify/pptx-service.log',
      format: jsonFormat,
      maxsize: 10 * 1024 * 1024, // 10MB
      maxFiles: 10,
    })
  );
}

// Create logger instance
export const logger = winston.createLogger({
  level: LOG_LEVEL,
  defaultMeta: { service: 'pptx-service' },
  transports,
});

// Export convenience methods
export const logInfo = (message: string, meta?: Record<string, any>) => {
  logger.info(message, meta);
};

export const logError = (message: string, error?: Error | any, meta?: Record<string, any>) => {
  logger.error(message, {
    ...meta,
    error: error instanceof Error ? {
      message: error.message,
      stack: error.stack,
      name: error.name,
    } : error,
  });
};

export const logWarn = (message: string, meta?: Record<string, any>) => {
  logger.warn(message, meta);
};

export const logDebug = (message: string, meta?: Record<string, any>) => {
  logger.debug(message, meta);
};

// Request logging middleware
export const requestLogger = (req: any, res: any, next: any) => {
  const startTime = Date.now();

  res.on('finish', () => {
    const duration = Date.now() - startTime;
    const logLevel = res.statusCode >= 500 ? 'error' : res.statusCode >= 400 ? 'warn' : 'info';

    logger.log(logLevel, `${req.method} ${req.originalUrl}`, {
      method: req.method,
      url: req.originalUrl,
      status: res.statusCode,
      duration: `${duration}ms`,
      ip: req.ip,
      userAgent: req.get('user-agent'),
    });
  });

  next();
};
