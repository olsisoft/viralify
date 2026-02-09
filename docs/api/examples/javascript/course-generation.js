/**
 * Viralify API - Course Generation Example (JavaScript/Node.js)
 *
 * This example demonstrates how to generate a complete video course
 * using the Viralify API.
 *
 * Requirements:
 *   npm install node-fetch dotenv
 *
 * Usage:
 *   export VIRALIFY_API_KEY="your_api_key"
 *   node course-generation.js
 */

import fetch from 'node-fetch';
import 'dotenv/config';

const API_KEY = process.env.VIRALIFY_API_KEY;
const BASE_URL = process.env.VIRALIFY_BASE_URL || 'https://api.viralify.io';

/**
 * Viralify API Client
 */
class ViralifyClient {
  constructor(apiKey, baseUrl = 'https://api.viralify.io') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  /**
   * Make an API request
   */
  async request(method, endpoint, body = null) {
    const options = {
      method,
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
    };

    if (body) {
      options.body = JSON.stringify(body);
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, options);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`API Error: ${error.message || response.statusText}`);
    }

    return response.json();
  }

  /**
   * Preview course outline
   */
  async previewOutline(config) {
    return this.request('POST', '/api/v1/courses/preview-outline', {
      topic: config.topic,
      difficulty_start: config.difficultyStart || 'beginner',
      difficulty_end: config.difficultyEnd || 'intermediate',
      structure: {
        number_of_sections: config.numSections || 4,
        lectures_per_section: config.lecturesPerSection || 3,
      },
      context: {
        category: config.category || 'tech',
      },
    });
  }

  /**
   * Start course generation
   */
  async generateCourse(config) {
    const result = await this.request('POST', '/api/v1/courses/generate', {
      topic: config.topic,
      difficulty_start: config.difficultyStart || 'beginner',
      difficulty_end: config.difficultyEnd || 'intermediate',
      structure: {
        number_of_sections: config.numSections || 4,
        lectures_per_section: config.lecturesPerSection || 3,
      },
      context: {
        category: config.category || 'tech',
      },
      language: config.language || 'en',
      quiz_config: {
        enabled: config.quizEnabled !== false,
        frequency: config.quizFrequency || 'per_section',
        questions_per_quiz: 5,
        passing_score: 70,
      },
      title_style: config.titleStyle || 'engaging',
      document_ids: config.documentIds || [],
    });

    return result.job_id;
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId) {
    return this.request('GET', `/api/v1/courses/jobs/${jobId}`);
  }

  /**
   * Wait for job completion
   */
  async waitForCompletion(jobId, options = {}) {
    const pollInterval = options.pollInterval || 15000;
    const timeout = options.timeout || 3600000;
    const onProgress = options.onProgress || (() => {});

    const startTime = Date.now();

    while (true) {
      if (Date.now() - startTime > timeout) {
        throw new Error(`Job ${jobId} timed out`);
      }

      const status = await this.getJobStatus(jobId);
      onProgress(status);

      if (status.status === 'completed') {
        return status;
      }

      if (status.status === 'failed') {
        throw new Error(`Job failed: ${status.error || 'Unknown error'}`);
      }

      await this.sleep(pollInterval);
    }
  }

  /**
   * Upload document
   */
  async uploadDocument(filePath, userId, role = 'auto') {
    const fs = await import('fs');
    const path = await import('path');
    const FormData = (await import('form-data')).default;

    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    form.append('user_id', userId);
    form.append('pedagogical_role', role);

    const response = await fetch(`${this.baseUrl}/api/v1/documents/upload`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: form,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Upload failed: ${error.message}`);
    }

    return response.json();
  }

  /**
   * Sleep helper
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Progress callback
 */
function onProgress(status) {
  const progress = status.progress || 0;
  const stage = status.status || 'unknown';
  console.log(`[${stage.toUpperCase()}] Progress: ${progress.toFixed(1)}%`);
}

/**
 * Main function
 */
async function main() {
  if (!API_KEY) {
    console.error('Error: VIRALIFY_API_KEY environment variable not set');
    process.exit(1);
  }

  const client = new ViralifyClient(API_KEY, BASE_URL);

  // Course configuration
  const config = {
    topic: 'Introduction to React Hooks',
    difficultyStart: 'beginner',
    difficultyEnd: 'intermediate',
    numSections: 4,
    lecturesPerSection: 3,
    category: 'tech',
    language: 'en',
    quizEnabled: true,
    quizFrequency: 'per_section',
    titleStyle: 'engaging',
  };

  console.log(`\n=== Generating Course: ${config.topic} ===\n`);

  try {
    // Step 1: Preview outline
    console.log('Step 1: Previewing outline...');
    const outline = await client.previewOutline(config);
    console.log(`  Title: ${outline.title}`);
    console.log(`  Sections: ${outline.section_count}`);
    console.log(`  Total Lectures: ${outline.total_lectures}`);
    console.log();

    // Step 2: Start generation
    console.log('Step 2: Starting course generation...');
    const jobId = await client.generateCourse(config);
    console.log(`  Job ID: ${jobId}`);
    console.log();

    // Step 3: Wait for completion
    console.log('Step 3: Waiting for completion...');
    const result = await client.waitForCompletion(jobId, {
      pollInterval: 15000,
      timeout: 3600000,
      onProgress,
    });

    console.log('\n=== Course Generation Complete! ===');
    console.log(`Videos: ${result.output_urls.videos.length} files`);
    console.log(`ZIP: ${result.output_urls.zip}`);

    console.log('\nGenerated Videos:');
    result.output_urls.videos.forEach((url, i) => {
      console.log(`  ${i + 1}. ${url}`);
    });

  } catch (error) {
    console.error(`\nError: ${error.message}`);
    process.exit(1);
  }
}

main();
