// API wrapper that integrates with demo mode
// This provides a unified api object for the dashboard

import { DEMO_MODE, demoApi, DEMO_STATS, DEMO_SCHEDULED_POSTS, DEMO_PLATFORM_ACCOUNTS } from './demo-mode';

// Type definitions for API responses
export interface DashboardStats {
  total_views: number;
  total_likes: number;
  total_comments: number;
  total_shares: number;
  avg_engagement_rate: number;
  growth_metrics?: {
    views_growth: number;
    likes_growth: number;
    followers_growth: number;
  };
  posting_frequency?: {
    daily_avg: number;
    weekly_total: number;
  };
}

export interface GeneratedScript {
  script_id?: string;
  hook: string;
  main_content: string;
  cta: string;
  full_script: string;
  duration_estimate_seconds: number;
  suggested_visuals: string[];
  trending_elements?: string[];
  engagement_score: number;
}

export interface GeneratedCaption {
  caption: string;
  hashtags: string[];
  character_count?: number;
  estimated_reach_multiplier?: number;
}

export interface Hashtag {
  id: string;
  hashtag: string;
  view_count: number;
  trend_score: number;
  growth_rate: number;
}

export interface ScheduledPost {
  id: string;
  title: string;
  caption?: string;
  hashtags: string[];
  videoUrl: string;
  thumbnailUrl?: string;
  scheduledAt: string;
  status: 'pending' | 'published' | 'failed' | 'cancelled' | 'processing';
  targetPlatforms: string[];
  platforms?: string[];
  platformStatuses?: {
    platform: string;
    status: string;
    platformPostId?: string;
    platformShareUrl?: string;
  }[];
}

// Demo data for trending hashtags
const DEMO_HASHTAGS: Hashtag[] = [
  { id: '1', hashtag: '#productivity', view_count: 1200000, trend_score: 95, growth_rate: 23 },
  { id: '2', hashtag: '#morningroutine', view_count: 890000, trend_score: 88, growth_rate: 15 },
  { id: '3', hashtag: '#lifehacks', view_count: 2100000, trend_score: 92, growth_rate: 18 },
  { id: '4', hashtag: '#motivation', view_count: 3500000, trend_score: 85, growth_rate: 8 },
  { id: '5', hashtag: '#cooking', view_count: 1800000, trend_score: 90, growth_rate: 12 },
  { id: '6', hashtag: '#fitness', view_count: 4200000, trend_score: 87, growth_rate: 5 },
  { id: '7', hashtag: '#tutorial', view_count: 980000, trend_score: 78, growth_rate: 20 },
  { id: '8', hashtag: '#viral', view_count: 8900000, trend_score: 99, growth_rate: 35 },
  { id: '9', hashtag: '#trending', view_count: 5600000, trend_score: 96, growth_rate: 28 },
  { id: '10', hashtag: '#fyp', view_count: 15000000, trend_score: 100, growth_rate: 2 },
];

// Demo dashboard stats
const DEMO_DASHBOARD_STATS = {
  total_views: DEMO_STATS.totalViews,
  total_likes: DEMO_STATS.totalLikes,
  total_comments: DEMO_STATS.totalComments,
  total_shares: DEMO_STATS.totalShares,
  avg_engagement_rate: DEMO_STATS.engagementRate,
  growth_metrics: {
    views_growth: 15.2,
    likes_growth: 12.8,
    followers_growth: DEMO_STATS.followerGrowth,
  },
  posting_frequency: {
    daily_avg: 2.3,
    weekly_total: DEMO_STATS.postsThisWeek,
  },
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
// All services should go through API Gateway in production
const MEDIA_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
const COACHING_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
const PRESENTATION_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
const COURSE_API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

// Helper function for API calls
async function apiCall<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const token = typeof window !== 'undefined' ? localStorage.getItem('accessToken') : null;

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }

  return response.json();
}

// Helper function for Media API calls (direct to media-generator service)
async function mediaApiCall<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${MEDIA_API_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Media API Error: ${response.status} - ${error}`);
  }

  return response.json();
}

// Helper function for Coaching API calls
async function coachingApiCall<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${COACHING_API_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Coaching API Error: ${response.status} - ${error}`);
  }

  return response.json();
}

// Helper function for Presentation API calls
async function presentationApiCall<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${PRESENTATION_API_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Presentation API Error: ${response.status} - ${error}`);
  }

  return response.json();
}

// Helper function for Course API calls
async function courseApiCall<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${COURSE_API_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Course API Error: ${response.status} - ${error}`);
  }

  return response.json();
}

export const api = {
  analytics: {
    getDashboard: async (): Promise<DashboardStats> => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 300));
        return DEMO_DASHBOARD_STATS;
      }
      return apiCall<DashboardStats>('/api/v1/analytics/dashboard');
    },
    getSummary: async (period: string = '7d') => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 300));
        return {
          period,
          total_views: DEMO_STATS.totalViews,
          total_likes: DEMO_STATS.totalLikes,
          total_comments: DEMO_STATS.totalComments,
          total_shares: DEMO_STATS.totalShares,
          avg_engagement_rate: DEMO_STATS.engagementRate,
        };
      }
      return apiCall(`/api/v1/analytics/summary?period=${period}`);
    },
    getTimeSeries: async (metric: string, period: string = '7d') => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 300));
        // Generate demo time series data
        const labels = [];
        const data = [];
        const days = period === '7d' ? 7 : period === '30d' ? 30 : 90;

        for (let i = days - 1; i >= 0; i--) {
          const date = new Date();
          date.setDate(date.getDate() - i);
          labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
          data.push(Math.floor(Math.random() * 50000) + 10000);
        }

        return {
          labels,
          datasets: [{
            label: metric,
            data,
            borderColor: '#fe2c55',
            backgroundColor: 'rgba(254, 44, 85, 0.1)',
          }],
        };
      }
      return apiCall(`/api/v1/analytics/time-series?metric=${metric}&period=${period}`);
    },
  },

  trends: {
    getHashtags: async (options?: { limit?: number; region?: string }): Promise<Hashtag[]> => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 200));
        return DEMO_HASHTAGS.slice(0, options?.limit || 10);
      }
      const params = new URLSearchParams();
      if (options?.limit) params.append('limit', String(options.limit));
      if (options?.region) params.append('region', options.region);
      return apiCall<Hashtag[]>(`/api/v1/trends/hashtags?${params}`);
    },
    getSounds: async (options?: { limit?: number; region?: string }) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 200));
        return [
          { id: '1', sound_title: 'Original Sound', sound_author: 'viral_creator', usage_count: 1500000, trend_score: 95 },
          { id: '2', sound_title: 'Trending Beat', sound_author: 'music_producer', usage_count: 890000, trend_score: 88 },
          { id: '3', sound_title: 'Funny Audio', sound_author: 'comedy_king', usage_count: 2100000, trend_score: 92 },
        ];
      }
      const params = new URLSearchParams();
      if (options?.limit) params.append('limit', String(options.limit));
      if (options?.region) params.append('region', options.region);
      return apiCall(`/api/v1/trends/sounds?${params}`);
    },
    getViralPatterns: async (niche?: string) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 300));
        return {
          patterns: [
            { name: 'Hook in first 3 seconds', description: 'Capture attention immediately', effectiveness: 95 },
            { name: 'Trending sound usage', description: 'Use popular audio clips', effectiveness: 88 },
            { name: 'Text overlays', description: 'Add engaging captions', effectiveness: 82 },
          ],
          common_elements: ['Quick cuts', 'Bright lighting', 'Clear audio'],
          best_posting_times: [
            { time: '7:00 PM', day: 'Tuesday', engagement_boost: 23 },
            { time: '12:00 PM', day: 'Saturday', engagement_boost: 18 },
          ],
        };
      }
      return apiCall(`/api/v1/trends/viral-patterns${niche ? `?niche=${niche}` : ''}`);
    },
  },

  scheduler: {
    getPendingPosts: async (): Promise<ScheduledPost[]> => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 200));
        return DEMO_SCHEDULED_POSTS.filter(p => p.status === 'pending') as ScheduledPost[];
      }
      return apiCall<ScheduledPost[]>('/api/v1/scheduler/posts/pending');
    },
    getPosts: async (): Promise<ScheduledPost[]> => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 200));
        return DEMO_SCHEDULED_POSTS as ScheduledPost[];
      }
      return apiCall<ScheduledPost[]>('/api/v1/scheduler/posts');
    },
    createPost: async (data: any) => {
      if (DEMO_MODE) {
        const result = await demoApi.createScheduledPost(data);
        return result.data;
      }
      return apiCall('/api/v1/scheduler/posts', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    cancelPost: async (postId: string) => {
      if (DEMO_MODE) {
        await demoApi.cancelScheduledPost(postId);
        return { success: true };
      }
      return apiCall(`/api/v1/scheduler/posts/${postId}`, { method: 'DELETE' });
    },
    getStats: async () => {
      if (DEMO_MODE) {
        return demoApi.getStats();
      }
      return apiCall('/api/v1/scheduler/stats');
    },
  },

  platforms: {
    getConnectedAccounts: async () => {
      if (DEMO_MODE) {
        return demoApi.getConnectedAccounts();
      }
      return apiCall('/api/v1/platforms/accounts');
    },
    connectPlatform: async (platform: string) => {
      if (DEMO_MODE) {
        return demoApi.connectPlatform(platform);
      }
      return apiCall(`/api/v1/auth/${platform.toLowerCase()}`);
    },
    disconnectPlatform: async (platform: string) => {
      if (DEMO_MODE) {
        return demoApi.disconnectPlatform(platform);
      }
      return apiCall(`/api/v1/platforms/${platform.toLowerCase()}`, { method: 'DELETE' });
    },
  },

  content: {
    generateScript: async (data: any): Promise<GeneratedScript> => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        return {
          script_id: `script-${Date.now()}`,
          hook: "You won't believe what happens when...",
          main_content: `Here's the amazing content about ${data.topic}. This is going to blow your mind! Follow these simple steps to achieve incredible results.`,
          cta: "Follow for more tips! Link in bio.",
          full_script: `Hook: You won't believe what happens when...\n\nMain: Here's the amazing content about ${data.topic}. This is going to blow your mind! Follow these simple steps to achieve incredible results.\n\nCTA: Follow for more tips! Link in bio.`,
          duration_estimate_seconds: data.duration_seconds || 30,
          suggested_visuals: ['Opening shot with text overlay', 'B-roll of the process', 'Final reveal shot'],
          trending_elements: ['#fyp', '#viral', '#trending'],
          engagement_score: 85,
        };
      }
      return apiCall<GeneratedScript>('/api/v1/content/generate/script', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    generateCaption: async (data: any): Promise<GeneratedCaption> => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 800));
        return {
          caption: `${data.script.substring(0, 100)}... Watch till the end!`,
          hashtags: ['#fyp', '#viral', '#trending', '#foryou', '#foryoupage'],
          character_count: 150,
          estimated_reach_multiplier: 2.3,
        };
      }
      return apiCall<GeneratedCaption>('/api/v1/content/generate/caption', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    generateHashtags: async (description: string, niche?: string, maxHashtags?: number) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 500));
        return {
          hashtags: ['#fyp', '#viral', '#trending', '#foryou', '#foryoupage', '#tiktokviral', '#explore'],
          reach_estimate: 5000000,
        };
      }
      return apiCall('/api/v1/content/generate/hashtags', {
        method: 'POST',
        body: JSON.stringify({ content_description: description, niche, max_hashtags: maxHashtags }),
      });
    },
    chat: async (data: any) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        return {
          conversation_id: data.conversation_id || `conv-${Date.now()}`,
          agent_name: data.agent_name,
          response: `Great question! Based on your content about "${data.message}", I'd recommend focusing on trending sounds and posting at peak hours (7-9 PM). Would you like me to generate a script for this topic?`,
          suggested_actions: [
            { type: 'generate_script', label: 'Generate Script' },
            { type: 'find_trends', label: 'Find Trends' },
          ],
          tokens_used: 150,
        };
      }
      return apiCall('/api/v1/content/chat', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
  },

  auth: {
    getCurrentUser: async () => {
      if (DEMO_MODE) {
        const user = typeof window !== 'undefined' ? localStorage.getItem('user') : null;
        if (user) {
          return JSON.parse(user);
        }
        return null;
      }
      return apiCall('/api/v1/auth/me');
    },
  },

  // NEW: Media Generator API - Connected to real backend at localhost:8004
  media: {
    generateImage: async (data: any) => {
      // For demo mode, return a placeholder image
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        const encodedPrompt = encodeURIComponent(data.prompt?.substring(0, 30) || 'Image');
        return {
          url: `https://placehold.co/1080x1920/667eea/ffffff?text=${encodedPrompt}`,
          image_url: `https://placehold.co/1080x1920/667eea/ffffff?text=${encodedPrompt}`,
        };
      }

      // Map frontend style names to backend enum values
      const styleMap: Record<string, string> = {
        'photorealistic': 'realistic',
        'digital-art': 'vibrant',
        'anime': 'anime',
        'illustration': 'illustration',
        '3d-render': '3d',
        'minimalist': 'minimalist',
      };

      const requestData = {
        prompt: data.prompt,
        style: styleMap[data.style] || 'vibrant',
        aspect_ratio: data.aspect_ratio || (data.preset === 'thumbnail' ? '16:9' :
                      data.preset === 'story' ? '9:16' :
                      data.preset === 'banner' ? '21:9' : '1:1'),
        quality: data.quality || 'standard',
      };

      return mediaApiCall('/api/v1/media/image', {
        method: 'POST',
        body: JSON.stringify(requestData),
      });
    },
    generateVoiceover: async (data: any) => {
      // Map frontend voice names to ElevenLabs voice IDs
      const voiceMap: Record<string, string> = {
        'rachel': '21m00Tcm4TlvDq8ikWAM',
        'domi': 'AZnzlk1XvdvUeBnXmlld',
        'bella': 'EXAVITQu4vr4xnSDxMaL',
        'elli': 'MF3mGyEYCl7XYWbV9V6O',
        'antoni': 'ErXwobaYiN019PkySvjV',
        'josh': 'TxGEqnHWrfWFTfGW9XjX',
        'arnold': 'VR6AewLTigWG4xSOukaG',
        'adam': 'pNInz6obpgDQGcFmaJgB',
        'sam': 'yoZ06aMxZJJ28mfd3POQ',
        // OpenAI voices (used directly)
        'nova': 'nova',
        'shimmer': 'shimmer',
        'echo': 'echo',
        'onyx': 'onyx',
        'fable': 'fable',
        'alloy': 'alloy',
      };

      const voiceId = voiceMap[data.voice?.toLowerCase()] || data.voice_id || '21m00Tcm4TlvDq8ikWAM';
      const isElevenLabsVoice = voiceId.length > 10; // ElevenLabs IDs are longer

      const requestData = {
        text: data.text,
        voice_id: voiceId,
        provider: isElevenLabsVoice ? 'elevenlabs' : 'openai',
        speed: data.speed || 1.0,
        emotion: data.emotion || null,
      };

      return mediaApiCall('/api/v1/media/voiceover', {
        method: 'POST',
        body: JSON.stringify(requestData),
      });
    },
    generateArticle: async (data: any) => {
      return mediaApiCall('/api/v1/media/article', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    generateDiagram: async (data: { description: string; style?: string; format?: string }) => {
      // For demo mode, return a placeholder
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 1500));
        // Generate a mermaid-based diagram URL or placeholder
        const encodedDesc = encodeURIComponent(data.description.substring(0, 50));
        return {
          url: `https://placehold.co/1080x1920/2d3748/e2e8f0?text=${encodedDesc}`,
          diagram_url: `https://placehold.co/1080x1920/2d3748/e2e8f0?text=${encodedDesc}`,
        };
      }
      return mediaApiCall('/api/v1/media/diagram', {
        method: 'POST',
        body: JSON.stringify({
          description: data.description,
          style: data.style || 'modern',
          format: data.format || 'png',
        }),
      });
    },
    searchStockVideos: async (query: string, options?: { orientation?: string; per_page?: number }) => {
      return mediaApiCall('/api/v1/media/video/stock/search', {
        method: 'POST',
        body: JSON.stringify({
          query,
          orientation: options?.orientation || 'portrait',
          per_page: options?.per_page || 10,
        }),
      });
    },
    getJobStatus: async (jobId: string) => {
      return mediaApiCall(`/api/v1/media/jobs/${jobId}`, {
        method: 'GET',
      });
    },
    composeVideo: async (data: { video_urls: string[]; output_format?: string; quality?: string }) => {
      return mediaApiCall('/api/v1/media/video/compose', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
  },

  // NEW: Coaching API
  coaching: {
    getProfile: async (userId: string) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 300));
        return {
          level: 'Creator',
          xp: 2450,
          nextLevelXp: 5000,
          progressPercent: 49,
          skills: { content_creation: 7, engagement: 5, trend_awareness: 8 },
        };
      }
      return apiCall(`/api/v1/coaching/profile/${userId}`);
    },
    getMissions: async (userId: string, type?: string) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 300));
        return [
          { id: '1', title: 'Daily Post', type: 'daily', xpReward: 50, progress: 1, target: 1, status: 'completed' },
          { id: '2', title: 'Community Builder', type: 'daily', xpReward: 75, progress: 6, target: 10, status: 'active' },
        ];
      }
      return apiCall(`/api/v1/coaching/missions/${type || 'today'}/${userId}`);
    },
    getBadges: async (userId: string) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 300));
        return [
          { id: '1', name: 'First Steps', icon: '🎯', rarity: 'common', earned: true },
          { id: '2', name: 'Week Warrior', icon: '🔥', rarity: 'uncommon', earned: true },
        ];
      }
      return apiCall(`/api/v1/coaching/badges/user/${userId}`);
    },
    getActivePlan: async (userId: string) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 300));
        return {
          id: 'plan-1',
          title: '30-Day Fame Accelerator',
          currentDay: 10,
          totalDays: 30,
          progressPercent: 33.3,
        };
      }
      return apiCall(`/api/v1/coaching/plan/active/${userId}`);
    },
    generatePlan: async (data: any) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        return {
          id: `plan-${Date.now()}`,
          title: '30-Day Fame Accelerator',
          description: 'Your personalized growth plan',
        };
      }
      return apiCall('/api/v1/coaching/plan/generate', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    getTips: async (userId: string) => {
      if (DEMO_MODE) {
        return [
          { id: '1', type: 'performance', title: 'Best posting time', content: 'Your audience is most active between 7-9 PM.' },
        ];
      }
      return apiCall(`/api/v1/coaching/tips/${userId}`);
    },
    getStreak: async (userId: string) => {
      if (DEMO_MODE) {
        return { dailyPostStreak: 7, longestStreak: 15 };
      }
      return apiCall(`/api/v1/coaching/streak/${userId}`);
    },
  },

  // NEW: Enhanced Analytics
  analyticsAdvanced: {
    predictViral: async (data: any) => {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        return {
          viral_score: 72,
          predicted_views_low: 5000,
          predicted_views_high: 25000,
          factors: {
            hook_strength: 20,
            trend_alignment: 15,
            posting_time: 12,
            hashtag_mix: 10,
          },
          recommendations: [
            'Add a trending sound to boost discoverability',
            'Post between 7-9 PM for maximum reach',
          ],
          confidence: 0.78,
        };
      }
      return apiCall('/api/v1/analytics/predict-viral', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    getBenchmark: async (niche: string, platform?: string) => {
      if (DEMO_MODE) {
        return {
          niche,
          platform: platform || 'tiktok',
          avg_engagement_rate: 6.5,
          avg_views: 25000,
          top_hashtags: ['#fyp', '#viral', '#trending'],
        };
      }
      return apiCall(`/api/v1/analytics/benchmark/${niche}?platform=${platform || 'tiktok'}`);
    },
    getContentGaps: async (userId: string, niche?: string) => {
      if (DEMO_MODE) {
        return {
          missing_content_types: ['Duets', 'Q&A', 'Tutorials'],
          opportunities: ['Create a content series', 'Collaborate with creators'],
          recommended_topics: ['Behind the scenes', 'Quick tips'],
        };
      }
      return apiCall(`/api/v1/analytics/content-gaps?user_id=${userId}&niche=${niche || ''}`);
    },
    getOptimalTimes: async (userId: string) => {
      if (DEMO_MODE) {
        return {
          best_overall: { day: 'Tuesday', time: '19:00', engagement_boost: 1.23 },
          optimal_times: [
            { day: 'Monday', times: ['12:00', '19:00'] },
            { day: 'Tuesday', times: ['19:00', '20:00'] },
          ],
        };
      }
      return apiCall(`/api/v1/analytics/optimal-times?user_id=${userId}`);
    },
  },

  // Presentation Generator API
  presentations: {
    generate: async (data: any) => {
      // Use v3 endpoint - includes VoiceoverEnforcer for proper video duration
      return presentationApiCall('/api/v1/presentations/generate/v3', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    getJobStatus: async (jobId: string) => {
      return presentationApiCall(`/api/v1/presentations/jobs/${jobId}`);
    },
    listJobs: async (limit: number = 20, offset: number = 0) => {
      return presentationApiCall(`/api/v1/presentations/jobs?limit=${limit}&offset=${offset}`);
    },
  },

  // Course Generator API
  courses: {
    // Get context questions for a category/niche
    getContextQuestions: async (data: { category: string; topic?: string; generate_ai_questions?: boolean }) => {
      return courseApiCall('/api/v1/courses/context-questions', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    // Get context questions by niche (convenience endpoint)
    getContextQuestionsByNiche: async (niche: string) => {
      return courseApiCall(`/api/v1/courses/context-questions/${encodeURIComponent(niche)}`);
    },
    previewOutline: async (data: any) => {
      return courseApiCall('/api/v1/courses/preview-outline', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    generate: async (data: any) => {
      return courseApiCall('/api/v1/courses/generate', {
        method: 'POST',
        body: JSON.stringify(data),
      });
    },
    getJobStatus: async (jobId: string) => {
      return courseApiCall(`/api/v1/courses/jobs/${jobId}`);
    },
    listJobs: async (limit: number = 20, offset: number = 0) => {
      return courseApiCall(`/api/v1/courses/jobs?limit=${limit}&offset=${offset}`);
    },
    deleteJob: async (jobId: string, force: boolean = false) => {
      return courseApiCall(`/api/v1/courses/jobs/${jobId}?force=${force}`, {
        method: 'DELETE',
      });
    },
    reorderOutline: async (jobId: string, sections: any[]) => {
      return courseApiCall(`/api/v1/courses/${jobId}/reorder`, {
        method: 'PUT',
        body: JSON.stringify({ sections }),
      });
    },
    downloadCourse: async (jobId: string) => {
      return courseApiCall(`/api/v1/courses/${jobId}/download`);
    },
    getDifficultyLevels: async () => {
      return courseApiCall('/api/v1/courses/config/difficulty-levels');
    },
    getLessonElements: async () => {
      return courseApiCall('/api/v1/courses/config/lesson-elements');
    },

    // Job Management API
    // Get error queue for a job
    getErrors: async (jobId: string) => {
      return presentationApiCall(`/api/v1/presentations/jobs/v3/${jobId}/errors`);
    },

    // Update lesson content before retry
    updateLessonContent: async (jobId: string, sceneIndex: number, content: { voiceover_text?: string; title?: string; slide_data?: any }) => {
      return presentationApiCall(`/api/v1/presentations/jobs/v3/${jobId}/lessons/${sceneIndex}`, {
        method: 'PATCH',
        body: JSON.stringify(content),
      });
    },

    // Retry a single lesson
    retryLesson: async (jobId: string, sceneIndex: number, options?: { rebuild_final?: boolean }) => {
      return presentationApiCall(`/api/v1/presentations/jobs/v3/${jobId}/lessons/${sceneIndex}/retry`, {
        method: 'POST',
        body: JSON.stringify(options || {}),
      });
    },

    // Retry all failed lessons
    retryAllFailed: async (jobId: string) => {
      return presentationApiCall(`/api/v1/presentations/jobs/v3/${jobId}/retry`, {
        method: 'POST',
      });
    },

    // Cancel job (gracefully)
    cancelJob: async (jobId: string, options?: { keep_completed?: boolean }) => {
      return presentationApiCall(`/api/v1/presentations/jobs/v3/${jobId}/cancel`, {
        method: 'POST',
        body: JSON.stringify(options || { keep_completed: true }),
      });
    },

    // Rebuild final video from completed lessons
    rebuildVideo: async (jobId: string) => {
      return presentationApiCall(`/api/v1/presentations/jobs/v3/${jobId}/rebuild`, {
        method: 'POST',
      });
    },

    // Get lessons for progressive download
    getLessons: async (jobId: string) => {
      return presentationApiCall(`/api/v1/presentations/jobs/v3/${jobId}/lessons`);
    },
  },
};

export default api;
