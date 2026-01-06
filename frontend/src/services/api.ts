import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios';

// Demo mode flag - set in lib/demo-mode.ts
const DEMO_MODE = true; // Sync with lib/demo-mode.ts

// ========================================
// API Client Configuration
// ========================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

class ApiClient {
  private client: AxiosInstance;
  private isRefreshing = false;
  private refreshSubscribers: ((token: string) => void)[] = [];

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor - add auth token
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('accessToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor - handle token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

        if (error.response?.status === 401 && !originalRequest._retry) {
          if (this.isRefreshing) {
            return new Promise((resolve) => {
              this.refreshSubscribers.push((token: string) => {
                originalRequest.headers = {
                  ...originalRequest.headers,
                  Authorization: `Bearer ${token}`,
                };
                resolve(this.client(originalRequest));
              });
            });
          }

          originalRequest._retry = true;
          this.isRefreshing = true;

          try {
            const refreshToken = localStorage.getItem('refreshToken');
            const response = await this.client.post('/api/v1/auth/refresh', {
              refreshToken,
            });

            const { accessToken, refreshToken: newRefreshToken } = response.data;
            localStorage.setItem('accessToken', accessToken);
            localStorage.setItem('refreshToken', newRefreshToken);

            this.refreshSubscribers.forEach((callback) => callback(accessToken));
            this.refreshSubscribers = [];

            return this.client(originalRequest);
          } catch (refreshError) {
            localStorage.removeItem('accessToken');
            localStorage.removeItem('refreshToken');
            localStorage.removeItem('user');
            window.location.href = '/auth/login';
            return Promise.reject(refreshError);
          } finally {
            this.isRefreshing = false;
          }
        }

        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.put<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.delete<T>(url, config);
    return response.data;
  }
}

const apiClient = new ApiClient();

// ========================================
// Auth Service
// ========================================

// ========================================
// Platform Types (Multi-Platform Support)
// ========================================

export type PlatformType = 'TIKTOK' | 'INSTAGRAM' | 'YOUTUBE';

export interface PlatformInfo {
  id: PlatformType;
  name: string;
  displayName: string;
  maxDurationSeconds: number;
  maxCaptionLength: number;
  maxHashtags: number;
  hashtagsInCaption: boolean;
  icon: string;
  color: string;
}

export const PLATFORMS: Record<PlatformType, PlatformInfo> = {
  TIKTOK: {
    id: 'TIKTOK',
    name: 'tiktok',
    displayName: 'TikTok',
    maxDurationSeconds: 600,
    maxCaptionLength: 2200,
    maxHashtags: 30,
    hashtagsInCaption: true,
    icon: 'video',
    color: '#000000',
  },
  INSTAGRAM: {
    id: 'INSTAGRAM',
    name: 'instagram',
    displayName: 'Instagram Reels',
    maxDurationSeconds: 90,
    maxCaptionLength: 2200,
    maxHashtags: 30,
    hashtagsInCaption: true,
    icon: 'instagram',
    color: '#E4405F',
  },
  YOUTUBE: {
    id: 'YOUTUBE',
    name: 'youtube',
    displayName: 'YouTube Shorts',
    maxDurationSeconds: 60,
    maxCaptionLength: 5000,
    maxHashtags: 500, // total chars for tags
    hashtagsInCaption: false,
    icon: 'youtube',
    color: '#FF0000',
  },
};

export interface PlatformAccount {
  id: string;
  platform: PlatformType;
  platformUserId: string;
  platformUsername: string;
  platformDisplayName?: string;
  platformAvatarUrl?: string;
  followerCount: number;
  accountStatus: 'active' | 'expired' | 'revoked' | 'error';
  lastSyncAt?: string;
}

export interface PlatformStatus {
  platform: PlatformType;
  status: 'pending' | 'processing' | 'uploading' | 'published' | 'failed' | 'cancelled' | 'skipped';
  platformPostId?: string;
  platformShareUrl?: string;
  errorMessage?: string;
  publishedAt?: string;
  adaptedCaption?: string;
  adaptedHashtags?: string[];
}

export interface PlatformSpecificSettings {
  // TikTok
  allowDuet?: boolean;
  allowStitch?: boolean;
  commercialContent?: boolean;
  brandedContent?: boolean;
  // Instagram
  locationId?: string;
  userTags?: string[];
  // YouTube
  playlistId?: string;
  categoryId?: string;
  visibility?: 'public' | 'unlisted' | 'private';
}

export interface User {
  id: string;
  email: string;
  fullName?: string;
  avatarUrl?: string;
  tiktokUserId?: string;
  tiktokUsername?: string;
  tiktokDisplayName?: string;
  tiktokAvatarUrl?: string;
  tiktokFollowerCount?: number;
  tiktokConnected: boolean;
  // Multi-platform support
  connectedPlatforms?: PlatformAccount[];
  planType: string;
  monthlyPostsLimit: number;
  monthlyPostsUsed: number;
  monthlyAiGenerationsLimit: number;
  monthlyAiGenerationsUsed: number;
}

export interface AuthResponse {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  user: User;
  tiktokConnected?: boolean;
}

export const authService = {
  async register(email: string, password: string, fullName?: string): Promise<AuthResponse> {
    return apiClient.post('/api/v1/auth/register', { email, password, fullName });
  },

  async login(email: string, password: string): Promise<AuthResponse> {
    return apiClient.post('/api/v1/auth/login', { email, password });
  },

  async logout(): Promise<void> {
    const token = localStorage.getItem('accessToken');
    await apiClient.post('/api/v1/auth/logout', { token });
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    localStorage.removeItem('user');
  },

  async getCurrentUser(): Promise<User> {
    return apiClient.get('/api/v1/auth/me');
  },

  getTikTokAuthUrl(): string {
    return `${API_BASE_URL}/api/v1/auth/tiktok`;
  },
};

// ========================================
// Content Service
// ========================================

export interface GenerateScriptRequest {
  topic: string;
  niche?: string;
  target_audience?: string;
  duration_seconds?: number;
  tone?: string;
  include_trends?: boolean;
}

export interface ScriptResponse {
  script_id: string;
  hook: string;
  main_content: string;
  cta: string;
  full_script: string;
  duration_estimate_seconds: number;
  suggested_visuals: string[];
  trending_elements: string[];
  engagement_score: number;
}

export interface GenerateCaptionRequest {
  script: string;
  max_length?: number;
  include_hashtags?: boolean;
  include_cta?: boolean;
}

export interface CaptionResponse {
  caption: string;
  hashtags: string[];
  character_count: number;
  estimated_reach_multiplier: number;
}

export interface ChatRequest {
  agent_name: string;
  message: string;
  conversation_id?: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  conversation_id: string;
  agent_name: string;
  response: string;
  suggested_actions: Array<{ type: string; label: string }>;
  tokens_used: number;
}

export const contentService = {
  async generateScript(request: GenerateScriptRequest): Promise<ScriptResponse> {
    return apiClient.post('/api/v1/content/generate/script', request);
  },

  async generateCaption(request: GenerateCaptionRequest): Promise<CaptionResponse> {
    return apiClient.post('/api/v1/content/generate/caption', request);
  },

  async generateHashtags(contentDescription: string, niche?: string, maxHashtags?: number) {
    return apiClient.post('/api/v1/content/generate/hashtags', {
      content_description: contentDescription,
      niche,
      max_hashtags: maxHashtags,
    });
  },

  async chatWithAgent(request: ChatRequest): Promise<ChatResponse> {
    return apiClient.post('/api/v1/content/chat', request);
  },

  async getAgents() {
    return apiClient.get('/api/v1/content/agents');
  },
};

// ========================================
// Scheduler Service
// ========================================

export interface ScheduledPost {
  id: string;
  userId: string;
  title: string;
  caption?: string;
  hashtags: string[];
  videoUrl: string;
  thumbnailUrl?: string;
  scheduledAt: string;
  privacyLevel: string;
  status: string;
  // Multi-platform support
  targetPlatforms: PlatformType[];
  platformStatuses?: PlatformStatus[];
  // Legacy TikTok fields (kept for backwards compatibility)
  tiktokPostId?: string;
  tiktokShareUrl?: string;
  errorMessage?: string;
  retryCount: number;
  publishedAt?: string;
  createdAt: string;
}

export interface CreateScheduledPostRequest {
  title: string;
  caption?: string;
  hashtags?: string[];
  videoUrl: string;
  videoSizeBytes?: number;
  videoDurationSeconds?: number;
  scheduledAt: string;
  privacyLevel?: string;
  allowComments?: boolean;
  // Multi-platform support
  targetPlatforms?: PlatformType[];
  platformSettings?: Record<PlatformType, PlatformSpecificSettings>;
  // TikTok specific (legacy, also in platformSettings)
  allowDuet?: boolean;
  allowStitch?: boolean;
}

// Forward declare for demo mode (will be imported at the top later)
let _demoScheduledPosts: any[] = [];

export const schedulerService = {
  async createPost(request: CreateScheduledPostRequest): Promise<ScheduledPost> {
    if (DEMO_MODE) {
      const { demoApi } = await import('@/lib/demo-mode');
      const result = await demoApi.createScheduledPost(request);
      return result.data;
    }
    return apiClient.post('/api/v1/scheduler/posts', request);
  },

  async getPosts(): Promise<ScheduledPost[]> {
    if (DEMO_MODE) {
      const { demoApi } = await import('@/lib/demo-mode');
      return demoApi.getScheduledPosts();
    }
    return apiClient.get('/api/v1/scheduler/posts');
  },

  async getPendingPosts(): Promise<ScheduledPost[]> {
    if (DEMO_MODE) {
      const { demoApi } = await import('@/lib/demo-mode');
      const posts = await demoApi.getScheduledPosts();
      return posts.filter((p: any) => p.status === 'pending');
    }
    return apiClient.get('/api/v1/scheduler/posts/pending');
  },

  async getPost(postId: string): Promise<ScheduledPost> {
    if (DEMO_MODE) {
      const { demoApi } = await import('@/lib/demo-mode');
      const posts = await demoApi.getScheduledPosts();
      return posts.find((p: any) => p.id === postId);
    }
    return apiClient.get(`/api/v1/scheduler/posts/${postId}`);
  },

  async updatePost(postId: string, data: Partial<CreateScheduledPostRequest>): Promise<ScheduledPost> {
    if (DEMO_MODE) {
      const { DEMO_SCHEDULED_POSTS } = await import('@/lib/demo-mode');
      const post = DEMO_SCHEDULED_POSTS.find((p: any) => p.id === postId);
      if (post) {
        Object.assign(post, data);
      }
      return post;
    }
    return apiClient.put(`/api/v1/scheduler/posts/${postId}`, data);
  },

  async cancelPost(postId: string): Promise<void> {
    if (DEMO_MODE) {
      const { demoApi } = await import('@/lib/demo-mode');
      await demoApi.cancelScheduledPost(postId);
      return;
    }
    return apiClient.delete(`/api/v1/scheduler/posts/${postId}`);
  },

  async getStats() {
    if (DEMO_MODE) {
      const { demoApi } = await import('@/lib/demo-mode');
      return demoApi.getStats();
    }
    return apiClient.get('/api/v1/scheduler/stats');
  },
};

// ========================================
// Analytics Service
// ========================================

export interface AnalyticsSummary {
  period: string;
  total_views: number;
  total_likes: number;
  total_comments: number;
  total_shares: number;
  avg_engagement_rate: number;
  top_performing_post?: {
    post_id: string;
    views: number;
    engagement_rate: number;
  };
  growth_rate: {
    views: number;
    engagement: number;
  };
}

export interface PerformanceInsights {
  best_posting_times: Array<{
    hour: number;
    day: string;
    avg_engagement: number;
  }>;
  content_performance_by_type: Record<string, any>;
  audience_insights: Record<string, any>;
  recommendations: string[];
}

export interface TimeSeriesData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
  }>;
}

export const analyticsService = {
  async getSummary(period: string = '7d'): Promise<AnalyticsSummary> {
    return apiClient.get(`/api/v1/analytics/summary?period=${period}`);
  },

  async getInsights(): Promise<PerformanceInsights> {
    return apiClient.get('/api/v1/analytics/insights');
  },

  async getTimeSeries(metric: string, period: string = '7d'): Promise<TimeSeriesData> {
    return apiClient.get(`/api/v1/analytics/time-series?metric=${metric}&period=${period}`);
  },

  async getDashboard() {
    return apiClient.get('/api/v1/analytics/dashboard');
  },

  async getPostAnalytics(postId: string) {
    return apiClient.get(`/api/v1/analytics/posts/${postId}`);
  },
};

// ========================================
// Trends Service
// ========================================

export interface TrendingHashtag {
  id: string;
  hashtag: string;
  region: string;
  category?: string;
  view_count: number;
  video_count: number;
  trend_score: number;
  growth_rate: number;
  is_trending: boolean;
}

export interface TrendingSound {
  id: string;
  sound_id: string;
  sound_title?: string;
  sound_author?: string;
  region: string;
  usage_count: number;
  trend_score: number;
  growth_rate: number;
  is_trending: boolean;
}

export interface ViralPatterns {
  patterns: Array<{
    name: string;
    description: string;
    effectiveness: number;
    examples: string[];
  }>;
  common_elements: string[];
  optimal_duration: Record<string, any>;
  best_posting_times: Array<{
    time: string;
    day: string;
    engagement_boost: number;
  }>;
}

export const trendsService = {
  async getHashtags(region: string = 'global', limit: number = 20): Promise<TrendingHashtag[]> {
    return apiClient.get(`/api/v1/trends/hashtags?region=${region}&limit=${limit}`);
  },

  async getSounds(region: string = 'global', limit: number = 20): Promise<TrendingSound[]> {
    return apiClient.get(`/api/v1/trends/sounds?region=${region}&limit=${limit}`);
  },

  async getNicheTrends(niche: string, region: string = 'global') {
    return apiClient.get(`/api/v1/trends/niche/${niche}?region=${region}`);
  },

  async getViralPatterns(niche?: string): Promise<ViralPatterns> {
    return apiClient.get(`/api/v1/trends/viral-patterns${niche ? `?niche=${niche}` : ''}`);
  },

  async predictTrend(trendName: string) {
    return apiClient.get(`/api/v1/trends/predict/${trendName}`);
  },

  async getBestPostingTimes(timezone?: string) {
    return apiClient.get(`/api/v1/trends/best-times${timezone ? `?timezone=${timezone}` : ''}`);
  },
};

// ========================================
// Platform Service (Multi-Platform)
// ========================================

export interface ContentAdaptation {
  platform: PlatformType;
  title: string;
  caption: string;
  hashtags: string[];
  description?: string; // YouTube
  tags?: string[]; // YouTube
  suggestedDurationSeconds?: number;
  contentWasModified: boolean;
  adaptationNotes: string[];
}

export const platformService = {
  async getConnectedAccounts(): Promise<PlatformAccount[]> {
    if (DEMO_MODE) {
      const { demoApi } = await import('@/lib/demo-mode');
      return demoApi.getConnectedAccounts();
    }
    return apiClient.get('/api/v1/platforms/accounts');
  },

  async getOAuthUrl(platform: PlatformType): Promise<{ url: string }> {
    if (DEMO_MODE) {
      // Return a fake URL that won't actually navigate
      return { url: `javascript:alert('Demo Mode: ${platform} OAuth would open here')` };
    }
    return apiClient.get(`/api/v1/auth/${platform.toLowerCase()}`);
  },

  async disconnectPlatform(platform: PlatformType): Promise<void> {
    if (DEMO_MODE) {
      const { demoApi, DEMO_PLATFORM_ACCOUNTS } = await import('@/lib/demo-mode');
      await demoApi.disconnectPlatform(platform);
      // Remove from demo accounts
      const index = DEMO_PLATFORM_ACCOUNTS.findIndex(a => a.platform === platform);
      if (index !== -1) {
        DEMO_PLATFORM_ACCOUNTS.splice(index, 1);
      }
      return;
    }
    return apiClient.delete(`/api/v1/platforms/${platform.toLowerCase()}`);
  },

  async getContentLimits(platform: PlatformType) {
    if (DEMO_MODE) {
      return PLATFORMS[platform];
    }
    return apiClient.get(`/api/v1/platforms/${platform.toLowerCase()}/limits`);
  },

  async previewAdaptation(
    title: string,
    caption: string,
    hashtags: string[],
    durationSeconds: number,
    platforms: PlatformType[]
  ): Promise<ContentAdaptation[]> {
    if (DEMO_MODE) {
      // Generate mock adaptations
      return platforms.map(platform => ({
        platform,
        title,
        caption: caption.substring(0, PLATFORMS[platform].maxCaptionLength),
        hashtags: hashtags.slice(0, PLATFORMS[platform].maxHashtags),
        suggestedDurationSeconds: Math.min(durationSeconds, PLATFORMS[platform].maxDurationSeconds),
        contentWasModified: durationSeconds > PLATFORMS[platform].maxDurationSeconds || caption.length > PLATFORMS[platform].maxCaptionLength,
        adaptationNotes: [],
      }));
    }
    return apiClient.post('/api/v1/content/adapt/preview', {
      title,
      caption,
      hashtags,
      durationSeconds,
      platforms,
    });
  },

  async refreshPlatformToken(platform: PlatformType): Promise<void> {
    if (DEMO_MODE) {
      return;
    }
    return apiClient.post(`/api/v1/platforms/${platform.toLowerCase()}/refresh`);
  },
};

// ========================================
// Auth Service Extensions (Platform OAuth)
// ========================================

export const authServiceExtended = {
  ...authService,

  getInstagramAuthUrl(): string {
    return `${API_BASE_URL}/api/v1/auth/instagram`;
  },

  getYouTubeAuthUrl(): string {
    return `${API_BASE_URL}/api/v1/auth/youtube`;
  },

  getPlatformAuthUrl(platform: PlatformType): string {
    return `${API_BASE_URL}/api/v1/auth/${platform.toLowerCase()}`;
  },
};

export default apiClient;
