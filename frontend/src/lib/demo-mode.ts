// Demo Mode Configuration
// This allows testing the frontend without a running backend

export const DEMO_MODE = false; // Set to true for demo without backend

export const DEMO_USER = {
  id: 'demo-user-123',
  email: 'demo@viralify.com',
  name: 'Demo User',
  avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=demo',
  plan: 'Pro',
  createdAt: new Date().toISOString(),
};

export const DEMO_TOKENS = {
  accessToken: 'demo-access-token-xyz',
  refreshToken: 'demo-refresh-token-xyz',
};

export const DEMO_PLATFORM_ACCOUNTS = [
  {
    id: 'pa-1',
    platform: 'TIKTOK' as const,
    platformUserId: 'tiktok-123',
    platformUsername: 'viralify_demo',
    platformDisplayName: 'Viralify Demo',
    platformAvatarUrl: 'https://api.dicebear.com/7.x/avataaars/svg?seed=tiktok',
    followerCount: 125000,
    accountStatus: 'active' as const,
    lastSyncAt: new Date().toISOString(),
  },
  {
    id: 'pa-2',
    platform: 'INSTAGRAM' as const,
    platformUserId: 'ig-456',
    platformUsername: 'viralify_demo',
    platformDisplayName: 'Viralify Demo',
    platformAvatarUrl: 'https://api.dicebear.com/7.x/avataaars/svg?seed=instagram',
    followerCount: 89000,
    accountStatus: 'active' as const,
    lastSyncAt: new Date().toISOString(),
  },
];

export const DEMO_SCHEDULED_POSTS = [
  {
    id: 'post-1',
    title: 'Morning Routine Tips',
    caption: 'Start your day right with these productivity hacks!',
    hashtags: ['productivity', 'morning', 'routine', 'tips'],
    videoUrl: 'https://example.com/video1.mp4',
    thumbnailUrl: 'https://picsum.photos/seed/post1/400/600',
    scheduledAt: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(), // 2 hours from now
    status: 'pending' as const,
    targetPlatforms: ['TIKTOK', 'INSTAGRAM'] as ('TIKTOK' | 'INSTAGRAM' | 'YOUTUBE')[],
    platformStatuses: [
      { platform: 'TIKTOK', status: 'pending' },
      { platform: 'INSTAGRAM', status: 'pending' },
    ],
  },
  {
    id: 'post-2',
    title: 'Quick Recipe: 5-min Breakfast',
    caption: 'Delicious and healthy breakfast in just 5 minutes!',
    hashtags: ['food', 'recipe', 'breakfast', 'healthy'],
    videoUrl: 'https://example.com/video2.mp4',
    thumbnailUrl: 'https://picsum.photos/seed/post2/400/600',
    scheduledAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(), // Tomorrow
    status: 'pending' as const,
    targetPlatforms: ['TIKTOK', 'YOUTUBE'] as ('TIKTOK' | 'INSTAGRAM' | 'YOUTUBE')[],
    platformStatuses: [
      { platform: 'TIKTOK', status: 'pending' },
      { platform: 'YOUTUBE', status: 'pending' },
    ],
  },
  {
    id: 'post-3',
    title: 'Viral Dance Tutorial',
    caption: 'Learn this trending dance in 60 seconds!',
    hashtags: ['dance', 'tutorial', 'viral', 'trending'],
    videoUrl: 'https://example.com/video3.mp4',
    thumbnailUrl: 'https://picsum.photos/seed/post3/400/600',
    scheduledAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(), // 2 hours ago
    status: 'published' as const,
    targetPlatforms: ['TIKTOK'] as ('TIKTOK' | 'INSTAGRAM' | 'YOUTUBE')[],
    platformStatuses: [
      { platform: 'TIKTOK', status: 'published', platformPostId: 'tt-123', platformShareUrl: 'https://tiktok.com/@demo/video/123' },
    ],
    tiktokPostId: 'tt-123',
  },
];

export const DEMO_STATS = {
  totalViews: 1250000,
  totalLikes: 89000,
  totalComments: 12500,
  totalShares: 5600,
  engagementRate: 7.2,
  followerGrowth: 12.5,
  postsThisWeek: 8,
  avgViewsPerPost: 156000,
};

// Demo API handlers
export const demoApi = {
  // Auth
  login: async (email: string, password: string) => {
    await simulateDelay();
    if (email && password.length >= 4) {
      return {
        success: true,
        data: {
          accessToken: DEMO_TOKENS.accessToken,
          refreshToken: DEMO_TOKENS.refreshToken,
          user: { ...DEMO_USER, email },
        },
      };
    }
    return { success: false, error: 'Invalid credentials' };
  },

  register: async (name: string, email: string, password: string) => {
    await simulateDelay();
    if (name && email && password.length >= 8) {
      return {
        success: true,
        data: {
          message: 'Account created successfully',
          user: { ...DEMO_USER, name, email },
        },
      };
    }
    return { success: false, error: 'Invalid registration data' };
  },

  // Platform accounts
  getConnectedAccounts: async () => {
    await simulateDelay();
    return DEMO_PLATFORM_ACCOUNTS;
  },

  connectPlatform: async (platform: string) => {
    await simulateDelay();
    return {
      success: true,
      message: `${platform} connected successfully (Demo Mode)`,
    };
  },

  disconnectPlatform: async (platform: string) => {
    await simulateDelay();
    return { success: true };
  },

  // Scheduled posts
  getScheduledPosts: async () => {
    await simulateDelay();
    return DEMO_SCHEDULED_POSTS;
  },

  createScheduledPost: async (postData: any) => {
    await simulateDelay();
    const newPost = {
      id: `post-${Date.now()}`,
      ...postData,
      status: 'pending',
      platformStatuses: postData.targetPlatforms.map((p: string) => ({
        platform: p,
        status: 'pending',
      })),
    };
    DEMO_SCHEDULED_POSTS.unshift(newPost);
    return { success: true, data: newPost };
  },

  cancelScheduledPost: async (postId: string) => {
    await simulateDelay();
    const index = DEMO_SCHEDULED_POSTS.findIndex(p => p.id === postId);
    if (index !== -1) {
      DEMO_SCHEDULED_POSTS.splice(index, 1);
    }
    return { success: true };
  },

  // Stats
  getStats: async () => {
    await simulateDelay();
    return DEMO_STATS;
  },
};

function simulateDelay(ms: number = 500) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Helper to check if we're in demo mode
export function isDemoMode(): boolean {
  return DEMO_MODE;
}

// Initialize demo session
export function initDemoSession() {
  if (DEMO_MODE && typeof window !== 'undefined') {
    const existingUser = localStorage.getItem('user');
    if (!existingUser) {
      // Don't auto-login, let user go through login/register flow
    }
  }
}
