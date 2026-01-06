'use client';
/* eslint-disable @next/next/no-img-element */

import Image from 'next/image';
import { Eye, Heart, MessageCircle, Play } from 'lucide-react';

// Demo data for recent posts
const DEMO_RECENT_POSTS = [
  {
    id: '1',
    title: 'Morning Routine Tips',
    thumbnail: 'https://picsum.photos/seed/post1/400/600',
    views: 125000,
    likes: 8900,
    comments: 342,
    publishedAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '2',
    title: '5-Minute Breakfast Recipe',
    thumbnail: 'https://picsum.photos/seed/post2/400/600',
    views: 89000,
    likes: 5600,
    comments: 234,
    publishedAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '3',
    title: 'Viral Dance Tutorial',
    thumbnail: 'https://picsum.photos/seed/post3/400/600',
    views: 456000,
    likes: 32000,
    comments: 1200,
    publishedAt: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
  },
];

export function RecentPosts() {
  const formatCount = (count: number) => {
    if (count >= 1000000) {
      return `${(count / 1000000).toFixed(1)}M`;
    }
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}K`;
    }
    return count.toString();
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
      return `${diffDays}d ago`;
    }
    if (diffHours > 0) {
      return `${diffHours}h ago`;
    }
    return 'Just now';
  };

  return (
    <div className="space-y-4">
      {DEMO_RECENT_POSTS.map((post) => (
        <div
          key={post.id}
          className="flex gap-4 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition cursor-pointer"
        >
          {/* Thumbnail */}
          <div className="relative w-20 h-28 rounded-lg overflow-hidden flex-shrink-0">
            <img
              src={post.thumbnail}
              alt={post.title}
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 flex items-center justify-center bg-black/30 opacity-0 hover:opacity-100 transition">
              <Play className="w-8 h-8 text-white" fill="white" />
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <h3 className="text-white font-medium truncate">{post.title}</h3>
            <p className="text-gray-400 text-sm mt-1">{formatTimeAgo(post.publishedAt)}</p>

            {/* Stats */}
            <div className="flex items-center gap-4 mt-3 text-sm text-gray-400">
              <div className="flex items-center gap-1">
                <Eye className="w-4 h-4" />
                <span>{formatCount(post.views)}</span>
              </div>
              <div className="flex items-center gap-1">
                <Heart className="w-4 h-4" />
                <span>{formatCount(post.likes)}</span>
              </div>
              <div className="flex items-center gap-1">
                <MessageCircle className="w-4 h-4" />
                <span>{formatCount(post.comments)}</span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
