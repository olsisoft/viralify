'use client';
/* eslint-disable @next/next/no-img-element */

import Image from 'next/image';
import { Calendar, Clock } from 'lucide-react';

interface Post {
  id: string;
  title: string;
  scheduledAt: string;
  thumbnailUrl?: string;
  targetPlatforms?: string[];
}

interface UpcomingPostsProps {
  posts: Post[];
}

export function UpcomingPosts({ posts }: UpcomingPostsProps) {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
      return `In ${diffDays} day${diffDays > 1 ? 's' : ''}`;
    }
    if (diffHours > 0) {
      return `In ${diffHours} hour${diffHours > 1 ? 's' : ''}`;
    }
    const diffMins = Math.floor(diffMs / (1000 * 60));
    if (diffMins > 0) {
      return `In ${diffMins} min${diffMins > 1 ? 's' : ''}`;
    }
    return 'Now';
  };

  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  if (!posts || posts.length === 0) {
    return (
      <div className="text-center py-8">
        <Calendar className="w-12 h-12 text-gray-600 mx-auto mb-3" />
        <p className="text-gray-400">No scheduled posts</p>
        <button className="mt-3 text-sm text-[#fe2c55] hover:underline">
          Schedule your first post
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {posts.slice(0, 3).map((post) => (
        <div
          key={post.id}
          className="flex items-center gap-3 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition cursor-pointer"
        >
          {/* Thumbnail */}
          <div className="w-12 h-16 rounded-lg overflow-hidden bg-gray-800 flex-shrink-0">
            {post.thumbnailUrl ? (
              <img
                src={post.thumbnailUrl}
                alt={post.title}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center">
                <Calendar className="w-5 h-5 text-gray-600" />
              </div>
            )}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <h4 className="text-white text-sm font-medium truncate">{post.title}</h4>
            <div className="flex items-center gap-2 mt-1">
              <Clock className="w-3 h-3 text-gray-500" />
              <span className="text-xs text-gray-400">{formatTime(post.scheduledAt)}</span>
            </div>
            {post.targetPlatforms && post.targetPlatforms.length > 0 && (
              <div className="flex gap-1 mt-1">
                {post.targetPlatforms.map((platform) => (
                  <span
                    key={platform}
                    className="px-2 py-0.5 text-xs rounded bg-white/10 text-gray-300"
                  >
                    {platform}
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Time Badge */}
          <div className="text-right flex-shrink-0">
            <span className="text-xs text-[#25f4ee] font-medium">
              {formatDate(post.scheduledAt)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
