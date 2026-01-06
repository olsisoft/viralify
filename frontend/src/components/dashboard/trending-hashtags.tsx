'use client';

import { TrendingUp } from 'lucide-react';

interface Hashtag {
  id: string;
  hashtag: string;
  view_count: number;
  trend_score: number;
  growth_rate?: number;
}

interface TrendingHashtagsProps {
  hashtags: Hashtag[];
}

export function TrendingHashtags({ hashtags }: TrendingHashtagsProps) {
  const formatCount = (count: number) => {
    if (count >= 1000000) {
      return `${(count / 1000000).toFixed(1)}M`;
    }
    if (count >= 1000) {
      return `${(count / 1000).toFixed(0)}K`;
    }
    return count.toString();
  };

  if (!hashtags || hashtags.length === 0) {
    return (
      <div className="text-center text-gray-400 py-4">
        No trending hashtags available
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {hashtags.slice(0, 5).map((tag, index) => (
        <div
          key={tag.id}
          className="flex items-center justify-between p-3 rounded-xl bg-white/5 hover:bg-white/10 transition cursor-pointer"
        >
          <div className="flex items-center gap-3">
            <span className="text-gray-500 text-sm w-4">{index + 1}</span>
            <div>
              <p className="text-white font-medium">{tag.hashtag}</p>
              <p className="text-xs text-gray-400">{formatCount(tag.view_count)} views</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {tag.growth_rate && tag.growth_rate > 0 && (
              <div className="flex items-center gap-1 text-green-400 text-xs">
                <TrendingUp className="w-3 h-3" />
                <span>+{tag.growth_rate}%</span>
              </div>
            )}
            <div className="w-12 h-2 rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[#25f4ee] to-[#fe2c55]"
                style={{ width: `${tag.trend_score}%` }}
              />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
