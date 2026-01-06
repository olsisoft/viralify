'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Award, Trophy, Star, Lock, ChevronLeft, Filter,
  Flame, Target, TrendingUp, Zap, Calendar, Crown
} from 'lucide-react';
import { DEMO_MODE } from '@/lib/demo-mode';

interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: 'milestone' | 'streak' | 'skill' | 'challenge';
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
  xpValue: number;
  earned: boolean;
  earnedAt?: Date;
  progress?: number;
  target?: number;
}

const DEMO_BADGES: Badge[] = [
  // Milestones
  { id: '1', name: 'First Steps', description: 'Published your first content', icon: '🎯', category: 'milestone', rarity: 'common', xpValue: 50, earned: true, earnedAt: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000) },
  { id: '2', name: 'Rising Creator', description: 'Reached 1,000 followers', icon: '⭐', category: 'milestone', rarity: 'uncommon', xpValue: 300, earned: true, earnedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) },
  { id: '3', name: 'Influencer Status', description: 'Reached 10,000 followers', icon: '🌟', category: 'milestone', rarity: 'epic', xpValue: 1500, earned: false, progress: 5234, target: 10000 },
  { id: '4', name: 'Celebrity', description: 'Reached 100,000 followers', icon: '👑', category: 'milestone', rarity: 'legendary', xpValue: 5000, earned: false, progress: 5234, target: 100000 },

  // Streaks
  { id: '5', name: 'Week Warrior', description: '7-day posting streak', icon: '🔥', category: 'streak', rarity: 'uncommon', xpValue: 200, earned: true, earnedAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000) },
  { id: '6', name: 'Consistency King', description: '30-day posting streak', icon: '💪', category: 'streak', rarity: 'epic', xpValue: 1000, earned: false, progress: 7, target: 30 },
  { id: '7', name: 'Unstoppable', description: '100-day posting streak', icon: '🏆', category: 'streak', rarity: 'legendary', xpValue: 3000, earned: false, progress: 7, target: 100 },

  // Skills
  { id: '8', name: 'Engagement Master', description: 'Achieved 10% engagement rate', icon: '💬', category: 'skill', rarity: 'rare', xpValue: 400, earned: false, progress: 7, target: 10 },
  { id: '9', name: 'Trend Rider', description: 'Used 5 trending sounds/hashtags', icon: '🌊', category: 'skill', rarity: 'uncommon', xpValue: 150, earned: true, earnedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000) },
  { id: '10', name: 'Viral Hit', description: 'Got 10,000+ views on a single video', icon: '🚀', category: 'skill', rarity: 'rare', xpValue: 500, earned: false, progress: 4500, target: 10000 },
  { id: '11', name: 'Million Views', description: 'Got 1,000,000+ views on a single video', icon: '🎆', category: 'skill', rarity: 'legendary', xpValue: 2500, earned: false, progress: 4500, target: 1000000 },

  // Challenges
  { id: '12', name: 'Early Bird', description: 'Post before 8 AM for 7 days', icon: '🌅', category: 'challenge', rarity: 'uncommon', xpValue: 250, earned: false, progress: 2, target: 7 },
  { id: '13', name: 'Night Owl', description: 'Post after 10 PM for 7 days', icon: '🌙', category: 'challenge', rarity: 'uncommon', xpValue: 250, earned: false, progress: 3, target: 7 },
  { id: '14', name: 'Collaborator', description: 'Collaborate with 5 different creators', icon: '🤝', category: 'challenge', rarity: 'rare', xpValue: 500, earned: false, progress: 1, target: 5 },
  { id: '15', name: 'Multi-Platform', description: 'Post on 3 different platforms', icon: '📱', category: 'challenge', rarity: 'uncommon', xpValue: 200, earned: true, earnedAt: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000) },
];

const RARITY_STYLES = {
  common: { border: 'border-gray-500', bg: 'bg-gray-500/10', text: 'text-gray-400', label: 'Common' },
  uncommon: { border: 'border-green-500', bg: 'bg-green-500/10', text: 'text-green-400', label: 'Uncommon' },
  rare: { border: 'border-blue-500', bg: 'bg-blue-500/10', text: 'text-blue-400', label: 'Rare' },
  epic: { border: 'border-purple-500', bg: 'bg-purple-500/10', text: 'text-purple-400', label: 'Epic' },
  legendary: { border: 'border-yellow-500', bg: 'bg-yellow-500/10', text: 'text-yellow-400', label: 'Legendary' },
};

const CATEGORY_INFO = {
  milestone: { icon: TrendingUp, label: 'Milestones', color: 'text-blue-400' },
  streak: { icon: Flame, label: 'Streaks', color: 'text-orange-400' },
  skill: { icon: Target, label: 'Skills', color: 'text-purple-400' },
  challenge: { icon: Trophy, label: 'Challenges', color: 'text-yellow-400' },
};

export default function AchievementsPage() {
  const [badges] = useState<Badge[]>(DEMO_BADGES);
  const [filter, setFilter] = useState<'all' | 'earned' | 'locked'>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  const filteredBadges = badges.filter(b => {
    if (filter === 'earned' && !b.earned) return false;
    if (filter === 'locked' && b.earned) return false;
    if (categoryFilter !== 'all' && b.category !== categoryFilter) return false;
    return true;
  });

  const earnedCount = badges.filter(b => b.earned).length;
  const totalXP = badges.filter(b => b.earned).reduce((acc, b) => acc + b.xpValue, 0);

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/dashboard/coaching"
            className="text-gray-400 hover:text-white flex items-center gap-2 mb-4"
          >
            <ChevronLeft className="w-4 h-4" />
            Back to Coaching
          </Link>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                <Award className="w-8 h-8 text-yellow-400" />
                Achievements
              </h1>
              <p className="text-gray-400 mt-1">Collect badges and show off your accomplishments</p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-white">{earnedCount}/{badges.length}</p>
              <p className="text-sm text-gray-400">Badges earned</p>
            </div>
          </div>
        </div>

        {/* Stats Banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-4 gap-4 mb-8"
        >
          {Object.entries(RARITY_STYLES).map(([rarity, style]) => {
            const count = badges.filter(b => b.rarity === rarity && b.earned).length;
            const total = badges.filter(b => b.rarity === rarity).length;
            return (
              <div key={rarity} className={`p-4 rounded-xl border-2 ${style.border} ${style.bg}`}>
                <div className={`text-2xl font-bold ${style.text}`}>{count}/{total}</div>
                <div className="text-sm text-gray-400">{style.label}</div>
              </div>
            );
          })}
        </motion.div>

        {/* Filters */}
        <div className="flex flex-wrap gap-4 mb-6">
          <div className="flex gap-2">
            {(['all', 'earned', 'locked'] as const).map((type) => (
              <button
                key={type}
                onClick={() => setFilter(type)}
                className={`px-4 py-2 rounded-xl text-sm transition ${
                  filter === type
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                {type === 'all' ? 'All' : type === 'earned' ? 'Earned' : 'Locked'}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => setCategoryFilter('all')}
              className={`px-3 py-2 rounded-xl text-sm transition ${
                categoryFilter === 'all'
                  ? 'bg-gray-700 text-white'
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700'
              }`}
            >
              All Categories
            </button>
            {Object.entries(CATEGORY_INFO).map(([cat, info]) => (
              <button
                key={cat}
                onClick={() => setCategoryFilter(cat)}
                className={`px-3 py-2 rounded-xl text-sm transition flex items-center gap-1 ${
                  categoryFilter === cat
                    ? 'bg-gray-700 text-white'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700'
                }`}
              >
                <info.icon className={`w-4 h-4 ${info.color}`} />
                {info.label}
              </button>
            ))}
          </div>
        </div>

        {/* Badges Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredBadges.map((badge, index) => (
            <BadgeCard key={badge.id} badge={badge} index={index} />
          ))}
        </div>

        {filteredBadges.length === 0 && (
          <div className="text-center py-12">
            <Award className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400">No badges match your filters</p>
          </div>
        )}

        {/* Total XP earned */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-8 p-4 bg-gradient-to-r from-purple-600/20 to-pink-600/20 border border-purple-500/30 rounded-xl text-center"
        >
          <p className="text-gray-400 text-sm">Total XP from badges</p>
          <p className="text-3xl font-bold text-white flex items-center justify-center gap-2">
            <Zap className="w-6 h-6 text-yellow-400" />
            {totalXP.toLocaleString()} XP
          </p>
        </motion.div>
      </div>
    </div>
  );
}

function BadgeCard({ badge, index }: { badge: Badge; index: number }) {
  const style = RARITY_STYLES[badge.rarity];
  const progressPercent = badge.target ? (badge.progress! / badge.target) * 100 : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className={`relative p-5 rounded-xl border-2 ${style.border} ${style.bg} ${
        !badge.earned ? 'opacity-70' : ''
      }`}
    >
      {!badge.earned && (
        <div className="absolute top-3 right-3">
          <Lock className="w-4 h-4 text-gray-500" />
        </div>
      )}

      <div className="flex items-start gap-4">
        <div className={`text-4xl ${!badge.earned ? 'grayscale' : ''}`}>
          {badge.icon}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-white font-semibold">{badge.name}</h3>
            <span className={`px-2 py-0.5 rounded text-xs ${style.text} bg-black/20`}>
              {style.label}
            </span>
          </div>
          <p className="text-gray-400 text-sm mt-1">{badge.description}</p>

          {badge.earned ? (
            <p className="text-green-400 text-xs mt-2 flex items-center gap-1">
              <Star className="w-3 h-3" />
              Earned {badge.earnedAt?.toLocaleDateString()}
            </p>
          ) : badge.target && (
            <div className="mt-3">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>Progress</span>
                <span>{badge.progress?.toLocaleString()}/{badge.target.toLocaleString()}</span>
              </div>
              <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${style.border.replace('border-', 'bg-')}`}
                  style={{ width: `${Math.min(progressPercent, 100)}%` }}
                />
              </div>
            </div>
          )}

          <div className="mt-2 flex items-center gap-1 text-yellow-400 text-sm">
            <Zap className="w-3 h-3" />
            +{badge.xpValue} XP
          </div>
        </div>
      </div>
    </motion.div>
  );
}
