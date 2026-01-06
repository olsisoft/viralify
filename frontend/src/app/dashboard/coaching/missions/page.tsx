'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Target, CheckCircle2, Clock, Zap, Calendar, Trophy,
  Flame, Star, ChevronLeft, Filter, RefreshCw
} from 'lucide-react';
import toast from 'react-hot-toast';
import { DEMO_MODE } from '@/lib/demo-mode';

interface Mission {
  id: string;
  title: string;
  description: string;
  type: 'daily' | 'weekly' | 'challenge';
  category: string;
  difficulty: 'easy' | 'medium' | 'hard';
  xpReward: number;
  progress: number;
  target: number;
  status: 'active' | 'completed' | 'expired';
  expiresAt?: Date;
  badgeReward?: string;
}

const DEMO_MISSIONS: Mission[] = [
  // Daily
  { id: '1', title: 'Daily Post', description: 'Post at least 1 piece of content today', type: 'daily', category: 'content', difficulty: 'easy', xpReward: 50, progress: 1, target: 1, status: 'completed' },
  { id: '2', title: 'Community Builder', description: 'Reply to 10 comments on your content', type: 'daily', category: 'engagement', difficulty: 'medium', xpReward: 75, progress: 6, target: 10, status: 'active' },
  { id: '3', title: 'Trend Surfer', description: 'Create content using a trending sound or hashtag', type: 'daily', category: 'growth', difficulty: 'medium', xpReward: 100, progress: 0, target: 1, status: 'active' },
  { id: '4', title: 'Story Time', description: 'Post a story/update on your account', type: 'daily', category: 'content', difficulty: 'easy', xpReward: 25, progress: 0, target: 1, status: 'active' },
  // Weekly
  { id: '5', title: 'Prolific Creator', description: 'Post 5 pieces of content this week', type: 'weekly', category: 'content', difficulty: 'medium', xpReward: 300, progress: 3, target: 5, status: 'active' },
  { id: '6', title: 'Network Builder', description: 'Collaborate or duet with another creator', type: 'weekly', category: 'growth', difficulty: 'hard', xpReward: 500, progress: 0, target: 1, status: 'active' },
  { id: '7', title: 'Engagement Master', description: 'Reply to 50 comments total', type: 'weekly', category: 'engagement', difficulty: 'medium', xpReward: 250, progress: 32, target: 50, status: 'active' },
  // Challenges
  { id: '8', title: '7-Day Streak Challenge', description: 'Post every day for 7 consecutive days', type: 'challenge', category: 'content', difficulty: 'hard', xpReward: 750, progress: 5, target: 7, status: 'active', badgeReward: 'Week Warrior' },
  { id: '9', title: 'Viral Video Challenge', description: 'Get 10,000 views on a single video', type: 'challenge', category: 'growth', difficulty: 'hard', xpReward: 1000, progress: 4500, target: 10000, status: 'active', badgeReward: 'Viral Hit' },
];

const DIFFICULTY_COLORS = {
  easy: 'bg-green-600/20 text-green-400',
  medium: 'bg-yellow-600/20 text-yellow-400',
  hard: 'bg-red-600/20 text-red-400',
};

const CATEGORY_ICONS: Record<string, string> = {
  content: '📹',
  engagement: '💬',
  growth: '📈',
  learning: '📚',
};

export default function MissionsPage() {
  const [missions, setMissions] = useState<Mission[]>(DEMO_MISSIONS);
  const [filter, setFilter] = useState<'all' | 'daily' | 'weekly' | 'challenge'>('all');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  const filteredMissions = missions.filter(m => {
    if (filter !== 'all' && m.type !== filter) return false;
    if (categoryFilter !== 'all' && m.category !== categoryFilter) return false;
    return true;
  });

  const dailyMissions = filteredMissions.filter(m => m.type === 'daily');
  const weeklyMissions = filteredMissions.filter(m => m.type === 'weekly');
  const challenges = filteredMissions.filter(m => m.type === 'challenge');

  const completedToday = missions.filter(m => m.type === 'daily' && m.status === 'completed').length;
  const totalDaily = missions.filter(m => m.type === 'daily').length;

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
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
                <Target className="w-8 h-8 text-purple-400" />
                Missions
              </h1>
              <p className="text-gray-400 mt-1">Complete missions to earn XP and level up</p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-white">{completedToday}/{totalDaily}</p>
              <p className="text-sm text-gray-400">Daily missions done</p>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-4 mb-6">
          <div className="flex gap-2">
            {(['all', 'daily', 'weekly', 'challenge'] as const).map((type) => (
              <button
                key={type}
                onClick={() => setFilter(type)}
                className={`px-4 py-2 rounded-xl text-sm transition ${
                  filter === type
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                {type === 'all' ? 'All' : type.charAt(0).toUpperCase() + type.slice(1)}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            {['all', 'content', 'engagement', 'growth'].map((cat) => (
              <button
                key={cat}
                onClick={() => setCategoryFilter(cat)}
                className={`px-3 py-2 rounded-xl text-sm transition flex items-center gap-1 ${
                  categoryFilter === cat
                    ? 'bg-gray-700 text-white'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700'
                }`}
              >
                {cat !== 'all' && <span>{CATEGORY_ICONS[cat]}</span>}
                {cat === 'all' ? 'All Categories' : cat.charAt(0).toUpperCase() + cat.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Daily Missions */}
        {(filter === 'all' || filter === 'daily') && dailyMissions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="flex items-center gap-2 mb-4">
              <Clock className="w-5 h-5 text-blue-400" />
              <h2 className="text-lg font-semibold text-white">Daily Missions</h2>
              <span className="text-sm text-gray-400">Resets in 8 hours</span>
            </div>
            <div className="space-y-3">
              {dailyMissions.map((mission) => (
                <MissionCard key={mission.id} mission={mission} />
              ))}
            </div>
          </motion.div>
        )}

        {/* Weekly Missions */}
        {(filter === 'all' || filter === 'weekly') && weeklyMissions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-8"
          >
            <div className="flex items-center gap-2 mb-4">
              <Calendar className="w-5 h-5 text-green-400" />
              <h2 className="text-lg font-semibold text-white">Weekly Missions</h2>
              <span className="text-sm text-gray-400">Resets in 3 days</span>
            </div>
            <div className="space-y-3">
              {weeklyMissions.map((mission) => (
                <MissionCard key={mission.id} mission={mission} />
              ))}
            </div>
          </motion.div>
        )}

        {/* Challenges */}
        {(filter === 'all' || filter === 'challenge') && challenges.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-8"
          >
            <div className="flex items-center gap-2 mb-4">
              <Trophy className="w-5 h-5 text-yellow-400" />
              <h2 className="text-lg font-semibold text-white">Challenges</h2>
              <span className="text-sm text-gray-400">Special achievements</span>
            </div>
            <div className="space-y-3">
              {challenges.map((mission) => (
                <MissionCard key={mission.id} mission={mission} />
              ))}
            </div>
          </motion.div>
        )}

        {filteredMissions.length === 0 && (
          <div className="text-center py-12">
            <Target className="w-12 h-12 text-gray-600 mx-auto mb-3" />
            <p className="text-gray-400">No missions match your filters</p>
          </div>
        )}
      </div>
    </div>
  );
}

function MissionCard({ mission }: { mission: Mission }) {
  const progressPercent = (mission.progress / mission.target) * 100;

  return (
    <motion.div
      whileHover={{ scale: 1.01 }}
      className={`p-4 rounded-xl border transition ${
        mission.status === 'completed'
          ? 'bg-green-600/10 border-green-500/30'
          : 'bg-gray-800 border-gray-700 hover:border-gray-600'
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className={`p-2 rounded-lg ${
            mission.status === 'completed'
              ? 'bg-green-600/20'
              : 'bg-purple-600/20'
          }`}>
            {mission.status === 'completed' ? (
              <CheckCircle2 className="w-5 h-5 text-green-400" />
            ) : mission.type === 'challenge' ? (
              <Trophy className="w-5 h-5 text-yellow-400" />
            ) : (
              <Target className="w-5 h-5 text-purple-400" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-white font-medium">{mission.title}</h3>
              <span className={`px-2 py-0.5 rounded text-xs ${DIFFICULTY_COLORS[mission.difficulty]}`}>
                {mission.difficulty}
              </span>
              <span className="text-xs text-gray-500">{CATEGORY_ICONS[mission.category]}</span>
            </div>
            <p className="text-gray-400 text-sm mt-1">{mission.description}</p>
            {mission.badgeReward && (
              <p className="text-yellow-400 text-xs mt-1 flex items-center gap-1">
                <Star className="w-3 h-3" />
                Unlocks &quot;{mission.badgeReward}&quot; badge
              </p>
            )}
          </div>
        </div>
        <span className="flex items-center gap-1 text-yellow-400 text-sm whitespace-nowrap">
          <Zap className="w-4 h-4" />
          +{mission.xpReward} XP
        </span>
      </div>

      {mission.status !== 'completed' && (
        <div className="mt-4">
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>Progress</span>
            <span>
              {mission.progress.toLocaleString()}/{mission.target.toLocaleString()}
            </span>
          </div>
          <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(progressPercent, 100)}%` }}
              transition={{ duration: 0.5 }}
              className={`h-full rounded-full ${
                mission.type === 'challenge'
                  ? 'bg-gradient-to-r from-yellow-500 to-orange-500'
                  : 'bg-gradient-to-r from-purple-500 to-pink-500'
              }`}
            />
          </div>
        </div>
      )}
    </motion.div>
  );
}
