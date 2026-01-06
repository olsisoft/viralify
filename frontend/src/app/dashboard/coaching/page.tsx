'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Trophy, Target, Flame, Star, TrendingUp, Award,
  ChevronRight, Zap, Calendar, CheckCircle2, Clock,
  Sparkles, Medal, Crown, Info
} from 'lucide-react';
import toast from 'react-hot-toast';
import { DEMO_MODE } from '@/lib/demo-mode';

// Types
interface SkillProfile {
  level: string;
  xp: number;
  nextLevelXp: number;
  progressPercent: number;
  skills: Record<string, number>;
}

interface Mission {
  id: string;
  title: string;
  description: string;
  type: string;
  xpReward: number;
  progress: number;
  target: number;
  status: 'active' | 'completed';
}

interface Badge {
  id: string;
  name: string;
  icon: string;
  rarity: string;
  earned: boolean;
  earnedAt?: Date;
}

interface CoachingTip {
  id: string;
  type: string;
  title: string;
  content: string;
  priority: number;
}

// Demo data
const DEMO_PROFILE: SkillProfile = {
  level: 'Creator',
  xp: 2450,
  nextLevelXp: 5000,
  progressPercent: 49,
  skills: {
    content_creation: 7,
    engagement: 5,
    trend_awareness: 8,
    consistency: 6,
    storytelling: 4,
    analytics: 3,
  },
};

const DEMO_MISSIONS: Mission[] = [
  { id: '1', title: 'Daily Post', description: 'Post 1 piece of content today', type: 'daily', xpReward: 50, progress: 1, target: 1, status: 'completed' },
  { id: '2', title: 'Engage with Community', description: 'Reply to 10 comments', type: 'daily', xpReward: 75, progress: 6, target: 10, status: 'active' },
  { id: '3', title: 'Trend Surfer', description: 'Use a trending sound/hashtag', type: 'daily', xpReward: 100, progress: 0, target: 1, status: 'active' },
];

const DEMO_BADGES: Badge[] = [
  { id: '1', name: 'First Steps', icon: '🎯', rarity: 'common', earned: true, earnedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) },
  { id: '2', name: 'Week Warrior', icon: '🔥', rarity: 'uncommon', earned: true, earnedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000) },
  { id: '3', name: 'Trend Rider', icon: '🌊', rarity: 'rare', earned: false },
  { id: '4', name: 'Viral Hit', icon: '🚀', rarity: 'epic', earned: false },
];

const DEMO_TIPS: CoachingTip[] = [
  { id: '1', type: 'performance', title: 'Best posting time', content: 'Your audience is most active between 7-9 PM. Try posting then!', priority: 5 },
  { id: '2', type: 'trend', title: 'Trending opportunity', content: 'The sound "Original Sound - trending" is gaining traction in your niche.', priority: 4 },
];

const LEVEL_COLORS: Record<string, string> = {
  Beginner: 'from-gray-500 to-gray-600',
  Creator: 'from-blue-500 to-cyan-500',
  'Rising Star': 'from-purple-500 to-pink-500',
  Influencer: 'from-orange-500 to-red-500',
  Celebrity: 'from-yellow-400 to-orange-500',
};

const RARITY_COLORS: Record<string, string> = {
  common: 'border-gray-500 bg-gray-500/10',
  uncommon: 'border-green-500 bg-green-500/10',
  rare: 'border-blue-500 bg-blue-500/10',
  epic: 'border-purple-500 bg-purple-500/10',
  legendary: 'border-yellow-500 bg-yellow-500/10',
};

export default function CoachingPage() {
  const [profile, setProfile] = useState<SkillProfile | null>(null);
  const [missions, setMissions] = useState<Mission[]>([]);
  const [badges, setBadges] = useState<Badge[]>([]);
  const [tips, setTips] = useState<CoachingTip[]>([]);
  const [streak, setStreak] = useState(7);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setIsLoading(true);
    try {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 500));
        setProfile(DEMO_PROFILE);
        setMissions(DEMO_MISSIONS);
        setBadges(DEMO_BADGES);
        setTips(DEMO_TIPS);
        setStreak(7);
      }
    } catch (error) {
      toast.error('Failed to load coaching data');
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="animate-spin w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Trophy className="w-8 h-8 text-yellow-400" />
              Fame Coaching
            </h1>
            <p className="text-gray-400 mt-1">Your personalized path to viral success</p>
          </div>
          {DEMO_MODE && (
            <div className="px-3 py-1 bg-yellow-500/20 border border-yellow-500/30 rounded-lg">
              <span className="text-sm text-yellow-200">Demo Mode</span>
            </div>
          )}
        </div>

        {/* Level & XP Card */}
        {profile && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-800 rounded-2xl p-6 border border-gray-700 mb-8"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-4">
                <div className={`w-16 h-16 rounded-2xl bg-gradient-to-r ${LEVEL_COLORS[profile.level]} flex items-center justify-center`}>
                  <Crown className="w-8 h-8 text-white" />
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-2xl font-bold text-white">{profile.level}</span>
                    <span className="px-2 py-0.5 bg-purple-600/20 text-purple-400 rounded text-sm">
                      {profile.xp.toLocaleString()} XP
                    </span>
                  </div>
                  <p className="text-gray-400 text-sm">
                    {profile.nextLevelXp - profile.xp} XP to next level
                  </p>
                </div>
              </div>

              {/* Streak */}
              <div className="flex items-center gap-2 px-4 py-2 bg-orange-600/20 rounded-xl">
                <Flame className="w-6 h-6 text-orange-400" />
                <div>
                  <span className="text-2xl font-bold text-white">{streak}</span>
                  <span className="text-sm text-gray-400 ml-1">day streak</span>
                </div>
              </div>
            </div>

            {/* XP Progress Bar */}
            <div className="relative">
              <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${profile.progressPercent}%` }}
                  transition={{ duration: 1, ease: 'easeOut' }}
                  className="h-full bg-gradient-to-r from-purple-600 to-pink-600 rounded-full"
                />
              </div>
              <div className="flex justify-between mt-2 text-xs text-gray-400">
                <span>0 XP</span>
                <span>{profile.nextLevelXp.toLocaleString()} XP</span>
              </div>
            </div>
          </motion.div>
        )}

        {/* Main Grid */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Left Column - Missions & Plan */}
          <div className="lg:col-span-2 space-y-6">
            {/* Today&apos;s Missions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Target className="w-5 h-5 text-purple-400" />
                  Today&apos;s Missions
                </h2>
                <Link
                  href="/dashboard/coaching/missions"
                  className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
                >
                  View All <ChevronRight className="w-4 h-4" />
                </Link>
              </div>

              <div className="space-y-3">
                {missions.map((mission) => (
                  <div
                    key={mission.id}
                    className={`p-4 rounded-xl border ${
                      mission.status === 'completed'
                        ? 'bg-green-600/10 border-green-500/30'
                        : 'bg-gray-700/50 border-gray-600'
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
                          ) : (
                            <Target className="w-5 h-5 text-purple-400" />
                          )}
                        </div>
                        <div>
                          <h3 className="text-white font-medium">{mission.title}</h3>
                          <p className="text-gray-400 text-sm">{mission.description}</p>
                        </div>
                      </div>
                      <span className="flex items-center gap-1 text-yellow-400 text-sm">
                        <Zap className="w-4 h-4" />
                        +{mission.xpReward} XP
                      </span>
                    </div>

                    {mission.status !== 'completed' && (
                      <div className="mt-3">
                        <div className="flex justify-between text-xs text-gray-400 mb-1">
                          <span>Progress</span>
                          <span>{mission.progress}/{mission.target}</span>
                        </div>
                        <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-purple-500 rounded-full"
                            style={{ width: `${(mission.progress / mission.target) * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Growth Plan */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-2xl p-6 border border-purple-500/30"
            >
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-purple-400" />
                    30-Day Growth Plan
                  </h2>
                  <p className="text-gray-400 text-sm mt-1">Your personalized path to 10K followers</p>
                </div>
                <span className="px-3 py-1 bg-green-600/20 text-green-400 rounded-full text-sm">
                  Day 10 of 30
                </span>
              </div>

              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="bg-gray-800/50 rounded-xl p-3 text-center">
                  <p className="text-2xl font-bold text-white">5,234</p>
                  <p className="text-xs text-gray-400">Current Followers</p>
                </div>
                <div className="bg-gray-800/50 rounded-xl p-3 text-center">
                  <p className="text-2xl font-bold text-purple-400">10,000</p>
                  <p className="text-xs text-gray-400">Target</p>
                </div>
                <div className="bg-gray-800/50 rounded-xl p-3 text-center">
                  <p className="text-2xl font-bold text-green-400">52%</p>
                  <p className="text-xs text-gray-400">Progress</p>
                </div>
              </div>

              <Link
                href="/dashboard/coaching/plan"
                className="block w-full py-3 bg-purple-600 text-white rounded-xl hover:bg-purple-700 transition text-center font-medium"
              >
                View Full Plan
              </Link>
            </motion.div>

            {/* Skill Radar */}
            {profile && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
              >
                <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-purple-400" />
                  Your Skills
                </h2>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {Object.entries(profile.skills).map(([skill, level]) => (
                    <div key={skill} className="bg-gray-700/50 rounded-xl p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-300 capitalize">
                          {skill.replace('_', ' ')}
                        </span>
                        <span className="text-sm text-purple-400">{level}/10</span>
                      </div>
                      <div className="h-2 bg-gray-600 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
                          style={{ width: `${level * 10}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </div>

          {/* Right Column - Badges & Tips */}
          <div className="space-y-6">
            {/* Recent Badges */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Award className="w-5 h-5 text-yellow-400" />
                  Achievements
                </h2>
                <Link
                  href="/dashboard/coaching/achievements"
                  className="text-sm text-purple-400 hover:text-purple-300 flex items-center"
                >
                  View All <ChevronRight className="w-4 h-4" />
                </Link>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {badges.map((badge) => (
                  <div
                    key={badge.id}
                    className={`p-3 rounded-xl border-2 ${RARITY_COLORS[badge.rarity]} ${
                      !badge.earned ? 'opacity-50 grayscale' : ''
                    }`}
                  >
                    <div className="text-2xl mb-1">{badge.icon}</div>
                    <p className="text-white text-sm font-medium">{badge.name}</p>
                    <p className="text-xs text-gray-400 capitalize">{badge.rarity}</p>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Coaching Tips */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
            >
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Info className="w-5 h-5 text-blue-400" />
                AI Coach Tips
              </h2>

              <div className="space-y-3">
                {tips.map((tip) => (
                  <div
                    key={tip.id}
                    className="p-3 bg-blue-600/10 border border-blue-500/30 rounded-xl"
                  >
                    <div className="flex items-center gap-2 mb-1">
                      {tip.priority >= 4 && <Star className="w-4 h-4 text-yellow-400" />}
                      <span className="text-white font-medium text-sm">{tip.title}</span>
                    </div>
                    <p className="text-gray-400 text-sm">{tip.content}</p>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Quick Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
            >
              <h2 className="text-lg font-semibold text-white mb-4">This Week</h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Missions Completed</span>
                  <span className="text-white font-semibold">12/15</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">XP Earned</span>
                  <span className="text-purple-400 font-semibold">+850</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Badges Earned</span>
                  <span className="text-yellow-400 font-semibold">2</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Posts Published</span>
                  <span className="text-green-400 font-semibold">8</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
