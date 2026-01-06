'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  TrendingUp, Target, Calendar, CheckCircle2, Circle,
  ChevronLeft, Sparkles, BarChart3, Users, Eye,
  Clock, Zap, RefreshCw, Loader2
} from 'lucide-react';
import toast from 'react-hot-toast';
import { DEMO_MODE } from '@/lib/demo-mode';

interface Goal {
  metric: string;
  current: number;
  target: number;
  icon: React.ReactNode;
}

interface Milestone {
  day: number;
  title: string;
  description: string;
  status: 'completed' | 'in_progress' | 'pending';
  tasks: string[];
}

interface GrowthPlan {
  id: string;
  type: string;
  title: string;
  description: string;
  startDate: Date;
  endDate: Date;
  currentDay: number;
  totalDays: number;
  goals: Goal[];
  milestones: Milestone[];
  dailyTasks: string[];
  weeklyFocus: string[];
}

const DEMO_PLAN: GrowthPlan = {
  id: 'plan-1',
  type: '30_day',
  title: '30-Day Fame Accelerator',
  description: 'Your personalized path to viral success and reaching 10K followers',
  startDate: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000),
  endDate: new Date(Date.now() + 20 * 24 * 60 * 60 * 1000),
  currentDay: 10,
  totalDays: 30,
  goals: [
    { metric: 'Followers', current: 5234, target: 10000, icon: <Users className="w-5 h-5" /> },
    { metric: 'Avg. Views', current: 23500, target: 50000, icon: <Eye className="w-5 h-5" /> },
    { metric: 'Engagement', current: 5.2, target: 8, icon: <BarChart3 className="w-5 h-5" /> },
  ],
  milestones: [
    {
      day: 7,
      title: 'Foundation Phase',
      description: 'Establish consistent posting schedule and brand voice',
      status: 'completed',
      tasks: [
        'Set up posting schedule (daily at optimal times)',
        'Define your niche and unique angle',
        'Create content templates',
        'Post 7 pieces of content',
      ],
    },
    {
      day: 14,
      title: 'Growth Acceleration',
      description: 'Leverage trends and collaborations for rapid growth',
      status: 'in_progress',
      tasks: [
        'Use 5+ trending sounds/hashtags',
        'Collaborate with 2 creators in your niche',
        'Analyze top-performing content',
        'Double down on what works',
      ],
    },
    {
      day: 21,
      title: 'Community Building',
      description: 'Focus on engagement and building loyal followers',
      status: 'pending',
      tasks: [
        'Respond to all comments within 1 hour',
        'Host a Q&A or live session',
        'Create user-generated content campaign',
        'Build email list or community',
      ],
    },
    {
      day: 30,
      title: 'Goal Achievement',
      description: 'Reach your target metrics and establish sustainable growth',
      status: 'pending',
      tasks: [
        'Review analytics and adjust strategy',
        'Plan next 30-day goals',
        'Celebrate achievements!',
        'Set up passive growth systems',
      ],
    },
  ],
  dailyTasks: [
    'Post at least 1 piece of content',
    'Engage with 20 accounts in your niche',
    'Reply to all comments within 1 hour',
    'Research trending topics for 15 minutes',
    'Analyze yesterday\'s performance',
  ],
  weeklyFocus: [
    'Week 1-2: Content consistency and quality improvement',
    'Week 2-3: Trend identification and participation',
    'Week 3-4: Audience engagement and community building',
    'Week 4: Analytics review and strategy adjustment',
  ],
};

export default function PlanPage() {
  const [plan, setPlan] = useState<GrowthPlan>(DEMO_PLAN);
  const [isRegenerating, setIsRegenerating] = useState(false);

  const progressPercent = (plan.currentDay / plan.totalDays) * 100;

  const handleRegenerate = async () => {
    setIsRegenerating(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));
      toast.success('Plan updated with latest insights! (Demo)');
    } catch (error) {
      toast.error('Failed to regenerate plan');
    } finally {
      setIsRegenerating(false);
    }
  };

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
                <TrendingUp className="w-8 h-8 text-purple-400" />
                Growth Plan
              </h1>
              <p className="text-gray-400 mt-1">{plan.title}</p>
            </div>
            <button
              onClick={handleRegenerate}
              disabled={isRegenerating}
              className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-xl text-gray-300 hover:bg-gray-700 transition flex items-center gap-2 disabled:opacity-50"
            >
              {isRegenerating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <RefreshCw className="w-4 h-4" />
              )}
              Regenerate
            </button>
          </div>
        </div>

        {/* Progress Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-purple-600/20 to-pink-600/20 rounded-2xl p-6 border border-purple-500/30 mb-8"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <span className="text-purple-400 text-sm font-medium">Day {plan.currentDay} of {plan.totalDays}</span>
              <h2 className="text-2xl font-bold text-white">{plan.title}</h2>
            </div>
            <div className="text-right">
              <span className="text-3xl font-bold text-white">{Math.round(progressPercent)}%</span>
              <p className="text-gray-400 text-sm">Complete</p>
            </div>
          </div>

          <div className="h-3 bg-gray-700 rounded-full overflow-hidden mb-4">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${progressPercent}%` }}
              transition={{ duration: 1, ease: 'easeOut' }}
              className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"
            />
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">
              <Calendar className="w-4 h-4 inline mr-1" />
              Started {plan.startDate.toLocaleDateString()}
            </span>
            <span className="text-gray-400">
              <Clock className="w-4 h-4 inline mr-1" />
              {plan.totalDays - plan.currentDay} days remaining
            </span>
          </div>
        </motion.div>

        {/* Goals */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid md:grid-cols-3 gap-4 mb-8"
        >
          {plan.goals.map((goal, index) => {
            const progress = (goal.current / goal.target) * 100;
            return (
              <div key={index} className="bg-gray-800 rounded-xl p-5 border border-gray-700">
                <div className="flex items-center justify-between mb-3">
                  <div className="p-2 bg-purple-600/20 rounded-lg text-purple-400">
                    {goal.icon}
                  </div>
                  <span className={`text-sm font-medium ${progress >= 100 ? 'text-green-400' : 'text-gray-400'}`}>
                    {progress >= 100 ? 'Achieved!' : `${Math.round(progress)}%`}
                  </span>
                </div>
                <h3 className="text-white font-medium mb-1">{goal.metric}</h3>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold text-white">
                    {typeof goal.current === 'number' && goal.current % 1 !== 0
                      ? goal.current.toFixed(1)
                      : goal.current.toLocaleString()}
                  </span>
                  <span className="text-gray-400">/ {goal.target.toLocaleString()}</span>
                </div>
                <div className="h-1.5 bg-gray-700 rounded-full mt-3 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${progress >= 100 ? 'bg-green-500' : 'bg-purple-500'}`}
                    style={{ width: `${Math.min(progress, 100)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Milestones */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
          >
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Target className="w-5 h-5 text-purple-400" />
              Milestones
            </h2>

            <div className="space-y-4">
              {plan.milestones.map((milestone, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-xl border ${
                    milestone.status === 'completed'
                      ? 'bg-green-600/10 border-green-500/30'
                      : milestone.status === 'in_progress'
                      ? 'bg-purple-600/10 border-purple-500/30'
                      : 'bg-gray-700/50 border-gray-600'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-1 rounded-full ${
                      milestone.status === 'completed'
                        ? 'bg-green-500'
                        : milestone.status === 'in_progress'
                        ? 'bg-purple-500'
                        : 'bg-gray-600'
                    }`}>
                      {milestone.status === 'completed' ? (
                        <CheckCircle2 className="w-4 h-4 text-white" />
                      ) : (
                        <Circle className="w-4 h-4 text-white" />
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <h3 className="text-white font-medium">{milestone.title}</h3>
                        <span className="text-xs text-gray-400">Day {milestone.day}</span>
                      </div>
                      <p className="text-gray-400 text-sm mt-1">{milestone.description}</p>

                      {milestone.status === 'in_progress' && (
                        <div className="mt-3 space-y-1">
                          {milestone.tasks.map((task, i) => (
                            <div key={i} className="flex items-center gap-2 text-xs text-gray-300">
                              <div className="w-1.5 h-1.5 rounded-full bg-purple-400" />
                              {task}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Daily Tasks & Weekly Focus */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
            >
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-green-400" />
                Daily Tasks
              </h2>

              <div className="space-y-2">
                {plan.dailyTasks.map((task, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-3 p-3 bg-gray-700/50 rounded-xl"
                  >
                    <input
                      type="checkbox"
                      className="w-5 h-5 rounded accent-green-500"
                    />
                    <span className="text-gray-300 text-sm">{task}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-gray-800 rounded-2xl p-6 border border-gray-700"
            >
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Calendar className="w-5 h-5 text-blue-400" />
                Weekly Focus Areas
              </h2>

              <div className="space-y-2">
                {plan.weeklyFocus.map((focus, index) => (
                  <div
                    key={index}
                    className={`p-3 rounded-xl ${
                      index === Math.floor(plan.currentDay / 7)
                        ? 'bg-blue-600/20 border border-blue-500/30'
                        : 'bg-gray-700/50'
                    }`}
                  >
                    <span className="text-gray-300 text-sm">{focus}</span>
                    {index === Math.floor(plan.currentDay / 7) && (
                      <span className="ml-2 text-xs text-blue-400">(Current)</span>
                    )}
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="p-4 bg-gradient-to-r from-green-600/10 to-emerald-600/10 border border-green-500/20 rounded-xl"
            >
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-5 h-5 text-green-400" />
                <span className="text-white font-medium">AI Recommendation</span>
              </div>
              <p className="text-gray-400 text-sm">
                Based on your progress, focus on collaborations this week.
                Creators who collaborate see 40% faster growth on average.
              </p>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}
