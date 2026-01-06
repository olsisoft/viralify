'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  TrendingUp, Eye, Heart, MessageCircle, Share2, Calendar,
  Sparkles, Zap, Target, BarChart3, Clock, ArrowUpRight,
  Play, Plus, Bell
} from 'lucide-react';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { StatsCard } from '@/components/dashboard/stats-card';
import { TrendingHashtags } from '@/components/dashboard/trending-hashtags';
import { RecentPosts } from '@/components/dashboard/recent-posts';
import { AIAssistantWidget } from '@/components/dashboard/ai-assistant-widget';
import { PerformanceChart } from '@/components/dashboard/performance-chart';
import { UpcomingPosts } from '@/components/dashboard/upcoming-posts';
import { QuickActions } from '@/components/dashboard/quick-actions';
import { api, DashboardStats } from '@/lib/api';

export default function DashboardPage() {
  const { data: stats, isLoading: statsLoading } = useQuery<DashboardStats>({
    queryKey: ['dashboard-stats'],
    queryFn: () => api.analytics.getDashboard(),
  });

  const { data: trends } = useQuery({
    queryKey: ['trending-hashtags'],
    queryFn: () => api.trends.getHashtags({ limit: 10 }),
  });

  const { data: scheduledPosts } = useQuery({
    queryKey: ['scheduled-posts'],
    queryFn: () => api.scheduler.getPendingPosts(),
  });

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white">
              Welcome back! 👋
            </h1>
            <p className="text-gray-400 mt-1">
              Here&apos;s what&apos;s happening with your TikTok content
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button className="glass px-4 py-2 rounded-xl text-white flex items-center gap-2 hover:bg-white/10 transition">
              <Bell className="w-4 h-4" />
              <span className="hidden sm:inline">Notifications</span>
            </button>
            <Link
              href="/dashboard/create"
              className="bg-gradient-to-r from-[#fe2c55] to-[#ff6b6b] px-4 py-2 rounded-xl text-white flex items-center gap-2 hover:opacity-90 transition"
            >
              <Plus className="w-4 h-4" />
              <span>Create Post</span>
            </Link>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <StatsCard
              title="Total Views"
              value={stats?.total_views || 0}
              change={stats?.growth_metrics?.views_growth || 0}
              icon={<Eye className="w-5 h-5" />}
              color="from-blue-500 to-cyan-500"
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <StatsCard
              title="Total Likes"
              value={stats?.total_likes || 0}
              change={stats?.growth_metrics?.likes_growth || 0}
              icon={<Heart className="w-5 h-5" />}
              color="from-pink-500 to-rose-500"
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <StatsCard
              title="Engagement Rate"
              value={`${stats?.avg_engagement_rate || 0}%`}
              change={12.5}
              icon={<TrendingUp className="w-5 h-5" />}
              color="from-green-500 to-emerald-500"
            />
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <StatsCard
              title="Posts This Week"
              value={stats?.posting_frequency?.daily_avg || 0}
              subtitle="/day avg"
              icon={<Calendar className="w-5 h-5" />}
              color="from-purple-500 to-violet-500"
            />
          </motion.div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Performance & AI */}
          <div className="lg:col-span-2 space-y-6">
            {/* Performance Chart */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="glass rounded-2xl p-6 card-glow"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-[#fe2c55]" />
                  Performance Overview
                </h2>
                <select className="bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-white">
                  <option value="7d">Last 7 days</option>
                  <option value="30d">Last 30 days</option>
                  <option value="90d">Last 90 days</option>
                </select>
              </div>
              <PerformanceChart />
            </motion.div>

            {/* AI Assistant Widget */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <AIAssistantWidget />
            </motion.div>

            {/* Recent Posts */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="glass rounded-2xl p-6"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold text-white flex items-center gap-2">
                  <Play className="w-5 h-5 text-[#25f4ee]" />
                  Recent Posts
                </h2>
                <button className="text-sm text-[#fe2c55] hover:underline">
                  View all
                </button>
              </div>
              <RecentPosts />
            </motion.div>
          </div>

          {/* Right Column - Trends & Schedule */}
          <div className="space-y-6">
            {/* Quick Actions */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <QuickActions />
            </motion.div>

            {/* Trending Hashtags */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="glass rounded-2xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  Trending Now
                </h2>
                <button className="text-sm text-[#fe2c55] hover:underline">
                  See all
                </button>
              </div>
              <TrendingHashtags hashtags={trends || []} />
            </motion.div>

            {/* Upcoming Posts */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
              className="glass rounded-2xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                  <Clock className="w-5 h-5 text-blue-500" />
                  Upcoming Posts
                </h2>
                <button className="text-sm text-[#fe2c55] hover:underline">
                  Manage
                </button>
              </div>
              <UpcomingPosts posts={scheduledPosts || []} />
            </motion.div>

            {/* AI Insights */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
              className="glass rounded-2xl p-6 border border-[#fe2c55]/20"
            >
              <div className="flex items-center gap-2 mb-4">
                <Sparkles className="w-5 h-5 text-[#fe2c55]" />
                <h2 className="text-lg font-semibold text-white">AI Insight</h2>
              </div>
              <p className="text-gray-300 text-sm leading-relaxed">
                Based on your recent performance, posting at <span className="text-[#25f4ee] font-medium">7 PM</span> on weekdays 
                could increase your engagement by <span className="text-green-400 font-medium">+23%</span>. 
                Try incorporating trending sound &quot;Original Sound - Creator&quot; in your next video!
              </p>
              <button className="mt-4 w-full py-2 bg-[#fe2c55]/10 border border-[#fe2c55]/20 rounded-xl text-[#fe2c55] text-sm font-medium hover:bg-[#fe2c55]/20 transition">
                Generate Optimized Script
              </button>
            </motion.div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
