'use client';
/* eslint-disable @next/next/no-img-element */

import { useState, useEffect, useCallback } from 'react';
import Image from 'next/image';
import { motion } from 'framer-motion';
import {
  BarChart3, TrendingUp, TrendingDown, Eye, Heart, MessageCircle,
  Share2, Clock, Target, Lightbulb, Users, Play, Info,
  ArrowUpRight, ArrowDownRight, Instagram, Youtube
} from 'lucide-react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts';
import { DEMO_MODE, DEMO_STATS } from '@/lib/demo-mode';
import { DashboardLayout } from '@/components/layout/dashboard-layout';

interface PlatformStats {
  platform: string;
  followers: number;
  views: number;
  engagement: number;
  growth: number;
  color: string;
  icon: any;
}

export default function AnalyticsPage() {
  const [period, setPeriod] = useState('7d');
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'platforms' | 'content'>('overview');

  const periods = [
    { value: '24h', label: '24h' },
    { value: '7d', label: '7 Days' },
    { value: '30d', label: '30 Days' },
    { value: '90d', label: '90 Days' }
  ];

  // Generate realistic demo data based on period
  const getDaysCount = useCallback(() => {
    switch (period) {
      case '24h': return 24;
      case '7d': return 7;
      case '30d': return 30;
      case '90d': return 90;
      default: return 7;
    }
  }, [period]);

  const generateViewsData = useCallback(() => {
    const days = getDaysCount();
    const data = [];
    const baseViews = DEMO_STATS.avgViewsPerPost;

    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      if (period === '24h') {
        date.setHours(date.getHours() - i);
      } else {
        date.setDate(date.getDate() - i);
      }

      const variation = 0.5 + Math.random();
      const dayOfWeek = date.getDay();
      const weekendBoost = (dayOfWeek === 0 || dayOfWeek === 6) ? 1.3 : 1;

      data.push({
        date: period === '24h'
          ? date.toLocaleTimeString('en-US', { hour: '2-digit' })
          : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        views: Math.floor(baseViews * variation * weekendBoost / (period === '24h' ? 24 : 1)),
        likes: Math.floor(baseViews * variation * 0.07 * weekendBoost / (period === '24h' ? 24 : 1)),
        comments: Math.floor(baseViews * variation * 0.01 * weekendBoost / (period === '24h' ? 24 : 1)),
      });
    }
    return data;
  }, [period, getDaysCount]);

  const generateEngagementData = useCallback(() => {
    const days = getDaysCount();
    const data = [];

    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      if (period === '24h') {
        date.setHours(date.getHours() - i);
      } else {
        date.setDate(date.getDate() - i);
      }

      data.push({
        date: period === '24h'
          ? date.toLocaleTimeString('en-US', { hour: '2-digit' })
          : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        rate: (5 + Math.random() * 5).toFixed(1),
      });
    }
    return data;
  }, [period, getDaysCount]);

  const [viewsData, setViewsData] = useState(generateViewsData());
  const [engagementData, setEngagementData] = useState(generateEngagementData());

  useEffect(() => {
    setIsLoading(true);
    // Simulate loading
    setTimeout(() => {
      setViewsData(generateViewsData());
      setEngagementData(generateEngagementData());
      setIsLoading(false);
    }, 500);
  }, [period, generateViewsData, generateEngagementData]);

  const formatNumber = (num: number) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toLocaleString();
  };

  const metrics = [
    {
      label: 'Total Views',
      value: formatNumber(DEMO_STATS.totalViews),
      change: 23.5,
      icon: Eye,
      color: 'from-blue-500 to-cyan-500',
      bgColor: 'bg-blue-500/20'
    },
    {
      label: 'Total Likes',
      value: formatNumber(DEMO_STATS.totalLikes),
      change: 15.2,
      icon: Heart,
      color: 'from-pink-500 to-rose-500',
      bgColor: 'bg-pink-500/20'
    },
    {
      label: 'Comments',
      value: formatNumber(DEMO_STATS.totalComments),
      change: 8.7,
      icon: MessageCircle,
      color: 'from-purple-500 to-indigo-500',
      bgColor: 'bg-purple-500/20'
    },
    {
      label: 'Shares',
      value: formatNumber(DEMO_STATS.totalShares),
      change: 12.1,
      icon: Share2,
      color: 'from-green-500 to-emerald-500',
      bgColor: 'bg-green-500/20'
    }
  ];

  const platformStats: PlatformStats[] = [
    {
      platform: 'TikTok',
      followers: 125000,
      views: 850000,
      engagement: 8.2,
      growth: 12.5,
      color: '#000000',
      icon: () => (
        <svg className="w-6 h-6" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5 20.1a6.34 6.34 0 0 0 10.86-4.43v-7a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1-.1z"/>
        </svg>
      )
    },
    {
      platform: 'Instagram',
      followers: 89000,
      views: 420000,
      engagement: 6.8,
      growth: 8.3,
      color: '#E1306C',
      icon: Instagram
    },
    {
      platform: 'YouTube',
      followers: 45000,
      views: 280000,
      engagement: 5.4,
      growth: 15.2,
      color: '#FF0000',
      icon: Youtube
    }
  ];

  const contentPerformance = [
    { name: 'Educational', value: 35, color: '#8b5cf6' },
    { name: 'Entertainment', value: 28, color: '#ec4899' },
    { name: 'Trending', value: 22, color: '#06b6d4' },
    { name: 'Personal', value: 15, color: '#10b981' }
  ];

  const topPosts = [
    { title: '5 Productivity Hacks', views: 156000, likes: 12400, engagement: 9.2, thumbnail: 'https://picsum.photos/seed/post1/100/150' },
    { title: 'Morning Routine', views: 98000, likes: 7800, engagement: 8.5, thumbnail: 'https://picsum.photos/seed/post2/100/150' },
    { title: 'Viral Dance Tutorial', views: 245000, likes: 21000, engagement: 11.2, thumbnail: 'https://picsum.photos/seed/post3/100/150' },
    { title: 'Quick Recipe', views: 67000, likes: 5200, engagement: 7.8, thumbnail: 'https://picsum.photos/seed/post4/100/150' },
  ];

  const bestPostingTimes = [
    { time: '7 AM', engagement: 45 },
    { time: '9 AM', engagement: 62 },
    { time: '12 PM', engagement: 78 },
    { time: '3 PM', engagement: 55 },
    { time: '6 PM', engagement: 85 },
    { time: '9 PM', engagement: 95 },
    { time: '11 PM', engagement: 72 },
  ];

  const insights = [
    {
      icon: Clock,
      title: 'Best Posting Time',
      description: 'Your audience is most active between 7 PM - 10 PM',
      action: 'Schedule posts for peak hours',
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-500/20'
    },
    {
      icon: Target,
      title: 'Top Performing Content',
      description: 'Educational content gets 40% more engagement',
      action: 'Create more tutorials and tips',
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/20'
    },
    {
      icon: TrendingUp,
      title: 'Growth Opportunity',
      description: 'Trending sounds boost views by 2.5x',
      action: 'Use trending audio in next video',
      color: 'text-green-400',
      bgColor: 'bg-green-500/20'
    }
  ];

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Demo Mode Banner */}
        {DEMO_MODE && (
          <div className="p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-xl flex items-center gap-2">
            <Info className="w-5 h-5 text-yellow-400 flex-shrink-0" />
            <p className="text-sm text-yellow-200">
              <strong>Demo Mode:</strong> Showing simulated analytics data.
            </p>
          </div>
        )}

        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center">
              <BarChart3 className="mr-3 h-8 w-8 text-purple-400" />
              Analytics
            </h1>
            <p className="mt-1 text-gray-400">Track your content performance across all platforms</p>
          </div>

          {/* Period Selector */}
          <div className="mt-4 md:mt-0 flex bg-gray-800 rounded-lg p-1">
            {periods.map((p) => (
              <button
                key={p.value}
                onClick={() => setPeriod(p.value)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition ${
                  period === p.value
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 border-b border-gray-700 pb-2">
          {[
            { id: 'overview', label: 'Overview' },
            { id: 'platforms', label: 'Platforms' },
            { id: 'content', label: 'Content' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-2 rounded-t-lg text-sm font-medium transition ${
                activeTab === tab.id
                  ? 'bg-gray-800 text-white border-b-2 border-purple-500'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-6"
          >
            {/* Metrics Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {metrics.map((metric, index) => (
                <motion.div
                  key={metric.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-800 rounded-xl p-5 border border-gray-700"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className={`p-2.5 rounded-xl ${metric.bgColor}`}>
                      <metric.icon className="h-5 w-5 text-white" />
                    </div>
                    <div className={`flex items-center text-sm ${
                      metric.change >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {metric.change >= 0 ? (
                        <ArrowUpRight className="h-4 w-4" />
                      ) : (
                        <ArrowDownRight className="h-4 w-4" />
                      )}
                      {Math.abs(metric.change)}%
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-white">{metric.value}</div>
                  <div className="text-sm text-gray-400 mt-1">{metric.label}</div>
                </motion.div>
              ))}
            </div>

            {/* Charts Row */}
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Views Chart */}
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-white">Views & Engagement</h3>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-full bg-purple-500"></span>
                      Views
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-full bg-pink-500"></span>
                      Likes
                    </span>
                  </div>
                </div>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={viewsData}>
                      <defs>
                        <linearGradient id="viewsGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="likesGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ec4899" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#ec4899" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="date" stroke="#9ca3af" fontSize={11} />
                      <YAxis stroke="#9ca3af" fontSize={11} tickFormatter={(v) => formatNumber(v)} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                        labelStyle={{ color: '#fff' }}
                        formatter={(value: number) => formatNumber(value)}
                      />
                      <Area type="monotone" dataKey="views" stroke="#8b5cf6" strokeWidth={2} fill="url(#viewsGradient)" />
                      <Area type="monotone" dataKey="likes" stroke="#ec4899" strokeWidth={2} fill="url(#likesGradient)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Best Posting Times */}
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">Best Posting Times</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={bestPostingTimes}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="time" stroke="#9ca3af" fontSize={11} />
                      <YAxis stroke="#9ca3af" fontSize={11} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                        labelStyle={{ color: '#fff' }}
                        formatter={(value: number) => [`${value}%`, 'Engagement']}
                      />
                      <Bar dataKey="engagement" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* AI Insights */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                <Lightbulb className="mr-2 h-5 w-5 text-yellow-400" />
                AI-Powered Insights
              </h3>
              <div className="grid md:grid-cols-3 gap-4">
                {insights.map((insight, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="p-4 bg-gray-700/50 rounded-xl"
                  >
                    <div className={`p-2 ${insight.bgColor} rounded-lg w-fit mb-3`}>
                      <insight.icon className={`h-5 w-5 ${insight.color}`} />
                    </div>
                    <h4 className="font-medium text-white">{insight.title}</h4>
                    <p className="text-sm text-gray-400 mt-1">{insight.description}</p>
                    <button className="mt-3 text-sm text-purple-400 hover:text-purple-300">
                      {insight.action} →
                    </button>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* Platforms Tab */}
        {activeTab === 'platforms' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-6"
          >
            {/* Platform Cards */}
            <div className="grid md:grid-cols-3 gap-6">
              {platformStats.map((platform, index) => (
                <motion.div
                  key={platform.platform}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-gray-800 rounded-xl p-6 border border-gray-700"
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div
                        className="p-3 rounded-xl"
                        style={{ backgroundColor: `${platform.color}20` }}
                      >
                        <platform.icon className="h-6 w-6" style={{ color: platform.color === '#000000' ? '#fff' : platform.color }} />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">{platform.platform}</h3>
                        <p className="text-sm text-gray-400">{formatNumber(platform.followers)} followers</p>
                      </div>
                    </div>
                    <div className={`flex items-center text-sm ${platform.growth >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      <ArrowUpRight className="h-4 w-4" />
                      {platform.growth}%
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className="bg-gray-700/50 rounded-lg p-3">
                      <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
                        <Eye className="h-4 w-4" />
                        Views
                      </div>
                      <p className="text-xl font-bold text-white">{formatNumber(platform.views)}</p>
                    </div>
                    <div className="bg-gray-700/50 rounded-lg p-3">
                      <div className="flex items-center gap-2 text-gray-400 text-sm mb-1">
                        <TrendingUp className="h-4 w-4" />
                        Engagement
                      </div>
                      <p className="text-xl font-bold text-white">{platform.engagement}%</p>
                    </div>
                  </div>

                  <div className="mt-4 pt-4 border-t border-gray-700">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Growth this month</span>
                      <span className="text-green-400">+{formatNumber(Math.floor(platform.followers * platform.growth / 100))}</span>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Platform Comparison Chart */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Platform Performance Comparison</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={platformStats.map(p => ({
                      name: p.platform,
                      views: p.views / 1000,
                      engagement: p.engagement * 10000,
                      followers: p.followers / 100
                    }))}
                    layout="vertical"
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" stroke="#9ca3af" fontSize={11} />
                    <YAxis dataKey="name" type="category" stroke="#9ca3af" fontSize={11} width={80} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                      labelStyle={{ color: '#fff' }}
                    />
                    <Bar dataKey="views" name="Views (K)" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </motion.div>
        )}

        {/* Content Tab */}
        {activeTab === 'content' && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-6"
          >
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Content Performance Pie */}
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">Content Categories</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={contentPerformance}
                        cx="50%"
                        cy="50%"
                        innerRadius={50}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {contentPerformance.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                        formatter={(value: number) => [`${value}%`, 'Share']}
                      />
                      <Legend
                        verticalAlign="bottom"
                        formatter={(value) => <span className="text-gray-300 text-sm">{value}</span>}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Top Posts */}
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold text-white mb-4">Top Performing Posts</h3>
                <div className="space-y-3">
                  {topPosts.map((post, index) => (
                    <div
                      key={index}
                      className="flex items-center gap-4 p-3 bg-gray-700/50 rounded-xl"
                    >
                      <div className="w-12 h-16 rounded-lg overflow-hidden bg-gray-600 flex-shrink-0">
                        <img src={post.thumbnail} alt="" className="w-full h-full object-cover" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="font-medium text-white truncate">{post.title}</h4>
                        <div className="flex items-center gap-4 mt-1 text-sm text-gray-400">
                          <span className="flex items-center gap-1">
                            <Eye className="h-3 w-3" />
                            {formatNumber(post.views)}
                          </span>
                          <span className="flex items-center gap-1">
                            <Heart className="h-3 w-3" />
                            {formatNumber(post.likes)}
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="text-green-400 text-sm font-medium">{post.engagement}%</span>
                        <p className="text-xs text-gray-500">engagement</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Engagement Rate Over Time */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Engagement Rate Trend</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={engagementData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="date" stroke="#9ca3af" fontSize={11} />
                    <YAxis stroke="#9ca3af" fontSize={11} domain={[0, 12]} tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                      labelStyle={{ color: '#fff' }}
                      formatter={(value: number) => [`${value}%`, 'Engagement Rate']}
                    />
                    <Line
                      type="monotone"
                      dataKey="rate"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={{ fill: '#10b981', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </DashboardLayout>
  );
}
