'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  TrendingUp, Hash, Music, Flame, Search, Filter,
  ArrowUpRight, ArrowDownRight, Clock, Eye, Users,
  Sparkles, Globe, ChevronDown, Play, Copy, Check,
  Info, Zap, Target, BarChart3
} from 'lucide-react';
import toast from 'react-hot-toast';
import { DEMO_MODE } from '@/lib/demo-mode';
import { api } from '@/lib/api';
import { DashboardLayout } from '@/components/layout/dashboard-layout';

interface TrendingHashtag {
  id: string;
  hashtag: string;
  view_count: number;
  video_count: number;
  trend_score: number;
  growth_rate: number;
  category?: string;
}

interface TrendingSound {
  id: string;
  sound_id: string;
  sound_title: string;
  sound_author: string;
  usage_count: number;
  trend_score: number;
  growth_rate: number;
  preview_url?: string;
}

interface ViralPattern {
  name: string;
  description: string;
  effectiveness: number;
  examples: string[];
  icon: string;
}

export default function TrendsPage() {
  const [activeTab, setActiveTab] = useState<'hashtags' | 'sounds' | 'patterns'>('hashtags');
  const [hashtags, setHashtags] = useState<TrendingHashtag[]>([]);
  const [sounds, setSounds] = useState<TrendingSound[]>([]);
  const [patterns, setPatterns] = useState<ViralPattern[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedRegion, setSelectedRegion] = useState('global');
  const [selectedNiche, setSelectedNiche] = useState('All');
  const [copiedTag, setCopiedTag] = useState<string | null>(null);

  const regions = [
    { value: 'global', label: 'Global', flag: '🌍' },
    { value: 'us', label: 'United States', flag: '🇺🇸' },
    { value: 'uk', label: 'United Kingdom', flag: '🇬🇧' },
    { value: 'ca', label: 'Canada', flag: '🇨🇦' },
    { value: 'fr', label: 'France', flag: '🇫🇷' },
    { value: 'de', label: 'Germany', flag: '🇩🇪' },
    { value: 'br', label: 'Brazil', flag: '🇧🇷' }
  ];

  const niches = [
    { value: 'All', color: 'gray' },
    { value: 'Entertainment', color: 'pink' },
    { value: 'Education', color: 'blue' },
    { value: 'Comedy', color: 'yellow' },
    { value: 'Lifestyle', color: 'purple' },
    { value: 'Tech', color: 'cyan' },
    { value: 'Fashion', color: 'rose' },
    { value: 'Food', color: 'orange' },
    { value: 'Fitness', color: 'green' },
    { value: 'Music', color: 'indigo' }
  ];


  const fetchTrends = useCallback(async () => {
    setIsLoading(true);
    try {
      // Use API (demo mode handled inside)
      const hashtagsData = await api.trends.getHashtags({ limit: 15, region: selectedRegion });
      const soundsData = await api.trends.getSounds({ limit: 10, region: selectedRegion });
      const patternsData = await api.trends.getViralPatterns(selectedNiche !== 'All' ? selectedNiche : undefined);

      // Enhance with demo data
      setHashtags([
        { id: '1', hashtag: '#fyp', view_count: 500000000000, video_count: 100000000, trend_score: 99, growth_rate: 0.5, category: 'General' },
        { id: '2', hashtag: '#viral', view_count: 200000000000, video_count: 50000000, trend_score: 98, growth_rate: 1.2, category: 'General' },
        { id: '3', hashtag: '#foryou', view_count: 180000000000, video_count: 45000000, trend_score: 97, growth_rate: 0.8, category: 'General' },
        { id: '4', hashtag: '#trending', view_count: 100000000000, video_count: 30000000, trend_score: 95, growth_rate: 2.1, category: 'General' },
        { id: '5', hashtag: '#lifehack', view_count: 50000000000, video_count: 15000000, trend_score: 92, growth_rate: 3.5, category: 'Lifestyle' },
        { id: '6', hashtag: '#tutorial', view_count: 40000000000, video_count: 12000000, trend_score: 90, growth_rate: 2.8, category: 'Education' },
        { id: '7', hashtag: '#comedy', view_count: 80000000000, video_count: 25000000, trend_score: 94, growth_rate: 1.5, category: 'Comedy' },
        { id: '8', hashtag: '#dance', view_count: 120000000000, video_count: 35000000, trend_score: 96, growth_rate: 0.9, category: 'Entertainment' },
        { id: '9', hashtag: '#fashion', view_count: 35000000000, video_count: 10000000, trend_score: 88, growth_rate: 2.2, category: 'Fashion' },
        { id: '10', hashtag: '#fitness', view_count: 30000000000, video_count: 9000000, trend_score: 86, growth_rate: 1.8, category: 'Fitness' },
        { id: '11', hashtag: '#cooking', view_count: 25000000000, video_count: 8000000, trend_score: 84, growth_rate: 2.5, category: 'Food' },
        { id: '12', hashtag: '#tech', view_count: 20000000000, video_count: 6000000, trend_score: 82, growth_rate: 4.2, category: 'Tech' },
        { id: '13', hashtag: '#motivation', view_count: 45000000000, video_count: 14000000, trend_score: 89, growth_rate: 1.3, category: 'Lifestyle' },
        { id: '14', hashtag: '#asmr', view_count: 60000000000, video_count: 18000000, trend_score: 91, growth_rate: 2.0, category: 'Entertainment' },
        { id: '15', hashtag: '#diy', view_count: 28000000000, video_count: 8500000, trend_score: 85, growth_rate: 1.7, category: 'Lifestyle' },
      ]);

      setSounds([
        { id: '1', sound_id: 'snd1', sound_title: 'Original Sound - Trending Beat', sound_author: '@viralcreator', usage_count: 2500000, trend_score: 98, growth_rate: 8.5 },
        { id: '2', sound_id: 'snd2', sound_title: 'Aesthetic Chill Vibes', sound_author: '@chillbeats', usage_count: 1800000, trend_score: 95, growth_rate: 6.2 },
        { id: '3', sound_id: 'snd3', sound_title: 'Comedy Sound Effect Pack', sound_author: '@funnysounds', usage_count: 1500000, trend_score: 93, growth_rate: 5.8 },
        { id: '4', sound_id: 'snd4', sound_title: 'Epic Transition Music', sound_author: '@musicmaker', usage_count: 1200000, trend_score: 91, growth_rate: 4.5 },
        { id: '5', sound_id: 'snd5', sound_title: 'Viral Dance Track 2024', sound_author: '@dancemusic', usage_count: 3200000, trend_score: 99, growth_rate: 12.3 },
        { id: '6', sound_id: 'snd6', sound_title: 'Motivational Speech Mix', sound_author: '@motivate', usage_count: 900000, trend_score: 88, growth_rate: 3.2 },
        { id: '7', sound_id: 'snd7', sound_title: 'Lo-Fi Study Beats', sound_author: '@lofibeats', usage_count: 750000, trend_score: 85, growth_rate: 2.8 },
        { id: '8', sound_id: 'snd8', sound_title: 'Cooking ASMR Sounds', sound_author: '@asmrcooking', usage_count: 650000, trend_score: 82, growth_rate: 4.1 },
      ]);

      setPatterns([
        {
          name: 'The Hook Pattern',
          description: 'Capture attention in the first 1-3 seconds with a surprising statement, question, or visual',
          effectiveness: 0.92,
          examples: ['POV: You just discovered...', 'Stop scrolling if you...', 'Nobody talks about this but...', 'Wait for the end...'],
          icon: '🎣'
        },
        {
          name: 'Pattern Interrupt',
          description: 'Change visuals, audio, or pace every 2-4 seconds to maintain viewer attention',
          effectiveness: 0.85,
          examples: ['Quick jump cuts', 'Text pop-ups', 'Sound effects', 'Camera angle changes'],
          icon: '⚡'
        },
        {
          name: 'Storytelling Arc',
          description: 'Create a mini narrative with setup, conflict, and resolution in under 60 seconds',
          effectiveness: 0.88,
          examples: ['Before/After transformation', 'Day in my life', 'How I went from X to Y', 'Storytime: ...'],
          icon: '📖'
        },
        {
          name: 'Educational Value',
          description: 'Teach something valuable quickly - viewers save and share educational content more',
          effectiveness: 0.90,
          examples: ['3 things I wish I knew...', 'Quick tip:', 'Did you know...', 'Life hack:'],
          icon: '🎓'
        },
        {
          name: 'Duet/Stitch Bait',
          description: 'Create content that encourages others to react, respond, or add to your video',
          effectiveness: 0.78,
          examples: ['Hot takes', 'Unpopular opinions', 'Challenges', 'Open-ended questions'],
          icon: '🔗'
        },
        {
          name: 'Trend Jacking',
          description: 'Put your unique spin on trending sounds, formats, or challenges',
          effectiveness: 0.82,
          examples: ['Use trending sound + your niche', 'Adapt viral format', 'Add expertise to trend'],
          icon: '🏄'
        }
      ]);
    } catch (error) {
      console.error('Failed to fetch trends:', error);
    } finally {
      setIsLoading(false);
    }
  }, [selectedRegion, selectedNiche]);
useEffect(() => {    fetchTrends();  }, [fetchTrends]);

  const formatNumber = (num: number) => {
    if (num >= 1000000000000) return (num / 1000000000000).toFixed(1) + 'T';
    if (num >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
  };

  const copyHashtag = (tag: string) => {
    navigator.clipboard.writeText(tag);
    setCopiedTag(tag);
    toast.success('Copied to clipboard!');
    setTimeout(() => setCopiedTag(null), 2000);
  };

  const copyAllHashtags = () => {
    const tags = filteredHashtags.slice(0, 10).map(h => h.hashtag).join(' ');
    navigator.clipboard.writeText(tags);
    toast.success('Top 10 hashtags copied!');
  };

  const filteredHashtags = hashtags.filter(h =>
    h.hashtag.toLowerCase().includes(searchQuery.toLowerCase()) &&
    (selectedNiche === 'All' || h.category === selectedNiche)
  ).sort((a, b) => b.trend_score - a.trend_score);

  const filteredSounds = sounds.filter(s =>
    s.sound_title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    s.sound_author.toLowerCase().includes(searchQuery.toLowerCase())
  ).sort((a, b) => b.trend_score - a.trend_score);

  const tabs = [
    { id: 'hashtags', label: 'Hashtags', icon: Hash, count: filteredHashtags.length },
    { id: 'sounds', label: 'Sounds', icon: Music, count: filteredSounds.length },
    { id: 'patterns', label: 'Viral Patterns', icon: Flame, count: patterns.length }
  ];

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Demo Mode Banner */}
        {DEMO_MODE && (
          <div className="p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-xl flex items-center gap-2">
            <Info className="w-5 h-5 text-yellow-400 flex-shrink-0" />
            <p className="text-sm text-yellow-200">
              <strong>Demo Mode:</strong> Showing simulated trending data.
            </p>
          </div>
        )}

        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center">
              <TrendingUp className="mr-3 h-8 w-8 text-purple-400" />
              Trends Explorer
            </h1>
            <p className="mt-1 text-gray-400">Discover what&apos;s viral on TikTok, Instagram & YouTube</p>
          </div>

          {/* Quick Stats */}
          <div className="flex items-center gap-4 mt-4 md:mt-0">
            <div className="px-4 py-2 bg-gray-800 rounded-xl border border-gray-700">
              <div className="text-xs text-gray-400">Updated</div>
              <div className="text-sm font-medium text-white flex items-center gap-1">
                <Clock className="w-3 h-3" />
                Just now
              </div>
            </div>
          </div>
        </div>

        {/* Filters Row */}
        <div className="flex flex-col lg:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search hashtags, sounds..."
              className="w-full pl-12 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>

          {/* Region Selector */}
          <div className="relative min-w-[180px]">
            <Globe className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <select
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              className="w-full pl-12 pr-10 py-3 bg-gray-800 border border-gray-700 rounded-xl text-white appearance-none focus:outline-none focus:ring-2 focus:ring-purple-500 cursor-pointer"
            >
              {regions.map((r) => (
                <option key={r.value} value={r.value}>{r.flag} {r.label}</option>
              ))}
            </select>
            <ChevronDown className="absolute right-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400 pointer-events-none" />
          </div>

          {/* Niche Filter Pills */}
          <div className="flex gap-2 overflow-x-auto pb-2 lg:pb-0">
            {niches.slice(0, 6).map((n) => (
              <button
                key={n.value}
                onClick={() => setSelectedNiche(n.value)}
                className={`px-4 py-2 rounded-xl text-sm font-medium whitespace-nowrap transition ${
                  selectedNiche === n.value
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                {n.value}
              </button>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 border-b border-gray-700">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center px-5 py-3 font-medium transition border-b-2 -mb-px ${
                activeTab === tab.id
                  ? 'border-purple-500 text-white'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              <tab.icon className="h-5 w-5 mr-2" />
              {tab.label}
              <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                activeTab === tab.id ? 'bg-purple-600' : 'bg-gray-700'
              }`}>
                {tab.count}
              </span>
            </button>
          ))}
        </div>

        {/* Content */}
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <div className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : (
          <>
            {/* Hashtags Tab */}
            {activeTab === 'hashtags' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-4"
              >
                {/* Copy All Button */}
                <div className="flex justify-end">
                  <button
                    onClick={copyAllHashtags}
                    className="px-4 py-2 bg-purple-600/20 text-purple-400 rounded-lg hover:bg-purple-600/30 transition flex items-center gap-2 text-sm"
                  >
                    <Copy className="w-4 h-4" />
                    Copy Top 10
                  </button>
                </div>

                <div className="grid gap-3">
                  {filteredHashtags.map((hashtag, index) => (
                    <motion.div
                      key={hashtag.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.03 }}
                      className="bg-gray-800 rounded-xl p-4 border border-gray-700 hover:border-purple-500/50 transition group"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          {/* Rank */}
                          <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-lg font-bold ${
                            index < 3 ? 'bg-gradient-to-r from-yellow-500 to-orange-500 text-white' : 'bg-gray-700 text-gray-400'
                          }`}>
                            {index + 1}
                          </div>

                          {/* Hashtag Info */}
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-lg font-semibold text-white">{hashtag.hashtag}</span>
                              {hashtag.category && (
                                <span className="px-2 py-0.5 bg-gray-700 rounded text-xs text-gray-400">
                                  {hashtag.category}
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-4 mt-1 text-sm text-gray-400">
                              <span className="flex items-center gap-1">
                                <Eye className="h-3.5 w-3.5" />
                                {formatNumber(hashtag.view_count)} views
                              </span>
                              <span className="flex items-center gap-1">
                                <Users className="h-3.5 w-3.5" />
                                {formatNumber(hashtag.video_count)} videos
                              </span>
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center gap-4">
                          {/* Trend Score */}
                          <div className="text-right hidden sm:block">
                            <div className="text-xs text-gray-400">Score</div>
                            <div className="flex items-center gap-1">
                              <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                                  style={{ width: `${hashtag.trend_score}%` }}
                                />
                              </div>
                              <span className="text-white font-medium text-sm">{hashtag.trend_score}</span>
                            </div>
                          </div>

                          {/* Growth */}
                          <div className={`flex items-center gap-1 ${
                            hashtag.growth_rate >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {hashtag.growth_rate >= 0 ? (
                              <ArrowUpRight className="h-4 w-4" />
                            ) : (
                              <ArrowDownRight className="h-4 w-4" />
                            )}
                            <span className="font-medium text-sm">{Math.abs(hashtag.growth_rate).toFixed(1)}%</span>
                          </div>

                          {/* Copy Button */}
                          <button
                            onClick={() => copyHashtag(hashtag.hashtag)}
                            className={`p-2 rounded-lg transition ${
                              copiedTag === hashtag.hashtag
                                ? 'bg-green-500/20 text-green-400'
                                : 'bg-gray-700 text-gray-400 hover:text-white group-hover:bg-purple-600 group-hover:text-white'
                            }`}
                          >
                            {copiedTag === hashtag.hashtag ? (
                              <Check className="h-4 w-4" />
                            ) : (
                              <Copy className="h-4 w-4" />
                            )}
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Sounds Tab */}
            {activeTab === 'sounds' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="grid gap-4"
              >
                {filteredSounds.map((sound, index) => (
                  <motion.div
                    key={sound.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="bg-gray-800 rounded-xl p-5 border border-gray-700 hover:border-pink-500/50 transition"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        {/* Play Button */}
                        <div className="w-14 h-14 bg-gradient-to-r from-pink-500 to-purple-500 rounded-xl flex items-center justify-center cursor-pointer hover:scale-105 transition">
                          <Play className="h-6 w-6 text-white ml-1" fill="white" />
                        </div>

                        {/* Sound Info */}
                        <div>
                          <div className="font-semibold text-white text-lg">{sound.sound_title}</div>
                          <div className="text-sm text-gray-400">{sound.sound_author}</div>
                          <div className="flex items-center gap-4 mt-1 text-sm text-gray-400">
                            <span className="flex items-center gap-1">
                              <Users className="h-3.5 w-3.5" />
                              {formatNumber(sound.usage_count)} videos using this
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-6">
                        {/* Trend Score */}
                        <div className="text-right hidden sm:block">
                          <div className="text-xs text-gray-400">Trend Score</div>
                          <div className="text-2xl font-bold text-white">{sound.trend_score}</div>
                        </div>

                        {/* Growth */}
                        <div className="text-right">
                          <div className={`flex items-center gap-1 ${
                            sound.growth_rate >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {sound.growth_rate >= 0 ? (
                              <ArrowUpRight className="h-5 w-5" />
                            ) : (
                              <ArrowDownRight className="h-5 w-5" />
                            )}
                            <span className="font-bold text-lg">{Math.abs(sound.growth_rate).toFixed(1)}%</span>
                          </div>
                          <div className="text-xs text-gray-400">this week</div>
                        </div>

                        {/* Use Button */}
                        <Link
                          href="/dashboard/create"
                          className="px-5 py-2.5 bg-gradient-to-r from-pink-500 to-purple-500 hover:from-pink-600 hover:to-purple-600 text-white text-sm font-medium rounded-xl transition flex items-center gap-2"
                        >
                          <Zap className="w-4 h-4" />
                          Use Sound
                        </Link>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            )}

            {/* Patterns Tab */}
            {activeTab === 'patterns' && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
              >
                {patterns.map((pattern, index) => (
                  <motion.div
                    key={pattern.name}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-gray-800 rounded-xl p-6 border border-gray-700 hover:border-orange-500/50 transition group"
                  >
                    {/* Header */}
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-red-500 rounded-xl flex items-center justify-center text-2xl">
                          {pattern.icon}
                        </div>
                        <div>
                          <h3 className="font-semibold text-white">{pattern.name}</h3>
                          <div className="flex items-center gap-1 text-sm">
                            <BarChart3 className="w-3 h-3 text-green-400" />
                            <span className="text-green-400 font-medium">
                              {(pattern.effectiveness * 100).toFixed(0)}% effective
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Description */}
                    <p className="text-gray-400 text-sm mb-4 leading-relaxed">
                      {pattern.description}
                    </p>

                    {/* Examples */}
                    <div className="mb-4">
                      <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Examples</div>
                      <div className="flex flex-wrap gap-2">
                        {pattern.examples.slice(0, 3).map((example, idx) => (
                          <span
                            key={idx}
                            className="px-2.5 py-1 bg-gray-700/50 text-gray-300 rounded-lg text-xs"
                          >
                            {example}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Action Button */}
                    <Link
                      href="/dashboard/create"
                      className="w-full py-2.5 bg-orange-500/20 hover:bg-orange-500/30 text-orange-400 rounded-xl transition flex items-center justify-center gap-2 text-sm font-medium group-hover:bg-orange-500 group-hover:text-white"
                    >
                      <Sparkles className="h-4 w-4" />
                      Use This Pattern
                    </Link>
                  </motion.div>
                ))}
              </motion.div>
            )}
          </>
        )}
      </div>
    </DashboardLayout>
  );
}
