'use client';
/* eslint-disable @next/next/no-img-element */

import { useState, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { motion } from 'framer-motion';
import {
  Sparkles, TrendingUp, Calendar, BarChart3,
  Zap, Brain, Clock, Shield, ChevronRight, ChevronDown,
  Play, Star, Users, Video, Check, ArrowRight,
  MessageSquare, Target, Rocket, Globe, Award,
  Instagram, Youtube, Menu, X
} from 'lucide-react';

export default function HomePage() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  useEffect(() => {
    const token = localStorage.getItem('accessToken');
    setIsLoggedIn(!!token);
  }, []);

  const features = [
    {
      icon: Brain,
      title: 'AI Script Generation',
      description: 'Generate viral scripts with hooks that stop the scroll. Our AI analyzes millions of videos to craft content that converts.',
      color: 'from-purple-500 to-indigo-500'
    },
    {
      icon: TrendingUp,
      title: 'Trend Detection',
      description: 'Stay ahead with real-time trend analysis. Discover viral sounds, hashtags, and patterns before they peak.',
      color: 'from-pink-500 to-rose-500'
    },
    {
      icon: Calendar,
      title: 'Smart Scheduling',
      description: 'Publish at optimal times across TikTok, Instagram Reels, and YouTube Shorts. One upload, three platforms.',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: BarChart3,
      title: 'Analytics Dashboard',
      description: 'Track performance with detailed insights. Understand what works and optimize your content strategy.',
      color: 'from-green-500 to-emerald-500'
    },
    {
      icon: Target,
      title: 'Content Optimization',
      description: 'AI-powered suggestions for captions, hashtags, and posting times to maximize your reach and engagement.',
      color: 'from-orange-500 to-amber-500'
    },
    {
      icon: MessageSquare,
      title: 'AI Chat Assistants',
      description: 'Chat with specialized AI agents for trend analysis, script writing, content optimization, and growth strategy.',
      color: 'from-violet-500 to-purple-500'
    }
  ];

  const stats = [
    { value: '10M+', label: 'Videos Analyzed' },
    { value: '500K+', label: 'Scripts Generated' },
    { value: '2.5M+', label: 'Posts Scheduled' },
    { value: '98%', label: 'User Satisfaction' }
  ];

  const howItWorks = [
    {
      step: '01',
      title: 'Connect Your Accounts',
      description: 'Link your TikTok, Instagram, and YouTube accounts in seconds with secure OAuth.',
      icon: Globe
    },
    {
      step: '02',
      title: 'Create with AI',
      description: 'Generate scripts, hooks, and captions using our AI agents. Upload your video or let AI guide you.',
      icon: Sparkles
    },
    {
      step: '03',
      title: 'Schedule & Publish',
      description: 'Pick your platforms, set the time, and let Viralify handle the rest. One click, multiple platforms.',
      icon: Rocket
    }
  ];

  const aiAgents = [
    {
      name: 'TrendScout',
      role: 'Trend Analyzer',
      description: 'Identifies emerging trends before they peak',
      avatar: '🔍',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      name: 'ScriptGenius',
      role: 'Script Writer',
      description: 'Creates viral hooks and engaging scripts',
      avatar: '✍️',
      color: 'from-purple-500 to-pink-500'
    },
    {
      name: 'ContentOptimizer',
      role: 'Content Optimizer',
      description: 'Maximizes reach and engagement',
      avatar: '🎯',
      color: 'from-orange-500 to-red-500'
    },
    {
      name: 'StrategyAdvisor',
      role: 'Growth Strategist',
      description: 'Develops growth strategies',
      avatar: '📈',
      color: 'from-green-500 to-emerald-500'
    }
  ];

  const pricing = [
    {
      name: 'Starter',
      price: 'Free',
      description: 'Perfect for getting started',
      features: [
        '3 scheduled posts/month',
        '1 connected account',
        'Basic analytics',
        'AI script generation (5/month)',
        'Community support'
      ],
      cta: 'Get Started',
      popular: false
    },
    {
      name: 'Pro',
      price: '$19',
      period: '/month',
      description: 'For serious creators',
      features: [
        'Unlimited scheduled posts',
        '3 connected accounts',
        'Advanced analytics',
        'Unlimited AI generation',
        'Trend alerts',
        'Priority support',
        'Content calendar'
      ],
      cta: 'Start Free Trial',
      popular: true
    },
    {
      name: 'Business',
      price: '$49',
      period: '/month',
      description: 'For teams and agencies',
      features: [
        'Everything in Pro',
        '10 connected accounts',
        'Team collaboration',
        'White-label reports',
        'API access',
        'Dedicated account manager',
        'Custom integrations'
      ],
      cta: 'Contact Sales',
      popular: false
    }
  ];

  const testimonials = [
    {
      name: 'Sarah Chen',
      role: 'Content Creator',
      avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=sarah',
      content: 'Viralify helped me go from 10K to 500K followers in 3 months. The AI scripts are incredibly engaging!',
      followers: '500K',
      platform: 'TikTok'
    },
    {
      name: 'Marcus Johnson',
      role: 'Digital Marketer',
      avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=marcus',
      content: 'Managing multiple clients across platforms was a nightmare. Viralify makes it effortless. Game changer!',
      followers: '1.2M',
      platform: 'Multi-platform'
    },
    {
      name: 'Emma Rodriguez',
      role: 'Lifestyle Influencer',
      avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=emma',
      content: 'The trend detection feature is incredible. I\'m always ahead of what\'s about to go viral.',
      followers: '750K',
      platform: 'Instagram'
    }
  ];

  const faqs = [
    {
      question: 'How does the multi-platform publishing work?',
      answer: 'Simply upload your video once, and Viralify automatically adapts and publishes it to TikTok, Instagram Reels, and YouTube Shorts. We handle format adjustments, hashtag optimization, and timing for each platform.'
    },
    {
      question: 'Is my content safe and secure?',
      answer: 'Absolutely. We use OAuth for secure authentication and never store your social media passwords. Your content is encrypted and you maintain full ownership of everything you create.'
    },
    {
      question: 'Can I cancel my subscription anytime?',
      answer: 'Yes! You can cancel your subscription at any time with no questions asked. Your access continues until the end of your billing period.'
    },
    {
      question: 'How accurate is the AI script generation?',
      answer: 'Our AI is trained on millions of viral videos and continuously learns from current trends. While no AI is perfect, our scripts consistently achieve 3-5x higher engagement than average.'
    },
    {
      question: 'Do you offer a free trial?',
      answer: 'Yes! Our Starter plan is completely free and includes core features. Pro plan also comes with a 14-day free trial so you can experience all premium features risk-free.'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-gray-950/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#fe2c55] to-[#25f4ee] flex items-center justify-center">
                <Video className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-white">Viralify</span>
            </Link>

            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-gray-400 hover:text-white transition text-sm">
                Features
              </a>
              <a href="#how-it-works" className="text-gray-400 hover:text-white transition text-sm">
                How it Works
              </a>
              <a href="#pricing" className="text-gray-400 hover:text-white transition text-sm">
                Pricing
              </a>
              <a href="#faq" className="text-gray-400 hover:text-white transition text-sm">
                FAQ
              </a>
            </div>

            <div className="hidden md:flex items-center space-x-4">
              {isLoggedIn ? (
                <Link
                  href="/dashboard"
                  className="px-4 py-2 bg-gradient-to-r from-[#fe2c55] to-[#25f4ee] text-white font-medium rounded-lg transition hover:opacity-90"
                >
                  Dashboard
                </Link>
              ) : (
                <>
                  <Link href="/auth/login" className="text-gray-400 hover:text-white transition text-sm">
                    Sign In
                  </Link>
                  <Link
                    href="/auth/register"
                    className="px-4 py-2 bg-gradient-to-r from-[#fe2c55] to-[#25f4ee] text-white font-medium rounded-lg transition hover:opacity-90"
                  >
                    Get Started Free
                  </Link>
                </>
              )}
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 text-gray-400 hover:text-white"
            >
              {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="md:hidden bg-gray-900 border-t border-white/5"
          >
            <div className="px-4 py-4 space-y-3">
              <a href="#features" className="block text-gray-300 hover:text-white py-2">Features</a>
              <a href="#how-it-works" className="block text-gray-300 hover:text-white py-2">How it Works</a>
              <a href="#pricing" className="block text-gray-300 hover:text-white py-2">Pricing</a>
              <a href="#faq" className="block text-gray-300 hover:text-white py-2">FAQ</a>
              <div className="pt-4 border-t border-white/10 space-y-3">
                <Link href="/auth/login" className="block text-gray-300 hover:text-white py-2">Sign In</Link>
                <Link
                  href="/auth/register"
                  className="block w-full text-center px-4 py-2 bg-gradient-to-r from-[#fe2c55] to-[#25f4ee] text-white font-medium rounded-lg"
                >
                  Get Started Free
                </Link>
              </div>
            </div>
          </motion.div>
        )}
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
        {/* Background effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-r from-[#fe2c55]/10 to-[#25f4ee]/10 rounded-full blur-3xl" />
        </div>

        <div className="max-w-7xl mx-auto relative">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            {/* Badge */}
            <div className="inline-flex items-center px-4 py-2 bg-white/5 border border-white/10 rounded-full text-sm mb-8">
              <Zap className="h-4 w-4 mr-2 text-yellow-400" />
              <span className="text-gray-300">Trusted by 50,000+ creators worldwide</span>
            </div>

            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
              Go Viral on
              <span className="block mt-2">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#fe2c55] via-purple-500 to-[#25f4ee]">
                  Every Platform
                </span>
              </span>
            </h1>

            <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-10">
              The AI-powered platform that helps you create, schedule, and optimize content for
              TikTok, Instagram Reels, and YouTube Shorts — all in one place.
            </p>

            {/* Platform logos */}
            <div className="flex items-center justify-center gap-6 mb-10">
              <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full">
                <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5 20.1a6.34 6.34 0 0 0 10.86-4.43v-7a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1-.1z"/>
                </svg>
                <span className="text-white text-sm">TikTok</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full">
                <Instagram className="w-5 h-5 text-white" />
                <span className="text-white text-sm">Reels</span>
              </div>
              <div className="flex items-center gap-2 px-4 py-2 bg-white/5 rounded-full">
                <Youtube className="w-5 h-5 text-white" />
                <span className="text-white text-sm">Shorts</span>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href={isLoggedIn ? "/dashboard" : "/auth/register"}
                className="w-full sm:w-auto px-8 py-4 bg-gradient-to-r from-[#fe2c55] to-[#25f4ee] text-white font-semibold rounded-xl transition hover:opacity-90 flex items-center justify-center group"
              >
                Start Creating for Free
                <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <a
                href="#how-it-works"
                className="w-full sm:w-auto px-8 py-4 bg-white/5 border border-white/10 hover:bg-white/10 text-white font-semibold rounded-xl transition flex items-center justify-center"
              >
                <Play className="mr-2 h-5 w-5" />
                See How It Works
              </a>
            </div>

            {/* No credit card required */}
            <p className="mt-6 text-sm text-gray-500">
              No credit card required • Free forever plan available
            </p>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-8"
          >
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-4xl font-bold text-white mb-2">{stat.value}</div>
                <div className="text-gray-500 text-sm">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 px-4 sm:px-6 lg:px-8 bg-gray-900/50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <p className="text-[#25f4ee] font-medium mb-4">FEATURES</p>
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
                Everything You Need to Go Viral
              </h2>
              <p className="text-gray-400 max-w-2xl mx-auto text-lg">
                Our AI-powered toolkit gives you the edge to create content that resonates with your audience and the algorithm.
              </p>
            </motion.div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="p-6 bg-gray-800/50 rounded-2xl border border-gray-700/50 hover:border-gray-600 transition group"
              >
                <div className={`inline-flex items-center justify-center w-12 h-12 bg-gradient-to-r ${feature.color} rounded-xl mb-4`}>
                  <feature.icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-gray-400 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <p className="text-[#fe2c55] font-medium mb-4">HOW IT WORKS</p>
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
                From Idea to Viral in 3 Steps
              </h2>
              <p className="text-gray-400 max-w-2xl mx-auto text-lg">
                Getting started is simple. Connect, create, and publish — we handle the complexity.
              </p>
            </motion.div>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {howItWorks.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="relative"
              >
                <div className="text-center">
                  <div className="text-6xl font-bold text-gray-800 mb-4">{item.step}</div>
                  <div className="w-16 h-16 bg-gradient-to-r from-[#fe2c55] to-[#25f4ee] rounded-2xl flex items-center justify-center mx-auto mb-6">
                    <item.icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-3">{item.title}</h3>
                  <p className="text-gray-400">{item.description}</p>
                </div>
                {index < howItWorks.length - 1 && (
                  <div className="hidden md:block absolute top-24 left-full w-full">
                    <div className="w-full h-0.5 bg-gradient-to-r from-gray-800 to-gray-800 via-gray-700" style={{ width: 'calc(100% - 4rem)', marginLeft: '2rem' }} />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* AI Agents Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gray-900/50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <p className="text-purple-400 font-medium mb-4">AI AGENTS</p>
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
                Meet Your AI Content Team
              </h2>
              <p className="text-gray-400 max-w-2xl mx-auto text-lg">
                Specialized AI agents work together to analyze, create, and optimize your content for maximum impact.
              </p>
            </motion.div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {aiAgents.map((agent, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="p-6 bg-gray-800/50 rounded-2xl border border-gray-700/50 hover:border-purple-500/50 transition text-center group"
              >
                <div className={`w-16 h-16 bg-gradient-to-r ${agent.color} rounded-2xl flex items-center justify-center text-3xl mx-auto mb-4 group-hover:scale-110 transition-transform`}>
                  {agent.avatar}
                </div>
                <h3 className="text-lg font-semibold text-white mb-1">{agent.name}</h3>
                <p className="text-purple-400 text-sm mb-3">{agent.role}</p>
                <p className="text-gray-400 text-sm">{agent.description}</p>
              </motion.div>
            ))}
          </div>

          <div className="mt-12 text-center">
            <Link
              href={isLoggedIn ? "/dashboard/ai-chat" : "/auth/register"}
              className="inline-flex items-center px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl transition"
            >
              <Brain className="mr-2 h-5 w-5" />
              Chat with AI Agents
            </Link>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <p className="text-[#25f4ee] font-medium mb-4">PRICING</p>
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
                Simple, Transparent Pricing
              </h2>
              <p className="text-gray-400 max-w-2xl mx-auto text-lg">
                Start free and scale as you grow. No hidden fees, cancel anytime.
              </p>
            </motion.div>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {pricing.map((plan, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className={`relative p-8 rounded-2xl border ${
                  plan.popular
                    ? 'bg-gradient-to-b from-purple-900/50 to-gray-900/50 border-purple-500/50'
                    : 'bg-gray-800/50 border-gray-700/50'
                }`}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-gradient-to-r from-[#fe2c55] to-[#25f4ee] text-white text-sm font-medium rounded-full">
                    Most Popular
                  </div>
                )}
                <div className="text-center mb-8">
                  <h3 className="text-xl font-semibold text-white mb-2">{plan.name}</h3>
                  <div className="flex items-baseline justify-center gap-1">
                    <span className="text-4xl font-bold text-white">{plan.price}</span>
                    {plan.period && <span className="text-gray-400">{plan.period}</span>}
                  </div>
                  <p className="text-gray-400 text-sm mt-2">{plan.description}</p>
                </div>
                <ul className="space-y-4 mb-8">
                  {plan.features.map((feature, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <Check className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                      <span className="text-gray-300 text-sm">{feature}</span>
                    </li>
                  ))}
                </ul>
                <Link
                  href="/auth/register"
                  className={`block w-full py-3 text-center font-medium rounded-xl transition ${
                    plan.popular
                      ? 'bg-gradient-to-r from-[#fe2c55] to-[#25f4ee] text-white hover:opacity-90'
                      : 'bg-white/10 text-white hover:bg-white/20'
                  }`}
                >
                  {plan.cta}
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-24 px-4 sm:px-6 lg:px-8 bg-gray-900/50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <p className="text-[#fe2c55] font-medium mb-4">TESTIMONIALS</p>
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
                Loved by Creators Worldwide
              </h2>
            </motion.div>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="p-6 bg-gray-800/50 rounded-2xl border border-gray-700/50"
              >
                <div className="flex items-center gap-1 mb-4">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="w-5 h-5 text-yellow-400 fill-yellow-400" />
                  ))}
                </div>
                <p className="text-gray-300 mb-6 leading-relaxed">&quot;{testimonial.content}&quot;</p>
                <div className="flex items-center gap-4">
                  <img
                    src={testimonial.avatar}
                    alt={testimonial.name}
                    className="w-12 h-12 rounded-full"
                  />
                  <div>
                    <div className="font-medium text-white">{testimonial.name}</div>
                    <div className="text-sm text-gray-400">{testimonial.followers} followers • {testimonial.platform}</div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-3xl mx-auto">
          <div className="text-center mb-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <p className="text-purple-400 font-medium mb-4">FAQ</p>
              <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
                Frequently Asked Questions
              </h2>
            </motion.div>
          </div>

          <div className="space-y-4">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="border border-gray-700/50 rounded-xl overflow-hidden"
              >
                <button
                  onClick={() => setOpenFaq(openFaq === index ? null : index)}
                  className="w-full px-6 py-4 flex items-center justify-between text-left bg-gray-800/50 hover:bg-gray-800 transition"
                >
                  <span className="font-medium text-white">{faq.question}</span>
                  <ChevronDown className={`w-5 h-5 text-gray-400 transition-transform ${openFaq === index ? 'rotate-180' : ''}`} />
                </button>
                {openFaq === index && (
                  <div className="px-6 py-4 bg-gray-900/50">
                    <p className="text-gray-400 leading-relaxed">{faq.answer}</p>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="relative overflow-hidden rounded-3xl bg-gradient-to-r from-[#fe2c55] to-[#25f4ee] p-12 text-center"
          >
            <div className="absolute inset-0 bg-black/20" />
            <div className="relative">
              <h2 className="text-3xl md:text-5xl font-bold text-white mb-4">
                Ready to Go Viral?
              </h2>
              <p className="text-white/80 mb-8 text-lg max-w-2xl mx-auto">
                Join 50,000+ creators using Viralify to grow their audience across TikTok, Instagram, and YouTube.
              </p>
              <Link
                href={isLoggedIn ? "/dashboard" : "/auth/register"}
                className="inline-flex items-center px-8 py-4 bg-white text-gray-900 font-semibold rounded-xl hover:bg-gray-100 transition group"
              >
                Get Started Free
                <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <p className="mt-4 text-white/60 text-sm">
                No credit card required
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-16 px-4 sm:px-6 lg:px-8 border-t border-gray-800">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-4 gap-12 mb-12">
            <div>
              <Link href="/" className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#fe2c55] to-[#25f4ee] flex items-center justify-center">
                  <Video className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold text-white">Viralify</span>
              </Link>
              <p className="text-gray-400 text-sm leading-relaxed">
                The AI-powered platform for creating viral content across TikTok, Instagram, and YouTube.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">Product</h4>
              <ul className="space-y-3">
                <li><a href="#features" className="text-gray-400 hover:text-white text-sm transition">Features</a></li>
                <li><a href="#pricing" className="text-gray-400 hover:text-white text-sm transition">Pricing</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white text-sm transition">Changelog</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white text-sm transition">Roadmap</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">Company</h4>
              <ul className="space-y-3">
                <li><a href="#" className="text-gray-400 hover:text-white text-sm transition">About</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white text-sm transition">Blog</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white text-sm transition">Careers</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white text-sm transition">Contact</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-white mb-4">Legal</h4>
              <ul className="space-y-3">
                <li><a href="#" className="text-gray-400 hover:text-white text-sm transition">Privacy Policy</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white text-sm transition">Terms of Service</a></li>
                <li><a href="#" className="text-gray-400 hover:text-white text-sm transition">Cookie Policy</a></li>
              </ul>
            </div>
          </div>

          <div className="pt-8 border-t border-gray-800 flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-gray-500 text-sm">
              © 2024 Viralify. All rights reserved.
            </div>
            <div className="flex items-center gap-4">
              <a href="#" className="text-gray-400 hover:text-white transition">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z"/></svg>
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition">
                <Instagram className="w-5 h-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition">
                <Youtube className="w-5 h-5" />
              </a>
              <a href="#" className="text-gray-400 hover:text-white transition">
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-5.2 1.74 2.89 2.89 0 0 1 2.31-4.64 2.93 2.93 0 0 1 .88.13V9.4a6.84 6.84 0 0 0-1-.05A6.33 6.33 0 0 0 5 20.1a6.34 6.34 0 0 0 10.86-4.43v-7a8.16 8.16 0 0 0 4.77 1.52v-3.4a4.85 4.85 0 0 1-1-.1z"/></svg>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
