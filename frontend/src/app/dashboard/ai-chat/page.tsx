'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import {
  Brain, Send, Sparkles, TrendingUp, FileText, Target,
  Loader2, Copy, Check, Trash2, MessageSquare, Plus,
  ChevronRight, Info, Lightbulb, Hash, Music, Video,
  Zap, RotateCcw, Download, Share2, Menu, X
} from 'lucide-react';
import toast from 'react-hot-toast';
import { DEMO_MODE } from '@/lib/demo-mode';
import { DashboardLayout } from '@/components/layout/dashboard-layout';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  agent?: string;
  timestamp: Date;
  suggestedActions?: Array<{ type: string; label: string; prompt?: string }>;
}

interface Agent {
  id: string;
  name: string;
  role: string;
  description: string;
  avatar: string;
  color: string;
  capabilities: string[];
}

interface ChatSession {
  id: string;
  title: string;
  agentId: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

const agents: Agent[] = [
  {
    id: 'trendscout',
    name: 'TrendScout',
    role: 'Trend Analyzer',
    description: 'Analyzes viral trends and identifies winning patterns across TikTok, Instagram & YouTube',
    avatar: '🔍',
    color: 'from-blue-500 to-cyan-500',
    capabilities: ['trend_detection', 'pattern_analysis', 'viral_prediction', 'competitor_analysis']
  },
  {
    id: 'scriptgenius',
    name: 'ScriptGenius',
    role: 'Script Writer',
    description: 'Creates engaging video scripts with viral hooks and storytelling techniques',
    avatar: '✍️',
    color: 'from-purple-500 to-pink-500',
    capabilities: ['script_writing', 'hook_creation', 'storytelling', 'call_to_action']
  },
  {
    id: 'contentoptimizer',
    name: 'ContentOptimizer',
    role: 'Content Optimizer',
    description: 'Optimizes captions, hashtags, and posting times for maximum engagement',
    avatar: '🎯',
    color: 'from-orange-500 to-red-500',
    capabilities: ['caption_optimization', 'hashtag_strategy', 'timing_analysis', 'seo']
  },
  {
    id: 'strategyadvisor',
    name: 'StrategyAdvisor',
    role: 'Growth Strategist',
    description: 'Develops comprehensive content strategies for sustainable account growth',
    avatar: '📈',
    color: 'from-green-500 to-emerald-500',
    capabilities: ['strategy_planning', 'content_calendar', 'growth_projection', 'monetization']
  }
];

const quickPrompts = [
  { icon: Sparkles, label: 'Generate viral script', prompt: 'Generate a viral TikTok script about ' },
  { icon: TrendingUp, label: 'Find trending topics', prompt: 'What are the current trending topics in the ' },
  { icon: Hash, label: 'Suggest hashtags', prompt: 'Suggest the best hashtags for a video about ' },
  { icon: Target, label: 'Optimize my caption', prompt: 'Help me optimize this caption for engagement: ' },
  { icon: Lightbulb, label: 'Content ideas', prompt: 'Give me 5 unique content ideas for ' },
  { icon: Video, label: 'Hook ideas', prompt: 'Create 3 attention-grabbing hooks for a video about ' },
];

// Demo mode AI responses
const generateDemoResponse = (agent: Agent, userMessage: string): { content: string; suggestedActions: Message['suggestedActions'] } => {
  const lowerMessage = userMessage.toLowerCase();

  // TrendScout responses
  if (agent.id === 'trendscout') {
    if (lowerMessage.includes('trend') || lowerMessage.includes('viral')) {
      return {
        content: `📊 **Current Trending Analysis**

Based on my analysis of the latest data, here are the top trends right now:

**🔥 Hot Topics This Week:**
1. **"Day in my life"** content - 340% increase in engagement
2. **AI-generated content reveals** - 280% growth
3. **Micro-tutorials (under 30 sec)** - Consistently high performance
4. **Behind-the-scenes content** - 45% higher completion rate

**🎵 Trending Sounds:**
- Original sounds with voiceovers performing 2x better than music-only
- Comedy sound effects seeing a resurgence

**💡 Key Insight:** Videos under 15 seconds are getting 50% more shares this week. Consider creating shorter, punchier content.

Would you like me to dive deeper into any of these trends?`,
        suggestedActions: [
          { type: 'explore', label: '🔍 Explore hashtag trends', prompt: 'Show me the top performing hashtags this week' },
          { type: 'analyze', label: '📈 Competitor analysis', prompt: 'Analyze what my competitors are doing right' }
        ]
      };
    }
    if (lowerMessage.includes('competitor') || lowerMessage.includes('analysis')) {
      return {
        content: `🔎 **Competitor Analysis Framework**

Here's how to analyze your competitors effectively:

**Step 1: Identify Top Performers**
- Look at accounts with similar content/niche
- Focus on those with high engagement rates, not just follower counts

**Step 2: Content Patterns to Track**
- Posting frequency (most successful: 1-3x daily)
- Video length patterns
- Hook styles they use
- Caption strategies

**Step 3: Engagement Analysis**
- Which videos get the most comments?
- What topics drive shares?
- Peak posting times

**🎯 Pro Tip:** Don't copy—adapt. Take successful formats and add your unique spin.

Want me to suggest specific accounts to study in your niche?`,
        suggestedActions: [
          { type: 'niche', label: '🎯 Find niche leaders', prompt: 'Find the top creators in the fitness niche' },
          { type: 'strategy', label: '📝 Create strategy', prompt: 'Create a content strategy based on competitor analysis' }
        ]
      };
    }
  }

  // ScriptGenius responses
  if (agent.id === 'scriptgenius') {
    if (lowerMessage.includes('script') || lowerMessage.includes('write')) {
      const topic = lowerMessage.includes('about')
        ? lowerMessage.split('about')[1]?.trim() || 'your topic'
        : 'your topic';
      return {
        content: `✍️ **Viral TikTok Script: ${topic}**

---

**🎬 HOOK (0-3 sec)**
"Stop scrolling if you want to ${topic}..."

**📖 SETUP (3-10 sec)**
"Most people don't know this, but [surprising fact about ${topic}]"

**💡 VALUE (10-40 sec)**
"Here's what you need to do:
1. First, [action step]
2. Then, [action step]
3. Finally, [action step]"

**🔥 PAYOFF (40-50 sec)**
"And that's how I [achieved result]. The difference is insane."

**📢 CTA (50-60 sec)**
"Follow for more tips like this, and comment 'MORE' if you want Part 2!"

---

**Pro Tips for This Script:**
- Maintain eye contact with camera
- Use text overlays for key points
- Add trending sound at low volume

Want me to create alternative hooks or a different style?`,
        suggestedActions: [
          { type: 'hooks', label: '🎣 More hook ideas', prompt: 'Give me 5 alternative hooks for this script' },
          { type: 'shorter', label: '⚡ Make it shorter', prompt: 'Create a 15-second version of this script' }
        ]
      };
    }
    if (lowerMessage.includes('hook')) {
      return {
        content: `🎣 **Attention-Grabbing Hooks**

Here are proven hook formulas that stop the scroll:

**1. The Curiosity Gap**
"Nobody talks about this but..."
"The thing about [topic] that nobody mentions..."

**2. The Controversial Take**
"Unpopular opinion: [bold statement]"
"I'm going to get hate for this but..."

**3. The Promise**
"This changed my life in 30 days"
"Watch till the end for [specific benefit]"

**4. The Pattern Interrupt**
*Start mid-action or mid-sentence*
"—and that's when everything changed"

**5. The Direct Address**
"If you're struggling with [problem], this is for you"
"POV: You just discovered [solution]"

**🎯 Best Practice:** Test 3 different hooks for the same content and see which performs best!`,
        suggestedActions: [
          { type: 'customize', label: '✨ Customize for my niche', prompt: 'Create custom hooks for the cooking niche' },
          { type: 'script', label: '📝 Full script', prompt: 'Now write a full script using hook #1' }
        ]
      };
    }
  }

  // ContentOptimizer responses
  if (agent.id === 'contentoptimizer') {
    if (lowerMessage.includes('hashtag')) {
      return {
        content: `#️⃣ **Hashtag Strategy Guide**

**The Perfect Hashtag Mix (aim for 5-10):**

🔥 **Trending (1-2):** High competition but massive reach
- #fyp #foryou #viral

📊 **Medium (3-4):** 100K-1M posts, good discoverability
- #[niche]tips #[niche]hack #[topic]advice

🎯 **Niche (3-4):** Under 100K posts, highly targeted
- #[specific_niche] #[micro_community]

**💡 Pro Tips:**
1. Place hashtags in caption, not comments (better for SEO)
2. Rotate hashtags to avoid shadowban
3. Check hashtag performance weekly
4. Use hashtag research tools to find rising tags

**🚫 Avoid:**
- Banned/restricted hashtags
- Irrelevant trending tags
- Overused generic tags only

Would you like specific hashtags for your content niche?`,
        suggestedActions: [
          { type: 'specific', label: '🏷️ Get specific tags', prompt: 'Give me hashtags for fitness content' },
          { type: 'caption', label: '✏️ Optimize caption', prompt: 'Help me write an optimized caption with hashtags' }
        ]
      };
    }
    if (lowerMessage.includes('caption') || lowerMessage.includes('optimize')) {
      return {
        content: `✏️ **Caption Optimization Framework**

**Structure for Maximum Engagement:**

**Line 1 - The Hook** (most important!)
→ Question, bold statement, or intrigue

**Line 2-3 - Value/Context**
→ What they'll learn or why it matters

**Line 4 - CTA (Call to Action)**
→ Tell them exactly what to do

---

**Example Transformation:**

❌ Before: "New video! Check it out 😊"

✅ After: "This ONE trick increased my engagement by 300%

I tried it for 30 days and the results were insane.

Save this and try it on your next post 👇

#contentcreator #socialmediatips #growthhack"

---

**Power Words to Use:**
Secret, Proven, Instant, Free, Easy, Ultimate, Shocking, Finally

Share your caption and I'll optimize it!`,
        suggestedActions: [
          { type: 'rewrite', label: '🔄 Rewrite my caption', prompt: 'Rewrite this caption to be more engaging: ' },
          { type: 'cta', label: '📢 Better CTA ideas', prompt: 'Give me 10 powerful call-to-action ideas' }
        ]
      };
    }
  }

  // StrategyAdvisor responses
  if (agent.id === 'strategyadvisor') {
    if (lowerMessage.includes('strategy') || lowerMessage.includes('grow')) {
      return {
        content: `📈 **30-Day Growth Strategy**

**Week 1: Foundation**
- Audit your current content performance
- Define your content pillars (3-4 main topics)
- Create a consistent posting schedule

**Week 2: Content Optimization**
- Post 1-2x daily at peak times
- Test different video lengths
- Implement hook formulas

**Week 3: Engagement Push**
- Respond to ALL comments within 1 hour
- Engage with 20 accounts in your niche daily
- Collaborate with similar-sized creators

**Week 4: Scale & Analyze**
- Double down on top-performing content types
- Repurpose best content for other platforms
- Review analytics and adjust strategy

**📊 Expected Results:**
- 20-50% follower growth
- 2x engagement rate improvement
- Clearer content direction

**🎯 Key Success Factors:**
1. Consistency > Perfection
2. Engage authentically
3. Provide value first

Want me to create a detailed content calendar?`,
        suggestedActions: [
          { type: 'calendar', label: '📅 Content calendar', prompt: 'Create a weekly content calendar for me' },
          { type: 'pillars', label: '🎯 Define content pillars', prompt: 'Help me define my content pillars' }
        ]
      };
    }
    if (lowerMessage.includes('calendar') || lowerMessage.includes('schedule')) {
      return {
        content: `📅 **Weekly Content Calendar Template**

**MONDAY - Educational**
🕐 Post at 7 AM & 6 PM
📝 "How to..." or "X tips for..."
🎯 Goal: Establish expertise

**TUESDAY - Trending**
🕐 Post at 12 PM & 8 PM
📝 Jump on current trends with your spin
🎯 Goal: Reach new audiences

**WEDNESDAY - Behind-the-scenes**
🕐 Post at 9 AM & 7 PM
📝 Process, workspace, day-in-life
🎯 Goal: Build connection

**THURSDAY - Value Bomb**
🕐 Post at 11 AM & 9 PM
📝 Your best tip/hack of the week
🎯 Goal: Get saves & shares

**FRIDAY - Entertainment**
🕐 Post at 2 PM & 10 PM
📝 Fun, relatable content
🎯 Goal: Increase engagement

**WEEKEND - Engagement**
🕐 Post at 10 AM & 5 PM
📝 Questions, polls, duets
🎯 Goal: Community building

**💡 Pro Tip:** Batch create content on Sundays for the week ahead!`,
        suggestedActions: [
          { type: 'ideas', label: '💡 Content ideas', prompt: 'Give me specific content ideas for each day' },
          { type: 'batch', label: '📦 Batch creation tips', prompt: 'How can I batch create content efficiently?' }
        ]
      };
    }
  }

  // Default response
  return {
    content: `Thanks for your question! I'm ${agent.name}, your ${agent.role}.

Based on what you're asking about, here's what I can help you with:

${agent.capabilities.map(cap => `• ${cap.replace(/_/g, ' ').charAt(0).toUpperCase() + cap.replace(/_/g, ' ').slice(1)}`).join('\n')}

Could you be more specific about what you'd like to achieve? For example:
- What platform are you focusing on?
- What's your content niche?
- What's your current challenge?

The more details you share, the better I can help! 🚀`,
    suggestedActions: [
      { type: 'script', label: '✍️ Write a script', prompt: 'Write a viral script about ' },
      { type: 'trends', label: '📊 Show trends', prompt: 'What are the current trends in ' },
      { type: 'strategy', label: '📈 Growth strategy', prompt: 'Create a growth strategy for my account' }
    ]
  };
};

export default function AIChatPage() {
  const [selectedAgent, setSelectedAgent] = useState<Agent>(agents[0]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(false);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load chat sessions from localStorage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('viralify_chat_sessions');
      if (saved) {
        const sessions = JSON.parse(saved);
        setChatSessions(sessions);
      }
    }
  }, []);

  // Save chat sessions to localStorage
  useEffect(() => {
    if (typeof window !== 'undefined' && chatSessions.length > 0) {
      localStorage.setItem('viralify_chat_sessions', JSON.stringify(chatSessions));
    }
  }, [chatSessions]);

  // Initialize with welcome message
  useEffect(() => {
    if (messages.length === 0) {
      const welcomeMessage: Message = {
        id: `welcome-${Date.now()}`,
        role: 'assistant',
        content: `👋 Hi! I'm **${selectedAgent.name}**, your ${selectedAgent.role}.

${selectedAgent.description}.

**I can help you with:**
${selectedAgent.capabilities.map(cap => `• ${cap.replace(/_/g, ' ').charAt(0).toUpperCase() + cap.replace(/_/g, ' ').slice(1)}`).join('\n')}

What would you like to work on today?`,
        agent: selectedAgent.name,
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
    }
  }, [selectedAgent, messages.length]);

  const startNewChat = () => {
    // Save current session if it has messages
    if (messages.length > 1 && currentSessionId) {
      const updatedSessions = chatSessions.map(s =>
        s.id === currentSessionId
          ? { ...s, messages, updatedAt: new Date() }
          : s
      );
      setChatSessions(updatedSessions);
    }

    const welcomeMessage: Message = {
      id: `welcome-${Date.now()}`,
      role: 'assistant',
      content: `👋 Hi! I'm **${selectedAgent.name}**, your ${selectedAgent.role}.

${selectedAgent.description}.

What would you like to work on today?`,
      agent: selectedAgent.name,
      timestamp: new Date()
    };

    const newSessionId = `session-${Date.now()}`;
    setCurrentSessionId(newSessionId);
    setMessages([welcomeMessage]);
    setShowSidebar(false);
  };

  const switchAgent = (agent: Agent) => {
    setSelectedAgent(agent);
    const welcomeMessage: Message = {
      id: `welcome-${Date.now()}`,
      role: 'assistant',
      content: `👋 Hi! I'm **${agent.name}**, your ${agent.role}.

${agent.description}.

**I can help you with:**
${agent.capabilities.map(cap => `• ${cap.replace(/_/g, ' ').charAt(0).toUpperCase() + cap.replace(/_/g, ' ').slice(1)}`).join('\n')}

What would you like to work on today?`,
      agent: agent.name,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
    setShowSidebar(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setIsLoading(true);

    try {
      if (DEMO_MODE) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));

        const { content, suggestedActions } = generateDemoResponse(selectedAgent, currentInput);

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content,
          agent: selectedAgent.name,
          timestamp: new Date(),
          suggestedActions
        };

        setMessages(prev => [...prev, assistantMessage]);
      } else {
        const token = localStorage.getItem('accessToken');
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/content/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
            agent_name: selectedAgent.name,
            message: currentInput,
            context: {}
          })
        });

        const data = await response.json();

        const assistantMessage: Message = {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: data.response || 'I apologize, but I encountered an error. Please try again.',
          agent: selectedAgent.name,
          timestamp: new Date(),
          suggestedActions: data.suggested_actions
        };

        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      toast.error('Failed to get response. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickPrompt = (prompt: string) => {
    setInput(prompt);
    inputRef.current?.focus();
  };

  const handleSuggestedAction = (action: { prompt?: string }) => {
    if (action.prompt) {
      setInput(action.prompt);
      inputRef.current?.focus();
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast.success('Copied to clipboard');
    setTimeout(() => setCopiedId(null), 2000);
  };

  const clearChat = () => {
    const welcomeMessage: Message = {
      id: `welcome-${Date.now()}`,
      role: 'assistant',
      content: `Chat cleared! 👋 I'm still here to help. What would you like to work on?`,
      agent: selectedAgent.name,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
    toast.success('Chat cleared');
  };

  const formatMessage = (content: string) => {
    // Simple markdown-like formatting
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\n/g, '<br/>');
  };

  return (
    <DashboardLayout>
      <div className="h-[calc(100vh-120px)] flex flex-col">
        {/* Demo Mode Banner */}
        {DEMO_MODE && (
          <div className="p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-xl flex items-center gap-2 mb-4">
            <Info className="w-5 h-5 text-yellow-400 flex-shrink-0" />
            <p className="text-sm text-yellow-200">
              <strong>Demo Mode:</strong> AI responses are simulated. Connect your backend for real AI chat.
            </p>
          </div>
        )}

        <div className="flex-1 flex overflow-hidden bg-gray-800/50 rounded-2xl border border-gray-700">
          {/* Mobile Sidebar Toggle */}
          <button
            onClick={() => setShowSidebar(!showSidebar)}
            className="lg:hidden fixed bottom-24 left-4 z-50 p-3 bg-purple-600 rounded-full shadow-lg"
          >
            {showSidebar ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>

          {/* Sidebar */}
          <div className={`${showSidebar ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 fixed lg:relative z-40 w-72 h-full bg-gray-900 lg:bg-transparent border-r border-gray-700 flex flex-col transition-transform duration-300`}>
            {/* Agent Selector */}
            <div className="p-4 border-b border-gray-700">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">AI Agents</h2>
                <button
                  onClick={startNewChat}
                  className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition"
                  title="New chat"
                >
                  <Plus className="h-4 w-4" />
                </button>
              </div>

              <div className="space-y-2">
                {agents.map((agent) => (
                  <button
                    key={agent.id}
                    onClick={() => switchAgent(agent)}
                    className={`w-full p-3 rounded-xl text-left transition ${
                      selectedAgent.id === agent.id
                        ? 'bg-gradient-to-r ' + agent.color + ' bg-opacity-20 border border-white/20'
                        : 'bg-gray-800/50 border border-transparent hover:bg-gray-700/50'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${agent.color} flex items-center justify-center text-xl`}>
                        {agent.avatar}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-white text-sm">{agent.name}</div>
                        <div className="text-xs text-gray-400 truncate">{agent.role}</div>
                      </div>
                      {selectedAgent.id === agent.id && (
                        <Check className="h-4 w-4 text-white" />
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Prompts */}
            <div className="flex-1 overflow-y-auto p-4">
              <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">Quick Prompts</h3>
              <div className="space-y-2">
                {quickPrompts.map((prompt, index) => (
                  <button
                    key={index}
                    onClick={() => handleQuickPrompt(prompt.prompt)}
                    className="w-full p-3 bg-gray-800/50 hover:bg-gray-700/50 rounded-xl text-left transition group"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-gray-700 rounded-lg group-hover:bg-purple-600/20 transition">
                        <prompt.icon className="h-4 w-4 text-gray-400 group-hover:text-purple-400" />
                      </div>
                      <span className="text-sm text-gray-300">{prompt.label}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="p-4 border-t border-gray-700">
              <button
                onClick={clearChat}
                className="w-full p-2 text-gray-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition flex items-center justify-center gap-2 text-sm"
              >
                <Trash2 className="h-4 w-4" />
                Clear Chat
              </button>
            </div>
          </div>

          {/* Main Chat Area */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Chat Header */}
            <div className="p-4 border-b border-gray-700 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-xl bg-gradient-to-r ${selectedAgent.color} flex items-center justify-center text-xl`}>
                  {selectedAgent.avatar}
                </div>
                <div>
                  <h1 className="font-semibold text-white">{selectedAgent.name}</h1>
                  <p className="text-xs text-gray-400">{selectedAgent.role} • Online</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    const chatText = messages.map(m => `${m.role}: ${m.content}`).join('\n\n');
                    navigator.clipboard.writeText(chatText);
                    toast.success('Chat exported to clipboard');
                  }}
                  className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition"
                  title="Export chat"
                >
                  <Download className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <AnimatePresence>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] lg:max-w-[70%] ${
                        message.role === 'user'
                          ? 'bg-purple-600 text-white rounded-2xl rounded-br-md'
                          : 'bg-gray-700/50 text-gray-100 rounded-2xl rounded-bl-md'
                      } p-4`}
                    >
                      {message.role === 'assistant' && (
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-lg">{selectedAgent.avatar}</span>
                          <span className="text-sm font-medium text-gray-300">{message.agent}</span>
                        </div>
                      )}

                      <div
                        className="text-sm leading-relaxed prose prose-invert max-w-none"
                        dangerouslySetInnerHTML={{ __html: formatMessage(message.content) }}
                      />

                      {message.role === 'assistant' && (
                        <div className="flex items-center justify-between mt-3 pt-3 border-t border-gray-600">
                          <span className="text-xs text-gray-500">
                            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                          <div className="flex items-center gap-1">
                            <button
                              onClick={() => copyToClipboard(message.content, message.id)}
                              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-600 rounded transition"
                              title="Copy"
                            >
                              {copiedId === message.id ? (
                                <Check className="h-4 w-4 text-green-400" />
                              ) : (
                                <Copy className="h-4 w-4" />
                              )}
                            </button>
                            <button
                              onClick={() => {
                                setInput(`Regenerate: ${messages.find(m => m.id === message.id)?.content.slice(0, 50)}...`);
                                inputRef.current?.focus();
                              }}
                              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-600 rounded transition"
                              title="Regenerate"
                            >
                              <RotateCcw className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      )}

                      {/* Suggested Actions */}
                      {message.suggestedActions && message.suggestedActions.length > 0 && (
                        <div className="flex flex-wrap gap-2 mt-3">
                          {message.suggestedActions.map((action, idx) => (
                            <button
                              key={idx}
                              onClick={() => handleSuggestedAction(action)}
                              className="px-3 py-1.5 bg-purple-500/20 text-purple-300 text-xs rounded-full hover:bg-purple-500/30 transition flex items-center gap-1"
                            >
                              <ChevronRight className="h-3 w-3" />
                              {action.label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex justify-start"
                >
                  <div className="bg-gray-700/50 rounded-2xl rounded-bl-md p-4">
                    <div className="flex items-center gap-3">
                      <span className="text-xl">{selectedAgent.avatar}</span>
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
                        <span className="text-sm text-gray-400">Thinking...</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-gray-700">
              <form onSubmit={handleSubmit} className="flex items-end gap-3">
                <div className="flex-1 relative">
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => {
                      setInput(e.target.value);
                      e.target.style.height = 'auto';
                      e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(e);
                      }
                    }}
                    placeholder={`Ask ${selectedAgent.name} anything...`}
                    rows={1}
                    className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none text-sm"
                    style={{ minHeight: '48px', maxHeight: '150px' }}
                  />
                </div>
                <button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  className="p-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white rounded-xl transition flex-shrink-0"
                >
                  {isLoading ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Send className="h-5 w-5" />
                  )}
                </button>
              </form>
              <p className="mt-2 text-xs text-gray-500 text-center">
                Press Enter to send • Shift + Enter for new line
              </p>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
