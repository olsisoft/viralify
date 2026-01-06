'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  FileText, Wand2, Download, Copy, RefreshCw,
  Loader2, Sparkles, Info, Check, ChevronDown,
  List, AlignLeft, Hash, Globe, Zap
} from 'lucide-react';
import toast from 'react-hot-toast';
import { DEMO_MODE } from '@/lib/demo-mode';

interface GeneratedArticle {
  id: string;
  title: string;
  content: string;
  outline: string[];
  seoKeywords: string[];
  wordCount: number;
  readingTime: number;
  createdAt: Date;
}

const articleTypes = [
  { id: 'blog', name: 'Blog Post', description: 'Engaging blog article', icon: '📝' },
  { id: 'listicle', name: 'Listicle', description: 'List-based article', icon: '📋' },
  { id: 'how-to', name: 'How-To Guide', description: 'Step-by-step tutorial', icon: '📖' },
  { id: 'news', name: 'News Article', description: 'Informational piece', icon: '📰' },
  { id: 'review', name: 'Review', description: 'Product/service review', icon: '⭐' },
  { id: 'opinion', name: 'Opinion Piece', description: 'Personal perspective', icon: '💭' },
];

const tones = [
  { id: 'professional', name: 'Professional' },
  { id: 'casual', name: 'Casual' },
  { id: 'conversational', name: 'Conversational' },
  { id: 'authoritative', name: 'Authoritative' },
  { id: 'friendly', name: 'Friendly' },
  { id: 'humorous', name: 'Humorous' },
];

const lengths = [
  { id: 'short', name: 'Short', words: '300-500', credits: 1 },
  { id: 'medium', name: 'Medium', words: '500-800', credits: 2 },
  { id: 'long', name: 'Long', words: '800-1200', credits: 3 },
  { id: 'comprehensive', name: 'Comprehensive', words: '1200-2000', credits: 5 },
];

// Demo article
const DEMO_ARTICLE = {
  title: "10 Productivity Hacks That Will Transform Your Daily Routine",
  outline: [
    "Introduction: The importance of productivity",
    "1. Start with the hardest task first",
    "2. Use the Pomodoro Technique",
    "3. Eliminate digital distractions",
    "4. Create a dedicated workspace",
    "5. Take regular breaks",
    "6. Use productivity apps",
    "7. Plan your day the night before",
    "8. Practice the two-minute rule",
    "9. Batch similar tasks together",
    "10. Review and reflect weekly",
    "Conclusion: Building lasting habits"
  ],
  content: `# 10 Productivity Hacks That Will Transform Your Daily Routine

In today's fast-paced world, being productive isn't just about working harder—it's about working smarter. Whether you're a busy professional, entrepreneur, or student, these ten proven productivity hacks will help you accomplish more while reducing stress.

## 1. Start with the Hardest Task First

Also known as "eating the frog," tackling your most challenging task first thing in the morning sets a positive tone for the entire day. Your willpower and energy are at their peak in the morning, making it the ideal time to handle complex work.

## 2. Use the Pomodoro Technique

Work in focused 25-minute intervals followed by 5-minute breaks. After four "pomodoros," take a longer 15-30 minute break. This technique helps maintain concentration while preventing burnout.

## 3. Eliminate Digital Distractions

Turn off non-essential notifications, use website blockers during work hours, and consider keeping your phone in another room. The average person checks their phone 96 times per day—imagine what you could accomplish with that time back.

## 4. Create a Dedicated Workspace

Your environment shapes your productivity. Having a clean, organized workspace signals to your brain that it's time to focus. Invest in good lighting, a comfortable chair, and keep only essential items on your desk.

## 5. Take Regular Breaks

Contrary to popular belief, working non-stop doesn't make you more productive. Regular breaks help prevent mental fatigue and maintain consistent performance throughout the day.

## 6. Use Productivity Apps

Leverage technology to your advantage. Apps like Notion for organization, Forest for focus, and Todoist for task management can streamline your workflow and keep you accountable.

## 7. Plan Your Day the Night Before

Spend 10 minutes each evening planning tomorrow's priorities. This practice reduces morning decision fatigue and helps you hit the ground running.

## 8. Practice the Two-Minute Rule

If a task takes less than two minutes, do it immediately. This prevents small tasks from piling up and overwhelming your to-do list.

## 9. Batch Similar Tasks Together

Group similar activities together to minimize context switching. Check emails at set times, make all phone calls in one block, and schedule meetings back-to-back when possible.

## 10. Review and Reflect Weekly

Set aside time each week to review what worked, what didn't, and adjust your strategies accordingly. Continuous improvement is key to long-term productivity gains.

## Conclusion

Productivity isn't about perfection—it's about progress. Start by implementing one or two of these hacks, and gradually add more as they become habits. Remember, the goal isn't to do more things, but to do the right things more effectively.

*What's your favorite productivity hack? Share in the comments below!*`,
  seoKeywords: ["productivity tips", "work efficiency", "time management", "productivity hacks", "daily routine"],
  wordCount: 456,
  readingTime: 3
};

export default function ArticlesPage() {
  const [topic, setTopic] = useState('');
  const [keywords, setKeywords] = useState('');
  const [selectedType, setSelectedType] = useState('blog');
  const [selectedTone, setSelectedTone] = useState('professional');
  const [selectedLength, setSelectedLength] = useState('medium');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedArticle, setGeneratedArticle] = useState<GeneratedArticle | null>(null);
  const [showOutline, setShowOutline] = useState(false);
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced options
  const [includeIntro, setIncludeIntro] = useState(true);
  const [includeConclusion, setIncludeConclusion] = useState(true);
  const [includeCTA, setIncludeCTA] = useState(true);
  const [targetAudience, setTargetAudience] = useState('');

  const getCredits = () => {
    return lengths.find(l => l.id === selectedLength)?.credits || 2;
  };

  const handleGenerate = async () => {
    if (!topic.trim()) {
      toast.error('Please enter a topic');
      return;
    }

    setIsGenerating(true);

    try {
      if (DEMO_MODE) {
        await new Promise(resolve => setTimeout(resolve, 3000));

        const article: GeneratedArticle = {
          id: `article-${Date.now()}`,
          title: DEMO_ARTICLE.title,
          content: DEMO_ARTICLE.content,
          outline: DEMO_ARTICLE.outline,
          seoKeywords: keywords ? keywords.split(',').map(k => k.trim()) : DEMO_ARTICLE.seoKeywords,
          wordCount: DEMO_ARTICLE.wordCount,
          readingTime: DEMO_ARTICLE.readingTime,
          createdAt: new Date(),
        };

        setGeneratedArticle(article);
        toast.success('Article generated! (Demo)');
      } else {
        const response = await fetch('/api/v1/media/article', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic,
            keywords: keywords.split(',').map(k => k.trim()).filter(k => k),
            article_type: selectedType,
            tone: selectedTone,
            length: selectedLength,
            include_intro: includeIntro,
            include_conclusion: includeConclusion,
            include_cta: includeCTA,
            target_audience: targetAudience,
          }),
        });

        if (!response.ok) throw new Error('Generation failed');
        // Handle response...
      }
    } catch (error) {
      toast.error('Failed to generate article');
    } finally {
      setIsGenerating(false);
    }
  };

  const copyToClipboard = (text: string, field: string) => {
    navigator.clipboard.writeText(text);
    setCopiedField(field);
    toast.success('Copied!');
    setTimeout(() => setCopiedField(null), 2000);
  };

  const downloadMarkdown = () => {
    if (!generatedArticle) return;

    const blob = new Blob([generatedArticle.content], { type: 'text/markdown' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${generatedArticle.title.toLowerCase().replace(/\s+/g, '-')}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    toast.success('Article downloaded!');
  };

  return (
    <div className="min-h-screen bg-gray-900 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-gradient-to-r from-green-500 to-emerald-500 rounded-xl">
              <FileText className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-white">Article Writer</h1>
          </div>
          <p className="text-gray-400">
            Generate SEO-optimized articles and blog posts with AI
          </p>
        </div>

        {/* Demo Banner */}
        {DEMO_MODE && (
          <div className="mb-6 p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-xl flex items-center gap-2">
            <Info className="w-5 h-5 text-yellow-400 flex-shrink-0" />
            <p className="text-sm text-yellow-200">
              <strong>Demo Mode:</strong> Showing sample article. Real GPT-4 generation in production.
            </p>
          </div>
        )}

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <div className="space-y-6">
            {/* Topic Input */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Article Topic *
              </label>
              <textarea
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="What should the article be about? Be specific for better results..."
                rows={3}
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500 resize-none"
              />
            </div>

            {/* Keywords */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                <Hash className="inline w-4 h-4 mr-1" />
                SEO Keywords (optional)
              </label>
              <input
                type="text"
                value={keywords}
                onChange={(e) => setKeywords(e.target.value)}
                placeholder="productivity, time management, work tips"
                className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500"
              />
              <p className="text-xs text-gray-500 mt-2">Separate keywords with commas</p>
            </div>

            {/* Article Type */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Article Type
              </label>
              <div className="grid grid-cols-3 gap-2">
                {articleTypes.map((type) => (
                  <button
                    key={type.id}
                    onClick={() => setSelectedType(type.id)}
                    className={`p-3 rounded-xl border text-center transition ${
                      selectedType === type.id
                        ? 'bg-green-600/20 border-green-500'
                        : 'bg-gray-700 border-gray-600 hover:border-gray-500'
                    }`}
                  >
                    <div className="text-xl mb-1">{type.icon}</div>
                    <div className="text-xs text-gray-300">{type.name}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Tone Selection */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Writing Tone
              </label>
              <div className="flex flex-wrap gap-2">
                {tones.map((tone) => (
                  <button
                    key={tone.id}
                    onClick={() => setSelectedTone(tone.id)}
                    className={`px-4 py-2 rounded-xl border transition ${
                      selectedTone === tone.id
                        ? 'bg-green-600/20 border-green-500 text-white'
                        : 'bg-gray-700 border-gray-600 text-gray-300'
                    }`}
                  >
                    {tone.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Length Selection */}
            <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
              <label className="block text-sm font-medium text-gray-300 mb-3">
                Article Length
              </label>
              <div className="grid grid-cols-2 gap-3">
                {lengths.map((length) => (
                  <button
                    key={length.id}
                    onClick={() => setSelectedLength(length.id)}
                    className={`p-3 rounded-xl border text-left transition ${
                      selectedLength === length.id
                        ? 'bg-green-600/20 border-green-500 text-white'
                        : 'bg-gray-700 border-gray-600 text-gray-300 hover:border-gray-500'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{length.name}</span>
                      <span className="text-xs flex items-center gap-1">
                        <Zap className="w-3 h-3 text-yellow-400" />
                        {length.credits}
                      </span>
                    </div>
                    <span className="text-xs text-gray-400">{length.words} words</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Advanced Options */}
            <div className="bg-gray-800 rounded-2xl border border-gray-700 overflow-hidden">
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full p-4 flex items-center justify-between text-gray-300 hover:bg-gray-700/50 transition"
              >
                <span className="font-medium">Advanced Options</span>
                <ChevronDown className={`w-5 h-5 transition ${showAdvanced ? 'rotate-180' : ''}`} />
              </button>

              {showAdvanced && (
                <div className="p-4 pt-0 space-y-4 border-t border-gray-700">
                  {/* Target Audience */}
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Target Audience</label>
                    <input
                      type="text"
                      value={targetAudience}
                      onChange={(e) => setTargetAudience(e.target.value)}
                      placeholder="e.g., Entrepreneurs, Students, Parents"
                      className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500"
                    />
                  </div>

                  {/* Toggles */}
                  <div className="space-y-3">
                    <label className="flex items-center justify-between cursor-pointer">
                      <span className="text-sm text-gray-400">Include Introduction</span>
                      <input
                        type="checkbox"
                        checked={includeIntro}
                        onChange={(e) => setIncludeIntro(e.target.checked)}
                        className="w-5 h-5 rounded accent-green-500"
                      />
                    </label>
                    <label className="flex items-center justify-between cursor-pointer">
                      <span className="text-sm text-gray-400">Include Conclusion</span>
                      <input
                        type="checkbox"
                        checked={includeConclusion}
                        onChange={(e) => setIncludeConclusion(e.target.checked)}
                        className="w-5 h-5 rounded accent-green-500"
                      />
                    </label>
                    <label className="flex items-center justify-between cursor-pointer">
                      <span className="text-sm text-gray-400">Include Call-to-Action</span>
                      <input
                        type="checkbox"
                        checked={includeCTA}
                        onChange={(e) => setIncludeCTA(e.target.checked)}
                        className="w-5 h-5 rounded accent-green-500"
                      />
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !topic.trim()}
              className="w-full py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white font-semibold rounded-xl hover:from-green-700 hover:to-emerald-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Generating Article...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Generate Article ({getCredits()} credits)
                </>
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {generatedArticle ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-6"
              >
                {/* Article Header */}
                <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h2 className="text-xl font-bold text-white mb-2">
                        {generatedArticle.title}
                      </h2>
                      <div className="flex items-center gap-4 text-sm text-gray-400">
                        <span className="flex items-center gap-1">
                          <AlignLeft className="w-4 h-4" />
                          {generatedArticle.wordCount} words
                        </span>
                        <span>{generatedArticle.readingTime} min read</span>
                      </div>
                    </div>
                    <button
                      onClick={() => copyToClipboard(generatedArticle.title, 'title')}
                      className="p-2 hover:bg-gray-700 rounded-lg transition"
                    >
                      {copiedField === 'title' ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-400" />
                      )}
                    </button>
                  </div>

                  {/* SEO Keywords */}
                  <div className="flex flex-wrap gap-2">
                    {generatedArticle.seoKeywords.map((keyword, i) => (
                      <span
                        key={i}
                        className="px-2 py-1 bg-green-600/20 text-green-400 rounded-lg text-xs"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Outline Toggle */}
                <div className="bg-gray-800 rounded-2xl border border-gray-700 overflow-hidden">
                  <button
                    onClick={() => setShowOutline(!showOutline)}
                    className="w-full p-4 flex items-center justify-between text-gray-300 hover:bg-gray-700/50 transition"
                  >
                    <span className="flex items-center gap-2">
                      <List className="w-4 h-4" />
                      Article Outline
                    </span>
                    <ChevronDown className={`w-5 h-5 transition ${showOutline ? 'rotate-180' : ''}`} />
                  </button>

                  {showOutline && (
                    <div className="p-4 pt-0 border-t border-gray-700">
                      <ul className="space-y-2">
                        {generatedArticle.outline.map((item, i) => (
                          <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                            <span className="text-green-400">•</span>
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {/* Article Content */}
                <div className="bg-gray-800 rounded-2xl p-6 border border-gray-700">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium text-gray-400">Article Content</h3>
                    <button
                      onClick={() => copyToClipboard(generatedArticle.content, 'content')}
                      className="p-2 hover:bg-gray-700 rounded-lg transition"
                    >
                      {copiedField === 'content' ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-400" />
                      )}
                    </button>
                  </div>

                  <div className="prose prose-invert prose-sm max-w-none max-h-96 overflow-y-auto">
                    <div className="whitespace-pre-wrap text-gray-300 text-sm leading-relaxed">
                      {generatedArticle.content}
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  <button
                    onClick={downloadMarkdown}
                    className="flex-1 py-3 bg-gray-700 text-white rounded-xl hover:bg-gray-600 transition flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download Markdown
                  </button>
                  <button
                    onClick={() => {
                      setGeneratedArticle(null);
                      handleGenerate();
                    }}
                    className="flex-1 py-3 bg-green-600 text-white rounded-xl hover:bg-green-700 transition flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    Regenerate
                  </button>
                </div>
              </motion.div>
            ) : (
              <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700 text-center">
                <div className="w-16 h-16 bg-gray-700 rounded-2xl flex items-center justify-center mx-auto mb-4">
                  <FileText className="w-8 h-8 text-gray-500" />
                </div>
                <h3 className="text-lg font-medium text-white mb-2">No article yet</h3>
                <p className="text-gray-400 text-sm">
                  Enter a topic and click generate to create your first article
                </p>
              </div>
            )}

            {/* Tips */}
            <div className="bg-gradient-to-r from-green-600/10 to-emerald-600/10 border border-green-500/20 rounded-xl p-4">
              <h4 className="text-white font-medium mb-2">Writing Tips</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Be specific about your topic for better results</li>
                <li>• Include target keywords for SEO optimization</li>
                <li>• Choose the right tone for your audience</li>
                <li>• Edit and personalize the generated content</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
