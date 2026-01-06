'use client';

import { useState } from 'react';
import { Sparkles, Send, Loader2 } from 'lucide-react';

export function AIAssistantWidget() {
  const [message, setMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<string | null>(null);

  const quickPrompts = [
    'Generate a viral script idea',
    'Best posting times for my niche',
    'Trending hashtags for cooking',
    'Optimize my last video',
  ];

  const handleSend = async () => {
    if (!message.trim() || isLoading) return;

    setIsLoading(true);
    setResponse(null);

    // Simulate AI response in demo mode
    await new Promise(resolve => setTimeout(resolve, 1500));

    setResponse(
      `Great question! Based on current trends, I recommend focusing on short-form content with hooks in the first 3 seconds. ` +
      `For "${message}", try using trending sounds and posting between 7-9 PM for maximum engagement. ` +
      `Would you like me to generate a specific script for this topic?`
    );

    setIsLoading(false);
    setMessage('');
  };

  return (
    <div className="glass rounded-2xl p-6 border border-[#fe2c55]/20">
      <div className="flex items-center gap-2 mb-4">
        <div className="p-2 rounded-lg bg-gradient-to-r from-[#fe2c55] to-[#ff6b6b]">
          <Sparkles className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-white">AI Assistant</h2>
          <p className="text-xs text-gray-400">Ask me anything about content creation</p>
        </div>
      </div>

      {/* Quick Prompts */}
      <div className="flex flex-wrap gap-2 mb-4">
        {quickPrompts.map((prompt, index) => (
          <button
            key={index}
            onClick={() => setMessage(prompt)}
            className="px-3 py-1.5 text-xs bg-white/5 border border-white/10 rounded-full text-gray-300 hover:bg-white/10 hover:text-white transition"
          >
            {prompt}
          </button>
        ))}
      </div>

      {/* Response */}
      {response && (
        <div className="mb-4 p-4 rounded-xl bg-white/5 border border-white/10">
          <p className="text-gray-300 text-sm leading-relaxed">{response}</p>
        </div>
      )}

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask AI for content ideas..."
          className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-[#fe2c55] focus:border-transparent"
        />
        <button
          onClick={handleSend}
          disabled={isLoading || !message.trim()}
          className="px-4 py-3 bg-gradient-to-r from-[#fe2c55] to-[#ff6b6b] text-white rounded-xl hover:opacity-90 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </div>
    </div>
  );
}
