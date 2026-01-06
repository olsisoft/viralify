'use client';

import Link from 'next/link';
import { Video, Sparkles, Calendar, TrendingUp, Settings, BarChart3 } from 'lucide-react';

const actions = [
  {
    label: 'Create Post',
    description: 'Upload and schedule content',
    icon: Video,
    href: '/dashboard/create',
    color: 'from-[#fe2c55] to-[#ff6b6b]',
  },
  {
    label: 'AI Script',
    description: 'Generate viral scripts',
    icon: Sparkles,
    href: '/dashboard/ai-chat',
    color: 'from-purple-500 to-pink-500',
  },
  {
    label: 'Schedule',
    description: 'View all scheduled posts',
    icon: Calendar,
    href: '/dashboard/scheduler',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    label: 'Trends',
    description: 'Explore trending content',
    icon: TrendingUp,
    href: '/dashboard/trends',
    color: 'from-green-500 to-emerald-500',
  },
  {
    label: 'Analytics',
    description: 'View performance stats',
    icon: BarChart3,
    href: '/dashboard/analytics',
    color: 'from-orange-500 to-amber-500',
  },
  {
    label: 'Settings',
    description: 'Manage your account',
    icon: Settings,
    href: '/dashboard/settings',
    color: 'from-gray-500 to-gray-600',
  },
];

export function QuickActions() {
  return (
    <div className="glass rounded-2xl p-6">
      <h2 className="text-lg font-semibold text-white mb-4">Quick Actions</h2>
      <div className="grid grid-cols-2 gap-3">
        {actions.map((action) => (
          <Link
            key={action.label}
            href={action.href}
            className="group p-4 rounded-xl bg-white/5 hover:bg-white/10 transition border border-transparent hover:border-white/10"
          >
            <div className={`w-10 h-10 rounded-lg bg-gradient-to-r ${action.color} flex items-center justify-center mb-3`}>
              <action.icon className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-white font-medium text-sm">{action.label}</h3>
            <p className="text-gray-500 text-xs mt-1">{action.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
