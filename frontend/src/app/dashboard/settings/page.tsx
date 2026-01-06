'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import {
  Settings,
  Link2,
  User,
  Bell,
  Shield,
  CreditCard,
  Palette,
  Globe,
  ChevronRight,
} from 'lucide-react';
import { DashboardLayout } from '@/components/layout/dashboard-layout';

interface SettingsCardProps {
  href: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  badge?: string;
  color: string;
}

const SettingsCard: React.FC<SettingsCardProps> = ({
  href,
  icon,
  title,
  description,
  badge,
  color,
}) => {
  return (
    <Link href={href}>
      <motion.div
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className="p-6 bg-gray-800/50 rounded-2xl border border-gray-700/50 hover:border-gray-600/50 transition-all cursor-pointer group"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div
              className={`w-12 h-12 rounded-xl flex items-center justify-center ${color}`}
            >
              {icon}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-semibold text-white">{title}</h3>
                {badge && (
                  <span className="px-2 py-0.5 text-xs font-medium bg-purple-500/20 text-purple-400 rounded-full">
                    {badge}
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-400">{description}</p>
            </div>
          </div>
          <ChevronRight className="w-5 h-5 text-gray-500 group-hover:text-white transition" />
        </div>
      </motion.div>
    </Link>
  );
};

export default function SettingsPage() {
  const settingsSections = [
    {
      title: 'Account',
      items: [
        {
          href: '/dashboard/settings/platforms',
          icon: <Link2 className="w-6 h-6 text-purple-400" />,
          title: 'Connected Platforms',
          description: 'Manage TikTok, Instagram, and YouTube connections',
          badge: 'New',
          color: 'bg-purple-500/20',
        },
        {
          href: '/dashboard/settings/profile',
          icon: <User className="w-6 h-6 text-blue-400" />,
          title: 'Profile',
          description: 'Update your account information and preferences',
          color: 'bg-blue-500/20',
        },
        {
          href: '/dashboard/settings/notifications',
          icon: <Bell className="w-6 h-6 text-yellow-400" />,
          title: 'Notifications',
          description: 'Configure email and push notification settings',
          color: 'bg-yellow-500/20',
        },
      ],
    },
    {
      title: 'Preferences',
      items: [
        {
          href: '/dashboard/settings/appearance',
          icon: <Palette className="w-6 h-6 text-pink-400" />,
          title: 'Appearance',
          description: 'Customize theme, colors, and display options',
          color: 'bg-pink-500/20',
        },
        {
          href: '/dashboard/settings/language',
          icon: <Globe className="w-6 h-6 text-cyan-400" />,
          title: 'Language & Region',
          description: 'Set your language, timezone, and date format',
          color: 'bg-cyan-500/20',
        },
      ],
    },
    {
      title: 'Security & Billing',
      items: [
        {
          href: '/dashboard/settings/security',
          icon: <Shield className="w-6 h-6 text-green-400" />,
          title: 'Security',
          description: 'Manage password, two-factor auth, and sessions',
          color: 'bg-green-500/20',
        },
        {
          href: '/dashboard/settings/billing',
          icon: <CreditCard className="w-6 h-6 text-orange-400" />,
          title: 'Billing & Plans',
          description: 'View your subscription and payment history',
          color: 'bg-orange-500/20',
        },
      ],
    },
  ];

  return (
    <DashboardLayout>
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Settings className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">Settings</h1>
              <p className="text-gray-400">
                Manage your account settings and preferences
              </p>
            </div>
          </div>
        </div>

        {/* Settings Sections */}
        <div className="space-y-8">
          {settingsSections.map((section, sectionIndex) => (
            <motion.div
              key={section.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: sectionIndex * 0.1 }}
            >
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
                {section.title}
              </h2>
              <div className="space-y-3">
                {section.items.map((item, itemIndex) => (
                  <motion.div
                    key={item.href}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: sectionIndex * 0.1 + itemIndex * 0.05 }}
                  >
                    <SettingsCard {...item} />
                  </motion.div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-12 pt-8 border-t border-gray-700/50 text-center"
        >
          <p className="text-sm text-gray-500">
            Viralify v1.0.0 - Made with love for content creators
          </p>
        </motion.div>
      </div>
    </DashboardLayout>
  );
}
