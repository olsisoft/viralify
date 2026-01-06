'use client';
/* eslint-disable @next/next/no-img-element */

import { useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import {
  LayoutDashboard,
  Calendar,
  TrendingUp,
  BarChart3,
  MessageSquare,
  PlusCircle,
  Settings,
  LogOut,
  ChevronLeft,
  ChevronRight,
  Video,
  Menu,
  X,
  Bell,
  User,
  Link2,
  Loader2,
  Wand2,
  Trophy,
  Users,
} from 'lucide-react';
import { useAuth } from '@/contexts/auth-context';
import toast from 'react-hot-toast';

interface NavItem {
  href: string;
  icon: React.ReactNode;
  label: string;
  badge?: string;
}

interface DashboardLayoutProps {
  children: React.ReactNode;
}

const navItems: NavItem[] = [
  {
    href: '/dashboard',
    icon: <LayoutDashboard className="w-5 h-5" />,
    label: 'Dashboard',
  },
  {
    href: '/dashboard/create',
    icon: <PlusCircle className="w-5 h-5" />,
    label: 'Create',
  },
  {
    href: '/dashboard/studio',
    icon: <Wand2 className="w-5 h-5" />,
    label: 'Studio',
    badge: 'New',
  },
  {
    href: '/dashboard/coaching',
    icon: <Trophy className="w-5 h-5" />,
    label: 'Coaching',
    badge: 'New',
  },
  {
    href: '/dashboard/scheduler',
    icon: <Calendar className="w-5 h-5" />,
    label: 'Scheduler',
  },
  {
    href: '/dashboard/trends',
    icon: <TrendingUp className="w-5 h-5" />,
    label: 'Trends',
  },
  {
    href: '/dashboard/analytics',
    icon: <BarChart3 className="w-5 h-5" />,
    label: 'Analytics',
  },
  {
    href: '/dashboard/ai-chat',
    icon: <MessageSquare className="w-5 h-5" />,
    label: 'AI Chat',
  },
];

const bottomNavItems: NavItem[] = [
  {
    href: '/dashboard/settings/profiles',
    icon: <Users className="w-5 h-5" />,
    label: 'Creator Profiles',
    badge: 'New',
  },
  {
    href: '/dashboard/settings',
    icon: <Settings className="w-5 h-5" />,
    label: 'Settings',
  },
  {
    href: '/dashboard/settings/platforms',
    icon: <Link2 className="w-5 h-5" />,
    label: 'Platforms',
  },
];

export function DashboardLayout({ children }: DashboardLayoutProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const pathname = usePathname();
  const { user, logout } = useAuth();

  const isActive = (href: string) => {
    if (href === '/dashboard') {
      return pathname === href;
    }
    return pathname.startsWith(href);
  };

  const handleLogout = async () => {
    setIsLoggingOut(true);
    try {
      // Small delay for UX
      await new Promise(resolve => setTimeout(resolve, 300));
      toast.success('Logged out successfully');
      logout();
    } catch (error) {
      toast.error('Logout failed');
      setIsLoggingOut(false);
    }
  };

  const NavLink = ({ item, collapsed = false }: { item: NavItem; collapsed?: boolean }) => {
    const active = isActive(item.href);

    return (
      <Link
        href={item.href}
        className={`
          relative flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200
          ${active
            ? 'bg-gradient-to-r from-[#fe2c55]/20 to-[#25f4ee]/20 text-white'
            : 'text-gray-400 hover:text-white hover:bg-white/5'
          }
          ${collapsed ? 'justify-center' : ''}
        `}
        onClick={() => setIsMobileMenuOpen(false)}
      >
        {active && (
          <motion.div
            layoutId="activeNav"
            className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-gradient-to-b from-[#fe2c55] to-[#25f4ee] rounded-full"
          />
        )}
        <span className={active ? 'text-[#fe2c55]' : ''}>{item.icon}</span>
        {!collapsed && (
          <span className="font-medium">{item.label}</span>
        )}
        {!collapsed && item.badge && (
          <span className="ml-auto px-2 py-0.5 text-xs font-medium bg-purple-500/20 text-purple-400 rounded-full">
            {item.badge}
          </span>
        )}
      </Link>
    );
  };

  const LogoutButton = ({ collapsed = false }: { collapsed?: boolean }) => (
    <button
      onClick={handleLogout}
      disabled={isLoggingOut}
      className={`
        w-full flex items-center gap-3 px-3 py-2.5 text-red-400 hover:bg-red-500/10 rounded-xl transition
        disabled:opacity-50 disabled:cursor-not-allowed
        ${collapsed ? 'justify-center' : ''}
      `}
    >
      {isLoggingOut ? (
        <Loader2 className="w-5 h-5 animate-spin" />
      ) : (
        <LogOut className="w-5 h-5" />
      )}
      {!collapsed && <span className="font-medium">{isLoggingOut ? 'Logging out...' : 'Logout'}</span>}
    </button>
  );

  // Get user initials for avatar
  const getUserInitials = () => {
    if (!user?.name) return 'U';
    return user.name
      .split(' ')
      .map(n => n[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-gray-800">
      {/* Mobile Header */}
      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 bg-gray-900/95 backdrop-blur-lg border-b border-white/10">
        <div className="flex items-center justify-between px-4 py-3">
          <Link href="/dashboard" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#fe2c55] to-[#25f4ee] flex items-center justify-center">
              <Video className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white">Viralify</span>
          </Link>

          <div className="flex items-center gap-2">
            <button className="p-2 text-gray-400 hover:text-white transition relative">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-[#fe2c55] rounded-full" />
            </button>
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-2 text-gray-400 hover:text-white transition"
            >
              {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="lg:hidden fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
            onClick={() => setIsMobileMenuOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Mobile Sidebar */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.aside
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="lg:hidden fixed left-0 top-0 bottom-0 w-[280px] z-50 bg-gray-900 border-r border-white/10 pt-16"
          >
            <div className="flex flex-col h-full p-4">
              {/* User Info - Mobile */}
              {user && (
                <div className="mb-4 p-3 bg-white/5 rounded-xl">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                      {user.avatar ? (
                        <img src={user.avatar} alt={user.name} className="w-10 h-10 rounded-full" />
                      ) : (
                        <span className="text-sm font-bold text-white">{getUserInitials()}</span>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-white truncate">{user.name}</p>
                      <p className="text-xs text-gray-400 truncate">{user.email}</p>
                    </div>
                  </div>
                </div>
              )}

              <nav className="flex-1 space-y-1">
                {navItems.map((item) => (
                  <NavLink key={item.href} item={item} />
                ))}
              </nav>

              <div className="border-t border-white/10 pt-4 space-y-1">
                {bottomNavItems.map((item) => (
                  <NavLink key={item.href} item={item} />
                ))}
                <LogoutButton />
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* Desktop Sidebar */}
      <aside
        className={`
          hidden lg:flex fixed left-0 top-0 bottom-0 z-40 flex-col
          bg-gray-900/50 backdrop-blur-xl border-r border-white/10
          transition-all duration-300
          ${isCollapsed ? 'w-[72px]' : 'w-[260px]'}
        `}
      >
        {/* Logo */}
        <div className={`flex items-center h-16 px-4 ${isCollapsed ? 'justify-center' : 'gap-3'}`}>
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#fe2c55] to-[#25f4ee] flex items-center justify-center flex-shrink-0">
            <Video className="w-6 h-6 text-white" />
          </div>
          {!isCollapsed && (
            <span className="text-xl font-bold text-white">Viralify</span>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink key={item.href} item={item} collapsed={isCollapsed} />
          ))}
        </nav>

        {/* Bottom Navigation */}
        <div className="px-3 py-4 border-t border-white/10 space-y-1">
          {bottomNavItems.map((item) => (
            <NavLink key={item.href} item={item} collapsed={isCollapsed} />
          ))}
          <LogoutButton collapsed={isCollapsed} />
        </div>

        {/* Collapse Toggle */}
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="absolute -right-3 top-20 w-6 h-6 bg-gray-800 border border-white/10 rounded-full flex items-center justify-center text-gray-400 hover:text-white transition"
        >
          {isCollapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </aside>

      {/* Main Content */}
      <main
        className={`
          pt-16 lg:pt-0 min-h-screen transition-all duration-300
          ${isCollapsed ? 'lg:pl-[72px]' : 'lg:pl-[260px]'}
        `}
      >
        {/* Desktop Header */}
        <header className="hidden lg:flex items-center justify-between h-16 px-6 border-b border-white/10">
          <div>
            {/* Breadcrumb or page title could go here */}
          </div>

          <div className="flex items-center gap-4">
            <button className="p-2 text-gray-400 hover:text-white transition relative">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-[#fe2c55] rounded-full" />
            </button>

            <Link
              href="/dashboard/settings"
              className="flex items-center gap-3 px-3 py-1.5 rounded-xl hover:bg-white/5 transition"
            >
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center overflow-hidden">
                {user?.avatar ? (
                  <img src={user.avatar} alt={user?.name} className="w-8 h-8 rounded-full object-cover" />
                ) : (
                  <span className="text-xs font-bold text-white">{getUserInitials()}</span>
                )}
              </div>
              <div className="text-left">
                <p className="text-sm font-medium text-white">{user?.name || 'User'}</p>
                <p className="text-xs text-gray-400">{user?.plan || 'Free'} Plan</p>
              </div>
            </Link>
          </div>
        </header>

        {/* Page Content */}
        <div className="p-4 lg:p-6">
          {children}
        </div>
      </main>
    </div>
  );
}

export default DashboardLayout;
