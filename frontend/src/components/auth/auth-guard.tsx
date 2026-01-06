'use client';

import { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/contexts/auth-context';
import { motion } from 'framer-motion';
import { Loader2, Video } from 'lucide-react';

interface AuthGuardProps {
  children: React.ReactNode;
}

// Routes that don't require authentication
const publicRoutes = [
  '/',
  '/auth/login',
  '/auth/register',
  '/auth/forgot-password',
  '/auth/reset-password',
  '/pricing',
  '/features',
  '/about',
  '/contact',
  '/privacy',
  '/terms',
];

// Routes that should redirect to dashboard if already authenticated
const authRoutes = [
  '/auth/login',
  '/auth/register',
];

export function AuthGuard({ children }: AuthGuardProps) {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  const isPublicRoute = publicRoutes.some(route =>
    pathname === route || pathname.startsWith('/auth/')
  );

  const isAuthRoute = authRoutes.includes(pathname);
  const isDashboardRoute = pathname.startsWith('/dashboard');

  useEffect(() => {
    if (isLoading) return;

    // Redirect authenticated users away from auth pages
    if (isAuthenticated && isAuthRoute) {
      router.replace('/dashboard');
      return;
    }

    // Redirect unauthenticated users to login from protected routes
    if (!isAuthenticated && isDashboardRoute) {
      router.replace('/auth/login');
      return;
    }
  }, [isAuthenticated, isLoading, isAuthRoute, isDashboardRoute, router]);

  // Show loading state while checking auth
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-gray-800 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex flex-col items-center gap-4"
        >
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#fe2c55] to-[#25f4ee] flex items-center justify-center">
            <Video className="w-8 h-8 text-white" />
          </div>
          <div className="flex items-center gap-2 text-white">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="text-lg font-medium">Loading...</span>
          </div>
        </motion.div>
      </div>
    );
  }

  // Prevent flash of protected content for unauthenticated users
  if (!isAuthenticated && isDashboardRoute) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-gray-800 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-4"
        >
          <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
          <span className="text-gray-400">Redirecting to login...</span>
        </motion.div>
      </div>
    );
  }

  // Prevent flash of auth pages for authenticated users
  if (isAuthenticated && isAuthRoute) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-gray-800 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-col items-center gap-4"
        >
          <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
          <span className="text-gray-400">Redirecting to dashboard...</span>
        </motion.div>
      </div>
    );
  }

  return <>{children}</>;
}

// HOC for protecting individual pages (alternative approach)
export function withAuth<P extends object>(Component: React.ComponentType<P>) {
  return function ProtectedComponent(props: P) {
    const { isAuthenticated, isLoading } = useAuth();
    const router = useRouter();

    useEffect(() => {
      if (!isLoading && !isAuthenticated) {
        router.replace('/auth/login');
      }
    }, [isAuthenticated, isLoading, router]);

    if (isLoading || !isAuthenticated) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-gray-800 flex items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-purple-500" />
        </div>
      );
    }

    return <Component {...props} />;
  };
}
