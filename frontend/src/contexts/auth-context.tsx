'use client';

import { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { DEMO_MODE, DEMO_USER, DEMO_TOKENS } from '@/lib/demo-mode';

interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  plan?: string;
  createdAt?: string;
}

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (name: string, email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
  updateUser: (userData: Partial<User>) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();


  const checkAuth = useCallback(async () => {
    try {
      const storedUser = localStorage.getItem('user');
      const accessToken = localStorage.getItem('accessToken');

      if (storedUser && accessToken) {
        const userData = JSON.parse(storedUser);
        setUser(userData);

        // In non-demo mode, validate token with backend
        if (!DEMO_MODE) {
          try {
            const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/auth/me`, {
              headers: {
                'Authorization': `Bearer ${accessToken}`
              }
            });

            if (response.ok) {
              const data = await response.json();
              setUser(data.user);
              localStorage.setItem('user', JSON.stringify(data.user));
            } else if (response.status === 401) {
              // Token expired, try to refresh
              const refreshed = await refreshToken();
              if (!refreshed) {
                clearAuth();
              }
            }
          } catch (error) {
            console.error('Auth validation error:', error);
            // Keep user logged in if network error in demo mode
            if (!DEMO_MODE) {
              clearAuth();
            }
          }
        }
      }
    } catch (error) {
      console.error('Auth check error:', error);
      clearAuth();
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Check for existing session on mount
  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  const refreshToken = async (): Promise<boolean> => {
    try {
      const refreshTokenValue = localStorage.getItem('refreshToken');
      if (!refreshTokenValue) return false;

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refreshToken: refreshTokenValue })
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('accessToken', data.accessToken);
        if (data.refreshToken) {
          localStorage.setItem('refreshToken', data.refreshToken);
        }
        return true;
      }
      return false;
    } catch (error) {
      console.error('Token refresh error:', error);
      return false;
    }
  };

  const clearAuth = () => {
    localStorage.removeItem('accessToken');
    localStorage.removeItem('refreshToken');
    localStorage.removeItem('user');
    setUser(null);
  };

  const login = async (email: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      if (DEMO_MODE) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));

        if (email && password.length >= 4) {
          const userData = { ...DEMO_USER, email };
          localStorage.setItem('accessToken', DEMO_TOKENS.accessToken);
          localStorage.setItem('refreshToken', DEMO_TOKENS.refreshToken);
          localStorage.setItem('user', JSON.stringify(userData));
          setUser(userData);
          return { success: true };
        }
        return { success: false, error: 'Invalid credentials' };
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      const data = await response.json();

      if (response.ok) {
        // Map API user fields to frontend User interface
        const userData = {
          id: data.user.id,
          email: data.user.email,
          name: data.user.fullName || data.user.email.split('@')[0], // Fallback to email prefix
          avatar: data.user.avatarUrl,
          plan: data.user.planType,
          createdAt: data.user.createdAt,
        };
        localStorage.setItem('accessToken', data.accessToken);
        localStorage.setItem('refreshToken', data.refreshToken);
        localStorage.setItem('user', JSON.stringify(userData));
        setUser(userData);
        return { success: true };
      }

      return { success: false, error: data.message || 'Login failed' };
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, error: 'Connection error. Please try again.' };
    }
  };

  const register = async (name: string, email: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      if (DEMO_MODE) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));

        if (name && email && password.length >= 8) {
          const userData = { ...DEMO_USER, name, email };
          localStorage.setItem('accessToken', DEMO_TOKENS.accessToken);
          localStorage.setItem('refreshToken', DEMO_TOKENS.refreshToken);
          localStorage.setItem('user', JSON.stringify(userData));
          setUser(userData);
          return { success: true };
        }
        return { success: false, error: 'Password must be at least 8 characters' };
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password })
      });

      const data = await response.json();

      if (response.ok) {
        // Auto-login after registration
        if (data.accessToken) {
          // Map API user fields to frontend User interface
          const userData = {
            id: data.user.id,
            email: data.user.email,
            name: data.user.fullName || name, // Use registration name if fullName is null
            avatar: data.user.avatarUrl,
            plan: data.user.planType,
            createdAt: data.user.createdAt,
          };
          localStorage.setItem('accessToken', data.accessToken);
          localStorage.setItem('refreshToken', data.refreshToken);
          localStorage.setItem('user', JSON.stringify(userData));
          setUser(userData);
        }
        return { success: true };
      }

      return { success: false, error: data.message || 'Registration failed' };
    } catch (error) {
      console.error('Register error:', error);
      return { success: false, error: 'Connection error. Please try again.' };
    }
  };

  const logout = useCallback(() => {
    clearAuth();
    router.push('/auth/login');
  }, [router]);

  const updateUser = (userData: Partial<User>) => {
    if (user) {
      const updatedUser = { ...user, ...userData };
      setUser(updatedUser);
      localStorage.setItem('user', JSON.stringify(updatedUser));
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
        updateUser
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
