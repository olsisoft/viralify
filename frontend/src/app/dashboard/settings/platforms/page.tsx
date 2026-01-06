'use client';
/* eslint-disable @next/next/no-img-element */

import { useState, useEffect } from 'react';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Video,
  Instagram,
  Youtube,
  Link2,
  Unlink,
  Check,
  AlertCircle,
  ExternalLink,
  RefreshCw,
  Shield,
  Users,
  Clock,
  ChevronRight,
  Settings,
  Info,
} from 'lucide-react';
import toast from 'react-hot-toast';
import {
  PlatformType,
  PlatformAccount,
  PLATFORMS,
  platformService,
} from '@/services/api';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { DEMO_MODE } from '@/lib/demo-mode';

interface PlatformCardProps {
  platform: PlatformType;
  account: PlatformAccount | null;
  onConnect: () => void;
  onDisconnect: () => void;
  isConnecting: boolean;
}

const PlatformIcon: React.FC<{ platform: PlatformType; className?: string }> = ({
  platform,
  className = 'w-8 h-8',
}) => {
  switch (platform) {
    case 'TIKTOK':
      return <Video className={className} />;
    case 'INSTAGRAM':
      return <Instagram className={className} />;
    case 'YOUTUBE':
      return <Youtube className={className} />;
    default:
      return <Video className={className} />;
  }
};

const StatusBadge: React.FC<{ status: PlatformAccount['accountStatus'] }> = ({ status }) => {
  const statusConfig = {
    active: {
      color: 'bg-green-500/20 text-green-400 border-green-500/30',
      icon: <Check className="w-3 h-3" />,
      label: 'Connected',
    },
    expired: {
      color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
      icon: <Clock className="w-3 h-3" />,
      label: 'Token Expired',
    },
    revoked: {
      color: 'bg-red-500/20 text-red-400 border-red-500/30',
      icon: <AlertCircle className="w-3 h-3" />,
      label: 'Access Revoked',
    },
    error: {
      color: 'bg-red-500/20 text-red-400 border-red-500/30',
      icon: <AlertCircle className="w-3 h-3" />,
      label: 'Error',
    },
  };

  const config = statusConfig[status];

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded-full border ${config.color}`}
    >
      {config.icon}
      {config.label}
    </span>
  );
};

const PlatformCard: React.FC<PlatformCardProps> = ({
  platform,
  account,
  onConnect,
  onDisconnect,
  isConnecting,
}) => {
  const platformInfo = PLATFORMS[platform];
  const isConnected = account !== null;
  const needsReauth = account?.accountStatus === 'expired' || account?.accountStatus === 'revoked';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`
        relative overflow-hidden rounded-2xl border transition-all duration-300
        ${isConnected
          ? 'bg-gradient-to-br from-gray-800/80 to-gray-900/80 border-gray-700/50'
          : 'bg-gray-800/30 border-gray-700/30 hover:border-gray-600/50'
        }
      `}
    >
      {/* Platform header */}
      <div className="p-6">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-4">
            <div
              className="w-14 h-14 rounded-xl flex items-center justify-center"
              style={{ backgroundColor: `${platformInfo.color}20` }}
            >
              <PlatformIcon
                platform={platform}
                className="w-7 h-7"
              />
              <style jsx>{`
                :global(svg) {
                  color: ${platformInfo.color};
                }
              `}</style>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">
                {platformInfo.displayName}
              </h3>
              <p className="text-sm text-gray-400">
                {isConnected
                  ? `@${account.platformUsername || account.platformDisplayName}`
                  : 'Not connected'}
              </p>
            </div>
          </div>

          {isConnected && <StatusBadge status={account.accountStatus} />}
        </div>

        {/* Account details when connected */}
        {isConnected && account && (
          <div className="mt-6 grid grid-cols-2 gap-4">
            {account.platformAvatarUrl && (
              <div className="col-span-2 flex items-center gap-3 p-3 bg-gray-700/30 rounded-xl">
                <img
                  src={account.platformAvatarUrl}
                  alt={account.platformDisplayName || account.platformUsername}
                  className="w-10 h-10 rounded-full object-cover"
                />
                <div>
                  <p className="text-sm font-medium text-white">
                    {account.platformDisplayName || account.platformUsername}
                  </p>
                  <p className="text-xs text-gray-400">
                    @{account.platformUsername}
                  </p>
                </div>
              </div>
            )}

            <div className="p-3 bg-gray-700/30 rounded-xl">
              <div className="flex items-center gap-2 text-gray-400 mb-1">
                <Users className="w-4 h-4" />
                <span className="text-xs">Followers</span>
              </div>
              <p className="text-lg font-semibold text-white">
                {account.followerCount?.toLocaleString() || 'N/A'}
              </p>
            </div>

            <div className="p-3 bg-gray-700/30 rounded-xl">
              <div className="flex items-center gap-2 text-gray-400 mb-1">
                <Clock className="w-4 h-4" />
                <span className="text-xs">Last Sync</span>
              </div>
              <p className="text-sm font-medium text-white">
                {account.lastSyncAt
                  ? new Date(account.lastSyncAt).toLocaleDateString()
                  : 'Never'}
              </p>
            </div>
          </div>
        )}

        {/* Platform limits info */}
        <div className="mt-4 p-3 bg-gray-700/20 rounded-xl">
          <p className="text-xs text-gray-400 mb-2">Platform Limits</p>
          <div className="flex flex-wrap gap-2">
            <span className="px-2 py-1 text-xs bg-gray-600/30 text-gray-300 rounded-lg">
              Max {platformInfo.maxDurationSeconds}s video
            </span>
            <span className="px-2 py-1 text-xs bg-gray-600/30 text-gray-300 rounded-lg">
              {platformInfo.maxCaptionLength} chars caption
            </span>
            <span className="px-2 py-1 text-xs bg-gray-600/30 text-gray-300 rounded-lg">
              {platformInfo.maxHashtags} hashtags
            </span>
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div className="px-6 py-4 bg-gray-900/50 border-t border-gray-700/50">
        {isConnected ? (
          <div className="flex items-center justify-between">
            {needsReauth ? (
              <button
                onClick={onConnect}
                disabled={isConnecting}
                className="flex items-center gap-2 px-4 py-2 bg-yellow-500/20 text-yellow-400 rounded-xl hover:bg-yellow-500/30 transition disabled:opacity-50"
              >
                {isConnecting ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <RefreshCw className="w-4 h-4" />
                )}
                Reconnect
              </button>
            ) : (
              <a
                href={`https://${platform.toLowerCase()}.com/${account?.platformUsername || ''}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition"
              >
                <ExternalLink className="w-4 h-4" />
                View Profile
              </a>
            )}

            <button
              onClick={onDisconnect}
              disabled={isConnecting}
              className="flex items-center gap-2 px-4 py-2 text-red-400 hover:bg-red-500/10 rounded-xl transition disabled:opacity-50"
            >
              <Unlink className="w-4 h-4" />
              Disconnect
            </button>
          </div>
        ) : (
          <button
            onClick={onConnect}
            disabled={isConnecting}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-white font-medium transition disabled:opacity-50"
            style={{ backgroundColor: platformInfo.color }}
          >
            {isConnecting ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Connecting...
              </>
            ) : (
              <>
                <Link2 className="w-4 h-4" />
                Connect {platformInfo.displayName}
              </>
            )}
          </button>
        )}
      </div>
    </motion.div>
  );
};

export default function PlatformsSettingsPage() {
  const [accounts, setAccounts] = useState<PlatformAccount[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [connectingPlatform, setConnectingPlatform] = useState<PlatformType | null>(null);

  const platforms: PlatformType[] = ['TIKTOK', 'INSTAGRAM', 'YOUTUBE'];

  useEffect(() => {
    fetchAccounts();
  }, []);

  const fetchAccounts = async () => {
    try {
      setIsLoading(true);
      const data = await platformService.getConnectedAccounts();
      setAccounts(data);
    } catch (error) {
      console.error('Failed to fetch accounts:', error);
      toast.error('Failed to load connected accounts');
    } finally {
      setIsLoading(false);
    }
  };

  const handleConnect = async (platform: PlatformType) => {
    try {
      setConnectingPlatform(platform);
      const { url } = await platformService.getOAuthUrl(platform);

      // Open OAuth URL in new window or redirect
      const width = 600;
      const height = 700;
      const left = window.screenX + (window.outerWidth - width) / 2;
      const top = window.screenY + (window.outerHeight - height) / 2;

      const popup = window.open(
        url,
        `Connect ${platform}`,
        `width=${width},height=${height},left=${left},top=${top},toolbar=no,menubar=no`
      );

      // Poll for popup close and refresh accounts
      const pollTimer = setInterval(() => {
        if (popup?.closed) {
          clearInterval(pollTimer);
          setConnectingPlatform(null);
          fetchAccounts();
          toast.success(`${PLATFORMS[platform].displayName} connection updated`);
        }
      }, 1000);

      // Fallback timeout
      setTimeout(() => {
        clearInterval(pollTimer);
        setConnectingPlatform(null);
      }, 300000); // 5 minutes timeout

    } catch (error) {
      console.error('Failed to get OAuth URL:', error);
      toast.error(`Failed to connect ${PLATFORMS[platform].displayName}`);
      setConnectingPlatform(null);
    }
  };

  const handleDisconnect = async (platform: PlatformType) => {
    const platformInfo = PLATFORMS[platform];

    if (!confirm(`Are you sure you want to disconnect ${platformInfo.displayName}? You won't be able to post to this platform until you reconnect.`)) {
      return;
    }

    try {
      setConnectingPlatform(platform);
      await platformService.disconnectPlatform(platform);
      setAccounts((prev) => prev.filter((a) => a.platform !== platform));
      toast.success(`${platformInfo.displayName} disconnected successfully`);
    } catch (error) {
      console.error('Failed to disconnect:', error);
      toast.error(`Failed to disconnect ${platformInfo.displayName}`);
    } finally {
      setConnectingPlatform(null);
    }
  };

  const getAccountForPlatform = (platform: PlatformType): PlatformAccount | null => {
    return accounts.find((a) => a.platform === platform) || null;
  };

  const connectedCount = accounts.filter((a) => a.accountStatus === 'active').length;

  return (
    <DashboardLayout>
      <div className="max-w-5xl mx-auto">
        {/* Demo Mode Banner */}
        {DEMO_MODE && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-yellow-500/20 border border-yellow-500/30 rounded-2xl flex items-center gap-3"
          >
            <Info className="w-5 h-5 text-yellow-400 flex-shrink-0" />
            <div>
              <p className="text-sm text-yellow-200">
                <strong>Demo Mode:</strong> Platform connections are simulated. In production, clicking &quot;Connect&quot; will open the official OAuth flow.
              </p>
            </div>
          </motion.div>
        )}

        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                <Settings className="w-8 h-8 text-purple-500" />
                Connected Platforms
              </h1>
              <p className="text-gray-400 mt-2">
                Manage your social media connections for cross-platform posting
              </p>
            </div>

            <div className="text-right">
              <p className="text-2xl font-bold text-white">{connectedCount}/{platforms.length}</p>
              <p className="text-sm text-gray-400">Platforms connected</p>
            </div>
          </div>
        </div>

        {/* Info Banner */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 p-4 bg-purple-500/10 border border-purple-500/20 rounded-2xl"
        >
          <div className="flex items-start gap-3">
            <Shield className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-purple-300 font-medium">Secure OAuth Connection</p>
              <p className="text-sm text-gray-400 mt-1">
                We use official OAuth 2.0 to connect to your accounts. We never store your passwords
                and you can revoke access at any time from your platform settings.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Loading State */}
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <RefreshCw className="w-8 h-8 text-purple-500 animate-spin" />
          </div>
        ) : (
          <>
            {/* Platform Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {platforms.map((platform, index) => (
                <motion.div
                  key={platform}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <PlatformCard
                    platform={platform}
                    account={getAccountForPlatform(platform)}
                    onConnect={() => handleConnect(platform)}
                    onDisconnect={() => handleDisconnect(platform)}
                    isConnecting={connectingPlatform === platform}
                  />
                </motion.div>
              ))}
            </div>

            {/* Additional Settings */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="mt-8 p-6 bg-gray-800/50 rounded-2xl border border-gray-700/50"
            >
              <h3 className="text-lg font-semibold text-white mb-4">Cross-Platform Settings</h3>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-gray-700/30 rounded-xl">
                  <div>
                    <p className="text-sm font-medium text-white">Auto-adapt content</p>
                    <p className="text-xs text-gray-400">
                      Automatically adjust captions and hashtags for each platform&apos;s limits
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" className="sr-only peer" defaultChecked />
                    <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-500"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between p-4 bg-gray-700/30 rounded-xl">
                  <div>
                    <p className="text-sm font-medium text-white">Unified scheduling</p>
                    <p className="text-xs text-gray-400">
                      Post to all platforms at the same scheduled time
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" className="sr-only peer" defaultChecked />
                    <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-500"></div>
                  </label>
                </div>

                <div className="flex items-center justify-between p-4 bg-gray-700/30 rounded-xl">
                  <div>
                    <p className="text-sm font-medium text-white">Sync analytics</p>
                    <p className="text-xs text-gray-400">
                      Collect performance metrics from all connected platforms
                    </p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" className="sr-only peer" defaultChecked />
                    <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-500"></div>
                  </label>
                </div>
              </div>
            </motion.div>

            {/* Help Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-8 p-6 bg-gray-800/30 rounded-2xl border border-gray-700/30"
            >
              <h3 className="text-lg font-semibold text-white mb-4">Need Help?</h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <a
                  href="#"
                  className="flex items-center justify-between p-4 bg-gray-700/20 rounded-xl hover:bg-gray-700/40 transition group"
                >
                  <div>
                    <p className="text-sm font-medium text-white">Connection Issues</p>
                    <p className="text-xs text-gray-400">Troubleshoot OAuth problems</p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-white transition" />
                </a>

                <a
                  href="#"
                  className="flex items-center justify-between p-4 bg-gray-700/20 rounded-xl hover:bg-gray-700/40 transition group"
                >
                  <div>
                    <p className="text-sm font-medium text-white">Platform Requirements</p>
                    <p className="text-xs text-gray-400">API access & permissions</p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-white transition" />
                </a>

                <a
                  href="#"
                  className="flex items-center justify-between p-4 bg-gray-700/20 rounded-xl hover:bg-gray-700/40 transition group"
                >
                  <div>
                    <p className="text-sm font-medium text-white">Content Guidelines</p>
                    <p className="text-xs text-gray-400">Best practices per platform</p>
                  </div>
                  <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-white transition" />
                </a>
              </div>
            </motion.div>
          </>
        )}
      </div>
    </DashboardLayout>
  );
}
