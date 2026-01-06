'use client';

import { ReactNode } from 'react';
import { ArrowUpRight, ArrowDownRight } from 'lucide-react';

interface StatsCardProps {
  title: string;
  value: string | number;
  change?: number;
  subtitle?: string;
  icon: ReactNode;
  color: string;
}

export function StatsCard({ title, value, change, subtitle, icon, color }: StatsCardProps) {
  const formatValue = (val: string | number) => {
    if (typeof val === 'number') {
      if (val >= 1000000) {
        return `${(val / 1000000).toFixed(1)}M`;
      }
      if (val >= 1000) {
        return `${(val / 1000).toFixed(1)}K`;
      }
      return val.toString();
    }
    return val;
  };

  return (
    <div className="glass rounded-2xl p-5 card-glow">
      <div className="flex items-start justify-between">
        <div className={`p-3 rounded-xl bg-gradient-to-r ${color}`}>
          {icon}
        </div>
        {change !== undefined && (
          <div className={`flex items-center gap-1 text-sm ${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {change >= 0 ? <ArrowUpRight className="w-4 h-4" /> : <ArrowDownRight className="w-4 h-4" />}
            <span>{Math.abs(change)}%</span>
          </div>
        )}
      </div>
      <div className="mt-4">
        <p className="text-gray-400 text-sm">{title}</p>
        <p className="text-2xl font-bold text-white mt-1">
          {formatValue(value)}
          {subtitle && <span className="text-sm font-normal text-gray-400 ml-1">{subtitle}</span>}
        </p>
      </div>
    </div>
  );
}
