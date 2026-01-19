'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  FileVideo,
  Download,
  RefreshCw,
  GraduationCap,
} from 'lucide-react';
import { api } from '@/lib/api';
import type { CourseJob } from '../lib/course-types';

interface CourseHistoryProps {
  onSelectJob?: (job: CourseJob) => void;
  refreshTrigger?: number;
}

export function CourseHistory({ onSelectJob, refreshTrigger = 0 }: CourseHistoryProps) {
  const router = useRouter();
  const [jobs, setJobs] = useState<CourseJob[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const handlePractice = (jobId: string) => {
    router.push(`/dashboard/studio/practice?courseId=${jobId}`);
  };

  const fetchJobs = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await api.courses.listJobs(10);
      setJobs(result as CourseJob[]);
    } catch (err: any) {
      setError(err.message || 'Failed to load history');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
  }, [refreshTrigger]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="w-4 h-4 text-green-400" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />;
      case 'processing':
      case 'queued':
        return <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-400';
      case 'failed':
        return 'text-red-400';
      case 'processing':
        return 'text-purple-400';
      default:
        return 'text-gray-400';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-400 text-sm mb-2">{error}</p>
        <button
          onClick={fetchJobs}
          className="text-purple-400 hover:text-purple-300 text-sm flex items-center gap-1 mx-auto"
        >
          <RefreshCw className="w-4 h-4" />
          Retry
        </button>
      </div>
    );
  }

  if (jobs.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <FileVideo className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-sm">No courses generated yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-gray-400">Recent Courses</h4>
        <button
          onClick={fetchJobs}
          className="text-gray-500 hover:text-gray-300 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {jobs.map((job) => (
        <button
          key={job.jobId}
          onClick={() => onSelectJob?.(job)}
          className="w-full flex items-center gap-3 p-3 bg-gray-800/30 hover:bg-gray-800/50 border border-gray-700 rounded-lg transition-colors text-left"
        >
          {getStatusIcon(job.status)}

          <div className="flex-1 min-w-0">
            <p className="text-white text-sm font-medium truncate">
              {job.outline?.title || 'Untitled Course'}
            </p>
            <p className="text-xs text-gray-500">
              {new Date(job.createdAt).toLocaleDateString()} at{' '}
              {new Date(job.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </p>
          </div>

          <div className="text-right">
            <p className={`text-xs font-medium ${getStatusColor(job.status)}`}>
              {job.status === 'processing' ? `${job.progress.toFixed(0)}%` : job.status}
            </p>
            {job.lecturesTotal > 0 && (
              <p className="text-xs text-gray-500">
                {job.lecturesCompleted}/{job.lecturesTotal} lectures
              </p>
            )}
          </div>

          {job.status === 'completed' && (
            <div className="flex items-center gap-1">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handlePractice(job.jobId);
                }}
                className="p-1.5 text-gray-500 hover:text-green-400 transition-colors"
                title="Mode Pratique"
              >
                <GraduationCap className="w-4 h-4" />
              </button>
              <a
                href={`${process.env.NEXT_PUBLIC_COURSE_API_URL || 'http://localhost:8007'}/api/v1/courses/${job.jobId}/download`}
                onClick={(e) => e.stopPropagation()}
                className="p-1.5 text-gray-500 hover:text-purple-400 transition-colors"
                title="Download"
              >
                <Download className="w-4 h-4" />
              </a>
            </div>
          )}
        </button>
      ))}
    </div>
  );
}
