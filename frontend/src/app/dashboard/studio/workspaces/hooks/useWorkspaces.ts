'use client';

import { useState, useCallback, useEffect } from 'react';
import {
  Workspace,
  WorkspaceResponse,
  TeamMember,
  TeamInvitation,
  TeamRole,
  ActivityLog,
} from '../lib/workspace-types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

interface UseWorkspacesReturn {
  // State
  workspaces: Workspace[];
  selectedWorkspace: WorkspaceResponse | null;
  activities: ActivityLog[];
  isLoading: boolean;
  error: string | null;

  // Workspace Actions
  fetchWorkspaces: (userId: string) => Promise<void>;
  selectWorkspace: (workspaceId: string, userId: string) => Promise<void>;
  createWorkspace: (name: string, ownerId: string, description?: string) => Promise<Workspace>;
  updateWorkspace: (workspaceId: string, userId: string, data: Partial<Workspace>) => Promise<void>;

  // Member Actions
  inviteMember: (workspaceId: string, email: string, role: TeamRole, inviterId: string) => Promise<TeamInvitation>;
  removeMember: (workspaceId: string, memberId: string, removerId: string) => Promise<void>;
  updateMemberRole: (workspaceId: string, memberId: string, newRole: TeamRole, updaterId: string) => Promise<void>;
  leaveWorkspace: (workspaceId: string, userId: string) => Promise<void>;

  // Activity
  fetchActivities: (workspaceId: string, userId: string) => Promise<void>;
}

export function useWorkspaces(userId?: string): UseWorkspacesReturn {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [selectedWorkspace, setSelectedWorkspace] = useState<WorkspaceResponse | null>(null);
  const [activities, setActivities] = useState<ActivityLog[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchWorkspaces = useCallback(async (userId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/workspaces?user_id=${userId}`);
      if (!response.ok) throw new Error('Failed to fetch workspaces');
      const data = await response.json();
      setWorkspaces(data.workspaces);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const selectWorkspace = useCallback(async (workspaceId: string, userId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/workspaces/${workspaceId}?user_id=${userId}`);
      if (!response.ok) throw new Error('Failed to fetch workspace');
      const data: WorkspaceResponse = await response.json();
      setSelectedWorkspace(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const createWorkspace = useCallback(async (
    name: string,
    ownerId: string,
    description?: string
  ): Promise<Workspace> => {
    const response = await fetch(`${API_BASE}/api/v1/workspaces`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        slug: name.toLowerCase().replace(/\s+/g, '-'),
        owner_id: ownerId,
        description,
      }),
    });

    if (!response.ok) throw new Error('Failed to create workspace');
    const workspace: Workspace = await response.json();
    setWorkspaces((prev) => [...prev, workspace]);
    return workspace;
  }, []);

  const updateWorkspace = useCallback(async (
    workspaceId: string,
    userId: string,
    data: Partial<Workspace>
  ) => {
    const response = await fetch(`${API_BASE}/api/v1/workspaces/${workspaceId}?user_id=${userId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) throw new Error('Failed to update workspace');
    await selectWorkspace(workspaceId, userId);
  }, [selectWorkspace]);

  const inviteMember = useCallback(async (
    workspaceId: string,
    email: string,
    role: TeamRole,
    inviterId: string
  ): Promise<TeamInvitation> => {
    const response = await fetch(
      `${API_BASE}/api/v1/workspaces/${workspaceId}/invite?inviter_id=${inviterId}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, role }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to invite member');
    }
    return await response.json();
  }, []);

  const removeMember = useCallback(async (
    workspaceId: string,
    memberId: string,
    removerId: string
  ) => {
    const response = await fetch(
      `${API_BASE}/api/v1/workspaces/${workspaceId}/members/${memberId}?remover_id=${removerId}`,
      { method: 'DELETE' }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to remove member');
    }
  }, []);

  const updateMemberRole = useCallback(async (
    workspaceId: string,
    memberId: string,
    newRole: TeamRole,
    updaterId: string
  ) => {
    const response = await fetch(
      `${API_BASE}/api/v1/workspaces/${workspaceId}/members/${memberId}/role?updater_id=${updaterId}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: memberId, new_role: newRole }),
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update role');
    }
  }, []);

  const leaveWorkspace = useCallback(async (workspaceId: string, userId: string) => {
    const response = await fetch(
      `${API_BASE}/api/v1/workspaces/${workspaceId}/leave?user_id=${userId}`,
      { method: 'POST' }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to leave workspace');
    }
    setWorkspaces((prev) => prev.filter((w) => w.id !== workspaceId));
  }, []);

  const fetchActivities = useCallback(async (workspaceId: string, userId: string) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/workspaces/${workspaceId}/activity?user_id=${userId}&limit=50`
      );
      if (!response.ok) throw new Error('Failed to fetch activities');
      const data = await response.json();
      setActivities(data.activities);
    } catch (err) {
      console.error('Activity fetch error:', err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    if (userId) {
      fetchWorkspaces(userId);
    }
  }, [fetchWorkspaces, userId]);

  return {
    workspaces,
    selectedWorkspace,
    activities,
    isLoading,
    error,
    fetchWorkspaces,
    selectWorkspace,
    createWorkspace,
    updateWorkspace,
    inviteMember,
    removeMember,
    updateMemberRole,
    leaveWorkspace,
    fetchActivities,
  };
}
