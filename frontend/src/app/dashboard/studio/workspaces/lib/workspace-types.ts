/**
 * Workspace/Collaboration Types
 * TypeScript types for team collaboration features
 */

export type TeamRole = 'owner' | 'admin' | 'editor' | 'viewer';
export type InvitationStatus = 'pending' | 'accepted' | 'declined' | 'expired';
export type SharePermission = 'view' | 'comment' | 'edit' | 'full';

export interface TeamMember {
  id: string;
  user_id: string;
  email: string;
  name: string;
  role: TeamRole;
  avatar_url?: string;
  joined_at: string;
  last_active_at?: string;
}

export interface TeamInvitation {
  id: string;
  workspace_id: string;
  email: string;
  role: TeamRole;
  status: InvitationStatus;
  invited_by: string;
  invite_token: string;
  expires_at: string;
  created_at: string;
}

export interface Workspace {
  id: string;
  name: string;
  slug: string;
  description?: string;
  logo_url?: string;
  owner_id: string;
  default_member_role: TeamRole;
  allow_external_sharing: boolean;
  members: TeamMember[];
  pending_invitations: TeamInvitation[];
  course_count: number;
  member_count: number;
  created_at: string;
  updated_at: string;
}

export interface WorkspaceResponse {
  workspace: Workspace;
  current_user_role: TeamRole;
  permissions: Record<string, boolean>;
}

export interface ActivityLog {
  id: string;
  workspace_id: string;
  user_id: string;
  user_name: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  resource_name?: string;
  details: Record<string, any>;
  created_at: string;
}

// Role permissions
export const ROLE_PERMISSIONS: Record<TeamRole, Record<string, boolean>> = {
  owner: {
    can_manage_team: true,
    can_invite_members: true,
    can_remove_members: true,
    can_change_roles: true,
    can_delete_workspace: true,
    can_create_courses: true,
    can_edit_all_courses: true,
    can_delete_courses: true,
    can_view_analytics: true,
    can_manage_billing: true,
  },
  admin: {
    can_manage_team: true,
    can_invite_members: true,
    can_remove_members: true,
    can_change_roles: false,
    can_delete_workspace: false,
    can_create_courses: true,
    can_edit_all_courses: true,
    can_delete_courses: true,
    can_view_analytics: true,
    can_manage_billing: false,
  },
  editor: {
    can_manage_team: false,
    can_invite_members: false,
    can_remove_members: false,
    can_change_roles: false,
    can_delete_workspace: false,
    can_create_courses: true,
    can_edit_all_courses: false,
    can_delete_courses: false,
    can_view_analytics: true,
    can_manage_billing: false,
  },
  viewer: {
    can_manage_team: false,
    can_invite_members: false,
    can_remove_members: false,
    can_change_roles: false,
    can_delete_workspace: false,
    can_create_courses: false,
    can_edit_all_courses: false,
    can_delete_courses: false,
    can_view_analytics: false,
    can_manage_billing: false,
  },
};

// Helper functions
export function getRoleLabel(role: TeamRole): string {
  const labels: Record<TeamRole, string> = {
    owner: 'Owner',
    admin: 'Admin',
    editor: 'Editor',
    viewer: 'Viewer',
  };
  return labels[role];
}

export function getRoleColor(role: TeamRole): string {
  const colors: Record<TeamRole, string> = {
    owner: 'text-yellow-400 bg-yellow-500/20',
    admin: 'text-purple-400 bg-purple-500/20',
    editor: 'text-blue-400 bg-blue-500/20',
    viewer: 'text-gray-400 bg-gray-500/20',
  };
  return colors[role];
}

export function getActionLabel(action: string): string {
  const labels: Record<string, string> = {
    created_workspace: 'created the workspace',
    updated_workspace: 'updated workspace settings',
    invited_member: 'invited a new member',
    joined_workspace: 'joined the workspace',
    removed_member: 'removed a member',
    updated_role: 'changed a member role',
    created_course: 'created a course',
    updated_course: 'updated a course',
    deleted_course: 'deleted a course',
    shared_course: 'shared a course',
  };
  return labels[action] || action;
}

export function formatRelativeTime(date: string): string {
  const now = new Date();
  const then = new Date(date);
  const diffMs = now.getTime() - then.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return then.toLocaleDateString();
}
