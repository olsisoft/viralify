'use client';

import { useState, useEffect } from 'react';
import { useWorkspaces } from './hooks/useWorkspaces';
import {
  TeamRole,
  getRoleLabel,
  getRoleColor,
  getActionLabel,
  formatRelativeTime,
  ROLE_PERMISSIONS,
} from './lib/workspace-types';

function MemberCard({
  member,
  currentUserRole,
  onRemove,
  onChangeRole,
}: {
  member: any;
  currentUserRole: TeamRole;
  onRemove: () => void;
  onChangeRole: (role: TeamRole) => void;
}) {
  const [showRoleMenu, setShowRoleMenu] = useState(false);
  const canManage = ROLE_PERMISSIONS[currentUserRole].can_remove_members;
  const canChangeRole = ROLE_PERMISSIONS[currentUserRole].can_change_roles;

  return (
    <div className="flex items-center justify-between p-4 bg-gray-800 rounded-lg border border-gray-700">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-purple-600 flex items-center justify-center text-white font-medium">
          {member.name.charAt(0).toUpperCase()}
        </div>
        <div>
          <div className="text-white font-medium">{member.name}</div>
          <div className="text-gray-400 text-sm">{member.email}</div>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <div className="relative">
          <button
            onClick={() => canChangeRole && member.role !== 'owner' && setShowRoleMenu(!showRoleMenu)}
            className={`px-3 py-1 rounded-full text-sm ${getRoleColor(member.role)} ${
              canChangeRole && member.role !== 'owner' ? 'cursor-pointer hover:opacity-80' : ''
            }`}
          >
            {getRoleLabel(member.role)}
          </button>
          {showRoleMenu && (
            <div className="absolute right-0 top-full mt-2 bg-gray-800 border border-gray-700 rounded-lg shadow-lg z-10">
              {(['admin', 'editor', 'viewer'] as TeamRole[]).map((role) => (
                <button
                  key={role}
                  onClick={() => {
                    onChangeRole(role);
                    setShowRoleMenu(false);
                  }}
                  className="block w-full px-4 py-2 text-left text-gray-300 hover:bg-gray-700 first:rounded-t-lg last:rounded-b-lg"
                >
                  {getRoleLabel(role)}
                </button>
              ))}
            </div>
          )}
        </div>
        {canManage && member.role !== 'owner' && (
          <button
            onClick={onRemove}
            className="p-2 text-gray-400 hover:text-red-400 transition-colors"
            title="Remove member"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
}

export default function WorkspacesPage() {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [newWorkspaceName, setNewWorkspaceName] = useState('');
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState<TeamRole>('editor');
  const [activeTab, setActiveTab] = useState<'members' | 'activity'>('members');

  // TODO: Get actual user ID from auth context
  const userId = 'demo-user-123';

  const {
    workspaces,
    selectedWorkspace,
    activities,
    isLoading,
    error,
    selectWorkspace,
    createWorkspace,
    inviteMember,
    removeMember,
    updateMemberRole,
    leaveWorkspace,
    fetchActivities,
  } = useWorkspaces(userId);

  useEffect(() => {
    if (selectedWorkspace) {
      fetchActivities(selectedWorkspace.workspace.id, userId);
    }
  }, [selectedWorkspace, fetchActivities, userId]);

  const handleCreateWorkspace = async () => {
    if (!newWorkspaceName.trim()) return;
    try {
      const workspace = await createWorkspace(newWorkspaceName, userId);
      await selectWorkspace(workspace.id, userId);
      setShowCreateModal(false);
      setNewWorkspaceName('');
    } catch (err) {
      console.error('Create workspace error:', err);
    }
  };

  const handleInviteMember = async () => {
    if (!inviteEmail.trim() || !selectedWorkspace) return;
    try {
      await inviteMember(selectedWorkspace.workspace.id, inviteEmail, inviteRole, userId);
      await selectWorkspace(selectedWorkspace.workspace.id, userId);
      setShowInviteModal(false);
      setInviteEmail('');
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleRemoveMember = async (memberId: string) => {
    if (!selectedWorkspace) return;
    if (!confirm('Remove this member from the workspace?')) return;
    try {
      await removeMember(selectedWorkspace.workspace.id, memberId, userId);
      await selectWorkspace(selectedWorkspace.workspace.id, userId);
    } catch (err: any) {
      alert(err.message);
    }
  };

  const handleChangeRole = async (memberId: string, newRole: TeamRole) => {
    if (!selectedWorkspace) return;
    try {
      await updateMemberRole(selectedWorkspace.workspace.id, memberId, newRole, userId);
      await selectWorkspace(selectedWorkspace.workspace.id, userId);
    } catch (err: any) {
      alert(err.message);
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-white">Workspaces</h1>
          <p className="text-gray-400 mt-1">Collaborate with your team on courses</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white font-medium transition-colors"
        >
          Create Workspace
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Workspace List */}
        <div className="lg:col-span-1">
          <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
            <div className="p-4 border-b border-gray-700">
              <h2 className="font-medium text-white">Your Workspaces</h2>
            </div>
            <div className="divide-y divide-gray-700">
              {isLoading && workspaces.length === 0 ? (
                <div className="p-4 text-gray-400 text-center">Loading...</div>
              ) : workspaces.length === 0 ? (
                <div className="p-4 text-gray-400 text-center">No workspaces yet</div>
              ) : (
                workspaces.map((workspace) => (
                  <button
                    key={workspace.id}
                    onClick={() => selectWorkspace(workspace.id, userId)}
                    className={`w-full p-4 text-left hover:bg-gray-700/50 transition-colors ${
                      selectedWorkspace?.workspace.id === workspace.id ? 'bg-gray-700/50' : ''
                    }`}
                  >
                    <div className="font-medium text-white">{workspace.name}</div>
                    <div className="text-sm text-gray-400 mt-1">
                      {workspace.member_count} members &middot; {workspace.course_count} courses
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Workspace Details */}
        <div className="lg:col-span-3">
          {selectedWorkspace ? (
            <div className="bg-gray-800 rounded-xl border border-gray-700">
              {/* Workspace Header */}
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-bold text-white">{selectedWorkspace.workspace.name}</h2>
                    {selectedWorkspace.workspace.description && (
                      <p className="text-gray-400 mt-1">{selectedWorkspace.workspace.description}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`px-3 py-1 rounded-full text-sm ${getRoleColor(selectedWorkspace.current_user_role)}`}>
                      {getRoleLabel(selectedWorkspace.current_user_role)}
                    </span>
                    {selectedWorkspace.permissions.can_invite_members && (
                      <button
                        onClick={() => setShowInviteModal(true)}
                        className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm transition-colors"
                      >
                        Invite
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Tabs */}
              <div className="flex border-b border-gray-700">
                <button
                  onClick={() => setActiveTab('members')}
                  className={`px-6 py-3 text-sm font-medium transition-colors ${
                    activeTab === 'members'
                      ? 'text-purple-400 border-b-2 border-purple-400'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Members ({selectedWorkspace.workspace.members.length})
                </button>
                <button
                  onClick={() => setActiveTab('activity')}
                  className={`px-6 py-3 text-sm font-medium transition-colors ${
                    activeTab === 'activity'
                      ? 'text-purple-400 border-b-2 border-purple-400'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Activity
                </button>
              </div>

              {/* Tab Content */}
              <div className="p-6">
                {activeTab === 'members' ? (
                  <div className="space-y-3">
                    {selectedWorkspace.workspace.members.map((member) => (
                      <MemberCard
                        key={member.id}
                        member={member}
                        currentUserRole={selectedWorkspace.current_user_role}
                        onRemove={() => handleRemoveMember(member.user_id)}
                        onChangeRole={(role) => handleChangeRole(member.user_id, role)}
                      />
                    ))}

                    {/* Pending Invitations */}
                    {selectedWorkspace.workspace.pending_invitations.length > 0 && (
                      <>
                        <h3 className="text-gray-400 text-sm font-medium mt-6 mb-3">Pending Invitations</h3>
                        {selectedWorkspace.workspace.pending_invitations.map((invitation) => (
                          <div
                            key={invitation.id}
                            className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg border border-gray-700"
                          >
                            <div>
                              <div className="text-white">{invitation.email}</div>
                              <div className="text-gray-400 text-sm">
                                Invited as {getRoleLabel(invitation.role)} &middot; Expires {formatRelativeTime(invitation.expires_at)}
                              </div>
                            </div>
                            <span className="px-3 py-1 rounded-full text-sm text-yellow-400 bg-yellow-500/20">
                              Pending
                            </span>
                          </div>
                        ))}
                      </>
                    )}
                  </div>
                ) : (
                  <div className="space-y-3">
                    {activities.length === 0 ? (
                      <div className="text-gray-400 text-center py-8">No activity yet</div>
                    ) : (
                      activities.map((activity) => (
                        <div key={activity.id} className="flex items-start gap-3 py-3 border-b border-gray-700 last:border-0">
                          <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-gray-400 text-sm flex-shrink-0">
                            {activity.user_name.charAt(0).toUpperCase()}
                          </div>
                          <div className="flex-1">
                            <div className="text-gray-300">
                              <span className="text-white font-medium">{activity.user_name}</span>{' '}
                              {getActionLabel(activity.action)}
                              {activity.resource_name && (
                                <span className="text-purple-400"> "{activity.resource_name}"</span>
                              )}
                            </div>
                            <div className="text-gray-500 text-sm mt-1">
                              {formatRelativeTime(activity.created_at)}
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-gray-800 rounded-xl border border-gray-700 p-12 text-center">
              <div className="text-4xl mb-4">&#128101;</div>
              <h3 className="text-white font-medium mb-2">Select a workspace</h3>
              <p className="text-gray-400 mb-4">Choose a workspace from the list or create a new one</p>
              <button
                onClick={() => setShowCreateModal(true)}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
              >
                Create Workspace
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Create Workspace Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-xl font-bold text-white mb-4">Create Workspace</h3>
            <input
              type="text"
              value={newWorkspaceName}
              onChange={(e) => setNewWorkspaceName(e.target.value)}
              placeholder="Workspace name"
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500 mb-4"
            />
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowCreateModal(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateWorkspace}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Invite Member Modal */}
      {showInviteModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-xl font-bold text-white mb-4">Invite Member</h3>
            <input
              type="email"
              value={inviteEmail}
              onChange={(e) => setInviteEmail(e.target.value)}
              placeholder="Email address"
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500 mb-4"
            />
            <select
              value={inviteRole}
              onChange={(e) => setInviteRole(e.target.value as TeamRole)}
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500 mb-4"
            >
              <option value="admin">Admin</option>
              <option value="editor">Editor</option>
              <option value="viewer">Viewer</option>
            </select>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowInviteModal(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleInviteMember}
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white transition-colors"
              >
                Send Invite
              </button>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-6 p-4 bg-red-500/20 border border-red-500/30 rounded-lg text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}
