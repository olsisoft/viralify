"""
Collaboration Service

Handles team workspaces, member management, and course sharing.
"""
import re
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from models.collaboration_models import (
    TeamRole,
    InvitationStatus,
    SharePermission,
    ROLE_PERMISSIONS,
    TeamMember,
    TeamInvitation,
    Workspace,
    CourseShare,
    ActivityLog,
    WorkspaceResponse,
)


class CollaborationRepository:
    """In-memory collaboration repository. Use PostgreSQL in production."""

    def __init__(self):
        self.workspaces: Dict[str, Workspace] = {}
        self.user_workspaces: Dict[str, List[str]] = {}  # user_id -> [workspace_ids]
        self.invitations: Dict[str, TeamInvitation] = {}  # token -> invitation
        self.course_shares: Dict[str, List[CourseShare]] = {}  # course_id -> shares
        self.activity_logs: Dict[str, List[ActivityLog]] = {}  # workspace_id -> logs

    async def save_workspace(self, workspace: Workspace) -> None:
        self.workspaces[workspace.id] = workspace

        # Index by members
        for member in workspace.members:
            if member.user_id not in self.user_workspaces:
                self.user_workspaces[member.user_id] = []
            if workspace.id not in self.user_workspaces[member.user_id]:
                self.user_workspaces[member.user_id].append(workspace.id)

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        return self.workspaces.get(workspace_id)

    async def get_workspace_by_slug(self, slug: str) -> Optional[Workspace]:
        for ws in self.workspaces.values():
            if ws.slug == slug:
                return ws
        return None

    async def get_user_workspaces(self, user_id: str) -> List[Workspace]:
        workspace_ids = self.user_workspaces.get(user_id, [])
        return [self.workspaces[wid] for wid in workspace_ids if wid in self.workspaces]

    async def delete_workspace(self, workspace_id: str) -> None:
        workspace = self.workspaces.pop(workspace_id, None)
        if workspace:
            for member in workspace.members:
                if member.user_id in self.user_workspaces:
                    self.user_workspaces[member.user_id] = [
                        wid for wid in self.user_workspaces[member.user_id]
                        if wid != workspace_id
                    ]

    async def save_invitation(self, invitation: TeamInvitation) -> None:
        self.invitations[invitation.invite_token] = invitation

    async def get_invitation(self, token: str) -> Optional[TeamInvitation]:
        return self.invitations.get(token)

    async def save_course_share(self, share: CourseShare) -> None:
        if share.course_id not in self.course_shares:
            self.course_shares[share.course_id] = []
        self.course_shares[share.course_id].append(share)

    async def get_course_shares(self, course_id: str) -> List[CourseShare]:
        return self.course_shares.get(course_id, [])

    async def save_activity(self, activity: ActivityLog) -> None:
        if activity.workspace_id not in self.activity_logs:
            self.activity_logs[activity.workspace_id] = []
        self.activity_logs[activity.workspace_id].insert(0, activity)
        # Keep last 1000 activities
        self.activity_logs[activity.workspace_id] = self.activity_logs[activity.workspace_id][:1000]

    async def get_activities(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ActivityLog]:
        logs = self.activity_logs.get(workspace_id, [])
        return logs[offset:offset + limit]


class CollaborationService:
    """
    Main collaboration service for team workspaces and sharing.
    """

    def __init__(self):
        self.repository = CollaborationRepository()
        print("[COLLABORATION] Service initialized", flush=True)

    def _generate_slug(self, name: str) -> str:
        """Generate URL-friendly slug from name."""
        slug = name.lower()
        slug = re.sub(r'[^a-z0-9\s-]', '', slug)
        slug = re.sub(r'[\s-]+', '-', slug)
        slug = slug.strip('-')
        return slug[:50]

    async def create_workspace(
        self,
        name: str,
        owner_id: str,
        owner_name: str,
        owner_email: str,
        description: Optional[str] = None,
    ) -> Workspace:
        """Create a new team workspace."""
        slug = self._generate_slug(name)

        # Ensure unique slug
        existing = await self.repository.get_workspace_by_slug(slug)
        if existing:
            slug = f"{slug}-{secrets.token_hex(4)}"

        # Create owner as first member
        owner_member = TeamMember(
            user_id=owner_id,
            email=owner_email,
            name=owner_name,
            role=TeamRole.OWNER,
        )

        workspace = Workspace(
            name=name,
            slug=slug,
            description=description,
            owner_id=owner_id,
            members=[owner_member],
            member_count=1,
        )

        await self.repository.save_workspace(workspace)

        # Log activity
        await self._log_activity(
            workspace_id=workspace.id,
            user_id=owner_id,
            user_name=owner_name,
            action="created_workspace",
            resource_type="workspace",
            resource_id=workspace.id,
            resource_name=workspace.name,
        )

        print(f"[COLLABORATION] Workspace created: {workspace.name}", flush=True)
        return workspace

    async def get_workspace(
        self,
        workspace_id: str,
        user_id: str,
    ) -> Optional[WorkspaceResponse]:
        """Get workspace with user's role and permissions."""
        workspace = await self.repository.get_workspace(workspace_id)
        if not workspace:
            return None

        # Find user's role
        user_role = None
        for member in workspace.members:
            if member.user_id == user_id:
                user_role = member.role
                break

        if not user_role:
            return None  # User not a member

        permissions = ROLE_PERMISSIONS.get(user_role.value, {})

        return WorkspaceResponse(
            workspace=workspace,
            current_user_role=user_role,
            permissions=permissions,
        )

    async def get_user_workspaces(self, user_id: str) -> List[Workspace]:
        """Get all workspaces user is a member of."""
        return await self.repository.get_user_workspaces(user_id)

    async def update_workspace(
        self,
        workspace_id: str,
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        logo_url: Optional[str] = None,
        default_member_role: Optional[TeamRole] = None,
        allow_external_sharing: Optional[bool] = None,
    ) -> Workspace:
        """Update workspace settings."""
        workspace = await self.repository.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        # Check permission
        if not await self._has_permission(workspace, user_id, "can_manage_team"):
            raise PermissionError("Permission denied")

        if name:
            workspace.name = name
        if description is not None:
            workspace.description = description
        if logo_url is not None:
            workspace.logo_url = logo_url
        if default_member_role:
            workspace.default_member_role = default_member_role
        if allow_external_sharing is not None:
            workspace.allow_external_sharing = allow_external_sharing

        workspace.updated_at = datetime.utcnow()
        await self.repository.save_workspace(workspace)

        return workspace

    async def invite_member(
        self,
        workspace_id: str,
        inviter_id: str,
        inviter_name: str,
        email: str,
        role: TeamRole = TeamRole.EDITOR,
    ) -> TeamInvitation:
        """Invite a new member to the workspace."""
        workspace = await self.repository.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        # Check permission
        if not await self._has_permission(workspace, inviter_id, "can_invite_members"):
            raise PermissionError("Permission denied")

        # Check if already a member
        for member in workspace.members:
            if member.email == email:
                raise ValueError("User is already a member")

        # Check existing pending invitation
        for inv in workspace.pending_invitations:
            if inv.email == email and inv.status == InvitationStatus.PENDING:
                raise ValueError("Invitation already pending")

        # Create invitation
        invitation = TeamInvitation(
            workspace_id=workspace_id,
            email=email,
            role=role,
            invited_by=inviter_id,
        )

        workspace.pending_invitations.append(invitation)
        await self.repository.save_workspace(workspace)
        await self.repository.save_invitation(invitation)

        # Log activity
        await self._log_activity(
            workspace_id=workspace_id,
            user_id=inviter_id,
            user_name=inviter_name,
            action="invited_member",
            resource_type="invitation",
            resource_id=invitation.id,
            details={"email": email, "role": role.value},
        )

        print(f"[COLLABORATION] Invitation sent to {email}", flush=True)
        return invitation

    async def accept_invitation(
        self,
        invite_token: str,
        user_id: str,
        user_name: str,
        user_email: str,
    ) -> Workspace:
        """Accept a team invitation."""
        invitation = await self.repository.get_invitation(invite_token)
        if not invitation:
            raise ValueError("Invalid invitation")

        if invitation.status != InvitationStatus.PENDING:
            raise ValueError("Invitation is no longer valid")

        if invitation.expires_at < datetime.utcnow():
            invitation.status = InvitationStatus.EXPIRED
            raise ValueError("Invitation has expired")

        workspace = await self.repository.get_workspace(invitation.workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        # Add member
        new_member = TeamMember(
            user_id=user_id,
            email=user_email,
            name=user_name,
            role=invitation.role,
        )
        workspace.members.append(new_member)
        workspace.member_count = len(workspace.members)

        # Update invitation status
        invitation.status = InvitationStatus.ACCEPTED
        invitation.responded_at = datetime.utcnow()

        # Remove from pending
        workspace.pending_invitations = [
            inv for inv in workspace.pending_invitations
            if inv.id != invitation.id
        ]

        workspace.updated_at = datetime.utcnow()
        await self.repository.save_workspace(workspace)

        # Log activity
        await self._log_activity(
            workspace_id=workspace.id,
            user_id=user_id,
            user_name=user_name,
            action="joined_workspace",
            resource_type="member",
            resource_id=user_id,
        )

        print(f"[COLLABORATION] {user_name} joined {workspace.name}", flush=True)
        return workspace

    async def remove_member(
        self,
        workspace_id: str,
        remover_id: str,
        remover_name: str,
        member_user_id: str,
    ) -> Workspace:
        """Remove a member from the workspace."""
        workspace = await self.repository.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        # Check permission
        if not await self._has_permission(workspace, remover_id, "can_remove_members"):
            raise PermissionError("Permission denied")

        # Can't remove owner
        if member_user_id == workspace.owner_id:
            raise ValueError("Cannot remove workspace owner")

        # Can't remove self using this method
        if member_user_id == remover_id:
            raise ValueError("Use leave_workspace to remove yourself")

        # Find and remove member
        removed_member = None
        workspace.members = [
            m for m in workspace.members
            if m.user_id != member_user_id or (removed_member := m) is None
        ]

        if not removed_member:
            raise ValueError("Member not found")

        workspace.member_count = len(workspace.members)
        workspace.updated_at = datetime.utcnow()
        await self.repository.save_workspace(workspace)

        # Log activity
        await self._log_activity(
            workspace_id=workspace_id,
            user_id=remover_id,
            user_name=remover_name,
            action="removed_member",
            resource_type="member",
            resource_id=member_user_id,
            resource_name=removed_member.name if removed_member else None,
        )

        print(f"[COLLABORATION] Member removed from {workspace.name}", flush=True)
        return workspace

    async def update_member_role(
        self,
        workspace_id: str,
        updater_id: str,
        member_user_id: str,
        new_role: TeamRole,
    ) -> TeamMember:
        """Update a member's role."""
        workspace = await self.repository.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        # Check permission
        if not await self._has_permission(workspace, updater_id, "can_change_roles"):
            raise PermissionError("Permission denied")

        # Can't change owner role
        if member_user_id == workspace.owner_id:
            raise ValueError("Cannot change owner's role")

        # Find member
        member = None
        for m in workspace.members:
            if m.user_id == member_user_id:
                member = m
                break

        if not member:
            raise ValueError("Member not found")

        member.role = new_role
        workspace.updated_at = datetime.utcnow()
        await self.repository.save_workspace(workspace)

        return member

    async def leave_workspace(
        self,
        workspace_id: str,
        user_id: str,
    ) -> None:
        """Leave a workspace."""
        workspace = await self.repository.get_workspace(workspace_id)
        if not workspace:
            raise ValueError("Workspace not found")

        # Owner can't leave without transferring ownership
        if user_id == workspace.owner_id:
            raise ValueError("Owner must transfer ownership before leaving")

        workspace.members = [m for m in workspace.members if m.user_id != user_id]
        workspace.member_count = len(workspace.members)
        workspace.updated_at = datetime.utcnow()
        await self.repository.save_workspace(workspace)

    async def share_course(
        self,
        course_id: str,
        sharer_id: str,
        permission: SharePermission,
        share_with_user_id: Optional[str] = None,
        share_with_workspace_id: Optional[str] = None,
        share_with_email: Optional[str] = None,
        create_public_link: bool = False,
    ) -> CourseShare:
        """Share a course with a user, workspace, or create public link."""
        share = CourseShare(
            course_id=course_id,
            shared_by=sharer_id,
            shared_with_user_id=share_with_user_id,
            shared_with_workspace_id=share_with_workspace_id,
            shared_with_email=share_with_email,
            permission=permission,
            is_public=create_public_link,
            public_token=secrets.token_urlsafe(16) if create_public_link else None,
        )

        await self.repository.save_course_share(share)
        print(f"[COLLABORATION] Course {course_id} shared", flush=True)
        return share

    async def get_course_shares(self, course_id: str) -> List[CourseShare]:
        """Get all shares for a course."""
        return await self.repository.get_course_shares(course_id)

    async def get_activity_log(
        self,
        workspace_id: str,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ActivityLog]:
        """Get workspace activity log."""
        workspace = await self.repository.get_workspace(workspace_id)
        if not workspace:
            return []

        # Check user is member
        is_member = any(m.user_id == user_id for m in workspace.members)
        if not is_member:
            return []

        return await self.repository.get_activities(workspace_id, limit, offset)

    async def _has_permission(
        self,
        workspace: Workspace,
        user_id: str,
        permission: str,
    ) -> bool:
        """Check if user has a specific permission."""
        for member in workspace.members:
            if member.user_id == user_id:
                role_perms = ROLE_PERMISSIONS.get(member.role.value, {})
                return role_perms.get(permission, False)
        return False

    async def _log_activity(
        self,
        workspace_id: str,
        user_id: str,
        user_name: str,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Log an activity."""
        activity = ActivityLog(
            workspace_id=workspace_id,
            user_id=user_id,
            user_name=user_name,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            details=details or {},
        )
        await self.repository.save_activity(activity)


# Global instance
collaboration_service: Optional[CollaborationService] = None


def get_collaboration_service() -> CollaborationService:
    """Get or create collaboration service instance."""
    global collaboration_service
    if collaboration_service is None:
        collaboration_service = CollaborationService()
    return collaboration_service
