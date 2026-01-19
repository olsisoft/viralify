"""
Collaboration Models

Pydantic models for team workspaces, member roles, and course sharing.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, EmailStr


class TeamRole(str, Enum):
    """Team member roles"""
    OWNER = "owner"       # Full access, can delete workspace
    ADMIN = "admin"       # Can manage members and settings
    EDITOR = "editor"     # Can create and edit courses
    VIEWER = "viewer"     # Can only view courses


class InvitationStatus(str, Enum):
    """Invitation status"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"


class SharePermission(str, Enum):
    """Course sharing permission levels"""
    VIEW = "view"         # Can view course
    COMMENT = "comment"   # Can view and comment
    EDIT = "edit"         # Can edit course
    FULL = "full"         # Full access including delete


# Role permissions mapping
ROLE_PERMISSIONS: Dict[str, Dict] = {
    "owner": {
        "can_manage_team": True,
        "can_invite_members": True,
        "can_remove_members": True,
        "can_change_roles": True,
        "can_delete_workspace": True,
        "can_create_courses": True,
        "can_edit_all_courses": True,
        "can_delete_courses": True,
        "can_view_analytics": True,
        "can_manage_billing": True,
    },
    "admin": {
        "can_manage_team": True,
        "can_invite_members": True,
        "can_remove_members": True,
        "can_change_roles": False,  # Can't change to owner
        "can_delete_workspace": False,
        "can_create_courses": True,
        "can_edit_all_courses": True,
        "can_delete_courses": True,
        "can_view_analytics": True,
        "can_manage_billing": False,
    },
    "editor": {
        "can_manage_team": False,
        "can_invite_members": False,
        "can_remove_members": False,
        "can_change_roles": False,
        "can_delete_workspace": False,
        "can_create_courses": True,
        "can_edit_all_courses": False,  # Only own courses
        "can_delete_courses": False,
        "can_view_analytics": True,
        "can_manage_billing": False,
    },
    "viewer": {
        "can_manage_team": False,
        "can_invite_members": False,
        "can_remove_members": False,
        "can_change_roles": False,
        "can_delete_workspace": False,
        "can_create_courses": False,
        "can_edit_all_courses": False,
        "can_delete_courses": False,
        "can_view_analytics": False,
        "can_manage_billing": False,
    },
}


class TeamMember(BaseModel):
    """Team member with role"""
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    user_id: str
    email: str
    name: str
    role: TeamRole
    avatar_url: Optional[str] = None
    joined_at: datetime = Field(default_factory=datetime.utcnow)
    last_active_at: Optional[datetime] = None


class TeamInvitation(BaseModel):
    """Team invitation"""
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    workspace_id: str
    email: str
    role: TeamRole = TeamRole.EDITOR
    status: InvitationStatus = InvitationStatus.PENDING
    invited_by: str  # user_id of inviter
    invite_token: str = Field(default_factory=lambda: __import__('secrets').token_urlsafe(32))
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + __import__('datetime').timedelta(days=7))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    responded_at: Optional[datetime] = None


class Workspace(BaseModel):
    """Team workspace"""
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    name: str
    slug: str  # URL-friendly name
    description: Optional[str] = None
    logo_url: Optional[str] = None
    owner_id: str

    # Settings
    default_member_role: TeamRole = TeamRole.EDITOR
    allow_external_sharing: bool = True
    require_approval_for_publish: bool = False

    # Membership
    members: List[TeamMember] = Field(default_factory=list)
    pending_invitations: List[TeamInvitation] = Field(default_factory=list)

    # Usage
    course_count: int = 0
    member_count: int = 1

    # Dates
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CourseShare(BaseModel):
    """Course sharing record"""
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    course_id: str
    shared_by: str  # user_id
    shared_with_user_id: Optional[str] = None  # For user shares
    shared_with_workspace_id: Optional[str] = None  # For workspace shares
    shared_with_email: Optional[str] = None  # For email invites
    permission: SharePermission
    is_public: bool = False  # Public link
    public_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ActivityLog(BaseModel):
    """Team activity log entry"""
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    workspace_id: str
    user_id: str
    user_name: str
    action: str  # e.g., "created_course", "edited_course", "invited_member"
    resource_type: str  # e.g., "course", "member", "workspace"
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    details: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Request/Response Models

class CreateWorkspaceRequest(BaseModel):
    """Request to create a workspace"""
    name: str
    slug: str
    description: Optional[str] = None
    owner_id: str


class UpdateWorkspaceRequest(BaseModel):
    """Request to update workspace settings"""
    name: Optional[str] = None
    description: Optional[str] = None
    logo_url: Optional[str] = None
    default_member_role: Optional[TeamRole] = None
    allow_external_sharing: Optional[bool] = None


class InviteMemberRequest(BaseModel):
    """Request to invite a team member"""
    email: str
    role: TeamRole = TeamRole.EDITOR
    message: Optional[str] = None


class UpdateMemberRoleRequest(BaseModel):
    """Request to update member role"""
    user_id: str
    new_role: TeamRole


class ShareCourseRequest(BaseModel):
    """Request to share a course"""
    course_id: str
    share_with_email: Optional[str] = None
    share_with_user_id: Optional[str] = None
    share_with_workspace_id: Optional[str] = None
    permission: SharePermission = SharePermission.VIEW
    create_public_link: bool = False


class AcceptInvitationRequest(BaseModel):
    """Request to accept an invitation"""
    invite_token: str
    user_id: str
    user_name: str
    user_email: str


class WorkspaceResponse(BaseModel):
    """Workspace response with member info"""
    workspace: Workspace
    current_user_role: TeamRole
    permissions: Dict


class WorkspaceListResponse(BaseModel):
    """List of workspaces"""
    workspaces: List[Workspace]
    total: int


class MemberListResponse(BaseModel):
    """List of team members"""
    members: List[TeamMember]
    pending_invitations: List[TeamInvitation]
    total: int


class ActivityLogResponse(BaseModel):
    """Activity log response"""
    activities: List[ActivityLog]
    total: int
    has_more: bool
