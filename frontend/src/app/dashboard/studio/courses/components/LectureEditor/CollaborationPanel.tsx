'use client';

import React, { useState, useCallback, useRef } from 'react';
import type {
  TimecodedComment,
  CommentReply,
  ProjectVersion,
  ShareLink,
  CollaborationState,
} from '../../lib/lecture-editor-types';
import { formatDuration } from '../../lib/lecture-editor-types';

interface CollaborationPanelProps {
  collaboration: CollaborationState;
  currentTime: number;
  onCollaborationChange: (collaboration: CollaborationState) => void;
  onSeek: (time: number) => void;
  onRestoreVersion: (version: ProjectVersion) => void;
  currentUserId: string;
  currentUserName: string;
}

// Generate unique ID
const generateId = () => `id-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

export function CollaborationPanel({
  collaboration,
  currentTime,
  onCollaborationChange,
  onSeek,
  onRestoreVersion,
  currentUserId,
  currentUserName,
}: CollaborationPanelProps) {
  const [activeTab, setActiveTab] = useState<'comments' | 'versions' | 'share'>('comments');
  const [newComment, setNewComment] = useState('');
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [replyText, setReplyText] = useState('');
  const [showResolved, setShowResolved] = useState(false);

  // Add comment
  const addComment = useCallback(() => {
    if (!newComment.trim()) return;

    const comment: TimecodedComment = {
      id: generateId(),
      userId: currentUserId,
      userName: currentUserName,
      timestamp: currentTime,
      text: newComment.trim(),
      createdAt: new Date().toISOString(),
      isResolved: false,
      replies: [],
    };

    onCollaborationChange({
      ...collaboration,
      comments: [...collaboration.comments, comment].sort((a, b) => a.timestamp - b.timestamp),
    });
    setNewComment('');
  }, [newComment, currentTime, currentUserId, currentUserName, collaboration, onCollaborationChange]);

  // Add reply
  const addReply = useCallback((commentId: string) => {
    if (!replyText.trim()) return;

    const reply: CommentReply = {
      id: generateId(),
      userId: currentUserId,
      userName: currentUserName,
      text: replyText.trim(),
      createdAt: new Date().toISOString(),
    };

    onCollaborationChange({
      ...collaboration,
      comments: collaboration.comments.map(c =>
        c.id === commentId ? { ...c, replies: [...c.replies, reply] } : c
      ),
    });
    setReplyText('');
    setReplyingTo(null);
  }, [replyText, currentUserId, currentUserName, collaboration, onCollaborationChange]);

  // Toggle resolve
  const toggleResolve = useCallback((commentId: string) => {
    onCollaborationChange({
      ...collaboration,
      comments: collaboration.comments.map(c =>
        c.id === commentId ? { ...c, isResolved: !c.isResolved } : c
      ),
    });
  }, [collaboration, onCollaborationChange]);

  // Delete comment
  const deleteComment = useCallback((commentId: string) => {
    onCollaborationChange({
      ...collaboration,
      comments: collaboration.comments.filter(c => c.id !== commentId),
    });
  }, [collaboration, onCollaborationChange]);

  // Create share link
  const createShareLink = useCallback((permissions: ShareLink['permissions']) => {
    const link: ShareLink = {
      id: generateId(),
      url: `${window.location.origin}/review/${generateId()}`,
      permissions,
      isActive: true,
      createdAt: new Date().toISOString(),
      accessCount: 0,
    };

    onCollaborationChange({
      ...collaboration,
      shareLinks: [...collaboration.shareLinks, link],
    });
  }, [collaboration, onCollaborationChange]);

  // Delete share link
  const deleteShareLink = useCallback((linkId: string) => {
    onCollaborationChange({
      ...collaboration,
      shareLinks: collaboration.shareLinks.filter(l => l.id !== linkId),
    });
  }, [collaboration, onCollaborationChange]);

  // Filter comments
  const filteredComments = showResolved
    ? collaboration.comments
    : collaboration.comments.filter(c => !c.isResolved);

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-white font-semibold text-sm flex items-center gap-2">
          <span>👥</span>
          Collaboration
        </h3>
        {collaboration.activeCollaborators.length > 0 && (
          <div className="flex -space-x-2">
            {collaboration.activeCollaborators.slice(0, 3).map((collab) => (
              <div
                key={collab.userId}
                className="w-6 h-6 rounded-full bg-purple-600 flex items-center justify-center text-xs text-white border-2 border-gray-900"
                title={collab.userName}
              >
                {collab.userName[0].toUpperCase()}
              </div>
            ))}
            {collaboration.activeCollaborators.length > 3 && (
              <div className="w-6 h-6 rounded-full bg-gray-700 flex items-center justify-center text-xs text-white border-2 border-gray-900">
                +{collaboration.activeCollaborators.length - 3}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-800">
        {(['comments', 'versions', 'share'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-2 text-sm transition-colors ${
              activeTab === tab
                ? 'text-purple-400 border-b-2 border-purple-400'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            {tab === 'comments' ? `Commentaires (${filteredComments.length})` :
             tab === 'versions' ? 'Versions' : 'Partager'}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Comments tab */}
        {activeTab === 'comments' && (
          <div className="p-4 space-y-4">
            {/* Add comment */}
            <div className="flex gap-2">
              <input
                type="text"
                value={newComment}
                onChange={(e) => setNewComment(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addComment()}
                placeholder={`Commenter à ${formatDuration(currentTime)}...`}
                className="flex-1 bg-gray-800 text-white text-sm rounded-lg px-3 py-2 border border-gray-700 focus:border-purple-500 focus:outline-none"
              />
              <button
                onClick={addComment}
                disabled={!newComment.trim()}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>

            {/* Show resolved toggle */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showResolved}
                onChange={(e) => setShowResolved(e.target.checked)}
                className="rounded border-gray-600 bg-gray-800 text-purple-600 focus:ring-purple-500"
              />
              <span className="text-gray-400 text-xs">Afficher les commentaires résolus</span>
            </label>

            {/* Comments list */}
            <div className="space-y-3">
              {filteredComments.length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-4">
                  Aucun commentaire
                </p>
              ) : (
                filteredComments.map(comment => (
                  <CommentItem
                    key={comment.id}
                    comment={comment}
                    isOwner={comment.userId === currentUserId}
                    replyingTo={replyingTo}
                    replyText={replyText}
                    onSeek={() => onSeek(comment.timestamp)}
                    onReply={() => setReplyingTo(comment.id)}
                    onReplyTextChange={setReplyText}
                    onSubmitReply={() => addReply(comment.id)}
                    onCancelReply={() => { setReplyingTo(null); setReplyText(''); }}
                    onToggleResolve={() => toggleResolve(comment.id)}
                    onDelete={() => deleteComment(comment.id)}
                  />
                ))
              )}
            </div>
          </div>
        )}

        {/* Versions tab */}
        {activeTab === 'versions' && (
          <div className="p-4 space-y-3">
            {collaboration.versions.length === 0 ? (
              <p className="text-gray-500 text-sm text-center py-4">
                Aucune version sauvegardée
              </p>
            ) : (
              collaboration.versions.map(version => (
                <div
                  key={version.id}
                  className="bg-gray-800 rounded-lg p-3"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-white text-sm font-medium">
                        {version.name}
                        {version.isAutoSave && (
                          <span className="ml-2 text-xs text-gray-500">(auto)</span>
                        )}
                      </p>
                      <p className="text-gray-500 text-xs mt-1">
                        {new Date(version.createdAt).toLocaleString('fr-FR')}
                      </p>
                      {version.description && (
                        <p className="text-gray-400 text-xs mt-1">{version.description}</p>
                      )}
                    </div>
                    <button
                      onClick={() => onRestoreVersion(version)}
                      className="px-3 py-1 text-xs bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
                    >
                      Restaurer
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* Share tab */}
        {activeTab === 'share' && (
          <div className="p-4 space-y-4">
            {/* Create link buttons */}
            <div className="space-y-2">
              <p className="text-gray-400 text-xs font-medium">Créer un lien de partage</p>
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => createShareLink('view')}
                  className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-xs"
                >
                  Lecture seule
                </button>
                <button
                  onClick={() => createShareLink('comment')}
                  className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-xs"
                >
                  Commentaires
                </button>
                <button
                  onClick={() => createShareLink('edit')}
                  className="py-2 px-3 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 text-xs"
                >
                  Édition
                </button>
              </div>
            </div>

            {/* Existing links */}
            <div className="space-y-2">
              {collaboration.shareLinks.length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-4">
                  Aucun lien de partage
                </p>
              ) : (
                collaboration.shareLinks.map(link => (
                  <div
                    key={link.id}
                    className="bg-gray-800 rounded-lg p-3"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        link.permissions === 'view' ? 'bg-blue-500/20 text-blue-400' :
                        link.permissions === 'comment' ? 'bg-green-500/20 text-green-400' :
                        'bg-yellow-500/20 text-yellow-400'
                      }`}>
                        {link.permissions === 'view' ? 'Lecture' :
                         link.permissions === 'comment' ? 'Commentaire' : 'Édition'}
                      </span>
                      <span className="text-gray-500 text-xs">
                        {link.accessCount} accès
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={link.url}
                        readOnly
                        className="flex-1 bg-gray-700 text-gray-300 text-xs rounded px-2 py-1.5 border border-gray-600"
                      />
                      <button
                        onClick={() => navigator.clipboard.writeText(link.url)}
                        className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
                        title="Copier"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                      </button>
                      <button
                        onClick={() => deleteShareLink(link.id)}
                        className="p-1.5 text-gray-400 hover:text-red-400 hover:bg-gray-700 rounded transition-colors"
                        title="Supprimer"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Comment item
interface CommentItemProps {
  comment: TimecodedComment;
  isOwner: boolean;
  replyingTo: string | null;
  replyText: string;
  onSeek: () => void;
  onReply: () => void;
  onReplyTextChange: (text: string) => void;
  onSubmitReply: () => void;
  onCancelReply: () => void;
  onToggleResolve: () => void;
  onDelete: () => void;
}

function CommentItem({
  comment,
  isOwner,
  replyingTo,
  replyText,
  onSeek,
  onReply,
  onReplyTextChange,
  onSubmitReply,
  onCancelReply,
  onToggleResolve,
  onDelete,
}: CommentItemProps) {
  const isReplying = replyingTo === comment.id;

  return (
    <div className={`bg-gray-800 rounded-lg p-3 ${comment.isResolved ? 'opacity-60' : ''}`}>
      <div className="flex items-start gap-3">
        {/* Avatar */}
        <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm text-white flex-shrink-0">
          {comment.userName[0].toUpperCase()}
        </div>

        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center gap-2 mb-1">
            <span className="text-white text-sm font-medium">{comment.userName}</span>
            <button
              onClick={onSeek}
              className="text-purple-400 text-xs hover:text-purple-300"
            >
              @{formatDuration(comment.timestamp)}
            </button>
            {comment.isResolved && (
              <span className="text-green-400 text-xs">Résolu</span>
            )}
          </div>

          {/* Text */}
          <p className="text-gray-300 text-sm">{comment.text}</p>

          {/* Actions */}
          <div className="flex items-center gap-3 mt-2">
            <button
              onClick={onReply}
              className="text-gray-400 text-xs hover:text-white"
            >
              Répondre
            </button>
            <button
              onClick={onToggleResolve}
              className="text-gray-400 text-xs hover:text-white"
            >
              {comment.isResolved ? 'Rouvrir' : 'Résoudre'}
            </button>
            {isOwner && (
              <button
                onClick={onDelete}
                className="text-gray-400 text-xs hover:text-red-400"
              >
                Supprimer
              </button>
            )}
          </div>

          {/* Replies */}
          {comment.replies.length > 0 && (
            <div className="mt-3 pl-3 border-l-2 border-gray-700 space-y-2">
              {comment.replies.map(reply => (
                <div key={reply.id} className="flex items-start gap-2">
                  <div className="w-6 h-6 rounded-full bg-gray-700 flex items-center justify-center text-xs text-white flex-shrink-0">
                    {reply.userName[0].toUpperCase()}
                  </div>
                  <div>
                    <span className="text-white text-xs font-medium">{reply.userName}</span>
                    <p className="text-gray-400 text-xs">{reply.text}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Reply input */}
          {isReplying && (
            <div className="mt-3 flex gap-2">
              <input
                type="text"
                value={replyText}
                onChange={(e) => onReplyTextChange(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && onSubmitReply()}
                placeholder="Répondre..."
                className="flex-1 bg-gray-700 text-white text-xs rounded px-2 py-1.5 border border-gray-600 focus:border-purple-500 focus:outline-none"
                autoFocus
              />
              <button
                onClick={onSubmitReply}
                disabled={!replyText.trim()}
                className="px-3 py-1.5 text-xs bg-purple-600 text-white rounded hover:bg-purple-500 disabled:opacity-50"
              >
                Envoyer
              </button>
              <button
                onClick={onCancelReply}
                className="px-3 py-1.5 text-xs bg-gray-700 text-white rounded hover:bg-gray-600"
              >
                Annuler
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CollaborationPanel;
