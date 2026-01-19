"""
Scene Validator Agent

Verifies audio-visual synchronization for a scene.
Can trigger regeneration if sync issues are detected.
"""

import os
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentResult, SyncStatus, WordTimestamp


class SyncIssueType(str, Enum):
    """Types of synchronization issues"""
    VISUAL_BEFORE_AUDIO = "visual_before_audio"  # Visual appears before it's mentioned
    VISUAL_AFTER_AUDIO = "visual_after_audio"    # Visual appears too late
    DURATION_MISMATCH = "duration_mismatch"      # Animation doesn't match audio length
    MISSING_VISUAL = "missing_visual"            # No visual for mentioned content
    OVERLAP_ISSUE = "overlap_issue"              # Visual elements overlap incorrectly
    TIMING_GAP = "timing_gap"                    # Long gaps with no visual changes


@dataclass
class SyncIssue:
    """A detected synchronization issue"""
    issue_type: SyncIssueType
    severity: str  # "critical", "warning", "info"
    timestamp: float
    description: str
    suggested_fix: str


class SceneValidatorAgent(BaseAgent):
    """Validates audio-visual synchronization for a scene"""

    def __init__(self):
        super().__init__("SCENE_VALIDATOR")
        # Tolerance for sync (in seconds)
        self.sync_tolerance = float(os.getenv("SYNC_TOLERANCE", "0.5"))
        # Minimum acceptable sync score
        self.min_sync_score = float(os.getenv("MIN_SYNC_SCORE", "0.7"))
        # Maximum iterations for regeneration
        self.max_iterations = int(os.getenv("MAX_SYNC_ITERATIONS", "3"))

    async def execute(self, state: Dict[str, Any]) -> AgentResult:
        """Validate synchronization for a scene"""
        scene_index = state.get("scene_index", 0)
        word_timestamps = state.get("word_timestamps", [])
        visual_elements = state.get("visual_elements", [])
        sync_map = state.get("sync_map", {})
        animations = state.get("animations", [])
        audio_duration = state.get("audio_duration", 0)
        timing_cues = state.get("timing_cues", [])
        iteration = state.get("iteration", 0)

        self.log(f"Scene {scene_index}: Validating sync (iteration {iteration})")

        try:
            # Convert word timestamps if needed
            word_ts = [
                WordTimestamp(**wt) if isinstance(wt, dict) else wt
                for wt in word_timestamps
            ]

            # Run validation checks
            issues = []

            # Check 1: Visual timing vs audio mentions
            issues.extend(self._check_visual_audio_sync(
                visual_elements, sync_map, word_ts, timing_cues
            ))

            # Check 2: Animation duration
            issues.extend(self._check_animation_timing(
                animations, audio_duration
            ))

            # Check 3: Content coverage
            issues.extend(self._check_content_coverage(
                visual_elements, word_ts, state.get("slide_data", {})
            ))

            # Check 4: Timing gaps
            issues.extend(self._check_timing_gaps(
                sync_map, audio_duration
            ))

            # Calculate sync score
            sync_score = self._calculate_sync_score(issues)

            # Determine status
            critical_issues = [i for i in issues if i.severity == "critical"]

            if not issues:
                sync_status = SyncStatus.IN_SYNC
            elif sync_score >= self.min_sync_score and not critical_issues:
                sync_status = SyncStatus.IN_SYNC
            elif iteration >= self.max_iterations:
                sync_status = SyncStatus.FAILED
                self.log(f"Scene {scene_index}: Max iterations reached, accepting with score {sync_score:.2f}")
            else:
                sync_status = SyncStatus.OUT_OF_SYNC

            # Generate fix suggestions if needed
            fixes = []
            if sync_status == SyncStatus.OUT_OF_SYNC:
                fixes = self._generate_fixes(issues)

            self.log(
                f"Scene {scene_index}: Sync score {sync_score:.2f}, "
                f"status={sync_status.value}, issues={len(issues)}"
            )

            return AgentResult(
                success=True,
                data={
                    "sync_status": sync_status.value,
                    "sync_score": sync_score,
                    "issues": [
                        {
                            "type": issue.issue_type.value,
                            "severity": issue.severity,
                            "timestamp": issue.timestamp,
                            "description": issue.description,
                            "suggested_fix": issue.suggested_fix
                        }
                        for issue in issues
                    ],
                    "fixes": fixes,
                    "needs_regeneration": sync_status == SyncStatus.OUT_OF_SYNC,
                    "iteration": iteration
                }
            )

        except Exception as e:
            self.log(f"Scene {scene_index}: Validation failed - {e}")
            return AgentResult(
                success=False,
                errors=[str(e)],
                data={
                    "sync_status": SyncStatus.FAILED.value,
                    "sync_score": 0,
                    "needs_regeneration": False
                }
            )

    def _check_visual_audio_sync(
        self,
        visual_elements: List[Dict[str, Any]],
        sync_map: Dict[str, Any],
        word_timestamps: List[WordTimestamp],
        timing_cues: List[Dict[str, Any]]
    ) -> List[SyncIssue]:
        """Check if visuals appear at the right time relative to audio"""
        issues = []

        for cue in timing_cues:
            cue_time = cue.get("timestamp", 0)
            target = cue.get("target", "").lower()
            event_type = cue.get("event_type", "")

            # Find when target is mentioned in audio
            mention_time = None
            for wt in word_timestamps:
                if target and target in wt.word.lower():
                    mention_time = wt.start
                    break

            if mention_time is not None:
                time_diff = cue_time - mention_time

                if time_diff > self.sync_tolerance:
                    # Visual appears too late
                    issues.append(SyncIssue(
                        issue_type=SyncIssueType.VISUAL_AFTER_AUDIO,
                        severity="critical" if time_diff > 2 else "warning",
                        timestamp=cue_time,
                        description=f"Visual '{target}' appears {time_diff:.1f}s after audio mention",
                        suggested_fix=f"Move visual timing to {mention_time:.1f}s"
                    ))
                elif time_diff < -self.sync_tolerance:
                    # Visual appears too early (usually OK, but check)
                    if abs(time_diff) > 3:
                        issues.append(SyncIssue(
                            issue_type=SyncIssueType.VISUAL_BEFORE_AUDIO,
                            severity="info",
                            timestamp=cue_time,
                            description=f"Visual '{target}' appears {abs(time_diff):.1f}s before audio mention",
                            suggested_fix="Consider if this is intentional"
                        ))

        return issues

    def _check_animation_timing(
        self,
        animations: List[Dict[str, Any]],
        audio_duration: float
    ) -> List[SyncIssue]:
        """Check animation durations match audio"""
        issues = []

        for anim in animations:
            anim_end = anim.get("start_time", 0) + anim.get("duration", 0)

            if anim_end > audio_duration + 0.5:
                issues.append(SyncIssue(
                    issue_type=SyncIssueType.DURATION_MISMATCH,
                    severity="critical",
                    timestamp=anim.get("start_time", 0),
                    description=f"Animation '{anim.get('element_id')}' ends at {anim_end:.1f}s but audio ends at {audio_duration:.1f}s",
                    suggested_fix=f"Reduce animation duration to fit within {audio_duration:.1f}s"
                ))

            if anim.get("element_type") == "typing":
                # Typing should end before audio ends
                if anim_end > audio_duration - 0.5:
                    issues.append(SyncIssue(
                        issue_type=SyncIssueType.DURATION_MISMATCH,
                        severity="warning",
                        timestamp=anim.get("start_time", 0),
                        description="Typing animation finishes too close to scene end",
                        suggested_fix="Speed up typing animation slightly"
                    ))

        return issues

    def _check_content_coverage(
        self,
        visual_elements: List[Dict[str, Any]],
        word_timestamps: List[WordTimestamp],
        slide_data: Dict[str, Any]
    ) -> List[SyncIssue]:
        """Check if all mentioned content has corresponding visuals"""
        issues = []

        # Key phrases that should have visuals
        key_phrases = ["code", "diagram", "example", "output", "result", "shows", "see"]

        words_text = " ".join([wt.word.lower() for wt in word_timestamps])
        visual_types = [ve.get("element_type", "") for ve in visual_elements]

        for phrase in key_phrases:
            if phrase in words_text:
                # Check if corresponding visual exists
                if phrase == "code" and "code" not in visual_types and "code_animation" not in visual_types:
                    if slide_data.get("code"):
                        issues.append(SyncIssue(
                            issue_type=SyncIssueType.MISSING_VISUAL,
                            severity="critical",
                            timestamp=0,
                            description="Audio mentions 'code' but no code visual found",
                            suggested_fix="Add code visual element"
                        ))

                elif phrase == "diagram" and "diagram" not in visual_types and "image" not in visual_types:
                    issues.append(SyncIssue(
                        issue_type=SyncIssueType.MISSING_VISUAL,
                        severity="warning",
                        timestamp=0,
                        description="Audio mentions 'diagram' but no diagram visual found",
                        suggested_fix="Add diagram or image visual element"
                    ))

        return issues

    def _check_timing_gaps(
        self,
        sync_map: Dict[str, Any],
        audio_duration: float
    ) -> List[SyncIssue]:
        """Check for long gaps with no visual changes"""
        issues = []
        max_gap = 5.0  # Maximum acceptable gap without visual change

        sync_points = sync_map.get("sync_points", [])
        if not sync_points:
            return issues

        # Sort by time
        sorted_points = sorted(sync_points, key=lambda x: x.get("time", 0))

        prev_time = 0
        for point in sorted_points:
            current_time = point.get("time", 0)
            gap = current_time - prev_time

            if gap > max_gap:
                issues.append(SyncIssue(
                    issue_type=SyncIssueType.TIMING_GAP,
                    severity="info",
                    timestamp=prev_time,
                    description=f"Long gap ({gap:.1f}s) with no visual changes",
                    suggested_fix="Consider adding intermediate visual cues"
                ))

            prev_time = current_time

        # Check gap at end
        if audio_duration - prev_time > max_gap:
            issues.append(SyncIssue(
                issue_type=SyncIssueType.TIMING_GAP,
                severity="info",
                timestamp=prev_time,
                description=f"Long gap at end of scene ({audio_duration - prev_time:.1f}s)",
                suggested_fix="Consider adding closing visual cue"
            ))

        return issues

    def _calculate_sync_score(self, issues: List[SyncIssue]) -> float:
        """Calculate overall sync score from issues"""
        if not issues:
            return 1.0

        # Deductions per severity
        deductions = {
            "critical": 0.3,
            "warning": 0.1,
            "info": 0.02
        }

        total_deduction = 0
        for issue in issues:
            total_deduction += deductions.get(issue.severity, 0.05)

        return max(0, 1.0 - total_deduction)

    def _generate_fixes(self, issues: List[SyncIssue]) -> List[Dict[str, Any]]:
        """Generate actionable fixes for issues"""
        fixes = []

        # Group issues by type
        critical_issues = [i for i in issues if i.severity == "critical"]

        for issue in critical_issues:
            if issue.issue_type == SyncIssueType.VISUAL_AFTER_AUDIO:
                fixes.append({
                    "action": "adjust_timing",
                    "target": "visual_cue",
                    "timestamp": issue.timestamp,
                    "fix": issue.suggested_fix
                })
            elif issue.issue_type == SyncIssueType.DURATION_MISMATCH:
                fixes.append({
                    "action": "adjust_duration",
                    "target": "animation",
                    "timestamp": issue.timestamp,
                    "fix": issue.suggested_fix
                })
            elif issue.issue_type == SyncIssueType.MISSING_VISUAL:
                fixes.append({
                    "action": "regenerate_visual",
                    "target": "visual_element",
                    "fix": issue.suggested_fix
                })

        return fixes
