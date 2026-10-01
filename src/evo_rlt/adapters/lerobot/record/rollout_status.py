"""Status-window content and key legend for policy rollouts (`evo-rlt-record collect|segment|full`).

The window itself is :class:`~evo_rlt.adapters.lerobot.record.status_view.StatusView`; this module
only turns the recording loop's state into the dict it draws, so the mapping can be tested
without OpenCV or hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CONTROL_VLA = "VLA"
CONTROL_RLT = "RLT"
CONTROL_HUMAN = "HUMAN"
CONTROL_RESET = "RESET"

# Requests that wait for a Piper leader to leave teach mode before they run.
HANDOVER_RL_START = "RL start"
HANDOVER_RELEASE = "hand back to policy"

_CONTROL_COLORS = {
    CONTROL_VLA: "white",
    CONTROL_RLT: "green",
    CONTROL_HUMAN: "yellow",
    CONTROL_RESET: "grey",
}


def rollout_controls_help(rl_key: str | None = "r", intervention_key: str | None = "space") -> str:
    """One-line key legend shared by the startup summary and the status window.

    ``None`` leaves a key out, for sessions without RL phases or without a leader arm.
    """
    keys = [f"{rl_key}=start RL"] if rl_key else []
    keys += ["s=success", "f=failure"]
    if intervention_key:
        keys.append(f"{intervention_key}=intervene/hand back")
    keys += ["<-=re-record", "Esc=stop"]
    return ", ".join(keys)


@dataclass
class RolloutStatus:
    task: str
    control: str
    critical: bool
    recording: bool
    step: int
    elapsed_s: float
    fps: float | None
    episode_index: int | None
    # None when the leader has no teach button (SO-series) or there is no leader.
    teach_mode: bool | None = None
    queued: str | None = None
    queued_key: str | None = None
    # Segment recording before the operator pressed r: frames are not written yet.
    waiting_for_rl: bool = False
    saved_episodes: int | None = None
    saved_frames: int | None = None
    successes: int = 0
    failures: int = 0
    last_outcome: str | None = None
    controls_help: str = rollout_controls_help()


def build_rollout_status(status: RolloutStatus) -> dict[str, Any]:
    """The dict :meth:`StatusView.update` expects, with the rollout lines in ``lines``."""
    control = status.control
    if status.critical and control in (CONTROL_VLA, CONTROL_RLT):
        control = f"{control} (critical segment)"
    lines: list[tuple[str, str, float]] = [
        (f"CONTROL: {control}", _CONTROL_COLORS.get(status.control, "white"), 0.9)
    ]
    if status.teach_mode is not None:
        if status.teach_mode:
            lines.append(("LEADER TEACH MODE: ON", "red", 0.6))
        else:
            lines.append(("leader teach mode: off", "grey", 0.5))
    if status.queued:
        cancel = f" ({status.queued_key} again cancels)" if status.queued_key else ""
        lines.append(
            (f"QUEUED: {status.queued} - press the teach button to leave teach mode{cancel}", "yellow", 0.55)
        )
    if status.waiting_for_rl:
        lines.append(("not recording yet - press r to start the RL segment", "grey", 0.5))
    session = f"this session: {status.successes} success | {status.failures} failure"
    if status.last_outcome:
        session += f" | last: {status.last_outcome}"
    lines.append((session, "white", 0.5))
    lines.append((status.controls_help, "grey", 0.45))

    return {
        "task": status.task,
        "step": status.step,
        "fps": status.fps,
        "elapsed": status.elapsed_s,
        "recording": status.recording,
        "episode_index": status.episode_index,
        "buffered_frames": status.step,
        "saved_episodes": status.saved_episodes,
        "saved_frames": status.saved_frames,
        "lines": lines,
    }
