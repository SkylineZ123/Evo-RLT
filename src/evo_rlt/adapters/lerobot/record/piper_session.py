"""Leader/follower handover for a Piper leader arm.

A Piper leader cannot be made backdrivable from software: the operator presses the teach
button on the arm. So the host must never let either arm jump onto the other's pose at
full speed. Before recording starts, both arms are brought together here with a slow
interpolation:

* the operator will teleoperate first -> the follower is ramped onto the leader, then the
  leader is released to the operator (teach button to drag);
* the policy will drive first -> the leader is ramped onto the follower through
  ``send_feedback``, so the first policy-synced command is a small step, not a jump.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from lerobot.processor import RobotAction

log = logging.getLogger(__name__)


def is_piper_leader(teleop: Any) -> bool:
    return callable(getattr(teleop, "begin_alignment", None)) and callable(
        getattr(teleop, "release_to_operator", None)
    )


def _interpolate(current: RobotAction, target: RobotAction, t: float) -> RobotAction:
    return {key: current[key] * (1.0 - t) + target[key] * t if key in target else current[key] for key in current}


def follower_smooth_move_to(
    robot: Any, current: RobotAction, target: RobotAction, duration_s: float = 3.0, fps: int = 30
) -> None:
    """Interpolate the follower from ``current`` to ``target`` (both in robot action keys)."""
    steps = max(int(duration_s * fps), 1)
    for step in range(steps + 1):
        robot.send_action(_interpolate(current, target, step / steps))
        time.sleep(1.0 / fps)


def leader_smooth_move_to(
    teleop: Any, current: RobotAction, target: RobotAction, duration_s: float = 3.0, fps: int = 30
) -> None:
    """Interpolate a commanded leader from ``current`` to ``target`` through ``send_feedback``."""
    steps = max(int(duration_s * fps), 1)
    for step in range(steps + 1):
        teleop.send_feedback(_interpolate(current, target, step / steps))
        time.sleep(1.0 / fps)


def _follower_pose(robot: Any, keys: list[str]) -> RobotAction:
    obs = robot.get_observation()
    missing = [key for key in keys if key not in obs]
    if missing:
        raise ValueError(f"Follower observation lacks leader action keys {missing}")
    return {key: float(obs[key]) for key in keys}


def prepare_piper_leader(
    robot: Any,
    teleop: Any,
    *,
    operator_first: bool,
    align_time_s: float = 3.0,
    fps: int = 30,
) -> None:
    """Bring a Piper leader and the follower onto one pose before the control loop starts.

    ``operator_first``: the session begins under teleoperation (teleop-only recording or
    ``--start-with-teleop``). Otherwise the policy drives first and the leader tracks it.
    """
    ramp_fps = max(20, fps)
    # Enabled follower role in CAN command mode: the leader holds its pose rigidly.
    teleop.begin_alignment()
    leader_pose = {key: float(value) for key, value in teleop.get_action().items()}
    follower_pose = _follower_pose(robot, list(leader_pose))
    if operator_first:
        log.info("Aligning follower to the Piper leader over %.1fs", align_time_s)
        follower_smooth_move_to(robot, follower_pose, leader_pose, duration_s=align_time_s, fps=ramp_fps)
        teleop.release_to_operator()
        log.info("Leader released: press the teach button on the leader arm to drag it.")
        return
    log.info("Aligning the Piper leader to the follower over %.1fs", align_time_s)
    leader_smooth_move_to(teleop, leader_pose, follower_pose, duration_s=align_time_s, fps=ramp_fps)
