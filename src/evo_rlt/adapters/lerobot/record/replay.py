"""Replay recorded Piper episodes on the follower to check a dataset's quality.

    evo-rlt-record replay --setup-json configs/record/piper_setup.json \\
        --dataset-root ~/lerobot_datasets/piper/0924_piper_teleop/teleop_190416 --episodes 0 1

Every selected episode is first checked offline (empty, non-finite values, joint jumps between
consecutive frames larger than ``--max-step-deg``); flagged episodes are never sent to the arm.
``--dry-run`` stops after this check and never touches the hardware.

On the robot, the follower is ramped slowly onto the episode's first frame, then the recorded
``action`` (or ``observation.state`` with ``--source state``) is streamed at the dataset fps
times ``--speed``. The joints measured each tick are compared with the recorded
``observation.state`` of the same frame: a large tracking error means the stored actions do not
reproduce what the follower did while recording (lag, dropped frames, a bad mapping).

Only the follower is used; the leader and the cameras stay off. Ctrl+C stops the replay, and
the follower goes to its safe pose before it is disabled, as after recording.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evo_rlt.adapters.lerobot.record.common import is_piper_setup, load_robot_setup, set_offline_env

log = logging.getLogger(__name__)

PIPER_JOINT_KEYS = [f"joint_{i}.pos" for i in range(1, 8)]
# joint_1..joint_6 are radians; the last entry is the gripper opening in meters.
ARM_JOINTS = 6


@dataclass(frozen=True)
class Episode:
    index: int
    label: str | None
    actions: np.ndarray  # (frames, 7) recorded `action`
    states: np.ndarray  # (frames, 7) recorded `observation.state`

    def commands(self, source: str) -> np.ndarray:
        return self.actions if source == "action" else self.states


def load_episodes(dataset_root: Path, episodes: list[int] | None) -> tuple[int, list[Episode]]:
    """Read the joint columns of *episodes* (all when None) without decoding any video."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_STATE

    # LeRobotDataset creates a missing root and then tries the Hub; fail on a typo instead.
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"{dataset_root} is not a LeRobot dataset (no meta/info.json)")
    dataset = LeRobotDataset(f"local/{dataset_root.name}", root=dataset_root, download_videos=False)
    for key in (ACTION, OBS_STATE):
        names = dataset.features.get(key, {}).get("names")
        if names != PIPER_JOINT_KEYS:
            raise ValueError(
                f"{dataset_root}: `{key}` names are {names}, expected Piper joints {PIPER_JOINT_KEYS}"
            )

    total = dataset.meta.total_episodes
    if total == 0:
        raise ValueError(f"{dataset_root} has no saved episodes")
    selected = list(range(total)) if episodes is None else list(episodes)
    missing = [index for index in selected if not 0 <= index < total]
    if missing:
        raise ValueError(f"Episodes {missing} do not exist; {dataset_root} has episodes 0..{total - 1}")

    columns = dataset.hf_dataset.select_columns([ACTION, OBS_STATE, "episode_index"]).with_format("numpy")[:]
    loaded = []
    for index in selected:
        rows = columns["episode_index"] == index
        loaded.append(
            Episode(
                index=index,
                label=dataset.meta.episodes[index].get("episode_success"),
                actions=columns[ACTION][rows].astype(np.float64),
                states=columns[OBS_STATE][rows].astype(np.float64),
            )
        )
    return dataset.fps, loaded


def find_problems(commands: np.ndarray, max_step_rad: float) -> list[str]:
    """Reasons not to send *commands* to the arm; empty when the episode is safe to replay."""
    if len(commands) == 0:
        return ["no frames"]
    if not np.isfinite(commands).all():
        return ["non-finite values in the commands"]
    steps = np.abs(np.diff(commands[:, :ARM_JOINTS], axis=0))
    if steps.size and steps.max() > max_step_rad:
        frame, joint = np.unravel_index(steps.argmax(), steps.shape)
        return [
            f"joint_{joint + 1} jumps {math.degrees(steps[frame, joint]):.1f} deg between frames "
            f"{frame} and {frame + 1} (limit {math.degrees(max_step_rad):.1f}, see --max-step-deg)"
        ]
    return []


def describe_episode(episode: Episode, fps: int, source: str) -> str:
    frames = len(episode.actions)
    text = f"episode {episode.index}: {frames} frames ({frames / fps:.1f}s), label={episode.label or '-'}"
    commands = episode.commands(source)
    if frames < 2 or not (np.isfinite(episode.actions).all() and np.isfinite(episode.states).all()):
        return text
    steps = np.abs(np.diff(commands[:, :ARM_JOINTS], axis=0)).max(axis=0)
    gap = np.abs(episode.actions - episode.states).max(axis=0)
    step_joint, gap_joint = int(steps.argmax()), int(gap[:ARM_JOINTS].argmax())
    return (
        f"{text}, max step {math.degrees(steps[step_joint]):.1f} deg/frame (joint_{step_joint + 1}), "
        f"max |action-state| {math.degrees(gap[gap_joint]):.1f} deg (joint_{gap_joint + 1}) "
        f"/ gripper {gap[ARM_JOINTS] * 1000:.1f} mm"
    )


def replay_episode(robot, commands: np.ndarray, *, fps: float) -> np.ndarray:
    """Stream *commands* at *fps*; returns the joints measured right before each command.

    Same order as the recording loop (observe, then act), so row ``t`` lines up with the
    recorded ``observation.state`` of frame ``t``.
    """
    from lerobot.utils.robot_utils import precise_sleep

    measured = np.empty_like(commands)
    for frame, command in enumerate(commands):
        tick_t = time.perf_counter()
        obs = robot.get_observation()
        measured[frame] = [obs[key] for key in PIPER_JOINT_KEYS]
        robot.send_action(dict(zip(PIPER_JOINT_KEYS, command.tolist(), strict=True)))
        precise_sleep(max(1.0 / fps - (time.perf_counter() - tick_t), 0.0))
    return measured


def tracking_error(measured: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint RMSE and max absolute error."""
    error = np.abs(measured - reference)
    return np.sqrt(np.mean(error**2, axis=0)), error.max(axis=0)


def _display_units(values: np.ndarray) -> str:
    """Arm joints in degrees, the gripper in millimeters."""
    arm = [f"{value:8.2f}" for value in np.degrees(values[:ARM_JOINTS])]
    return " ".join([*arm, f"{values[ARM_JOINTS] * 1000:8.1f}"])


def print_tracking_report(results: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    if not results:
        return
    header = " ".join([*(f"{f'joint_{i}':>8}" for i in range(1, ARM_JOINTS + 1)), f"{'gripper':>8}"])
    print("\nTracking error vs recorded observation.state (joints: deg, gripper: mm)")
    print(f"{'episode':>7} {'metric':<6} {header}")
    for index, rmse, max_abs in results:
        print(f"{index:>7} {'rmse':<6} {_display_units(rmse)}")
        print(f"{'':>7} {'max':<6} {_display_units(max_abs)}")


def _ask_before(episode: Episode) -> str:
    prompt = f"\nEpisode {episode.index}: reset the scene, then Enter to replay (s=skip, q=quit): "
    return input(prompt).strip().lower()


def run_replay(args: argparse.Namespace) -> None:
    set_offline_env()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not 0 < args.speed <= 1:
        raise ValueError(f"--speed must be in (0, 1], got {args.speed}")
    setup = load_robot_setup(args.setup_json)
    if not is_piper_setup(setup):
        raise ValueError("`evo-rlt-record replay` supports robot_type=piper manifests only")

    from evo_rlt.adapters.lerobot.record.teleop_collect import build_piper_follower_config

    dataset_root = Path(args.dataset_root).expanduser()
    fps, episodes = load_episodes(dataset_root, args.episodes)
    replay_fps = fps * args.speed
    print("\nPiper replay")
    print(f"Dataset: {dataset_root}")
    print(f"Follower: {setup.followers[0]['port']}  Source: {args.source}  "
          f"fps: {replay_fps:g} ({args.speed:g}x)")

    replayable = []
    for episode in episodes:
        print(describe_episode(episode, fps, args.source))
        problems = find_problems(episode.commands(args.source), math.radians(args.max_step_deg))
        for problem in problems:
            print(f"  !! {problem} -> skipped")
        if not problems:
            replayable.append(episode)

    robot_cfg = build_piper_follower_config(setup, with_cameras=False)
    if args.dry_run:
        print(f"\nDry run robot config: {robot_cfg}")
        return
    if not replayable:
        log.warning("No episode passed the offline check; nothing to replay.")
        return

    from evo_rlt.adapters.lerobot.hardware.piper import Piper

    save_dir = Path(args.save_trace).expanduser() if args.save_trace else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    robot = Piper(robot_cfg)
    results = []
    try:
        robot.connect(calibrate=False)
        for episode in replayable:
            if args.confirm:
                reply = _ask_before(episode)
                if reply == "q":
                    break
                if reply == "s":
                    continue
            commands = episode.commands(args.source)
            log.info("Moving to the first frame of episode %d", episode.index)
            robot.bus.move_to_joint_smoothly(commands[0].tolist(), duration_s=args.approach_time_s)
            log.info("Replaying episode %d: %d frames at %g fps", episode.index, len(commands), replay_fps)
            measured = replay_episode(robot, commands, fps=replay_fps)
            results.append((episode.index, *tracking_error(measured, episode.states)))
            if save_dir is not None:
                np.savez(
                    save_dir / f"episode_{episode.index:03d}.npz",
                    command=commands,
                    recorded_state=episode.states,
                    measured_state=measured,
                    fps=replay_fps,
                )
    except KeyboardInterrupt:
        log.warning("Interrupted; stopping the replay.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        print_tracking_report(results)
