from __future__ import annotations

import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

DEFAULT_SETUP_PATH = Path.home() / ".roboclaw/workspace/embodied/manifest.json"
DEFAULT_DATASET_ROOT = Path.home() / ".roboclaw/workspace/embodied/datasets"


def load_setup_json(path: str | None = None) -> dict[str, Any]:
    setup_path = Path(path).expanduser() if path else DEFAULT_SETUP_PATH
    with open(setup_path) as fh:
        return json.load(fh)


def resolve_dataset_root(setup: dict[str, Any]) -> Path:
    dataset_root = setup.get("datasets", {}).get("root", "")
    if not dataset_root:
        return DEFAULT_DATASET_ROOT
    return Path(dataset_root).expanduser()


def get_sorted_followers(setup: dict[str, Any]) -> list[dict[str, Any]]:
    followers = [arm for arm in setup["arms"] if "follower" in arm["type"]]
    followers.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return followers


def get_sorted_leaders(setup: dict[str, Any]) -> list[dict[str, Any]]:
    leaders = [arm for arm in setup["arms"] if "leader" in arm["type"]]
    leaders.sort(key=lambda arm: 0 if "left" in arm.get("alias", "") else 1)
    return leaders

log = logging.getLogger(__name__)

CAMERA_RENAME = {"left_wrist": "wrist", "right_wrist": "wrist", "right_front": "front"}
LEFT_CAMERA_ALIASES = {"left_wrist"}
RIGHT_CAMERA_ALIASES = {"right_wrist", "right_front"}
TELEOP_ID = "bimanual_leader"
PIPER_ROBOT_ID = "piper_follower"
PIPER_TELEOP_ID = "piper_leader"

ROBOT_TYPE_BI_SO = "bi_so"
ROBOT_TYPE_PIPER = "piper"
ROBOT_TYPES = (ROBOT_TYPE_BI_SO, ROBOT_TYPE_PIPER)

# `--resume` with no directory: continue the most recent dataset of this tag.
RESUME_LATEST = "latest"


@dataclass(frozen=True)
class RobotSetup:
    setup: dict[str, Any]
    followers: list[dict[str, Any]]
    leaders: list[dict[str, Any]]
    left_cameras: dict[str, Any]
    right_cameras: dict[str, Any]
    robot_type: str = ROBOT_TYPE_BI_SO
    # Flat camera dict for single-arm robots, keyed by the manifest alias.
    cameras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunPaths:
    dataset_name: str
    dataset_root: Path
    day_dir: Path
    log_file: Path
    # Set when appending to an existing dataset (`--resume`): episodes and tasks it already holds.
    resumed_episodes: int | None = None
    resumed_tasks: tuple[str, ...] = ()

    @property
    def resume(self) -> bool:
        return self.resumed_episodes is not None


def get_robot_type(setup: Any) -> str:
    """Robot family of a manifest (`dict`) or a loaded `RobotSetup`; `bi_so` when unset."""
    if isinstance(setup, dict):
        robot_type = setup.get("robot_type", ROBOT_TYPE_BI_SO)
    else:
        robot_type = getattr(setup, "robot_type", ROBOT_TYPE_BI_SO)
    if robot_type not in ROBOT_TYPES:
        raise ValueError(f"Unsupported robot_type {robot_type!r}; expected one of {ROBOT_TYPES}")
    return robot_type


def is_piper_setup(setup: Any) -> bool:
    return get_robot_type(setup) == ROBOT_TYPE_PIPER


def load_robot_setup(setup_json: str | None) -> RobotSetup:
    setup = load_setup_json(setup_json)
    followers = get_sorted_followers(setup)
    leaders = get_sorted_leaders(setup)
    if is_piper_setup(setup):
        if len(followers) != 1:
            raise ValueError(f"Piper setup needs exactly 1 follower arm, got {len(followers)}")
        if len(leaders) > 1:
            raise ValueError(f"Piper setup supports at most 1 leader arm, got {len(leaders)}")
        return RobotSetup(
            setup,
            followers,
            leaders,
            {},
            {},
            robot_type=ROBOT_TYPE_PIPER,
            cameras=build_flat_camera_configs(setup.get("cameras", [])),
        )
    if len(followers) < 2:
        raise ValueError(f"Need at least 2 follower arms, got {len(followers)}")
    left_cameras, right_cameras = build_camera_configs(setup.get("cameras", []))
    return RobotSetup(setup, followers, leaders, left_cameras, right_cameras)


def build_camera_configs(cameras: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    left_cameras: dict[str, Any] = {}
    right_cameras: dict[str, Any] = {}
    for camera in cameras:
        alias = camera["alias"]
        camera_config = build_camera_config(camera)
        target_name = CAMERA_RENAME.get(alias, alias)
        if alias in LEFT_CAMERA_ALIASES:
            left_cameras[target_name] = camera_config
        elif alias in RIGHT_CAMERA_ALIASES:
            right_cameras[target_name] = camera_config
    return left_cameras, right_cameras


def build_camera_config(camera: dict[str, Any]) -> dict[str, Any]:
    """One manifest camera entry -> a LeRobot camera config dict (opencv or realsense)."""
    camera_type = camera.get("type", "opencv")
    if camera_type in {"realsense", "intelrealsense"}:
        serial = camera.get("serial", camera.get("port"))
        if serial is None:
            raise ValueError(f"RealSense camera {camera.get('alias')!r} needs a 'serial' entry")
        config: dict[str, Any] = {
            "type": "intelrealsense",
            "serial_number_or_name": str(serial),
            "width": camera.get("width", 640),
            "height": camera.get("height", 480),
            "fps": camera.get("fps", 30),
        }
        if camera.get("use_depth"):
            config["use_depth"] = True
        return config
    if camera_type != "opencv":
        raise ValueError(f"Unsupported camera type {camera_type!r} for {camera.get('alias')!r}")
    config = {
        "type": "opencv",
        "index_or_path": camera["port"],
        "width": camera.get("width", 640),
        "height": camera.get("height", 480),
        "fps": camera.get("fps", 30),
    }
    if camera.get("fourcc"):
        config["fourcc"] = camera["fourcc"]
    return config


def build_flat_camera_configs(cameras: list[dict[str, Any]]) -> dict[str, Any]:
    """Single-arm robots: every camera keeps its manifest alias as the feature name."""
    return {camera["alias"]: build_camera_config(camera) for camera in cameras}


def resolve_run_paths(setup: dict[str, Any], dataset_tag: str, dataset_prefix: str) -> RunPaths:
    now = datetime.now()
    date_folder = f"{now:%m%d}_{dataset_tag}"
    dataset_leaf = f"{dataset_prefix}_{now:%H%M%S}"
    day_dir = resolve_dataset_root(setup) / date_folder
    dataset_root = day_dir / dataset_leaf
    return RunPaths(
        dataset_name=f"local/{dataset_leaf}",
        dataset_root=dataset_root,
        day_dir=day_dir,
        log_file=day_dir / f"{dataset_leaf}.log",
    )


def _check_resumable_files(root: Path, info: dict[str, Any]) -> None:
    """Refuse a dataset whose last session did not finalize, before any hardware starts.

    A session killed before `finalize()` leaves parquet files without a footer, or fewer
    episode rows than `info.json` counts; appending to it yields a dataset that cannot be read.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    for subdir, expected in (("data", info["total_frames"]), ("meta/episodes", info["total_episodes"])):
        rows = 0
        for path in sorted((root / subdir).glob("*/*.parquet")):
            try:
                rows += pq.read_metadata(path).num_rows
            except pa.ArrowException as error:
                raise ValueError(
                    f"--resume: cannot read {path} ({error}); the previous session was probably not finalized"
                ) from error
        if rows != expected:
            raise ValueError(
                f"--resume: {root / subdir} holds {rows} rows, but meta/info.json expects {expected}"
            )


def resolve_resume_paths(dataset_root: str | Path) -> RunPaths:
    """Paths that append to an existing, finalized dataset instead of creating a new one."""
    import pandas as pd

    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"--resume: {root} is not a LeRobot dataset (no meta/info.json)")
    info = json.loads(info_path.read_text())
    if not info.get("total_episodes"):
        raise ValueError(f"--resume: {root} has no saved episodes; start a new recording instead")
    _check_resumable_files(root, info)
    return RunPaths(
        dataset_name=f"local/{root.name}",
        dataset_root=root,
        day_dir=root.parent,
        # Same file as the first session's log; logging appends to it.
        log_file=root.parent / f"{root.name}.log",
        resumed_episodes=int(info["total_episodes"]),
        resumed_tasks=tuple(pd.read_parquet(root / "meta" / "tasks.parquet").index),
    )


def find_latest_dataset(setup: dict[str, Any], dataset_tag: str, dataset_prefix: str) -> Path:
    """Most recently written dataset with saved episodes at `<root>/<MMDD>_<tag>/<prefix>_<HHMMSS>`."""
    day_dir = re.compile(rf"\d{{4}}_{re.escape(dataset_tag)}")
    leaf = re.compile(rf"{re.escape(dataset_prefix)}_\d{{6}}")
    root = resolve_dataset_root(setup)
    candidates = []
    for info_path in root.glob("*/*/meta/info.json"):
        dataset_dir = info_path.parent.parent
        if not (day_dir.fullmatch(dataset_dir.parent.name) and leaf.fullmatch(dataset_dir.name)):
            continue
        # Sessions quit before their first save leave empty datasets behind; skip those.
        if json.loads(info_path.read_text()).get("total_episodes"):
            candidates.append((info_path.stat().st_mtime, dataset_dir))
    if not candidates:
        raise FileNotFoundError(
            f"--resume: no dataset with saved episodes at "
            f"{root}/<MMDD>_{dataset_tag}/{dataset_prefix}_<HHMMSS>; "
            "pass the dataset directory, or drop --resume to start a new one"
        )
    return max(candidates)[1]


def resolve_record_paths(
    setup: dict[str, Any], dataset_tag: str, dataset_prefix: str, resume: str | None
) -> RunPaths:
    """New timestamped dataset paths, or an existing dataset's when *resume* is set.

    *resume* is a dataset directory, or `RESUME_LATEST` for this tag's most recent dataset.
    """
    if not resume:
        return resolve_run_paths(setup, dataset_tag, dataset_prefix)
    if resume == RESUME_LATEST:
        return resolve_resume_paths(find_latest_dataset(setup, dataset_tag, dataset_prefix))
    return resolve_resume_paths(resume)


def print_dataset_target(paths: RunPaths, task: str) -> None:
    print(f"Dataset: {paths.dataset_name} -> {paths.dataset_root}")
    if not paths.resume:
        return
    print(f"Resume: appending after {paths.resumed_episodes} saved episodes")
    if task not in paths.resumed_tasks:
        print(f"WARNING: task {task!r} is new to this dataset; "
              f"it was recorded with {list(paths.resumed_tasks)}")


def configure_logging(log_file: Path, log_level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)


def remove_existing_dataset(dataset_root: Path) -> None:
    if dataset_root.exists():
        log.info("Removing existing dataset dir: %s", dataset_root)
        shutil.rmtree(dataset_root)


def stage_arm_calibration(arm: dict[str, Any], dst: Path) -> None:
    serial = Path(arm["calibration_dir"]).name
    src = Path(arm["calibration_dir"]).expanduser() / f"{serial}.json"
    if src.exists():
        shutil.copy2(src, dst)
        log.info("Calibration staged: %s -> %s", src, dst)
        return
    log.warning("Calibration file not found: %s", src)


def stage_follower_calibrations(followers: list[dict[str, Any]], cal_dir: str) -> None:
    for side, arm in (("left", followers[0]), ("right", followers[1])):
        stage_arm_calibration(arm, Path(cal_dir) / f"bimanual_{side}.json")


def build_teleop_argv(
    leaders: list[dict[str, Any]], no_teleop: bool, robot_type: str = ROBOT_TYPE_BI_SO
) -> list[str]:
    if no_teleop:
        log.warning("Teleop disabled by --no-teleop")
        return []
    if robot_type == ROBOT_TYPE_PIPER:
        return build_piper_teleop_argv(leaders)
    if len(leaders) < 2:
        log.warning("Teleop disabled: need 2 leader arms, got %d", len(leaders))
        return []
    log.info("Teleop enabled: left=%s, right=%s", leaders[0]["port"], leaders[1]["port"])
    return [
        "--teleop.type=bi_so_leader",
        f"--teleop.left_arm_config.port={leaders[0]['port']}",
        "--teleop.left_arm_config.use_degrees=true",
        f"--teleop.right_arm_config.port={leaders[1]['port']}",
        "--teleop.right_arm_config.use_degrees=true",
        f"--teleop.id={TELEOP_ID}",
    ]


def build_piper_teleop_argv(leaders: list[dict[str, Any]]) -> list[str]:
    if not leaders:
        log.warning("Teleop disabled: Piper setup has no leader arm")
        return []
    leader = leaders[0]
    log.info("Teleop enabled: piper_leader on %s", leader["port"])
    # Same-model leader/follower share one key space (radians + gripper meters), so the
    # leader runs in absolute passthrough and needs no calibration file.
    argv = [
        "--teleop.type=piper_leader",
        f"--teleop.port={leader['port']}",
        f"--teleop.id={PIPER_TELEOP_ID}",
        "--teleop.require_calibration=false",
    ]
    for key in ("arm_model", "firmware_version", "gripper_force_n"):
        if key in leader:
            argv.append(f"--teleop.{key}={leader[key]}")
    return argv


def stage_setup_follower_calibrations(setup: RobotSetup, cal_dir: str) -> None:
    """Stage SO follower calibrations; Piper followers are position-controlled and need none."""
    if is_piper_setup(setup):
        return
    stage_follower_calibrations(setup.followers, cal_dir)


def stage_leader_calibrations(
    leaders: list[dict[str, Any]], teleop_argv: list[str]
) -> TemporaryDirectory[str] | None:
    if not teleop_argv:
        return None
    if "--teleop.type=piper_leader" in teleop_argv:
        return None
    leader_cal_dir = TemporaryDirectory(prefix="record-leader-cal-")
    for side, arm in (("left", leaders[0]), ("right", leaders[1])):
        stage_arm_calibration(arm, Path(leader_cal_dir.name) / f"{TELEOP_ID}_{side}.json")
    teleop_argv.append(f"--teleop.calibration_dir={leader_cal_dir.name}")
    return leader_cal_dir


def build_robot_argv(
    followers: list[dict[str, Any]],
    left_cameras: dict[str, Any],
    right_cameras: dict[str, Any],
    cal_dir: str,
) -> list[str]:
    return [
        "--robot.type=bi_so_follower",
        "--robot.id=bimanual",
        f"--robot.calibration_dir={cal_dir}",
        f"--robot.left_arm_config.port={followers[0]['port']}",
        "--robot.left_arm_config.use_degrees=true",
        f"--robot.left_arm_config.cameras={json.dumps(left_cameras)}",
        f"--robot.right_arm_config.port={followers[1]['port']}",
        "--robot.right_arm_config.use_degrees=true",
        f"--robot.right_arm_config.cameras={json.dumps(right_cameras)}",
    ]


def build_piper_robot_argv(follower: dict[str, Any], cameras: dict[str, Any]) -> list[str]:
    argv = [
        "--robot.type=piper",
        f"--robot.id={PIPER_ROBOT_ID}",
        f"--robot.can_port={follower['port']}",
        f"--robot.cameras={json.dumps(cameras)}",
    ]
    for key in ("arm_model", "firmware_version", "speed_percent", "gripper_force_n"):
        if key in follower:
            argv.append(f"--robot.{key}={follower[key]}")
    if "home_position" in follower:
        argv.append(f"--robot.home_position={json.dumps(follower['home_position'])}")
    return argv


def build_setup_robot_argv(setup: RobotSetup, cal_dir: str) -> list[str]:
    if is_piper_setup(setup):
        return build_piper_robot_argv(setup.followers[0], setup.cameras)
    return build_robot_argv(setup.followers, setup.left_cameras, setup.right_cameras, cal_dir)


def build_dataset_argv(
    *,
    dataset_name: str,
    dataset_root: Path,
    task: str,
    num_episodes: int,
    episode_time_s: int,
    fps: int,
    vcodec: str,
    resume: bool = False,
) -> list[str]:
    return [
        *(["--resume=true"] if resume else []),
        f"--dataset.repo_id={dataset_name}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={task}",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.fps={fps}",
        f"--dataset.vcodec={vcodec}",
        "--dataset.push_to_hub=false",
        f"--dataset.video_encoding_batch_size={num_episodes + 1}",
        "--dataset.streaming_encoding=true",
    ]


def build_policy_overrides(
    *,
    policy_path: str | None,
    vla_path: str | None,
    rl_token_path: str | None,
    phase_mode: str | None = None,
    chunk_exec_steps: int | None = None,
) -> list[str]:
    if policy_path is None:
        return []
    overrides = [f"--policy.path={policy_path}"]
    if phase_mode is not None:
        overrides.append(f"--policy.phase_mode={phase_mode}")
    if chunk_exec_steps is not None:
        overrides.append(f"--policy.chunk_exec_steps={chunk_exec_steps}")
    if vla_path is not None:
        overrides.append(f"--policy.vla_pretrained_path={vla_path}")
    if rl_token_path is not None:
        overrides.append(f"--policy.rl_token_pretrained_path={rl_token_path}")
    return overrides


def build_rtc_argv(
    *,
    enabled: bool,
    execution_horizon: int,
    max_guidance_weight: float,
    prefix_attention_schedule: str,
    vla_execution_horizon: int | None,
    action_queue_size_to_get_new_actions: int | None,
) -> list[str]:
    argv = [
        f"--rlt.rtc_enabled={'true' if enabled else 'false'}",
        f"--rlt.rtc_execution_horizon={execution_horizon}",
        f"--rlt.rtc_max_guidance_weight={max_guidance_weight}",
        f"--rlt.rtc_prefix_attention_schedule={prefix_attention_schedule}",
    ]
    if vla_execution_horizon is not None:
        argv.append(f"--rlt.vla_rtc_execution_horizon={vla_execution_horizon}")
    if action_queue_size_to_get_new_actions is not None:
        argv.append(
            "--rlt.rtc_action_queue_size_to_get_new_actions="
            f"{action_queue_size_to_get_new_actions}"
        )
    return argv


def preflight_piper_connections(
    follower: dict[str, Any], leader: dict[str, Any] | None, *, timeout_s: float = 2.0
) -> None:
    """Check both CAN arms answer before the (slow) policy load, without enabling either.

    Passive on purpose: enabling and then disabling a follower that is not resting in its
    safe pose would let it drop. Opening the bus and reading one status frame proves the
    interface is up and the arm is powered.
    """
    from evo_rlt.adapters.lerobot.hardware.piper.agx_arm import make_piper_arm, read_piper_ctrl_mode

    arms = [("follower", follower)] + ([("leader", leader)] if leader is not None else [])
    for role, arm_cfg in arms:
        port = arm_cfg["port"]
        log.info("Preflight checking Piper %s on %s", role, port)
        arm, _gripper, _profile = make_piper_arm(
            channel=port,
            arm_model=arm_cfg.get("arm_model", "piper"),
            firmware_version=arm_cfg.get("firmware_version", "auto"),
            with_gripper=False,
        )
        try:
            if read_piper_ctrl_mode(arm, timeout_s=timeout_s) is None:
                raise ConnectionError(
                    f"Piper {role} on CAN '{port}' sent no status frame within {timeout_s:.1f}s; "
                    "check power, the CAN interface name and `ip link show`."
                )
        finally:
            arm.disconnect()
        log.info("Preflight Piper %s check passed", role)


def preflight_motor_connections(
    followers: list[dict[str, Any]],
    leaders: list[dict[str, Any]],
    cal_dir: str,
    leader_cal_dir: str | None,
) -> None:
    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
    from lerobot.robots.so_follower import SOFollowerConfig
    from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
    from lerobot.teleoperators.so_leader import SOLeaderConfig

    def disconnect(device: Any) -> None:
        for arm_name in ("left_arm", "right_arm"):
            arm = getattr(device, arm_name, None)
            if arm is not None and arm.is_connected:
                arm.disconnect()
        if getattr(device, "is_connected", False):
            device.disconnect()

    log.info("Preflight checking follower motor connections before loading policy")
    robot = BiSOFollower(
        BiSOFollowerConfig(
            id="bimanual",
            calibration_dir=Path(cal_dir),
            left_arm_config=SOFollowerConfig(port=followers[0]["port"], use_degrees=True),
            right_arm_config=SOFollowerConfig(port=followers[1]["port"], use_degrees=True),
        )
    )
    try:
        robot.connect(calibrate=True)
        log.info("Preflight follower motor check passed")
    finally:
        disconnect(robot)

    if not leaders or leader_cal_dir is None:
        return

    log.info("Preflight checking leader motor connections before loading policy")
    teleop = BiSOLeader(
        BiSOLeaderConfig(
            id=TELEOP_ID,
            calibration_dir=Path(leader_cal_dir),
            left_arm_config=SOLeaderConfig(port=leaders[0]["port"], use_degrees=True),
            right_arm_config=SOLeaderConfig(port=leaders[1]["port"], use_degrees=True),
        )
    )
    try:
        teleop.connect(calibrate=True)
        log.info("Preflight leader motor check passed")
    finally:
        disconnect(teleop)


def set_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
