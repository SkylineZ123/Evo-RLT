"""Interactive teleoperation recording for a Piper leader/follower pair.

Ported from the LeRobot 0.6 working copy's ``lerobot_record.interactive_record_loop``: the
operator, not a timer, decides when an episode starts, pauses, and is kept or dropped::

    c        start recording, or resume after a pause
    space    pause / resume within the episode (never splits it)
    s        save the buffered episode (labelled success) and go back to idle
    r        discard the buffer and go back to idle
    q / Esc  end the session

At startup the follower is ramped onto the leader's pose, then the leader is released to
the operator, who drags it with the teach button. The follower tracks the leader for the
whole session, recording or not, so the scene can be reset with the arms live.

Saved episodes carry ``episode_success="success"`` so the dataset can be fed straight into
``evo-rlt-build-bucket-cache --bucket-mode human_expert`` as well as pi0.5 SFT.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from evo_rlt.adapters.lerobot.record.common import (
    PIPER_ROBOT_ID,
    PIPER_TELEOP_ID,
    RobotSetup,
    configure_logging,
    is_piper_setup,
    load_robot_setup,
    print_dataset_target,
    remove_existing_dataset,
    resolve_record_paths,
    set_offline_env,
)

log = logging.getLogger(__name__)

TELEOP_CONTROLS_HELP = "c=start/resume, space=pause, s=save, r=discard, q=stop"
TELEOP_DATASET_PREFIX = "teleop"
EPISODE_SUCCESS_LABEL = "success"


def camera_config_from_dict(camera: dict[str, Any]):
    """Turn a camera dict from `record.common.build_camera_config` into a LeRobot config."""
    params = dict(camera)
    camera_type = params.pop("type")
    if camera_type == "intelrealsense":
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        return RealSenseCameraConfig(**params)
    if camera_type == "opencv":
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        return OpenCVCameraConfig(**params)
    raise ValueError(f"Unsupported camera type {camera_type!r}")


def build_piper_follower_config(setup: RobotSetup, *, with_cameras: bool = True):
    """Follower config for a Piper setup manifest (mirrors `build_piper_robot_argv`)."""
    from evo_rlt.adapters.lerobot.hardware.piper import PiperConfig

    follower = setup.followers[0]
    robot_kwargs = {key: follower[key] for key in ("arm_model", "firmware_version", "speed_percent",
                                                    "gripper_force_n", "home_position") if key in follower}
    cameras = setup.cameras if with_cameras else {}
    return PiperConfig(
        id=PIPER_ROBOT_ID,
        can_port=follower["port"],
        cameras={name: camera_config_from_dict(cam) for name, cam in cameras.items()},
        **robot_kwargs,
    )


def build_piper_configs(setup: RobotSetup):
    """Follower + leader configs for a Piper setup manifest (mirrors `build_piper_*_argv`)."""
    from evo_rlt.adapters.lerobot.hardware.piper import PiperLeaderConfig

    leader = setup.leaders[0]
    leader_kwargs = {key: leader[key] for key in ("arm_model", "firmware_version", "gripper_force_n")
                     if key in leader}
    robot_cfg = build_piper_follower_config(setup)
    teleop_cfg = PiperLeaderConfig(
        id=PIPER_TELEOP_ID,
        port=leader["port"],
        require_calibration=False,
        **leader_kwargs,
    )
    return robot_cfg, teleop_cfg


def _guard_press(controls):
    """Key backends swallow handler exceptions; log illegal transitions instead of going quiet."""

    def press(key: str) -> None:
        try:
            controls.press(key)
        except ValueError as error:
            logging.warning("%s", error)

    return press


def _announce(event: str, payload: dict) -> None:
    message = payload.get("message")
    if message:
        logging.info(message)


def interactive_teleop_loop(
    *,
    robot,
    teleop,
    controls,
    dataset,
    fps: int,
    task: str,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    num_episodes: int | None = None,
    status_view=None,
    forward_window_keys: bool = False,
) -> None:
    """Teleoperate at *fps* for the whole session; episodes are cut by the operator's keys.

    *num_episodes* counts saves in this session, so a resumed dataset gets that many more.
    """
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.robot_utils import precise_sleep

    if dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    window_key_handler = _guard_press(controls) if forward_window_keys else None
    session_start_episodes = dataset.num_episodes
    buffered = 0
    episode_start_t = time.perf_counter()
    measured_fps = float(fps)
    prev_tick_t: float | None = None

    while not controls.stop.is_set():
        tick_t = time.perf_counter()
        if prev_tick_t is not None and tick_t > prev_tick_t:
            measured_fps = 0.9 * measured_fps + 0.1 / (tick_t - prev_tick_t)
        prev_tick_t = tick_t

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        action_values = teleop_action_processor((teleop.get_action(), obs))
        robot.send_action(robot_action_processor((action_values, obs)))

        if controls.recording.is_set():
            frame = {
                **build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR),
                **build_dataset_frame(dataset.features, action_values, prefix=ACTION),
                "task": task,
            }
            dataset.add_frame(frame)
            buffered += 1

        if status_view is not None:
            status_view.update(
                obs_processed,
                {
                    "task": task,
                    "skill_text": "",
                    "recording": controls.recording.is_set(),
                    "step": buffered,
                    "elapsed": time.perf_counter() - episode_start_t,
                    "fps": measured_fps,
                    "episode_index": dataset.num_episodes,
                    "buffered_frames": buffered,
                    "saved_episodes": dataset.num_episodes,
                    "saved_frames": dataset.num_frames,
                },
            )
            if not status_view.render_once(window_key_handler):
                controls.apply("stop")

        if controls.save.is_set():
            controls.save.clear()
            if dataset.has_pending_frames():
                episode_index = dataset.num_episodes
                dataset.save_episode(extra_episode_metadata={"episode_success": EPISODE_SUCCESS_LABEL})
                logging.info("Saved episode %d (%d frames)", episode_index, buffered)
            else:
                logging.warning("Nothing to save — the episode buffer is empty.")
            buffered = 0
            episode_start_t = time.perf_counter()
            if num_episodes and dataset.num_episodes - session_start_episodes >= num_episodes:
                logging.info("Reached the episode target (%d)", num_episodes)
                controls.apply("stop")

        if controls.discard.is_set():
            controls.discard.clear()
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
                logging.info("Discarded episode (%d frames)", buffered)
            buffered = 0
            episode_start_t = time.perf_counter()

        precise_sleep(max(1.0 / fps - (time.perf_counter() - tick_t), 0.0))

    # Settled here, inside the caller's VideoEncodingManager, whose exit finalizes the dataset.
    if dataset.has_pending_frames():
        if controls.stop_pending == "save" and not controls.emergency.is_set():
            logging.info("Saving the last episode")
            dataset.save_episode(extra_episode_metadata={"episode_success": EPISODE_SUCCESS_LABEL})
        else:
            logging.info("Discarding %d unsaved frames (stop_pending=%s).", buffered, controls.stop_pending)
            dataset.clear_episode_buffer()


def open_teleop_dataset(paths, robot, features: dict, *, fps: int, vcodec: str):
    """Create the session's dataset, or reopen it for appending when `paths.resume` is set."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility

    writer_kwargs = {
        "image_writer_processes": 0,
        "image_writer_threads": 4 * len(robot.cameras),
        "batch_encoding_size": 1,
        "vcodec": vcodec,
        "streaming_encoding": True,
    }
    if not paths.resume:
        return LeRobotDataset.create(
            paths.dataset_name,
            fps,
            root=paths.dataset_root,
            robot_type=robot.name,
            features=features,
            use_videos=True,
            **writer_kwargs,
        )
    dataset = LeRobotDataset.resume(paths.dataset_name, root=paths.dataset_root, **writer_kwargs)
    try:
        # Same fps, robot type and features (camera set and resolution included).
        sanity_check_dataset_robot_compatibility(dataset, robot, fps, features)
    except ValueError:
        dataset.finalize()  # stop the image-writer threads resume() started
        raise
    return dataset


def print_teleop_summary(args: argparse.Namespace, setup: RobotSetup, paths) -> None:
    print("\nPiper teleoperation recording")
    print_dataset_target(paths, args.task)
    print(f"Log: {paths.log_file}")
    print(f"Follower: {setup.followers[0]['port']}  Leader: {setup.leaders[0]['port']}")
    print(f"Cameras: {', '.join(setup.cameras) or '(none)'}")
    print(f"Task: {args.task}")
    print(f"Episodes this session: {args.num_episodes}  fps: {args.fps}")
    print(f"Controls: {TELEOP_CONTROLS_HELP}")
    print("Drag the leader with its teach button (it lights up green).")


def run_teleop(args: argparse.Namespace) -> None:
    set_offline_env()
    setup = load_robot_setup(args.setup_json)
    if not is_piper_setup(setup):
        raise ValueError(
            "`evo-rlt-record teleop` supports robot_type=piper manifests; "
            "for bi-SO use `evo-rlt-record full --initial-source teleop`."
        )
    if not setup.leaders:
        raise ValueError("Teleoperation recording needs a leader arm in the setup manifest")
    paths = resolve_record_paths(setup.setup, args.dataset_tag, TELEOP_DATASET_PREFIX, args.resume)
    print_teleop_summary(args, setup, paths)
    if args.dry_run:
        robot_cfg, teleop_cfg = build_piper_configs(setup)
        print(f"\nDry run robot config: {robot_cfg}")
        print(f"Dry run teleop config: {teleop_cfg}")
        return

    configure_logging(paths.log_file, args.log_level)
    if not paths.resume:
        remove_existing_dataset(paths.dataset_root)

    if args.status_view:
        # Must load before PyAV (imported by lerobot.datasets below), or the window deadlocks;
        # see `status_view.shadowed_x11_libs`.
        import cv2  # noqa: F401

    from evo_rlt.adapters.lerobot.record.runner import prepare_lerobot_runtime

    # Registers the Piper plugin and the `save_episode(extra_episode_metadata=...)` patch.
    prepare_lerobot_runtime(background_episode_video_encoding=True)

    from lerobot.datasets.feature_utils import combine_feature_dicts
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.datasets.video_utils import VideoEncodingManager
    from lerobot.processor import make_default_processors

    from evo_rlt.adapters.lerobot.hardware.piper import Piper, PiperLeader
    from evo_rlt.adapters.lerobot.record.controls import RecordControls
    from evo_rlt.adapters.lerobot.record.key_input import create_key_listener
    from evo_rlt.adapters.lerobot.record.piper_session import prepare_piper_leader

    robot_cfg, teleop_cfg = build_piper_configs(setup)
    robot = Piper(robot_cfg)
    teleop = PiperLeader(teleop_cfg)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    listener = None
    status_view = None
    dataset = open_teleop_dataset(paths, robot, dataset_features, fps=args.fps, vcodec=args.vcodec)
    session_start_episodes = dataset.num_episodes
    try:
        # Leader first, so the follower is not left enabled and idle during leader init.
        teleop.connect(calibrate=False)
        robot.connect()
        prepare_piper_leader(robot, teleop, operator_first=True, align_time_s=args.align_time_s, fps=args.fps)

        controls = RecordControls(on_change=_announce)
        controls.stop_pending = args.stop_pending
        listener = create_key_listener(_guard_press(controls), controls_help=TELEOP_CONTROLS_HELP)
        if args.status_view:
            from evo_rlt.adapters.lerobot.record.status_view import StatusView

            status_view = StatusView(None, window_name=f"evo-rlt-record · {paths.dataset_name}").start()
        logging.info("Ready to record. Press c to start — %s", TELEOP_CONTROLS_HELP)

        with VideoEncodingManager(dataset):
            interactive_teleop_loop(
                robot=robot,
                teleop=teleop,
                controls=controls,
                dataset=dataset,
                fps=args.fps,
                task=args.task,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                num_episodes=args.num_episodes,
                status_view=status_view,
                forward_window_keys=not listener.captures_global_keys,
            )
    finally:
        if status_view is not None:
            status_view.stop()
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        logging.info(
            "Saved %d episodes this session (%d total) to %s",
            dataset.num_episodes - session_start_episodes,
            dataset.num_episodes,
            paths.dataset_root,
        )
