import json
from types import SimpleNamespace

import pytest

from evo_rlt.adapters.lerobot.record.common import (
    RobotSetup,
    build_setup_robot_argv,
    build_teleop_argv,
    load_robot_setup,
    stage_leader_calibrations,
)

PIPER_MANIFEST = {
    "robot_type": "piper",
    "arms": [
        {"alias": "follower", "type": "follower", "port": "can_f", "speed_percent": 50},
        {"alias": "leader", "type": "leader", "port": "can_l"},
    ],
    "cameras": [
        {"alias": "wrist", "type": "realsense", "serial": "148522072680", "width": 640, "height": 480, "fps": 30},
        {"alias": "front", "port": "/dev/video4", "fourcc": "MJPG"},
    ],
}
JOINTS = [f"joint_{i}.pos" for i in range(1, 8)]


def _write_manifest(tmp_path, manifest):
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(manifest))
    return str(path)


def test_piper_manifest_builds_piper_argv(tmp_path):
    setup = load_robot_setup(_write_manifest(tmp_path, PIPER_MANIFEST))
    assert setup.robot_type == "piper"
    assert setup.cameras["wrist"] == {
        "type": "intelrealsense",
        "serial_number_or_name": "148522072680",
        "width": 640,
        "height": 480,
        "fps": 30,
    }
    assert setup.cameras["front"]["type"] == "opencv"
    assert setup.cameras["front"]["fourcc"] == "MJPG"

    robot_argv = build_setup_robot_argv(setup, "/unused")
    assert robot_argv[:3] == ["--robot.type=piper", "--robot.id=piper_follower", "--robot.can_port=can_f"]
    assert "--robot.speed_percent=50" in robot_argv
    assert json.loads(robot_argv[3].split("=", 1)[1]) == setup.cameras

    teleop_argv = build_teleop_argv(setup.leaders, False, setup.robot_type)
    assert teleop_argv == [
        "--teleop.type=piper_leader",
        "--teleop.port=can_l",
        "--teleop.id=piper_leader",
        "--teleop.require_calibration=false",
    ]
    # Piper leaders run in absolute passthrough: nothing to stage, no calibration_dir arg.
    assert stage_leader_calibrations(setup.leaders, teleop_argv) is None
    assert not any(arg.startswith("--teleop.calibration_dir") for arg in teleop_argv)


def test_piper_manifest_rejects_two_followers(tmp_path):
    manifest = dict(PIPER_MANIFEST)
    manifest["arms"] = [*PIPER_MANIFEST["arms"], {"alias": "f2", "type": "follower", "port": "can_x"}]
    with pytest.raises(ValueError, match="exactly 1 follower"):
        load_robot_setup(_write_manifest(tmp_path, manifest))


def test_unknown_robot_type_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported robot_type"):
        load_robot_setup(_write_manifest(tmp_path, {**PIPER_MANIFEST, "robot_type": "ur5"}))


def test_bi_so_setup_without_robot_type_keeps_so_argv():
    setup = RobotSetup(setup={}, followers=[{"port": "l"}, {"port": "r"}], leaders=[], left_cameras={},
                       right_cameras={})
    assert build_setup_robot_argv(setup, "/cal")[0] == "--robot.type=bi_so_follower"


def test_piper_types_resolve_through_lerobot_factories():
    from lerobot.robots import RobotConfig, make_robot_from_config
    from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config

    from evo_rlt.adapters.lerobot import register
    from evo_rlt.adapters.lerobot.hardware.piper import Piper, PiperLeader

    register()
    robot = make_robot_from_config(RobotConfig.get_choice_class("piper")(can_port="can_f"))
    teleop = make_teleoperator_from_config(
        TeleoperatorConfig.get_choice_class("piper_leader")(port="can_l", require_calibration=False)
    )
    assert isinstance(robot, Piper)
    assert isinstance(teleop, PiperLeader)
    assert list(robot.action_features) == JOINTS


def test_teleop_dry_run_does_not_touch_hardware(tmp_path, capsys):
    from evo_rlt.adapters.lerobot.record.cli import main

    manifest = {**PIPER_MANIFEST, "datasets": {"root": str(tmp_path / "ds")}}
    main(["teleop", "--setup-json", _write_manifest(tmp_path, manifest), "--task", "t", "--dry-run"])
    out = capsys.readouterr().out
    assert "can_port='can_f'" in out
    assert "require_calibration=False" in out
    assert not (tmp_path / "ds").exists()


# --------------------------------------------------------------------------- handover


class FakeFollower:
    def __init__(self, pose):
        self.pose = dict(pose)
        self.sent = []

    def get_observation(self):
        return dict(self.pose)

    def send_action(self, action):
        self.sent.append(dict(action))
        self.pose.update(action)
        return action


class FakePiperLeader:
    def __init__(self, pose, teach=False):
        self.pose = dict(pose)
        self.teach = teach
        self.calls = []
        self.feedback = []
        self.manual = None

    def begin_alignment(self):
        self.calls.append("begin_alignment")

    def release_to_operator(self):
        self.calls.append("release_to_operator")

    def get_action(self):
        return dict(self.pose)

    def send_feedback(self, feedback):
        self.feedback.append(dict(feedback))
        self.pose.update(feedback)

    def set_manual_control(self, enabled):
        self.manual = enabled
        self.calls.append(f"manual={enabled}")

    def is_teach_mode_active(self):
        return self.teach


def _pose(value):
    return dict.fromkeys(JOINTS, value)


@pytest.fixture
def no_sleep(monkeypatch):
    from evo_rlt.adapters.lerobot.record import piper_session

    monkeypatch.setattr(piper_session.time, "sleep", lambda _s: None)


def test_operator_first_ramps_follower_onto_leader_then_releases(no_sleep):
    from evo_rlt.adapters.lerobot.record.piper_session import prepare_piper_leader

    robot, leader = FakeFollower(_pose(0.0)), FakePiperLeader(_pose(1.0))
    prepare_piper_leader(robot, leader, operator_first=True, align_time_s=1.0, fps=30)

    assert leader.calls == ["begin_alignment", "release_to_operator"]
    assert robot.sent[-1] == _pose(1.0)
    steps = [s["joint_1.pos"] for s in robot.sent]
    assert steps == sorted(steps) and max(b - a for a, b in zip(steps, steps[1:])) < 0.1
    assert leader.feedback == []


def test_policy_first_ramps_leader_onto_follower(no_sleep):
    from evo_rlt.adapters.lerobot.record.piper_session import prepare_piper_leader

    robot, leader = FakeFollower(_pose(0.5)), FakePiperLeader(_pose(-0.5))
    prepare_piper_leader(robot, leader, operator_first=False, align_time_s=1.0, fps=30)

    assert leader.calls == ["begin_alignment"]
    assert leader.feedback[-1] == _pose(0.5)
    assert robot.sent == []


# --------------------------------------------------------------------------- recorder


class FakeDataset:
    def __init__(self, fps=1000):
        self.fps = fps
        self.features = {
            "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINTS},
            "action": {"dtype": "float32", "shape": (7,), "names": JOINTS},
        }
        self.buffer = []
        self.episodes = []
        self.num_frames = 0

    @property
    def num_episodes(self):
        return len(self.episodes)

    def add_frame(self, frame):
        self.buffer.append(frame)

    def has_pending_frames(self):
        return bool(self.buffer)

    def save_episode(self, extra_episode_metadata=None):
        self.episodes.append((list(self.buffer), extra_episode_metadata))
        self.num_frames += len(self.buffer)
        self.buffer = []

    def clear_episode_buffer(self):
        self.buffer = []


class ScriptedFollower(FakeFollower):
    """Presses recorder keys at given ticks, as an operator would."""

    def __init__(self, script, controls):
        super().__init__(_pose(0.0))
        self.script = script
        self.controls = controls
        self.tick = 0

    def get_observation(self):
        from evo_rlt.adapters.lerobot.record.teleop_collect import _guard_press

        for key in self.script.get(self.tick, ()):
            _guard_press(self.controls)(key)
        self.tick += 1
        return super().get_observation()


def _identity(x):
    return x[0] if isinstance(x, tuple) else x


def _run_recorder(script, stop_pending="discard", num_episodes=None, existing_episodes=0):
    from evo_rlt.adapters.lerobot.record.controls import RecordControls
    from evo_rlt.adapters.lerobot.record.teleop_collect import interactive_teleop_loop

    controls = RecordControls()
    controls.stop_pending = stop_pending
    robot = ScriptedFollower(script, controls)
    dataset = FakeDataset()
    dataset.episodes = [([], {"episode_success": "success"})] * existing_episodes
    interactive_teleop_loop(
        robot=robot,
        teleop=FakePiperLeader(_pose(0.25)),
        controls=controls,
        dataset=dataset,
        fps=dataset.fps,
        task="insert",
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
        robot_observation_processor=_identity,
        num_episodes=num_episodes,
    )
    return robot, dataset


def test_recorder_keys_start_pause_save_discard():
    # tick: 0 c | 1,2 rec | 3 space | 4 paused | 5 space | 6 rec | 7 s | 8 c | 9 rec | 10 r | 11 q
    script = {0: ["c"], 3: ["space"], 5: ["space"], 7: ["s"], 8: ["c"], 10: ["r"], 11: ["q"]}
    robot, dataset = _run_recorder(script)

    assert len(dataset.episodes) == 1
    frames, meta = dataset.episodes[0]
    # Pausing does not split the episode: ticks 0,1,2 then 5,6 are recorded.
    assert len(frames) == 5
    assert meta == {"episode_success": "success"}
    assert frames[0]["task"] == "insert"
    assert list(frames[0]["action"]) == [0.25] * 7
    # The follower tracks the leader on every tick, recording or not.
    assert len(robot.sent) == robot.tick


def test_recorder_quit_honours_stop_pending_and_episode_target():
    # q lands mid-tick 3, whose frame is still written: ticks 0..3.
    _, dataset = _run_recorder({0: ["c"], 3: ["q"]}, stop_pending="save")
    assert [len(frames) for frames, _ in dataset.episodes] == [4]

    _, dataset = _run_recorder({0: ["c"], 3: ["q"]}, stop_pending="discard")
    assert dataset.episodes == []

    robot, dataset = _run_recorder({0: ["c"], 2: ["s"], 3: ["c"], 5: ["s"], 6: ["c"]}, num_episodes=2)
    assert dataset.num_episodes == 2
    assert robot.tick == 6


def test_episode_target_counts_this_session_on_a_resumed_dataset():
    robot, dataset = _run_recorder(
        {0: ["c"], 2: ["s"], 3: ["c"], 5: ["s"], 6: ["c"]}, num_episodes=2, existing_episodes=50
    )
    assert dataset.num_episodes == 52
    assert robot.tick == 6


def test_space_before_start_is_rejected_not_fatal():
    _, dataset = _run_recorder({0: ["space"], 1: ["q"]})
    assert dataset.episodes == []


# --------------------------------------------------------------------------- HIL guard


def _make_hil_leader_class():
    from lerobot.teleoperators import Teleoperator

    class HILPiperLeader(FakePiperLeader, Teleoperator):
        name = "fake_piper_leader"
        config_class = object

        def __init__(self, pose, teach=False):
            FakePiperLeader.__init__(self, pose, teach)
            self.id = "fake"

        action_features = property(lambda self: dict.fromkeys(JOINTS, float))
        feedback_features = property(lambda self: {})
        is_connected = property(lambda self: True)
        is_calibrated = property(lambda self: True)

        def connect(self, calibrate=True):
            pass

        def calibrate(self):
            pass

        def configure(self):
            pass

        def disconnect(self):
            pass

    return HILPiperLeader


class HILDataset(FakeDataset):
    root = None

    def __init__(self):
        super().__init__(fps=1000)
        self.features.update(
            {
                "complementary_info.policy_action": {"dtype": "float32", "shape": (7,), "names": JOINTS},
                "complementary_info.is_intervention": {"dtype": "float32", "shape": (1,), "names": ["x"]},
                "complementary_info.state": {"dtype": "float32", "shape": (1,), "names": ["x"]},
                "complementary_info.phase": {"dtype": "float32", "shape": (1,), "names": ["x"]},
            }
        )

    @property
    def episode_buffer(self):
        return {"size": len(self.buffer)}


def _run_hil_loop(monkeypatch, leader, toggles):
    """Run record_loop with a stub policy; `toggles` maps tick -> teach-mode state + space press."""
    from evo_rlt.adapters.lerobot.record import loop

    policy_pose = _pose(0.9)
    monkeypatch.setattr(loop, "_predict_policy_action_with_acp_inference", lambda **_kw: "policy")
    monkeypatch.setattr(loop, "make_robot_action", lambda _action, _features: dict(policy_pose))
    monkeypatch.setattr(loop, "_validate_policy_image_features", lambda *_a: None)
    monkeypatch.setattr(loop, "log_say", lambda *_a, **_kw: None)

    events = {"exit_early": False, "toggle_intervention": False}
    states = []

    class Follower(FakeFollower):
        robot_type = "piper"
        name = "piper"
        action_features = dict.fromkeys(JOINTS, float)

        def __init__(self):
            super().__init__(_pose(0.0))
            self.tick = 0

        def get_observation(self):
            if self.tick in toggles:
                teach, press = toggles[self.tick]
                leader.teach = teach
                events["toggle_intervention"] = press
            if self.tick >= max(toggles) + 2:
                events["exit_early"] = True
            self.tick += 1
            return super().get_observation()

        def send_action(self, action):
            states.append("policy" if action == policy_pose else "human")
            return super().send_action(action)

    reset = SimpleNamespace(reset=lambda: None)
    policy = SimpleNamespace(config=SimpleNamespace(device="cpu", use_amp=False), reset=lambda: None)
    loop.record_loop(
        robot=Follower(),
        events=events,
        fps=1000,
        teleop_action_processor=_identity,
        robot_action_processor=_identity,
        robot_observation_processor=_identity,
        dataset=HILDataset(),
        teleop=leader,
        policy=policy,
        preprocessor=reset,
        postprocessor=reset,
        control_time_s=5,
        single_task="insert",
    )
    return states


def test_release_is_refused_while_leader_teach_button_is_engaged(monkeypatch):
    leader = _make_hil_leader_class()(_pose(0.2))
    # A toggle pressed on tick N takes effect on tick N+1.
    # tick 1: space -> S1. tick 3: operator in teach mode presses space -> refused, stays S1.
    # tick 5: teach button released, space -> back to the policy.
    states = _run_hil_loop(monkeypatch, leader, {1: (False, True), 3: (True, True), 5: (False, True)})

    assert states == ["policy", "policy", "human", "human", "human", "human", "policy", "policy"]
    assert leader.calls == ["manual=False", "manual=True", "manual=False"]


def test_episode_starts_under_operator_when_teach_button_already_engaged(monkeypatch):
    leader = _make_hil_leader_class()(_pose(0.2), teach=True)
    states = _run_hil_loop(monkeypatch, leader, {2: (False, True)})

    assert states == ["human", "human", "human", "policy", "policy"]
    assert leader.calls == ["manual=True", "manual=False"]
