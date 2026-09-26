import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from evo_rlt.adapters.lerobot.record import replay
from evo_rlt.adapters.lerobot.record.replay import (
    PIPER_JOINT_KEYS,
    Episode,
    find_problems,
    load_episodes,
    replay_episode,
    tracking_error,
)

PIPER_MANIFEST = {
    "robot_type": "piper",
    "arms": [{"alias": "follower", "type": "follower", "port": "can_f"}],
    "cameras": [{"alias": "wrist", "type": "realsense", "serial": "1"}],
}


def _ramp(frames, stop=0.5):
    return np.linspace(0.0, stop, frames)[:, None].repeat(7, axis=1)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    import lerobot.utils.robot_utils

    monkeypatch.setattr(lerobot.utils.robot_utils, "precise_sleep", lambda _s: None)


class FakeBus:
    def __init__(self, robot):
        self.robot = robot

    def move_to_joint_smoothly(self, target, duration_s=None):
        self.robot.approached.append(list(target))
        self.robot.pose = list(target)


class FakePiper:
    instances = []

    def __init__(self, config):
        self.config = config
        self.pose = [0.0] * 7
        self.sent = []
        self.approached = []
        self.is_connected = False
        self.bus = FakeBus(self)
        FakePiper.instances.append(self)

    def connect(self, calibrate=True):
        assert calibrate is False
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        return dict(zip(PIPER_JOINT_KEYS, self.pose, strict=True))

    def send_action(self, action):
        self.sent.append([action[key] for key in PIPER_JOINT_KEYS])
        self.pose = self.sent[-1]
        return action


def test_find_problems_flags_jumps_and_bad_values():
    assert find_problems(_ramp(30), math.radians(10)) == []
    jump = _ramp(30)
    jump[12, 2] += 0.5
    (problem,) = find_problems(jump, math.radians(10))
    assert "joint_3 jumps" in problem and "frames 11 and 12" in problem
    nan = _ramp(5)
    nan[3, 0] = np.nan
    assert find_problems(nan, 1.0) == ["non-finite values in the commands"]
    assert find_problems(np.empty((0, 7)), 1.0) == ["no frames"]
    # The gripper is in meters and is not part of the joint-jump check.
    gripper = _ramp(5)
    gripper[2, 6] = 5.0
    assert find_problems(gripper, math.radians(10)) == []


def test_replay_episode_observes_before_each_command():
    robot = FakePiper(None)
    commands = _ramp(4)
    measured = replay_episode(robot, commands, fps=1000)
    assert np.allclose(robot.sent, commands)
    # Row t is the pose before command t: the start pose, then the previous command.
    assert np.allclose(measured[0], 0.0)
    assert np.allclose(measured[1:], commands[:-1])
    rmse, max_abs = tracking_error(measured, commands)
    assert np.allclose(max_abs, commands[1, 0])
    assert (rmse <= max_abs).all()


def _args(tmp_path, **overrides):
    setup_json = tmp_path / "setup.json"
    setup_json.write_text(json.dumps(PIPER_MANIFEST))
    args = dict(
        setup_json=str(setup_json),
        dataset_root=str(tmp_path / "ds"),
        episodes=None,
        source="action",
        speed=1.0,
        approach_time_s=3.0,
        max_step_deg=10.0,
        confirm=False,
        save_trace=None,
        log_level="INFO",
        dry_run=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.fixture
def fake_robot(monkeypatch):
    import evo_rlt.adapters.lerobot.hardware.piper as piper_pkg

    FakePiper.instances.clear()
    monkeypatch.setattr(piper_pkg, "Piper", FakePiper)
    return FakePiper.instances


def _episodes():
    jumpy = _ramp(10)
    jumpy[5:, 0] += 1.0
    return [
        Episode(0, "success", _ramp(10), _ramp(10)),
        Episode(1, "failure", jumpy, jumpy),
        Episode(2, None, _ramp(6, stop=-0.2), _ramp(6, stop=-0.2)),
    ]


def test_run_replay_skips_flagged_episodes_and_reports(monkeypatch, tmp_path, fake_robot, capsys):
    monkeypatch.setattr(replay, "load_episodes", lambda root, episodes: (30, _episodes()))
    replay.run_replay(_args(tmp_path, save_trace=str(tmp_path / "trace")))

    (robot,) = fake_robot
    assert robot.config.cameras == {} and robot.config.can_port == "can_f"
    assert not robot.is_connected
    # Episode 1 jumps 57 deg in one frame, so only episodes 0 and 2 reach the arm.
    assert np.allclose(robot.approached, [_ramp(10)[0], _ramp(6, stop=-0.2)[0]])
    assert np.allclose(robot.sent, np.concatenate([_ramp(10), _ramp(6, stop=-0.2)]))
    out = capsys.readouterr().out
    assert "joint_1 jumps" in out and "-> skipped" in out
    assert "Tracking error" in out
    trace = np.load(tmp_path / "trace" / "episode_002.npz")
    assert trace["measured_state"].shape == (6, 7) and float(trace["fps"]) == 30
    assert not (tmp_path / "trace" / "episode_001.npz").exists()


def test_run_replay_confirm_prompt_skip_and_quit(monkeypatch, tmp_path, fake_robot):
    monkeypatch.setattr(replay, "load_episodes", lambda root, episodes: (30, _episodes()))
    replies = iter(["s", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(replies))
    replay.run_replay(_args(tmp_path, confirm=True))

    (robot,) = fake_robot
    assert robot.sent == [] and robot.approached == []
    assert not robot.is_connected


def test_dry_run_and_bad_speed_never_touch_hardware(monkeypatch, tmp_path, fake_robot, capsys):
    monkeypatch.setattr(replay, "load_episodes", lambda root, episodes: (30, _episodes()))
    replay.run_replay(_args(tmp_path, dry_run=True))
    assert "Dry run robot config" in capsys.readouterr().out
    with pytest.raises(ValueError, match="--speed"):
        replay.run_replay(_args(tmp_path, speed=1.5))
    assert fake_robot == []


def test_load_episodes_reads_a_recorded_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    joint = {"dtype": "float32", "shape": (7,), "names": PIPER_JOINT_KEYS}
    root = tmp_path / "ds"
    dataset = LeRobotDataset.create(
        "local/ds", 30, root=root, robot_type="piper",
        features={"action": joint, "observation.state": joint}, use_videos=False,
    )
    for episode, frames in enumerate((4, 3)):
        for frame in range(frames):
            value = np.full(7, episode + frame / 10, dtype=np.float32)
            dataset.add_frame({"action": value, "observation.state": value - 0.01, "task": "t"})
        dataset.save_episode()
    dataset.finalize()

    fps, episodes = load_episodes(root, [1])
    assert fps == 30
    (episode,) = episodes
    assert episode.index == 1 and episode.label is None
    assert episode.actions.shape == (3, 7)
    assert np.allclose(episode.actions[:, 0], [1.0, 1.1, 1.2])
    assert np.allclose(episode.commands("state"), episode.actions - 0.01)
    with pytest.raises(ValueError, match="do not exist"):
        load_episodes(root, [2])
    with pytest.raises(FileNotFoundError):
        load_episodes(tmp_path / "missing", None)
    assert not (tmp_path / "missing").exists()


def test_cli_parses_replay_subcommand():
    from evo_rlt.adapters.lerobot.record.cli import parse_args

    args = parse_args(
        ["replay", "--dataset-root", "/d", "--episodes", "0", "3", "--no-confirm", "--speed", "0.5"]
    )
    assert args.func.__name__ == "run_replay"
    assert args.episodes == [0, 3] and args.confirm is False and args.speed == 0.5 and args.source == "action"
