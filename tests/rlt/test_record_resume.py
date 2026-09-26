import json
import os
import shutil

import numpy as np
import pytest
import yaml

from evo_rlt.adapters.lerobot.record.cli import parse_args
from evo_rlt.adapters.lerobot.record.common import (
    RESUME_LATEST,
    find_latest_dataset,
    resolve_record_paths,
    resolve_resume_paths,
)

JOINTS = [f"joint_{i}.pos" for i in range(1, 8)]
FEATURES = {
    "action": {"dtype": "float32", "shape": (7,), "names": JOINTS},
    "observation.state": {"dtype": "float32", "shape": (7,), "names": JOINTS},
}
TASK = "insert"


def _add_episode(dataset, value, frames=3):
    for _ in range(frames):
        joints = np.full(7, value, dtype=np.float32)
        dataset.add_frame({"action": joints, "observation.state": joints, "task": TASK})
    dataset.save_episode()


def _record(root, episodes=2):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create("local/" + root.name, 30, root=root, robot_type="piper",
                                    features=FEATURES, use_videos=False)
    for episode in range(episodes):
        _add_episode(dataset, episode)
    dataset.finalize()
    return root


@pytest.fixture
def recorded(tmp_path):
    return _record(tmp_path / "0925_tag" / "teleop_101010")


# --------------------------------------------------------------------------- resume paths


def test_resume_paths_point_at_the_existing_dataset(recorded):
    paths = resolve_resume_paths(recorded)
    assert paths.resume and paths.resumed_episodes == 2
    assert paths.dataset_root == recorded and paths.dataset_name == "local/teleop_101010"
    assert paths.resumed_tasks == (TASK,)
    # The first session's log, appended to.
    assert paths.log_file == recorded.parent / "teleop_101010.log"
    assert not resolve_record_paths({}, "tag", "teleop", None).resume


def test_resume_refuses_missing_empty_and_unfinalized_datasets(tmp_path, recorded):
    with pytest.raises(FileNotFoundError, match="not a LeRobot dataset"):
        resolve_resume_paths(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()

    empty = _record(tmp_path / "empty", episodes=0)
    with pytest.raises(ValueError, match="no saved episodes"):
        resolve_resume_paths(empty)

    truncated = shutil.copytree(recorded, tmp_path / "truncated")
    data_file = next((truncated / "data").glob("*/*.parquet"))
    data_file.write_bytes(data_file.read_bytes()[:-20])  # a killed session leaves no footer
    with pytest.raises(ValueError, match="not finalized"):
        resolve_resume_paths(truncated)

    miscounted = shutil.copytree(recorded, tmp_path / "miscounted")
    info_path = miscounted / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = 3
    info_path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="expects 3"):
        resolve_resume_paths(miscounted)


def _fake_dataset(root, day, leaf, episodes, mtime):
    info = root / day / leaf / "meta" / "info.json"
    info.parent.mkdir(parents=True)
    info.write_text(json.dumps({"total_episodes": episodes}))
    os.utime(info, (mtime, mtime))


def test_latest_dataset_is_the_newest_non_empty_one_of_the_tag(tmp_path):
    setup = {"datasets": {"root": str(tmp_path)}}
    _fake_dataset(tmp_path, "0924_tag", "teleop_100000", 20, mtime=1000)
    _fake_dataset(tmp_path, "0925_tag", "teleop_080000", 5, mtime=2000)
    _fake_dataset(tmp_path, "0925_tag", "teleop_090000", 0, mtime=3000)  # quit before any save
    _fake_dataset(tmp_path, "0925_other_tag", "teleop_100000", 9, mtime=4000)  # tag is a suffix only
    _fake_dataset(tmp_path, "0925_tag", "eval_vla_full_100000", 9, mtime=4000)  # other subcommand

    assert find_latest_dataset(setup, "tag", "teleop") == tmp_path / "0925_tag" / "teleop_080000"
    latest_full = find_latest_dataset(setup, "tag", "eval_vla_full")
    assert latest_full == tmp_path / "0925_tag" / "eval_vla_full_100000"
    with pytest.raises(FileNotFoundError, match="no dataset with saved episodes"):
        find_latest_dataset(setup, "tag", "eval_vla_segment")


def test_resume_latest_goes_through_the_integrity_check(tmp_path, recorded):
    setup = {"datasets": {"root": str(tmp_path)}}
    paths = resolve_record_paths(setup, "tag", "teleop", RESUME_LATEST)
    assert paths.dataset_root == recorded and paths.resumed_episodes == 2


# --------------------------------------------------------------------------- CLI / YAML


@pytest.mark.parametrize(
    ("argv", "yaml_resume", "expected"),
    [
        (["--resume"], None, RESUME_LATEST),
        (["--resume", "/data/teleop_101010"], None, "/data/teleop_101010"),
        ([], True, RESUME_LATEST),
        ([], False, None),
        ([], "/data/teleop_101010", "/data/teleop_101010"),
        (["--resume", "/cli"], True, "/cli"),
    ],
)
def test_resume_flag_and_yaml_key(tmp_path, argv, yaml_resume, expected):
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({"command": "teleop", "task": "t", "resume": yaml_resume}))
    assert parse_args(["--config", str(config), *argv]).resume == expected


@pytest.mark.parametrize("command", [["teleop", "--task", "t"], ["collect", "--policy-path", "p"],
                                     ["segment", "--initial-source", "vla", "--critical-source", "vla"],
                                     ["full", "--initial-source", "vla"]])
def test_every_recording_subcommand_takes_resume(command):
    assert parse_args([*command, "--resume"]).resume == RESUME_LATEST
    assert parse_args(command).resume is None


# --------------------------------------------------------------------------- appending


class _Robot:
    name = robot_type = "piper"
    cameras = {}


def test_teleop_dataset_resume_appends_and_checks_compatibility(recorded):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from evo_rlt.adapters.lerobot.record.teleop_collect import open_teleop_dataset

    with pytest.raises(ValueError, match="fps"):
        open_teleop_dataset(resolve_resume_paths(recorded), _Robot(), FEATURES, fps=15, vcodec="h264")

    dataset = open_teleop_dataset(resolve_resume_paths(recorded), _Robot(), FEATURES, fps=30, vcodec="h264")
    assert dataset.num_episodes == 2
    _add_episode(dataset, 7.0)
    assert dataset.num_episodes == 3
    dataset.finalize()

    reloaded = LeRobotDataset("local/" + recorded.name, root=recorded)
    assert reloaded.num_episodes == 3 and reloaded.num_frames == 9
    assert [int(ep) for ep in reloaded.hf_dataset["episode_index"]] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert float(reloaded[8]["action"][0]) == 7.0


def test_segment_dry_run_resumes_without_removing_the_dataset(tmp_path, recorded, capsys):
    from evo_rlt.adapters.lerobot.record.cli import main

    setup_json = tmp_path / "setup.json"
    setup_json.write_text(json.dumps({
        "robot_type": "piper",
        "datasets": {"root": str(tmp_path)},
        "arms": [{"alias": "follower", "type": "follower", "port": "can_f"}],
    }))
    main(["segment", "--setup-json", str(setup_json), "--initial-source", "vla", "--critical-source", "vla",
          "--policy-path", "p", "--task", "other", "--resume", str(recorded), "--no-teleop", "--dry-run"])

    out = capsys.readouterr().out
    assert "Resume: appending after 2 saved episodes" in out
    assert "task 'other' is new to this dataset" in out
    assert "--resume=true" in out and f"--dataset.root={recorded}" in out
    assert resolve_resume_paths(recorded).resumed_episodes == 2
