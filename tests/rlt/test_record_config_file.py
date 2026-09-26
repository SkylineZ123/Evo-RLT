from pathlib import Path

import pytest
import yaml

from evo_rlt.adapters.lerobot.record.cli import parse_args
from evo_rlt.adapters.lerobot.record.common import RESUME_LATEST

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_CONFIGS = sorted((REPO_ROOT / "configs/record").glob("*.yaml"))


def _write_yaml(tmp_path, data, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_shipped_record_configs_exist():
    assert {path.name for path in RECORD_CONFIGS} >= {
        "piper_teleop.yaml",
        "piper_collect_hil.yaml",
        "piper_segment.yaml",
        "piper_full_vla.yaml",
    }


@pytest.mark.parametrize("config_path", RECORD_CONFIGS, ids=lambda path: path.name)
def test_shipped_record_config_parses_with_every_key_known(config_path):
    raw = yaml.safe_load(config_path.read_text())
    args = parse_args(["--config", str(config_path)])

    assert args.command == raw["command"]
    assert args.fps == 30
    assert args.setup_json.endswith("piper_setup.json")
    for key, value in raw.items():
        if key != "command" and value is not None:
            # `resume: true` is the bare `--resume` flag.
            expected = RESUME_LATEST if key == "resume" and value is True else value
            assert getattr(args, key) == expected, key


def test_cli_flags_override_config_and_subcommand_may_be_repeated(tmp_path):
    path = _write_yaml(tmp_path, {"command": "teleop", "task": "from yaml", "num_episodes": 50})

    args = parse_args(["--config", path, "--num-episodes", "3"])
    assert (args.task, args.num_episodes) == ("from yaml", 3)

    args = parse_args(["teleop", f"--config={path}", "--task", "from cli"])
    assert (args.task, args.num_episodes) == ("from cli", 50)


def test_config_satisfies_required_options(tmp_path):
    path = _write_yaml(tmp_path, {"initial_source": "vla", "critical_source": "rlt", "policy_path": "/ac"})
    args = parse_args(["segment", "--config", path])
    assert (args.initial_source, args.critical_source, args.policy_path) == ("vla", "rlt", "/ac")
    assert args.dataset_tag == "vla_segment"


def test_null_keeps_the_builtin_default(tmp_path):
    path = _write_yaml(tmp_path, {"command": "collect", "policy_path": "/ac", "fps": None, "task": None})
    args = parse_args(["--config", path])
    assert args.fps == 30
    assert args.task == "Insert the copper screw into the black sleeve."


def test_hyphenated_keys_are_accepted(tmp_path):
    path = _write_yaml(tmp_path, {"command": "collect", "policy-path": "/ac", "only-critical": True})
    args = parse_args(["--config", path])
    assert args.policy_path == "/ac"
    assert args.only_critical is True


@pytest.mark.parametrize(
    ("data", "argv_prefix", "message"),
    [
        ({"command": "teleop", "task": "t", "polcy_path": "/ac"}, [], "unknown option 'polcy_path'"),
        ({"command": "segment", "initial_source": "rlt"}, [], "must be one of"),
        ({"command": "teleop", "task": "t", "status_view": "yes"}, [], "must be true or false"),
        ({"command": "teleop", "task": "t", "fps": "fast"}, [], "must be int"),
        ({"command": "teleop", "task": "t", "fps": True}, [], "does not take true/false"),
        ({"command": "nope"}, [], "unknown command"),
        ({"task": "t"}, [], "set `command:`"),
        ({"command": "teleop", "task": "t"}, ["collect"], "is a 'teleop' config"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path, data, argv_prefix, message, capsys):
    path = _write_yaml(tmp_path, data)
    with pytest.raises(SystemExit) as excinfo:
        parse_args([*argv_prefix, "--config", path])
    assert message in str(excinfo.value)


def test_missing_required_option_still_errors(tmp_path):
    path = _write_yaml(tmp_path, {"command": "collect"})
    with pytest.raises(SystemExit):
        parse_args(["--config", path])


def test_teleop_config_dry_run(monkeypatch, capsys):
    from evo_rlt.adapters.lerobot.record.cli import main

    monkeypatch.chdir(REPO_ROOT)
    main(["--config", "configs/record/piper_teleop.yaml", "--dry-run"])
    out = capsys.readouterr().out
    assert "can_port='can_follower'" in out
    assert "Task: Pick up the peg and insert it into the hole." in out


# --------------------------------------------------------------------------- keyboard patch


@pytest.fixture
def patched_listener(monkeypatch):
    """Install the record keyboard patch over a stand-in for LeRobot 0.5.1's listener."""
    control_utils = pytest.importorskip("lerobot.utils.control_utils")
    from evo_rlt.adapters.lerobot.record import pedal_listener
    from evo_rlt.adapters.lerobot.record.runner import _patch_double_tap_episode_outcome_listener

    captured = {}

    class FakePedalListener:
        def __init__(self, on_press):
            captured["on_press"] = on_press

        def start(self):
            return True

        def stop(self):
            pass

    def lerobot_051_init_keyboard_listener():  # takes no arguments, like LeRobot 0.5.1
        return None, {"exit_early": False, "rerecord_episode": False, "stop_recording": False}

    monkeypatch.setattr(control_utils, "is_headless", lambda: True)
    monkeypatch.setattr(control_utils, "init_keyboard_listener", lerobot_051_init_keyboard_listener)
    monkeypatch.setattr(pedal_listener, "PedalListener", FakePedalListener)
    _patch_double_tap_episode_outcome_listener(0.0, None)
    return control_utils.init_keyboard_listener, captured


def test_listener_without_outcome_key_accepts_record_kwargs_and_labels_with_s_f(patched_listener):
    init_keyboard_listener, captured = patched_listener
    # `full` without a pedal: RLT is off, so s/f are free to label the episode.
    listener, events = init_keyboard_listener(
        intervention_toggle_key="i", episode_success_key="s", episode_failure_key="f"
    )
    captured["on_press"]("i")
    assert events["toggle_intervention"] is True
    captured["on_press"]("f")
    assert (events["episode_outcome"], events["exit_early"]) == ("failure", True)
    listener.stop()


def test_listener_without_outcome_key_leaves_s_f_to_rlt_phase_keys(patched_listener):
    init_keyboard_listener, captured = patched_listener
    # `collect --only-critical` / `segment`: s/f already end the RL phase.
    listener, events = init_keyboard_listener(
        intervention_toggle_key=" ",
        episode_success_key="s",
        episode_failure_key="f",
        rl_phase_key="r",
        end_success_key="s",
        end_failure_key="f",
    )
    captured["on_press"]("s")
    assert events["end_phase_success"] is True
    assert events["episode_outcome"] is None
    assert events["exit_early"] is False
    captured["on_press"]("r")
    assert events["start_rl_phase"] is True
    listener.stop()
