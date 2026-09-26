import json
import sys
from pathlib import Path

import pytest
import yaml

from evo_rlt.cli.train_pi05_sft import (
    DEFAULT_RENAME_MAP,
    LAST_TRAIN_CONFIG,
    build_accelerate_command,
    build_lerobot_argv,
    dataset_camera_keys,
    drop_unused_cameras,
    main,
    parse_args,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED_CONFIG = REPO_ROOT / "configs/train/pi05_sft_piper_blood_gas.yaml"


def _write_yaml(tmp_path, data, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _flags(argv):
    return dict(token[2:].split("=", 1) for token in argv)


def _fake_run(run_dir: Path) -> Path:
    config_path = run_dir / LAST_TRAIN_CONFIG
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}")
    return config_path


def test_shipped_config_parses_with_every_key_known():
    raw = yaml.safe_load(SHIPPED_CONFIG.read_text())
    args, extra = parse_args(["--config", str(SHIPPED_CONFIG)])

    assert extra == []
    for key, value in raw.items():
        if value is not None:
            assert getattr(args, key) == value, key


def test_new_run_argv_from_yaml_with_cli_override(tmp_path):
    config = _write_yaml(
        tmp_path,
        {
            "dataset_root": "/data/teleop_1",
            "output_dir": str(tmp_path / "run"),
            "steps": 1000,
            "episodes": [0, 2],
            "wandb": True,
            "wandb_mode": "offline",
            "lr": 1e-5,
        },
    )
    args, extra = parse_args(["--config", config, "--batch-size", "2", "--policy.optimizer_eps=1e-6"])
    flags = _flags(build_lerobot_argv(args, extra))

    assert flags["dataset.repo_id"] == "local/teleop_1"
    assert flags["dataset.root"] == "/data/teleop_1"
    assert json.loads(flags["dataset.episodes"]) == [0, 2]
    assert json.loads(flags["rename_map"]) == DEFAULT_RENAME_MAP
    assert flags["policy.path"] == "lerobot/pi05_base"
    assert flags["policy.gradient_checkpointing"] == "true"
    assert flags["policy.scheduler_decay_steps"] == "1000"
    assert flags["policy.optimizer_lr"] == "1e-05"
    assert flags["batch_size"] == "2"
    assert flags["steps"] == "1000"
    assert flags["eval_freq"] == "0"
    assert flags["wandb.enable"] == "true"
    assert flags["wandb.mode"] == "offline"
    assert flags["wandb.disable_artifact"] == "true"
    assert flags["policy.optimizer_eps"] == "1e-6"  # forwarded unknown flag


def test_yaml_rename_map_and_output_dir_default(tmp_path):
    config = _write_yaml(
        tmp_path,
        {"dataset_root": "/d", "job_name": "exp", "rename_map": {"observation.images.cam": "observation.images.x"}},
    )
    args, _ = parse_args(["--config", config])

    assert args.output_dir == str(Path("outputs/pi05_sft/exp"))
    assert json.loads(_flags(build_lerobot_argv(args))["rename_map"]) == {
        "observation.images.cam": "observation.images.x"
    }


def test_unknown_yaml_key_is_rejected(tmp_path):
    config = _write_yaml(tmp_path, {"dataset_root": "/d", "bogus": 1})
    with pytest.raises(SystemExit, match="unknown option 'bogus'"):
        parse_args(["--config", config])


def test_new_run_refuses_existing_output_dir(tmp_path):
    args, _ = parse_args(["--dataset-root", "/d", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="--resume"):
        build_lerobot_argv(args)
    # accelerate ranks skip the check: rank 0 may already have created the directory.
    assert build_lerobot_argv(args, check_output_dir=False)


def test_resume_uses_checkpoint_config_and_run_overrides(tmp_path):
    run_dir = tmp_path / "run"
    config_path = _fake_run(run_dir)
    config = _write_yaml(tmp_path, {"dataset_root": "/d", "output_dir": str(run_dir), "resume": True})
    args, _ = parse_args(["--config", config, "--steps", "50000"])
    argv = build_lerobot_argv(args)

    assert argv[:2] == ["--resume=true", f"--config_path={config_path}"]
    flags = _flags(argv)
    assert flags["steps"] == "50000"
    assert "policy.path" not in flags and "dataset.root" not in flags


def test_resume_explicit_dir_and_missing_checkpoint(tmp_path):
    config_path = _fake_run(tmp_path / "other")
    args, _ = parse_args(["--dataset-root", "/d", "--resume", str(tmp_path / "other")])
    assert build_lerobot_argv(args)[1] == f"--config_path={config_path}"

    args, _ = parse_args(["--dataset-root", "/d", "--output-dir", str(tmp_path / "empty"), "--resume"])
    with pytest.raises(SystemExit, match="no checkpoint"):
        build_lerobot_argv(args)


def test_multi_gpu_relaunches_under_accelerate(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    argv = ["--dataset-root", "/d", "--output-dir", str(tmp_path / "run"), "--num-gpus", "2", "--dry-run"]
    args, _ = parse_args(argv)
    command = build_accelerate_command(args, argv)

    assert command[:3] == [sys.executable, "-m", "accelerate.commands.launch"]
    assert "--num_processes=2" in command
    assert command[command.index("-m", 3) + 1] == "evo_rlt.cli.train_pi05_sft"
    assert command[-len(argv):] == argv

    main(argv)
    assert "accelerate.commands.launch" in capsys.readouterr().out


def test_compile_flags(tmp_path):
    args, _ = parse_args(["--dataset-root", "/d", "--output-dir", str(tmp_path / "run")])
    flags = _flags(build_lerobot_argv(args))
    assert flags["policy.compile_model"] == "false"
    assert "policy.compile_mode" not in flags

    args, _ = parse_args(
        ["--dataset-root", "/d", "--output-dir", str(tmp_path / "run"), "--compile-model", "--compile-mode", "default"]
    )
    flags = _flags(build_lerobot_argv(args))
    assert flags["policy.compile_model"] == "true"
    assert flags["policy.compile_mode"] == "default"


def _write_info(root: Path, features: dict) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"features": features}))
    return root


def test_dataset_camera_keys_apply_rename_map(tmp_path):
    root = _write_info(
        tmp_path / "ds",
        {
            "observation.images.front": {"dtype": "video"},
            "observation.images.wrist": {"dtype": "video"},
            "observation.images.side": {"dtype": "image"},
            "observation.state": {"dtype": "float32"},
        },
    )
    assert dataset_camera_keys(root, DEFAULT_RENAME_MAP) == {
        "observation.images.base_0_rgb",
        "observation.images.right_wrist_0_rgb",
        "observation.images.side",
    }
    with pytest.raises(SystemExit, match="info.json"):
        dataset_camera_keys(tmp_path / "missing", DEFAULT_RENAME_MAP)


def test_drop_unused_cameras():
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature

    def make_cfg():
        visual = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        return SimpleNamespace(
            input_features={
                "observation.images.base_0_rgb": visual,
                "observation.images.left_wrist_0_rgb": visual,
                "observation.images.right_wrist_0_rgb": visual,
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(32,)),
            }
        )

    cfg = make_cfg()
    keep = {"observation.images.base_0_rgb", "observation.images.right_wrist_0_rgb"}
    assert drop_unused_cameras(cfg, keep) == ["observation.images.left_wrist_0_rgb"]
    assert list(cfg.input_features) == [
        "observation.images.base_0_rgb",
        "observation.images.right_wrist_0_rgb",
        "observation.state",
    ]

    with pytest.raises(SystemExit, match="no dataset camera"):
        drop_unused_cameras(make_cfg(), {"observation.images.front"})
