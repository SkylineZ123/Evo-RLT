from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_CAMERAS = ["left_wrist", "right_wrist", "right_front"]
DEFAULT_ACTION_DIM = 12
DEFAULT_PROPRIO_DIM = 12
DEFAULT_VLA_HORIZON = 50
DEFAULT_CHUNK_LENGTH = 10

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def configure_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def json_dict(value: str) -> dict[str, str]:
    """argparse type for a JSON object of strings, e.g. --camera-name-map '{"front": "..."}'."""
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {value!r}")
    return {str(k): str(v) for k, v in parsed.items()}


def apply_run_section(parser: argparse.ArgumentParser, run, config_path: str) -> None:
    """Load a YAML `run:` section as parser defaults; explicit CLI flags still win."""
    from evo_rlt.adapters.lerobot.record.cli import _coerce_config_value

    if not isinstance(run, dict):
        raise SystemExit(f"{config_path}: `run` must be a mapping of option -> value")
    actions = {action.dest: action for action in parser._actions if action.dest not in ("help", "config")}
    defaults = {}
    for raw_key, value in run.items():
        key = str(raw_key).replace("-", "_")
        action = actions.get(key)
        if action is None:
            raise SystemExit(f"{config_path}: unknown run option '{raw_key}'. Valid options: {sorted(actions)}")
        if value is None:
            # `key: null` leaves the built-in default (or the CLI) in charge.
            continue
        if key == "camera_name_map":
            if not isinstance(value, dict):
                raise SystemExit(f"{config_path}: run.{raw_key} must be a mapping, got {value!r}")
            defaults[key] = {str(k): str(v) for k, v in value.items()}
        elif key == "active_cameras" and isinstance(value, list):
            defaults[key] = ",".join(str(camera) for camera in value)
        elif action.nargs in ("+", "*"):
            items = value if isinstance(value, list) else [value]
            defaults[key] = [_coerce_config_value(action, raw_key, item) for item in items]
        else:
            defaults[key] = _coerce_config_value(action, raw_key, value)
        action.required = False
    parser.set_defaults(**defaults)


def parse_args_with_run_section(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    """Parse argv; with `--config PATH.yaml` the file's `run:` section supplies the option defaults."""
    import yaml

    from evo_rlt.adapters.lerobot.record.cli import _pop_config_path

    argv = sys.argv[1:] if argv is None else list(argv)
    config_path, argv = _pop_config_path(argv)
    if config_path is not None:
        with open(config_path) as fh:
            raw = yaml.safe_load(fh) or {}
        apply_run_section(parser, raw.get("run") or {}, config_path)
    args = parser.parse_args(argv)
    args.config = config_path
    return args


def load_training_config(config_path: str | None):
    """Load an RLT YAML config; shape fields the YAML leaves out get the 12-dim bimanual defaults."""
    import yaml

    from evo_rlt.core.config import RLTConfig

    config = RLTConfig.from_yaml(config_path) if config_path else RLTConfig()
    raw = {}
    if config_path:
        with open(config_path) as fh:
            raw = yaml.safe_load(fh) or {}
    defaults = {
        "action_dim": DEFAULT_ACTION_DIM,
        "proprio_dim": DEFAULT_PROPRIO_DIM,
        "vla_horizon": DEFAULT_VLA_HORIZON,
        "chunk_length": DEFAULT_CHUNK_LENGTH,
        "cameras": list(DEFAULT_CAMERAS),
    }
    for key, value in defaults.items():
        if key not in raw:
            setattr(config, key, value)
    return config


def build_pi05_policy(
    config,
    model_path: str,
    task_instruction: str,
    device: str,
    token_pool_size: int,
    dtype: str,
    rl_token_checkpoint: str | None = None,
    vla_cache_dir: str | None = None,
    image_only: bool = False,
    active_cameras: list[str] | None = None,
    tokenizer_path: str | None = None,
    camera_name_map: dict[str, str] | None = None,
):
    from evo_rlt.adapters.lerobot.pi05_adapter import Pi05VLAAdapter
    from evo_rlt.core.policy import RLTPolicy
    from evo_rlt.core.rl_token import load_rl_token_encoder
    import torch

    rl_token_state = None
    if rl_token_checkpoint is not None:
        checkpoint = torch.load(rl_token_checkpoint, map_location="cpu", weights_only=False)
        rl_token_state = checkpoint["rl_token_state_dict"]
        config.rl_token.update_from_state_dict(rl_token_state)

    vla = Pi05VLAAdapter(
        model_path=model_path,
        actual_action_dim=config.action_dim,
        actual_proprio_dim=config.proprio_dim,
        camera_name_map=camera_name_map,
        task_instruction=task_instruction,
        dtype=dtype,
        device=device,
        cache_dir=vla_cache_dir,
        token_pool_size=token_pool_size,
        image_only=image_only,
        active_cameras=active_cameras,
        tokenizer_path=tokenizer_path,
    )
    policy = RLTPolicy(config, vla).to(device)
    if rl_token_state is not None:
        load_rl_token_encoder(policy.rl_token, rl_token_state)
    return policy
