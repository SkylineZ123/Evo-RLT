#!/usr/bin/env python
"""Train the RL token module (demo adaptation) on a LeRobot demo dataset; pi0.5 stays frozen.

One YAML drives the run: the RLTConfig sections (`rl_token`, `demo_adaptation`, ...) plus a `run:`
section whose keys are this script's options (`model_path` == `--model-path`). CLI flags override it.

    evo-rlt-train-rl-token --config configs/train/rl_token_piper.yaml
    evo-rlt-train-rl-token --config configs/train/rl_token_piper.yaml --steps 20000 --output-dir outputs/rl_token/exp2

Fine-tune the VLA itself with evo-rlt-train-pi05-sft first, then point `model_path` at that checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evo_rlt.adapters.lerobot.record.cli import _coerce_config_value, _pop_config_path
from evo_rlt.cli.common import build_pi05_policy, configure_logging, load_training_config

logger = configure_logging(__name__)

DEMO_REPO_ID = "rlt_demo"


def _json_dict(value: str) -> dict[str, str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {value!r}")
    return {str(k): str(v) for k, v in parsed.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the RL token module on a raw demo dataset. --config PATH.yaml holds the RLTConfig sections "
            "plus a `run:` section of defaults for the options below; CLI flags override the file."
        )
    )
    parser.add_argument("--config", default=None, help="RLT YAML config (RLTConfig sections + optional `run:` options).")
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument("--demo-dataset-path", required=True, help="Local LeRobot dataset directory.")
    parser.add_argument("--output-dir", default="outputs/rl_token")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None, help="Override demo_adaptation.steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override demo_adaptation.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override demo_adaptation.lr.")
    parser.add_argument(
        "--task-instruction",
        default=None,
        help="Task text in the pi0.5 prompt; default: the dataset's task (what SFT trained on).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume-checkpoint", default=None, help="Path to an RL token checkpoint to resume from.")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--token-pool-size", type=int, default=0, help="Pool prefix tokens before RL token encoding (0 disables pooling).")
    parser.add_argument("--image-only", action="store_true", help="Drop language tokens before RL token encode (image-patch tokens only).")
    parser.add_argument("--active-cameras", default=None,
                        help="Comma-separated camera names (e.g. 'right_wrist' or 'left_wrist,right_wrist'). Implies image-only and overrides it.")
    parser.add_argument(
        "--camera-name-map",
        type=_json_dict,
        default=None,
        help=(
            'JSON {dataset camera: pi0.5 slot}, e.g. {"front": "observation.images.base_0_rgb"}; '
            "default: the bimanual left_wrist/right_wrist/right_front map."
        ),
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantile-normalize state/actions to [-1, 1] from dataset stats, as pi0.5 and build_transition_cache do.",
    )
    parser.add_argument("--norm-stats", default=None, help="Path to a .pt file with 'std' tensor for per-dim weighted MSE.")
    parser.add_argument("--norm-gamma", type=float, default=0.0, help="Per-dim weighting exponent. 0=raw MSE, 0.5=partial whitening, 1=full whitening.")
    parser.add_argument("--num-rl-tokens", type=int, default=None, help="Override config.rl_token.num_rl_tokens.")
    parser.add_argument("--enc-layers", type=int, default=None, help="Override config.rl_token.enc_layers.")
    parser.add_argument("--dec-layers", type=int, default=None, help="Override config.rl_token.dec_layers.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="VLA model dtype")
    parser.add_argument("--vla-cache-dir", default=None, help="Optional Pi0.5 cache directory.")
    parser.add_argument("--tokenizer-path", default=None, help="PaliGemma tokenizer repo id or local snapshot path.")
    return parser


def _apply_run_section(parser: argparse.ArgumentParser, run, config_path: str) -> None:
    """Load the YAML `run:` section as parser defaults; explicit CLI flags still win."""
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
        else:
            defaults[key] = _coerce_config_value(action, raw_key, value)
        action.required = False
    parser.set_defaults(**defaults)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    config_path, argv = _pop_config_path(argv)
    if config_path is not None:
        with open(config_path) as fh:
            raw = yaml.safe_load(fh) or {}
        vla_ft_weight = (raw.get("demo_adaptation") or {}).get("vla_ft_weight") or 0.0
        if vla_ft_weight > 0:
            raise SystemExit(
                f"{config_path}: demo_adaptation.vla_ft_weight={vla_ft_weight} is not supported here: this script "
                "keeps pi0.5 frozen and saves only the RL token. Fine-tune the VLA with evo-rlt-train-pi05-sft, "
                "point run.model_path at that checkpoint and set vla_ft_weight to 0."
            )
        _apply_run_section(parser, raw.get("run") or {}, config_path)
    args = parser.parse_args(argv)
    args.config = config_path
    return args


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_dataset_meta(dataset_path: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if not (Path(dataset_path) / "meta" / "info.json").is_file():
        raise SystemExit(f"--demo-dataset-path {dataset_path}: not a LeRobot dataset (no meta/info.json)")
    return LeRobotDatasetMetadata(repo_id=DEMO_REPO_ID, root=dataset_path, revision="main")


def _resolve_task(task_instruction: str | None, meta) -> str:
    """The prompt must match what SFT trained on, so default to the dataset's own task."""
    tasks = list(meta.tasks.index) if meta.tasks is not None else []
    if task_instruction:
        if tasks and task_instruction not in tasks:
            logger.warning("--task-instruction %r differs from the dataset task(s) %s", task_instruction, tasks)
        return task_instruction
    if len(tasks) != 1:
        raise SystemExit(f"dataset has {len(tasks)} tasks {tasks}; pick one with --task-instruction")
    return tasks[0]


def _check_cameras(meta, camera_name_map: dict[str, str]) -> None:
    """Fail before loading pi0.5 when no dataset camera reaches a pi0.5 slot (it would see blank images)."""
    cameras = [key.split(".")[-1] for key in meta.camera_keys]
    mapped = [camera for camera in cameras if camera in camera_name_map]
    if not mapped:
        raise SystemExit(
            f"none of the dataset cameras {cameras} is in camera_name_map {sorted(camera_name_map)}, so pi0.5 "
            "would only see blank images. Set run.camera_name_map (or --camera-name-map), e.g. "
            '{"front": "observation.images.base_0_rgb", "wrist": "observation.images.right_wrist_0_rgb"}'
        )
    unmapped = sorted(set(cameras) - set(mapped))
    if unmapped:
        logger.warning("Dataset cameras %s have no pi0.5 slot in camera_name_map and are ignored", unmapped)


def main() -> None:
    args = parse_args()

    from evo_rlt.adapters.lerobot.demo_loader import make_demo_loader
    from evo_rlt.adapters.lerobot.pi05_adapter import DEFAULT_CAMERA_NAME_MAP
    from evo_rlt.core.algorithm import RLTAlgorithm
    from evo_rlt.core.trainer import demo_adaptation

    config = load_training_config(args.config)
    if args.steps is not None:
        config.demo_adaptation.steps = args.steps
    if args.batch_size is not None:
        config.demo_adaptation.batch_size = args.batch_size
    if args.lr is not None:
        config.demo_adaptation.lr = args.lr
    if args.num_rl_tokens is not None:
        config.rl_token.num_rl_tokens = args.num_rl_tokens
    if args.enc_layers is not None:
        config.rl_token.enc_layers = args.enc_layers
    if args.dec_layers is not None:
        config.rl_token.dec_layers = args.dec_layers
    # pi0.5 is frozen and not saved here (parse_args rejects a positive YAML value; the dataclass default is 1.0).
    config.demo_adaptation.vla_ft_weight = 0.0
    _seed_everything(config.seed)

    meta = _load_dataset_meta(args.demo_dataset_path)
    task_instruction = _resolve_task(args.task_instruction, meta)
    camera_name_map = args.camera_name_map or dict(DEFAULT_CAMERA_NAME_MAP)
    _check_cameras(meta, camera_name_map)
    active_cameras = [camera.strip() for camera in args.active_cameras.split(",") if camera.strip()] if args.active_cameras else None

    logger.info("Loading pi0.5 from %s (task=%r, cameras=%s)", args.model_path, task_instruction, camera_name_map)
    policy = build_pi05_policy(
        config=config,
        model_path=args.model_path,
        task_instruction=task_instruction,
        device=args.device,
        token_pool_size=args.token_pool_size,
        dtype=args.dtype,
        vla_cache_dir=args.vla_cache_dir,
        image_only=args.image_only,
        active_cameras=active_cameras,
        tokenizer_path=args.tokenizer_path,
        camera_name_map=camera_name_map,
    )
    algorithm = RLTAlgorithm(policy, config)
    logger.info(
        "Policy ready. RL token params: %.1fM",
        sum(parameter.numel() for parameter in policy.rl_token.parameters()) / 1e6,
    )

    rl_token_full = algorithm.build_rl_token_full(args.device)
    start_step = 0
    prior_losses = None
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=False)
        rl_token_full.load_state_dict(checkpoint["rl_token_state_dict"], strict=False)
        start_step = checkpoint.get("step", 0)
        prior_losses = checkpoint.get("losses", [])
        logger.info("Resumed RL token checkpoint at step %d", start_step)

    dim_std = None
    if args.norm_stats:
        stats = torch.load(args.norm_stats, map_location=args.device, weights_only=False)
        dim_std = stats["std"].to(args.device)
        logger.info("Loaded dim_std from %s (shape=%s, gamma=%.2f)", args.norm_stats, tuple(dim_std.shape), args.norm_gamma)

    demo_loader = make_demo_loader(
        dataset_path=args.demo_dataset_path,
        batch_size=config.demo_adaptation.batch_size,
        chunk_length=config.vla_horizon,
        repo_id=DEMO_REPO_ID,
        num_workers=args.num_workers,
        device=args.device,
        normalize_actions=args.normalize,
    )
    optimizer = torch.optim.Adam(rl_token_full.parameters(), lr=config.demo_adaptation.lr)

    logger.info(
        "Training RL token: steps=%d batch_size=%d lr=%.2e normalize=%s",
        config.demo_adaptation.steps,
        config.demo_adaptation.batch_size,
        config.demo_adaptation.lr,
        args.normalize,
    )
    start_time = time.time()
    losses = demo_adaptation(
        algorithm=algorithm,
        config=config,
        demo_loader=demo_loader,
        demo_optimizer=optimizer,
        rl_token_full=rl_token_full,
        save_dir=args.output_dir,
        save_every=args.save_every,
        start_step=start_step,
        prior_losses=prior_losses,
        metadata={
            "vla_model": args.model_path,
            "dataset": args.demo_dataset_path,
            "config": args.config,
            "task_instruction": task_instruction,
            "camera_name_map": camera_name_map,
            "normalize": args.normalize,
            "image_only": args.image_only,
            "active_cameras": active_cameras,
            "token_pool_size": args.token_pool_size,
            "num_rl_tokens": config.rl_token.num_rl_tokens,
            "norm_gamma": args.norm_gamma,
            "norm_stats": args.norm_stats,
        },
        dim_std=dim_std,
        norm_gamma=args.norm_gamma,
    )
    elapsed = time.time() - start_time
    final_loss = losses[-1] if losses else 0.0
    avg_last_100 = sum(losses[-100:]) / min(len(losses), 100) if losses else 0.0
    logger.info("RL token training finished in %.1fs. final=%.4f avg100=%.4f", elapsed, final_loss, avg_last_100)


if __name__ == "__main__":
    main()
