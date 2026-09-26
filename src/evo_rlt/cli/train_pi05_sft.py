#!/usr/bin/env python
"""pi0.5 SFT on a local LeRobot dataset, driven by a YAML config.

A thin wrapper around LeRobot's own `lerobot_train` (accelerate, wandb, checkpoints and resume all
come from there): options are gathered from `--config PATH.yaml` plus CLI flags, turned into
`lerobot_train` arguments, and run in-process after Evo-RLT's `register()`. With `--num-gpus N > 1`
the same command is relaunched under `accelerate launch`.

    evo-rlt-train-pi05-sft --config configs/train/pi05_sft_piper_blood_gas.yaml
    evo-rlt-train-pi05-sft --config configs/train/pi05_sft_piper_blood_gas.yaml --num-gpus 2
    evo-rlt-train-pi05-sft --config configs/train/pi05_sft_piper_blood_gas.yaml --resume
    evo-rlt-train-pi05-sft --config ... --dry-run            # print the lerobot command only

Unknown flags (e.g. `--policy.optimizer_lr=1e-5`) are forwarded to `lerobot_train` unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from evo_rlt.adapters.lerobot.record.cli import _coerce_config_value, _pop_config_path

DEFAULT_RENAME_MAP = {
    "observation.images.front": "observation.images.base_0_rgb",
    "observation.images.wrist": "observation.images.right_wrist_0_rgb",
}
# `--resume` without a value: resume the run in --output-dir.
RESUME_OUTPUT_DIR = "<output_dir>"
LAST_TRAIN_CONFIG = Path("checkpoints") / "last" / "pretrained_model" / "train_config.json"


def _json_dict(value: str) -> dict[str, str]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {value!r}")
    return parsed


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(" ", "").split(",") if item]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "pi0.5 SFT via LeRobot's lerobot_train. --config PATH.yaml supplies defaults (keys are the "
            "option names, `policy_path` == `--policy-path`); CLI flags override the file. Unknown flags "
            "are forwarded to lerobot_train."
        )
    )
    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset-root", required=True, help="Local LeRobot v3 dataset directory.")
    data.add_argument("--dataset-repo-id", default=None, help="Dataset name; default local/<root dir name>.")
    data.add_argument("--episodes", type=_int_list, default=None, help="Episode subset, e.g. '0,1,2'.")
    data.add_argument(
        "--rename-map",
        type=_json_dict,
        default=dict(DEFAULT_RENAME_MAP),
        help="JSON {dataset_key: policy_key}; maps dataset cameras onto pi05_base's camera slots.",
    )
    data.add_argument("--video-backend", default="pyav")
    data.add_argument("--tolerance-s", type=float, default=1e-4)
    data.add_argument("--image-transforms", action=argparse.BooleanOptionalAction, default=False)

    model = parser.add_argument_group("model")
    model.add_argument("--policy-path", default="lerobot/pi05_base")
    model.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    model.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--freeze-vision-encoder", action=argparse.BooleanOptionalAction, default=False)
    model.add_argument("--train-expert-only", action=argparse.BooleanOptionalAction, default=False)
    model.add_argument("--chunk-size", type=int, default=50)
    model.add_argument("--n-action-steps", type=int, default=50)
    model.add_argument("--device", default="cuda")
    model.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the policy forward; slower first steps, faster afterwards.",
    )
    model.add_argument("--compile-mode", default=None, help="torch.compile mode; default pi05 preset (max-autotune).")
    model.add_argument(
        "--drop-unused-cameras",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Remove policy camera slots that no dataset camera maps onto. Otherwise each one is fed a blank "
            "image that still runs through SigLIP and adds 256 masked prefix tokens."
        ),
    )

    train = parser.add_argument_group("training")
    train.add_argument("--output-dir", default=None, help="Run directory; default outputs/pi05_sft/<job_name>.")
    train.add_argument("--job-name", default="pi05_sft")
    train.add_argument("--steps", type=int, default=30_000)
    train.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size.")
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--save-freq", type=int, default=5_000)
    train.add_argument("--log-freq", type=int, default=100)
    train.add_argument("--seed", type=int, default=1000)
    train.add_argument("--lr", type=float, default=None, help="Peak LR; default pi05 preset (2.5e-5).")
    train.add_argument("--warmup-steps", type=int, default=None, help="Default pi05 preset (1000).")
    train.add_argument("--decay-steps", type=int, default=None, help="Cosine decay length; default --steps.")
    train.add_argument("--decay-lr", type=float, default=None, help="Final LR; default pi05 preset (2.5e-6).")

    wandb = parser.add_argument_group("wandb")
    wandb.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    wandb.add_argument("--wandb-project", default="evo_rlt_pi05_sft")
    wandb.add_argument("--wandb-entity", default=None)
    wandb.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    wandb.add_argument("--wandb-notes", default=None)
    wandb.add_argument(
        "--wandb-disable-artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip uploading each (~7GB) checkpoint as a wandb artifact.",
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "--resume",
        nargs="?",
        const=RESUME_OUTPUT_DIR,
        default=None,
        metavar="OUTPUT_DIR",
        help=(
            "Resume from OUTPUT_DIR/checkpoints/last (YAML `resume: true` = --output-dir). The checkpoint's "
            "train config wins; only steps/save_freq/log_freq/num_workers are re-applied, so a run can be "
            "extended. wandb continues the same run."
        ),
    )
    run.add_argument("--num-gpus", type=int, default=1, help=">1 relaunches under `accelerate launch` (DDP).")
    run.add_argument("--gpu-ids", default=None, help="Sets CUDA_VISIBLE_DEVICES, e.g. '0,1'.")
    run.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="no")
    run.add_argument(
        "--hf-offline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Set HF_HUB_OFFLINE=1: load the policy and tokenizer from the local HF cache only. Avoids Hub "
            "429s (transformers queries model_info on every tokenizer load when online)."
        ),
    )
    run.add_argument("--dry-run", action="store_true", default=False, help="Print the command and exit.")
    return parser


def _apply_train_config(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Load a flat YAML train config as parser defaults; explicit CLI flags still win."""
    import yaml

    with open(config_path) as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path}: top level must be a mapping of option -> value")

    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    defaults = {}
    for raw_key, value in config.items():
        key = str(raw_key).replace("-", "_")
        action = actions.get(key)
        if action is None:
            raise SystemExit(f"{config_path}: unknown option '{raw_key}'. Valid options: {sorted(actions)}")
        if value is None:
            # `key: null` leaves the built-in default (or the CLI) in charge.
            continue
        if key == "rename_map":
            if not isinstance(value, dict):
                raise SystemExit(f"config key '{raw_key}' must be a mapping, got {value!r}")
            defaults[key] = {str(k): str(v) for k, v in value.items()}
        elif key == "episodes":
            defaults[key] = [int(ep) for ep in value] if isinstance(value, list) else _int_list(str(value))
        else:
            defaults[key] = _coerce_config_value(action, raw_key, value)
        action.required = False
    parser.set_defaults(**defaults)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    config_path, argv = _pop_config_path(argv)
    if config_path is not None:
        _apply_train_config(parser, config_path)
    args, extra = parser.parse_known_args(argv)
    if args.output_dir is None:
        args.output_dir = str(Path("outputs") / "pi05_sft" / args.job_name)
    if args.resume == RESUME_OUTPUT_DIR:
        args.resume = args.output_dir
    return args, extra


def resolve_resume_config(run_dir: str | Path) -> Path:
    config_path = Path(run_dir) / LAST_TRAIN_CONFIG
    if not config_path.is_file():
        raise SystemExit(
            f"--resume: no checkpoint at {config_path}; the run needs at least one saved checkpoint "
            "(checkpoints/last is written on every save)"
        )
    return config_path


def _bool(value: bool) -> str:
    return "true" if value else "false"


def build_lerobot_argv(
    args: argparse.Namespace, extra: list[str] = (), check_output_dir: bool = True
) -> list[str]:
    """`lerobot_train` arguments for a new run, or for resuming `args.resume`.

    *check_output_dir* is off on accelerate ranks: the launcher already checked, and rank 0 may
    have created the directory by the time a slower rank gets here.
    """
    run_overrides = [
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--num_workers={args.num_workers}",
    ]
    if args.resume:
        config_path = resolve_resume_config(args.resume)
        return ["--resume=true", f"--config_path={config_path}", *run_overrides, *extra]

    if check_output_dir and Path(args.output_dir).exists():
        raise SystemExit(
            f"{args.output_dir} already exists; pass --resume to continue that run, or pick another --output-dir"
        )
    dataset_root = Path(args.dataset_root)
    repo_id = args.dataset_repo_id or f"local/{dataset_root.name}"
    argv = [
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={args.video_backend}",
        f"--dataset.image_transforms.enable={_bool(args.image_transforms)}",
        f"--tolerance_s={args.tolerance_s}",
        f"--policy.path={args.policy_path}",
        f"--policy.device={args.device}",
        f"--policy.dtype={args.dtype}",
        f"--policy.gradient_checkpointing={_bool(args.gradient_checkpointing)}",
        f"--policy.freeze_vision_encoder={_bool(args.freeze_vision_encoder)}",
        f"--policy.train_expert_only={_bool(args.train_expert_only)}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.compile_model={_bool(args.compile_model)}",
        f"--policy.scheduler_decay_steps={args.decay_steps or args.steps}",
        "--policy.push_to_hub=false",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--batch_size={args.batch_size}",
        f"--seed={args.seed}",
        "--eval_freq=0",
        *run_overrides,
        f"--wandb.enable={_bool(args.wandb)}",
        f"--wandb.project={args.wandb_project}",
        f"--wandb.disable_artifact={_bool(args.wandb_disable_artifact)}",
    ]
    if args.episodes:
        argv.append(f"--dataset.episodes={json.dumps(args.episodes)}")
    if args.rename_map:
        argv.append(f"--rename_map={json.dumps(args.rename_map)}")
    if args.compile_mode:
        argv.append(f"--policy.compile_mode={args.compile_mode}")
    if args.lr is not None:
        argv.append(f"--policy.optimizer_lr={args.lr}")
    if args.warmup_steps is not None:
        argv.append(f"--policy.scheduler_warmup_steps={args.warmup_steps}")
    if args.decay_lr is not None:
        argv.append(f"--policy.scheduler_decay_lr={args.decay_lr}")
    if args.wandb_entity:
        argv.append(f"--wandb.entity={args.wandb_entity}")
    if args.wandb_mode:
        argv.append(f"--wandb.mode={args.wandb_mode}")
    if args.wandb_notes:
        argv.append(f"--wandb.notes={args.wandb_notes}")
    return [*argv, *extra]


def dataset_camera_keys(dataset_root: str | Path, rename_map: dict[str, str]) -> set[str]:
    """Policy-side keys of the dataset's cameras, after *rename_map*."""
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"--drop-unused-cameras: no {info_path} to read the dataset cameras from")
    features = json.loads(info_path.read_text())["features"]
    return {rename_map.get(key, key) for key, ft in features.items() if ft.get("dtype") in ("video", "image")}


def drop_unused_cameras(policy_cfg, keep: set[str]) -> list[str]:
    """Remove visual input features not in *keep* from *policy_cfg*; returns the removed keys."""
    from lerobot.configs.types import FeatureType

    visual = [key for key, ft in policy_cfg.input_features.items() if ft.type is FeatureType.VISUAL]
    dropped = [key for key in visual if key not in keep]
    if len(dropped) == len(visual):
        raise SystemExit(
            f"--drop-unused-cameras: no dataset camera maps onto the policy's slots {visual} "
            f"(dataset cameras after rename_map: {sorted(keep)})"
        )
    for key in dropped:
        del policy_cfg.input_features[key]
    return dropped


def install_camera_pruning(keep: set[str]) -> None:
    """Drop unused camera slots right after lerobot_train loads the policy config.

    A `--policy.input_features=...` override can't do it: draccus merges dicts, so keys can only be
    added. The pruned features are saved with every checkpoint, so resume and deployment see the
    same cameras.
    """
    from lerobot.configs.train import TrainPipelineConfig

    validate = TrainPipelineConfig.validate

    def validate_and_prune(self) -> None:
        validate(self)
        dropped = drop_unused_cameras(self.policy, keep)
        if dropped and os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[pi05-sft] dropped unused camera slots: {dropped}")

    TrainPipelineConfig.validate = validate_and_prune


def build_accelerate_command(args: argparse.Namespace, argv: list[str]) -> list[str]:
    """Relaunch this module under accelerate; each rank then runs lerobot_train in-process."""
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        f"--num_processes={args.num_gpus}",
        "--num_machines=1",
        f"--mixed_precision={args.mixed_precision}",
        "--dynamo_backend=no",
        "-m",
        "evo_rlt.cli.train_pi05_sft",
        *argv,
    ]


def _print_summary(args: argparse.Namespace) -> None:
    if args.resume:
        print(f"[pi05-sft] resume {args.resume} -> steps={args.steps}")
        return
    info_path = Path(args.dataset_root) / "meta" / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        dataset = f"{info.get('total_episodes')} episodes / {info.get('total_frames')} frames @ {info.get('fps')}fps"
    else:
        dataset = f"(no {info_path})"
    print(
        f"[pi05-sft] dataset {args.dataset_root}: {dataset}\n"
        f"[pi05-sft] policy {args.policy_path} | batch {args.batch_size} x {args.num_gpus} gpu | "
        f"steps {args.steps} | output {args.output_dir} | wandb {'on' if args.wandb else 'off'}"
    )


def main(argv: list[str] | None = None) -> None:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args, extra = parse_args(raw_argv)
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.hf_offline:
        # Must precede any huggingface_hub/transformers import; accelerate ranks inherit it.
        os.environ["HF_HUB_OFFLINE"] = "1"

    launch_distributed = args.num_gpus > 1 and "LOCAL_RANK" not in os.environ
    if launch_distributed:
        # Validate (resume checkpoint, output dir) before spawning ranks.
        build_lerobot_argv(args, extra)
        command = build_accelerate_command(args, raw_argv)
        _print_summary(args)
        print("[pi05-sft] " + shlex.join(command))
        if args.dry_run:
            return
        raise SystemExit(subprocess.run(command).returncode)

    is_rank = "LOCAL_RANK" in os.environ
    lerobot_argv = build_lerobot_argv(args, extra, check_output_dir=not is_rank)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        _print_summary(args)
        print("[pi05-sft] lerobot_train " + shlex.join(lerobot_argv))
    if args.dry_run:
        return

    from accelerate.utils import check_cuda_p2p_ib_support

    if not check_cuda_p2p_ib_support():
        # RTX 40xx: Accelerator() refuses to start unless these are set (`accelerate launch` does it).
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")

    from evo_rlt.adapters.lerobot import register

    register()
    if args.drop_unused_cameras and not args.resume:
        # A resumed run's checkpoint config already carries the pruned camera set.
        install_camera_pruning(dataset_camera_keys(args.dataset_root, args.rename_map))
    from lerobot.scripts import lerobot_train

    sys.argv = ["lerobot_train", *lerobot_argv]
    lerobot_train.main()


if __name__ == "__main__":
    main()
