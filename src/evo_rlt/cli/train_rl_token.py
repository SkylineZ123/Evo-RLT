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
import dataclasses
import functools
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

from evo_rlt.adapters.lerobot.record.cli import _pop_config_path
from evo_rlt.cli.common import (
    apply_run_section,
    build_pi05_policy,
    configure_logging,
    json_dict,
    load_training_config,
)

logger = configure_logging(__name__)

DEMO_REPO_ID = "rlt_demo"
TOKEN_STATS_FILE = "token_stats.pt"


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
        type=json_dict,
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
    parser.add_argument(
        "--norm-stats",
        default=None,
        help=(
            "Per-dim std for weighted MSE: a .pt file with a 'std' tensor, or 'auto' to estimate it from "
            f"--norm-stats-batches training batches (saved as <output-dir>/{TOKEN_STATS_FILE} and reused on resume)."
        ),
    )
    parser.add_argument("--norm-stats-batches", type=int, default=50, help="Batches used by --norm-stats auto.")
    parser.add_argument("--norm-gamma", type=float, default=0.0, help="Per-dim weighting exponent. 0=raw MSE, 0.5=partial whitening, 1=full whitening.")
    parser.add_argument(
        "--val-episodes",
        type=int,
        default=0,
        help="Hold out this many episodes (picked with the config seed) for validation; 0 = train on all.",
    )
    parser.add_argument(
        "--val-samples", type=int, default=64, help="Held-out frames evaluated (spread over the val episodes)."
    )
    parser.add_argument("--eval-every", type=int, default=1000, help="Validate every N steps (needs --val-episodes).")
    parser.add_argument("--num-rl-tokens", type=int, default=None, help="Override config.rl_token.num_rl_tokens.")
    parser.add_argument("--enc-layers", type=int, default=None, help="Override config.rl_token.enc_layers.")
    parser.add_argument("--dec-layers", type=int, default=None, help="Override config.rl_token.dec_layers.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="VLA model dtype")
    parser.add_argument("--vla-cache-dir", default=None, help="Optional Pi0.5 cache directory.")
    parser.add_argument("--tokenizer-path", default=None, help="PaliGemma tokenizer repo id or local snapshot path.")
    return parser


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
        apply_run_section(parser, raw.get("run") or {}, config_path)
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


def _split_episodes(num_episodes: int, num_val: int, seed: int) -> tuple[list[int] | None, list[int]]:
    """Pick `num_val` held-out episodes. Returns (train episodes, or None for all; val episodes)."""
    if num_val <= 0:
        return None, []
    if num_val >= num_episodes:
        raise SystemExit(f"--val-episodes {num_val} leaves no training episodes (dataset has {num_episodes})")
    val = sorted(random.Random(seed).sample(range(num_episodes), num_val))
    return [episode for episode in range(num_episodes) if episode not in val], val


def _to_device(obs, device: str):
    from evo_rlt.core.interfaces import Observation

    return Observation(images={k: v.to(device) for k, v in obs.images.items()}, proprio=obs.proprio.to(device))


@torch.no_grad()
def _estimate_token_stats(vla, demo_loader, num_batches: int) -> dict:
    """Per-dim mean/std of the prefix tokens the RL token encodes, merged batch by batch (Chan et al.)."""
    n = 0
    mean = m2 = None
    for i, (obs, _) in enumerate(demo_loader):
        if i >= num_batches:
            break
        x = vla.prefix_tokens(obs).float().reshape(-1, vla.token_dim)
        batch_n = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(dim=0)
        if mean is None:
            mean, m2 = torch.zeros_like(batch_mean), torch.zeros_like(batch_m2)
        delta = batch_mean - mean
        total = n + batch_n
        mean = mean + delta * batch_n / total
        m2 = m2 + batch_m2 + delta ** 2 * n * batch_n / total
        n = total
    std = (m2 / max(n - 1, 1)).sqrt()
    return {"mean": mean.cpu(), "std": std.cpu(), "n": n}


def _resolve_dim_std(args: argparse.Namespace, vla, demo_loader, prefix: dict) -> torch.Tensor | None:
    """Per-dim std for the weighted reconstruction loss: from --norm-stats PATH, or estimated for 'auto'."""
    if not args.norm_stats:
        return None
    if args.norm_stats != "auto":
        stats = torch.load(args.norm_stats, map_location="cpu", weights_only=False)
        logger.info("Loaded dim_std from %s (gamma=%.2f)", args.norm_stats, args.norm_gamma)
        return stats["std"].to(args.device)

    path = Path(args.output_dir) / TOKEN_STATS_FILE
    if path.is_file():
        stats = torch.load(path, map_location="cpu", weights_only=False)
        if stats.get("prefix") != prefix:
            raise SystemExit(
                f"{path} was computed for prefix tokens {stats.get('prefix')} but this run uses {prefix}; "
                "delete it or use another --output-dir"
            )
        logger.info("Reusing prefix-token stats from %s", path)
    else:
        logger.info(
            "Estimating per-dim prefix-token std from %d batches (--norm-stats auto)", args.norm_stats_batches
        )
        stats = {**_estimate_token_stats(vla, demo_loader, args.norm_stats_batches), "prefix": prefix}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, path)
        var = stats["std"] ** 2
        logger.info(
            "Saved %s (%d tokens): std median=%.4f max=%.4f; top-10 dims hold %.1f%% of the variance",
            path,
            stats["n"],
            stats["std"].median(),
            stats["std"].max(),
            100 * var.topk(10).values.sum() / var.sum(),
        )
    return stats["std"].to(args.device)


@torch.no_grad()
def _build_val_batches(
    vla, args: argparse.Namespace, episodes: list[int], chunk_length: int, batch_size: int, normalization_stats
):
    """Prefix tokens of --val-samples frames spread over the held-out episodes, computed once (pi0.5 is frozen)."""
    from evo_rlt.adapters.lerobot.demo_loader import RLTDemoDataset, rlt_demo_collate

    dataset = RLTDemoDataset(
        dataset_path=args.demo_dataset_path,
        repo_id=DEMO_REPO_ID,
        chunk_length=chunk_length,
        normalize_actions=args.normalize,
        episodes=episodes,
        normalization_stats=normalization_stats,
    )
    num_samples = min(args.val_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, num_samples).round().astype(int).tolist()
    num_batches = max(1, num_samples // batch_size)
    batches = []
    for b in range(num_batches):
        # Strided, so every batch spans all held-out episodes and the shuffled-z_rl baseline
        # swaps in the z_rl of a genuinely different state.
        obs, _ = rlt_demo_collate([dataset[i] for i in indices[b::num_batches]])
        batches.append(vla.prefix_tokens(_to_device(obs, args.device)).cpu())
    return batches


def main() -> None:
    args = parse_args()

    from evo_rlt.adapters.lerobot.demo_loader import IMAGE_RESIZE, load_policy_normalization_stats, make_demo_loader
    from evo_rlt.adapters.lerobot.pi05_adapter import DEFAULT_CAMERA_NAME_MAP
    from evo_rlt.core.algorithm import RLTAlgorithm
    from evo_rlt.core.trainer import demo_adaptation, evaluate_reconstruction

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
    train_episodes, val_episodes = _split_episodes(meta.total_episodes, args.val_episodes, config.seed)

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
    rl_token_full = algorithm.build_rl_token_full(args.device)
    logger.info(
        "Policy ready. RL token arch=%s seq_len=%s num_rl_tokens=%d, params: %.1fM (encoder %.1fM)",
        rl_token_full.arch,
        rl_token_full.seq_len,
        rl_token_full.num_rl_tokens,
        sum(parameter.numel() for parameter in rl_token_full.parameters()) / 1e6,
        sum(parameter.numel() for parameter in policy.rl_token.parameters()) / 1e6,
    )

    start_step = 0
    prior_losses = None
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=args.device, weights_only=False)
        rl_token_full.load_state_dict(checkpoint["rl_token_state_dict"])
        start_step = checkpoint.get("step", 0)
        prior_losses = checkpoint.get("losses", [])
        logger.info("Resumed RL token checkpoint at step %d", start_step)

    # pi0.5 reads the normalized state (prompt tokens), so it gets the SFT quantiles, not the dataset's.
    normalization_stats = load_policy_normalization_stats(args.model_path) if args.normalize else None
    demo_loader = make_demo_loader(
        dataset_path=args.demo_dataset_path,
        batch_size=config.demo_adaptation.batch_size,
        chunk_length=config.vla_horizon,
        repo_id=DEMO_REPO_ID,
        num_workers=args.num_workers,
        device=args.device,
        normalize_actions=args.normalize,
        episodes=train_episodes,
        normalization_stats=normalization_stats,
    )
    prefix = {
        "model_path": args.model_path,
        "dataset": args.demo_dataset_path,
        "episodes": train_episodes,
        "camera_name_map": camera_name_map,
        "image_only": args.image_only,
        "active_cameras": active_cameras,
        "token_pool_size": args.token_pool_size,
        "image_resize": IMAGE_RESIZE,
    }
    dim_std = _resolve_dim_std(args, policy.vla, demo_loader, prefix)

    eval_fn = None
    if val_episodes:
        val_batches = _build_val_batches(
            policy.vla, args, val_episodes, config.vla_horizon, config.demo_adaptation.batch_size, normalization_stats
        )
        logger.info(
            "Validation on held-out episodes %s: %d frames in %d batches, every %d steps",
            val_episodes, sum(len(batch) for batch in val_batches), len(val_batches), args.eval_every,
        )
        eval_fn = functools.partial(
            evaluate_reconstruction,
            token_batches=val_batches,
            dim_std=dim_std,
            norm_gamma=args.norm_gamma,
            device=args.device,
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
            "image_resize": IMAGE_RESIZE,
            "num_rl_tokens": config.rl_token.num_rl_tokens,
            "rl_token": {**dataclasses.asdict(config.rl_token), "seq_len": rl_token_full.seq_len},
            "val_episodes": val_episodes,
            "norm_gamma": args.norm_gamma,
            "norm_stats": args.norm_stats,
        },
        dim_std=dim_std,
        norm_gamma=args.norm_gamma,
        eval_fn=eval_fn,
        eval_every=args.eval_every,
    )
    elapsed = time.time() - start_time
    final_loss = losses[-1] if losses else 0.0
    avg_last_100 = sum(losses[-100:]) / min(len(losses), 100) if losses else 0.0
    logger.info("RL token training finished in %.1fs. final=%.4f avg100=%.4f", elapsed, final_loss, avg_last_100)


if __name__ == "__main__":
    main()
