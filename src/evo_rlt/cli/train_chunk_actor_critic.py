#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evo_rlt.cli.common import (
    build_pi05_policy,
    configure_logging,
    load_training_config,
    parse_args_with_run_section,
)

logger = configure_logging(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train chunk-level actor-critic from a transition cache or raw demo dataset. --config PATH.yaml holds "
            "the RLTConfig sections plus a `run:` section of defaults for the options below; CLI flags override it."
        )
    )
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument(
        "--transition-cache-dir", nargs="+", default=None,
        help="One or more chunk-transition cache directories (e.g. a warmup_vla bucket and rl_rollout buckets).",
    )
    parser.add_argument(
        "--cache-mix-weights", nargs="+", type=float, default=None,
        help=(
            "Sampling weight per --transition-cache-dir (normalized): every batch takes that share of its "
            "transitions from each cache. Default: each cache's share of all train transitions."
        ),
    )
    parser.add_argument("--demo-dataset-path", default=None, help="Raw demo dataset path. Required when no transition cache is provided.")
    parser.add_argument("--config", default=None, help="Path to an RLT YAML config")
    parser.add_argument("--output-dir", default="outputs/rlt_actor_critic")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rl-token-checkpoint", default=None, help="RL token checkpoint to initialize the policy encoder.")
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--token-pool-size", type=int, default=0, help="Pool prefix tokens before RL token encoding (0 disables pooling).")
    parser.add_argument("--image-only", action="store_true", help="Drop language tokens before RL token encode (image-patch tokens only).")
    parser.add_argument("--task-instruction", default="pick up the object")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--tokenizer-path", default=None, help="PaliGemma tokenizer repo id or local snapshot path.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return parse_args_with_run_section(build_parser(), argv)


def apply_overrides(config, args: argparse.Namespace) -> None:
    if args.gradient_steps is not None:
        config.offline_rl.num_gradient_steps = args.gradient_steps
    if args.eval_every is not None:
        config.offline_rl.eval_every = args.eval_every
    if args.save_every is not None:
        config.offline_rl.save_every = args.save_every
    if args.log_every is not None:
        config.offline_rl.log_every = args.log_every
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.beta is not None:
        config.training.beta = args.beta
    if args.actor_lr is not None:
        config.actor.lr = args.actor_lr
    if args.critic_lr is not None:
        config.critic.lr = args.critic_lr


def load_cached_replay_buffers(
    transition_cache_dirs: str | list[str], capacity: int, mix_weights: list[float] | None = None
) -> tuple:
    """Train/val buffers over one or more caches; several caches are mixed per batch by `mix_weights`.

    A cache without a val split (e.g. a rollout bucket built with val_ratio 0) only feeds training; val is
    None when no cache has one.
    """
    from evo_rlt.adapters.lerobot.offline_dataset import load_transition_cache

    dirs = [transition_cache_dirs] if isinstance(transition_cache_dirs, str) else list(transition_cache_dirs)
    if mix_weights is not None:
        if len(mix_weights) != len(dirs):
            raise ValueError(f"cache_mix_weights has {len(mix_weights)} values for {len(dirs)} transition cache dirs")
        if any(w <= 0 for w in mix_weights):
            raise ValueError(f"cache_mix_weights must be positive, got {mix_weights}")
    names = [f"{i}:{Path(d).name}" for i, d in enumerate(dirs)]
    train = {name: load_transition_cache(d, "train", capacity=capacity) for name, d in zip(names, dirs)}
    val = {
        name: load_transition_cache(d, "val", capacity=capacity)
        for name, d in zip(names, dirs)
        if (Path(d) / "chunk_transitions_val.pt").exists()
    }
    if len(train) == 1:
        (name,) = train
        if name not in val:
            logger.warning("%s has no val split; training without validation", dirs[0])
        return train[name], val.get(name)

    total = sum(len(buf) for buf in train.values())
    weights = dict(zip(names, mix_weights or [len(buf) / total for buf in train.values()]))
    weight_sum = sum(weights.values())
    for name, d in zip(names, dirs):
        logger.info(
            "Train cache %s: %d transitions, %.1f%% of each batch; val: %s",
            d, len(train[name]), 100.0 * weights[name] / weight_sum,
            f"{len(val[name])} transitions" if name in val else "none (train only)",
        )
    return _mixed_buffer(train, weights), _mixed_buffer(val, weights)


def _mixed_buffer(buffers: dict, weights: dict[str, float]):
    from evo_rlt.core.mix_dataset import WeightedMixReplayBuffer

    if not buffers:
        return None
    if len(buffers) == 1:
        return next(iter(buffers.values()))
    return WeightedMixReplayBuffer(buffers, {name: weights[name] for name in buffers})


def build_live_replay_buffers(policy, args: argparse.Namespace, config) -> tuple:
    from evo_rlt.adapters.lerobot.demo_loader import load_policy_normalization_stats
    from evo_rlt.adapters.lerobot.offline_dataset import build_transition_replay_buffer

    normalization_stats = load_policy_normalization_stats(args.model_path)
    logger.info("Building train replay buffer from %s", args.demo_dataset_path)
    train_buffer = build_transition_replay_buffer(
        policy=policy,
        demo_dataset_path=args.demo_dataset_path,
        config=config,
        split="train",
        device=args.device,
        normalization_stats=normalization_stats,
    )
    logger.info("Building val replay buffer from %s", args.demo_dataset_path)
    val_buffer = build_transition_replay_buffer(
        policy=policy,
        demo_dataset_path=args.demo_dataset_path,
        config=config,
        split="val",
        device=args.device,
        normalization_stats=normalization_stats,
    )
    return train_buffer, val_buffer


def configure_action_bounds(actor, buffer, margin: float | None) -> None:
    """Bound the TD-target action to the range of the executed actions the critic is trained on."""
    from evo_rlt.core.mix_dataset import WeightedMixReplayBuffer

    if margin is None:
        logger.info("Action bounds: fixed [-1, 1] (actor.action_bound_margin is null)")
        return
    buffers = buffer.buckets.values() if isinstance(buffer, WeightedMixReplayBuffer) else [buffer]
    actions = torch.cat([t.exec_chunk for buf in buffers for t in buf.buffer])  # (N * C, action_dim)
    low, high = actions.amin(dim=0), actions.amax(dim=0)
    pad = margin * (high - low)
    actor.set_action_bounds(low - pad, high + pad)
    logger.info(
        "Action bounds from %d executed actions (margin %.2f): low=%s high=%s",
        actions.shape[0], margin,
        [round(v, 3) for v in (low - pad).tolist()], [round(v, 3) for v in (high + pad).tolist()],
    )


def create_algorithm_with_cached_transitions(config, rl_token_checkpoint: str | None, device: str):
    from evo_rlt.core.algorithm import RLTAlgorithm
    from evo_rlt.core.policy import RLTPolicy
    from evo_rlt.core.rl_token import load_rl_token_encoder
    from evo_rlt.core.vla_adapter import DummyVLAAdapter

    rl_token_state = None
    if rl_token_checkpoint is not None:
        checkpoint = torch.load(rl_token_checkpoint, map_location="cpu", weights_only=False)
        rl_token_state = checkpoint["rl_token_state_dict"]
        config.rl_token.update_from_state_dict(rl_token_state)

    vla = DummyVLAAdapter(
        token_dim=config.rl_token.token_dim,
        action_dim=config.action_dim,
        num_tokens=64,
        horizon=config.vla_horizon,
    )
    policy = RLTPolicy(config, vla).to(device)
    if rl_token_state is not None:
        load_rl_token_encoder(policy.rl_token, rl_token_state)
        logger.info("Loaded RL token checkpoint from %s", rl_token_checkpoint)

    policy.freeze_vla()
    policy.freeze_rl_token_encoder()
    algorithm = RLTAlgorithm(policy, config)
    algorithm.to(device)
    return algorithm


def create_algorithm_with_pi05(config, args: argparse.Namespace):
    from evo_rlt.core.algorithm import RLTAlgorithm

    policy = build_pi05_policy(
        config=config,
        model_path=args.model_path,
        task_instruction=args.task_instruction,
        device=args.device,
        token_pool_size=args.token_pool_size,
        dtype=args.dtype,
        rl_token_checkpoint=args.rl_token_checkpoint,
        image_only=args.image_only,
        tokenizer_path=args.tokenizer_path,
    )
    policy.freeze_vla()
    policy.freeze_rl_token_encoder()

    algorithm = RLTAlgorithm(policy, config)
    algorithm.to(args.device)
    return algorithm


def main() -> None:
    args = parse_args()

    from evo_rlt.core.evaluator import evaluate_offline
    from evo_rlt.core.trainer import offline_rl_loop

    config = load_training_config(args.config)
    apply_overrides(config, args)

    metadata = None
    if args.transition_cache_dir is not None:
        logger.info("Loading transition cache from %s", args.transition_cache_dir)
        train_buffer, val_buffer = load_cached_replay_buffers(
            args.transition_cache_dir, config.replay.capacity, args.cache_mix_weights
        )
        metadata = {"transition_cache_dirs": list(args.transition_cache_dir), "cache_mix_weights": args.cache_mix_weights}
        algorithm = create_algorithm_with_cached_transitions(config, args.rl_token_checkpoint, args.device)
    else:
        if args.cache_mix_weights is not None:
            raise ValueError("--cache-mix-weights needs --transition-cache-dir")
        if args.demo_dataset_path is None:
            raise ValueError("--demo-dataset-path is required when --transition-cache-dir is not provided")
        algorithm = create_algorithm_with_pi05(config, args)
        train_buffer, val_buffer = build_live_replay_buffers(algorithm.policy, args, config)
    configure_action_bounds(algorithm.policy.actor, train_buffer, config.actor.action_bound_margin)

    logger.info(
        "Train transitions: %d, Val transitions: %s", len(train_buffer), len(val_buffer) if val_buffer else "none"
    )
    logger.info(
        "Training chunk actor-critic: steps=%d batch_size=%d beta=%.2f actor_lr=%.2e critic_lr=%.2e",
        config.offline_rl.num_gradient_steps,
        config.training.batch_size,
        config.training.beta,
        config.actor.lr,
        config.critic.lr,
    )

    actor_optimizer = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=config.actor.lr)
    critic_optimizer = torch.optim.Adam(algorithm.critic.parameters(), lr=config.critic.lr)

    start_time = time.time()
    metrics = offline_rl_loop(
        algorithm=algorithm,
        config=config,
        replay_buffer=train_buffer,
        val_buffer=val_buffer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        save_dir=args.output_dir,
        metadata=metadata,
    )
    elapsed = time.time() - start_time
    logger.info(
        "Actor-critic training finished in %.1fs. critic_updates=%d actor_updates=%d",
        elapsed,
        len(metrics.critic_losses),
        len(metrics.actor_losses),
    )

    if val_buffer is None:
        logger.info("No val split in any transition cache; skipping offline eval")
        return
    eval_metrics = evaluate_offline(algorithm, val_buffer, config, num_batches=10)
    logger.info(
        "Eval: expert_mse=%.4f ref_mse=%.4f q_policy=%.4f q_expert=%.4f q_gap=%.4f td_err=%.4f",
        eval_metrics.expert_action_mse,
        eval_metrics.ref_action_mse,
        eval_metrics.mean_q_policy,
        eval_metrics.mean_q_expert,
        eval_metrics.q_gap,
        eval_metrics.mean_critic_td_error,
    )


if __name__ == "__main__":
    main()
