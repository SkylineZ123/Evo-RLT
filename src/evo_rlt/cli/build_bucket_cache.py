#!/usr/bin/env python
"""Build a 3-bucket chunk-transition cache for RLT online-offline training.

For each bucket:
  - warmup_vla:   ref_chunk = VLA-proposed action (no replacement)
  - human_expert: ref_chunk = exec_chunk (always, teleop-only dataset)
  - rl_rollout:   ref_chunk = exec_chunk only for chunks where the dominant
                  per-frame source is human intervention (chunk-level vote,
                  matching online_collector.py semantics)

--human-label replace_ref (above, the RLT paper's Alg. 1) turns a human chunk into both the actor's input
and its BC target. --human-label bc_target keeps the pi0.5 ref as the actor input (what deploy feeds it)
and stores bc_target_chunk instead: the human action on intervened steps, the VLA ref elsewhere
(openpi-RLT's per-step mask; human_expert treats every step as intervened).

--config PATH.yaml holds the RLTConfig sections plus a `run:` section of defaults
for the options below (`bucket_mode` == `--bucket-mode`); CLI flags override it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evo_rlt.cli.common import configure_logging, json_dict, load_training_config, parse_args_with_run_section

logger = configure_logging(__name__)


HUMAN_LABEL_MODES = ("replace_ref", "bc_target")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-dataset-path", default=None)
    parser.add_argument("--transition-cache-dir", required=True)
    parser.add_argument("--bucket-mode", choices=["warmup_vla", "human_expert", "rl_rollout"], required=True)
    parser.add_argument("--model-path", default="lerobot/pi05_base")
    parser.add_argument("--rl-token-checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mark-critical", action="store_true")
    parser.add_argument(
        "--human-label",
        choices=HUMAN_LABEL_MODES,
        default="replace_ref",
        help="replace_ref: human chunks become the ref (actor input and BC target); "
        "bc_target: ref stays the VLA's, the human action is only the BC target (per step).",
    )
    parser.add_argument("--token-pool-size", type=int, default=64)
    parser.add_argument("--image-only", action="store_true", help="Drop language tokens before RL token encode.")
    parser.add_argument("--active-cameras", default=None,
                        help="Comma-separated camera names (e.g. 'front,wrist'). Implies image-only.")
    parser.add_argument(
        "--camera-name-map",
        type=json_dict,
        default=None,
        help=(
            'JSON {dataset camera: pi0.5 slot}, e.g. {"front": "observation.images.base_0_rgb"}; '
            "default: the bimanual left_wrist/right_wrist/right_front map. Must match RL token training."
        ),
    )
    parser.add_argument("--tokenizer-path", default=None, help="PaliGemma tokenizer repo id or local snapshot path.")
    parser.add_argument("--task-instruction", default="Insert the copper screw into the black sleeve.")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for video decoding.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--repo-id", default="rlt_demo")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return parse_args_with_run_section(build_parser(), argv)


def _collect_intervention_array(dataset) -> torch.Tensor | None:
    """Return (N,) float tensor of is_intervention flags if present, else None."""
    hf = getattr(dataset._dataset, "hf_dataset", None)
    if hf is None:
        return None
    cols = getattr(hf, "column_names", [])
    key = "complementary_info.is_intervention"
    if key not in cols:
        return None
    raw = hf[key]
    return torch.tensor([float(v) for v in raw], dtype=torch.float32)


def _dominant_is_human(interventions: torch.Tensor, start: int, length: int) -> bool:
    """Return True when human frames win the majority vote over [start, start+length)."""
    window = interventions[start : start + length]
    human = int((window > 0.5).sum().item())
    return human >= (length - human)  # human wins ties, matches online_collector rule


def apply_bucket_refs(
    transitions: list,
    start_frames: list[int],
    bucket_mode: str,
    interventions: torch.Tensor | None,
    chunk_length: int,
    human_label: str = "replace_ref",
) -> int:
    """Label one episode's human chunks for `bucket_mode`; returns how many chunks are human.

    A chunk is human for human_expert always, for rl_rollout when its executed frames are mostly human.
    replace_ref: a human chunk's ref becomes its exec_chunk, and each next_ref_chunk then follows the
    replaced ref of the transition starting at t+C.
    bc_target: refs stay the VLA's; every transition with a human step gets bc_target_chunk = the executed
    action on human steps and the ref on the others (and on the padded tail of a truncated chunk).
    """
    if human_label not in HUMAN_LABEL_MODES:
        raise ValueError(f"human_label must be one of {HUMAN_LABEL_MODES}, got {human_label!r}")
    replaced = [False] * len(transitions)
    for i, (t, start) in enumerate(zip(transitions, start_frames)):
        steps = int(t.actual_steps)
        human_chunk = bucket_mode == "human_expert" or (
            bucket_mode == "rl_rollout" and _dominant_is_human(interventions, start, steps)
        )
        if human_chunk:
            t.intervention = torch.tensor(1.0)
            replaced[i] = True
        if human_label == "replace_ref":
            if human_chunk:
                t.ref_chunk = t.exec_chunk.clone()
            continue
        mask = torch.zeros(t.ref_chunk.shape[0], dtype=torch.bool)
        if bucket_mode == "human_expert":
            mask[:steps] = True
        elif bucket_mode == "rl_rollout":
            mask[:steps] = interventions[start : start + steps] > 0.5
        if mask.any():
            t.bc_target_chunk = torch.where(mask[:, None], t.exec_chunk, t.ref_chunk)
    if human_label == "bc_target":
        return sum(replaced)

    start_to_idx = {start: i for i, start in enumerate(start_frames)}
    for t, start in zip(transitions, start_frames):
        nxt = start_to_idx.get(start + chunk_length)
        if nxt is not None and replaced[nxt]:
            t.next_ref_chunk = transitions[nxt].ref_chunk.clone()
    return sum(replaced)


def main() -> None:
    args = parse_args()

    from evo_rlt.cli.common import build_pi05_policy
    from evo_rlt.adapters.lerobot.demo_loader import (
        RLTDemoDataset,
        load_policy_normalization_stats,
        rlt_demo_collate,
    )
    from evo_rlt.adapters.lerobot.offline_dataset import (
        _count_episodes,
        _episode_frame_range,
        build_overlap_frame_indices,
        build_transitions_from_demos,
        save_transition_cache,
        split_episode_indices,
        transition_start_frames,
    )

    config = load_training_config(args.config)
    config.offline_rl.frame_stride = args.frame_stride
    config.offline_rl.train_ratio = args.train_ratio
    config.offline_rl.val_ratio = args.val_ratio

    logger.info("Loading pi0.5 from %s", args.model_path)
    policy = build_pi05_policy(
        config=config,
        model_path=args.model_path,
        task_instruction=args.task_instruction,
        device=args.device,
        token_pool_size=args.token_pool_size,
        dtype=args.dtype,
        rl_token_checkpoint=args.rl_token_checkpoint,
        image_only=args.image_only,
        active_cameras=[c.strip() for c in args.active_cameras.split(",") if c.strip()] if args.active_cameras else None,
        tokenizer_path=args.tokenizer_path,
        camera_name_map=args.camera_name_map,
    )
    policy.freeze_vla()
    policy.freeze_rl_token_encoder()
    policy.eval()

    # The SFT pi0.5's quantiles, not the dataset's: a rollout dataset has its own, narrower ones.
    dataset = RLTDemoDataset(
        dataset_path=args.demo_dataset_path,
        repo_id=args.repo_id,
        chunk_length=config.vla_horizon,
        normalize_actions=True,
        normalization_stats=load_policy_normalization_stats(args.model_path),
    )
    num_episodes = _count_episodes(dataset)
    splits = split_episode_indices(
        num_episodes,
        train_ratio=config.offline_rl.train_ratio,
        val_ratio=config.offline_rl.val_ratio,
        seed=config.seed,
    )
    logger.info(
        "[%s] %d episodes -> train=%d val=%d test=%d",
        args.bucket_mode, num_episodes,
        len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    interventions: torch.Tensor | None = None
    if args.bucket_mode == "rl_rollout":
        interventions = _collect_intervention_array(dataset)
        if interventions is None:
            raise RuntimeError("rl_rollout bucket requires complementary_info.is_intervention column")
        logger.info(
            "rl_rollout: loaded %d intervention flags, human_frac=%.2f",
            interventions.numel(), float(interventions.mean()),
        )

    total_start = time.time()
    for split_name, episode_ids in splits.items():
        if not episode_ids:
            continue

        split_start = time.time()
        transitions = []
        n_ref_replaced = 0
        n_total = 0
        for ep_index, episode_id in enumerate(sorted(episode_ids), start=1):
            frame_start, frame_stop = _episode_frame_range(dataset, episode_id)
            frame_indices = build_overlap_frame_indices(
                episode_start=frame_start,
                episode_stop=frame_stop,
                chunk_length=config.chunk_length,
                stride=config.offline_rl.frame_stride,
            )
            loader = DataLoader(
                Subset(dataset, frame_indices),
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=rlt_demo_collate,
                num_workers=args.num_workers,
                drop_last=False,
            )
            episode_success = dataset.get_episode_success(episode_id)
            ep_transitions = build_transitions_from_demos(
                policy=policy,
                demo_loader=loader,
                frame_indices=frame_indices,
                episode_last_frame=frame_stop - 1,
                chunk_length=config.chunk_length,
                device=args.device,
                episode_id=episode_id,
                is_critical=float(args.mark_critical),
                stride=config.offline_rl.frame_stride,
                source=_bucket_source_id(args.bucket_mode),
                episode_success=episode_success,
            )

            start_anchors = transition_start_frames(frame_indices, frame_stop - 1)
            if len(start_anchors) != len(ep_transitions):
                raise RuntimeError(
                    f"Episode {episode_id}: start_anchors={len(start_anchors)} "
                    f"vs transitions={len(ep_transitions)}"
                )

            n_total += len(ep_transitions)
            n_ref_replaced += apply_bucket_refs(
                ep_transitions, start_anchors, args.bucket_mode, interventions, config.chunk_length,
                human_label=args.human_label,
            )
            transitions.extend(ep_transitions)
            if ep_index % 20 == 0:
                logger.info(
                    "[%s/%s] %d/%d ep, %d transitions, %d human chunks",
                    args.bucket_mode, split_name, ep_index, len(episode_ids),
                    len(transitions), n_ref_replaced,
                )

        save_transition_cache(transitions, args.transition_cache_dir, split_name)
        logger.info(
            "[%s/%s] split done: %d transitions, %d human chunks (%.1f%%, human_label=%s), %.1fs",
            args.bucket_mode, split_name, len(transitions), n_ref_replaced,
            100.0 * n_ref_replaced / max(n_total, 1), args.human_label,
            time.time() - split_start,
        )

    logger.info(
        "[%s] total cache built at %s in %.1fs",
        args.bucket_mode, args.transition_cache_dir, time.time() - total_start,
    )


def _bucket_source_id(mode: str) -> int:
    return {"warmup_vla": 1, "human_expert": 3, "rl_rollout": 2}[mode]


if __name__ == "__main__":
    main()
