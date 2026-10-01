"""WeightedMixReplayBuffer for multi-bucket offline RL training.

Holds multiple ReplayBuffer sub-buffers and samples from them according to
fixed per-bucket probabilities. Drops buckets that are empty.
"""
from __future__ import annotations

import random
from typing import Iterable

import torch

from evo_rlt.core.interfaces import (
    ACTUAL_STEPS,
    BC_TARGET_FLAT,
    DONE,
    EPISODE_ID,
    EXEC_CHUNK_FLAT,
    IS_CRITICAL,
    NEXT_REF_FLAT,
    NEXT_STATE_VEC,
    REF_CHUNK_FLAT,
    REWARD_SEQ,
    SOURCE,
    STATE_VEC,
)
from evo_rlt.core.replay_buffer import ReplayBuffer

BATCH_KEYS = (
    STATE_VEC, EXEC_CHUNK_FLAT, REF_CHUNK_FLAT, REWARD_SEQ,
    NEXT_STATE_VEC, NEXT_REF_FLAT, DONE, ACTUAL_STEPS,
    SOURCE, EPISODE_ID, IS_CRITICAL, BC_TARGET_FLAT,
)


class WeightedMixReplayBuffer:
    """Deterministic-ratio mix over named ReplayBuffer sub-buckets.

    sample(batch_size) returns a dict with the same keys as ReplayBuffer.sample,
    concatenated across buckets using sample counts that round ``probs * batch_size``
    while guaranteeing the total equals ``batch_size``.
    """

    def __init__(self, buckets: dict[str, ReplayBuffer], probs: dict[str, float]) -> None:
        if not buckets:
            raise ValueError("buckets must be non-empty")
        missing = [name for name in buckets if name not in probs]
        if missing:
            raise ValueError(f"probs missing buckets: {missing}")
        # Drop empty buckets, renormalize over the rest
        alive = {name: buf for name, buf in buckets.items() if len(buf) > 0}
        if not alive:
            raise ValueError("all buckets are empty")
        p_alive = {name: float(probs[name]) for name in alive}
        s = sum(p_alive.values())
        if s <= 0:
            raise ValueError(f"probs sum to {s} after dropping empty buckets")
        self.buckets: dict[str, ReplayBuffer] = alive
        self.probs: dict[str, float] = {name: p / s for name, p in p_alive.items()}
        self._names: list[str] = list(self.buckets.keys())

    def __len__(self) -> int:
        return sum(len(buf) for buf in self.buckets.values())

    @property
    def capacity(self) -> int:
        return sum(buf.capacity for buf in self.buckets.values())

    def bucket_sizes(self) -> dict[str, int]:
        return {name: len(buf) for name, buf in self.buckets.items()}

    def _split_batch_size(self, batch_size: int) -> dict[str, int]:
        raw = {name: self.probs[name] * batch_size for name in self._names}
        floors = {name: int(raw[name]) for name in self._names}
        remainder = batch_size - sum(floors.values())
        if remainder > 0:
            # distribute remaining slots to buckets with largest fractional parts
            order = sorted(
                self._names,
                key=lambda name: raw[name] - floors[name],
                reverse=True,
            )
            for name in order[:remainder]:
                floors[name] += 1
        return floors

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        split = self._split_batch_size(batch_size)
        pieces: list[dict[str, torch.Tensor]] = []
        for name, count in split.items():
            if count <= 0:
                continue
            pieces.append(self.buckets[name].sample(count))
        if not pieces:
            raise RuntimeError("WeightedMixReplayBuffer.sample produced no slices")
        out: dict[str, torch.Tensor] = {}
        for key in BATCH_KEYS:
            out[key] = torch.cat([p[key] for p in pieces], dim=0)
        # Shuffle so per-bucket order does not leak into gradients
        perm = torch.randperm(out[STATE_VEC].shape[0])
        return {k: v[perm] for k, v in out.items()}


def shuffle_bucket(buf: ReplayBuffer) -> ReplayBuffer:
    """Return a new ReplayBuffer with the same transitions in random order.

    Used when loading caches that may have been stored with temporal order.
    """
    items = list(buf.buffer)
    random.shuffle(items)
    shuffled = ReplayBuffer(capacity=buf.capacity)
    for t in items:
        shuffled.add(t)
    return shuffled


def load_bucket_cache(cache_dir: str, split: str, capacity: int = 200_000) -> ReplayBuffer:
    """Thin wrapper over evo_rlt.core.offline_dataset.load_transition_cache."""
    from evo_rlt.adapters.lerobot.offline_dataset import load_transition_cache
    return load_transition_cache(cache_dir, split, capacity=capacity)


def build_mix_from_paths(
    bucket_paths: dict[str, str],
    probs: dict[str, float],
    split: str,
    capacity: int = 200_000,
) -> WeightedMixReplayBuffer:
    """Build a WeightedMixReplayBuffer by loading each bucket cache directory."""
    buckets = {
        name: load_bucket_cache(path, split, capacity=capacity)
        for name, path in bucket_paths.items()
    }
    return WeightedMixReplayBuffer(buckets, probs)
