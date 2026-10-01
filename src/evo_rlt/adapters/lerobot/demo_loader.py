from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
from torch.utils.data import DataLoader, Dataset

from evo_rlt.core.interfaces import Observation

logger = logging.getLogger(__name__)

# How frames are fit to the model resolution; recorded with the RL token so stale stats/checkpoints are caught.
IMAGE_RESIZE = "resize_with_pad"


def letterbox_image(img: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Fit a (C, H, W) float [0, 1] frame into image_size without distortion, as pi0.5 SFT and deploy do.

    PI05Policy._preprocess_images resizes with resize_with_pad_torch (aspect ratio kept, black bars):
    a 640x480 frame becomes 224x168 of content between two 28-pixel bars. Stretching it to 224x224
    would feed pi0.5, and so the RL token, images it never saw in SFT or at deploy.
    """
    h, w = image_size
    if img.shape[1:] == (h, w):
        return img
    return resize_with_pad_torch(img.unsqueeze(0), h, w).squeeze(0)


def normalize_quantiles(
    tensor: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize from raw space to [-1, 1] using QUANTILES mode (matching lerobot convention).

    Formula: normalized = (raw - q01) / (q99 - q01) * 2.0 - 1.0
    """
    denom = q99 - q01
    denom = torch.where(denom.abs() < eps, torch.tensor(eps, dtype=denom.dtype), denom)
    return (tensor - q01) / denom * 2.0 - 1.0


def load_policy_normalization_stats(pretrained_path: str | Path | None) -> dict[str, dict[str, torch.Tensor]] | None:
    """Stats of the normalizer step saved with a LeRobot policy, keyed like `meta.stats` (`stats["action"]["q01"]`).

    These are the quantiles pi0.5 was SFT-trained with, and the export copies this processor into rlt_ac, so
    deploy normalizes with them too. A dataset's own `meta.stats` only match them for the SFT dataset: a
    critical-segment rollout covers a small slice of the workspace, and its own quantiles would put states
    and actions in a different [-1, 1] frame (a gripper that barely moves gets a ~1e-4 wide range).

    Returns None, with a warning, when *pretrained_path* is not a local directory with a saved normalizer
    (e.g. a hub id such as lerobot/pi05_base); callers then fall back to each dataset's own stats.
    """
    root = Path(pretrained_path) if pretrained_path is not None else None
    config_path = root / "policy_preprocessor.json" if root is not None else None
    state_files = []
    if config_path is not None and config_path.is_file():
        steps = json.loads(config_path.read_text()).get("steps", [])
        state_files = [
            step["state_file"]
            for step in steps
            if step.get("registry_name") == "normalizer_processor" and step.get("state_file")
        ]
    if not state_files:
        logger.warning(
            "No saved normalizer under %s: normalizing with each dataset's own q01/q99, which matches the "
            "policy only for its SFT dataset.",
            pretrained_path,
        )
        return None

    from safetensors.torch import load_file

    stats: dict[str, dict[str, torch.Tensor]] = {}
    for name, value in load_file(root / state_files[0]).items():
        # Flat "<feature>.<stat>" keys, e.g. "observation.state.q01".
        feature, _, stat = name.rpartition(".")
        stats.setdefault(feature, {})[stat] = value
    logger.info("Normalization stats from the policy normalizer %s", root / state_files[0])
    return stats


class RLTDemoDataset(Dataset):
    """Wraps a LeRobotDataset to yield (images, proprio, expert_actions) for RLT demo adaptation.

    Loads action chunks via delta_timestamps so each sample contains a
    chunk_length-step action trajectory starting from the current frame.
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        repo_id: str = "rlt_demo",
        chunk_length: int = 50,
        camera_keys: list[str] | None = None,
        image_size: tuple[int, int] = (224, 224),
        state_key: str = "observation.state",
        action_key: str = "action",
        normalize_actions: bool = False,
        tolerance_s: float = 0.04,
        episodes: list[int] | None = None,
        normalization_stats: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        """normalization_stats: q01/q99 to normalize with instead of the dataset's own `meta.stats`,
        normally `load_policy_normalization_stats(<SFT pi0.5 dir>)`."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        fps = self._read_fps(dataset_path, repo_id)
        delta_timestamps = {
            action_key: [i / fps for i in range(chunk_length)],
        }
        self._dataset = LeRobotDataset(
            repo_id=repo_id,
            root=dataset_path,
            episodes=episodes,
            revision="main",
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            video_backend="pyav",
        )

        self._camera_keys = camera_keys or self._detect_camera_keys()
        self._image_size = image_size
        self._state_key = state_key
        self._action_key = action_key
        self._chunk_length = chunk_length

        # Action + state normalization: raw degrees -> [-1, 1] via QUANTILES
        self._normalize_actions = normalize_actions
        self._action_q01: torch.Tensor | None = None
        self._action_q99: torch.Tensor | None = None
        self._state_q01: torch.Tensor | None = None
        self._state_q99: torch.Tensor | None = None
        if normalize_actions:
            self._init_quantiles(action_key, state_key, normalization_stats)

        logger.info(
            "RLTDemoDataset: %d samples, cameras=%s, chunk=%d, normalize_actions=%s",
            len(self._dataset), self._camera_keys, chunk_length, self._normalize_actions,
        )

    def _init_quantiles(
        self,
        action_key: str,
        state_key: str,
        normalization_stats: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Load q01/q99 for action and state normalization.

        Given normalization_stats (the policy's normalizer) they are used as is, and a missing entry is
        an error: silently falling back to the dataset's own stats would change the normalized frame.
        """
        from_policy = normalization_stats is not None
        stats = normalization_stats if from_policy else self._dataset.meta.stats
        source = "policy normalizer" if from_policy else "dataset meta/stats.json"
        for key, attr_prefix in [(action_key, "_action"), (state_key, "_state")]:
            key_stats = stats.get(key, {})
            q01_raw = key_stats.get("q01")
            q99_raw = key_stats.get("q99")
            if q01_raw is not None and q99_raw is not None:
                q01 = torch.as_tensor(q01_raw, dtype=torch.float32)
                q99 = torch.as_tensor(q99_raw, dtype=torch.float32)
                feature_dim = self._dataset.meta.features[key]["shape"][-1]
                if q01.shape[-1] != feature_dim or q99.shape[-1] != feature_dim:
                    raise ValueError(
                        f"{source} has {q01.shape[-1]}-dim q01/q99 for {key!r}, but the dataset's {key!r} "
                        f"is {feature_dim}-dim"
                    )
                setattr(self, f"{attr_prefix}_q01", q01)
                setattr(self, f"{attr_prefix}_q99", q99)
                logger.info(
                    "%s normalization (%s): q01[:4]=%s, q99[:4]=%s",
                    key, source, q01[:4].tolist(), q99[:4].tolist(),
                )
            elif from_policy:
                raise ValueError(f"policy normalizer has no q01/q99 for {key!r} (has: {sorted(stats)})")
            else:
                logger.warning("Dataset lacks q01/q99 stats for %s; skipping normalization", key)
                if attr_prefix == "_action":
                    self._normalize_actions = False

    def _read_fps(self, dataset_path: str | None, repo_id: str) -> float:
        """Read fps from the dataset metadata."""
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_path, revision="main")
        return meta.fps

    def _detect_camera_keys(self) -> list[str]:
        """Auto-detect camera keys from dataset features."""
        return [k for k in self._dataset.features if k.startswith("observation.images.")]

    def __len__(self) -> int:
        return len(self._dataset)

    def get_episode_success(self, episode_idx: int) -> bool:
        """Return per-episode success flag stored at recording time.

        The column ``episode_success`` is written by ``LeRobotDataset.save_episode
        (extra_episode_metadata=...)`` in ``lerobot_record.py`` around the episode-
        end hook. Canonical labels are ``"success"`` / ``"failure"`` (strings); bool
        and 0/1 numeric are also accepted. Strict on missing column: any dataset
        that lacks ``episode_success`` must be relabeled before use — see
        docs/rlt/rlt_pipeline_review_20260415_1839.md S2-2.
        """
        raw = self._dataset.meta.episodes["episode_success"][episode_idx]
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized == "success":
                return True
            if normalized == "failure":
                return False
        elif isinstance(raw, bool):
            return raw
        elif isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        raise ValueError(
            f"Unrecognized episode_success value for episode {episode_idx}: {raw!r}"
        )

    def __getitem__(self, idx: int) -> dict:
        item = self._dataset[idx]
        result = {}

        # Images: convert to float [0, 1] tensors of consistent size
        for cam_key in self._camera_keys:
            img = item[cam_key]
            if not isinstance(img, torch.Tensor):
                img = torch.as_tensor(img, dtype=torch.float32)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            # Ensure (C, H, W) format
            if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
            img = letterbox_image(img, self._image_size)
            # Use short camera name (e.g. "left_wrist" not "observation.images.left_wrist")
            short_name = cam_key.split(".")[-1]
            result[short_name] = img

        # Proprio state
        state = item[self._state_key]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        state = state.float()
        if self._normalize_actions and self._state_q01 is not None:
            state = normalize_quantiles(state, self._state_q01, self._state_q99)
        result["proprio"] = state

        # Action chunk: (chunk_length, action_dim)
        action = item[self._action_key]
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        action = action.float()
        if self._normalize_actions and self._action_q01 is not None:
            # No clamp: q01/q99 are per-episode averages, so real demo actions exceed [-1, 1]
            # (the SFT preprocessor does not clamp them either).
            action = normalize_quantiles(action, self._action_q01, self._action_q99)
        result["expert_actions"] = action

        return result


def rlt_demo_collate(batch: list[dict]) -> tuple[Observation, torch.Tensor]:
    """Collate a list of dataset items into (Observation, expert_actions).

    Returns:
        obs: Observation with batched images and proprio
        expert_actions: (B, chunk_length, action_dim)
    """
    # Gather all image keys from first item
    image_keys = [k for k in batch[0] if k not in ("proprio", "expert_actions")]

    images = {}
    for key in image_keys:
        images[key] = torch.stack([item[key] for item in batch])

    proprio = torch.stack([item["proprio"] for item in batch])
    expert_actions = torch.stack([item["expert_actions"] for item in batch])

    obs = Observation(images=images, proprio=proprio)
    return obs, expert_actions


def make_demo_loader(
    dataset_path: str | None = None,
    batch_size: int = 32,
    chunk_length: int = 50,
    repo_id: str = "rlt_demo",
    camera_keys: list[str] | None = None,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 2,
    device: str = "cuda",
    tolerance_s: float = 0.04,
    normalize_actions: bool = False,
    episodes: list[int] | None = None,
    normalization_stats: Mapping[str, Mapping[str, Any]] | None = None,
) -> Iterator[tuple[Observation, torch.Tensor]]:
    """Create an infinite-cycling DataLoader for demo adaptation.

    Yields (Observation, expert_actions) tuples moved to device. `episodes`
    restricts the dataset to those episode indices (None = all).
    """
    dataset = RLTDemoDataset(
        dataset_path=dataset_path,
        repo_id=repo_id,
        chunk_length=chunk_length,
        camera_keys=camera_keys,
        image_size=image_size,
        normalize_actions=normalize_actions,
        tolerance_s=tolerance_s,
        episodes=episodes,
        normalization_stats=normalization_stats,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rlt_demo_collate,
        drop_last=True,
        pin_memory=(device != "cpu"),
    )

    def _cycle_with_cleanup() -> Iterator[tuple[Observation, torch.Tensor]]:
        """Restart the DataLoader each epoch to avoid memory accumulation."""
        import gc
        while True:
            for obs, expert_actions in loader:
                obs_device = Observation(
                    images={k: v.to(device) for k, v in obs.images.items()},
                    proprio=obs.proprio.to(device),
                )
                yield obs_device, expert_actions.to(device)
            gc.collect()

    return _cycle_with_cleanup()
