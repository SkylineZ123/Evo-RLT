#!/usr/bin/env python
"""Does the RL token carry the task state? Offline evaluation of a trained RL token checkpoint.

Stage 1 (pi0.5 forward, GPU): every --frame-stride-th frame of a LeRobot dataset is encoded three ways and
saved with its labels to <output-dir>/features.pt:
  z         the trained RL token -- what the actor/critic see, next to proprio
  z_random  a randomly initialised encoder of the same architecture -- the control for what training added
  pooled    the prefix tokens mean-pooled -- a training-free 2048-d summary of the same tokens
The pass also accumulates the decoder's reconstruction R^2 per camera and re-encodes --perturb-frames frames
under photometric / shift / noise changes and with one camera blacked out.

Stage 2 (CPU): <output-dir>/report.json + report.md
  A. reconstruction  share of the frame-to-frame prefix-token variation the decoder rebuilds from z_rl
  B. geometry        effective rank, pairwise cosine, variance next to proprio (the actor does not normalize)
  C. probes          episode-grouped ridge + kNN for proprio / progress / future motion / MC return /
                     success / intervention; gain of [z, proprio] over proprio and over the two controls
  D. temporal        distance vs time lag inside an episode; task phase of cross-episode nearest neighbours
  E. robustness      nuisance-to-signal ratios; how far z moves when one camera goes black

    evo-rlt-eval-rl-token --rl-token-checkpoint output/checkpoint/rl_token/<run>/demo_adapt_checkpoint.pt
    evo-rlt-eval-rl-token --rl-token-checkpoint ... --demo-dataset-path <rollout dataset> --output-dir ...
    evo-rlt-eval-rl-token --features <output-dir>/features.pt      # re-run stage 2 only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evo_rlt.cli.common import configure_logging

logger = configure_logging(__name__)

FEATURES_FILE = "features.pt"
FEATURE_SETS = ("z", "z_random", "pooled")
# Probe inputs for targets that proprio does not already contain; "a+b" concatenates.
PROBE_SETS = ("proprio", "z", "z_random", "pooled", "z+proprio", "z_random+proprio", "pooled+proprio")
VISUAL_PROBE_SETS = ("z", "z_random", "pooled")
KNN_SETS = ("proprio", "z", "z_random", "pooled")
NUISANCE_KINDS = ("photometric", "shift", "noise")
# Ridge strength relative to the number of training frames (features are standardized).
RIDGE_GRID = tuple(float(r) for r in np.logspace(-4, 2, 9))
TEMPORAL_LAGS_S = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
# The within-episode change over this lag is the "signal" nuisance perturbations are compared to.
SIGNAL_LAG_S = 0.5
PHASE_BINS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how much task state a trained RL token carries: reconstruction, geometry, "
            "episode-grouped probes, temporal structure and robustness."
        )
    )
    parser.add_argument("--rl-token-checkpoint", default=None, help="demo_adapt_checkpoint.pt of evo-rlt-train-rl-token.")
    parser.add_argument("--features", default=None, help="features.pt of an earlier run: skip stage 1, only re-analyse.")
    parser.add_argument(
        "--demo-dataset-path",
        default=None,
        help="LeRobot dataset to evaluate on; default: the RL token's training dataset (held-out episodes reported apart).",
    )
    parser.add_argument("--model-path", default=None, help="SFT pi0.5; default: the one the RL token was trained on.")
    parser.add_argument("--output-dir", default=None, help="Default: <checkpoint dir>/eval_<dataset name>.")
    parser.add_argument("--frame-stride", type=int, default=3, help="Evaluate every N-th frame of each episode.")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Cap on evaluated frames, spread over the dataset (0 = all); for smoke tests."
    )
    parser.add_argument(
        "--motion-horizon", type=int, default=10, help="Future action steps in the future-motion target (actor chunk_length)."
    )
    parser.add_argument("--perturb-frames", type=int, default=128, help="Frames re-encoded under perturbations (0 = skip).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="pi0.5 dtype.")
    parser.add_argument("--tokenizer-path", default=None, help="PaliGemma tokenizer repo id or local snapshot path.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Per-step discount of the MC-return target.")
    parser.add_argument("--folds", type=int, default=5, help="Episode-grouped cross-validation folds of the probes.")
    parser.add_argument("--knn", type=int, default=10, help="Neighbours of the kNN probe.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


# ----------------------------------------------------------------------------------------------------
# Stage 1: features
# ----------------------------------------------------------------------------------------------------


class ReconAccumulator:
    """Per-position decoder error and token variance of the prefix tokens, per per-dim weighting and frame group.

    R^2 = 1 - SSE / SST with SST around the per-position mean of the evaluated frames: the share of the
    frame-to-frame variation of the prefix tokens that the decoder rebuilds from z_rl. The part that is the
    same in every frame (static background, letterbox bars) is in neither, so it cannot inflate the score.
    """

    def __init__(self, weights: dict[str, torch.Tensor | None]):
        self.weights = {name: None if w is None else w.double().cpu() for name, w in weights.items()}
        self.stats: dict[tuple[str, str], dict] = {}

    def add(self, groups: list[str], tokens: torch.Tensor, pred: torch.Tensor) -> dict[str, torch.Tensor]:
        """Accumulate one batch; returns each weighting's per-frame, per-position squared error (B, P)."""
        tokens = tokens.detach().double().cpu()
        diff = pred.detach().double().cpu() - tokens
        errors = {}
        for space, weight in self.weights.items():
            x = tokens if weight is None else tokens * weight
            err = (diff if weight is None else diff * weight).pow(2).sum(-1)
            errors[space] = err
            for group in set(groups):
                mask = torch.tensor([g == group for g in groups])
                s = self.stats.setdefault(
                    (group, space),
                    {
                        "n": 0,
                        "sum": torch.zeros(x.shape[1:], dtype=torch.float64),
                        "sumsq": torch.zeros(x.shape[1], dtype=torch.float64),
                        "sse": torch.zeros(x.shape[1], dtype=torch.float64),
                    },
                )
                s["n"] += int(mask.sum())
                s["sum"] += x[mask].sum(0)
                s["sumsq"] += x[mask].pow(2).sum((0, 2))
                s["sse"] += err[mask].sum(0)
        return errors

    def _groups(self, space: str) -> dict[str, dict]:
        parts = {g: s for (g, sp), s in sorted(self.stats.items()) if sp == space}
        if len(parts) > 1:
            parts["all"] = {
                key: sum(s[key] for s in parts.values()) for key in ("n", "sum", "sumsq", "sse")
            }
        return parts

    @staticmethod
    def _sst(stats: dict) -> torch.Tensor:
        return stats["sumsq"] - stats["sum"].pow(2).sum(-1) / max(stats["n"], 1)

    def r2(self, camera_slices: dict[str, slice]) -> dict:
        out: dict = {}
        for space in self.weights:
            for group, s in self._groups(space).items():
                sst = self._sst(s)
                out.setdefault(space, {})[group] = {
                    "frames": s["n"],
                    **{cam: float(1 - s["sse"][sl].sum() / sst[sl].sum().clamp_min(1e-12)) for cam, sl in camera_slices.items()},
                }
        return out

    def mean_sst(self, space: str, camera_slices: dict[str, slice]) -> dict[str, float]:
        """Average squared distance of a frame's tokens from the per-position mean, per camera (all frames)."""
        groups = self._groups(space)
        s = groups.get("all") or next(iter(groups.values()))
        sst = self._sst(s)
        return {cam: float(sst[sl].sum() / max(s["n"], 1)) for cam, sl in camera_slices.items()}


def perturb_images(images: dict[str, torch.Tensor], kind: str, generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Apply one perturbation to (B, 3, H, W) [0, 1] frames.

    photometric / shift / noise are nuisances a state representation should barely notice;
    `blackout:<camera>` removes that camera, which should move it by roughly that camera's share.
    """
    if kind.startswith("blackout:"):
        camera = kind.split(":", 1)[1]
        return {name: torch.zeros_like(img) if name == camera else img for name, img in images.items()}
    out = {}
    for name, img in images.items():
        if kind == "photometric":
            # +10% brightness, -10% contrast
            mean = img.mean(dim=(1, 2, 3), keepdim=True)
            img = (img - mean) * 0.9 + mean * 1.1
        elif kind == "shift":
            # 4 px (~2% of 224) to the right, the uncovered strip black
            img = torch.roll(img, shifts=4, dims=-1)
            img[..., :4] = 0.0
        elif kind == "noise":
            img = img + 0.02 * torch.randn(img.shape, generator=generator).to(img)
        else:
            raise ValueError(f"unknown perturbation {kind!r}")
        out[name] = img.clamp(0.0, 1.0)
    return out


def _load_rl_tokens(checkpoint_path: str, device: str, seed: int):
    """(trained full module, randomly initialised encoder, step, metadata) of an RL token checkpoint."""
    from evo_rlt.core.rl_token import RLTokenModule, rl_token_arch_from_state_dict

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata") or {}
    for key in ("rl_token", "vla_model", "dataset", "camera_name_map", "task_instruction"):
        if key not in metadata:
            raise SystemExit(f"{checkpoint_path}: metadata has no {key!r}; re-save it with the current evo-rlt-train-rl-token")
    state_dict = checkpoint["rl_token_state_dict"]
    cfg = {**metadata["rl_token"], **rl_token_arch_from_state_dict(state_dict)}

    def build(inference_only: bool) -> RLTokenModule:
        return RLTokenModule(
            token_dim=cfg["token_dim"],
            nhead=cfg["nhead"],
            num_enc_layers=cfg["enc_layers"],
            num_dec_layers=cfg["dec_layers"],
            ff_dim=cfg["ff_dim"],
            num_rl_tokens=cfg["num_rl_tokens"],
            inference_only=inference_only,
            arch=cfg["arch"],
            seq_len=cfg["seq_len"],
        )

    trained = build(inference_only=False)
    trained.load_state_dict(state_dict)
    torch.manual_seed(seed)
    random_init = build(inference_only=True)
    return trained.to(device).eval(), random_init.to(device).eval(), checkpoint.get("step"), metadata


def _load_token_std(checkpoint_path: str, metadata: dict) -> torch.Tensor | None:
    """Per-dim std of the prefix tokens the RL token was trained with (for the weighted R^2)."""
    norm_stats = metadata.get("norm_stats")
    path = Path(norm_stats) if norm_stats and norm_stats != "auto" else Path(checkpoint_path).parent / "token_stats.pt"
    if not path.is_file():
        logger.warning("No token stats at %s: reconstruction R^2 only in the raw token space", path)
        return None
    return torch.load(path, map_location="cpu", weights_only=False)["std"]


def _camera_slices(vla) -> dict[str, slice]:
    """Prefix-token positions of each camera (image tokens first, in pi0.5 slot order), plus "all"."""
    total = vla.num_prefix_tokens
    if vla.token_pool_size > 0:
        return {"all": slice(0, total)}
    slot_to_camera = {slot: camera for camera, slot in vla.camera_name_map.items()}
    indices = vla._active_camera_indices or list(range(len(vla.camera_order)))
    n = vla._num_per_camera
    slices = {
        slot_to_camera.get(vla.camera_order[i], vla.camera_order[i]): slice(k * n, (k + 1) * n)
        for k, i in enumerate(indices)
    }
    if total > len(indices) * n:
        slices["language"] = slice(len(indices) * n, total)
    return {**slices, "all": slice(0, total)}


def _denormalize(x: torch.Tensor, q01: torch.Tensor | None, q99: torch.Tensor | None) -> torch.Tensor:
    return x if q01 is None else (x + 1.0) / 2.0 * (q99 - q01) + q01


def _to_device(obs, device: str):
    from evo_rlt.core.interfaces import Observation

    return Observation(images={k: v.to(device) for k, v in obs.images.items()}, proprio=obs.proprio.to(device))


@torch.no_grad()
def extract_features(args: argparse.Namespace, output_dir: Path) -> dict:
    from torch.utils.data import DataLoader, Subset

    from evo_rlt.adapters.lerobot.demo_loader import RLTDemoDataset, load_policy_normalization_stats, rlt_demo_collate
    from evo_rlt.adapters.lerobot.offline_dataset import _count_episodes, _episode_frame_range
    from evo_rlt.adapters.lerobot.pi05_adapter import Pi05VLAAdapter
    from evo_rlt.cli.build_bucket_cache import _collect_intervention_array
    from evo_rlt.cli.train_rl_token import DEMO_REPO_ID, _check_cameras, _load_dataset_meta
    from evo_rlt.core.interfaces import Observation

    rl_token, rl_random, step, metadata = _load_rl_tokens(args.rl_token_checkpoint, args.device, args.seed)
    dataset_path = args.demo_dataset_path or metadata["dataset"]
    model_path = args.model_path or metadata["vla_model"]
    camera_name_map = metadata["camera_name_map"]
    meta = _load_dataset_meta(dataset_path)
    _check_cameras(meta, camera_name_map)
    state_dim = meta.features["observation.state"]["shape"][-1]
    action_dim = meta.features["action"]["shape"][-1]

    logger.info("Loading pi0.5 from %s (RL token step %s, dataset %s)", model_path, step, dataset_path)
    vla = Pi05VLAAdapter(
        model_path=model_path,
        actual_action_dim=action_dim,
        actual_proprio_dim=state_dim,
        camera_name_map=camera_name_map,
        task_instruction=metadata["task_instruction"],
        dtype=args.dtype,
        device=args.device,
        token_pool_size=metadata.get("token_pool_size", 0),
        image_only=metadata.get("image_only", False),
        active_cameras=metadata.get("active_cameras"),
        tokenizer_path=args.tokenizer_path,
    )
    vla.eval()
    camera_slices = _camera_slices(vla)
    cameras = [name for name in camera_slices if name not in ("all", "language")]

    normalize = metadata.get("normalize", True)
    dataset = RLTDemoDataset(
        dataset_path=dataset_path,
        repo_id=DEMO_REPO_ID,
        chunk_length=args.motion_horizon,
        normalize_actions=normalize,
        normalization_stats=load_policy_normalization_stats(model_path) if normalize else None,
    )

    # Frames: (global index, episode, frame in episode, episode length).
    rows = []
    for episode in range(_count_episodes(dataset)):
        start, stop = _episode_frame_range(dataset, episode)
        rows.extend((index, episode, index - start, stop - start) for index in range(start, stop, args.frame_stride))
    if args.max_frames and len(rows) > args.max_frames:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, args.max_frames).round().astype(int)]

    same_dataset = Path(dataset_path).resolve() == Path(metadata["dataset"]).resolve()
    val_episodes = set(metadata.get("val_episodes") or [])
    if same_dataset:
        group_of = {ep: "rl_heldout" if ep in val_episodes else "rl_train" for _, ep, _, _ in rows}
    else:
        group_of = {ep: "external" for _, ep, _, _ in rows}

    success_of = {}
    for _, episode, _, _ in rows:
        if episode not in success_of:
            try:
                success_of[episode] = float(dataset.get_episode_success(episode))
            except (KeyError, ValueError):
                success_of[episode] = float("nan")
    interventions = _collect_intervention_array(dataset)

    std = _load_token_std(args.rl_token_checkpoint, metadata)
    norm_gamma = float(metadata.get("norm_gamma") or 0.0)
    weights: dict[str, torch.Tensor | None] = {"raw": None}
    if std is not None:
        weights["train_metric"] = std.clamp_min(1e-6).pow(-norm_gamma)
        weights["whitened"] = std.clamp_min(1e-6).pow(-1.0)
    error_space = "whitened" if std is not None else "raw"
    recon = ReconAccumulator(weights)

    def loader_for(indices: list[int]) -> DataLoader:
        return DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=rlt_demo_collate,
            num_workers=args.num_workers,
        )

    feats: dict[str, list[torch.Tensor]] = {name: [] for name in FEATURE_SETS}
    proprio, actions, frame_errors = [], [], []
    logger.info("Stage 1: %d frames from %d episodes (stride %d)", len(rows), len(success_of), args.frame_stride)
    start_time = time.time()
    offset = 0
    for batch_index, (obs, expert_actions) in enumerate(loader_for([row[0] for row in rows])):
        batch = len(obs.proprio)
        groups = [group_of[rows[offset + i][1]] for i in range(batch)]
        offset += batch
        tokens = vla.prefix_tokens(_to_device(obs, args.device))
        z_multi = rl_token.encode_multi(tokens)
        feats["z"].append(z_multi.mean(dim=1).float().cpu())
        feats["z_random"].append(rl_random.encode(tokens).float().cpu())
        feats["pooled"].append(tokens.mean(dim=1).float().cpu())
        errors = recon.add(groups, tokens, rl_token._reconstruct(z_multi, tokens))[error_space]
        frame_errors.append(torch.stack([errors[:, camera_slices[c]].sum(-1) for c in camera_slices], dim=-1))
        proprio.append(obs.proprio.float().cpu())
        actions.append(expert_actions.float().cpu())
        if (batch_index + 1) % 50 == 0:
            logger.info("  %d/%d frames, %.1fs", offset, len(rows), time.time() - start_time)

    # Per-frame error relative to the average frame's variation, per camera: < 1 beats the per-position mean.
    mean_sst = recon.mean_sst(error_space, camera_slices)
    frame_errors = torch.cat(frame_errors).double() / torch.tensor([mean_sst[c] for c in camera_slices]).clamp_min(1e-12)

    proprio = torch.cat(proprio)
    actions = torch.cat(actions)
    features = {
        "meta": {
            "rl_token_checkpoint": str(args.rl_token_checkpoint),
            "rl_token_step": step,
            "rl_token_dataset": metadata["dataset"],
            "rl_token_val_episodes": sorted(val_episodes),
            "dataset": str(dataset_path),
            "model_path": str(model_path),
            "fps": float(meta.fps),
            "frame_stride": args.frame_stride,
            "motion_horizon": args.motion_horizon,
            "cameras": cameras,
            "norm_gamma": norm_gamma,
            "error_space": error_space,
        },
        "episode": torch.tensor([row[1] for row in rows]),
        "frame": torch.tensor([row[2] for row in rows]),
        "length": torch.tensor([row[3] for row in rows]),
        "group": [group_of[row[1]] for row in rows],
        "success": torch.tensor([success_of[row[1]] for row in rows]),
        "intervention": None if interventions is None else interventions[[row[0] for row in rows]],
        "proprio": proprio,
        "proprio_raw": _denormalize(proprio, dataset._state_q01, dataset._state_q99),
        "actions_raw": _denormalize(actions, dataset._action_q01, dataset._action_q99),
        "features": {name: torch.cat(values) for name, values in feats.items()},
        "recon": recon.r2(camera_slices),
        "recon_error": frame_errors.float(),
        "recon_error_cameras": list(camera_slices),
        "perturb": None,
    }

    if args.perturb_frames > 0:
        picks = np.linspace(0, len(rows) - 1, min(args.perturb_frames, len(rows))).round().astype(int).tolist()
        kinds = ["none", *NUISANCE_KINDS, *(f"blackout:{camera}" for camera in cameras)]
        perturb = {"row": torch.tensor(picks), "z": {k: [] for k in kinds}, "pooled": {k: [] for k in kinds}}
        generator = torch.Generator().manual_seed(args.seed)
        logger.info("Stage 1: re-encoding %d frames under %s", len(picks), kinds[1:])
        for obs, _ in loader_for([rows[i][0] for i in picks]):
            obs = _to_device(obs, args.device)
            for kind in kinds:
                images = obs.images if kind == "none" else perturb_images(obs.images, kind, generator)
                tokens = vla.prefix_tokens(Observation(images=images, proprio=obs.proprio))
                perturb["z"][kind].append(rl_token.encode(tokens).float().cpu())
                perturb["pooled"][kind].append(tokens.mean(dim=1).float().cpu())
        for name in ("z", "pooled"):
            perturb[name] = {k: torch.cat(v) for k, v in perturb[name].items()}
        features["perturb"] = perturb

    path = output_dir / FEATURES_FILE
    torch.save(features, path)
    logger.info("Stage 1 done in %.1fs: %s", time.time() - start_time, path)
    return features


# ----------------------------------------------------------------------------------------------------
# Stage 2: analysis (pure tensor code, CPU)
# ----------------------------------------------------------------------------------------------------


def r2_per_column(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - SSE/SST per column; NaN for columns that do not vary."""
    sst = (target - target.mean(0)).pow(2).sum(0)
    sse = (pred - target).pow(2).sum(0)
    r2 = 1 - sse / sst.clamp_min(1e-12)
    return torch.where(sst > 1e-10 * max(float(sst.max()), 1e-30), r2, torch.full_like(r2, float("nan")))


def mean_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    r2 = r2_per_column(pred, target)
    return float(r2[~r2.isnan()].mean()) if (~r2.isnan()).any() else float("nan")


def auc_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """P(score of a positive > score of a negative), ties counting half."""
    pos, neg = scores[labels > 0.5], scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for chunk in pos.split(2048):
        diff = chunk[:, None] - neg[None, :]
        wins += float((diff > 0).sum()) + 0.5 * float((diff == 0).sum())
    return wins / (len(pos) * len(neg))


def episode_folds(episode: torch.Tensor, folds: int, seed: int) -> torch.Tensor:
    """Fold index per frame; every episode lies in exactly one fold."""
    episodes = episode.unique()
    order = episodes[torch.randperm(len(episodes), generator=torch.Generator().manual_seed(seed))]
    k = min(folds, len(episodes))
    fold_of = {int(ep): i % k for i, ep in enumerate(order)}
    return torch.tensor([fold_of[int(ep)] for ep in episode])


def _standardize(train: torch.Tensor, *others: torch.Tensor) -> list[torch.Tensor]:
    mean, std = train.mean(0), train.std(0).nan_to_num(0.0).clamp_min(1e-8)
    return [(x - mean) / std for x in (train, *others)]


def ridge_path(x_train: torch.Tensor, y_train: torch.Tensor, x_eval: torch.Tensor, ridges=RIDGE_GRID) -> torch.Tensor:
    """Ridge predictions (len(ridges), m, targets) on standardized features, one eigendecomposition for all."""
    a, b = _standardize(x_train, x_eval)
    y_mean = y_train.mean(0)
    evals, evecs = torch.linalg.eigh(a.T @ a)
    proj = evecs.T @ (a.T @ (y_train - y_mean))
    bv = b @ evecs
    n = len(a)
    return torch.stack([bv @ (proj / (evals.clamp_min(0) + r * n)[:, None]) for r in ridges]) + y_mean


def ridge_probe(
    x: torch.Tensor, y: torch.Tensor, fold: torch.Tensor, episode: torch.Tensor, blocks: dict[str, slice], seed: int
) -> torch.Tensor:
    """Out-of-fold ridge predictions; each target block picks its ridge on an inner split of the training episodes."""
    oof = torch.empty_like(y)
    for k in fold.unique():
        test = fold == k
        train = ~test
        train_episodes = episode[train].unique()
        chosen = {name: len(RIDGE_GRID) // 2 for name in blocks}
        if len(train_episodes) >= 2:
            perm = train_episodes[torch.randperm(len(train_episodes), generator=torch.Generator().manual_seed(seed + int(k)))]
            inner_val_episodes = perm[: max(1, len(perm) // 4)]
            inner_val = torch.isin(episode, inner_val_episodes) & train
            inner_train = train & ~inner_val
            path = ridge_path(x[inner_train], y[inner_train], x[inner_val])
            for name, sl in blocks.items():
                scores = [mean_r2(p[:, sl], y[inner_val][:, sl]) for p in path]
                scores = [-math.inf if math.isnan(s) else s for s in scores]
                chosen[name] = int(np.argmax(scores))
        path = ridge_path(x[train], y[train], x[test])
        pred = torch.empty((int(test.sum()), y.shape[1]), dtype=y.dtype)
        for name, sl in blocks.items():
            pred[:, sl] = path[chosen[name]][:, sl]
        oof[test] = pred
    return oof


def knn_probe(x: torch.Tensor, y: torch.Tensor, fold: torch.Tensor, k: int) -> torch.Tensor:
    """Out-of-fold mean target of the k cosine-nearest training frames (standardized features)."""
    oof = torch.empty_like(y)
    for f in fold.unique():
        test = fold == f
        a, b = _standardize(x[~test], x[test])
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        idx = torch.cat([(chunk @ a.T).topk(min(k, len(a)), dim=-1).indices for chunk in b.split(1024)])
        oof[test] = y[~test][idx].mean(dim=1)
    return oof


def geometry(x: torch.Tensor, max_pairs_samples: int = 2000, seed: int = 0) -> dict:
    """Is the representation spread over many directions, or a constant plus a few?"""
    n, d = x.shape
    xc = x - x.mean(0)
    gram = xc.T @ xc if n > d else xc @ xc.T
    eig = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0) / max(n - 1, 1)
    share = eig.cumsum(0) / eig.sum().clamp_min(1e-30)
    std = xc.pow(2).mean(0).sqrt()
    sub = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:max_pairs_samples]

    def mean_cos(v: torch.Tensor) -> float:
        v = torch.nn.functional.normalize(v[sub], dim=-1)
        cos = v @ v.T
        m = len(v)
        return float((cos.sum() - cos.diagonal().sum()) / max(m * (m - 1), 1))

    return {
        "dim": d,
        "effective_rank": float(eig.sum().pow(2) / eig.pow(2).sum().clamp_min(1e-30)),
        "pcs_90": int((share < 0.9).sum()) + 1,
        "pcs_99": int((share < 0.99).sum()) + 1,
        "top1_share": float(share[0]),
        "cos_raw": mean_cos(x),
        "cos_centered": mean_cos(xc),
        "total_var": float(xc.pow(2).sum(1).mean()),
        "mean_norm_over_spread": float(x.mean(0).norm() / xc.pow(2).sum(1).mean().sqrt().clamp_min(1e-30)),
        "dead_dim_frac": float((std < 1e-3 * std.median().clamp_min(1e-30)).double().mean()),
    }


def _sorted_steps(episode: torch.Tensor, frame: torch.Tensor) -> tuple[torch.Tensor, float]:
    order = torch.from_numpy(np.lexsort((frame.numpy(), episode.numpy())))
    ep, fr = episode[order], frame[order]
    same = ep[1:] == ep[:-1]
    step = float((fr[1:] - fr[:-1])[same].double().median()) if same.any() else 1.0
    return order, step


def lag_pairs(episode: torch.Tensor, frame: torch.Tensor, lag_frames: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Index pairs (i, j) in the same episode about lag_frames apart; returns the lag actually used (frames)."""
    order, step = _sorted_steps(episode, frame)
    shift = max(1, round(lag_frames / step))
    ep = episode[order]
    if len(order) <= shift:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), shift * step
    i, j = order[:-shift], order[shift:]
    keep = ep[:-shift] == ep[shift:]
    return i[keep], j[keep], shift * step


def random_pair_distance(x: torch.Tensor, seed: int, pairs: int = 20000) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    i = torch.randint(len(x), (pairs,), generator=g)
    j = torch.randint(len(x), (pairs,), generator=g)
    keep = i != j
    return (x[i[keep]] - x[j[keep]]).norm(dim=-1)


def temporal_curve(x: torch.Tensor, episode: torch.Tensor, frame: torch.Tensor, fps: float, seed: int) -> dict:
    """Mean distance at each time lag inside an episode / mean distance of random frame pairs."""
    reference = float(random_pair_distance(x, seed).mean())
    curve = {}
    for lag_s in TEMPORAL_LAGS_S:
        i, j, used = lag_pairs(episode, frame, lag_s * fps)
        key = f"{used / fps:.2f}s"
        if len(i) and key not in curve:
            curve[key] = float((x[i] - x[j]).norm(dim=-1).mean() / max(reference, 1e-30))
    return curve


def phase_alignment(x: torch.Tensor, episode: torch.Tensor, progress: torch.Tensor, seed: int, queries: int = 2000) -> float:
    """Mean |progress difference| between a frame and its nearest neighbour from another episode."""
    q = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:queries]
    xc = x - x.mean(0)
    dist = torch.cdist(xc[q], xc)
    dist[episode[q][:, None] == episode[None, :]] = math.inf
    valid = torch.isfinite(dist).any(dim=1)
    if not valid.any():
        return float("nan")
    nn = dist[valid].argmin(dim=1)
    return float((progress[q][valid] - progress[nn]).abs().mean())


def _targets(features: dict, gamma: float) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Probe targets (N, k) and their kind ("reg" or "cls")."""
    episode = features["episode"]
    frame, length = features["frame"].double(), features["length"].double()
    progress = frame / (length - 1).clamp_min(1)
    success = features["success"].double()
    known = ~success.isnan()
    actions = features["actions_raw"].double()
    proprio_raw = features["proprio_raw"].double()
    if actions.shape[-1] == proprio_raw.shape[-1]:
        motion = actions - proprio_raw[:, None, :]
    else:
        motion = actions
    targets = {
        "proprio": features["proprio"].double(),
        "progress": progress[:, None],
        "future_motion": motion.flatten(1),
        # Success-weighted discounted return of a terminal reward; unlabeled episodes count as successes.
        "mc_return": (torch.where(known, success, torch.ones_like(success)) * gamma ** (length - 1 - frame))[:, None],
    }
    kinds = {name: "reg" for name in targets}

    def two_classes(labels: torch.Tensor) -> bool:
        return all(len(episode[labels == c].unique()) >= 2 for c in (0.0, 1.0))

    if known.all() and two_classes(success):
        targets["success"], kinds["success"] = success[:, None], "cls"
    intervention = features.get("intervention")
    if intervention is not None:
        flag = (intervention.double() > 0.5).double()
        if len(episode[flag == 1].unique()) >= 2 and (flag == 0).any():
            targets["intervention"], kinds["intervention"] = flag[:, None], "cls"
    return targets, kinds


def _input_sets(features: dict) -> dict[str, torch.Tensor]:
    base = {name: features["features"][name].double() for name in FEATURE_SETS}
    base["proprio"] = features["proprio"].double()
    return {name: torch.cat([base[part] for part in name.split("+")], dim=-1) for name in PROBE_SETS}


def run_probes(features: dict, folds: int, knn: int, gamma: float, seed: int) -> dict:
    episode = features["episode"]
    targets, kinds = _targets(features, gamma)
    if len(episode.unique()) < 2:
        logger.warning("Probes need at least 2 episodes; skipped")
        return {}
    fold = episode_folds(episode, folds, seed)
    inputs = _input_sets(features)

    names = list(targets)
    widths = [targets[name].shape[1] for name in names]
    bounds = np.cumsum([0, *widths])
    blocks = {name: slice(int(bounds[i]), int(bounds[i + 1])) for i, name in enumerate(names)}
    y = torch.cat([targets[name] for name in names], dim=-1)

    def score(pred: torch.Tensor, name: str) -> float:
        target = targets[name]
        return auc_score(pred[:, 0], target[:, 0]) if kinds[name] == "cls" else mean_r2(pred, target)

    results: dict = {name: {"metric": "auc" if kinds[name] == "cls" else "r2", "ridge": {}, "knn": {}} for name in names}
    for set_name in PROBE_SETS:
        start = time.time()
        pred = ridge_probe(inputs[set_name], y, fold, episode, blocks, seed)
        for name in names:
            if name == "proprio" and set_name not in VISUAL_PROBE_SETS:
                continue
            results[name]["ridge"][set_name] = score(pred[:, blocks[name]], name)
        if set_name in KNN_SETS:
            pred = knn_probe(inputs[set_name], y, fold, knn)
            for name in names:
                if name == "proprio" and set_name == "proprio":
                    continue
                results[name]["knn"][set_name] = score(pred[:, blocks[name]], name)
        logger.info("  probes on %-18s %.1fs", set_name, time.time() - start)
    return results


def robustness(features: dict, seed: int) -> dict:
    """Median perturbation-induced change / median change over SIGNAL_LAG_S inside an episode."""
    perturb = features.get("perturb")
    if not perturb:
        return {}
    episode, frame, fps = features["episode"], features["frame"], features["meta"]["fps"]
    i, j, used = lag_pairs(episode, frame, SIGNAL_LAG_S * fps)
    out = {"signal_lag_s": used / fps}
    for name in ("z", "pooled"):
        x = features["features"][name].double()
        signal = float((x[i] - x[j]).norm(dim=-1).median()) if len(i) else float("nan")
        spread = float(random_pair_distance(x, seed).median())
        base = perturb[name]["none"].double()
        out[name] = {
            kind: {
                "over_signal": float((values.double() - base).norm(dim=-1).median() / max(signal, 1e-30)),
                "over_random_pair": float((values.double() - base).norm(dim=-1).median() / max(spread, 1e-30)),
            }
            for kind, values in perturb[name].items()
            if kind != "none"
        }
    return out


def recon_by_phase(features: dict) -> dict:
    """Mean per-frame reconstruction error (1 = as bad as the per-position mean) per progress bin and group."""
    errors = features.get("recon_error")
    if errors is None:
        return {}
    progress = features["frame"].double() / (features["length"].double() - 1).clamp_min(1)
    bins = (progress * PHASE_BINS).long().clamp_max(PHASE_BINS - 1)
    cameras = features["recon_error_cameras"]
    groups = np.array(features["group"])
    out: dict = {"by_phase": {}, "by_group": {}}
    for b in range(PHASE_BINS):
        mask = bins == b
        if mask.any():
            label = f"{b / PHASE_BINS:.1f}-{(b + 1) / PHASE_BINS:.1f}"
            out["by_phase"][label] = {cam: float(errors[mask, c].mean()) for c, cam in enumerate(cameras)}
    for group in sorted(set(groups)):
        mask = torch.from_numpy(groups == group)
        out["by_group"][group] = {cam: float(errors[mask, c].mean()) for c, cam in enumerate(cameras)}
    return out


def _flags(report: dict) -> list[str]:
    """Heuristic warnings; thresholds are rules of thumb, read the numbers next to them."""
    flags = []
    geo = report["geometry"]
    if geo["z"]["effective_rank"] < 3:
        flags.append(f"z_rl is close to collapsed: effective rank {geo['z']['effective_rank']:.1f} < 3")
    ratio = geo["z"]["total_var"] / max(geo["proprio"]["total_var"], 1e-30)
    if ratio > 100 or ratio < 0.01:
        flags.append(
            f"z_rl variance is {ratio:.3g}x proprio's; the actor/critic MLP takes [z_rl, proprio] unnormalized, "
            "so one of them may drown the other"
        )
    probes = report.get("probes") or {}
    for target in ("future_motion", "mc_return", "progress"):
        ridge = probes.get(target, {}).get("ridge", {})
        if not ridge:
            continue
        gain = ridge["z+proprio"] - ridge["proprio"]
        if gain < 0.02:
            flags.append(f"{target}: [z_rl, proprio] adds only {gain:+.3f} R^2 over proprio alone")
        if ridge["z+proprio"] - ridge["z_random+proprio"] < 0:
            flags.append(f"{target}: the trained encoder is no better than a random one ({ridge['z+proprio']:.3f} vs {ridge['z_random+proprio']:.3f})")
        if ridge["z+proprio"] < ridge["pooled+proprio"] - 0.1:
            flags.append(f"{target}: z_rl loses information that mean-pooled tokens keep ({ridge['z+proprio']:.3f} vs {ridge['pooled+proprio']:.3f})")
    recon = (report.get("reconstruction") or {}).get("whitened") or (report.get("reconstruction") or {}).get("raw") or {}
    if "rl_train" in recon and "rl_heldout" in recon:
        gap = recon["rl_train"]["all"] - recon["rl_heldout"]["all"]
        if gap > 0.15:
            flags.append(f"reconstruction R^2 drops by {gap:.2f} from training to held-out episodes (overfitting)")
    rob = (report.get("robustness") or {}).get("z") or {}
    for kind in NUISANCE_KINDS:
        if kind in rob and rob[kind]["over_signal"] > 0.5:
            flags.append(f"{kind} perturbation moves z_rl {rob[kind]['over_signal']:.2f}x as far as {SIGNAL_LAG_S}s of motion")
    blackout = {k.split(":", 1)[1]: v["over_random_pair"] for k, v in rob.items() if k.startswith("blackout:")}
    if len(blackout) >= 2:
        weakest = min(blackout, key=blackout.get)
        if blackout[weakest] < 0.1 * max(blackout.values()):
            flags.append(f"z_rl barely depends on camera {weakest!r} (blackout moves it {blackout[weakest]:.3f} of a random pair)")
    return flags


def analyze(features: dict, folds: int = 5, knn: int = 10, gamma: float = 0.99, seed: int = 0) -> dict:
    meta = features["meta"]
    episode, frame = features["episode"], features["frame"]
    progress = frame.double() / (features["length"].double() - 1).clamp_min(1)
    sets = {name: features["features"][name].double() for name in FEATURE_SETS}
    sets["proprio"] = features["proprio"].double()

    logger.info("Stage 2: %d frames, %d episodes", len(episode), len(episode.unique()))
    report = {
        "meta": meta,
        "frames": len(episode),
        "episodes": len(episode.unique()),
        "groups": {g: int(sum(1 for x in features["group"] if x == g)) for g in sorted(set(features["group"]))},
        "reconstruction": features.get("recon"),
        "reconstruction_error": recon_by_phase(features),
        "geometry": {name: geometry(x, seed=seed) for name, x in sets.items()},
    }
    report["temporal"] = {name: temporal_curve(x, episode, frame, meta["fps"], seed) for name, x in sets.items()}
    sample = torch.randperm(len(progress), generator=torch.Generator().manual_seed(seed))[:2000]
    chance = float((progress[sample][:, None] - progress[sample][None, :]).abs().mean())
    report["phase_alignment"] = {
        "chance": chance,
        **{name: phase_alignment(x, episode, progress, seed) for name, x in sets.items()},
        "z+proprio": phase_alignment(_input_sets(features)["z+proprio"], episode, progress, seed),
    }
    report["probes"] = run_probes(features, folds, knn, gamma, seed)
    report["robustness"] = robustness(features, seed)
    report["flags"] = _flags(report)
    return report


# ----------------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------------


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in rows]
    return lines + [""]


def render_markdown(report: dict) -> str:
    meta = report["meta"]
    lines = [
        "# RL token evaluation",
        "",
        f"- checkpoint: `{meta['rl_token_checkpoint']}` (step {meta['rl_token_step']})",
        f"- dataset: `{meta['dataset']}`: {report['frames']} frames, {report['episodes']} episodes, groups {report['groups']}",
        "",
        "## Flags (heuristic)",
        "",
        *([f"- {flag}" for flag in report["flags"]] or ["- none"]),
        "",
        "## A. Reconstruction R^2 (share of frame-to-frame token variation rebuilt from z_rl)",
        "",
    ]
    for space, groups in (report.get("reconstruction") or {}).items():
        cameras = [k for k in next(iter(groups.values())) if k != "frames"]
        lines.append(f"space `{space}`")
        lines += _table(["group", "frames", *cameras], [[g, v["frames"], *(v[c] for c in cameras)] for g, v in groups.items()])
    err = report.get("reconstruction_error") or {}
    if err.get("by_phase"):
        cameras = list(next(iter(err["by_phase"].values())))
        lines.append(f"per-frame error / average frame variation (`{meta['error_space']}` space; 1 = per-position mean), by task progress")
        lines += _table(["progress", *cameras], [[k, *(v[c] for c in cameras)] for k, v in err["by_phase"].items()])

    lines += ["## B. Geometry", ""]
    geo = report["geometry"]
    keys = ["dim", "effective_rank", "pcs_90", "pcs_99", "top1_share", "cos_raw", "cos_centered", "total_var", "mean_norm_over_spread", "dead_dim_frac"]
    lines += _table(["features", *keys], [[name, *(g[k] for k in keys)] for name, g in geo.items()])

    lines += ["## C. Probes (episode-grouped CV; R^2, or AUC for success/intervention)", ""]
    for target, res in (report.get("probes") or {}).items():
        sets = list(res["ridge"])
        lines.append(f"**{target}** ({res['metric']})")
        lines += _table(
            ["probe", *sets],
            [["ridge", *(res["ridge"].get(s) for s in sets)], ["knn", *(res["knn"].get(s) for s in sets)]],
        )

    lines += ["## D. Temporal structure", "", "mean distance at a time lag inside an episode / mean distance of random frame pairs", ""]
    temporal = report["temporal"]
    lags = list(next(iter(temporal.values())))
    lines += _table(["features", *lags], [[name, *(curve.get(lag) for lag in lags)] for name, curve in temporal.items()])
    phase = report["phase_alignment"]
    lines.append("mean |progress difference| to the nearest neighbour in another episode (lower = aligned task phase)")
    lines += _table(list(phase), [list(phase.values())])

    rob = report.get("robustness") or {}
    if rob:
        lines += [
            "## E. Robustness",
            "",
            f"median change under a perturbation / median change over {rob['signal_lag_s']:.2f}s of motion "
            "(over_signal), and / median random-pair distance (over_random_pair)",
            "",
        ]
        kinds = list(rob["z"])
        rows = [[f"{name} {metric}", *(rob[name][k][metric] for k in kinds)] for name in ("z", "pooled") for metric in ("over_signal", "over_random_pair")]
        lines += _table(["features", *kinds], rows)
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    if args.features:
        features = torch.load(args.features, map_location="cpu", weights_only=False)
        output_dir = Path(args.output_dir or Path(args.features).parent)
    elif args.rl_token_checkpoint:
        checkpoint = Path(args.rl_token_checkpoint)
        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir is None:
            dataset_name = Path(args.demo_dataset_path).name if args.demo_dataset_path else "train_dataset"
            output_dir = checkpoint.parent / f"eval_{dataset_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        features = extract_features(args, output_dir)
    else:
        raise SystemExit("pass --rl-token-checkpoint (full run) or --features (analysis only)")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = analyze(features, folds=args.folds, knn=args.knn, gamma=args.gamma, seed=args.seed)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    markdown = render_markdown(report)
    (output_dir / "report.md").write_text(markdown)
    print(markdown)
    logger.info("Report: %s", output_dir / "report.md")


if __name__ == "__main__":
    main()
