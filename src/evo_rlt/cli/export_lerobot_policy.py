#!/usr/bin/env python
"""Export native RL token + actor-critic checkpoints as LeRobot policy directories.

evo-rlt-train-rl-token writes demo_adapt_checkpoint.pt and evo-rlt-train-actor-critic writes
rl_checkpoint.pt, but evo-rlt-record loads LeRobot `pretrained_model` directories. This writes

    <output_dir>/rlt_token/   config.json (type rlt_token) + model.safetensors (RL token encoder + decoder)
    <output_dir>/rlt_ac/      config.json (type rlt_ac)    + model.safetensors (actor, critic, target critic)

and copies the SFT pi0.5 pre/post-processor files into both, so deployment normalizes observations and
un-normalizes actions with the VLA's stats. Point evo-rlt-record's `policy_path` at <output_dir>/rlt_ac;
its config references the rlt_token directory by absolute path (`rl_token_path` overrides it).

    evo-rlt-export-lerobot-policy \\
        --rl-token-checkpoint output/checkpoint/rl_token/<run>/demo_adapt_checkpoint.pt \\
        --ac-checkpoint output/checkpoint/actor_critic/<run>/rl_checkpoint.pt \\
        --output-dir output/checkpoint/actor_critic/<run>/lerobot

Architectures, shapes and camera layout come from the checkpoints' state dicts and metadata; weights keep
their saved dtype. The export refuses a camera layout that deploy-time prefix capture cannot reproduce.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from evo_rlt.cli.common import configure_logging, parse_args_with_run_section

logger = configure_logging(__name__)

RLT_TOKEN_DIR = "rlt_token"
RLT_AC_DIR = "rlt_ac"
EXPORT_INFO_FILE = "export_info.json"
IMAGE_KEY_PREFIX = "observation.images."
# SigLIP in PaliGemma-3b-224 emits (224 / 14) ** 2 patch tokens per camera.
TOKENS_PER_CAMERA = 256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export demo_adapt_checkpoint.pt + rl_checkpoint.pt as rlt_token / rlt_ac LeRobot policy "
            "directories for evo-rlt-record. --config PATH.yaml supplies option defaults via a `run:` section."
        )
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--rl-token-checkpoint", required=True, help="demo_adapt_checkpoint.pt from evo-rlt-train-rl-token.")
    parser.add_argument("--ac-checkpoint", required=True, help="rl_checkpoint.pt from evo-rlt-train-actor-critic.")
    parser.add_argument("--output-dir", required=True, help=f"Writes <output-dir>/{RLT_TOKEN_DIR} and <output-dir>/{RLT_AC_DIR}.")
    parser.add_argument(
        "--vla-path",
        default=None,
        help="SFT pi0.5 pretrained_model dir; default: the vla_model recorded in the RL token checkpoint.",
    )
    parser.add_argument(
        "--phase-mode",
        default="manual",
        choices=["manual", "always_rl", "always_vla"],
        help="rlt_ac deploy phase. evo-rlt-record switches VLA/RL itself, which needs manual.",
    )
    parser.add_argument("--chunk-exec-steps", type=int, default=25, help="VLA-phase actions executed per pi0.5 chunk.")
    parser.add_argument("--vla-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer-path", default=None, help="Local PaliGemma tokenizer; default: the SFT preprocessor's.")
    parser.add_argument("--overwrite", action="store_true", help="Write into export directories that already exist.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return parse_args_with_run_section(build_parser(), argv)


# ---------------------------------------------------------------------------
# Architecture recovery
# ---------------------------------------------------------------------------


def _layer_count(state_dict: dict[str, torch.Tensor], stack: str) -> int:
    indices = {int(key.split(".")[2]) for key in state_dict if key.startswith(f"{stack}.layers.")}
    return max(indices) + 1 if indices else 0


def rl_token_architecture(state_dict: dict[str, torch.Tensor], meta: dict) -> dict:
    """RLTokenModule kwargs from the weights, cross-checked against the saved metadata."""
    from evo_rlt.core.rl_token import rl_token_arch_from_state_dict

    recovered = rl_token_arch_from_state_dict(state_dict)
    arch = {
        "arch": recovered["arch"],
        "seq_len": recovered["seq_len"],
        "num_rl_tokens": recovered["num_rl_tokens"],
        "token_dim": int(state_dict["rl_token_embed"].shape[-1]),
        "enc_layers": _layer_count(state_dict, "encoder"),
        "dec_layers": _layer_count(state_dict, "decoder"),
        "ff_dim": int(state_dict["encoder.layers.0.linear1.weight"].shape[0]),
    }
    saved = meta.get("rl_token") or {}
    for key, value in arch.items():
        if saved.get(key) is not None and saved[key] != value:
            raise ValueError(f"RL token checkpoint metadata says {key}={saved[key]!r} but its weights have {value!r}")
    # Head count is not recoverable from the weights.
    arch["nhead"] = int(saved.get("nhead") or 8)
    if not arch["dec_layers"]:
        raise ValueError("RL token checkpoint has no decoder weights; export needs the full demo_adapt_checkpoint.pt")
    return arch


def _mlp_architecture(state_dict: dict[str, torch.Tensor], saved: dict, name: str) -> dict:
    from evo_rlt.core.utils import infer_actor_architecture

    inferred = infer_actor_architecture(state_dict)
    if not saved:
        logger.warning("%s has no saved architecture metadata; inferred %s (activation assumed relu)", name, inferred)
        return inferred
    saved = {"arch": "mlp", **saved}  # checkpoints from before `arch` existed are all "mlp"
    # layer_norm / residual only shape the "mlp" head; "openpi" ignores them
    checked = ("arch", "hidden_dim", "num_layers") + (("layer_norm", "residual") if inferred["arch"] == "mlp" else ())
    for key in checked:
        if key in saved and saved[key] != inferred[key]:
            raise ValueError(f"{name} metadata says {key}={saved[key]!r} but its weights have {inferred[key]!r}")
    return {**inferred, **{key: saved[key] for key in inferred if key in saved}}


def actor_critic_architecture(ckpt: dict, token_dim: int) -> dict:
    """Actor / critic kwargs and chunk shapes, checked against the actor's input width."""
    meta = ckpt.get("metadata") or {}
    for key in ("chunk_length", "action_dim", "proprio_dim"):
        if key not in meta:
            raise ValueError(f"AC checkpoint metadata lacks {key!r}; re-save it with evo-rlt-train-actor-critic")
    chunk_dim = meta["chunk_length"] * meta["action_dim"]
    actor_sd = ckpt["actor_state_dict"]
    if "net.z_proj.weight" in actor_sd:  # openpi head: one input projection per stream
        in_dims = tuple(actor_sd[f"net.{p}_proj.weight"].shape[1] for p in ("z", "proprio", "chunk"))
    else:
        in_dims = (next(v for k, v in actor_sd.items() if k.endswith(".weight") and v.ndim == 2).shape[1],)
    expected = (token_dim, meta["proprio_dim"], chunk_dim)
    if in_dims not in (expected, (sum(expected),)):
        raise ValueError(
            f"actor input width {in_dims} != z_rl {token_dim} + proprio {meta['proprio_dim']} + ref chunk {chunk_dim}: "
            "the AC checkpoint was not trained on this RL token's state"
        )
    if "action_low" in actor_sd and actor_sd["action_low"].numel() != chunk_dim:
        raise ValueError(f"actor action bounds cover {actor_sd['action_low'].numel()} values, expected {chunk_dim}")
    critic_q1 = {k.removeprefix("q1."): v for k, v in ckpt["critic_state_dict"].items() if k.startswith("q1.")}
    return {
        "chunk_length": meta["chunk_length"],
        "action_dim": meta["action_dim"],
        "proprio_dim": meta["proprio_dim"],
        "actor": _mlp_architecture(actor_sd, meta.get("actor") or {}, "actor"),
        "critic": _mlp_architecture(critic_q1, meta.get("critic") or {}, "critic"),
        "training": meta.get("training") or {},
    }


# ---------------------------------------------------------------------------
# Camera layout
# ---------------------------------------------------------------------------


def _sft_rename_map(sft_dir: Path) -> dict[str, str]:
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME

    config = json.loads((sft_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json").read_text())
    for step in config.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            return dict(step.get("config", {}).get("rename_map") or {})
    return {}


def deploy_camera_keys(rl_meta: dict, sft_dir: Path) -> list[str]:
    """pi0.5 camera slots whose image tokens the RL token encodes, in deploy prefix order.

    Native training (Pi05VLAAdapter) lays the prefix out in PI05_CAMERA_SLOTS order with a masked empty
    slot for each camera it lacks, then keeps the `active_cameras` slots. At deploy LeRobot's PI05Policy
    puts the cameras it receives first, in its config's image_features order, and appends the empty ones,
    and ChunkACPolicy keeps the first len(camera_keys) cameras. Masked slots take no position ids and are
    never attended to, so both sides see the same tokens exactly when the native selection is the set of
    cameras the SFT preprocessor delivers, in the same order.
    """
    from evo_rlt.adapters.lerobot.pi05_adapter import DEFAULT_CAMERA_NAME_MAP, PI05_CAMERA_SLOTS

    camera_name_map = rl_meta.get("camera_name_map") or dict(DEFAULT_CAMERA_NAME_MAP)
    active = rl_meta.get("active_cameras")
    if active:
        slots = {camera_name_map.get(camera, camera) for camera in active}
        unknown = sorted(slots - set(PI05_CAMERA_SLOTS))
        if unknown:
            raise ValueError(f"active cameras {active} resolve to {unknown}, which are not pi0.5 camera slots")
        native = sorted(slots, key=PI05_CAMERA_SLOTS.index)
    elif rl_meta.get("image_only"):
        native = list(PI05_CAMERA_SLOTS)
    else:
        raise ValueError(
            "the RL token was trained on image + language tokens (image_only false); deploy-time prefix "
            "capture only supports image tokens"
        )

    rename_map = _sft_rename_map(sft_dir)
    for camera, slot in camera_name_map.items():
        deploy_slot = rename_map.get(IMAGE_KEY_PREFIX + camera, IMAGE_KEY_PREFIX + camera)
        if deploy_slot != slot:
            raise ValueError(
                f"camera {camera!r} went to {slot} in RL token training, but the SFT preprocessor sends it to "
                f"{deploy_slot}"
            )
    sft_config = json.loads((sft_dir / "config.json").read_text())
    image_features = [key for key, ft in sft_config["input_features"].items() if ft["type"] == "VISUAL"]
    delivered = set(rename_map.values()) | set(camera_name_map.values())
    deploy = [key for key in image_features if key in delivered]
    if deploy != native:
        raise ValueError(
            f"the RL token encodes the tokens of {native} but deploy would capture those of {deploy}; "
            "train the RL token with active_cameras set to exactly the cameras the robot provides"
        )
    return [key.removeprefix(IMAGE_KEY_PREFIX) for key in deploy]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _file_info(path: Path) -> dict:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _features(camera_keys: list[str], proprio_dim: int, action_dim: int):
    from lerobot.configs.types import FeatureType, PolicyFeature

    inputs = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(proprio_dim,))}
    for key in camera_keys:
        inputs[IMAGE_KEY_PREFIX + key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
    return inputs, {"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}


def _prepare_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"{path} already exists and is not empty; pass --overwrite to replace the export")
    path.mkdir(parents=True, exist_ok=True)


def copy_processor_files(sft_dir: Path, out_dir: Path) -> None:
    """Copy the SFT pi0.5 pre/post-processor configs and their state files."""
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

    for name in (POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME):
        config_path = sft_dir / f"{name}.json"
        shutil.copy2(config_path, out_dir / config_path.name)
        for step in json.loads(config_path.read_text()).get("steps", []):
            if step.get("state_file"):
                shutil.copy2(sft_dir / step["state_file"], out_dir / step["state_file"])


def _save_tensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().cpu().clone().contiguous() for k, v in tensors.items()}, str(path), metadata={"format": "pt"})


def _check_strict_load(module: torch.nn.Module, state_dict: dict[str, torch.Tensor], name: str) -> None:
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{name}: exported config does not rebuild the checkpoint (missing {missing}, unexpected {unexpected})")


def export(
    rl_token_checkpoint: str | Path,
    ac_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    vla_path: str | Path | None = None,
    phase_mode: str = "manual",
    chunk_exec_steps: int = 25,
    vla_dtype: str = "bfloat16",
    device: str = "cuda",
    tokenizer_path: str | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write the rlt_token and rlt_ac policy directories; returns their paths."""
    from evo_rlt.adapters.lerobot.demo_loader import IMAGE_RESIZE
    from evo_rlt.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
    from evo_rlt.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
    from evo_rlt.adapters.lerobot.policies.processor_rlt_common import _read_tokenizer_path
    from evo_rlt.core.actor import ChunkActor
    from evo_rlt.core.critic import TwinCritic
    from evo_rlt.core.rl_token import RLTokenModule

    rl_token_checkpoint, ac_checkpoint = Path(rl_token_checkpoint).resolve(), Path(ac_checkpoint).resolve()
    rl_ckpt = torch.load(rl_token_checkpoint, map_location="cpu", weights_only=False)
    ac_ckpt = torch.load(ac_checkpoint, map_location="cpu", weights_only=False)
    rl_meta = rl_ckpt.get("metadata") or {}
    rl_sd = rl_ckpt["rl_token_state_dict"]

    vla_path = vla_path or rl_meta.get("vla_model")
    if not vla_path:
        raise SystemExit("the RL token checkpoint does not record its pi0.5; pass --vla-path")
    sft_dir = Path(vla_path).resolve()
    if not (sft_dir / "config.json").is_file():
        raise SystemExit(f"--vla-path {sft_dir}: not a pi0.5 pretrained_model directory (no config.json)")
    if rl_meta.get("vla_model") and Path(rl_meta["vla_model"]).resolve() != sft_dir:
        logger.warning("RL token was trained on pi0.5 %s, exporting against %s", rl_meta["vla_model"], sft_dir)
    if rl_meta.get("image_resize") != IMAGE_RESIZE:
        logger.warning(
            "RL token checkpoint predates letterboxed training frames: it (and a cache built with it) saw "
            "frames stretched to 224x224, while deploy letterboxes them like pi0.5 SFT. Retrain it for deployment."
        )
    tokenizer_path = tokenizer_path or _read_tokenizer_path(str(sft_dir))

    token = rl_token_architecture(rl_sd, rl_meta)
    ac = actor_critic_architecture(ac_ckpt, token["token_dim"])
    camera_keys = deploy_camera_keys(rl_meta, sft_dir)
    token_pool_size = int(rl_meta.get("token_pool_size") or 0)
    if token["arch"] == "perceiver":
        expected = len(camera_keys) * TOKENS_PER_CAMERA
        if token_pool_size > 0:
            expected = min(expected, token_pool_size)
        if token["seq_len"] != expected:
            raise ValueError(
                f"RL token encodes {token['seq_len']} tokens but deploy captures {expected} "
                f"({len(camera_keys)} cameras x {TOKENS_PER_CAMERA}, pool {token_pool_size})"
            )
    input_features, output_features = _features(camera_keys, ac["proprio_dim"], ac["action_dim"])
    actor, critic, training = ac["actor"], ac["critic"], ac["training"]

    output_dir = Path(output_dir).resolve()
    token_dir, ac_dir = output_dir / RLT_TOKEN_DIR, output_dir / RLT_AC_DIR
    _prepare_dir(token_dir, overwrite)
    _prepare_dir(ac_dir, overwrite)

    # --- rlt_token: the full RL token module (the encoder is what deploy uses) ---
    token_cfg = RLTokenPolicyConfig(
        vla_pretrained_path=str(sft_dir),
        vla_dtype=vla_dtype,
        rl_token_dim=token["token_dim"],
        rl_token_nhead=token["nhead"],
        rl_token_enc_layers=token["enc_layers"],
        rl_token_dec_layers=token["dec_layers"],
        rl_token_ff_dim=token["ff_dim"],
        rl_token_num_rl_tokens=token["num_rl_tokens"],
        rl_token_arch=token["arch"],
        rl_token_seq_len=token["seq_len"],
        token_pool_size=token_pool_size,
        image_only=True,
        norm_gamma=float(rl_meta.get("norm_gamma") or 0.0),
        vla_ft_weight=0.0,
        action_dim=ac["action_dim"],
        proprio_dim=ac["proprio_dim"],
        camera_keys=camera_keys,
        tokenizer_path=tokenizer_path,
        device=device,
        push_to_hub=False,
        input_features=input_features,
        output_features=output_features,
    )
    rl_module = RLTokenModule(
        token_dim=token["token_dim"],
        nhead=token["nhead"],
        num_enc_layers=token["enc_layers"],
        num_dec_layers=token["dec_layers"],
        ff_dim=token["ff_dim"],
        num_rl_tokens=token["num_rl_tokens"],
        arch=token["arch"],
        seq_len=token["seq_len"],
    )
    _check_strict_load(rl_module, rl_sd, "RL token")
    token_cfg._save_pretrained(token_dir)
    _save_tensors({f"rl_token.{k}": v for k, v in rl_sd.items()}, token_dir / "model.safetensors")
    copy_processor_files(sft_dir, token_dir)

    # --- rlt_ac: actor + critics; pi0.5 and the RL token load from the paths in its config ---
    ac_cfg = ChunkACPolicyConfig(
        vla_pretrained_path=str(sft_dir),
        vla_dtype=vla_dtype,
        rl_token_pretrained_path=str(token_dir),
        rl_token_dim=token["token_dim"],
        rl_token_num_rl_tokens=token["num_rl_tokens"],
        token_pool_size=token_pool_size,
        image_only=True,
        actor_arch=actor["arch"],
        actor_hidden_dim=actor["hidden_dim"],
        actor_num_layers=actor["num_layers"],
        actor_fixed_std=actor["fixed_std"],
        actor_ref_dropout_p=actor["ref_dropout_p"],
        actor_activation=actor["activation"],
        actor_layer_norm=actor["layer_norm"],
        actor_residual=actor["residual"],
        critic_arch=critic["arch"],
        critic_hidden_dim=critic["hidden_dim"],
        critic_num_layers=critic["num_layers"],
        critic_activation=critic["activation"],
        critic_layer_norm=critic["layer_norm"],
        critic_residual=critic["residual"],
        **{key: training[key] for key in ("gamma", "beta", "tau", "utd_ratio", "actor_update_interval") if key in training},
        chunk_length=ac["chunk_length"],
        action_dim=ac["action_dim"],
        proprio_dim=ac["proprio_dim"],
        chunk_exec_steps=chunk_exec_steps,
        phase_mode=phase_mode,
        camera_keys=camera_keys,
        tokenizer_path=tokenizer_path,
        device=device,
        push_to_hub=False,
        input_features=input_features,
        output_features=output_features,
    )
    state_dim = token["token_dim"] + ac["proprio_dim"]
    chunk_dim = ac["chunk_length"] * ac["action_dim"]
    actor_module = ChunkActor(
        state_dim=state_dim,
        chunk_dim=chunk_dim,
        hidden_dim=actor["hidden_dim"],
        num_layers=actor["num_layers"],
        activation=actor["activation"],
        layer_norm=actor["layer_norm"],
        residual=actor["residual"],
        arch=actor["arch"],
        proprio_dim=ac["proprio_dim"],
    )
    critic_module = TwinCritic(
        state_dim=state_dim,
        chunk_dim=chunk_dim,
        hidden_dim=critic["hidden_dim"],
        num_layers=critic["num_layers"],
        activation=critic["activation"],
        layer_norm=critic["layer_norm"],
        residual=critic["residual"],
        arch=critic["arch"],
        proprio_dim=ac["proprio_dim"],
    )
    actor_sd = dict(ac_ckpt["actor_state_dict"])
    if "action_low" not in actor_sd:
        # Pre-bounds checkpoints were trained against the fixed [-1, 1] clamp (see ChunkActor._load_from_state_dict).
        logger.warning("AC checkpoint has no action bounds; exporting the fixed [-1, 1] it was trained with")
        actor_sd["action_low"] = torch.full((chunk_dim,), -1.0)
        actor_sd["action_high"] = torch.full((chunk_dim,), 1.0)
    _check_strict_load(actor_module, actor_sd, "actor")
    _check_strict_load(critic_module, ac_ckpt["critic_state_dict"], "critic")
    _check_strict_load(critic_module, ac_ckpt["target_critic_state_dict"], "target critic")
    tensors = {f"actor.{k}": v for k, v in actor_sd.items()}
    tensors.update({f"critic.{k}": v for k, v in ac_ckpt["critic_state_dict"].items()})
    tensors.update({f"target_critic.{k}": v for k, v in ac_ckpt["target_critic_state_dict"].items()})
    # One entry per critic update, so lerobot-train resumes the actor-update cadence where it stopped.
    tensors["_critic_step"] = torch.tensor(len(ac_ckpt.get("critic_losses") or []), dtype=torch.long)
    ac_cfg._save_pretrained(ac_dir)
    _save_tensors(tensors, ac_dir / "model.safetensors")
    copy_processor_files(sft_dir, ac_dir)

    info = {
        "exported_by": "evo-rlt-export-lerobot-policy",
        "rl_token_checkpoint": {**_file_info(rl_token_checkpoint), "step": rl_ckpt.get("step")},
        "ac_checkpoint": {**_file_info(ac_checkpoint), "step": ac_ckpt.get("step")},
        "vla_pretrained_path": str(sft_dir),
        "camera_keys": camera_keys,
        "rl_token_metadata": rl_meta,
        "ac_metadata": ac_ckpt.get("metadata") or {},
    }
    for directory in (token_dir, ac_dir):
        (directory / EXPORT_INFO_FILE).write_text(json.dumps(info, indent=2, default=str) + "\n")

    logger.info(
        "Exported RL token (%s, %d tokens from cameras %s) -> %s", token["arch"], token["seq_len"] or 0, camera_keys, token_dir
    )
    logger.info(
        "Exported actor-critic (step %s, actor %dx%d, chunk %dx%d, phase_mode=%s) -> %s",
        ac_ckpt.get("step"), actor["hidden_dim"], actor["num_layers"], ac["chunk_length"], ac["action_dim"], phase_mode, ac_dir,
    )
    return token_dir, ac_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, ac_dir = export(
        args.rl_token_checkpoint,
        args.ac_checkpoint,
        args.output_dir,
        vla_path=args.vla_path,
        phase_mode=args.phase_mode,
        chunk_exec_steps=args.chunk_exec_steps,
        vla_dtype=args.vla_dtype,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
        overwrite=args.overwrite,
    )
    print(f"\npolicy_path: {ac_dir}")


if __name__ == "__main__":
    main()
