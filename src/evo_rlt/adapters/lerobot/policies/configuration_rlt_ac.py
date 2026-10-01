from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

_DEFAULT_CAMERA_KEYS: list[str] = ["left_wrist", "right_wrist", "right_front"]


@dataclass
class ChunkACPolicyConfig(PreTrainedConfig):
    """Config for the chunk-level TD3+BC actor-critic policy on top of RL Token.

    Train via lerobot-train CLI with a ChunkTransitionDataset; deploy via
    make_policy + lerobot-record. forward(batch) returns a single scalar TD3+BC
    loss combining critic MSE + (every actor_update_interval steps) the actor
    Q-maximization + BC reg. Target critic is soft-updated with tau after every
    critic step. UTD ratio is achieved by setting outer --steps=outer*utd
    (each forward() == one critic update; actor update gated by counter).
    """

    # --- VLA + RL Token backbones (frozen, not serialized) ---
    vla_pretrained_path: str = "lerobot/pi05_base"
    vla_revision: str | None = None
    vla_dtype: str = "bfloat16"
    rl_token_pretrained_path: str = ""

    # --- RL Token arch (must match the loaded RLTokenPolicy ckpt) ---
    rl_token_dim: int = 2048
    rl_token_num_rl_tokens: int = 1
    token_pool_size: int = 0
    image_only: bool = False
    active_camera_indices: list[int] | None = None
    num_per_camera: int = 0

    # --- Actor (arch: "mlp" | "openpi", see evo_rlt.core.actor.build_head) ---
    actor_arch: str = "mlp"
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    actor_fixed_std: float = 0.05
    actor_ref_dropout_p: float = 0.5
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = False

    # --- Critic + target ---
    critic_arch: str = "mlp"
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    critic_activation: str = "relu"
    critic_layer_norm: bool = False
    critic_residual: bool = False

    # --- TD3+BC hyperparams ---
    gamma: float = 0.99
    beta: float = 0.3
    tau: float = 0.005
    utd_ratio: int = 5
    actor_update_interval: int = 2
    target_q_clip: float = 100.0

    # --- Shapes ---
    chunk_length: int = 10
    action_dim: int = 12
    proprio_dim: int = 12

    # --- Deploy ---
    chunk_exec_steps: int = 25
    phase_mode: str = "always_rl"
    deterministic: bool = True
    # torch.compile the pi0.5 sampler (ChunkACPolicy._compile_vla). The CUDA-graph modes
    # ("reduce-overhead", "max-autotune") replay its kernels as graphs, so inference stops
    # waiting on the GIL behind the robot's reader threads. Not supported together with RTC.
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"

    # --- Observation mapping (for deploy preprocessor) ---
    camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_CAMERA_KEYS))

    # --- Normalization MATCHES PI05 (deploy parity) ---
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # --- pi05 proxy fields ---
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    tokenizer_path: str | None = None

    @classmethod
    def ensure_registered(cls) -> None:
        import lerobot.policies  # noqa: F401

        PreTrainedConfig._choice_registry["rlt_ac"] = cls

    def __post_init__(self) -> None:
        self.ensure_registered()
        super().__post_init__()
        if self.phase_mode not in ("always_rl", "always_vla", "manual"):
            raise ValueError(
                f"phase_mode must be 'always_rl', 'always_vla', or 'manual', got {self.phase_mode!r}"
            )

    @property
    def type(self) -> str:
        return "rlt_ac"

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {}
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.proprio_dim,)
            )
        for cam_key in self.camera_keys:
            img_key = f"{OBS_IMAGES}.{cam_key}"
            if img_key not in self.input_features:
                self.input_features[img_key] = PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, *self.image_resolution)
                )
        if not self.output_features:
            self.output_features = {}
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION, shape=(self.action_dim,)
            )

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=3e-4, weight_decay=0.0, grad_clip_norm=1.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_length))

    @property
    def reward_delta_indices(self) -> None:
        return None
