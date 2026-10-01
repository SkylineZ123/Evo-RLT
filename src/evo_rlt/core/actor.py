from __future__ import annotations

import torch
import torch.nn as nn

from evo_rlt.core.utils import build_mlp, _get_activation

# "mlp": one MLP over the concatenated [z_rl, proprio, chunk] (plain or residual, see ResidualMLP).
# "openpi": openpi-RLT's head (rlt_online_rl/networks.py), see OpenpiMLP.
HEAD_ARCHS = ("mlp", "openpi")


class OpenpiMLP(nn.Module):
    """openpi-RLT's actor/critic network, taking the same concatenated input as the "mlp" head.

    Each input stream gets its own projection before the trunk:
      z_rl -> Linear(256) -> LN;  proprio -> Linear(64) -> LN -> tanh;  chunk -> Linear(256) -> LN -> tanh.
    The 576-wide concat then runs through num_layers x [Linear(hidden) -> LN -> GELU] and a linear output.
    Without this, the 7 proprio dims and the 70 chunk dims share one first layer with 2048 z_rl dims of
    much larger scale. As in openpi, LayerNorm has no affine parameters, GELU uses the tanh approximation,
    weights are Xavier-uniform and biases zero.
    """

    Z_PROJ, PROPRIO_PROJ, CHUNK_PROJ = 256, 64, 256

    def __init__(
        self, z_dim: int, proprio_dim: int, chunk_dim: int, hidden_dim: int, num_layers: int, out_dim: int
    ):
        super().__init__()
        self.z_dim, self.proprio_dim = z_dim, proprio_dim
        self.z_proj = self._linear(z_dim, self.Z_PROJ)
        self.proprio_proj = self._linear(proprio_dim, self.PROPRIO_PROJ)
        self.chunk_proj = self._linear(chunk_dim, self.CHUNK_PROJ)
        dims = [self.Z_PROJ + self.PROPRIO_PROJ + self.CHUNK_PROJ] + [hidden_dim] * num_layers
        self.hidden = nn.ModuleList(self._linear(dims[i], dims[i + 1]) for i in range(num_layers))
        self.out = self._linear(dims[-1], out_dim)

    @staticmethod
    def _linear(in_dim: int, out_dim: int) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    @staticmethod
    def _ln(x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(x, x.shape[-1:], eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split = self.z_dim + self.proprio_dim
        h = torch.cat([
            self._ln(self.z_proj(x[..., :self.z_dim])),
            torch.tanh(self._ln(self.proprio_proj(x[..., self.z_dim:split]))),
            torch.tanh(self._ln(self.chunk_proj(x[..., split:]))),
        ], dim=-1)
        for layer in self.hidden:
            h = nn.functional.gelu(self._ln(layer(h)), approximate="tanh")
        return self.out(h)


def build_head(
    arch: str,
    state_dim: int,
    chunk_dim: int,
    out_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str = "relu",
    layer_norm: bool = False,
    residual: bool = False,
    proprio_dim: int | None = None,
) -> nn.Module:
    """Network over [state_vec, chunk] (state_vec = [z_rl, proprio]) for the actor and the critics.

    activation / layer_norm / residual only apply to arch "mlp"; "openpi" needs proprio_dim to split
    state_vec back into z_rl and proprio.
    """
    if arch == "openpi":
        if not proprio_dim:
            raise ValueError("arch='openpi' needs proprio_dim to split state_vec into z_rl and proprio")
        return OpenpiMLP(state_dim - proprio_dim, proprio_dim, chunk_dim, hidden_dim, num_layers, out_dim)
    if arch != "mlp":
        raise ValueError(f"arch must be one of {HEAD_ARCHS}, got {arch!r}")
    mlp = ResidualMLP if residual else build_mlp
    return mlp(state_dim + chunk_dim, hidden_dim, out_dim, num_layers, activation=activation, layer_norm=layer_norm)


class ResidualMLP(nn.Module):
    """MLP with residual connections between hidden layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: str = "relu",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            block_layers: list[nn.Module] = [nn.Linear(hidden_dim, hidden_dim)]
            if layer_norm:
                block_layers.append(nn.LayerNorm(hidden_dim))
            block_layers.append(_get_activation(activation))
            blocks.append(nn.Sequential(*block_layers))
        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.output_proj(h)


class ChunkActor(nn.Module):
    """Actor that predicts an action chunk conditioned on RL state and VLA reference.

    Uses binary reference dropout (per batch element) during training to avoid
    over-reliance on the VLA reference chunk.
    """

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        fixed_std: float = 0.05,
        ref_dropout_p: float = 0.5,
        activation: str = "relu",
        layer_norm: bool = False,
        residual: bool = False,
        arch: str = "mlp",
        proprio_dim: int | None = None,
    ):
        super().__init__()
        self.net = build_head(
            arch, state_dim, chunk_dim, chunk_dim, hidden_dim, num_layers,
            activation=activation, layer_norm=layer_norm, residual=residual, proprio_dim=proprio_dim,
        )
        self.fixed_std = fixed_std
        self.ref_dropout_p = ref_dropout_p
        # Per-element bounds of the flattened chunk in the normalized action space: the range the
        # critic has seen actions in. Saved with the weights so the TD target and deploy share them.
        self.register_buffer("action_low", torch.full((chunk_dim,), -1.0))
        self.register_buffer("action_high", torch.full((chunk_dim,), 1.0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before the bounds existed were trained with the fixed [-1, 1] clamp.
        state_dict.setdefault(prefix + "action_low", torch.full_like(self.action_low, -1.0))
        state_dict.setdefault(prefix + "action_high", torch.full_like(self.action_high, 1.0))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def set_action_bounds(self, low: torch.Tensor, high: torch.Tensor) -> None:
        """Set per-dimension bounds of shape (action_dim,), repeated over every step of the chunk."""
        low, high = torch.as_tensor(low).flatten(), torch.as_tensor(high).flatten()
        repeats = self.action_low.numel() // max(low.numel(), 1)
        if low.shape != high.shape or repeats * low.numel() != self.action_low.numel():
            raise ValueError(
                f"bounds of shape {tuple(low.shape)}/{tuple(high.shape)} do not tile a chunk of {self.action_low.numel()}"
            )
        if (low > high).any():
            raise ValueError("action_low must not exceed action_high")
        self.action_low.copy_(low.repeat(repeats))
        self.action_high.copy_(high.repeat(repeats))

    def clamp_action(self, action_flat: torch.Tensor) -> torch.Tensor:
        """Clamp a flattened action chunk (..., chunk_dim) to the action bounds."""
        return action_flat.clamp(self.action_low, self.action_high)

    def forward(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (mu, std)."""
        if training:
            mask = (
                torch.rand(state_vec.shape[0], 1, device=state_vec.device) > self.ref_dropout_p
            ).float()
            ref_chunk_flat = ref_chunk_flat * mask
        x = torch.cat([state_vec, ref_chunk_flat], dim=-1)
        mu = self.net(x)
        std = torch.full_like(mu, self.fixed_std)
        return mu, std

    def sample(
        self,
        state_vec: torch.Tensor,
        ref_chunk_flat: torch.Tensor,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action with Gaussian noise. Returns (action, mu)."""
        mu, std = self.forward(state_vec, ref_chunk_flat, training)
        return mu + std * torch.randn_like(std), mu
