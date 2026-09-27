from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# "ar": the paper's teacher-forced autoregressive decoder (Eq. 2). The decoder sees the ground-truth
#       prefix tokens, so it can reconstruct them from its own context and ignore z_rl.
# "perceiver": openpi-RLT's cross-attention encoder/decoder. The decoder's queries are learned
#       positions and its only input-dependent signal is z_rl, so reconstruction must go through it.
RL_TOKEN_ARCHS = ("ar", "perceiver")

# State-dict prefixes that only the reconstruction decoder uses (dropped for inference).
DECODER_KEY_PREFIXES = ("decoder.", "out_proj.")


def sinusoidal_pos_embed(length: int, dim: int) -> torch.Tensor:
    """(length, dim) [sin | cos] table; openpi-RLT's init for its learned queries and positions."""
    position = torch.arange(length, dtype=torch.float32)[:, None]
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    return torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=-1)


class CrossAttentionStack(nn.Module):
    """Perceiver-style stack ported from openpi-RLT (`openpi.models.rl_token`).

    Each layer is pre-LN self-attention over the queries, cross-attention from the queries to
    `context`, then an MLP (openpi's `CrossAttentionLayer`; the MLP here is a plain GELU FFN of
    width `ff_dim` instead of openpi's GeGLU, which would be ~170M params per layer at 2048 wide).
    `context_pos` is a learned position embedding added to the context (openpi's `y_pos_enc`).
    With `num_queries > 0` the stack owns its queries (the decoder's per-position queries).
    """

    def __init__(
        self,
        token_dim: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        context_len: int,
        num_queries: int = 0,
    ):
        super().__init__()
        self.context_pos = nn.Parameter(sinusoidal_pos_embed(context_len, token_dim).unsqueeze(0))
        if num_queries > 0:
            self.query = nn.Parameter(sinusoidal_pos_embed(num_queries, token_dim).unsqueeze(0))
        else:
            self.register_parameter("query", None)
        self.layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=token_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )

    def forward(self, context: torch.Tensor, queries: torch.Tensor | None = None) -> torch.Tensor:
        expected = self.context_pos.shape[1]
        if context.shape[1] != expected:
            raise ValueError(
                f"RL token was built for {expected} context tokens but got {context.shape[1]}: "
                "image_only / active_cameras / token_pool_size must match RL token training"
            )
        if queries is None:
            queries = self.query.expand(context.shape[0], -1, -1)
        context = context + self.context_pos
        x = queries
        for layer in self.layers:
            x = layer(x, context)
        return x


class RLTokenModule(nn.Module):
    """RL Token encoder-decoder with configurable number of RL tokens.

    Encoder compresses VLA tokens into N learned <rl> tokens z_rl (B, N, D). For downstream RL,
    z_rl is mean-pooled to (B, D). The decoder reconstructs the VLA tokens from z_rl and only
    exists to train the encoder. See RL_TOKEN_ARCHS for the two architectures.
    """

    def __init__(
        self,
        token_dim: int = 2048,
        nhead: int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        ff_dim: int | None = None,
        num_rl_tokens: int = 1,
        inference_only: bool = False,
        arch: str = "ar",
        seq_len: int | None = None,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * token_dim
        if arch not in RL_TOKEN_ARCHS:
            raise ValueError(f"arch must be one of {RL_TOKEN_ARCHS}, got {arch!r}")
        if arch == "perceiver" and not seq_len:
            raise ValueError("arch='perceiver' needs seq_len (the number of VLA tokens it encodes)")

        self.token_dim = token_dim
        self.num_rl_tokens = num_rl_tokens
        self.inference_only = inference_only
        self.arch = arch
        self.seq_len = seq_len if arch == "perceiver" else None

        if arch == "perceiver":
            self.rl_token_embed = nn.Parameter(sinusoidal_pos_embed(num_rl_tokens, token_dim).unsqueeze(0))
            self.encoder = CrossAttentionStack(token_dim, nhead, num_enc_layers, ff_dim, context_len=seq_len)
            if not inference_only:
                self.decoder = CrossAttentionStack(
                    token_dim, nhead, num_dec_layers, ff_dim, context_len=num_rl_tokens, num_queries=seq_len
                )
                self.out_proj = nn.Linear(token_dim, token_dim)
            return

        self.rl_token_embed = nn.Parameter(torch.randn(1, num_rl_tokens, token_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        if not inference_only:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=token_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
            self.out_proj = nn.Linear(token_dim, token_dim)

    def encode(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """Encode VLA tokens into RL token(s).

        Args:
            vla_tokens: (B, M, D) -- final VLA token embeddings (should be detached)

        Returns:
            z_rl: (B, D) -- mean-pooled RL token for downstream RL state
        """
        z_rl_multi = self.encode_multi(vla_tokens)
        return z_rl_multi.mean(dim=1)  # (B, D)

    def encode_multi(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """Encode VLA tokens into multiple RL tokens (for decoder).

        Returns:
            z_rl: (B, N, D) where N = num_rl_tokens
        """
        B = vla_tokens.shape[0]
        rl = self.rl_token_embed.expand(B, -1, -1)  # (B, N, D)
        if self.arch == "perceiver":
            return self.encoder(vla_tokens, queries=rl)
        x = torch.cat([vla_tokens, rl], dim=1)  # (B, M+N, D)
        out = self.encoder(x)
        return out[:, -self.num_rl_tokens:, :]  # (B, N, D)

    def decode(self, z_rl: torch.Tensor, teacher_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruct the VLA tokens from z_rl.

        Args:
            z_rl: (B, D) single token or (B, N, D) multi-token
            teacher_tokens: (B, M, D) stop-gradiented VLA embeddings. Required by the "ar"
                decoder (teacher forcing); the "perceiver" decoder must not see them.

        Returns:
            pred: (B, M, D)
        """
        # Handle both single and multi-token z_rl
        memory = z_rl.unsqueeze(1) if z_rl.dim() == 2 else z_rl  # (B, N, D)
        if self.arch == "perceiver":
            if teacher_tokens is not None:
                raise ValueError(
                    "the perceiver decoder reconstructs from z_rl alone; do not pass teacher_tokens"
                )
            return self.out_proj(self.decoder(memory))
        if teacher_tokens is None:
            raise ValueError("the 'ar' decoder needs teacher_tokens for teacher forcing")

        M = teacher_tokens.shape[1]
        N = memory.shape[1]
        # Shifted input: [rl_tokens..., z_1, ..., z_{M-N}]
        dec_input = torch.cat([memory, teacher_tokens[:, :M - N]], dim=1)  # (B, M, D)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            M, device=z_rl.device
        )
        out = self.decoder(tgt=dec_input, memory=memory, tgt_mask=causal_mask)
        return self.out_proj(out)

    def _reconstruct(self, z_rl_multi: torch.Tensor, z_bar: torch.Tensor) -> torch.Tensor:
        return self.decode(z_rl_multi, z_bar if self.arch == "ar" else None)

    @staticmethod
    def _weighted_mse(
        pred: torch.Tensor, target: torch.Tensor, dim_std: torch.Tensor | None, gamma: float
    ) -> torch.Tensor:
        diff = pred - target
        if gamma > 0 and dim_std is not None:
            weight = dim_std.clamp_min(1e-6).pow(-gamma).to(device=diff.device, dtype=diff.dtype)
            diff = diff * weight
        return (diff ** 2).mean()
    def reconstruction_loss(
        self,
        vla_tokens: torch.Tensor,
        dim_std: torch.Tensor | None = None,
        gamma: float = 0.0,
    ) -> torch.Tensor:
        """L_ro = E[|| (pred_i - z_bar_i) * D^{-gamma} ||^2].
        gamma=0 reduces to vanilla MSE (paper Eq. 2). gamma=1 is full whitening
        in the per-dim std metric. gamma=0.5 partially compensates the few
        high-variance dims that otherwise dominate the gradient.
        """
        z_bar = vla_tokens.detach()
        z_rl_multi = self.encode_multi(z_bar)
        pred = self._reconstruct(z_rl_multi, z_bar)
        return self._weighted_mse(pred, z_bar, dim_std, gamma)

    @torch.no_grad()
    def reconstruction_diagnostics(
        self,
        vla_tokens: torch.Tensor,
        dim_std: torch.Tensor | None = None,
        gamma: float = 0.0,
    ) -> dict[str, float]:
        """Does the decoder actually read z_rl? (eval mode, no grad)

        recon: loss with each sample's own z_rl. recon_shuffled: loss with z_rl taken from another
        sample of the batch. z_rl_usage = 1 - recon / recon_shuffled is the share of the error
        that the right z_rl removes; ~0 means the decoder ignores z_rl. z_rl_cos is the mean
        pairwise cosine of the pooled z_rl across the batch; ~1 means the encoder emits a constant.
        """
        was_training = self.training
        self.eval()
        try:
            z_bar = vla_tokens.detach()
            z_rl = self.encode_multi(z_bar)
            recon = self._weighted_mse(self._reconstruct(z_rl, z_bar), z_bar, dim_std, gamma).item()
            out = {"recon": recon}
            B = z_rl.shape[0]
            if B > 1:
                shuffled = torch.roll(z_rl, shifts=B // 2, dims=0)
                recon_shuffled = self._weighted_mse(
                    self._reconstruct(shuffled, z_bar), z_bar, dim_std, gamma
                ).item()
                pooled = F.normalize(z_rl.mean(dim=1), dim=-1)
                cos = pooled @ pooled.T
                off_diag = cos[~torch.eye(B, dtype=torch.bool, device=cos.device)]
                out.update(
                    recon_shuffled=recon_shuffled,
                    z_rl_usage=1.0 - recon / max(recon_shuffled, 1e-12),
                    z_rl_cos=off_diag.mean().item(),
                )
            return out
        finally:
            self.train(was_training)


def rl_token_arch_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, str | int | None]:
    """Recover arch / seq_len / num_rl_tokens from an RLTokenModule state_dict."""
    arch: dict[str, str | int | None] = {"arch": "ar", "seq_len": None}
    if "encoder.context_pos" in state_dict:
        arch = {"arch": "perceiver", "seq_len": int(state_dict["encoder.context_pos"].shape[1])}
    if "rl_token_embed" in state_dict:
        arch["num_rl_tokens"] = int(state_dict["rl_token_embed"].shape[1])
    return arch


def load_rl_token_encoder(module: RLTokenModule, state_dict: dict[str, torch.Tensor]) -> None:
    """Load the encoder half of an RL token checkpoint into `module`, failing on any mismatch.

    Decoder weights in the checkpoint are ignored (and may be missing from `module`), but every
    encoder weight must be present on both sides with matching shapes -- a silent partial load
    would leave a randomly initialised encoder behind.
    """
    encoder_sd = {k: v for k, v in state_dict.items() if not k.startswith(DECODER_KEY_PREFIXES)}
    missing, unexpected = module.load_state_dict(encoder_sd, strict=False)
    missing = [k for k in missing if not k.startswith(DECODER_KEY_PREFIXES)]
    if missing or unexpected:
        raise RuntimeError(
            f"RL token checkpoint does not match the module (arch={module.arch!r}): "
            f"missing encoder keys {missing[:5]}, unexpected keys {unexpected[:5]}"
        )
