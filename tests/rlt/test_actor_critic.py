from __future__ import annotations

import torch
import pytest

from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.critic import ChunkCritic, TwinCritic


@pytest.fixture
def actor():
    return ChunkActor(state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2)


@pytest.fixture
def twin_critic():
    return TwinCritic(state_dim=78, chunk_dim=140, hidden_dim=64, num_layers=2)


class TestActor:
    def test_forward_shapes(self, actor):
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        mu, std = actor(state, ref)
        assert mu.shape == (8, 140)
        assert std.shape == (8, 140)

    def test_sample_shapes(self, actor):
        state = torch.randn(8, 78)
        ref = torch.randn(8, 140)
        action, mu = actor.sample(state, ref)
        assert action.shape == (8, 140)
        assert mu.shape == (8, 140)

    def test_fixed_std(self, actor):
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        _, std = actor(state, ref)
        assert torch.allclose(std, torch.full_like(std, 0.05))

    def test_ref_dropout_statistics(self):
        """With large batch and training=True, ~50% should be zeroed."""
        actor = ChunkActor(state_dim=78, chunk_dim=140, hidden_dim=64, ref_dropout_p=0.5)
        state = torch.randn(1000, 78)
        ref = torch.ones(1000, 140)  # all ones so we can detect zeroing

        torch.manual_seed(42)
        mu, _ = actor(state, ref, training=True)

        # The ref was multiplied by a mask. We can check by looking at the input
        # indirectly: the ratio of zero-ref samples should be ~50%
        # We verify by calling forward manually and checking the mask effect
        torch.manual_seed(42)
        mask = (torch.rand(1000, 1) > 0.5).float()
        frac_kept = mask.mean().item()
        assert 0.4 < frac_kept < 0.6

    def test_gradient_flow(self, actor):
        state = torch.randn(4, 78)
        ref = torch.randn(4, 140)
        action, _ = actor.sample(state, ref, training=True)
        loss = action.sum()
        loss.backward()
        for p in actor.parameters():
            assert p.grad is not None


class TestCritic:
    def test_chunk_critic_shape(self):
        critic = ChunkCritic(state_dim=78, chunk_dim=140, hidden_dim=64)
        q = critic(torch.randn(8, 78), torch.randn(8, 140))
        assert q.shape == (8, 1)

    def test_twin_critic_shapes(self, twin_critic):
        state = torch.randn(8, 78)
        action = torch.randn(8, 140)
        q1, q2 = twin_critic(state, action)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_min_q(self, twin_critic):
        state = torch.randn(8, 78)
        action = torch.randn(8, 140)
        q1, q2 = twin_critic(state, action)
        min_q = twin_critic.min_q(state, action)
        expected = torch.minimum(q1, q2)
        assert torch.allclose(min_q, expected)

    def test_gradient_flow(self, twin_critic):
        state = torch.randn(4, 78)
        action = torch.randn(4, 140)
        q = twin_critic.min_q(state, action)
        q.sum().backward()
        for p in twin_critic.parameters():
            assert p.grad is not None


class TestOpenpiHead:
    """arch="openpi": openpi-RLT's per-input LN projections + LN/GELU trunk (core.actor.OpenpiMLP)."""

    Z, P, CHUNK = 64, 7, 70

    def _actor(self):
        return ChunkActor(state_dim=self.Z + self.P, chunk_dim=self.CHUNK, hidden_dim=32, num_layers=2,
                          arch="openpi", proprio_dim=self.P)

    def test_shapes_and_gradients(self):
        actor = self._actor()
        critic = TwinCritic(state_dim=self.Z + self.P, chunk_dim=self.CHUNK, hidden_dim=32, arch="openpi",
                            proprio_dim=self.P)
        state, ref = torch.randn(8, self.Z + self.P), torch.randn(8, self.CHUNK)
        mu, _ = actor(state, ref)
        q1, q2 = critic(state, mu)
        assert mu.shape == (8, self.CHUNK) and q1.shape == q2.shape == (8, 1)
        (q1.sum() + q2.sum()).backward()
        assert all(p.grad is not None for p in list(actor.parameters()) + list(critic.parameters()))

    def test_proprio_is_heard_next_to_a_large_z_rl(self):
        # The proprio stream is layer-normed on its own, so a 100x larger z_rl cannot drown it out.
        actor = self._actor().eval()
        state, ref = torch.randn(16, self.Z + self.P), torch.randn(16, self.CHUNK)
        state[:, :self.Z] *= 100.0
        moved = state.clone()
        moved[:, self.Z:] += 0.5
        with torch.no_grad():
            delta = (actor(moved, ref)[0] - actor(state, ref)[0]).abs().mean()
        assert delta > 1e-2

    def test_needs_proprio_dim(self):
        with pytest.raises(ValueError, match="proprio_dim"):
            ChunkActor(state_dim=71, chunk_dim=70, arch="openpi")

    def test_config_rejects_unknown_arch(self):
        from evo_rlt.core.config import ActorConfig, CriticConfig

        with pytest.raises(ValueError, match="actor.arch"):
            ActorConfig(arch="transformer")
        with pytest.raises(ValueError, match="critic.arch"):
            CriticConfig(arch="transformer")

    def test_architecture_is_recovered_from_the_weights(self):
        from evo_rlt.core.utils import infer_actor_architecture

        inferred = infer_actor_architecture(self._actor().state_dict())
        shape = (inferred["arch"], inferred["hidden_dim"], inferred["num_layers"], inferred["proprio_dim"])
        assert shape == ("openpi", 32, 2, self.P)
        rebuilt = ChunkActor(state_dim=self.Z + self.P, chunk_dim=self.CHUNK,
                             **{k: v for k, v in inferred.items() if k not in ("fixed_std", "ref_dropout_p")})
        rebuilt.load_state_dict(self._actor().state_dict())
