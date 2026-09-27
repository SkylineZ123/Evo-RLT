from __future__ import annotations

import torch
import pytest

from evo_rlt.core.rl_token import RLTokenModule, load_rl_token_encoder, rl_token_arch_from_state_dict
from evo_rlt.core.utils import filter_encoder_only


@pytest.fixture
def rl_token():
    return RLTokenModule(token_dim=64, nhead=4, num_enc_layers=1, num_dec_layers=1, ff_dim=128)


def test_encode_shape(rl_token):
    tokens = torch.randn(3, 8, 64)
    z_rl = rl_token.encode(tokens)
    assert z_rl.shape == (3, 64)


def test_decode_shape(rl_token):
    z_rl = torch.randn(3, 64)
    teacher = torch.randn(3, 8, 64)
    pred = rl_token.decode(z_rl, teacher)
    assert pred.shape == (3, 8, 64)


def test_reconstruction_loss_scalar(rl_token):
    tokens = torch.randn(3, 8, 64)
    loss = rl_token.reconstruction_loss(tokens)
    assert loss.shape == ()
    assert not torch.isnan(loss)


def test_reconstruction_loss_decreases(rl_token):
    """Verify that reconstruction loss decreases on fixed data after optimization."""
    tokens = torch.randn(4, 8, 64)
    optimizer = torch.optim.Adam(rl_token.parameters(), lr=1e-3)

    initial_loss = rl_token.reconstruction_loss(tokens).item()
    for _ in range(50):
        loss = rl_token.reconstruction_loss(tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = rl_token.reconstruction_loss(tokens).item()

    assert final_loss < initial_loss


def test_gradients_flow_to_encoder_decoder(rl_token):
    """Verify gradients flow to encoder/decoder params but not input tokens."""
    tokens = torch.randn(2, 8, 64, requires_grad=True)
    loss = rl_token.reconstruction_loss(tokens)
    loss.backward()

    # Input tokens are detached inside reconstruction_loss, so no grad
    assert tokens.grad is None or (tokens.grad == 0).all()

    # Encoder/decoder params should have gradients
    assert rl_token.rl_token_embed.grad is not None
    for p in rl_token.encoder.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_causal_mask_applied(rl_token):
    """Verify causal masking by checking decode doesn't leak future information.

    We modify a teacher token at position k and verify that output positions
    before k (in shifted-input terms) are unaffected.
    """
    rl_token.eval()  # disable dropout for deterministic comparison

    z_rl = torch.randn(1, 64)
    teacher = torch.randn(1, 8, 64)

    pred1 = rl_token.decode(z_rl, teacher)

    # Modify teacher token at position 3. In shifted input, positions 0..3 are
    # [z_rl, teacher_0, teacher_1, teacher_2], which are unchanged.
    # So output positions 0..3 should be identical.
    teacher_modified = teacher.clone()
    teacher_modified[:, 3, :] = torch.randn(1, 64)

    pred2 = rl_token.decode(z_rl, teacher_modified)

    # Positions 0..3 should be identical (causal mask: they can't attend to position 4+)
    assert torch.allclose(pred1[:, :4, :], pred2[:, :4, :], atol=1e-5)
    # Position 4+ may differ (since shifted input at position 4 = teacher[:, 3] which changed)
    # We don't assert they differ (they might or might not depending on weights)


# ---------------------------------------------------------------------------
# perceiver arch (openpi-RLT style)
# ---------------------------------------------------------------------------

def _perceiver(**kwargs):
    defaults = dict(
        token_dim=64, nhead=4, num_enc_layers=1, num_dec_layers=1, ff_dim=128, arch="perceiver", seq_len=8
    )
    return RLTokenModule(**{**defaults, **kwargs})


def test_perceiver_shapes():
    rl_token = _perceiver(num_rl_tokens=2)
    tokens = torch.randn(3, 8, 64)
    assert rl_token.encode_multi(tokens).shape == (3, 2, 64)
    assert rl_token.encode(tokens).shape == (3, 64)
    assert rl_token.decode(rl_token.encode_multi(tokens)).shape == (3, 8, 64)
    assert rl_token.reconstruction_loss(tokens).shape == ()


def test_perceiver_requires_seq_len_and_checks_it():
    with pytest.raises(ValueError, match="seq_len"):
        RLTokenModule(token_dim=64, nhead=4, arch="perceiver")
    with pytest.raises(ValueError, match="built for 8 context tokens but got 12"):
        _perceiver().encode(torch.randn(2, 12, 64))


def test_perceiver_decoder_reads_only_z_rl():
    """No teacher forcing: the reconstruction is a function of z_rl alone."""
    rl_token = _perceiver().eval()
    with pytest.raises(ValueError, match="teacher_tokens"):
        rl_token.decode(torch.randn(2, 64), torch.randn(2, 8, 64))
    z_rl = torch.randn(1, 64)
    torch.testing.assert_close(rl_token.decode(z_rl.expand(2, -1))[0], rl_token.decode(z_rl.expand(2, -1))[1])
    assert not torch.allclose(rl_token.decode(z_rl), rl_token.decode(torch.randn(1, 64)))


def test_perceiver_learns_to_use_z_rl():
    """Tokens that repeat one per-sample vector: an AR decoder can copy its previous input,
    the perceiver decoder can only get them right by encoding that vector into z_rl."""
    torch.manual_seed(0)
    rl_token = _perceiver(token_dim=32, ff_dim=64)
    tokens = torch.randn(16, 1, 32).expand(-1, 8, -1).contiguous()
    optimizer = torch.optim.Adam(rl_token.parameters(), lr=3e-3)
    for _ in range(200):
        loss = rl_token.reconstruction_loss(tokens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    diag = rl_token.reconstruction_diagnostics(tokens)
    assert diag["z_rl_usage"] > 0.5
    assert diag["z_rl_cos"] < 0.9
    assert rl_token.training  # diagnostics restore the train/eval mode


def test_perceiver_encoder_round_trip_and_arch_detection():
    full = _perceiver(num_rl_tokens=2)
    state = full.state_dict()
    assert rl_token_arch_from_state_dict(state) == {"arch": "perceiver", "seq_len": 8, "num_rl_tokens": 2}
    encoder_only, skipped = filter_encoder_only(state)
    assert "decoder.query" in skipped and "encoder.context_pos" in encoder_only

    deploy = _perceiver(num_rl_tokens=2, inference_only=True)
    load_rl_token_encoder(deploy, state)
    tokens = torch.randn(2, 8, 64)
    torch.testing.assert_close(deploy.encode(tokens), full.encode(tokens))


def test_load_rl_token_encoder_rejects_other_arch(rl_token):
    assert rl_token_arch_from_state_dict(rl_token.state_dict())["arch"] == "ar"
    with pytest.raises(RuntimeError, match="does not match"):
        load_rl_token_encoder(_perceiver(inference_only=True), rl_token.state_dict())
