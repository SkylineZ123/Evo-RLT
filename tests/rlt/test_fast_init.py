import torch
from torch import nn
from transformers import GemmaConfig, GemmaForCausalLM

from evo_rlt.adapters.lerobot.fast_init import _reinit_missing, skip_weight_init


def _tiny_gemma() -> GemmaForCausalLM:
    cfg = GemmaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
    )
    return GemmaForCausalLM(cfg)


def test_skip_weight_init_consumes_no_rng_and_restores():
    original_normal = torch.nn.init.normal_
    torch.manual_seed(0)
    state = torch.get_rng_state()
    with skip_weight_init():
        nn.Linear(8, 8)
        model = _tiny_gemma()
    assert torch.equal(state, torch.get_rng_state())
    assert torch.nn.init.normal_ is original_normal
    # tie_weights still runs, so lm_head shares storage with embed_tokens.
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    # RoPE buffers are computed in __init__, not by the skipped init.
    assert torch.isfinite(model.model.rotary_emb.inv_freq).all()


def test_reinit_missing_only_touches_missing_params():
    with skip_weight_init():
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    loaded_bias = torch.full((4,), 3.0)
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].bias.copy_(loaded_bias)
        model[1].weight.fill_(float("nan"))
        model[1].bias.fill_(float("nan"))

    _reinit_missing(model, ["1.weight", "1.bias", "not_a_param"])

    assert torch.equal(model[0].weight, torch.ones(4, 4))
    assert torch.equal(model[0].bias, loaded_bias)
    assert torch.isfinite(model[1].weight).all() and torch.isfinite(model[1].bias).all()


def test_reinit_missing_keeps_loaded_sibling_param():
    with skip_weight_init():
        layer = nn.Linear(4, 4)
    with torch.no_grad():
        layer.weight.fill_(float("nan"))
        layer.bias.fill_(2.0)
    _reinit_missing(layer, ["weight"])
    assert torch.isfinite(layer.weight).all()
    assert torch.equal(layer.bias, torch.full((4,), 2.0))
