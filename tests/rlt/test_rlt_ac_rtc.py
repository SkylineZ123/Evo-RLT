from __future__ import annotations

from collections import deque
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from evo_rlt.adapters.lerobot.policies.action_modifier import RLTActionModifier, RLTStepMetadata
from evo_rlt.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from evo_rlt.core.actor import ChunkActor
from evo_rlt.core.phase_controller import PhaseController
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker


def _make_policy(chunk: torch.Tensor | None = None) -> ChunkACPolicy:
    policy = object.__new__(ChunkACPolicy)
    rtc_config = RTCConfig(enabled=True, execution_horizon=2)
    policy._rtc_config = rtc_config
    policy._vla_rtc_config = rtc_config
    policy._active_pi05_rtc_config = rtc_config
    policy._rtc_action_queue = ActionQueue(rtc_config)
    policy._rtc_latency_tracker = LatencyTracker()
    policy._rtc_fps = 10.0
    policy._rtc_action_queue_size_to_get_new_actions = 1
    policy._rtc_worker = None
    policy._rtc_worker_error = None
    policy._rtc_generation = 0
    policy._rtc_lock = Lock()
    policy._rtc_inference_lock = Lock()
    policy._rtc_step_metadata = deque()
    policy._rtc_selected_step_metadata = deque()
    policy.predict_calls = []
    if chunk is None:
        chunk = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    def predict_action_chunk(batch, **kwargs):
        policy.predict_calls.append((batch, kwargs))
        metadata = [RLTStepMetadata(phase=1.0, source_type=1.0) for _ in range(chunk.shape[1])]
        policy._ensure_modifier()._step_metadata.extend(metadata)
        return chunk.clone()

    object.__setattr__(policy, "predict_action_chunk", predict_action_chunk)
    object.__setattr__(policy, "_ensure_modifier", lambda: policy.modifier)
    policy.modifier = type("_Modifier", (), {"_step_metadata": deque()})()
    return policy


def test_configure_rtc_accepts_vla_rtc_config_and_switches_by_phase() -> None:
    policy = object.__new__(ChunkACPolicy)
    policy.config = SimpleNamespace(chunk_length=10, action_dim=3, proprio_dim=3, compile_model=False)
    policy._rtc_lock = Lock()
    policy._rtc_inference_lock = Lock()
    policy._rtc_step_metadata = deque()
    policy._rtc_selected_step_metadata = deque()
    policy.vla_seen_configs = []

    class FakePi05:
        def __init__(self) -> None:
            self.config = SimpleNamespace(rtc_config=None)
            self.init_calls: list[RTCConfig] = []

        def init_rtc_processor(self) -> None:
            self.init_calls.append(self.config.rtc_config)

        def predict_action_chunk(self, batch, **kwargs):
            policy.vla_seen_configs.append(self.config.rtc_config)
            return torch.zeros(1, 4, 3)

    fake_pi05 = FakePi05()
    policy._rl_token_policy = SimpleNamespace(_pi05=fake_pi05)
    policy._prefix_capture = SimpleNamespace(consume=lambda: torch.zeros(1, 1, 1))
    policy.modifier = SimpleNamespace(
        is_rl_phase=False,
        compute_chunk=lambda vla_chunk, proprio, prefix_tokens: vla_chunk,
    )
    object.__setattr__(policy, "eval", lambda: policy)
    object.__setattr__(policy, "_ensure_modifier", lambda: policy.modifier)

    rlt_rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    vla_rtc_config = RTCConfig(enabled=True, execution_horizon=25)
    policy.configure_rtc(rlt_rtc_config, fps=30, vla_rtc_config=vla_rtc_config)

    policy.predict_action_chunk({"observation.state": torch.ones(1, 3)})
    policy.modifier.is_rl_phase = True
    policy.predict_action_chunk({"observation.state": torch.ones(1, 3)})

    assert fake_pi05.init_calls == [rlt_rtc_config, vla_rtc_config, rlt_rtc_config]
    assert policy.vla_seen_configs == [vla_rtc_config, rlt_rtc_config]


def test_prepare_rtc_request_uses_leftover_and_latency() -> None:
    policy = _make_policy()
    old_actions = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    obs = torch.ones(1, 3)
    policy._rtc_action_queue.merge(old_actions, old_actions, real_delay=0)
    policy._rtc_action_queue.get()
    policy._rtc_latency_tracker.add(0.21)

    batch, prev_actions, inference_delay, action_index, _, _ = policy._prepare_rtc_request(
        {"observation.state": obs, "task": ["insert"]}
    )

    assert action_index == 1
    assert inference_delay == 3
    assert torch.equal(prev_actions, old_actions[1:])
    assert batch["task"] == ["insert"]
    assert torch.equal(batch["observation.state"], obs)
    assert batch["observation.state"] is not obs


def test_predict_and_merge_rtc_chunk_uses_actual_consumed_delay() -> None:
    policy = _make_policy()
    old_actions = torch.full((4, 3), -1.0)
    policy._rtc_action_queue.merge(old_actions, old_actions, real_delay=0)
    action_index_before_inference = policy._rtc_action_queue.get_action_index()
    policy._rtc_action_queue.get()
    policy._rtc_action_queue.get()

    policy._predict_and_merge_rtc_chunk(
        {"observation.state": torch.ones(1, 3)},
        prev_actions=old_actions[2:],
        inference_delay=1,
        action_index_before_inference=action_index_before_inference,
        request_start_time=0.0,
        generation=0,
    )

    assert policy._rtc_action_queue.qsize() == 2
    assert torch.equal(policy._rtc_action_queue.get(), torch.tensor([6.0, 7.0, 8.0]))
    assert len(policy._rtc_step_metadata) == 2
    _, kwargs = policy.predict_calls[0]
    assert kwargs["inference_delay"] == 1
    assert torch.equal(kwargs["prev_chunk_left_over"], old_actions[2:])


def test_select_action_rtc_returns_matching_metadata() -> None:
    policy = _make_policy()
    policy._rtc_latency_tracker.add(1.0)

    action = policy._select_action_rtc({"observation.state": torch.ones(1, 3)})
    metadata = policy.pop_step_metadata()

    assert action.shape == (1, 3)
    assert torch.equal(action, torch.tensor([[0.0, 1.0, 2.0]]))
    assert metadata == RLTStepMetadata(phase=1.0, source_type=1.0)
    assert policy._rtc_action_queue.qsize() == 3


def test_reset_rtc_runtime_invalidates_inflight_request_without_joining() -> None:
    policy = _make_policy()
    request = policy._prepare_rtc_request({"observation.state": torch.ones(1, 3)})
    policy._reset_rtc_runtime()

    policy._predict_and_merge_rtc_chunk(*request)

    assert policy._rtc_action_queue.qsize() == 0
    assert len(policy._rtc_step_metadata) == 0
    assert policy.predict_calls == []


def _make_rl_phase_policy(actor_out: list[float], low: list[float], high: list[float], rtc: bool) -> ChunkACPolicy:
    """ChunkACPolicy in RL phase with the real predict_action_chunk + RLTActionModifier; only pi0.5 is faked."""
    C, A, P, Z = 4, 3, 2, 6
    actor = ChunkActor(state_dim=Z + P, chunk_dim=C * A, hidden_dim=16, num_layers=1)
    with torch.no_grad():
        actor.net[-1].weight.zero_()
        actor.net[-1].bias.copy_(torch.tensor(actor_out).repeat(C))
    actor.set_action_bounds(torch.tensor(low), torch.tensor(high))

    class _StubRLToken(torch.nn.Module):
        def encode(self, prefix_tokens):
            return torch.zeros(prefix_tokens.shape[0], Z)

    phase = PhaseController(mode="manual")
    phase.trigger_critical()
    mod = RLTActionModifier(
        rl_token=_StubRLToken(), actor=actor, phase_ctrl=phase, chunk_length=C, action_dim=A, proprio_dim=P,
    )
    if rtc:
        policy = _make_policy()
        del policy.__dict__["predict_action_chunk"]  # use the real one
        policy._rtc_latency_tracker.add(0.1)
    else:
        policy = object.__new__(ChunkACPolicy)
        policy._rtc_config = None
    object.__setattr__(policy, "modifier", mod)
    object.__setattr__(policy, "_ensure_modifier", lambda: mod)
    object.__setattr__(policy, "eval", lambda: policy)
    policy.config = SimpleNamespace(chunk_length=C, action_dim=A, proprio_dim=P, compile_model=False)
    fake_pi05 = SimpleNamespace(
        config=SimpleNamespace(rtc_config=None),
        init_rtc_processor=lambda: None,
        predict_action_chunk=lambda batch, **kwargs: torch.zeros(1, 2 * C, A),
    )
    policy._rl_token_policy = SimpleNamespace(_pi05=fake_pi05)
    policy._prefix_capture = SimpleNamespace(consume=lambda: torch.zeros(1, 5, 8))
    return policy


@pytest.mark.parametrize("rtc", [True, False])
def test_rl_phase_select_action_passes_actor_output_beyond_unit_range(rtc: bool) -> None:
    policy = _make_rl_phase_policy([1.8, -1.5, 0.5], low=[-2.0] * 3, high=[2.0] * 3, rtc=rtc)
    action = policy.select_action({"observation.state": torch.zeros(1, 2)})
    assert torch.allclose(action, torch.tensor([[1.8, -1.5, 0.5]]))


@pytest.mark.parametrize("rtc", [True, False])
def test_rl_phase_select_action_clamps_to_actor_bounds(rtc: bool) -> None:
    policy = _make_rl_phase_policy([5.0, -5.0, 5.0], low=[-1.0, -2.0, -3.0], high=[1.5, 2.5, 3.5], rtc=rtc)
    action = policy.select_action({"observation.state": torch.zeros(1, 2)})
    assert torch.equal(action, torch.tensor([[1.5, -2.0, 3.5]]))


def test_compiled_policy_rejects_rtc() -> None:
    policy = object.__new__(ChunkACPolicy)
    policy.config = SimpleNamespace(chunk_length=10, action_dim=3, proprio_dim=3, compile_model=True)

    with pytest.raises(ValueError, match="RTC runs pi0.5 on a background thread"):
        policy.configure_rtc(RTCConfig(enabled=True, execution_horizon=10), fps=30)


def test_compile_vla_wraps_the_pi05_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    def sample_actions(*args, **kwargs):
        return torch.zeros(1)

    compiled = []

    def fake_compile(fn, mode):
        compiled.append((fn, mode))
        return lambda *args, **kwargs: fn(*args, **kwargs)

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = SimpleNamespace(sample_actions=sample_actions)
    policy = object.__new__(ChunkACPolicy)
    policy.config = SimpleNamespace(compile_mode="reduce-overhead")
    policy._rl_token_policy = SimpleNamespace(_pi05=SimpleNamespace(model=model))

    policy._compile_vla()

    assert compiled == [(sample_actions, "reduce-overhead")]
    assert model.sample_actions is not sample_actions


def test_compiled_predict_action_chunk_copies_graph_outputs() -> None:
    """CUDA-graph outputs are overwritten by the next replay, so nothing downstream may alias them."""
    graph_actions = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    graph_prefix = torch.ones(1, 5, 8)
    seen_prefix = []
    policy = object.__new__(ChunkACPolicy)
    policy.config = SimpleNamespace(chunk_length=4, action_dim=3, proprio_dim=3, compile_model=True)
    policy._rtc_config = None
    policy._rl_token_policy = SimpleNamespace(_pi05=SimpleNamespace(predict_action_chunk=lambda batch, **kw: graph_actions))
    policy._prefix_capture = SimpleNamespace(consume=lambda: graph_prefix)
    policy.modifier = SimpleNamespace(
        compute_chunk=lambda vla_chunk, proprio, prefix_tokens: seen_prefix.append(prefix_tokens) or vla_chunk
    )
    object.__setattr__(policy, "eval", lambda: policy)
    object.__setattr__(policy, "_ensure_modifier", lambda: policy.modifier)

    chunk = policy.predict_action_chunk({"observation.state": torch.zeros(1, 3)})

    assert torch.equal(chunk, graph_actions)
    assert chunk.data_ptr() != graph_actions.data_ptr()
    assert seen_prefix[0].data_ptr() != graph_prefix.data_ptr()
