"""Skip random weight init when pi0.5 is built only to be overwritten by a checkpoint.

LeRobot's ``PI05Policy.from_pretrained`` first runs ``cls(config)`` — which builds
PaliGemma + the Gemma action expert (~3.6B params) and randomly initializes every
weight on CPU in float32 (torch ``reset_parameters`` + transformers
``initialize_weights``) — and only then loads ``model.safetensors`` over it. The
random init is pure overhead and dominates VLA load time.

``install_pi05_fast_load()`` wraps ``PI05Policy.from_pretrained`` so construction
runs under ``skip_weight_init()``. Keys the checkpoint does not provide are
re-initialized afterwards, so their values match the old behaviour.
"""

from __future__ import annotations

import functools
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

log = logging.getLogger(__name__)


@contextmanager
def skip_weight_init() -> Iterator[None]:
    """Make torch/transformers weight init a no-op; tensors keep their allocated (empty) values.

    Unlike ``transformers.initialization.no_init_weights`` this keeps
    ``PreTrainedModel.tie_weights`` active, so lm_head/embed_tokens stay tied.
    """
    from transformers import initialization as hf_init
    from transformers.modeling_utils import PreTrainedModel

    def _skip(tensor=None, *args, **kwargs):
        return tensor

    originals: list[tuple[object, str, object]] = []
    try:
        for module_name in hf_init.TORCH_MODULES_TO_PATCH:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            for func_name in hf_init.TORCH_INIT_FUNCTIONS:
                if hasattr(module, func_name):
                    originals.append((module, func_name, getattr(module, func_name)))
                    setattr(module, func_name, _skip)
        originals.append((PreTrainedModel, "initialize_weights", PreTrainedModel.initialize_weights))
        PreTrainedModel.initialize_weights = lambda self: None
        yield
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def _reinit_missing(model: nn.Module, missing_keys: list[str]) -> None:
    """Run the normal init for parameters the checkpoint did not provide."""
    param_names = dict(model.named_parameters(remove_duplicate=False))
    missing = [k for k in missing_keys if k in param_names]
    if not missing:
        return
    by_module: dict[str, set[str]] = {}
    for key in missing:
        module_name, _, param_name = key.rpartition(".")
        by_module.setdefault(module_name, set()).add(param_name)
    for module_name, names in by_module.items():
        module = model.get_submodule(module_name)
        if not hasattr(module, "reset_parameters"):
            log.warning("no reset_parameters() on %s; missing params %s left uninitialized", module_name, sorted(names))
            continue
        # reset_parameters() touches every direct param; keep the ones that were loaded.
        loaded = {n: p.detach().clone() for n, p in module.named_parameters(recurse=False) if n not in names}
        with torch.no_grad():
            module.reset_parameters()
            for n, value in loaded.items():
                getattr(module, n).copy_(value)
    log.warning("re-initialized %d pi0.5 params missing from the checkpoint", len(missing))


def install_pi05_fast_load() -> None:
    """Patch ``PI05Policy.from_pretrained`` to skip the throwaway random init. Idempotent."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    if getattr(PI05Policy.from_pretrained, "_evo_rlt_fast_load", False):
        return
    original = PI05Policy.from_pretrained.__func__

    @functools.wraps(original)
    def from_pretrained(cls, *args, **kwargs):
        # LeRobot swallows load failures and returns the constructed model; record the
        # load_state_dict result so an unloaded (never initialized) model cannot slip through.
        results = []
        had_own = "load_state_dict" in cls.__dict__
        base_load = cls.load_state_dict

        def recording_load_state_dict(self, *a, **kw):
            result = base_load(self, *a, **kw)
            results.append(result)
            return result

        cls.load_state_dict = recording_load_state_dict
        try:
            with skip_weight_init():
                policy = original(cls, *args, **kwargs)
        finally:
            if had_own:
                cls.load_state_dict = base_load
            else:
                del cls.load_state_dict

        if not results:
            raise RuntimeError("PI05Policy.from_pretrained did not load any weights (see LeRobot warning above)")
        _reinit_missing(policy, list(results[-1].missing_keys))
        return policy

    from_pretrained._evo_rlt_fast_load = True
    PI05Policy.from_pretrained = classmethod(from_pretrained)
