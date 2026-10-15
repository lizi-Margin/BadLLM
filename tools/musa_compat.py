"""Compatibility shims for torch_musa 1.3.0 on the MTT S4000 stack.

The special MUSA build (torch 2.2.0 + torch_musa 1.3.0) is intentionally kept
untouched, so any `torch.musa` APIs that transformers / accelerate expect but
torch_musa 1.3.0 does not provide are added here instead of patching the MUSA
packages themselves.

Usage: ``import musa_compat`` (before or right after ``import torch_musa``).
Importing is idempotent and a no-op when no MUSA device is available.
"""
import math
import sys
import types

import torch
import torch_musa  # noqa: F401  (registers the torch.musa namespace)

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    if not torch.musa.is_available():
        return

    # 1) transformers >= 4.46 calls torch.musa.is_bf16_supported() (the analog of
    #    torch.cuda.is_bf16_supported), which torch_musa 1.3.0 does not expose.
    #    bf16 forward/backward was verified working on the MTT S4000, so report True.
    if not hasattr(torch.musa, "is_bf16_supported"):
        torch.musa.is_bf16_supported = lambda: True

    # 2) accelerate's torch.compile path sets torch.backends.musa.matmul.allow_tf32,
    #    but torch_musa 1.3.0 has no torch.backends.musa namespace. The real TF32
    #    switch on MUSA is torch.backends.mudnn.allow_tf32, so wire it through.
    if not hasattr(torch.backends, "musa"):

        class _MatmulNamespace:
            @property
            def allow_tf32(self):
                return torch.backends.mudnn.allow_tf32

            @allow_tf32.setter
            def allow_tf32(self, value):
                torch.backends.mudnn.allow_tf32 = value

        class _MusaBackendNamespace:
            matmul = _MatmulNamespace()

        torch.backends.musa = _MusaBackendNamespace()

    # 3) accelerate's test utilities call torch.musa.reset_max_memory_allocated();
    #    alias it to the existing reset_peak_memory_stats.
    if not hasattr(torch.musa, "reset_max_memory_allocated"):
        torch.musa.reset_max_memory_allocated = torch.musa.reset_peak_memory_stats

    # 4) MUSA's torch.cat kernel does not implement PyTorch's legacy special case
    #    for 1-D empty tensors mixed with higher-rank tensors
    #    (e.g. torch.cat([torch.tensor([]), x4d], dim=-2)), which transformers' KV
    #    cache lazy initialization relies on. Emulate the native semantics by
    #    dropping the 1-D empty tensors before delegating to the original cat.
    #    The guard is narrow: all other cat calls pass through untouched.
    if not getattr(torch.cat, "_musa_1d_empty_shim", False):
        _orig_cat = torch.cat

        def _cat(tensors, dim=0, *args, **kwargs):
            filtered = None
            empty_result = False
            try:
                has_1d_empty = any(t.dim() == 1 and t.numel() == 0 for t in tensors)
                if has_1d_empty and any(t.dim() > 1 for t in tensors):
                    filtered = [t for t in tensors if not (t.dim() == 1 and t.numel() == 0)]
                    if not filtered:
                        filtered = None
                elif (
                    has_1d_empty
                    and dim in (-1, 0)
                    and not args
                    and not kwargs
                    and all(t.dim() == 1 and t.numel() == 0 for t in tensors)
                ):
                    # every tensor is a 1-D empty tensor: native torch yields a fresh
                    # empty 1-D tensor of the promoted dtype
                    empty_result = True
            except Exception:  # noqa: BLE001  (never break normal cat calls)
                filtered = None
                empty_result = False
            if filtered is not None:
                return _orig_cat(filtered, dim, *args, **kwargs)
            if empty_result:
                return torch.empty(0, dtype=torch.result_type(*tensors), device=tensors[0].device)
            return _orig_cat(tensors, dim, *args, **kwargs)

        _cat._musa_1d_empty_shim = True
        torch.cat = _cat
        if hasattr(torch, "concat"):
            torch.concat = _cat

    # 5) MUSA's SDPA implementation strictly requires `scale` to equal
    #    1/sqrt(head_dim) bit-for-bit, computed as 1.0/math.sqrt(d) in double.
    #    HF models such as Qwen3 derive it as head_dim**-0.5 (pow), which differs
    #    by 1 ULP for non-power-of-two head dims (e.g. 128) and is rejected.
    #    Normalize mathematically-identical scales to the exact accepted value.
    if not getattr(torch.nn.functional.scaled_dot_product_attention, "_musa_scale_shim", False):
        _orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def _sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
            if scale is not None:
                try:
                    exact = 1.0 / math.sqrt(query.size(-1))
                    if scale != exact and math.isclose(scale, exact, rel_tol=1e-9):
                        scale = exact
                except Exception:  # noqa: BLE001  (never break normal sdpa calls)
                    pass
            return _orig_sdpa(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
            )

        _sdpa._musa_scale_shim = True
        torch.nn.functional.scaled_dot_product_attention = _sdpa

    # 6) transformers' checkpoint resume calls torch.musa.random.set_rng_state()
    #    (the torch.cuda.random-style namespace), which torch_musa 1.3.0 does
    #    not provide; the RNG functions themselves exist directly on torch.musa.
    if not hasattr(torch.musa, "random"):
        torch.musa.random = types.SimpleNamespace(
            get_rng_state=torch.musa.get_rng_state,
            set_rng_state=torch.musa.set_rng_state,
            get_rng_state_all=torch.musa.get_rng_state_all,
            set_rng_state_all=torch.musa.set_rng_state_all,
            manual_seed=torch.musa.manual_seed,
            manual_seed_all=torch.musa.manual_seed_all,
            initial_seed=torch.musa.initial_seed,
        )


def patch_transformers_torch_load_gate() -> bool:
    """Lift transformers' torch<2.6 ``torch.load`` gate (CVE-2025-32434 guard).

    transformers >= 4.56 calls ``check_torch_load_is_safe()`` before every
    ``torch.load``; with torch 2.2.0 (the MUSA build cannot be upgraded) it
    always raises, so ``--resume`` from a checkpoint is impossible. This server
    only ever loads checkpoints it has just written itself, so the guard is
    lifted here. Call after ``import transformers``; idempotent.
    """
    try:
        import transformers.utils.import_utils as tu
    except ImportError:
        return False
    original = getattr(tu, "check_torch_load_is_safe", None)
    if original is None:
        return False
    if getattr(original, "_musa_gate_lifted", False):
        return True

    def _lifted(*args, **kwargs):
        return None

    _lifted._musa_gate_lifted = True
    tu.check_torch_load_is_safe = _lifted
    # Modules that did ``from ... import check_torch_load_is_safe`` hold their
    # own binding; rebind every module still pointing at the original.
    for module in list(sys.modules.values()):
        try:
            if getattr(module, "check_torch_load_is_safe", None) is original:
                module.check_torch_load_is_safe = _lifted
        except Exception:  # noqa: BLE001  (never break other modules)
            pass
    return True


apply()
