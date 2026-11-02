from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from loguru import logger


_DEFAULT_IGNORE_INDEX = -100
_LOGGED_MESSAGES: set[str] = set()
_ORIGINAL_COMPUTE_FUSED_CE_LOSS: Any | None = None
_ORIGINAL_UNSLOTH_FUSED_CE_LOSS: Any | None = None


def _log_once(message: str) -> None:
    if message in _LOGGED_MESSAGES:
        return
    logger.info(message)
    _LOGGED_MESSAGES.add(message)


# --------------------------------------------------------------------------------------
# Inner loss function for Unsloth's chunked UnslothFusedLoss. Unsloth differentiates this
# under torch.func.grad_and_value and compiles the per-chunk accumulator with torch.compile.
# --------------------------------------------------------------------------------------
def compute_fused_dft_loss(
    hidden_states: Any,
    lm_head_weight: Any,
    lm_head_bias: Any,
    labels: Any,
    n_items: Any | None = None,
    scaling: Any | None = None,
    shift_labels: bool = True,
    **kwargs: Any,
) -> tuple[Any, tuple[Any]]:
    """Dense per-chunk DFT loss. Differentiated by Unsloth via torch.func.grad_and_value, so
    reweighting per-token CE by the detached target probability yields the DFT gradient."""

    import torch

    ignore_index = int(kwargs.get("ignore_index", _DEFAULT_IGNORE_INDEX))
    label_smoothing = float(kwargs.get("label_smoothing", 0.0))
    if label_smoothing != 0.0:
        raise ValueError("Fused DFT loss does not support label_smoothing != 0.0")
    device = lm_head_weight.device

    if shift_labels:
        shifted_labels = torch.empty_like(labels, device=device)
        shifted_labels[..., :-1] = labels[..., 1:]
        shifted_labels[..., -1] = ignore_index
        labels = shifted_labels

    vocab_size = lm_head_weight.shape[0]
    flat_labels = labels.reshape(-1).to(device=device).contiguous()
    valid = flat_labels != ignore_index
    logits = torch.nn.functional.linear(
        hidden_states.to(dtype=lm_head_weight.dtype, device=device),
        lm_head_weight,
        lm_head_bias,
    )
    # Prevent ignored rows with overflowed logits from contaminating transforms/backward.
    logits.view(-1, vocab_size).masked_fill_(~valid.unsqueeze(1), 0.0)

    logit_scale_multiply = kwargs.get("logit_scale_multiply", None)
    logit_scale_divide = kwargs.get("logit_scale_divide", None)
    logit_softcapping = kwargs.get("logit_softcapping", None)
    if logit_scale_multiply != 0 and logit_scale_multiply is not None:
        logits = logits * logit_scale_multiply
    if logit_scale_divide != 0 and logit_scale_divide is not None:
        logits = logits / logit_scale_divide
    if logit_softcapping != 0 and logit_softcapping is not None:
        logits = logits / logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * logit_softcapping

    ce = torch.nn.functional.cross_entropy(
        input=logits.view(-1, vocab_size).float().contiguous(),
        target=flat_labels,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=0.0,
    )
    loss = ((-ce).exp().detach() * ce).sum()

    divisor = n_items
    if divisor is None:
        divisor = (flat_labels != ignore_index).sum().clamp_min(1)
    if not torch.is_tensor(divisor):
        divisor = torch.tensor(divisor, dtype=torch.float32, device=device)
    if divisor.numel() != 1:
        divisor = divisor.ravel()[0]
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
    loss = loss / divisor.to(dtype=loss.dtype, device=device)

    scaled_loss = loss * scaling if scaling is not None else loss
    return scaled_loss, (loss.detach(),)


# --------------------------------------------------------------------------------------
# Public seam used by generated model forwards. It always delegates to Unsloth's original
# chunked wrapper; install_fused_dft patches that wrapper's inner compute function to DFT.
# --------------------------------------------------------------------------------------
def unsloth_fused_dft_loss(
    trainer: Any,
    hidden_states: Any,
    lm_head_weight: Any,
    lm_head_bias: Any,
    labels: Any,
    mask: Any | None = None,
    n_items: Any | None = None,
    scaling: Any | None = None,
    target_gb: Any | None = None,
    torch_compile: bool | None = True,
    overwrite: bool | None = False,
    shift_labels: bool = True,
    **kwargs: Any,
) -> Any:
    import torch

    if float(kwargs.get("label_smoothing", 0.0)) != 0.0:
        raise ValueError("Fused DFT loss does not support label_smoothing != 0.0")
    if _ORIGINAL_UNSLOTH_FUSED_CE_LOSS is None:
        raise RuntimeError("install_fused_dft() must be called before using fused DFT")
    if not shift_labels:
        raise ValueError("Unsloth's fused DFT wrapper requires causal shift_labels=True")

    # The stock wrapper always shifts labels itself; forwarding shift_labels through its
    # **kwargs duplicates the accumulator's argument. It also flattens with view().
    # Canonicalize the selected ignore value for its hard-coded shift/mask/count sentinel.
    ignore_index = int(kwargs.get("ignore_index", _DEFAULT_IGNORE_INDEX))
    if ignore_index != -100:
        labels = labels.masked_fill(labels == ignore_index, -100)
        kwargs["ignore_index"] = -100
    labels = labels.contiguous()
    if n_items is not None and not torch.is_tensor(n_items):
        n_items = torch.as_tensor(n_items, dtype=torch.float32, device=lm_head_weight.device)
    if target_gb is None and not os.environ.get("UNSLOTH_CE_LOSS_TARGET_GB"):
        kwargs.setdefault(
            "n_chunks", max(int(os.environ.get("UNSLOTH_CE_LOSS_N_CHUNKS", "16")), 1)
        )

    _log_once("DFT-fused routing through Unsloth's compiled dense chunked loss")
    return _ORIGINAL_UNSLOTH_FUSED_CE_LOSS(
        trainer,
        hidden_states,
        lm_head_weight,
        lm_head_bias,
        labels,
        mask=mask,
        n_items=n_items,
        scaling=scaling,
        target_gb=target_gb,
        torch_compile=torch_compile,
        overwrite=overwrite,
        **kwargs,
    )


def _repatch_generated_modules() -> int:
    """Rebind loss symbols in the loaded Llama namespace and generated Unsloth modules.

    Qwen2/3 fast forwards resolve the public loss in unsloth.models.llama. Generated
    modules can import a stale re-export from unsloth_zoo.loss_utils even after the
    source loss module is patched, so keep this pass both at install and after loading.
    """
    count = 0
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name != "unsloth.models.llama" and "compiled_module" not in name:
            continue
        if getattr(module, "unsloth_fused_ce_loss", None) not in (
            None,
            unsloth_fused_dft_loss,
        ):
            module.unsloth_fused_ce_loss = unsloth_fused_dft_loss
            count += 1
        if getattr(module, "compute_fused_ce_loss", None) not in (
            None,
            compute_fused_dft_loss,
        ):
            module.compute_fused_ce_loss = compute_fused_dft_loss
            count += 1
    if count:
        logger.info("Re-bound fused DFT loss into {} Llama/generated-module symbol(s)", count)
    return count


def install_fused_dft(ignore_index: int = -100) -> None:
    """Patch Unsloth's loss seams to compute fused DFT.

    Replaces the inner CE function with dense DFT and routes model forwards through Unsloth's
    chunked, torch-compiled fused-loss wrapper. Intentionally version-fragile: an import or
    attribute error should fail the run rather than silently train NLL.
    """

    global \
        _DEFAULT_IGNORE_INDEX, \
        _ORIGINAL_COMPUTE_FUSED_CE_LOSS, \
        _ORIGINAL_UNSLOTH_FUSED_CE_LOSS
    _DEFAULT_IGNORE_INDEX = int(ignore_index)
    # Route CausalLM models through the chunked wrapper too (ConditionalGeneration always does).
    os.environ["UNSLOTH_ENABLE_CCE"] = "0"
    os.environ.setdefault("UNSLOTH_CE_LOSS_N_CHUNKS", "16")

    ce_mod = importlib.import_module("unsloth_zoo.fused_losses.cross_entropy_loss")

    if _ORIGINAL_UNSLOTH_FUSED_CE_LOSS is None:
        _ORIGINAL_UNSLOTH_FUSED_CE_LOSS = getattr(ce_mod, "unsloth_fused_ce_loss")
    if _ORIGINAL_COMPUTE_FUSED_CE_LOSS is None:
        _ORIGINAL_COMPUTE_FUSED_CE_LOSS = getattr(ce_mod, "compute_fused_ce_loss")

    setattr(ce_mod, "compute_fused_ce_loss", compute_fused_dft_loss)
    setattr(ce_mod, "unsloth_fused_ce_loss", unsloth_fused_dft_loss)
    # Resolve chunk defaults at our public seam, so the stock wrapper's import-time
    # environment snapshot cannot override an explicit n_chunks argument.
    if hasattr(ce_mod, "N_CHUNKS"):
        ce_mod.N_CHUNKS = None
    _repatch_generated_modules()
    logger.info("Installed compiled dense chunked DFT loss patch into unsloth_zoo")


__all__ = [
    "compute_fused_dft_loss",
    "unsloth_fused_dft_loss",
    "install_fused_dft",
    "_repatch_generated_modules",
]
