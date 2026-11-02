# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified in this project from ASFTTrainer in unsloth/trainer.py at upstream
# commit 5dc8ee0b3597c5c78de66db132499b431a0aa8c2. Changes: extract the class,
# import the local loss with checkpointed sequence-chunk KL, request full logits
# before parent initialization when ASFT is enabled, create frozen references
# only when KL is active, support detached update-window normalization with RNG
# replay, and adapt types, documentation, and warnings.

"""Local ASFT trainer.

A thin subclass of unsloth's ``UnslothTrainer`` that routes ``compute_loss``
through the vendored ASFT objective in :mod:`src.train.asft` without requiring
an Unsloth fork.

It mirrors ``ASFTTrainer`` from unslothai/unsloth PR #4239, adapted for the
current TRL/Transformers stack and the vendored loss module. Local changes vs.
the PR:

* ``kl_weight == 0.0`` no longer allocates a frozen reference copy because the
  KL term is inactive.
* When ASFT is enabled we force ``UNSLOTH_RETURN_LOGITS=1`` so the model
  materialises full per-position logits. Setting this before ``super().__init__``
  also makes unsloth's ``SFTTrainer`` init disable padding-free / sample-packing
  (see ``unsloth.trainer._patch_sft_trainer_auto_packing``), which would
  otherwise flatten the batch and break the ``(B, T)`` assumptions in
  ``compute_asft_loss``.
* ``update_weights_v1`` measures a complete prefetched optimizer-update window,
  then replays each stochastic forward with the same RNG state.

With ``asft_enabled=False`` (the default) this class is behaviourally identical
to ``UnslothTrainer``: no environment change, and ``compute_loss`` delegates to
the parent.
"""

from __future__ import annotations

import os
import warnings
from copy import deepcopy
from typing import Literal, Optional

import torch

from unsloth.trainer import UnslothTrainer

from src.train.asft import (
    ASFTStreamingConfig,
    _capture_rng_state,
    _restore_rng_state,
    compute_asft_loss,
    resolve_effective_mode,
)

__all__ = ["ASFTTrainer", "ASFTStreamingConfig"]

# The vendored loss accepts the full 4-mode set (incl. the "sft" debug alias);
# qlora's config exposes only the {"dft", "sft+kl", "asft"} subset.
_AsftMode = Literal["sft", "dft", "sft+kl", "asft"]
_KlDirection = Literal["forward", "reverse"]
_ReferencePolicy = Literal["disable_adapter", "frozen_copy"]
_NormalizationSemantics = Literal[
    "token_count_v0", "microbatch_weights_v0", "update_weights_v1"
]


class ASFTTrainer(UnslothTrainer):
    """``UnslothTrainer`` + an optional Anchored-SFT loss path."""

    def __init__(
        self,
        *args,
        asft_enabled: bool = False,
        asft_mode: _AsftMode = "asft",
        kl_weight: float = 0.0,
        kl_direction: _KlDirection = "forward",
        reference_policy: _ReferencePolicy = "disable_adapter",
        asft_streaming: Optional[ASFTStreamingConfig] = None,
        normalization_semantics: _NormalizationSemantics = "token_count_v0",
        **kwargs,
    ):
        if asft_enabled and normalization_semantics not in (
            "token_count_v0", "microbatch_weights_v0", "update_weights_v1"
        ):
            raise ValueError(f"Unknown normalization_semantics: {normalization_semantics}")
        if asft_enabled:
            # ASFT reads outputs.logits from a label-free forward; force unsloth to
            # return full logits (and, as a side effect, disable padding-free /
            # packing at SFTTrainer init time).
            os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        super().__init__(*args, **kwargs)

        self.asft_enabled: bool = asft_enabled
        self.asft_mode: _AsftMode = asft_mode
        self.kl_weight: float = kl_weight
        self.kl_direction: _KlDirection = kl_direction
        self.reference_policy: _ReferencePolicy = reference_policy
        self.asft_streaming: ASFTStreamingConfig = (
            asft_streaming or ASFTStreamingConfig()
        )
        self.normalization_semantics = normalization_semantics
        # Lazily created only if a frozen reference copy is actually required.
        self._asft_original_model = None
        self._asft_replay_states = []
        self._asft_terminal_rng_state = None
        self._asft_active_replay_state = None
        self._asft_update_denominator = None

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """Measure one complete update window, including a final partial window."""
        batch_samples, num_items_in_batch = super().get_batch_samples(
            epoch_iterator, num_batches, device
        )
        if not (
            self.asft_enabled
            and self.normalization_semantics == "update_weights_v1"
        ):
            return batch_samples, num_items_in_batch

        self._asft_replay_states = []
        self._asft_update_denominator = None
        if not batch_samples:
            self._asft_terminal_rng_state = _capture_rng_state()
            return batch_samples, num_items_in_batch

        model = getattr(self, "model_wrapped", None)
        if model is None:
            model = self.model
        model.train()
        denominator = None
        with torch.no_grad():
            for batch in batch_samples:
                self._asft_replay_states.append(_capture_rng_state())
                inputs = dict(self._prepare_inputs(batch))
                received = []
                compute_asft_loss(
                    model=model,
                    inputs=inputs,
                    asft_mode="dft",
                    kl_weight=0.0,
                    streaming_config=self.asft_streaming,
                    normalization_semantics="token_count_v0",
                    denominator_stats=received.append,
                )
                if len(received) != 1:
                    raise RuntimeError("ASFT denominator pass did not emit exactly one statistic")
                value = received[0].detach().to(dtype=torch.float64)
                denominator = value if denominator is None else denominator + value

        self._asft_terminal_rng_state = _capture_rng_state()
        if hasattr(self.accelerator, "reduce"):
            denominator = self.accelerator.reduce(denominator, reduction="sum")
        elif int(getattr(self.accelerator, "num_processes", 1)) > 1:
            raise RuntimeError("update_weights_v1 requires a distributed sum reduction")
        self._asft_update_denominator = denominator.detach()
        return batch_samples, num_items_in_batch

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Replay measured RNG state while retaining the parent backward path."""
        if not (
            self.asft_enabled
            and self.normalization_semantics == "update_weights_v1"
        ):
            return super().training_step(model, inputs, num_items_in_batch)
        if not self._asft_replay_states or self._asft_update_denominator is None:
            raise RuntimeError(
                "update_weights_v1 training_step requires a prefetched get_batch_samples window"
            )
        if (
            (not self.model_accepts_loss_kwargs or num_items_in_batch is None)
            and self.compute_loss_func is None
            and self.current_gradient_accumulation_steps != 1
        ):
            # This parent path divides by the current window length before
            # Accelerate's backward division. The source scaling contract only
            # compensates Accelerate, so do not silently train a different loss.
            raise RuntimeError(
                "update_weights_v1 requires Trainer item-count loss handling; "
                "the parent would apply an additional accumulation division"
            )

        self._asft_active_replay_state = self._asft_replay_states.pop(0)
        loss_scale = self._update_loss_scale()
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        finally:
            terminal_state = self._asft_terminal_rng_state
            self._asft_active_replay_state = None
            if terminal_state is not None:
                _restore_rng_state(terminal_state)
        return loss / loss_scale

    def _update_loss_scale(self) -> int:
        # Accelerate divides the backward loss by accumulation steps, and DDP
        # averages gradients over processes. Undo both for a globally summed
        # numerator over the globally summed denominator.
        accumulation_steps = int(
            getattr(self.accelerator, "gradient_accumulation_steps", 0)
            or self.args.gradient_accumulation_steps
        )
        process_count = int(getattr(self.accelerator, "num_processes", 1))
        return accumulation_steps * process_count

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ASFT disabled -> behave exactly like the parent trainer.
        if not self.asft_enabled:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

        if self.normalization_semantics == "update_weights_v1":
            denominator = self._asft_update_denominator
            replay_state = self._asft_active_replay_state
            if denominator is None or replay_state is None:
                raise RuntimeError(
                    "update_weights_v1 compute_loss requires the get_batch_samples/training_step seam"
                )
        else:
            denominator = None

        inputs = dict(inputs)
        num_items_in_batch = kwargs.get("num_items_in_batch")
        if num_items_in_batch is not None:
            inputs["num_items_in_batch"] = num_items_in_batch

        _, kl_active = resolve_effective_mode(self.asft_mode, self.kl_weight)

        # Allocate a frozen reference copy only when KL is active
        # and a frozen copy is the only way to obtain a reference distribution.
        if kl_active:
            needs_frozen_copy = self.reference_policy == "frozen_copy" or (
                self.reference_policy == "disable_adapter"
                and not hasattr(model, "disable_adapter")
            )
            if needs_frozen_copy and self._asft_original_model is None:
                if self.reference_policy == "frozen_copy":
                    warnings.warn(
                        "ASFT: creating a frozen copy of the model for the KL "
                        "reference. This doubles VRAM; prefer 'disable_adapter' "
                        "for LoRA.",
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        "ASFT: 'disable_adapter' unavailable on this model; "
                        "falling back to a frozen copy for the KL reference. "
                        "This doubles VRAM.",
                        stacklevel=2,
                    )
                self._asft_original_model = deepcopy(model)
                self._asft_original_model.eval()
                self._asft_original_model.requires_grad_(False)

        if self.normalization_semantics == "update_weights_v1":
            _restore_rng_state(replay_state)
        result = compute_asft_loss(
            model=model,
            inputs=inputs,
            asft_mode=self.asft_mode,
            kl_weight=self.kl_weight,
            kl_direction=self.kl_direction,
            reference_policy=self.reference_policy,
            streaming_config=self.asft_streaming,
            original_model=self._asft_original_model,
            normalization_semantics=self.normalization_semantics,
            update_weight_sum=denominator[0] if denominator is not None else None,
            update_token_count=denominator[1] if denominator is not None else None,
            return_outputs=return_outputs,
        )
        if self.normalization_semantics != "update_weights_v1":
            return result

        loss_scale = self._update_loss_scale()
        if return_outputs:
            loss, outputs = result
            return loss * loss_scale, outputs
        return result * loss_scale
