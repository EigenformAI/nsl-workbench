"""ASFT objective/update-window oracles with CPU and real CUDA CE paths."""

import random
import unittest
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from src.train import asft
from src.train.asft_trainer import ASFTTrainer
from unsloth.trainer import UnslothTrainer


def _cpu_ce(logits, labels, logit_softcapping=0, logit_scaling=0):
    logits = asft.effective_logits(
        logits, logit_softcapping=logit_softcapping, logit_scaling=logit_scaling
    )
    labels = labels.reshape(-1)
    return (
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels, reduction="none"),
        labels != -100,
    )


class _DropoutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # A power-of-two vocabulary avoids testing the upstream CE kernel's
        # softcap treatment of padded vocabulary lanes instead of ASFT semantics.
        self.embedding = torch.nn.Embedding(8, 5)
        self.projection = torch.nn.Linear(5, 8)
        self.config = SimpleNamespace(logit_scale=1.7, final_logit_softcapping=2.3)

    def forward(self, input_ids):
        # Exercise all three host RNGs as well as actual Torch dropout.
        perturbation = 0.1 * (random.random() + float(np.random.random()))
        hidden = F.dropout(self.embedding(input_ids), p=0.35, training=self.training)
        return SimpleNamespace(logits=self.projection(hidden + perturbation))


def _terms(model, reference, batch):
    """Independent full-logit oracle; never calls the production loss helpers."""
    logits = model(input_ids=batch["input_ids"]).logits.float()
    logits = 2.3 * torch.tanh(1.7 * logits / 2.3)
    labels = batch["labels"][:, 1:]
    valid = labels != -100
    logp = logits[:, :-1].log_softmax(-1)
    ce = -logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)[valid]
    weights = (-ce).exp().detach()
    # Reference forward is deliberately deterministic: reference dropout and host
    # RNG effects are disabled so the training replay measures current-model RNG.
    ref_logits = reference(batch["input_ids"]).float()
    ref_logp = (2.3 * torch.tanh(1.7 * ref_logits / 2.3))[:, :-1].log_softmax(-1)
    kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1)[valid]
    return (ce * weights).sum(), weights.sum(), kl.sum(), valid.sum()


def _reference(model):
    embedding = model.embedding.weight.detach().clone() * 0.8
    weight = model.projection.weight.detach().clone() * 1.2
    bias = model.projection.bias.detach().clone() + 0.1
    return lambda ids: F.linear(F.embedding(ids, embedding), weight, bias)


def _batches():
    return [
        {"input_ids": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]]),
         "labels": torch.tensor([[-100, 1, 2, 3], [-100, 5, -100, 0]])},
        {"input_ids": torch.tensor([[2, 3, 4]]),
         "labels": torch.tensor([[-100, -100, 4]])},
        {"input_ids": torch.tensor([[5, 6, 0, 1, 2]]),
         "labels": torch.tensor([[-100, 6, 0, -100, 2]])},
    ]


def _seed():
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)


def _next_random(device):
    return random.random(), float(np.random.random()), torch.rand(5), torch.rand(5, device=device)


class _Accelerator:
    def __init__(self, steps=4, processes=1, remote=None):
        self.gradient_accumulation_steps = steps
        self.num_processes = processes
        self.remote = remote
        self.reduced = []

    def reduce(self, value, reduction):
        assert reduction == "sum"
        assert not value.requires_grad
        self.reduced.append(value.detach().clone())
        return value if self.remote is None else value + self.remote.to(value)

    def backward(self, loss):
        (loss / self.gradient_accumulation_steps).backward()


def _trainer(model, accelerator=None):
    trainer = object.__new__(ASFTTrainer)
    trainer.asft_enabled = True
    trainer.asft_mode = "asft"
    trainer.kl_weight = 0.3
    trainer.kl_direction = "forward"
    trainer.reference_policy = "disable_adapter"
    trainer.asft_streaming = asft.ASFTStreamingConfig(seq_chunk_size=2)
    trainer.normalization_semantics = "update_weights_v1"
    # Avoid a frozen-copy allocation in these tests by supplying the deterministic
    # reference callable at the loss boundary.
    trainer._asft_original_model = deepcopy(model).requires_grad_(False).eval()
    trainer.model_wrapped = model
    trainer.model = model
    trainer.args = SimpleNamespace(gradient_accumulation_steps=4)
    trainer.accelerator = accelerator or _Accelerator()
    trainer.model_accepts_loss_kwargs = True
    trainer.compute_loss_func = None
    trainer.current_gradient_accumulation_steps = 2
    trainer._prepare_inputs = lambda inputs: inputs
    trainer._asft_replay_states = []
    trainer._asft_update_denominator = None
    trainer._asft_active_replay_state = None
    trainer._asft_terminal_rng_state = None
    return trainer


def _parent_training_step(trainer, model, inputs, num_items_in_batch=None):
    loss = trainer.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
    trainer.accelerator.backward(loss)
    return loss.detach()


class TestNormalizationSemantics(unittest.TestCase):
    def test_explicit_objectives_and_independent_anchor_denominator(self):
        model = _DropoutModel()
        batch = _batches()[0]
        reference = _reference(model)
        _seed()
        numerator, weight_sum, kl_sum, token_count = _terms(model, reference, batch)
        expected = {
            "token_count_v0": (numerator + 0.3 * kl_sum) / 11,
            "microbatch_weights_v0": numerator / weight_sum + 0.3 * kl_sum / token_count,
            "update_weights_v1": numerator / 2.5 + 0.3 * kl_sum / 13,
        }
        for semantics, oracle in expected.items():
            with self.subTest(semantics=semantics), patch.object(
                asft, "fast_cross_entropy_loss_per_token", side_effect=_cpu_ce
            ), patch.object(asft, "get_reference_forward_callable", return_value=lambda **kw: reference(kw["input_ids"])):
                _seed()
                actual = asft.compute_asft_loss(
                    model, {**batch, "num_items_in_batch": 11}, kl_weight=0.3,
                    normalization_semantics=semantics,
                    update_weight_sum=torch.tensor(2.5), update_token_count=torch.tensor(13),
                )
                torch.testing.assert_close(actual, oracle)
        # This fixture must reject the old, combined DFT+KL weight denominator.
        old_combined = (numerator + 0.3 * kl_sum) / weight_sum
        self.assertGreater(abs(float((expected["microbatch_weights_v0"] - old_combined).detach())), 1e-3)

    def test_default_token_normalization_uses_local_count_and_clamps_zero_items(self):
        model = _DropoutModel()
        batch = _batches()[0]
        with patch.object(asft, "fast_cross_entropy_loss_per_token", side_effect=_cpu_ce):
            _seed()
            local = asft.compute_asft_loss(model, batch, asft_mode="dft")
            _seed()
            zero_items = asft.compute_asft_loss(
                model, {**batch, "num_items_in_batch": 0}, asft_mode="dft"
            )
        torch.testing.assert_close(local * 5, zero_items)

    def test_denominators_are_detached_and_both_update_statistics_required(self):
        numerator = torch.tensor(6.0, requires_grad=True)
        anchor = torch.tensor(8.0, requires_grad=True)
        weights = torch.tensor(6.0, requires_grad=True)
        tokens = torch.tensor(10.0, requires_grad=True)
        loss = asft.normalized_objective(
            numerator, weight_sum=torch.tensor(2.0), token_count=torch.tensor(4.0),
            normalization_semantics="update_weights_v1", update_weight_sum=weights,
            update_token_count=tokens, kl_sum=anchor, kl_weight=0.5,
        )
        self.assertAlmostEqual(loss.item(), 1.4)
        loss.backward()
        self.assertIsNone(weights.grad)
        self.assertIsNone(tokens.grad)
        self.assertAlmostEqual(numerator.grad.item(), 1 / 6)
        self.assertAlmostEqual(anchor.grad.item(), 0.05)
        for missing in ("update_weight_sum", "update_token_count"):
            kwargs = {"update_weight_sum": weights, "update_token_count": tokens}
            kwargs.pop(missing)
            with self.assertRaisesRegex(ValueError, missing):
                asft.normalized_objective(
                    numerator, weight_sum=weights, token_count=tokens,
                    normalization_semantics="update_weights_v1", **kwargs,
                )

    def test_empty_window_emits_zero_denominator_statistics_and_zero_gradient(self):
        model = _DropoutModel()
        batch = _batches()[0]
        batch["labels"] = torch.full_like(batch["labels"], -100)
        stats = []
        loss, outputs = asft.compute_asft_loss(
            model, batch, asft_mode="dft", denominator_stats=stats.append, return_outputs=True
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(stats[0], torch.zeros(2, dtype=torch.float64))
        self.assertFalse(stats[0].requires_grad)
        self.assertIsNotNone(outputs.logits)
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))

    def test_weight_normalization_rejects_non_dft_objective(self):
        with patch.object(asft, "fast_cross_entropy_loss_per_token", side_effect=_cpu_ce):
            with self.assertRaisesRegex(ValueError, "DFT or ASFT"):
                asft.compute_asft_loss(
                    _DropoutModel(), _batches()[0], asft_mode="sft",
                    normalization_semantics="microbatch_weights_v0",
                )


class TestUpdateWindowOracle(unittest.TestCase):
    def test_unequal_and_final_partial_windows_match_full_update_with_dropout(self):
        self._assert_update_windows(device=torch.device("cpu"), cpu_ce=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires the CUDA Triton CE kernel")
    def test_cuda_kernel_and_rng_replay_match_full_update_with_dropout(self):
        self._assert_update_windows(device=torch.device("cuda"), cpu_ce=False)

    def _assert_update_windows(self, *, device, cpu_ce):
        # Two distinct optimizer updates: one full 2-microbatch window and a
        # final 1-microbatch window. Neither has equal token counts per batch.
        all_batches = [
            {key: value.to(device) for key, value in batch.items()} for batch in _batches()
        ]
        for batches in (all_batches[:2], all_batches[2:]):
            with self.subTest(microbatches=len(batches)):
                _seed()
                model = _DropoutModel().to(device)
                oracle_model = deepcopy(model)
                reference = _reference(model)
                trainer = _trainer(model, _Accelerator(steps=2))
                _seed()
                terms = [_terms(oracle_model, reference, batch) for batch in batches]
                weight_sum = sum(term[1].to(torch.float64) for term in terms).detach()
                token_count = sum(term[3] for term in terms)
                oracle = sum(term[0] / weight_sum + 0.3 * term[2] / token_count for term in terms)
                oracle.backward()
                expected_next = _next_random(device)
                # Summing independently normalized microbatches is a credible
                # wrong implementation; ensure the unequal fixture rejects it.
                if len(terms) > 1:
                    wrong = sum(term[0] / term[1] + 0.3 * term[2] / term[3] for term in terms)
                    self.assertGreater(abs(float(wrong.detach() - oracle.detach())), 0.1)
                _seed()
                ce_context = patch.object(
                    asft, "fast_cross_entropy_loss_per_token", side_effect=_cpu_ce
                ) if cpu_ce else nullcontext()
                with patch.object(UnslothTrainer, "get_batch_samples", return_value=(batches, 999)), patch.object(
                    UnslothTrainer, "training_step", new=_parent_training_step
                ), ce_context, patch.object(
                    asft, "get_reference_forward_callable", return_value=lambda **kw: reference(kw["input_ids"])
                ):
                    returned, count = trainer.get_batch_samples(iter(batches), 2, device)
                    losses = [trainer.training_step(model, batch, count) for batch in returned]
                candidate_next = _next_random(device)
                torch.testing.assert_close(sum(losses), oracle.detach())
                torch.testing.assert_close(trainer._asft_update_denominator, torch.stack((weight_sum, token_count)))
                self.assertEqual(candidate_next[:2], expected_next[:2])
                for candidate, expected in zip(candidate_next[2:], expected_next[2:]):
                    torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
                candidate_optimizer = torch.optim.SGD(model.parameters(), lr=0.07)
                oracle_optimizer = torch.optim.SGD(oracle_model.parameters(), lr=0.07)
                for candidate, expected in zip(model.parameters(), oracle_model.parameters()):
                    torch.testing.assert_close(candidate.grad, expected.grad, atol=2e-7, rtol=2e-6)
                candidate_optimizer.step()
                oracle_optimizer.step()
                for candidate, expected in zip(model.parameters(), oracle_model.parameters()):
                    torch.testing.assert_close(candidate, expected, atol=2e-7, rtol=2e-6)
                self.assertEqual(trainer._asft_replay_states, [])

    def test_distributed_denominator_sum_and_gradient_scaling(self):
        model = _DropoutModel()
        remote = torch.tensor([0.7, 3.0], requires_grad=True)
        accelerator = _Accelerator(steps=4, processes=2, remote=remote)
        trainer = _trainer(model, accelerator)
        batches = _batches()[:2]
        with patch.object(UnslothTrainer, "get_batch_samples", return_value=(batches, None)), patch.object(
            asft, "fast_cross_entropy_loss_per_token", side_effect=_cpu_ce
        ):
            trainer.get_batch_samples(iter(batches), 4, torch.device("cpu"))
        torch.testing.assert_close(trainer._asft_update_denominator, accelerator.reduced[0] + remote.detach())
        self.assertFalse(trainer._asft_update_denominator.requires_grad)
        self.assertEqual(trainer._update_loss_scale(), 8)
        with patch("src.train.asft_trainer.compute_asft_loss", return_value=model.projection.weight.sum()), patch.object(
            UnslothTrainer, "training_step", new=_parent_training_step
        ):
            reported = trainer.training_step(model, batches[0], num_items_in_batch=8)
        # Accelerator divides by 4; distributed gradient averaging divides by
        # 2. The trainer must supply 8x before backward, but report unscaled loss.
        torch.testing.assert_close(model.projection.weight.grad / 2, torch.ones_like(model.projection.weight))
        torch.testing.assert_close(reported, model.projection.weight.detach().sum())
        self.assertIsNone(remote.grad)

    def test_update_semantics_cannot_silently_use_a_microbatch_denominator(self):
        trainer = _trainer(_DropoutModel())
        with self.assertRaisesRegex(RuntimeError, "get_batch_samples"):
            trainer.compute_loss(trainer.model, _batches()[0])
        with self.assertRaisesRegex(RuntimeError, "prefetched"):
            trainer.training_step(trainer.model, _batches()[0])

    def test_distributed_update_requires_a_sum_reducer(self):
        trainer = _trainer(_DropoutModel())
        trainer.accelerator = SimpleNamespace(num_processes=2)
        batches = _batches()[:1]
        with patch.object(UnslothTrainer, "get_batch_samples", return_value=(batches, None)), patch.object(
            asft, "fast_cross_entropy_loss_per_token", side_effect=_cpu_ce
        ):
            with self.assertRaisesRegex(RuntimeError, "distributed sum reduction"):
                trainer.get_batch_samples(iter(batches), 4, torch.device("cpu"))
        self.assertIsNone(trainer._asft_update_denominator)

    def test_update_rejects_parent_extra_accumulation_division(self):
        for accepts_items, item_count in ((False, 2), (True, None)):
            with self.subTest(accepts_items=accepts_items, item_count=item_count):
                trainer = _trainer(_DropoutModel())
                trainer.model_accepts_loss_kwargs = accepts_items
                trainer._asft_replay_states = [asft._capture_rng_state()]
                trainer._asft_update_denominator = torch.tensor([1.0, 2.0])
                with patch.object(UnslothTrainer, "training_step") as parent:
                    with self.assertRaisesRegex(RuntimeError, "additional accumulation division"):
                        trainer.training_step(trainer.model, _batches()[0], item_count)
                    parent.assert_not_called()

    def test_training_step_failure_restores_terminal_rng_state(self):
        trainer = _trainer(_DropoutModel())
        _seed()
        replay = asft._capture_rng_state()
        _next_random(torch.device("cpu"))
        terminal = asft._capture_rng_state()
        expected = _next_random(torch.device("cpu"))
        trainer._asft_replay_states = [replay]
        trainer._asft_terminal_rng_state = terminal
        trainer._asft_update_denominator = torch.tensor([1.0, 2.0])

        def failing_step(*_args):
            asft._restore_rng_state(replay)
            _next_random(torch.device("cpu"))
            raise RuntimeError("backward failed")

        with patch.object(UnslothTrainer, "training_step", side_effect=failing_step):
            with self.assertRaisesRegex(RuntimeError, "backward failed"):
                trainer.training_step(trainer.model, _batches()[0], num_items_in_batch=2)
        actual = _next_random(torch.device("cpu"))
        self.assertEqual(actual[:2], expected[:2])
        for value, expected_value in zip(actual[2:], expected[2:]):
            torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
        self.assertIsNone(trainer._asft_active_replay_state)

    def test_disabled_trainer_ignores_inactive_asft_normalization(self):
        trainer = _trainer(_DropoutModel())
        trainer.asft_enabled = False
        with patch.object(UnslothTrainer, "compute_loss", return_value=torch.tensor(2.0)) as parent:
            self.assertEqual(trainer.compute_loss(trainer.model, _batches()[0]).item(), 2.0)
            parent.assert_called_once()


class TestPreservedTrainerProtections(unittest.TestCase):
    def test_enabled_trainer_requests_full_logits_before_parent_initialization(self):
        import os

        observed = []
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UNSLOTH_RETURN_LOGITS", None)
            with patch.object(
                UnslothTrainer, "__init__",
                side_effect=lambda *_args, **_kwargs: observed.append(os.getenv("UNSLOTH_RETURN_LOGITS")),
            ):
                ASFTTrainer(asft_enabled=True)
        self.assertEqual(observed, ["1"])

    def test_zero_kl_never_allocates_or_runs_a_reference(self):
        model = _DropoutModel()
        trainer = _trainer(model)
        trainer.normalization_semantics = "token_count_v0"
        trainer.kl_weight = 0.0
        trainer._asft_original_model = None
        with patch.object(asft, "fast_cross_entropy_loss_per_token", side_effect=_cpu_ce), patch.object(
            asft, "get_reference_forward_callable", side_effect=AssertionError("KL is inactive")
        ):
            loss, outputs = trainer.compute_loss(model, _batches()[0], return_outputs=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(outputs.logits)
        self.assertIsNone(trainer._asft_original_model)


if __name__ == "__main__":
    unittest.main()
