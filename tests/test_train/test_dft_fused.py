import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types
import unittest
from unittest.mock import patch


try:
    import torch
except Exception as exc:  # pragma: no cover - environment-dependent import guard
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

IGNORE_INDEX = -100


def _shift_labels(labels, ignore_index=IGNORE_INDEX):
    shifted = torch.empty_like(labels)
    shifted[..., :-1] = labels[..., 1:]
    shifted[..., -1] = ignore_index
    return shifted


def _softcap(logits, softcap):
    if softcap is None or softcap == 0:
        return logits
    return torch.tanh(logits / softcap) * softcap


def _reference_dft_loss(hidden, weight, bias, labels, *, n_items=None, softcap=30.0):
    logits = torch.nn.functional.linear(hidden.to(weight.dtype), weight, bias)
    logits = _softcap(logits, softcap)
    shifted = _shift_labels(labels)
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, weight.shape[0]).float().contiguous(),
        shifted.view(-1).to(logits.device).contiguous(),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    )
    denom = n_items
    if denom is None:
        denom = (shifted.view(-1) != IGNORE_INDEX).sum().clamp_min(1)
    return ((-ce).exp().detach() * ce).sum() / denom


def _reference_nll_loss(hidden, weight, bias, labels, *, n_items=None, softcap=30.0):
    logits = torch.nn.functional.linear(hidden.to(weight.dtype), weight, bias)
    logits = _softcap(logits, softcap)
    shifted = _shift_labels(labels)
    reduction = "sum" if n_items is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, weight.shape[0]).float().contiguous(),
        shifted.view(-1).to(logits.device).contiguous(),
        reduction=reduction,
        ignore_index=IGNORE_INDEX,
    )
    return loss / n_items if n_items is not None else loss


def _oracle_dft_loss(
    hidden,
    weight,
    bias,
    labels,
    *,
    shift_labels=False,
    ignore_index=IGNORE_INDEX,
    n_items=None,
    scaling=1.0,
    multiply=None,
    divide=None,
    softcap=None,
    detach_weights=True,
):
    """Independent oracle: select valid rows before matmul, then logsumexp/gather."""
    targets = _shift_labels(labels, ignore_index) if shift_labels else labels
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    logits = hidden.reshape(-1, hidden.shape[-1])[valid].to(weight.dtype) @ weight.t()
    if bias is not None:
        logits = logits + bias
    if multiply:
        logits = logits * multiply
    if divide:
        logits = logits / divide
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    logits = logits.float()
    target_logits = logits.gather(1, targets[valid, None]).squeeze(1)
    ce = torch.logsumexp(logits, dim=1) - target_logits
    weights = torch.exp(-ce)
    if detach_weights:
        weights = weights.detach()
    denominator = max(int(valid.sum()), 1) if n_items is None else n_items
    denominator = torch.as_tensor(denominator, device=logits.device).reshape(-1)[0]
    if denominator == 0:
        denominator = 1
    return (weights * ce).sum() / denominator * scaling


class TestFusedDftInstall(unittest.TestCase):
    def setUp(self):
        from src.train import dft_fused

        self.enterContext(patch.dict(os.environ))
        for name, value in (
            ("_ORIGINAL_COMPUTE_FUSED_CE_LOSS", None),
            ("_ORIGINAL_UNSLOTH_FUSED_CE_LOSS", None),
            ("_DEFAULT_IGNORE_INDEX", IGNORE_INDEX),
        ):
            self.enterContext(patch.object(dft_fused, name, value))

    def test_install_fused_dft_patches_both_unsloth_symbols(self):
        from src.train.dft_fused import (
            compute_fused_dft_loss,
            install_fused_dft,
            unsloth_fused_dft_loss,
        )

        ce_mod = types.ModuleType("unsloth_zoo.fused_losses.cross_entropy_loss")

        def original_compute(*_args, **_kwargs):
            raise AssertionError("original compute loss should be patched")

        def original_fused(*_args, **_kwargs):
            raise AssertionError("original fused loss should be patched")

        ce_mod.compute_fused_ce_loss = original_compute
        ce_mod.unsloth_fused_ce_loss = original_fused
        fused_losses_mod = types.ModuleType("unsloth_zoo.fused_losses")
        zoo_mod = types.ModuleType("unsloth_zoo")

        with patch.dict(
            sys.modules,
            {
                "unsloth_zoo": zoo_mod,
                "unsloth_zoo.fused_losses": fused_losses_mod,
                "unsloth_zoo.fused_losses.cross_entropy_loss": ce_mod,
            },
        ):
            install_fused_dft()
            # Model forwards retain the stock wrapper, with its inner objective replaced by DFT.
            self.assertIs(ce_mod.unsloth_fused_ce_loss, unsloth_fused_dft_loss)
            self.assertIs(ce_mod.compute_fused_ce_loss, compute_fused_dft_loss)
            install_fused_dft()  # idempotent
            self.assertIs(ce_mod.unsloth_fused_ce_loss, unsloth_fused_dft_loss)
            self.assertIs(ce_mod.compute_fused_ce_loss, compute_fused_dft_loss)

    @unittest.skipIf(torch is None, f"torch unavailable: {TORCH_IMPORT_ERROR}")
    def test_generated_model_routes_to_dft_after_repeated_install(self):
        from src.train import dft_fused

        ce_mod = types.ModuleType("unsloth_zoo.fused_losses.cross_entropy_loss")

        def original_compute(*_args, **_kwargs):
            raise AssertionError("must not execute the original NLL objective")

        def original_wrapper(trainer, hidden, weight, bias, labels, **kwargs):
            return ce_mod.compute_fused_ce_loss(
                hidden,
                weight,
                bias,
                labels,
                n_items=kwargs["n_items"],
                scaling=kwargs["scaling"],
            )[0]

        ce_mod.compute_fused_ce_loss = original_compute
        ce_mod.unsloth_fused_ce_loss = original_wrapper
        generated = types.ModuleType("unsloth_compiled_module_dft_regression")
        generated.compute_fused_ce_loss = original_compute
        generated.unsloth_fused_ce_loss = original_wrapper
        with patch.dict(
            sys.modules,
            {
                "unsloth_zoo.fused_losses.cross_entropy_loss": ce_mod,
                generated.__name__: generated,
            },
        ):
            dft_fused.install_fused_dft()
            dft_fused.install_fused_dft()
            self.assertEqual(dft_fused._repatch_generated_modules(), 0)
        # Do not evict modules lazily imported by autograd when restoring sys.modules.
        torch.manual_seed(23)
        hidden = torch.randn(1, 5, 3, requires_grad=True)
        weight = torch.randn(7, 3, requires_grad=True)
        labels = torch.tensor([[1, 2, IGNORE_INDEX, 4, 5]])
        candidate = generated.unsloth_fused_ce_loss(
            None, hidden, weight, None, labels
        )
        expected = _oracle_dft_loss(
            hidden, weight, None, labels, shift_labels=True
        )
        torch.testing.assert_close(candidate, expected)
        actual_grads = torch.autograd.grad(candidate, (hidden, weight))
        expected_grads = torch.autograd.grad(expected, (hidden, weight))
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        inner, _ = generated.compute_fused_ce_loss(
            hidden, weight, None, labels
        )
        torch.testing.assert_close(inner, candidate)

    def test_missing_inner_loss_seam_fails_installation(self):
        from src.train import dft_fused

        ce_mod = types.ModuleType("unsloth_zoo.fused_losses.cross_entropy_loss")
        ce_mod.unsloth_fused_ce_loss = lambda *_args, **_kwargs: None
        with patch.dict(
            sys.modules, {"unsloth_zoo.fused_losses.cross_entropy_loss": ce_mod}
        ):
            with self.assertRaises(AttributeError):
                dft_fused.install_fused_dft()

    @unittest.skipIf(torch is None, f"torch unavailable: {TORCH_IMPORT_ERROR}")
    def test_chunk_default_environment_and_explicit_override(self):
        from src.train import dft_fused

        ce_mod = types.ModuleType("unsloth_zoo.fused_losses.cross_entropy_loss")
        received_chunks = []

        def original_wrapper(*_args, **kwargs):
            # Match the stock wrapper's import-time environment override.
            if ce_mod.N_CHUNKS:
                kwargs["n_chunks"] = int(ce_mod.N_CHUNKS)
            received_chunks.append(kwargs.get("n_chunks"))
            return 0

        ce_mod.compute_fused_ce_loss = lambda *_args, **_kwargs: None
        ce_mod.unsloth_fused_ce_loss = original_wrapper
        os.environ.pop("UNSLOTH_CE_LOSS_TARGET_GB", None)
        with patch.dict(
            sys.modules, {"unsloth_zoo.fused_losses.cross_entropy_loss": ce_mod}
        ):
            for configured, expected in ((None, 16), ("7", 7)):
                with self.subTest(configured=configured):
                    if configured is None:
                        os.environ.pop("UNSLOTH_CE_LOSS_N_CHUNKS", None)
                    else:
                        os.environ["UNSLOTH_CE_LOSS_N_CHUNKS"] = configured
                    ce_mod.N_CHUNKS = configured
                    dft_fused.install_fused_dft()
                    args = (None, None, None, None, torch.tensor([[1, 2]]))
                    dft_fused.unsloth_fused_dft_loss(*args)
                    dft_fused.unsloth_fused_dft_loss(*args, n_chunks=2)
                    dft_fused.unsloth_fused_dft_loss(*args, target_gb=1.5)
                    self.assertEqual(received_chunks[-3:], [expected, 2, None])
                    self.assertEqual(
                        os.environ["UNSLOTH_CE_LOSS_N_CHUNKS"], str(expected)
                    )


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_IMPORT_ERROR}")
class TestFusedDftLoss(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(7)
        hidden = torch.randn(2, 8, 5, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(13, 5, dtype=torch.float32, requires_grad=True)
        bias = torch.randn(13, dtype=torch.float32, requires_grad=True)
        labels = torch.randint(0, 13, (2, 8), dtype=torch.long)
        labels[0, 3] = IGNORE_INDEX
        labels[1, 6] = IGNORE_INDEX
        shifted = _shift_labels(labels)
        n_items = (shifted.view(-1) != IGNORE_INDEX).sum().to(torch.float32)
        return hidden, weight, bias, labels, n_items

    def _clone_inputs(self, hidden, weight, bias):
        return (
            hidden.detach().clone().requires_grad_(True),
            weight.detach().clone().requires_grad_(True),
            bias.detach().clone().requires_grad_(True),
        )

    def test_fused_dft_gradient_matches_full_logit_reference(self):
        from src.train.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, n_items = self._inputs()
        candidate_loss, _ = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        candidate_loss.backward()
        candidate_grads = (hidden.grad.clone(), weight.grad.clone(), bias.grad.clone())

        ref_hidden, ref_weight, ref_bias = self._clone_inputs(hidden, weight, bias)
        ref_loss = _reference_dft_loss(
            ref_hidden,
            ref_weight,
            ref_bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )
        ref_loss.backward()
        ref_grads = (ref_hidden.grad, ref_weight.grad, ref_bias.grad)

        for candidate_grad, ref_grad in zip(candidate_grads, ref_grads):
            self.assertTrue(
                torch.allclose(candidate_grad, ref_grad, rtol=1e-5, atol=1e-6),
                (candidate_grad - ref_grad).abs().max().item(),
            )

        nll_hidden, nll_weight, nll_bias = self._clone_inputs(hidden, weight, bias)
        nll_loss = _reference_nll_loss(
            nll_hidden,
            nll_weight,
            nll_bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )
        nll_loss.backward()
        self.assertFalse(
            torch.allclose(candidate_grads[0], nll_hidden.grad, rtol=1e-5, atol=1e-6)
        )

    def test_fused_dft_loss_value_and_scale(self):
        from src.train.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, n_items = self._inputs()
        candidate_loss, (detached_loss,) = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        ref_loss = _reference_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )
        nll_loss = _reference_nll_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )

        self.assertTrue(torch.allclose(candidate_loss, ref_loss, rtol=1e-6, atol=1e-7))
        self.assertTrue(
            torch.allclose(detached_loss, ref_loss.detach(), rtol=1e-6, atol=1e-7)
        )
        self.assertLess(candidate_loss.item(), nll_loss.item())

    def test_softcap_is_applied(self):
        from src.train.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, n_items = self._inputs()
        hidden = (hidden.detach() * 40).requires_grad_(True)
        weight = (weight.detach() * 8).requires_grad_(True)

        candidate_loss, _ = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        softcapped_ref = _reference_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            softcap=30.0,
        )
        uncapped_ref = _reference_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            softcap=None,
        )

        self.assertTrue(
            torch.allclose(candidate_loss, softcapped_ref, rtol=1e-6, atol=1e-7)
        )
        self.assertFalse(
            torch.allclose(candidate_loss, uncapped_ref, rtol=1e-4, atol=1e-5)
        )

    def test_mask_and_normalizer(self):
        from src.train.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, _n_items = self._inputs()
        labels = labels.clone()
        labels[0, 2] = IGNORE_INDEX
        shifted = _shift_labels(labels)
        n_items = (shifted.view(-1) != IGNORE_INDEX).sum().to(torch.float32)

        loss, _ = compute_fused_dft_loss(
            hidden,
            weight,
            bias,
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        loss.backward()

        self.assertTrue(
            torch.allclose(hidden.grad[0, 1], torch.zeros_like(hidden.grad[0, 1]))
        )

        changed_hidden = hidden.detach().clone()
        changed_hidden[0, 1] = changed_hidden[0, 1] + 1000
        changed_loss, _ = compute_fused_dft_loss(
            changed_hidden.requires_grad_(True),
            weight.detach().clone().requires_grad_(True),
            bias.detach().clone().requires_grad_(True),
            labels,
            n_items=n_items,
            shift_labels=True,
            logit_softcapping=30.0,
            ignore_index=IGNORE_INDEX,
        )
        self.assertTrue(
            torch.allclose(loss.detach(), changed_loss.detach(), rtol=1e-6, atol=1e-7)
        )

    def test_noncontiguous_labels_match_independent_value_and_gradient_oracle(self):
        from src.train.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, _ = self._inputs()
        labels = labels.t().contiguous().t()
        self.assertFalse(labels.is_contiguous())
        candidate, _ = compute_fused_dft_loss(
            hidden, weight, bias, labels, shift_labels=False
        )
        expected = _oracle_dft_loss(hidden, weight, bias, labels)
        torch.testing.assert_close(candidate, expected)
        actual_grads = torch.autograd.grad(candidate, (hidden, weight, bias))
        expected_grads = torch.autograd.grad(expected, (hidden, weight, bias))
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_zero_denominator_matches_unit_denominator_oracle(self):
        from src.train.dft_fused import compute_fused_dft_loss

        for denominator in (0, torch.tensor(0), torch.tensor([0, 9])):
            with self.subTest(denominator=denominator):
                hidden, weight, bias, labels, _ = self._inputs()
                candidate, _ = compute_fused_dft_loss(
                    hidden, weight, bias, labels, n_items=denominator, shift_labels=False
                )
                expected = _oracle_dft_loss(hidden, weight, bias, labels, n_items=1)
                torch.testing.assert_close(candidate, expected)
                actual_grads = torch.autograd.grad(candidate, (hidden, weight, bias))
                expected_grads = torch.autograd.grad(expected, (hidden, weight, bias))
                for actual, expected in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_ignored_overflowed_logits_match_valid_row_oracle(self):
        from src.train.dft_fused import compute_fused_dft_loss

        largest = torch.finfo(torch.float32).max
        for softcap in (None, 1.0):
            with self.subTest(softcap=softcap):
                hidden = torch.tensor(
                    [[[0.2, -0.1], [largest, largest]]], requires_grad=True
                )
                weight = torch.tensor(
                    [[0.3, -0.1], [2.0, -2.0], [-2.0, 2.0]], requires_grad=True
                )
                bias = torch.tensor([0.1, -0.2, 0.3], requires_grad=True)
                labels = torch.tensor([[1, IGNORE_INDEX]])
                raw_logits = torch.nn.functional.linear(hidden, weight, bias)
                self.assertFalse(torch.isfinite(raw_logits[0, 1]).all())
                candidate, _ = compute_fused_dft_loss(
                    hidden,
                    weight,
                    bias,
                    labels,
                    shift_labels=False,
                    logit_softcapping=softcap,
                )
                expected = _oracle_dft_loss(hidden, weight, bias, labels, softcap=softcap)
                torch.testing.assert_close(candidate, expected)
                actual_grads = torch.autograd.grad(candidate, (hidden, weight, bias))
                expected_grads = torch.autograd.grad(expected, (hidden, weight, bias))
                for actual, expected in zip(actual_grads, expected_grads):
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rejects_label_smoothing(self):
        from src.train.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, _ = self._inputs()
        with self.assertRaisesRegex(ValueError, "label_smoothing"):
            compute_fused_dft_loss(
                hidden, weight, bias, labels, label_smoothing=0.1
            )

    def test_scaling_transforms_and_detached_weighting_under_functorch(self):
        from src.train.dft_fused import compute_fused_dft_loss

        hidden, weight, bias, labels, _ = self._inputs()
        (actual_grads, (actual, (unscaled,))) = torch.func.grad_and_value(
            compute_fused_dft_loss, argnums=(0, 1, 2), has_aux=True
        )(
            hidden,
            weight,
            bias,
            labels,
            n_items=19,
            scaling=8.0,
            shift_labels=True,
            logit_scale_multiply=1.5,
            logit_scale_divide=2.0,
            logit_softcapping=1.25,
        )
        oracle_args = dict(
            shift_labels=True, n_items=19, scaling=8.0,
            multiply=1.5, divide=2.0, softcap=1.25,
        )
        expected = _oracle_dft_loss(hidden, weight, bias, labels, **oracle_args)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(unscaled, expected.detach() / 8.0)
        self.assertFalse(unscaled.requires_grad)
        expected_grads = torch.autograd.grad(expected, (hidden, weight, bias))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)

        # The same forward value with trainable weights is a credible wrong objective.
        wrong = _oracle_dft_loss(
            hidden, weight, bias, labels, detach_weights=False, **oracle_args
        )
        wrong_grad = torch.autograd.grad(wrong, hidden)[0]
        self.assertFalse(torch.allclose(actual_grads[0], wrong_grad))

    def test_all_ignored_rows_have_zero_value_and_gradients(self):
        from src.train.dft_fused import compute_fused_dft_loss

        for denominator in (None, 0):
            with self.subTest(denominator=denominator):
                hidden, weight, bias, labels, _ = self._inputs()
                labels.fill_(IGNORE_INDEX)
                loss, (unscaled,) = compute_fused_dft_loss(
                    hidden, weight, bias, labels,
                    n_items=denominator, scaling=8.0,
                )
                self.assertEqual(loss.item(), 0.0)
                self.assertEqual(unscaled.item(), 0.0)
                for gradient in torch.autograd.grad(loss, (hidden, weight, bias)):
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertEqual(torch.count_nonzero(gradient).item(), 0)


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_IMPORT_ERROR}")
class TestUnslothFusedDftLossRouting(unittest.TestCase):
    def test_delegates_mask_scaling_and_compilation_controls(self):
        from src.train import dft_fused

        original = unittest.mock.Mock(return_value=torch.tensor(0.25))
        hidden = torch.randn(1, 3, 4, dtype=torch.bfloat16)
        weight = torch.randn(7, 4, dtype=torch.bfloat16)
        labels = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[1, 1, 0]])
        trainer = object()
        with patch.object(dft_fused, "_ORIGINAL_UNSLOTH_FUSED_CE_LOSS", original):
            loss = dft_fused.unsloth_fused_dft_loss(
                trainer,
                hidden,
                weight,
                None,
                labels,
                mask=mask,
                n_items=7,
                scaling=8.0,
                target_gb=2.0,
                torch_compile=True,
                overwrite=True,
                n_chunks=4,
                logit_softcapping=1.0,
            )
        self.assertEqual(loss.item(), 0.25)
        forwarded_denominator = original.call_args.kwargs["n_items"]
        self.assertIsInstance(forwarded_denominator, torch.Tensor)
        self.assertEqual(forwarded_denominator.dtype, torch.float32)
        self.assertEqual(forwarded_denominator.device, weight.device)
        self.assertEqual(forwarded_denominator.item(), 7)
        original.assert_called_once_with(
            trainer,
            hidden,
            weight,
            None,
            labels,
            mask=mask,
            n_items=forwarded_denominator,
            scaling=8.0,
            target_gb=2.0,
            torch_compile=True,
            overwrite=True,
            n_chunks=4,
            logit_softcapping=1.0,
        )

    def test_public_loss_rejects_smoothing_before_wrapper_dispatch(self):
        from src.train import dft_fused

        original = unittest.mock.Mock()
        with patch.object(dft_fused, "_ORIGINAL_UNSLOTH_FUSED_CE_LOSS", original):
            with self.assertRaisesRegex(ValueError, "label_smoothing"):
                dft_fused.unsloth_fused_dft_loss(
                    None, None, None, None, None, label_smoothing=0.1
                )
        original.assert_not_called()

    def test_missing_installation_fails_instead_of_using_nll(self):
        from src.train import dft_fused

        with patch.object(dft_fused, "_ORIGINAL_UNSLOTH_FUSED_CE_LOSS", None):
            with self.assertRaisesRegex(RuntimeError, "install_fused_dft"):
                dft_fused.unsloth_fused_dft_loss(None, None, None, None, None)

    def test_public_wrapper_rejects_unshifted_labels_instead_of_silent_shift(self):
        from src.train import dft_fused

        original = unittest.mock.Mock()
        with patch.object(dft_fused, "_ORIGINAL_UNSLOTH_FUSED_CE_LOSS", original):
            with self.assertRaisesRegex(ValueError, "causal shift_labels=True"):
                dft_fused.unsloth_fused_dft_loss(
                    None, None, None, None, None, shift_labels=False
                )
        original.assert_not_called()


@unittest.skipUnless(
    torch is not None
    and torch.cuda.is_available()
    and importlib.util.find_spec("unsloth") is not None,
    "CUDA and the installed Unsloth stack are required",
)
class TestFusedDftCUDA(unittest.TestCase):
    def test_compiled_stock_wrapper_matches_independent_oracle(self):
        # Unsloth's process-wide patches must not leak into the rest of the unit suite.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""\
                    import os
                    import sys
                    os.environ['UNSLOTH_ENABLE_CCE'] = '0'
                    os.environ['UNSLOTH_CE_LOSS_N_CHUNKS'] = '7'
                    import unsloth
                    import torch
                    from unittest.mock import patch
                    import unsloth_zoo.fused_losses.cross_entropy_loss as ce_mod
                    from src.train import dft_fused
                    sys.path.insert(0, 'tests/test_train')
                    from test_dft_fused import _oracle_dft_loss

                    dft_fused.install_fused_dft()
                    torch.manual_seed(31)
                    hidden = torch.randn(2, 8, 5, device='cuda', requires_grad=True)
                    weight = torch.randn(13, 5, device='cuda', requires_grad=True)
                    bias = torch.randn(13, device='cuda', requires_grad=True)
                    labels = torch.randint(0, 13, (2, 8), device='cuda')
                    labels[:, 3] = -100
                    labels = labels.t().contiguous().t()
                    actual = dft_fused.unsloth_fused_dft_loss(
                        None, hidden, weight, bias, labels,
                        shift_labels=True,
                        scaling=8.0, n_items=torch.tensor(11, device='cuda'),
                        n_chunks=2, torch_compile=True,
                        logit_scale_multiply=1.5, logit_scale_divide=2.0,
                        logit_softcapping=1.25,
                    )
                    expected = _oracle_dft_loss(
                        hidden, weight, bias, labels, shift_labels=True,
                        n_items=11, multiply=1.5, divide=2.0, softcap=1.25,
                    )
                    torch.testing.assert_close(actual, expected)
                    actual_grads = torch.autograd.grad(actual, (hidden, weight, bias))
                    expected_grads = torch.autograd.grad(expected * 8.0, (hidden, weight, bias))
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=1e-6)

                    mask = torch.ones_like(labels)
                    mask[:, 5] = 0
                    with patch.object(torch, 'chunk', wraps=torch.chunk) as chunk:
                        masked = dft_fused.unsloth_fused_dft_loss(
                            None, hidden, weight, bias, labels,
                            mask=mask, n_chunks=2, torch_compile=False,
                        )
                    assert chunk.call_count == 3
                    assert all(call.args[1] == 2 for call in chunk.call_args_list), 'environment overrode explicit chunks'
                    expected_masked = _oracle_dft_loss(
                        hidden, weight, bias, labels.masked_fill(mask == 0, -100),
                        shift_labels=True,
                    )
                    torch.testing.assert_close(masked, expected_masked)
                    for actual_grad, expected_grad in zip(
                        torch.autograd.grad(masked, (hidden, weight, bias)),
                        torch.autograd.grad(expected_masked, (hidden, weight, bias)),
                    ):
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=1e-6)

                    zero = dft_fused.unsloth_fused_dft_loss(
                        None, hidden, weight, bias, torch.full_like(labels, -100),
                        n_items=0, n_chunks=2, torch_compile=True,
                    )
                    assert zero.item() == 0.0
                    for grad in torch.autograd.grad(zero, (hidden, weight, bias)):
                        assert torch.count_nonzero(grad).item() == 0

                    largest = torch.finfo(torch.float32).max
                    overflow_hidden = torch.tensor(
                        [[[0.2, -0.1], [largest, largest]]],
                        device='cuda', requires_grad=True,
                    )
                    overflow_weight = torch.tensor(
                        [[0.3, -0.1], [2.0, -2.0], [-2.0, 2.0]],
                        device='cuda', requires_grad=True,
                    )
                    overflow_labels = torch.tensor([[0, 1]], device='cuda')
                    assert not torch.isfinite(
                        torch.nn.functional.linear(overflow_hidden, overflow_weight)[0, 1]
                    ).all()
                    overflow_loss = dft_fused.unsloth_fused_dft_loss(
                        None, overflow_hidden, overflow_weight, None, overflow_labels,
                        n_chunks=1, torch_compile=True, logit_softcapping=1.0,
                    )
                    overflow_expected = _oracle_dft_loss(
                        overflow_hidden, overflow_weight, None, overflow_labels,
                        shift_labels=True, softcap=1.0,
                    )
                    torch.testing.assert_close(overflow_loss, overflow_expected)
                    for actual_grad, expected_grad in zip(
                        torch.autograd.grad(overflow_loss, (overflow_hidden, overflow_weight)),
                        torch.autograd.grad(overflow_expected, (overflow_hidden, overflow_weight)),
                    ):
                        assert torch.isfinite(actual_grad).all()
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=1e-6)

                    bf_hidden = hidden.detach().to(torch.bfloat16).requires_grad_(True)
                    bf_weight = weight.detach().to(torch.bfloat16)
                    bf_loss = dft_fused.unsloth_fused_dft_loss(
                        None, bf_hidden, bf_weight, None, labels,
                        n_chunks=2, torch_compile=True,
                    )
                    bf_expected = _oracle_dft_loss(
                        bf_hidden, bf_weight, None, labels, shift_labels=True,
                    )
                    torch.testing.assert_close(bf_loss, bf_expected, rtol=2e-2, atol=1e-4)
                    torch.testing.assert_close(
                        torch.autograd.grad(bf_loss, bf_hidden)[0],
                        torch.autograd.grad(bf_expected, bf_hidden)[0],
                        rtol=2e-2, atol=1e-4,
                    )
                    assert ce_mod._FUSED_CE_COMPILE_SUPPORTED is True, 'Unsloth fell back to eager execution'
                    print('CUDA compiled stock-wrapper DFT: transforms, mask, chunks, zero, overflow, bf16; scaled gradients match oracle')
                """),
            ],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("scaled gradients match oracle", result.stdout)

    def test_custom_ignore_indices_match_independent_value_and_gradient_oracle(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""\
                    import os
                    import sys
                    os.environ['UNSLOTH_ENABLE_CCE'] = '0'
                    import unsloth
                    import torch
                    import unsloth_zoo.fused_losses.cross_entropy_loss as ce_mod
                    from src.train import dft_fused
                    sys.path.insert(0, 'tests/test_train')
                    from test_dft_fused import _oracle_dft_loss

                    cases = 0
                    for device in ('cpu', 'cuda'):
                        for installed_ignore in (-7, 0):
                            dft_fused.install_fused_dft(ignore_index=installed_ignore)
                            for override in (None, -9, -100):
                                ignore = installed_ignore if override is None else override
                                torch.manual_seed(41)
                                hidden = torch.randn(2, 6, 3, device=device, requires_grad=True)
                                weight = torch.randn(5, 3, device=device, requires_grad=True)
                                bias = torch.randn(5, device=device, requires_grad=True)
                                labels = torch.tensor(
                                    [[0, 1, 2, 3, 4, 1], [2, 3, 4, 1, 0, 2]], device=device,
                                )
                                labels[:, 2] = ignore
                                mask = torch.ones_like(labels)
                                mask[:, 4] = 0
                                kwargs = {} if override is None else dict(ignore_index=override)
                                denominator = None if override is None else 9
                                actual = dft_fused.unsloth_fused_dft_loss(
                                    None, hidden, weight, bias, labels, mask=mask,
                                    n_items=denominator, scaling=4.0, n_chunks=2,
                                    torch_compile=(device == 'cuda'),
                                    logit_scale_multiply=1.5, logit_softcapping=2.0,
                                    **kwargs,
                                )
                                expected = _oracle_dft_loss(
                                    hidden, weight, bias, labels.masked_fill(mask == 0, ignore),
                                    shift_labels=True, ignore_index=ignore, n_items=denominator,
                                    multiply=1.5, softcap=2.0,
                                )
                                torch.testing.assert_close(actual, expected)
                                for actual_grad, expected_grad in zip(
                                    torch.autograd.grad(actual, (hidden, weight, bias)),
                                    torch.autograd.grad(expected * 4.0, (hidden, weight, bias)),
                                ):
                                    assert torch.isfinite(actual_grad).all()
                                    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=1e-6)
                                cases += 1
                    assert ce_mod._FUSED_CE_COMPILE_SUPPORTED is True, 'Unsloth fell back to eager execution'
                    print(f'custom-ignore stock-wrapper oracle: {cases} CPU/CUDA cases passed')
                """),
            ],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("12 CPU/CUDA cases passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
