"""Regression tests for the facts the Mehldau update-budget diagnosis relies on.

* train_qlora's loop takes one optimizer update per ``gradient_accumulation``
  batches, so 16 songs / batch 4 / accumulation 4 is one update per epoch.
* A LoRA-wrapped checkpoint saved the way train_qlora saves it reloads through
  load_model_with_lora with identical logits.
* Only LoRA tensors change during adapter training.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "music_transformer"))

from utilities.device import use_cuda

use_cuda(False)

from model.music_transformer import MusicTransformer
from scripts.generate import load_model_with_lora
from scripts.run_mehldau_update_budget_diag import fixed_crops, token_identity
from scripts.train_qlora import add_lora_to_model, train_epoch

TINY = dict(n_layers=1, num_heads=2, d_model=16, dim_feedforward=32, max_sequence=32, rpr=True)


def tiny_lora_model() -> nn.Module:
    torch.manual_seed(0)
    model = MusicTransformer(**TINY)
    model, _ = add_lora_to_model(model, r=2, alpha=4, dropout=0.0)
    return model


class OptimizerUpdateCountTest(unittest.TestCase):
    def _updates(self, batches: int, accumulation: int) -> int:
        model = tiny_lora_model()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        data = [(torch.randint(0, 300, (4, 16)), torch.randint(0, 300, (4, 16)))
                for _ in range(batches)]
        loss_fn = nn.CrossEntropyLoss()
        train_epoch(model, data, optimizer, None, loss_fn, torch.device("cpu"), 1,
                    gradient_accumulation=accumulation)
        steps = {int(s["step"]) for s in optimizer.state_dict()["state"].values()}
        self.assertEqual(len(steps), 1)
        return steps.pop()

    def test_accumulation_four_over_four_batches_is_one_update(self) -> None:
        # The Mehldau run: 16 train songs, batch 4 -> 4 batches, accumulation 4.
        self.assertEqual(self._updates(batches=4, accumulation=4), 1)

    def test_accumulation_one_updates_every_batch(self) -> None:
        self.assertEqual(self._updates(batches=4, accumulation=1), 4)

    def test_trailing_partial_accumulation_still_steps(self) -> None:
        self.assertEqual(self._updates(batches=5, accumulation=4), 2)

    def test_only_lora_tensors_change(self) -> None:
        model = tiny_lora_model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        data = [(torch.randint(0, 300, (2, 16)), torch.randint(0, 300, (2, 16)))]
        train_epoch(model, data, optimizer, None, nn.CrossEntropyLoss(),
                    torch.device("cpu"), 1, gradient_accumulation=1)
        changed = {k for k, v in model.state_dict().items() if not torch.equal(before[k], v)}
        self.assertTrue(changed)
        self.assertTrue(all("lora_" in k for k in changed), changed)


class SaveLoadIdentityTest(unittest.TestCase):
    def test_full_checkpoint_reload_gives_identical_logits(self) -> None:
        model = tiny_lora_model()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(0, 0.1)
        model.eval()
        x = torch.randint(0, 300, (1, 12))
        with torch.no_grad():
            expected = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_epoch1.pt"
            torch.save({"model_config": {**TINY, "lora_r": 2, "lora_alpha": 4},
                        "model_state_dict": model.state_dict()}, path)
            loaded = load_model_with_lora(lora_path=tmp, checkpoint_path=str(path),
                                          prefer_full_checkpoint=True)
        with torch.no_grad():
            actual = loaded(x)
        self.assertTrue(torch.equal(expected, actual))


class DiagnosticHelpersTest(unittest.TestCase):
    def test_fixed_crops_are_deterministic_and_non_overlapping(self) -> None:
        seq = np.arange(25)
        crops = fixed_crops([seq], length=8)
        self.assertEqual([int(c[0]) for c in crops], [0, 8, 16])
        self.assertTrue(all(len(c) == 9 for c in crops))
        self.assertEqual([c.tolist() for c in crops],
                         [c.tolist() for c in fixed_crops([seq], length=8)])

    def test_fixed_crops_cap_per_song(self) -> None:
        self.assertEqual(len(fixed_crops([np.arange(100)], length=8, max_per_song=2)), 2)

    def test_token_identity(self) -> None:
        self.assertEqual(token_identity([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(token_identity([1, 2, 3, 4], [1, 9]), 0.25)
        self.assertEqual(token_identity([], []), 1.0)


if __name__ == "__main__":
    unittest.main()


class ExportSnapshotTest(unittest.TestCase):
    def test_export_merges_snapshot_and_refuses_overwrite(self) -> None:
        from scripts.export_lora_snapshot import main as export_main

        base = tiny_lora_model()
        tuned = tiny_lora_model()
        with torch.no_grad():
            for name, param in tuned.named_parameters():
                if "lora_B" in name:
                    param.normal_(0, 0.1)
        cfg = {**TINY, "lora_r": 2, "lora_alpha": 4, "lora_dropout": 0.0}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            torch.save({"model_config": cfg, "model_state_dict": base.state_dict()}, tmp / "base.pt")
            torch.save({k: v for k, v in tuned.state_dict().items() if "lora_" in k}, tmp / "snap.pt")
            out = tmp / "export" / "checkpoint_update1.pt"
            args = ["--base", str(tmp / "base.pt"), "--snapshot", str(tmp / "snap.pt"),
                    "--output", str(out)]
            self.assertEqual(export_main(args), 0)
            loaded = load_model_with_lora(lora_path=str(out.parent), checkpoint_path=str(out),
                                          prefer_full_checkpoint=True)
            tuned.eval()
            x = torch.randint(0, 300, (1, 12))
            with torch.no_grad():
                self.assertTrue(torch.equal(tuned(x), loaded(x)))
            with self.assertRaises(SystemExit):
                export_main(args)


class LoraTargetsTest(unittest.TestCase):
    def _cfg(self):
        return {**TINY, "lora_r": 2, "lora_alpha": 4, "lora_dropout": 0.0}

    def test_default_targets_unchanged(self) -> None:
        from scripts.train_qlora import lora_targets_in_state_dict

        model = tiny_lora_model()
        self.assertEqual(lora_targets_in_state_dict(model.state_dict()), ["out_proj"])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(trainable, 2 * 16 * 2)  # one layer, out_proj A+B at r=2, d=16

    def test_qkv_and_ffn_start_at_zero_delta_and_keep_base_keys(self) -> None:
        from scripts.train_qlora import add_lora_targets, lora_targets_in_state_dict

        model = tiny_lora_model()
        model.eval()
        x = torch.randint(0, 300, (1, 12))
        with torch.no_grad():
            before = model(x)
        add_lora_targets(model, ["qkv", "ffn"], r=2, alpha=4, dropout=0.0)
        model.eval()
        with torch.no_grad():
            after = model(x)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        keys = model.state_dict().keys()
        self.assertIn("transformer.encoder.layers.0.self_attn.in_proj_weight", keys)
        self.assertEqual(lora_targets_in_state_dict(model.state_dict()), ["out_proj", "qkv", "ffn"])
        new = {k for k, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all("lora_" in k for k in new))
        self.assertTrue(any("lora_A_q" in k for k in new))
        self.assertTrue(any("linear1.lora_A" in k for k in new))

    def test_qkv_lora_changes_output_and_gets_gradient(self) -> None:
        from scripts.train_qlora import add_lora_targets

        model = tiny_lora_model()
        add_lora_targets(model, ["qkv"], r=2, alpha=4)
        attn = model.transformer.encoder.layers[0].self_attn
        with torch.no_grad():
            attn.lora_B_v.normal_(0, 0.5)
        model.train()
        out = model(torch.randint(0, 300, (1, 12)))
        out.sum().backward()
        self.assertIsNotNone(attn.lora_A_v.grad)
        self.assertIsNone(attn._parameters["in_proj_weight"].grad)

    def test_extended_checkpoint_reloads_identically(self) -> None:
        from scripts.train_qlora import add_lora_targets

        model = tiny_lora_model()
        add_lora_targets(model, ["qkv", "ffn"], r=2, alpha=4, dropout=0.0)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(0, 0.1)
        model.eval()
        x = torch.randint(0, 300, (1, 12))
        with torch.no_grad():
            expected = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_epoch1.pt"
            torch.save({"model_config": {**self._cfg(), "lora_targets": ["out_proj", "qkv", "ffn"]},
                        "model_state_dict": model.state_dict()}, path)
            loaded = load_model_with_lora(lora_path=tmp, checkpoint_path=str(path),
                                          prefer_full_checkpoint=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(expected, loaded(x)))


class LoraLoadingFailClosedTest(unittest.TestCase):
    """Review H1/M1: snapshot keys must never be dropped silently, and a base that
    already carries QKV must be usable as a base."""

    CFG = {**TINY, "lora_r": 2, "lora_alpha": 4, "lora_dropout": 0.0}

    def _qkv_model(self, seed=0):
        from scripts.train_qlora import add_lora_to_model

        torch.manual_seed(seed)
        model = MusicTransformer(**TINY)
        model, _ = add_lora_to_model(model, r=2, alpha=4, dropout=0.0, targets=("out_proj", "qkv"))
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(0, 0.1)
        return model.eval()

    @staticmethod
    def _lora(model):
        return {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}

    def test_qkv_snapshot_into_out_proj_model_is_rejected(self) -> None:
        from scripts.train_qlora import load_lora_snapshot

        out_proj_only = tiny_lora_model()
        with self.assertRaises(ValueError) as ctx:
            load_lora_snapshot(out_proj_only, self._lora(self._qkv_model()))
        self.assertIn("unexpected 6", str(ctx.exception))

    def test_partial_snapshot_is_rejected(self) -> None:
        from scripts.train_qlora import load_lora_snapshot

        model = self._qkv_model()
        partial = {k: v for k, v in self._lora(model).items() if "lora_A_q" not in k}
        with self.assertRaises(ValueError) as ctx:
            load_lora_snapshot(self._qkv_model(seed=1), partial)
        self.assertIn("missing 1", str(ctx.exception))

    def test_targets_inferred_from_snapshot_restore_exact_logits(self) -> None:
        from scripts.train_qlora import (build_lora_model_from_state, load_lora_snapshot,
                                         lora_targets_in_state_dict)

        trained = self._qkv_model()
        snapshot = self._lora(trained)
        targets = lora_targets_in_state_dict(snapshot)
        self.assertEqual(targets, ["out_proj", "qkv"])
        # base = the out_proj-only layout every existing base checkpoint has
        base = tiny_lora_model()
        base_state = {k: v for k, v in trained.state_dict().items() if "lora_" not in k
                      or ".out_proj." in k}
        base_state.update({k: v for k, v in base.state_dict().items() if ".out_proj.lora_" in k})
        model, built = build_lora_model_from_state(self.CFG, base_state, extra_targets=targets)
        self.assertEqual(built, ["out_proj", "qkv"])
        load_lora_snapshot(model, snapshot)
        model.eval()
        x = torch.randint(0, 300, (1, 12))
        with torch.no_grad():
            self.assertTrue(torch.equal(trained(x), model(x)))

    def test_qkv_base_can_be_reused_with_extra_target(self) -> None:
        from scripts.train_qlora import build_lora_model_from_state, lora_targets_in_state_dict

        b_full = self._qkv_model().state_dict()
        model, targets = build_lora_model_from_state(self.CFG, b_full, extra_targets=["ffn"])
        self.assertEqual(targets, ["out_proj", "qkv", "ffn"])
        self.assertEqual(lora_targets_in_state_dict(model.state_dict()), ["out_proj", "qkv", "ffn"])
        # every carried-over tensor is kept exactly; wrapping FFN renames
        # linear{1,2}.weight/bias to linear{1,2}.original_layer.*
        new_state = model.state_dict()
        for k, v in b_full.items():
            key = k
            for lin in (".linear1.", ".linear2."):
                if lin in k and "original_layer" not in k:
                    key = k.replace(lin, lin + "original_layer.")
            self.assertTrue(torch.equal(new_state[key], v), k)

    def test_plain_base_gets_default_out_proj(self) -> None:
        from scripts.train_qlora import build_lora_model_from_state

        torch.manual_seed(0)
        plain = MusicTransformer(**TINY).state_dict()
        model, targets = build_lora_model_from_state(self.CFG, plain)
        self.assertEqual(targets, ["out_proj"])
        trainable = {k for k, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable and all(".out_proj.lora_" in k for k in trainable))

    def test_export_infers_targets_and_rejects_wrong_declaration(self) -> None:
        from scripts.export_lora_snapshot import main as export_main

        trained = self._qkv_model()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # base that already carries QKV (the M1 case) plus a matching snapshot
            torch.save({"model_config": self.CFG, "model_state_dict": trained.state_dict()},
                       tmp / "b_base.pt")
            torch.save(self._lora(trained), tmp / "snap.pt")
            out = tmp / "export" / "checkpoint_update1.pt"
            args = ["--base", str(tmp / "b_base.pt"), "--snapshot", str(tmp / "snap.pt"),
                    "--output", str(out)]
            self.assertEqual(export_main(args), 0)
            saved = torch.load(out, map_location="cpu", weights_only=False)
            self.assertEqual(saved["model_config"]["lora_targets"], ["out_proj", "qkv"])
            loaded = load_model_with_lora(lora_path=str(out.parent), checkpoint_path=str(out),
                                          prefer_full_checkpoint=True)
            x = torch.randint(0, 300, (1, 12))
            with torch.no_grad():
                self.assertTrue(torch.equal(trained(x), loaded(x)))
            with self.assertRaises(SystemExit):
                export_main(["--base", str(tmp / "b_base.pt"), "--snapshot", str(tmp / "snap.pt"),
                             "--output", str(tmp / "other.pt"), "--lora-targets", "out_proj"])
