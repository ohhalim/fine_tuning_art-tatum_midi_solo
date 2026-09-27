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
