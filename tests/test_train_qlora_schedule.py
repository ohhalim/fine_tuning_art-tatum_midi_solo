from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "music_transformer"))

from utilities.device import use_cuda

use_cuda(False)

from scripts.train_qlora import evaluate, optimizer_updates_per_epoch, scheduler_total_steps


class SchedulerStepsTest(unittest.TestCase):
    def test_updates_per_epoch(self) -> None:
        self.assertEqual(optimizer_updates_per_epoch(4, 4), 1)
        self.assertEqual(optimizer_updates_per_epoch(5, 4), 2)
        self.assertEqual(optimizer_updates_per_epoch(4, 1), 4)

    def test_optimizer_update_mode_matches_actual_updates(self) -> None:
        # Mehldau run: 4 batches, accumulation 4, 8 epochs -> 8 updates.
        self.assertEqual(scheduler_total_steps(4, 4, 8), 8)

    def test_legacy_mode_counts_batches(self) -> None:
        self.assertEqual(scheduler_total_steps(4, 4, 8, "legacy_batches"), 32)

    def test_cosine_reaches_floor_in_optimizer_update_mode(self) -> None:
        opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=3e-4)
        t_max = scheduler_total_steps(4, 4, 8)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=1e-6)
        for _ in range(8):
            opt.step()
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-6, places=9)

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            scheduler_total_steps(4, 4, 8, "nope")


class RandomCropDataset(torch.utils.data.Dataset):
    """Mimics MidiDataset: a random crop start per item from ``random``."""

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        start = random.randint(0, 100)
        x = torch.full((4,), start % 10, dtype=torch.long)
        return x, x


class ScoreModel(nn.Module):
    def forward(self, x):
        logits = torch.zeros(*x.shape, 10)
        logits[..., 0] = x.float()
        return logits


class DeterministicValTest(unittest.TestCase):
    def _loss(self, crop_seed):
        loader = torch.utils.data.DataLoader(RandomCropDataset(), batch_size=3)
        return evaluate(ScoreModel(), loader, nn.CrossEntropyLoss(), torch.device("cpu"),
                        crop_seed=crop_seed)

    def test_seeded_val_is_repeatable(self) -> None:
        random.seed(1)
        a = self._loss(0)
        random.seed(2)
        b = self._loss(0)
        self.assertEqual(a, b)

    def test_seeded_val_restores_training_random_stream(self) -> None:
        random.seed(5)
        expected = [random.random() for _ in range(3)]
        random.seed(5)
        self._loss(0)
        self.assertEqual([random.random() for _ in range(3)], expected)


if __name__ == "__main__":
    unittest.main()
