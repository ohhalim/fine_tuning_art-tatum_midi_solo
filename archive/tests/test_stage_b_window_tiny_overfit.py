from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.run_stage_b_window_tiny_overfit import token_stats
from utilities.constants import VOCAB_SIZE


class StageBWindowTinyOverfitTest(unittest.TestCase):
    def test_token_stats_rejects_empty_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stats = token_stats(Path(tmp_dir))

            self.assertEqual(stats["files"], 0)
            self.assertFalse(stats["has_tokenized_records"])
            self.assertFalse(stats["fits_vocab"])

    def test_token_stats_accepts_stage_b_tokens_within_model_vocab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_dir = Path(tmp_dir) / "train"
            train_dir.mkdir(parents=True)
            np.save(train_dir / "00000.npy", np.array([0, VOCAB_SIZE - 1], dtype=np.int32))

            stats = token_stats(Path(tmp_dir))

            self.assertEqual(stats["files"], 1)
            self.assertEqual(stats["non_empty_files"], 1)
            self.assertEqual(stats["max_token_id"], VOCAB_SIZE - 1)
            self.assertTrue(stats["has_tokenized_records"])
            self.assertTrue(stats["fits_vocab"])


if __name__ == "__main__":
    unittest.main()
