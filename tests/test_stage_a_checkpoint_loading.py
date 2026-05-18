from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.generate import infer_checkpoint_max_sequence, resolve_full_checkpoint_path


class StageACheckpointLoadingTest(unittest.TestCase):
    def test_resolves_latest_epoch_checkpoint_from_lora_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            older = root / "checkpoint_epoch1.pt"
            newer = root / "checkpoint_epoch12.pt"
            unrelated = root / "lora_weights.pt"
            older.write_bytes(b"old")
            newer.write_bytes(b"new")
            unrelated.write_bytes(b"lora")

            resolved = resolve_full_checkpoint_path(root)

        self.assertEqual(resolved, newer)

    def test_explicit_checkpoint_path_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            auto_checkpoint = root / "checkpoint_epoch12.pt"
            explicit_checkpoint = root / "custom_full_checkpoint.pt"
            auto_checkpoint.write_bytes(b"auto")
            explicit_checkpoint.write_bytes(b"explicit")

            resolved = resolve_full_checkpoint_path(root, explicit_checkpoint)

        self.assertEqual(resolved, explicit_checkpoint)

    def test_missing_explicit_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.pt"

            with self.assertRaises(FileNotFoundError):
                resolve_full_checkpoint_path(tmp_dir, missing)

    def test_infers_model_max_sequence_from_checkpoint_state(self) -> None:
        class FakeTensor:
            shape = (512, 1, 512)

        state_dict = {"positional_encoding.pe": FakeTensor()}

        self.assertEqual(infer_checkpoint_max_sequence(state_dict, fallback=256), 512)


if __name__ == "__main__":
    unittest.main()
