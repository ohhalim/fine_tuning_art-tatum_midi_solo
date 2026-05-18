from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pretty_midi

from inference.app.metrics import compute_midi_metrics, validate_metrics
from inference.app.schemas import GenerationRequest
from scripts.run_stage_a_tiny_overfit import prepare_tiny_dataset


class StageATinyOverfitDatasetTest(unittest.TestCase):
    def test_prepare_tiny_dataset_writes_known_good_midi_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = prepare_tiny_dataset(Path(tmp_dir), sample_count=2, bpm=124)

            midi_paths = [Path(sample["midi_path"]) for sample in manifest["samples"]]
            train_paths = [Path(sample["train_tokens_path"]) for sample in manifest["samples"]]
            val_paths = [Path(sample["val_tokens_path"]) for sample in manifest["samples"]]

            self.assertEqual(manifest["sample_count"], 2)
            self.assertTrue(all(path.exists() for path in midi_paths))
            self.assertTrue(all(path.exists() for path in train_paths))
            self.assertTrue(all(path.exists() for path in val_paths))

            first_midi = pretty_midi.PrettyMIDI(str(midi_paths[0]))
            notes = first_midi.instruments[0].notes
            self.assertGreaterEqual(len(notes), 24)
            self.assertTrue(all(note.end > note.start for note in notes))

            first_tokens = np.load(train_paths[0])
            self.assertGreater(len(first_tokens), 50)

            request = GenerationRequest(
                bpm=124,
                chord_progression=["Cm7", "Fm7", "Bb7", "Ebmaj7"],
                bars=2,
                density="medium",
            )
            metrics = compute_midi_metrics(midi_paths[0], 0, False, request=request)
            valid, reason = validate_metrics(metrics, "medium", bars=2)
            self.assertTrue(valid, reason)


if __name__ == "__main__":
    unittest.main()
