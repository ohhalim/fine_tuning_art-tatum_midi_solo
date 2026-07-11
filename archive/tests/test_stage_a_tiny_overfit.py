from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pretty_midi

from inference.app.metrics import compute_midi_metrics, validate_metrics
from inference.app.schemas import GenerationRequest
from scripts.compare_stage_a_tiny_modes import build_decision
from scripts.control_tokens import SEQUENCE_FORMAT_CONTROL_V1, control_prefix_tokens
from scripts.run_control_v1_tiny_overfit import prepare_control_v1_tiny_dataset
from scripts.run_stage_a_tiny_overfit import (
    parse_best_validation_loss,
    prepare_tiny_dataset,
    summarize_report,
)


class StageATinyOverfitDatasetTest(unittest.TestCase):
    def test_parse_best_validation_loss_uses_last_logged_value(self) -> None:
        text = "Best validation loss: 4.8228\nother\nBest validation loss: 0.0568\n"

        self.assertEqual(parse_best_validation_loss(text), 0.0568)

    def test_summarize_report_extracts_gate_and_training_fields(self) -> None:
        report = {
            "training_mode": "full_model_tiny",
            "train_result": {"stdout_tail": "Best validation loss: 0.0568"},
            "raw_sample_metrics": [{"valid": True}, {"valid": False}],
            "inference_result": {
                "status": "COMPLETED",
                "fallback_used": False,
                "model_failure_reason": None,
            },
        }

        summary = summarize_report(report)

        self.assertTrue(summary["passed_mvp_gate"])
        self.assertEqual(summary["best_validation_loss"], 0.0568)
        self.assertEqual(summary["valid_raw_sample_count"], 1)
        self.assertEqual(summary["raw_sample_count"], 2)

    def test_compare_decision_rejects_random_base_lora_only_when_full_model_passes(self) -> None:
        decision = build_decision(
            {"passed_mvp_gate": True},
            {"passed_mvp_gate": False},
        )

        self.assertIn("do not rely on random-base LoRA-only", decision)

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

    def test_prepare_control_v1_tiny_dataset_writes_control_prompt_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = prepare_control_v1_tiny_dataset(Path(tmp_dir), sample_count=1, bpm=124)
            first_sample = manifest["samples"][0]
            tokens = np.load(first_sample["train_tokens_path"])

            self.assertEqual(manifest["sequence_format"], SEQUENCE_FORMAT_CONTROL_V1)
            self.assertEqual(tokens[:3].tolist(), control_prefix_tokens("lead", 124))
            self.assertGreater(first_sample["conditioning_token_count"], 0)
            self.assertGreater(first_sample["target_token_count"], 0)
            self.assertGreater(first_sample["token_count"], first_sample["target_token_count"])


if __name__ == "__main__":
    unittest.main()
