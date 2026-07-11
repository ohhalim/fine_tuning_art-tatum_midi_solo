from __future__ import annotations

import unittest
from pathlib import Path

from scripts.check_stage_b_midi_to_solo_training_resource_probe import (
    BOUNDARY,
    NEXT_BOUNDARY,
    REQUIRED_CONTEXT_FIELDS,
    StageBMidiToSoloTrainingResourceProbeError,
    build_resource_probe_report,
    validate_resource_probe_report,
)


def context_report(*, missing_field: bool = False, final_claim: bool = False) -> dict:
    event = {field: None for field in REQUIRED_CONTEXT_FIELDS}
    event.update(
        {
            "bar_index": 0,
            "position_index": 0,
            "tempo": 120.0,
            "chord_root": "C",
            "chord_quality": "maj7",
            "next_chord_root": "F",
            "next_chord_quality": "dom7",
            "bass_note": 36,
            "chord_confidence": 0.9,
            "chord_source": "pitch_class_inference",
        }
    )
    if missing_field:
        event.pop("chord_source")
    return {
        "boundary": "stage_b_midi_to_solo_context_extraction_mvp",
        "summary": {
            "context_bars": 8,
            "positions_per_bar": 16,
            "context_event_count": 128,
            "inferred_chord_bar_count": 4,
            "carry_forward_chord_bar_count": 4,
            "unknown_chord_bar_count": 0,
            "low_confidence_bar_count": 4,
            "bass_note_bar_count": 4,
        },
        "context": {"context_events": [event] * 128},
        "readiness": {
            "context_extraction_completed": True,
            "required_context_fields_present": True,
            "midi_to_solo_mvp_claimed": final_claim,
            "harmony_analysis_quality_claimed": False,
        },
    }


def full_window_report(*, train_files: int = 154136, quality_claim: bool = False) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_full_manifest_window_preparation",
            "full_manifest_window_preparation_ready": True,
            "full_training_executed": False,
            "broad_trained_model_quality_claimed": quality_claim,
            "brad_style_adaptation_claimed": False,
        },
        "input": {
            "train_file_count": 2433,
            "val_file_count": 270,
            "window_bars": 2,
            "window_stride_bars": 2,
            "min_window_target_notes": 4,
        },
        "dataset_summary": {"sequence_format": "stage_b_v1"},
        "token_stats": {
            "tokenized_train_files": train_files,
            "tokenized_val_files": 21845,
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": True,
        },
    }


def scale_smoke_report(*, checkpoint_count: int = 1, loss: float | None = 5.9031) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_base_training_scale_smoke",
            "training_scale_smoke_passed": True,
            "generic_base_scale_checkpoint_generation_probe_ready": True,
            "full_generic_training_executed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "input": {
            "selected_train_records": 128,
            "selected_val_records": 32,
        },
        "token_stats": {
            "files": 160,
            "max_token_id": 544,
            "vocab_size": 547,
            "fits_vocab": True,
        },
        "training": {
            "returncode": 0,
            "best_validation_loss": loss,
        },
        "artifacts": {
            "checkpoint_count": checkpoint_count,
            "lora_weights_exists": True,
        },
    }


def build_report(**overrides: dict) -> dict:
    return build_resource_probe_report(
        context_report=overrides.get("context_report") or context_report(),
        full_window_report=overrides.get("full_window_report") or full_window_report(),
        scale_smoke_report=overrides.get("scale_smoke_report") or scale_smoke_report(),
        output_dir=Path("outputs/resource_probe"),
        issue_number=485,
    )


class StageBMidiToSoloTrainingResourceProbeTest(unittest.TestCase):
    def test_accepts_ready_training_resource_without_final_claim(self) -> None:
        summary = validate_resource_probe_report(
            build_report(),
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_ready=True,
            require_no_final_claim=True,
            min_context_events=128,
            min_full_train_records=100000,
            min_full_val_records=10000,
            min_scale_train_records=64,
            min_scale_val_records=16,
        )

        self.assertTrue(summary["midi_to_solo_training_resource_ready"])
        self.assertTrue(summary["conditioned_generation_probe_ready"])
        self.assertEqual(summary["context_event_count"], 128)
        self.assertEqual(summary["full_tokenized_train_files"], 154136)
        self.assertEqual(summary["scale_selected_train_records"], 128)
        self.assertEqual(summary["scale_checkpoint_count"], 1)
        self.assertEqual(summary["scale_best_validation_loss"], 5.9031)
        self.assertFalse(summary["midi_to_solo_mvp_claimed"])
        self.assertFalse(summary["conditioned_generation_completed"])
        self.assertFalse(summary["broad_training_executed"])

    def test_rejects_missing_context_field(self) -> None:
        with self.assertRaises(StageBMidiToSoloTrainingResourceProbeError):
            validate_resource_probe_report(
                build_report(context_report=context_report(missing_field=True)),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_ready=True,
                require_no_final_claim=True,
                min_context_events=128,
                min_full_train_records=100000,
                min_full_val_records=10000,
                min_scale_train_records=64,
                min_scale_val_records=16,
            )

    def test_rejects_small_full_window_resource(self) -> None:
        with self.assertRaises(StageBMidiToSoloTrainingResourceProbeError):
            validate_resource_probe_report(
                build_report(full_window_report=full_window_report(train_files=999)),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_ready=True,
                require_no_final_claim=True,
                min_context_events=128,
                min_full_train_records=100000,
                min_full_val_records=10000,
                min_scale_train_records=64,
                min_scale_val_records=16,
            )

    def test_rejects_missing_scale_checkpoint(self) -> None:
        with self.assertRaises(StageBMidiToSoloTrainingResourceProbeError):
            validate_resource_probe_report(
                build_report(scale_smoke_report=scale_smoke_report(checkpoint_count=0)),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_ready=True,
                require_no_final_claim=True,
                min_context_events=128,
                min_full_train_records=100000,
                min_full_val_records=10000,
                min_scale_train_records=64,
                min_scale_val_records=16,
            )

    def test_rejects_upstream_final_claims(self) -> None:
        with self.assertRaises(StageBMidiToSoloTrainingResourceProbeError):
            validate_resource_probe_report(
                build_report(context_report=context_report(final_claim=True)),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_ready=True,
                require_no_final_claim=True,
                min_context_events=128,
                min_full_train_records=100000,
                min_full_val_records=10000,
                min_scale_train_records=64,
                min_scale_val_records=16,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_training_resource_probe")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_conditioned_generation_probe")


if __name__ == "__main__":
    unittest.main()
