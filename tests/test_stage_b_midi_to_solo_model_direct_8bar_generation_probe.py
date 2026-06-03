from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_model_direct_8bar_generation_probe import (
    BOUNDARY,
    FAIL_NEXT_BOUNDARY,
    StageBMidiToSoloModelDirect8BarGenerationProbeError,
    build_direct_8bar_generation_probe_report,
    validate_direct_8bar_generation_probe_report,
)


def sequence_budget_repair_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke",
        "repair_result": {
            "previous_max_sequence": 96,
            "repaired_max_sequence": 160,
            "minimum_contract_tokens": 123,
            "repaired_direct_note_capacity": 33,
            "target_min_note_count": 24,
        },
        "readiness": {
            "model_direct_8bar_generation_probe_ready": True,
            "model_direct_generation_quality_claimed": quality_claim,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def context_report() -> dict:
    bars = [
        ("C", "maj7"),
        ("F", "dom7"),
        ("G", "dom7"),
        ("C", "maj7"),
        ("C", "maj7"),
        ("C", "maj7"),
        ("C", "maj7"),
        ("C", "maj7"),
    ]
    return {
        "boundary": "stage_b_midi_to_solo_context_extraction_mvp",
        "summary": {
            "context_bars": 8,
            "context_event_count": 128,
            "unknown_chord_bar_count": 0,
            "low_confidence_bar_count": 4,
        },
        "context": {
            "bar_contexts": [
                {
                    "bar_index": index,
                    "tempo": 120.0,
                    "chord_root": root,
                    "chord_quality": quality,
                }
                for index, (root, quality) in enumerate(bars)
            ]
        },
        "readiness": {"context_extraction_completed": True},
    }


def scale_smoke_report(checkpoint_dir: Path, *, broad_claim: bool = False) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_epoch1.pt").write_bytes(b"stub")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "readiness": {
            "boundary": "stage_b_generic_base_training_scale_smoke",
            "training_scale_smoke_passed": True,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "training_config": {"max_sequence": 160},
        "training": {"best_validation_loss": 6.1293},
        "artifacts": {
            "checkpoint_count": 1,
            "lora_weights_exists": True,
        },
    }


def generation_report(root: Path, *, missing_midi: bool = False) -> dict:
    paths: list[str] = []
    for index in range(1, 4):
        path = root / f"stage_b_sample_{index}.mid"
        if not missing_midi:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"midi")
        paths.append(str(path))
    return {
        "summary": {
            "sample_count": 3,
            "valid_sample_count": 0,
            "strict_valid_sample_count": 0,
            "grammar_gate_sample_count": 3,
            "passed_generation_gate": False,
            "passed_grammar_gate": True,
            "passed_strict_review_gate": False,
            "collapse_warning_sample_count": 3,
            "collapse_warning_sample_rate": 1.0,
            "avg_postprocess_removal_ratio": 0.541667,
            "max_postprocess_removal_ratio": 0.583333,
            "avg_onset_coverage_ratio": 0.1875,
            "avg_sustained_coverage_ratio": 0.908854,
            "diagnostic_failure_reasons": {
                "note count too low: 11 < 24; collapse=postprocess_removed_majority": 1,
            },
            "strict_failure_reasons": {
                "postprocess removal ratio too high: 0.542 > 0.490": 1,
            },
        },
        "samples": [
            {
                "midi_path": paths[index - 1],
                "grammar": {"complete_note_groups": 24},
                "postprocess": {
                    "before_note_count": 24,
                    "after_note_count": 9 + index,
                },
                "metrics": {"note_count": 9 + index},
            }
            for index in range(1, 4)
        ],
    }


class StageBMidiToSoloModelDirect8BarGenerationProbeTest(unittest.TestCase):
    def test_records_direct_generated_midi_and_review_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_direct_8bar_generation_probe_report(
                sequence_budget_repair=sequence_budget_repair_report(),
                context_report=context_report(),
                repaired_training_scale_smoke=scale_smoke_report(root / "checkpoints"),
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report=generation_report(root / "samples"),
                generation_report_path=root / "generation" / "report.json",
                output_dir=root / "out",
                issue_number=497,
                target_bars=8,
                note_groups_per_bar=3,
            )
            summary = validate_direct_8bar_generation_probe_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=FAIL_NEXT_BOUNDARY,
                require_probe_completed=True,
                require_generated_midi=True,
                require_grammar_gate=True,
                require_no_quality_claim=True,
            )

        self.assertEqual(summary["generation_source"], "model_checkpoint_direct_constrained")
        self.assertEqual(summary["target_bars"], 8)
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["grammar_gate_sample_count"], 3)
        self.assertEqual(summary["valid_sample_count"], 0)
        self.assertTrue(summary["direct_generated_midi_written"])
        self.assertTrue(summary["direct_generation_grammar_gate_passed"])
        self.assertFalse(summary["direct_generation_review_gate_passed"])
        self.assertEqual(summary["min_pre_postprocess_note_groups"], 24)
        self.assertEqual(summary["min_postprocess_note_count"], 10)
        self.assertEqual(summary["max_postprocess_note_count"], 12)
        self.assertFalse(summary["model_direct_generation_quality_claimed"])

    def test_rejects_missing_generated_midi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_direct_8bar_generation_probe_report(
                sequence_budget_repair=sequence_budget_repair_report(),
                context_report=context_report(),
                repaired_training_scale_smoke=scale_smoke_report(root / "checkpoints"),
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report=generation_report(root / "samples", missing_midi=True),
                generation_report_path=root / "generation" / "report.json",
                output_dir=root / "out",
                issue_number=497,
                target_bars=8,
                note_groups_per_bar=3,
            )
            with self.assertRaises(StageBMidiToSoloModelDirect8BarGenerationProbeError):
                validate_direct_8bar_generation_probe_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=FAIL_NEXT_BOUNDARY,
                    require_probe_completed=True,
                    require_generated_midi=True,
                    require_grammar_gate=True,
                    require_no_quality_claim=True,
                )

    def test_rejects_upstream_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_direct_8bar_generation_probe_report(
                sequence_budget_repair=sequence_budget_repair_report(quality_claim=True),
                context_report=context_report(),
                repaired_training_scale_smoke=scale_smoke_report(root / "checkpoints"),
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report=generation_report(root / "samples"),
                generation_report_path=root / "generation" / "report.json",
                output_dir=root / "out",
                issue_number=497,
                target_bars=8,
                note_groups_per_bar=3,
            )
            with self.assertRaises(StageBMidiToSoloModelDirect8BarGenerationProbeError):
                validate_direct_8bar_generation_probe_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=FAIL_NEXT_BOUNDARY,
                    require_probe_completed=True,
                    require_generated_midi=True,
                    require_grammar_gate=True,
                    require_no_quality_claim=True,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_direct_8bar_generation_probe")
        self.assertEqual(FAIL_NEXT_BOUNDARY, "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair")


if __name__ == "__main__":
    unittest.main()
