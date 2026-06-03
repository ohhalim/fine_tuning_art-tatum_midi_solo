from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_model_direct_monophonic_overlap_repair import (
    BOUNDARY,
    PASS_NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectMonophonicOverlapRepairError,
    build_monophonic_overlap_repair_report,
    validate_monophonic_overlap_repair_report,
)


def previous_direct_probe(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": "stage_b_midi_to_solo_model_direct_8bar_generation_probe",
        "generation_summary": {
            "sample_count": 3,
            "valid_sample_count": 0,
            "strict_valid_sample_count": 0,
            "grammar_gate_sample_count": 3,
            "min_pre_postprocess_note_groups": 24,
            "min_postprocess_note_count": 10,
            "max_postprocess_note_count": 12,
            "avg_postprocess_removal_ratio": 0.541667,
            "collapse_warning_sample_rate": 1.0,
        },
        "readiness": {
            "direct_generated_midi_written": True,
            "direct_generation_review_gate_passed": False,
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


def sequence_budget_repair() -> dict:
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
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {"next_boundary": "stage_b_midi_to_solo_model_direct_8bar_generation_probe"},
    }


def context_report() -> dict:
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
                {"bar_index": index, "tempo": 120.0, "chord_root": "C", "chord_quality": "maj7"}
                for index in range(8)
            ]
        },
        "readiness": {"context_extraction_completed": True},
    }


def scale_smoke(checkpoint_dir: Path) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_epoch1.pt").write_bytes(b"stub")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "readiness": {
            "boundary": "stage_b_generic_base_training_scale_smoke",
            "training_scale_smoke_passed": True,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "training_config": {"max_sequence": 160},
        "training": {"best_validation_loss": 6.1293},
        "artifacts": {"checkpoint_count": 1, "lora_weights_exists": True},
    }


def repaired_generation(root: Path, *, valid_count: int = 3) -> dict:
    paths: list[str] = []
    for index in range(1, 4):
        path = root / f"stage_b_sample_{index}.mid"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"midi")
        paths.append(str(path))
    passed = valid_count == 3
    return {
        "summary": {
            "sample_count": 3,
            "valid_sample_count": valid_count,
            "strict_valid_sample_count": valid_count,
            "grammar_gate_sample_count": 3,
            "passed_generation_gate": passed,
            "passed_grammar_gate": True,
            "passed_strict_review_gate": passed,
            "collapse_warning_sample_count": 0,
            "collapse_warning_sample_rate": 0.0,
            "avg_postprocess_removal_ratio": 0.0,
            "max_postprocess_removal_ratio": 0.0,
            "avg_onset_coverage_ratio": 0.1875,
            "avg_sustained_coverage_ratio": 0.6354,
        },
        "samples": [
            {
                "midi_path": paths[index - 1],
                "grammar": {"complete_note_groups": 24},
                "postprocess": {
                    "before_note_count": 24,
                    "after_note_count": 24,
                },
                "metrics": {"note_count": 24},
            }
            for index in range(1, 4)
        ],
    }


class StageBMidiToSoloModelDirectMonophonicOverlapRepairTest(unittest.TestCase):
    def test_records_overlap_repair_and_review_gate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_monophonic_overlap_repair_report(
                previous_direct_probe=previous_direct_probe(),
                sequence_budget_repair=sequence_budget_repair(),
                context_report=context_report(),
                repaired_training_scale_smoke=scale_smoke(root / "checkpoints"),
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report=repaired_generation(root / "samples"),
                generation_report_path=root / "generation" / "report.json",
                output_dir=root / "out",
                issue_number=499,
                target_bars=8,
                note_groups_per_bar=3,
            )
            summary = validate_monophonic_overlap_repair_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=PASS_NEXT_BOUNDARY,
                require_repair_completed=True,
                require_review_gate_repaired=True,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["cap_duration_to_next_position"])
        self.assertEqual(summary["previous_valid_sample_count"], 0)
        self.assertEqual(summary["repaired_valid_sample_count"], 3)
        self.assertEqual(summary["previous_avg_postprocess_removal_ratio"], 0.541667)
        self.assertEqual(summary["repaired_avg_postprocess_removal_ratio"], 0.0)
        self.assertEqual(summary["previous_min_postprocess_note_count"], 10)
        self.assertEqual(summary["repaired_min_postprocess_note_count"], 24)
        self.assertTrue(summary["postprocess_removal_reduced"])
        self.assertTrue(summary["direct_generation_review_gate_passed"])
        self.assertFalse(summary["model_direct_generation_quality_claimed"])

    def test_rejects_unrepaired_review_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_monophonic_overlap_repair_report(
                previous_direct_probe=previous_direct_probe(),
                sequence_budget_repair=sequence_budget_repair(),
                context_report=context_report(),
                repaired_training_scale_smoke=scale_smoke(root / "checkpoints"),
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report=repaired_generation(root / "samples", valid_count=0),
                generation_report_path=root / "generation" / "report.json",
                output_dir=root / "out",
                issue_number=499,
                target_bars=8,
                note_groups_per_bar=3,
            )
            with self.assertRaises(StageBMidiToSoloModelDirectMonophonicOverlapRepairError):
                validate_monophonic_overlap_repair_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=PASS_NEXT_BOUNDARY,
                    require_repair_completed=True,
                    require_review_gate_repaired=True,
                    require_no_quality_claim=True,
                )

    def test_rejects_previous_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_monophonic_overlap_repair_report(
                previous_direct_probe=previous_direct_probe(quality_claim=True),
                sequence_budget_repair=sequence_budget_repair(),
                context_report=context_report(),
                repaired_training_scale_smoke=scale_smoke(root / "checkpoints"),
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report=repaired_generation(root / "samples"),
                generation_report_path=root / "generation" / "report.json",
                output_dir=root / "out",
                issue_number=499,
                target_bars=8,
                note_groups_per_bar=3,
            )
            with self.assertRaises(StageBMidiToSoloModelDirectMonophonicOverlapRepairError):
                validate_monophonic_overlap_repair_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=PASS_NEXT_BOUNDARY,
                    require_repair_completed=True,
                    require_review_gate_repaired=True,
                    require_no_quality_claim=True,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_model_direct_monophonic_overlap_repair")
        self.assertEqual(PASS_NEXT_BOUNDARY, "stage_b_midi_to_solo_model_direct_audio_render_package")


if __name__ == "__main__":
    unittest.main()
