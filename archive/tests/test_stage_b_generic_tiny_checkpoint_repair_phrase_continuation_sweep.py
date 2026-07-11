from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_sweep import (
    StageBGenericTinyCheckpointRepairPhraseContinuationSweepError,
    build_sweep_report,
    target_failure_reasons,
    validate_sweep_report,
)


def args() -> argparse.Namespace:
    return argparse.Namespace(
        issue_number=413,
        num_samples=2,
        seed=62,
        temperature=0.78,
        top_k=5,
        note_groups_per_bar=8,
        max_simultaneous_notes=1,
        min_note_count=8,
        min_phrase_coverage_ratio=0.85,
        max_tail_empty_steps=2,
        min_pitch_role_chord_tone_ratio=0.5,
        max_postprocess_removal_ratio=0.49,
        min_target_qualified=1,
    )


def decision_report() -> dict:
    return {
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep",
        },
    }


def sample_row(*, strict: bool = True, tail_empty: int = 2, chord_ratio: float = 0.5625) -> dict:
    return {
        "sample_index": 1,
        "sample_seed": 62,
        "midi_path": "sample.mid",
        "valid": True,
        "strict_valid": strict,
        "grammar_gate_passed": True,
        "metrics": {
            "note_count": 9,
            "phrase_coverage_ratio": 0.90625,
            "dead_air_ratio": 0.625,
            "max_simultaneous_notes": 1,
            "chord_tone_ratio": 0.3333,
        },
        "temporal_coverage": {
            "tail_empty_steps": tail_empty,
            "position_span_ratio": 0.90625,
        },
        "pitch_roles": {
            "chord_tone_ratio": chord_ratio,
        },
        "collapse": {
            "postprocess_removal_ratio": 0.4375,
        },
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationSweepTest(unittest.TestCase):
    def test_target_candidate_passes_phrase_continuation_filters(self) -> None:
        self.assertEqual(target_failure_reasons(sample_row(), args=args()), [])

    def test_tail_empty_failure_is_recorded(self) -> None:
        self.assertIn("tail_empty_above_target", target_failure_reasons(sample_row(tail_empty=4), args=args()))

    def test_validate_rejects_below_target_candidate_count(self) -> None:
        report = build_sweep_report(
            run_dir=Path("outputs/sweep"),
            checkpoint_dir=Path("checkpoints"),
            generation_result={"returncode": 0},
            generation_report_path=Path("report.json"),
            generation_report={
                "summary": {
                    "sample_count": 1,
                    "valid_sample_count": 1,
                    "strict_valid_sample_count": 1,
                    "grammar_gate_sample_count": 1,
                    "collapse_warning_sample_rate": 0.0,
                },
                "samples": [sample_row(chord_ratio=0.25)],
            },
            decision_report_path=Path("decision.json"),
            decision_report=decision_report(),
            args=args(),
        )

        with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationSweepError):
            validate_sweep_report(
                report,
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep",
                min_target_qualified=1,
                require_no_quality_claim=True,
            )

    def test_builds_sweep_report_without_quality_claim(self) -> None:
        report = build_sweep_report(
            run_dir=Path("outputs/sweep"),
            checkpoint_dir=Path("checkpoints"),
            generation_result={"returncode": 0},
            generation_report_path=Path("report.json"),
            generation_report={
                "summary": {
                    "sample_count": 2,
                    "valid_sample_count": 1,
                    "strict_valid_sample_count": 1,
                    "grammar_gate_sample_count": 2,
                    "collapse_warning_sample_rate": 0.0,
                },
                "samples": [sample_row(), sample_row(strict=False)],
            },
            decision_report_path=Path("decision.json"),
            decision_report=decision_report(),
            args=args(),
        )
        summary = validate_sweep_report(
            report,
            expected_boundary="stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep",
            min_target_qualified=1,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["target_qualified_count"], 1)
        self.assertTrue(summary["target_passed"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertEqual(
            summary["next_boundary"],
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_audio_render_package",
        )

    def test_rejects_wrong_decision_boundary(self) -> None:
        bad_decision = {"decision": {"next_boundary": "other"}}
        with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationSweepError):
            build_sweep_report(
                run_dir=Path("outputs/sweep"),
                checkpoint_dir=Path("checkpoints"),
                generation_result={"returncode": 0},
                generation_report_path=Path("report.json"),
                generation_report={"summary": {}, "samples": [sample_row()]},
                decision_report_path=Path("decision.json"),
                decision_report=bad_decision,
                args=args(),
            )


if __name__ == "__main__":
    unittest.main()
