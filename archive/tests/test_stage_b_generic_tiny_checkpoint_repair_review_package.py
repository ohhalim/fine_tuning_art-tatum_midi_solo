from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_generic_tiny_checkpoint_repair_review_package import (
    StageBGenericTinyCheckpointRepairReviewPackageError,
    build_candidate_row,
    build_review_package_report,
    sort_candidates,
    validate_review_package_report,
)


def sample_row(
    *,
    sample_index: int,
    seed: int,
    strict: bool = True,
    dead_air: float = 0.5,
    coverage: float = 0.8,
    failure: str | None = None,
) -> dict:
    return {
        "sample_index": sample_index,
        "sample_seed": seed,
        "midi_path": f"/tmp/sample_{sample_index}.mid",
        "valid": strict,
        "strict_valid": strict,
        "grammar_gate_passed": True,
        "diagnostic_failure_reason": failure,
        "metrics": {
            "note_count": 7,
            "unique_pitch_count": 6,
            "dead_air_ratio": dead_air,
            "phrase_coverage_ratio": coverage,
            "chord_tone_ratio": 0.4,
            "max_simultaneous_notes": 2,
            "max_note_duration_ratio": 0.15,
        },
        "temporal_coverage": {
            "onset_coverage_ratio": 0.2,
            "sustained_coverage_ratio": 0.4,
        },
        "phrase_contour": {
            "adjacent_repeated_pitch_ratio": 0.0,
            "direction_change_ratio": 0.8,
        },
        "pitch_roles": {
            "root_tone_ratio": 0.1,
            "tension_ratio": 0.2,
        },
        "rhythm_profile": {
            "syncopated_onset_ratio": 0.75,
        },
        "collapse": {
            "postprocess_removal_ratio": 0.1,
        },
    }


def package_report(*, ready: bool = True, musical_claim: bool = False, candidate_count: int = 5) -> dict:
    return {
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_review_package",
            "review_package_ready": ready,
            "musical_quality_claimed": musical_claim,
            "raw_generation_quality_claimed": False,
            "constrained_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_listening_notes",
            "critical_user_input_required": False,
        },
        "source_summary": {
            "sample_count": 6,
            "strict_valid_sample_count": 5,
            "grammar_gate_sample_count": 6,
        },
        "review_package": {
            "candidate_count": candidate_count,
            "failed_candidate_count": 1,
        },
        "next_recommended_issue": "Stage B generic tiny checkpoint repair listening notes",
    }


class StageBGenericTinyCheckpointRepairReviewPackageTest(unittest.TestCase):
    def test_sorts_candidates_for_review_priority(self) -> None:
        rows = [
            build_candidate_row(sample_row(sample_index=1, seed=42, dead_air=0.6, coverage=0.9)),
            build_candidate_row(sample_row(sample_index=2, seed=43, dead_air=0.4, coverage=0.7)),
            build_candidate_row(sample_row(sample_index=3, seed=44, dead_air=0.4, coverage=0.9)),
        ]

        sorted_rows = sort_candidates(rows)

        self.assertEqual([row["sample_index"] for row in sorted_rows], [3, 2, 1])

    def test_accepts_ready_package_without_quality_claim(self) -> None:
        summary = validate_review_package_report(
            package_report(),
            expected_boundary="stage_b_generic_tiny_checkpoint_repair_review_package",
            require_review_package_ready=True,
            require_no_musical_quality_claim=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 5)
        self.assertTrue(summary["review_package_ready"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])

    def test_rejects_musical_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairReviewPackageError):
            validate_review_package_report(
                package_report(musical_claim=True),
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_review_package",
                require_review_package_ready=True,
                require_no_musical_quality_claim=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_empty_package(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairReviewPackageError):
            validate_review_package_report(
                package_report(ready=False, candidate_count=0),
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_review_package",
                require_review_package_ready=True,
                require_no_musical_quality_claim=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_build_report_routes_ready_package_to_listening_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = argparse.Namespace(issue_number=399, min_candidate_count=5)
            candidates = [
                build_candidate_row(sample_row(sample_index=index, seed=40 + index), package_midi_path=root / f"{index}.mid")
                for index in range(1, 6)
            ]
            report = build_review_package_report(
                run_dir=root,
                repeatability_report_path=root / "repeatability.json",
                repeatability_report={
                    "generation": {
                        "report_path": str(root / "generation.json"),
                        "summary": {
                            "sample_count": 6,
                            "valid_sample_count": 5,
                            "strict_valid_sample_count": 5,
                            "grammar_gate_sample_count": 6,
                        },
                    }
                },
                generation_report={},
                candidates=candidates,
                failed_rows=[build_candidate_row(sample_row(sample_index=6, seed=46, strict=False, failure="dead-air"))],
                args=args,
            )

        self.assertTrue(report["readiness"]["review_package_ready"])
        self.assertEqual(report["decision"]["next_boundary"], "stage_b_generic_tiny_checkpoint_repair_listening_notes")
        self.assertFalse(report["readiness"]["musical_quality_claimed"])


if __name__ == "__main__":
    unittest.main()
