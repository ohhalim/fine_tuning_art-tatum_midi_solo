from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation import (
    StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError,
    build_objective_evidence_consolidation,
    validate_objective_evidence_consolidation,
)


def repair_sweep(*, broad_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_sweep_v1",
        "repair_summary": {
            "boundary": "outside_soloing_pitch_role_repair_candidates",
            "repaired_source_candidate_count": 2,
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": "outside_soloing_pitch_role_repair_candidates",
            "broad_model_quality_claimed": broad_claim,
        },
        "source_repair_results": [
            {
                "source_candidate_id": "source_155",
                "sample_seed": 155,
                "source_selected_dead_air_ratio": 0.3333333333333333,
                "source_selected_max_interval": 6,
                "selected_candidate": {
                    "candidate_id": "source_155_repaired",
                    "repair_policy": "contour_resolution",
                    "midi_path": "outputs/source_155_repaired.mid",
                    "outside_soloing_gate": {"qualified": True, "flags": []},
                    "metrics": {
                        "dead_air_ratio": 0.3333333333333333,
                        "chord_tone_ratio": 1.0,
                    },
                    "focused_solo_metrics": {
                        "focused_unique_pitch_count": 10,
                        "focused_max_interval": 7,
                    },
                    "pitch_role_metrics": {
                        "max_non_chord_tone_run": 0,
                    },
                },
            },
            {
                "source_candidate_id": "source_131",
                "sample_seed": 131,
                "source_selected_dead_air_ratio": 0.35294117647058826,
                "source_selected_max_interval": 11,
                "selected_candidate": {
                    "candidate_id": "source_131_repaired",
                    "repair_policy": "contour_resolution",
                    "midi_path": "outputs/source_131_repaired.mid",
                    "outside_soloing_gate": {"qualified": True, "flags": []},
                    "metrics": {
                        "dead_air_ratio": 0.35294117647058826,
                        "chord_tone_ratio": 1.0,
                    },
                    "focused_solo_metrics": {
                        "focused_unique_pitch_count": 9,
                        "focused_max_interval": 5,
                    },
                    "pitch_role_metrics": {
                        "max_non_chord_tone_run": 0,
                    },
                },
            },
        ],
    }


def user_review_fill(*, preference_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill_v1",
        "claim_boundary": {
            "boundary": "outside_soloing_repair_audio_review_pending",
            "human_audio_preference_claimed": preference_claim,
            "broad_model_quality_claimed": False,
        },
        "decision": {
            "objective_auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
    }


class StageBDurationCoverageFillOutsideSoloingRepairObjectiveEvidenceConsolidationTest(unittest.TestCase):
    def test_consolidates_objective_evidence_without_preference_claim(self) -> None:
        report = build_objective_evidence_consolidation(
            repair_sweep=repair_sweep(),
            user_review_fill=user_review_fill(),
            output_dir=Path("outputs/objective_consolidation"),
            min_repaired_source_candidates=2,
            min_chord_tone_ratio=0.72,
            max_non_chord_run=1,
            max_interval=7,
        )
        summary = validate_objective_evidence_consolidation(
            report,
            expected_boundary="outside_soloing_repair_objective_evidence_support",
            min_repaired_source_candidates=2,
            require_no_preference_claim=True,
            require_no_broad_quality_claim=True,
        )

        self.assertEqual(summary["source_candidate_count"], 2)
        self.assertEqual(summary["qualified_source_candidate_count"], 2)
        self.assertEqual(summary["dead_air_preserved_source_candidate_count"], 2)
        self.assertEqual(summary["chord_tone_pass_source_candidate_count"], 2)
        self.assertEqual(summary["non_chord_run_pass_source_candidate_count"], 2)
        self.assertEqual(summary["interval_pass_source_candidate_count"], 2)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["broad_model_quality_claimed"])
        self.assertIn("human_audio_preference", report["not_proven"])

    def test_rejects_preference_claim_before_review(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError):
            build_objective_evidence_consolidation(
                repair_sweep=repair_sweep(),
                user_review_fill=user_review_fill(preference_claim=True),
                output_dir=Path("outputs/objective_consolidation"),
                min_repaired_source_candidates=2,
                min_chord_tone_ratio=0.72,
                max_non_chord_run=1,
                max_interval=7,
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairObjectiveEvidenceConsolidationError):
            build_objective_evidence_consolidation(
                repair_sweep=repair_sweep(broad_claim=True),
                user_review_fill=user_review_fill(),
                output_dir=Path("outputs/objective_consolidation"),
                min_repaired_source_candidates=2,
                min_chord_tone_ratio=0.72,
                max_non_chord_run=1,
                max_interval=7,
            )


if __name__ == "__main__":
    unittest.main()
