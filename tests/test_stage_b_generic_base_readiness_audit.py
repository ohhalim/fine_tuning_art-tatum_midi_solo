from __future__ import annotations

import unittest
from pathlib import Path

from scripts.assess_stage_b_generic_base_readiness import (
    StageBGenericBaseReadinessError,
    build_readiness_report,
    validate_readiness_report,
)


def dataset_audit(*, non_brad: int = 2703, brad: int = 72, duplicate_groups: int = 0) -> dict:
    return {
        "input_dir": "midi_dataset/midi",
        "summary": {
            "readable_file_count": 2777,
            "unreadable_file_count": 0,
            "candidate_file_count": 2775,
            "candidate_non_brad_file_count": non_brad,
            "candidate_brad_file_count": brad,
            "duplicate_exact_hash_group_count": duplicate_groups,
            "duplicate_exact_file_count": 0,
        },
    }


def final_decision(*, broad_claim: bool = False, brad_claim: bool = False, boundary: str = "") -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_final_decision_v1",
        "decision": {
            "final_boundary": boundary or "outside_soloing_repair_objective_path_complete",
            "next_boundary": "stage_b_model_core_evidence_readme_refresh",
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
        },
        "objective_repeatability": {
            "source_candidate_count": 2,
            "qualified_source_candidate_count": 2,
            "supported_repair_policy_count": 3,
            "total_qualified_variant_count": 6,
        },
        "claim_boundary": {
            "outside_soloing_repair_objective_path_claimed": True,
            "human_audio_preference_claimed": False,
            "multi_reviewer_preference_claimed": False,
            "broad_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": brad_claim,
            "production_ready_improviser_claimed": False,
        },
    }


class StageBGenericBaseReadinessAuditTest(unittest.TestCase):
    def test_marks_phase4_prep_ready_without_claiming_broad_quality(self) -> None:
        report = build_readiness_report(
            dataset_audit(),
            final_decision(),
            output_dir=Path("outputs/readiness"),
            min_non_brad_candidates=1000,
            min_brad_holdout_candidates=20,
        )
        summary = validate_readiness_report(
            report,
            expected_boundary="stage_b_generic_base_readiness_audit",
            expected_next_boundary="stage_b_generic_base_manifest_contract",
            require_phase4_prep_ready=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertTrue(summary["phase4_prep_ready"])
        self.assertTrue(summary["generic_candidate_pool_ready"])
        self.assertTrue(summary["brad_holdout_available"])
        self.assertTrue(summary["stage_b_objective_path_ready"])
        self.assertFalse(summary["broad_training_execution_ready"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])
        self.assertFalse(summary["brad_style_adaptation_claimed"])

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseReadinessError):
            build_readiness_report(
                dataset_audit(),
                final_decision(broad_claim=True),
                output_dir=Path("outputs/readiness"),
                min_non_brad_candidates=1000,
                min_brad_holdout_candidates=20,
            )

    def test_rejects_brad_style_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseReadinessError):
            build_readiness_report(
                dataset_audit(),
                final_decision(brad_claim=True),
                output_dir=Path("outputs/readiness"),
                min_non_brad_candidates=1000,
                min_brad_holdout_candidates=20,
            )

    def test_requires_objective_final_boundary(self) -> None:
        with self.assertRaises(StageBGenericBaseReadinessError):
            build_readiness_report(
                dataset_audit(),
                final_decision(boundary="stage_b_model_core_evidence_readme_refresh"),
                output_dir=Path("outputs/readiness"),
                min_non_brad_candidates=1000,
                min_brad_holdout_candidates=20,
            )

    def test_phase4_prep_not_ready_when_generic_pool_is_too_small(self) -> None:
        report = build_readiness_report(
            dataset_audit(non_brad=20),
            final_decision(),
            output_dir=Path("outputs/readiness"),
            min_non_brad_candidates=1000,
            min_brad_holdout_candidates=20,
        )

        self.assertFalse(report["dataset_pool"]["generic_candidate_pool_ready"])
        self.assertFalse(report["readiness"]["phase4_prep_ready"])


if __name__ == "__main__":
    unittest.main()
