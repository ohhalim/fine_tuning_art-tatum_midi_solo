from __future__ import annotations

import unittest
from pathlib import Path

from scripts.summarize_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep import (
    StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError,
    build_broader_repeatability_sweep_report,
    validate_broader_repeatability_sweep,
)


def next_decision(*, critical: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_next_decision_v1",
        "decision": {
            "next_boundary": "outside_soloing_repair_broader_repeatability_sweep",
            "auto_progress_allowed": True,
            "critical_user_input_required": critical,
        },
        "claim_boundary": {
            "boundary": "outside_soloing_repair_next_decision",
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": False,
        },
    }


def variant(policy: str, *, max_interval: int = 7) -> dict:
    return {
        "candidate_id": f"candidate_{policy}",
        "repair_policy": policy,
        "outside_soloing_gate": {"qualified": True, "flags": []},
        "metrics": {
            "dead_air_ratio": 0.3333333333333333,
            "chord_tone_ratio": 1.0,
        },
        "focused_solo_metrics": {
            "focused_max_interval": max_interval,
            "focused_unique_pitch_count": 10,
        },
        "pitch_role_metrics": {
            "max_non_chord_tone_run": 0,
        },
    }


def repair_sweep(*, broad_claim: bool = False) -> dict:
    policies = ["chord_tone_snap", "guide_tone_landing", "contour_resolution"]
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_sweep_v1",
        "repair_summary": {
            "boundary": "outside_soloing_pitch_role_repair_candidates",
            "source_candidate_count": 2,
            "repaired_source_candidate_count": 2,
            "total_variant_count": 6,
            "total_qualified_variant_count": 6,
            "broad_model_quality_claimed": False,
        },
        "claim_boundary": {
            "boundary": "outside_soloing_pitch_role_repair_candidates",
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": broad_claim,
        },
        "source_repair_results": [
            {
                "source_candidate_id": "source_155",
                "sample_seed": 155,
                "source_selected_dead_air_ratio": 0.3333333333333333,
                "variants": [variant(policy, max_interval=7) for policy in policies],
            },
            {
                "source_candidate_id": "source_131",
                "sample_seed": 131,
                "source_selected_dead_air_ratio": 0.35294117647058826,
                "variants": [variant(policy, max_interval=5) for policy in policies],
            },
        ],
    }


class StageBDurationCoverageFillOutsideSoloingRepairBroaderRepeatabilitySweepTest(unittest.TestCase):
    def test_records_policy_repeatability_support(self) -> None:
        report = build_broader_repeatability_sweep_report(
            next_decision=next_decision(),
            repair_sweep=repair_sweep(),
            output_dir=Path("outputs/repeatability"),
            min_source_candidates=2,
            min_policy_repeatability_count=3,
            min_source_candidates_per_policy=2,
            min_chord_tone_ratio=0.72,
            max_non_chord_run=1,
            max_interval=7,
        )
        summary = validate_broader_repeatability_sweep(
            report,
            expected_boundary="outside_soloing_repair_policy_repeatability_support",
            min_source_candidates=2,
            min_policy_repeatability_count=3,
            require_no_preference_claim=True,
            require_no_broad_quality_claim=True,
        )

        self.assertEqual(summary["source_candidate_count"], 2)
        self.assertEqual(summary["repair_policy_count"], 3)
        self.assertEqual(summary["supported_repair_policy_count"], 3)
        self.assertEqual(summary["total_variant_count"], 6)
        self.assertEqual(summary["total_qualified_variant_count"], 6)
        self.assertFalse(summary["human_audio_preference_claimed"])
        for policy in report["policy_summaries"]:
            self.assertTrue(policy["policy_repeatability_supported"])

    def test_rejects_critical_next_decision(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError):
            build_broader_repeatability_sweep_report(
                next_decision=next_decision(critical=True),
                repair_sweep=repair_sweep(),
                output_dir=Path("outputs/repeatability"),
                min_source_candidates=2,
                min_policy_repeatability_count=3,
                min_source_candidates_per_policy=2,
                min_chord_tone_ratio=0.72,
                max_non_chord_run=1,
                max_interval=7,
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairBroaderRepeatabilitySweepError):
            build_broader_repeatability_sweep_report(
                next_decision=next_decision(),
                repair_sweep=repair_sweep(broad_claim=True),
                output_dir=Path("outputs/repeatability"),
                min_source_candidates=2,
                min_policy_repeatability_count=3,
                min_source_candidates_per_policy=2,
                min_chord_tone_ratio=0.72,
                max_non_chord_run=1,
                max_interval=7,
            )


if __name__ == "__main__":
    unittest.main()
