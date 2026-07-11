from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_listening_review_quality_gap import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    SCHEMA_VERSION,
    StageBMidiToSoloListeningReviewQualityGapError,
    build_listening_review_quality_gap_report,
    validate_listening_review_quality_gap_report,
)
from scripts.decide_stage_b_midi_to_solo_quality_gap import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BOUNDARY as QUALITY_GAP_BOUNDARY,
    CURRENT_EVIDENCE_SCHEMA_VERSION,
    LISTENING_REVIEW_NEXT_BOUNDARY,
    LISTENING_REVIEW_TARGET,
    MVP_COMPLETION_AUDIT_SCHEMA_VERSION,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    SCHEMA_VERSION as QUALITY_GAP_DECISION_SCHEMA_VERSION,
)


SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
}


def quality_gap_decision(
    *,
    selected_target: str = LISTENING_REVIEW_TARGET,
    next_boundary: str = LISTENING_REVIEW_NEXT_BOUNDARY,
    quality_claim: bool = False,
    changed_ratio_supported: bool = True,
    outside_soloing_repair_supported: bool = True,
) -> dict:
    return {
        "schema_version": QUALITY_GAP_DECISION_SCHEMA_VERSION,
        "boundary": QUALITY_GAP_BOUNDARY,
        "source_schema_version": MVP_COMPLETION_AUDIT_SCHEMA_VERSION,
        "source_current_evidence_schema_version": CURRENT_EVIDENCE_SCHEMA_VERSION,
        "quality_gap": {
            "technical_model_core_mvp_completed": True,
            "phrase_bank_cli_technical_path_completed": True,
            "model_conditioned_pitch_contour_objective_completed": True,
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed": (
                changed_ratio_supported
            ),
            "outside_soloing_repair_objective_completed": outside_soloing_repair_supported,
            "outside_soloing_repair_source_context_preserved": outside_soloing_repair_supported,
            "outside_soloing_repair_schema_context_preserved": outside_soloing_repair_supported,
            "outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "musical_quality_mvp_completed": False,
            "human_audio_preference_completed": False,
            "product_mvp_completed": False,
            "fallback_path_active": True,
            "model_conditioned_input_path_alignment_required": False,
            "model_conditioned_pitch_contour_objective_path_ready": True,
            "pitch_contour_changed_ratio_review_required": True,
            "pitch_contour_changed_ratio_repair_objective_path_ready": (
                changed_ratio_supported
            ),
            "pitch_contour_changed_ratio_repair_target_supported": changed_ratio_supported,
            "outside_soloing_repair_objective_path_ready": outside_soloing_repair_supported,
            "outside_soloing_repair_target_supported": outside_soloing_repair_supported,
            "outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "outside_soloing_repair_source_targeted": False,
            "outside_soloing_repair_source_residual_risk_preserved": True,
            "human_review_required_now": False,
            **SOURCE_CONTEXT,
        },
        "mvp_completion_summary": {
            "current_evidence_schema_version": CURRENT_EVIDENCE_SCHEMA_VERSION,
            "model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count": 3,
            "model_conditioned_pitch_contour_changed_ratio_repair_max_interval": (
                12 if changed_ratio_supported else 13
            ),
            "model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold": 12,
            "model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio": (
                0.4348 if changed_ratio_supported else 0.7174
            ),
            "model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio": 0.5,
            "outside_soloing_repair_rendered_audio_file_count": 6,
            "outside_soloing_repair_changed_note_total": 2,
            "outside_soloing_repair_pitch_role_risk_count_after": (
                0 if outside_soloing_repair_supported else 1
            ),
            "outside_soloing_repair_pitch_role_risk_delta": 2,
            "outside_soloing_repair_objective_path_supported": outside_soloing_repair_supported,
            "outside_soloing_repair_schema_context_preserved": outside_soloing_repair_supported,
            "outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "outside_soloing_repair_weak_landing_target_supported": True,
            "outside_soloing_repair_final_landing_target_supported": True,
            "outside_soloing_repair_non_chord_run_target_supported": True,
        },
        "selected_target": {
            "selected_target": selected_target,
            "selected_next_boundary": next_boundary,
            "fallback_path_active": True,
            "human_review_required_now": False,
        },
        "readiness": {
            "quality_gap_decision_completed": True,
            "selected_target": selected_target,
            "next_boundary_selected": next_boundary,
            "outside_soloing_repair_source_context_preserved": outside_soloing_repair_supported,
            "outside_soloing_repair_schema_context_preserved": outside_soloing_repair_supported,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloListeningReviewQualityGapTest(unittest.TestCase):
    def test_routes_to_mvp_delivery_package_without_quality_claim(self) -> None:
        report = build_listening_review_quality_gap_report(
            quality_gap_decision=quality_gap_decision(),
            output_dir=Path("outputs/listening_review_quality_gap"),
            issue_number=1158,
        )
        summary = validate_listening_review_quality_gap_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            expected_target=SELECTED_TARGET,
            require_delivery_package_ready=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_target"], SELECTED_TARGET)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            summary["source_schema_version"],
            QUALITY_GAP_DECISION_SCHEMA_VERSION,
        )
        self.assertEqual(
            summary["source_mvp_completion_audit_schema_version"],
            MVP_COMPLETION_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(
            summary["source_current_evidence_schema_version"],
            CURRENT_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertEqual(
            summary["current_evidence_schema_version"],
            CURRENT_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertTrue(summary["technical_model_core_mvp_completed"])
        self.assertTrue(summary["changed_ratio_repair_objective_completed"])
        self.assertTrue(summary["outside_soloing_repair_objective_completed"])
        self.assertTrue(summary["outside_soloing_repair_source_context_preserved"])
        self.assertTrue(summary["outside_soloing_repair_schema_context_preserved"])
        self.assertEqual(
            summary["outside_soloing_repair_objective_schema_version"],
            OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
        )
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertEqual(summary["max_repaired_interval"], 12)
        self.assertEqual(summary["max_interval_threshold"], 12)
        self.assertAlmostEqual(summary["max_repaired_pitch_changed_ratio"], 0.4348, places=4)
        self.assertAlmostEqual(summary["target_max_pitch_changed_ratio"], 0.5, places=4)
        self.assertTrue(summary["outside_soloing_repair_objective_path_ready"])
        self.assertTrue(summary["outside_soloing_repair_target_supported"])
        self.assertEqual(summary["outside_soloing_repair_rendered_audio_file_count"], 6)
        self.assertEqual(summary["outside_soloing_repair_changed_note_total"], 2)
        self.assertEqual(
            summary["outside_soloing_repair_source_objective_pitch_role_risk_count"],
            5,
        )
        self.assertEqual(
            summary["outside_soloing_repair_source_pitch_role_risk_count_before"],
            5,
        )
        self.assertEqual(
            summary["outside_soloing_repair_source_pitch_role_risk_count_after"],
            2,
        )
        self.assertEqual(summary["outside_soloing_repair_source_pitch_role_risk_delta"], 3)
        self.assertFalse(summary["outside_soloing_repair_source_targeted"])
        self.assertTrue(summary["outside_soloing_repair_source_residual_risk_preserved"])
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_count_after"], 0)
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_delta"], 2)
        for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
            self.assertEqual(summary[key], SOURCE_CONTEXT[key])
        self.assertTrue(summary["outside_soloing_repair_objective_path_supported"])
        self.assertTrue(summary["outside_soloing_repair_weak_landing_target_supported"])
        self.assertTrue(summary["outside_soloing_repair_final_landing_target_supported"])
        self.assertTrue(summary["outside_soloing_repair_non_chord_run_target_supported"])
        self.assertTrue(summary["listening_review_quality_gap_open"])
        self.assertTrue(summary["technical_mvp_delivery_package_ready"])
        self.assertFalse(summary["human_review_required_now"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertEqual(
            summary["next_recommended_issue"],
            "Stage B MIDI-to-solo MVP delivery package source-context refresh",
        )
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["source_schema_version"], QUALITY_GAP_DECISION_SCHEMA_VERSION)
        self.assertEqual(
            report["source_mvp_completion_audit_schema_version"],
            MVP_COMPLETION_AUDIT_SCHEMA_VERSION,
        )
        self.assertEqual(
            report["source_current_evidence_schema_version"],
            CURRENT_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertEqual(report["issue_number"], 1158)

    def test_rejects_quality_gap_decision_schema_mismatch(self) -> None:
        source = quality_gap_decision()
        source["schema_version"] = "stale_schema"
        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=source,
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=1158,
            )

    def test_rejects_quality_gap_decision_source_schema_mismatch(self) -> None:
        source = quality_gap_decision()
        source["source_schema_version"] = "stale_schema"
        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=source,
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=1158,
            )

    def test_rejects_wrong_selected_target(self) -> None:
        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=quality_gap_decision(selected_target="other_target"),
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=736,
            )

    def test_rejects_wrong_next_boundary(self) -> None:
        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=quality_gap_decision(next_boundary="other_boundary"),
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=736,
            )

    def test_rejects_missing_changed_ratio_repair_support(self) -> None:
        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=quality_gap_decision(changed_ratio_supported=False),
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=736,
            )

    def test_rejects_missing_outside_soloing_repair_support(self) -> None:
        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=quality_gap_decision(
                    outside_soloing_repair_supported=False
                ),
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=820,
            )

    def test_rejects_missing_outside_soloing_source_context(self) -> None:
        source = quality_gap_decision()
        source["quality_gap"].pop("outside_soloing_repair_source_residual_risk_preserved")

        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=source,
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=906,
            )

    def test_rejects_missing_outside_soloing_source_context_field(self) -> None:
        source = quality_gap_decision()
        source["quality_gap"].pop(
            "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
        )

        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=source,
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=990,
            )

    def test_rejects_false_outside_soloing_source_context_preserved_flag(self) -> None:
        source = quality_gap_decision()
        source["quality_gap"][
            "repair_sweep_source_outside_soloing_source_context_preserved"
        ] = False

        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=source,
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=1074,
            )

    def test_rejects_false_outside_soloing_schema_context_preserved_flag(self) -> None:
        source = quality_gap_decision()
        source["quality_gap"]["outside_soloing_repair_schema_context_preserved"] = False

        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=source,
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=1158,
            )

    def test_rejects_listening_review_quality_gap_schema_mismatch(self) -> None:
        report = build_listening_review_quality_gap_report(
            quality_gap_decision=quality_gap_decision(),
            output_dir=Path("outputs/listening_review_quality_gap"),
            issue_number=1158,
        )
        report["schema_version"] = "stale_schema"
        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            validate_listening_review_quality_gap_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                require_delivery_package_ready=True,
                require_no_quality_claim=True,
            )

    def test_rejects_targeted_outside_soloing_source_repair(self) -> None:
        source = quality_gap_decision()
        source["quality_gap"]["outside_soloing_repair_source_targeted"] = True

        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=source,
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=906,
            )

    def test_rejects_outside_soloing_source_risk_delta_mismatch(self) -> None:
        source = quality_gap_decision()
        source["quality_gap"][
            "outside_soloing_repair_source_pitch_role_risk_delta"
        ] = 2

        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=source,
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=906,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloListeningReviewQualityGapError):
            build_listening_review_quality_gap_report(
                quality_gap_decision=quality_gap_decision(quality_claim=True),
                output_dir=Path("outputs/listening_review_quality_gap"),
                issue_number=736,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_listening_review_quality_gap")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_mvp_delivery_package")
        self.assertEqual(SELECTED_TARGET, "mvp_delivery_package")
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_listening_review_quality_gap_v4",
        )


if __name__ == "__main__":
    unittest.main()
