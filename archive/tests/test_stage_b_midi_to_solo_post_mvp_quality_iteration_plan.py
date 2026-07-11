from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_final_status import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BOUNDARY as FINAL_STATUS_BOUNDARY,
    CURRENT_EVIDENCE_SCHEMA_VERSION,
    DELIVERY_SCHEMA_VERSION,
    LISTENING_GAP_SCHEMA_VERSION,
    NEXT_BOUNDARY as FINAL_STATUS_NEXT_BOUNDARY,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    QUALITY_GAP_DECISION_SCHEMA_VERSION,
    SCHEMA_VERSION as FINAL_STATUS_SCHEMA_VERSION,
)
from scripts.plan_stage_b_midi_to_solo_post_mvp_quality_iteration import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    SCHEMA_VERSION,
    StageBMidiToSoloPostMvpQualityIterationPlanError,
    build_post_mvp_quality_iteration_plan_report,
    validate_post_mvp_quality_iteration_plan_report,
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


def final_status_audit(
    *,
    quality_claim: bool = False,
    listening_gap_open: bool = True,
    cli_count: int = 3,
    wav_count: int = 3,
    outside_ready: bool = True,
    outside_wav_count: int = 6,
    outside_risk_after: int = 0,
) -> dict:
    return {
        "schema_version": FINAL_STATUS_SCHEMA_VERSION,
        "boundary": FINAL_STATUS_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_mvp_delivery_package",
        "source_schema_versions": {
            "delivery_package": DELIVERY_SCHEMA_VERSION,
            "listening_review_quality_gap": LISTENING_GAP_SCHEMA_VERSION,
            "quality_gap_decision": QUALITY_GAP_DECISION_SCHEMA_VERSION,
            "current_evidence": CURRENT_EVIDENCE_SCHEMA_VERSION,
        },
        "final_status": {
            "technical_mvp_complete": True,
            "technical_mvp_ready_for_local_review": True,
            "readme_final_evidence_reflected": True,
            "delivery_package_schema_version": DELIVERY_SCHEMA_VERSION,
            "delivery_source_listening_gap_schema_version": LISTENING_GAP_SCHEMA_VERSION,
            "delivery_source_quality_gap_schema_version": QUALITY_GAP_DECISION_SCHEMA_VERSION,
            "delivery_source_current_evidence_schema_version": CURRENT_EVIDENCE_SCHEMA_VERSION,
            "input_to_ranked_midi_ready": True,
            "input_to_rendered_wav_evidence_ready": True,
            "changed_ratio_repair_audio_evidence_ready": True,
            "cli_candidate_count": cli_count,
            "changed_ratio_repair_wav_count": wav_count,
            "outside_soloing_repair_evidence_ready": outside_ready,
            "outside_soloing_repair_source_context_preserved": outside_ready,
            "outside_soloing_repair_schema_context_preserved": outside_ready,
            "outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            "outside_soloing_repair_wav_count": outside_wav_count,
            "outside_soloing_repair_changed_note_total": 2,
            "outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "outside_soloing_repair_source_targeted": False,
            "outside_soloing_repair_source_residual_risk_preserved": True,
            "outside_soloing_repair_pitch_role_risk_count_after": outside_risk_after,
            "outside_soloing_repair_pitch_role_risk_delta": 2,
            "listening_review_quality_gap_open": listening_gap_open,
            "raw_artifact_upload_required": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            **SOURCE_CONTEXT,
        },
        "readiness": {
            "final_status_audit_completed": True,
            "technical_mvp_complete": True,
            "technical_mvp_ready_for_local_review": True,
            "readme_final_evidence_reflected": True,
            "outside_soloing_repair_source_context_preserved": outside_ready,
            "outside_soloing_repair_schema_context_preserved": outside_ready,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": FINAL_STATUS_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "next_recommended_issue": "Stage B MIDI-to-solo post-MVP musical quality iteration plan",
    }


class StageBMidiToSoloPostMvpQualityIterationPlanTest(unittest.TestCase):
    def test_selects_quality_rubric_baseline_without_quality_claim(self) -> None:
        report = build_post_mvp_quality_iteration_plan_report(
            final_status_audit=final_status_audit(),
            output_dir=Path("outputs/post_mvp_quality"),
            issue_number=1082,
        )
        summary = validate_post_mvp_quality_iteration_plan_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            expected_target=SELECTED_TARGET,
            require_quality_rubric=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["issue_number"], 1082)
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["source_final_status_schema_version"], FINAL_STATUS_SCHEMA_VERSION)
        self.assertEqual(summary["source_delivery_package_schema_version"], DELIVERY_SCHEMA_VERSION)
        self.assertEqual(summary["source_listening_gap_schema_version"], LISTENING_GAP_SCHEMA_VERSION)
        self.assertEqual(
            summary["source_quality_gap_schema_version"],
            QUALITY_GAP_DECISION_SCHEMA_VERSION,
        )
        self.assertEqual(
            summary["source_current_evidence_schema_version"],
            CURRENT_EVIDENCE_SCHEMA_VERSION,
        )
        self.assertTrue(summary["post_mvp_quality_iteration_plan_completed"])
        self.assertTrue(summary["technical_mvp_complete"])
        self.assertTrue(summary["local_review_ready"])
        self.assertEqual(summary["selected_target"], SELECTED_TARGET)
        self.assertTrue(summary["quality_rubric_required"])
        self.assertTrue(summary["outside_soloing_repair_evidence_ready"])
        self.assertTrue(summary["outside_soloing_repair_source_context_preserved"])
        self.assertTrue(summary["outside_soloing_repair_schema_context_preserved"])
        self.assertEqual(
            summary["outside_soloing_repair_objective_schema_version"],
            OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
        )
        self.assertEqual(summary["outside_soloing_repair_wav_count"], 6)
        self.assertEqual(summary["outside_soloing_repair_changed_note_total"], 2)
        self.assertEqual(summary["outside_soloing_repair_source_objective_pitch_role_risk_count"], 5)
        self.assertEqual(summary["outside_soloing_repair_source_pitch_role_risk_count_before"], 5)
        self.assertEqual(summary["outside_soloing_repair_source_pitch_role_risk_count_after"], 2)
        self.assertEqual(summary["outside_soloing_repair_source_pitch_role_risk_delta"], 3)
        self.assertFalse(summary["outside_soloing_repair_source_targeted"])
        self.assertTrue(summary["outside_soloing_repair_source_residual_risk_preserved"])
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_count_after"], 0)
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_delta"], 2)
        for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
            self.assertEqual(summary[key], SOURCE_CONTEXT[key])
        self.assertTrue(summary["candidate_failure_labeling_required"])
        self.assertTrue(summary["targeted_quality_repair_sweep_required"])
        self.assertTrue(summary["audio_review_package_required"])
        self.assertGreaterEqual(summary["ordered_work_count"], 4)
        self.assertGreaterEqual(summary["quality_failure_taxonomy_seed_count"], 7)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(
            summary["next_recommended_issue"],
            "Stage B MIDI-to-solo quality rubric baseline source-context refresh",
        )
        self.assertEqual(
            report["source_schema_versions"]["final_status_audit"],
            FINAL_STATUS_SCHEMA_VERSION,
        )

    def test_rejects_post_mvp_schema_mismatch(self) -> None:
        report = build_post_mvp_quality_iteration_plan_report(
            final_status_audit=final_status_audit(),
            output_dir=Path("outputs/post_mvp_quality"),
            issue_number=1166,
        )
        report["schema_version"] = "stale_schema"
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            validate_post_mvp_quality_iteration_plan_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                require_quality_rubric=True,
                require_no_quality_claim=True,
            )

    def test_rejects_final_status_schema_mismatch(self) -> None:
        source = final_status_audit()
        source["schema_version"] = "stale_schema"
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=source,
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1166,
            )

    def test_rejects_wrong_source_boundary(self) -> None:
        source = final_status_audit()
        source["boundary"] = "wrong_boundary"
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=source,
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(quality_claim=True),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )

    def test_rejects_closed_listening_gap(self) -> None:
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(listening_gap_open=False),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )

    def test_rejects_low_candidate_or_wav_count(self) -> None:
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(cli_count=2),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(wav_count=2),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )

    def test_rejects_incomplete_outside_soloing_repair_evidence(self) -> None:
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(outside_ready=False),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(outside_wav_count=5),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=final_status_audit(outside_risk_after=1),
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )

    def test_rejects_missing_outside_soloing_source_context_field(self) -> None:
        source = final_status_audit()
        source["final_status"].pop(
            "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
        )
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=source,
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )

    def test_rejects_false_source_context_preserved_field(self) -> None:
        source = final_status_audit()
        source["final_status"][
            "repair_sweep_source_outside_soloing_source_context_preserved"
        ] = False
        with self.assertRaises(StageBMidiToSoloPostMvpQualityIterationPlanError):
            build_post_mvp_quality_iteration_plan_report(
                final_status_audit=source,
                output_dir=Path("outputs/post_mvp_quality"),
                issue_number=1082,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_post_mvp_quality_iteration_plan")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_quality_rubric_baseline")
        self.assertEqual(SELECTED_TARGET, "quality_rubric_baseline")
        self.assertEqual(SCHEMA_VERSION, "stage_b_midi_to_solo_post_mvp_quality_iteration_plan_v4")


if __name__ == "__main__":
    unittest.main()
