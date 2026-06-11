from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_final_status import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    CURRENT_EVIDENCE_SCHEMA_VERSION,
    DELIVERY_SCHEMA_VERSION,
    LISTENING_GAP_SCHEMA_VERSION,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    QUALITY_GAP_DECISION_SCHEMA_VERSION,
    SCHEMA_VERSION as FINAL_STATUS_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_quality_rubric_baseline import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    SCHEMA_VERSION,
    StageBMidiToSoloQualityRubricBaselineError,
    build_quality_rubric_baseline_report,
    validate_quality_rubric_baseline_report,
)
from scripts.plan_stage_b_midi_to_solo_post_mvp_quality_iteration import (
    BOUNDARY as POST_MVP_BOUNDARY,
    NEXT_BOUNDARY as POST_MVP_NEXT_BOUNDARY,
    SELECTED_TARGET as POST_MVP_SELECTED_TARGET,
    SCHEMA_VERSION as POST_MVP_SCHEMA_VERSION,
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


def post_mvp_quality_plan(
    *,
    selected_target: str = POST_MVP_SELECTED_TARGET,
    quality_claim: bool = False,
    ordered_work_count: int = 4,
    taxonomy_count: int = 7,
    outside_ready: bool = True,
    outside_wav_count: int = 6,
    outside_risk_after: int = 0,
) -> dict:
    ordered_targets = [
        "quality_rubric_baseline",
        "candidate_failure_labeling",
        "targeted_quality_repair_sweep",
        "audio_review_package",
    ][:ordered_work_count]
    return {
        "schema_version": POST_MVP_SCHEMA_VERSION,
        "boundary": POST_MVP_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_final_status_audit",
        "source_schema_versions": {
            "final_status_audit": FINAL_STATUS_SCHEMA_VERSION,
            "delivery_package": DELIVERY_SCHEMA_VERSION,
            "listening_review_quality_gap": LISTENING_GAP_SCHEMA_VERSION,
            "quality_gap_decision": QUALITY_GAP_DECISION_SCHEMA_VERSION,
            "current_evidence": CURRENT_EVIDENCE_SCHEMA_VERSION,
        },
        "post_mvp_status": {
            "source_final_status_schema_version": FINAL_STATUS_SCHEMA_VERSION,
            "source_delivery_package_schema_version": DELIVERY_SCHEMA_VERSION,
            "source_listening_gap_schema_version": LISTENING_GAP_SCHEMA_VERSION,
            "source_quality_gap_schema_version": QUALITY_GAP_DECISION_SCHEMA_VERSION,
            "source_current_evidence_schema_version": CURRENT_EVIDENCE_SCHEMA_VERSION,
            "technical_mvp_complete": True,
            "local_review_ready": True,
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
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            **SOURCE_CONTEXT,
        },
        "selected_next_target": {
            "selected_target": selected_target,
            "selected_next_boundary": POST_MVP_NEXT_BOUNDARY,
        },
        "ordered_work": [{"target": target} for target in ordered_targets],
        "quality_failure_taxonomy_seed": [f"failure_{index}" for index in range(taxonomy_count)],
        "readiness": {
            "post_mvp_quality_iteration_plan_completed": True,
            "quality_rubric_required": True,
            "candidate_failure_labeling_required": True,
            "targeted_quality_repair_sweep_required": True,
            "audio_review_package_required": True,
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
            "next_boundary": POST_MVP_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "next_recommended_issue": "Stage B MIDI-to-solo quality rubric baseline source-context refresh",
    }


class StageBMidiToSoloQualityRubricBaselineTest(unittest.TestCase):
    def test_builds_quality_rubric_without_quality_claim(self) -> None:
        report = build_quality_rubric_baseline_report(
            post_mvp_quality_plan=post_mvp_quality_plan(),
            output_dir=Path("outputs/quality_rubric"),
            issue_number=1168,
        )
        summary = validate_quality_rubric_baseline_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            expected_target=SELECTED_TARGET,
            min_rubric_item_count=8,
            require_candidate_labeling_ready=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["issue_number"], 1168)
        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["source_post_mvp_plan_schema_version"], POST_MVP_SCHEMA_VERSION)
        self.assertEqual(summary["source_final_status_schema_version"], FINAL_STATUS_SCHEMA_VERSION)
        self.assertEqual(summary["source_delivery_package_schema_version"], DELIVERY_SCHEMA_VERSION)
        self.assertEqual(summary["source_listening_gap_schema_version"], LISTENING_GAP_SCHEMA_VERSION)
        self.assertEqual(summary["source_quality_gap_schema_version"], QUALITY_GAP_DECISION_SCHEMA_VERSION)
        self.assertEqual(summary["source_current_evidence_schema_version"], CURRENT_EVIDENCE_SCHEMA_VERSION)
        self.assertTrue(summary["quality_rubric_baseline_completed"])
        self.assertTrue(summary["candidate_failure_labeling_ready"])
        self.assertEqual(summary["selected_target"], SELECTED_TARGET)
        self.assertEqual(summary["rubric_item_count"], 8)
        self.assertGreaterEqual(summary["required_metric_group_count"], 20)
        self.assertTrue(summary["outside_soloing_repair_evidence_ready"])
        self.assertTrue(summary["outside_soloing_repair_source_context_preserved"])
        self.assertTrue(summary["outside_soloing_repair_schema_context_preserved"])
        self.assertEqual(
            summary["outside_soloing_repair_objective_schema_version"],
            OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
        )
        self.assertEqual(summary["outside_soloing_repair_wav_count"], 6)
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
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(
            summary["next_recommended_issue"],
            "Stage B MIDI-to-solo candidate failure labeling source-context refresh",
        )

    def test_rejects_wrong_source_boundary(self) -> None:
        source = post_mvp_quality_plan()
        source["boundary"] = "wrong_boundary"
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=source,
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )

    def test_rejects_wrong_selected_target(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(selected_target="repair_sweep"),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(quality_claim=True),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )

    def test_rejects_incomplete_plan_inputs(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(ordered_work_count=3),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(taxonomy_count=6),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )

    def test_rejects_incomplete_outside_soloing_repair_context(self) -> None:
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(outside_ready=False),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(outside_wav_count=5),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=post_mvp_quality_plan(outside_risk_after=1),
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )

    def test_rejects_post_mvp_source_schema_mismatch(self) -> None:
        source = post_mvp_quality_plan()
        source["source_schema_versions"]["final_status_audit"] = "wrong_schema"
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=source,
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )

    def test_rejects_missing_outside_soloing_schema_context(self) -> None:
        source = post_mvp_quality_plan()
        source["post_mvp_status"]["outside_soloing_repair_schema_context_preserved"] = False
        source["readiness"]["outside_soloing_repair_schema_context_preserved"] = False
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=source,
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )

    def test_rejects_missing_outside_soloing_source_context_field(self) -> None:
        source = post_mvp_quality_plan()
        source["post_mvp_status"].pop(
            "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
        )
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=source,
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )

    def test_rejects_false_source_context_preserved_field(self) -> None:
        source = post_mvp_quality_plan()
        source["post_mvp_status"][
            "followup_repair_sweep_source_outside_soloing_source_context_preserved"
        ] = False
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            build_quality_rubric_baseline_report(
                post_mvp_quality_plan=source,
                output_dir=Path("outputs/quality_rubric"),
                issue_number=1168,
            )

    def test_rejects_missing_required_rubric_item(self) -> None:
        report = build_quality_rubric_baseline_report(
            post_mvp_quality_plan=post_mvp_quality_plan(),
            output_dir=Path("outputs/quality_rubric"),
            issue_number=1168,
        )
        report["rubric_items"] = [
            item for item in report["rubric_items"] if item["id"] != "weak_chord_tone_landing"
        ]
        with self.assertRaises(StageBMidiToSoloQualityRubricBaselineError):
            validate_quality_rubric_baseline_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                min_rubric_item_count=7,
                require_candidate_labeling_ready=True,
                require_no_quality_claim=True,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_quality_rubric_baseline")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_candidate_failure_labeling")
        self.assertEqual(SELECTED_TARGET, "candidate_failure_labeling")
        self.assertEqual(SCHEMA_VERSION, "stage_b_midi_to_solo_quality_rubric_baseline_v4")


if __name__ == "__main__":
    unittest.main()
