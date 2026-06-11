from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_final_status import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
)
from scripts.decide_stage_b_midi_to_solo_targeted_quality_repair_objective_next import (
    BOUNDARY,
    FOLLOWUP_DECISION_NEXT_BOUNDARY,
    SCHEMA_VERSION,
    StageBMidiToSoloTargetedQualityRepairObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.guard_stage_b_midi_to_solo_targeted_quality_repair_listening_review_input import (
    BOUNDARY as SOURCE_BOUNDARY,
    EXPECTED_SOURCE_SCHEMA_VERSIONS as INPUT_GUARD_SOURCE_SCHEMA_VERSIONS,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SCHEMA_VERSION as SOURCE_INPUT_GUARD_SCHEMA_VERSION,
)


SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
}


def input_guard_report(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": SOURCE_INPUT_GUARD_SCHEMA_VERSION,
        "boundary": SOURCE_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_targeted_quality_repair_listening_review_package",
        "source_schema_versions": dict(INPUT_GUARD_SOURCE_SCHEMA_VERSIONS),
        "guard_result": {
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "review_item_count": 6,
            "required_input_field_count": 4,
            "source_summary": {
                "source_targeted_quality_repair_listening_review_input_guard_schema_version": (
                    SOURCE_INPUT_GUARD_SCHEMA_VERSION
                ),
                "source_targeted_quality_repair_listening_review_package_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS[
                        "targeted_quality_repair_listening_review_package"
                    ]
                ),
                "source_targeted_quality_repair_audio_package_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS[
                        "targeted_quality_repair_audio_package"
                    ]
                ),
                "source_targeted_quality_repair_sweep_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS[
                        "targeted_quality_repair_sweep"
                    ]
                ),
                "source_candidate_failure_labeling_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS["candidate_failure_labeling"]
                ),
                "source_quality_rubric_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS["quality_rubric_baseline"]
                ),
                "source_post_mvp_plan_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS[
                        "post_mvp_quality_iteration_plan"
                    ]
                ),
                "source_final_status_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS["final_status_audit"]
                ),
                "source_delivery_package_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS["delivery_package"]
                ),
                "source_listening_gap_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS[
                        "listening_review_quality_gap"
                    ]
                ),
                "source_quality_gap_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS["quality_gap_decision"]
                ),
                "source_current_evidence_schema_version": (
                    INPUT_GUARD_SOURCE_SCHEMA_VERSIONS["current_evidence"]
                ),
                "technical_wav_validation": True,
                "rendered_audio_file_count": 6,
                "sample_rate": 44100,
                "duration_min_seconds": 18.422,
                "duration_max_seconds": 18.984,
                "failure_label_delta": 4,
                "source_outside_soloing_repair_evidence_ready": True,
                "source_outside_soloing_repair_source_context_preserved": True,
                "source_outside_soloing_repair_schema_context_preserved": True,
                "source_outside_soloing_repair_objective_schema_version": (
                    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
                ),
                "source_outside_soloing_repair_wav_count": 6,
                "source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
                "source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
                "source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
                "source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
                "source_outside_soloing_repair_source_targeted": False,
                "source_outside_soloing_repair_source_residual_risk_preserved": True,
                "source_outside_soloing_repair_pitch_role_risk_count_after": 0,
                "source_outside_soloing_repair_pitch_role_risk_delta": 2,
                "source_outside_soloing_not_evaluable_count": 6,
                "repaired_outside_soloing_not_evaluable_count": 6,
                "audio_review_required": True,
                **SOURCE_CONTEXT,
            },
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "listening_review_input_guard_completed": True,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": SOURCE_BOUNDARY,
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloTargetedQualityRepairObjectiveNextTest(unittest.TestCase):
    def test_routes_pending_review_to_followup_decision_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_objective_next_report(
                input_guard_report=input_guard_report(),
                output_dir=root / "objective_next",
                issue_number=1180,
            )
            summary = validate_objective_next_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=FOLLOWUP_DECISION_NEXT_BOUNDARY,
                require_objective_decision=True,
                require_followup_required=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(report["issue_number"], 1180)
            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertEqual(
                report["source_schema_versions"][
                    "targeted_quality_repair_listening_review_input_guard"
                ],
                SOURCE_INPUT_GUARD_SCHEMA_VERSION,
            )
            for key, expected in INPUT_GUARD_SOURCE_SCHEMA_VERSIONS.items():
                self.assertEqual(report["source_schema_versions"][key], expected)
            self.assertEqual(
                summary[
                    "source_targeted_quality_repair_listening_review_input_guard_schema_version"
                ],
                SOURCE_INPUT_GUARD_SCHEMA_VERSION,
            )
            for key, expected in INPUT_GUARD_SOURCE_SCHEMA_VERSIONS.items():
                summary_key = "source_" + {
                    "targeted_quality_repair_listening_review_package": (
                        "targeted_quality_repair_listening_review_package"
                    ),
                    "targeted_quality_repair_audio_package": (
                        "targeted_quality_repair_audio_package"
                    ),
                    "targeted_quality_repair_sweep": "targeted_quality_repair_sweep",
                    "candidate_failure_labeling": "candidate_failure_labeling",
                    "quality_rubric_baseline": "quality_rubric",
                    "post_mvp_quality_iteration_plan": "post_mvp_plan",
                    "final_status_audit": "final_status",
                    "delivery_package": "delivery_package",
                    "listening_review_quality_gap": "listening_gap",
                    "quality_gap_decision": "quality_gap",
                    "current_evidence": "current_evidence",
                }[key] + "_schema_version"
                self.assertEqual(summary[summary_key], expected)
            self.assertTrue(summary["objective_next_decision_completed"])
            self.assertFalse(summary["validated_review_input_present"])
            self.assertFalse(summary["preference_fill_allowed"])
            self.assertTrue(summary["technical_wav_validation"])
            self.assertEqual(summary["rendered_audio_file_count"], 6)
            self.assertEqual(summary["failure_label_delta"], 4)
            self.assertTrue(summary["source_outside_soloing_repair_evidence_ready"])
            self.assertTrue(summary["source_outside_soloing_repair_source_context_preserved"])
            self.assertTrue(summary["source_outside_soloing_repair_schema_context_preserved"])
            self.assertEqual(
                summary["source_outside_soloing_repair_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
                self.assertEqual(summary[key], SOURCE_CONTEXT[key])
            self.assertEqual(summary["source_outside_soloing_repair_wav_count"], 6)
            self.assertEqual(
                summary["source_outside_soloing_repair_source_objective_pitch_role_risk_count"], 5
            )
            self.assertEqual(
                summary["source_outside_soloing_repair_source_pitch_role_risk_count_before"], 5
            )
            self.assertEqual(
                summary["source_outside_soloing_repair_source_pitch_role_risk_count_after"], 2
            )
            self.assertEqual(summary["source_outside_soloing_repair_source_pitch_role_risk_delta"], 3)
            self.assertFalse(summary["source_outside_soloing_repair_source_targeted"])
            self.assertTrue(summary["source_outside_soloing_repair_source_residual_risk_preserved"])
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_count_after"], 0)
            self.assertEqual(summary["source_outside_soloing_repair_pitch_role_risk_delta"], 2)
            self.assertEqual(summary["source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["repaired_outside_soloing_not_evaluable_count"], 6)
            self.assertTrue(summary["targeted_quality_followup_required"])
            self.assertFalse(summary["current_quality_claim_ready"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
            self.assertEqual(
                summary["next_recommended_issue"],
                "Stage B MIDI-to-solo targeted quality repair follow-up decision source-context refresh",
            )

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=input_guard_report(quality_claim=True),
                    output_dir=root / "objective_next",
                    issue_number=1180,
                )

    def test_rejects_missing_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"][
                "source_outside_soloing_repair_evidence_ready"
            ] = False
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1180,
                )

    def test_rejects_source_risk_delta_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"][
                "source_outside_soloing_repair_source_pitch_role_risk_delta"
            ] = 1
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1180,
                )

    def test_rejects_missing_source_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"].pop(
                "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
            )
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1180,
                )

    def test_rejects_false_source_context_preserved_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"][
                "followup_repair_sweep_source_outside_soloing_source_context_preserved"
            ] = False
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1180,
                )

    def test_rejects_source_schema_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["source_schema_versions"][
                "targeted_quality_repair_listening_review_package"
            ] = "wrong_schema"
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1180,
                )

    def test_rejects_missing_schema_context_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["guard_result"]["source_summary"][
                "source_outside_soloing_repair_schema_context_preserved"
            ] = False
            with self.assertRaises(StageBMidiToSoloTargetedQualityRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1180,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_objective_only_next_decision")
        self.assertEqual(FOLLOWUP_DECISION_NEXT_BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_followup_decision")
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_targeted_quality_repair_objective_next_v5",
        )


if __name__ == "__main__":
    unittest.main()
