from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_targeted_quality_repair_objective_next import (
    BOUNDARY,
    FOLLOWUP_DECISION_NEXT_BOUNDARY,
    StageBMidiToSoloTargetedQualityRepairObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.guard_stage_b_midi_to_solo_targeted_quality_repair_listening_review_input import (
    BOUNDARY as SOURCE_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


def input_guard_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": SOURCE_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_targeted_quality_repair_listening_review_package",
        "guard_result": {
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "review_item_count": 6,
            "required_input_field_count": 4,
            "source_summary": {
                "technical_wav_validation": True,
                "rendered_audio_file_count": 6,
                "sample_rate": 44100,
                "duration_min_seconds": 18.422,
                "duration_max_seconds": 18.984,
                "failure_label_delta": 4,
                "source_outside_soloing_repair_evidence_ready": True,
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
                issue_number=758,
            )
            summary = validate_objective_next_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=FOLLOWUP_DECISION_NEXT_BOUNDARY,
                require_objective_decision=True,
                require_followup_required=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["objective_next_decision_completed"])
            self.assertFalse(summary["validated_review_input_present"])
            self.assertFalse(summary["preference_fill_allowed"])
            self.assertTrue(summary["technical_wav_validation"])
            self.assertEqual(summary["rendered_audio_file_count"], 6)
            self.assertEqual(summary["failure_label_delta"], 4)
            self.assertTrue(summary["source_outside_soloing_repair_evidence_ready"])
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
                    issue_number=758,
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
                    issue_number=842,
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
                    issue_number=928,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_objective_only_next_decision")
        self.assertEqual(FOLLOWUP_DECISION_NEXT_BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_followup_decision")


if __name__ == "__main__":
    unittest.main()
