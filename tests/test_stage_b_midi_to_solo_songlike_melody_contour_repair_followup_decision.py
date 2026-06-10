from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_repair_followup import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    TIE_TARGET_LABELS,
    StageBMidiToSoloSonglikeMelodyContourRepairFollowupDecisionError,
    build_followup_decision_report,
    validate_followup_decision_report,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_repair_objective_next import (
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_repair_sweep import (
    BOUNDARY as SWEEP_BOUNDARY,
)


def objective_next_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "objective_summary": {
            "review_item_count": 6,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 6,
            "failure_label_delta": 4,
            "songlike_failure_delta": 5,
            "source_outside_soloing_repair_evidence_ready": True,
            "objective_source_outside_soloing_repair_wav_count": 6,
            "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "objective_source_outside_soloing_repair_source_targeted": False,
            "objective_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "objective_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "objective_source_outside_soloing_repair_pitch_role_risk_delta": 2,
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
            "current_quality_claim_ready": False,
        },
        "readiness": {
            "objective_next_decision_completed": True,
            "songlike_contour_followup_required": True,
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
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def repair_sweep_report(*, technical_regression_count: int = 0) -> dict:
    return {
        "boundary": SWEEP_BOUNDARY,
        "candidate_repairs": [
            {
                "repaired_labeling": {
                    "failure_labels": list(TIE_TARGET_LABELS),
                    "not_evaluable_labels": [
                        "outside_soloing_without_context",
                        "weak_chord_tone_landing",
                    ],
                }
            }
            for _ in range(6)
        ],
        "aggregate": {
            "candidate_count": 6,
            "source_total_failure_label_count": 8,
            "repaired_total_failure_label_count": 4,
            "failure_label_delta": 4,
            "improved_candidate_count": 4,
            "technical_regression_count": technical_regression_count,
            "repaired_failure_counts": {
                "phrase_shape_missing_tension_release": 2,
                "rhythmic_monotony": 2,
            },
            "source_outside_soloing_repair_evidence_ready": True,
            "objective_source_outside_soloing_repair_wav_count": 6,
            "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": 5,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": 2,
            "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": 3,
            "objective_source_outside_soloing_repair_source_targeted": False,
            "objective_source_outside_soloing_repair_source_residual_risk_preserved": True,
            "objective_source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "objective_source_outside_soloing_repair_pitch_role_risk_delta": 2,
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
        },
        "readiness": {
            "songlike_melody_contour_repair_sweep_completed": True,
            "songlike_melody_contour_repair_target_supported": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloSonglikeMelodyContourRepairFollowupDecisionTest(unittest.TestCase):
    def test_selects_phrase_rhythm_tie_repair_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_followup_decision_report(
                objective_next_report=objective_next_report(),
                repair_sweep_report=repair_sweep_report(),
                output_dir=Path(tmp) / "followup",
                issue_number=856,
            )
            summary = validate_followup_decision_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                require_followup_decision=True,
                require_phrase_rhythm_tie_target=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["followup_decision_completed"])
            self.assertTrue(summary["phrase_rhythm_tie_target_selected"])
            self.assertEqual(
                summary["primary_remaining_failure_labels"],
                list(TIE_TARGET_LABELS),
            )
            self.assertEqual(summary["primary_remaining_failure_count"], 2)
            self.assertEqual(summary["failure_label_delta"], 4)
            self.assertEqual(summary["technical_regression_count"], 0)
            self.assertTrue(summary["objective_source_outside_soloing_repair_evidence_ready"])
            self.assertEqual(summary["objective_source_outside_soloing_repair_wav_count"], 6)
            self.assertEqual(
                summary[
                    "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertEqual(
                summary[
                    "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
                ],
                2,
            )
            self.assertEqual(
                summary["objective_source_outside_soloing_repair_source_pitch_role_risk_delta"],
                3,
            )
            self.assertFalse(summary["objective_source_outside_soloing_repair_source_targeted"])
            self.assertTrue(
                summary[
                    "objective_source_outside_soloing_repair_source_residual_risk_preserved"
                ]
            )
            self.assertEqual(summary["objective_source_outside_soloing_repair_pitch_role_risk_count_after"], 0)
            self.assertEqual(
                summary["objective_source_outside_soloing_repair_pitch_role_risk_delta"],
                2,
            )
            self.assertEqual(summary["objective_source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["objective_repaired_outside_soloing_not_evaluable_count"], 6)
            self.assertTrue(summary["repair_sweep_source_outside_soloing_repair_evidence_ready"])
            self.assertEqual(
                summary[
                    "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertEqual(
                summary[
                    "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after"
                ],
                2,
            )
            self.assertEqual(
                summary[
                    "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta"
                ],
                3,
            )
            self.assertFalse(
                summary["repair_sweep_source_outside_soloing_repair_source_targeted"]
            )
            self.assertTrue(
                summary[
                    "repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved"
                ]
            )
            self.assertEqual(summary["repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after"], 0)
            self.assertEqual(
                summary["repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta"],
                2,
            )
            self.assertEqual(summary["repair_sweep_source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["repair_sweep_repaired_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_objective_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(quality_claim=True),
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=856,
                )

    def test_rejects_technical_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=repair_sweep_report(technical_regression_count=1),
                    output_dir=Path(tmp) / "followup",
                    issue_number=856,
                )

    def test_rejects_missing_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = repair_sweep_report()
            source["aggregate"]["source_outside_soloing_not_evaluable_count"] = 0
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=source,
                    output_dir=Path(tmp) / "followup",
                    issue_number=856,
                )

    def test_rejects_source_context_delta_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = repair_sweep_report()
            source["aggregate"]["source_outside_soloing_repair_source_pitch_role_risk_delta"] = 1
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=source,
                    output_dir=Path(tmp) / "followup",
                    issue_number=942,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_repair_followup_decision")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep")
        self.assertEqual(SELECTED_TARGET, "songlike_melody_contour_phrase_rhythm_repair_sweep")


if __name__ == "__main__":
    unittest.main()
