from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (
    BOUNDARY as BRIDGE_BOUNDARY,
    NEXT_BOUNDARY as BRIDGE_NEXT_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective import (
    BOUNDARY,
    NEXT_BOUNDARY,
    PRIMARY_RISK_LABEL,
    SELECTED_TARGET,
    StageBMidiToSoloPitchRoleObjectiveDecisionError,
    build_objective_decision_report,
    validate_objective_decision_report,
)


def bridge_report(*, quality_claim: bool = False, weak_count: int = 6, outside_count: int = 5) -> dict:
    return {
        "boundary": BRIDGE_BOUNDARY,
        "summary": {
            "candidate_count": 6,
            "chord_context_available_count": 6,
            "pitch_role_metrics_defined_count": 6,
            "not_evaluable_before_count": 12,
            "not_evaluable_after_count": 0,
            "bridge_flag_counts": {
                "weak_chord_tone_landing_risk": weak_count,
                "outside_soloing_pitch_role_risk": outside_count,
            },
            "min_chord_tone_ratio": 0.216,
            "max_outside_ratio": 0.027,
            "max_non_chord_tone_run": 5,
        },
        "readiness": {
            "chord_context_pitch_role_bridge_completed": True,
            "candidate_count": 6,
            "chord_context_available_count": 6,
            "pitch_role_metrics_defined_count": 6,
            "not_evaluable_before_count": 12,
            "not_evaluable_after_count": 0,
            "followup_objective_source_outside_soloing_not_evaluable_count": 6,
            "followup_objective_repaired_outside_soloing_not_evaluable_count": 6,
            "followup_repair_sweep_source_outside_soloing_not_evaluable_count": 6,
            "followup_repair_sweep_repaired_outside_soloing_not_evaluable_count": 6,
            "repair_sweep_source_outside_soloing_not_evaluable_count": 6,
            "repair_sweep_repaired_outside_soloing_not_evaluable_count": 6,
            "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
            "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
            "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
            "followup_objective_source_outside_soloing_source_context_preserved": True,
            "followup_objective_source_outside_soloing_source_targeted": False,
            "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
            "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
            "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
            "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
            "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
            "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
            "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
            "followup_repair_sweep_source_outside_soloing_source_targeted": False,
            "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
            "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
            "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
            "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
            "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
            "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
            "repair_sweep_source_outside_soloing_source_context_preserved": True,
            "repair_sweep_source_outside_soloing_source_targeted": False,
            "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
            "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
            "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
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
            "next_boundary": BRIDGE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloPitchRoleObjectiveDecisionTest(unittest.TestCase):
    def test_selects_chord_tone_landing_repair_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_objective_decision_report(
                bridge_report=bridge_report(),
                output_dir=Path(tmp) / "decision",
                issue_number=958,
            )
            summary = validate_objective_decision_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                require_objective_decision=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["pitch_role_objective_decision_completed"])
            self.assertEqual(summary["primary_risk_label"], PRIMARY_RISK_LABEL)
            self.assertEqual(summary["weak_chord_tone_landing_risk_count"], 6)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_count"], 5)
            self.assertEqual(summary["not_evaluable_after_count"], 0)
            self.assertEqual(
                summary["followup_objective_source_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(
                summary["followup_objective_repaired_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(
                summary["followup_repair_sweep_source_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(
                summary["followup_repair_sweep_repaired_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(summary["repair_sweep_source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(
                summary["repair_sweep_repaired_outside_soloing_not_evaluable_count"],
                6,
            )
            self.assertEqual(
                summary[
                    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertEqual(
                summary[
                    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after"
                ],
                2,
            )
            self.assertEqual(
                summary["followup_objective_source_outside_soloing_source_pitch_role_risk_delta"],
                3,
            )
            self.assertFalse(
                summary["followup_objective_source_outside_soloing_source_targeted"]
            )
            self.assertTrue(
                summary[
                    "followup_objective_source_outside_soloing_source_residual_risk_preserved"
                ]
            )
            self.assertEqual(
                summary[
                    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after"
                ],
                0,
            )
            self.assertEqual(
                summary["followup_objective_source_outside_soloing_current_pitch_role_risk_delta"],
                2,
            )
            self.assertEqual(
                summary[
                    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before"
                ],
                5,
            )
            self.assertEqual(
                summary["repair_sweep_source_outside_soloing_source_pitch_role_risk_delta"],
                3,
            )
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_bridge_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(StageBMidiToSoloPitchRoleObjectiveDecisionError):
                build_objective_decision_report(
                    bridge_report=bridge_report(quality_claim=True),
                    output_dir=Path(tmp) / "decision",
                    issue_number=958,
                )

    def test_routes_outside_soloing_when_it_dominates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_objective_decision_report(
                bridge_report=bridge_report(weak_count=2, outside_count=5),
                output_dir=Path(tmp) / "decision",
                issue_number=958,
            )
            summary = validate_objective_decision_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=(
                    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_outside_soloing_pitch_role_repair_sweep"
                ),
                expected_target=(
                    "songlike_melody_contour_phrase_rhythm_outside_soloing_pitch_role_repair_sweep"
                ),
                require_objective_decision=True,
                require_no_quality_claim=True,
            )

            self.assertEqual(summary["primary_risk_label"], "outside_soloing_pitch_role_risk")

    def test_rejects_missing_bridge_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = bridge_report()
            del source["readiness"]["followup_objective_source_outside_soloing_not_evaluable_count"]

            with self.assertRaises(StageBMidiToSoloPitchRoleObjectiveDecisionError):
                build_objective_decision_report(
                    bridge_report=source,
                    output_dir=Path(tmp) / "decision",
                    issue_number=958,
                )

    def test_rejects_missing_bridge_source_context_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = bridge_report()
            del source["readiness"][
                "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
            ]

            with self.assertRaises(StageBMidiToSoloPitchRoleObjectiveDecisionError):
                build_objective_decision_report(
                    bridge_report=source,
                    output_dir=Path(tmp) / "decision",
                    issue_number=958,
                )

    def test_rejects_source_context_targeted_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = bridge_report()
            source["readiness"][
                "repair_sweep_source_outside_soloing_source_targeted"
            ] = True

            with self.assertRaises(StageBMidiToSoloPitchRoleObjectiveDecisionError):
                build_objective_decision_report(
                    bridge_report=source,
                    output_dir=Path(tmp) / "decision",
                    issue_number=958,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective_decision",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep",
        )
        self.assertEqual(
            SELECTED_TARGET,
            "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep",
        )


if __name__ == "__main__":
    unittest.main()
